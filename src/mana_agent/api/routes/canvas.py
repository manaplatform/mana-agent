"""Authenticated REST endpoints for Canvas recovery and renderer actions."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from mana_agent.api.exceptions import ManaApiError
from mana_agent.api.routes.conversations import (
    _require_mutation_token,
    _resolve_root,
    _service,
)
from mana_agent.canvas.catalog import catalog_metadata
from mana_agent.canvas.config import (
    CanvasConfig,
    IMPLEMENTATION_VERSION,
    LOCAL_CATALOG_PATH,
    is_loopback_url,
)
from mana_agent.canvas.models import RendererAction, RendererCapabilities
from mana_agent.canvas.reducer import CanvasStateError
from mana_agent.canvas.service import canvas_service_for_root
from mana_agent.config.settings import Settings


router = APIRouter(prefix="/api/v1", tags=["canvas"])


class CanvasActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1, max_length=160)
    version: str = "v0.9"
    source_component_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    correlation_id: str = Field(min_length=1, max_length=160)
    context: dict[str, Any] = Field(default_factory=dict)
    timestamp: str | None = None
    root: str | None = None
    repository_id: str | None = None


class CanvasCloseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_id: str = Field(min_length=1, max_length=160)
    root: str | None = None
    repository_id: str | None = None


def _canvas(root: str | None, repository_id: str | None):
    path, _ = _resolve_root(root=root, repository_id=repository_id)
    return path, canvas_service_for_root(
        path, config=CanvasConfig.from_settings(Settings())
    )


def _require_conversation(
    conversation_id: str, root: str | None, repository_id: str | None
) -> None:
    try:
        _service(root=root, repository_id=repository_id).get_or_raise(conversation_id)
    except (FileNotFoundError, ValueError) as exc:
        raise ManaApiError(404, "Conversation not found.") from exc


@router.get("/canvas/capabilities")
def canvas_capabilities(request: Request) -> dict[str, Any]:
    config = CanvasConfig.from_settings(Settings())
    catalog_ids = list(config.allowed_catalogs)
    local_catalog_id = str(request.base_url).rstrip("/") + LOCAL_CATALOG_PATH
    if config.allow_localhost and is_loopback_url(local_catalog_id):
        catalog_ids.append(local_catalog_id)
    return {
        "ok": True,
        "enabled": config.enabled,
        "implementation_version": IMPLEMENTATION_VERSION,
        "renderer": RendererCapabilities(
            protocol_versions=config.protocol_versions,
            catalog_ids=tuple(dict.fromkeys(catalog_ids)),
            inline_catalogs=config.accept_inline_catalogs,
            max_components=config.max_components_per_surface,
        ).model_dump(mode="json"),
        "catalog": catalog_metadata(),
    }


@router.get("/canvas/catalogs/core/v1/catalog.json")
def canvas_core_catalog() -> dict[str, Any]:
    """Serve the built-in catalog from the same local API as the renderer."""
    return catalog_metadata()


@router.get("/conversations/{conversation_id}/canvas/surfaces")
def list_canvas_surfaces(
    conversation_id: str,
    root: str | None = None,
    repository_id: str | None = None,
    include_deleted: bool = False,
) -> dict[str, Any]:
    _require_conversation(conversation_id, root, repository_id)
    _, service = _canvas(root, repository_id)
    return {
        "ok": True,
        "conversation_id": conversation_id,
        "surfaces": [
            item.model_dump(mode="json")
            for item in service.list_surfaces(
                conversation_id, include_deleted=include_deleted
            )
            if item.conversation_id == conversation_id
        ],
    }


@router.get("/conversations/{conversation_id}/canvas/surfaces/{surface_id}")
def get_canvas_surface(
    conversation_id: str,
    surface_id: str,
    root: str | None = None,
    repository_id: str | None = None,
    after_sequence: int = 0,
) -> dict[str, Any]:
    _require_conversation(conversation_id, root, repository_id)
    _, service = _canvas(root, repository_id)
    try:
        snapshot, events = service.replay(
            conversation_id, surface_id, after_sequence=after_sequence
        )
    except CanvasStateError as exc:
        raise ManaApiError(
            410 if "expired" in str(exc).lower() else 404, str(exc)
        ) from exc
    if snapshot.conversation_id != conversation_id:
        raise ManaApiError(403, "Surface does not belong to this conversation.")
    return {
        "ok": True,
        "snapshot": snapshot.model_dump(mode="json"),
        "events": [item.model_dump(mode="json") for item in events],
    }


@router.post("/conversations/{conversation_id}/canvas/surfaces/{surface_id}/actions")
def submit_canvas_action(
    conversation_id: str,
    surface_id: str,
    payload: CanvasActionRequest,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    _require_mutation_token(authorization)
    _require_conversation(conversation_id, payload.root, payload.repository_id)
    _, service = _canvas(payload.root, payload.repository_id)
    try:
        action_values = payload.model_dump(
            exclude={"root", "repository_id", "timestamp"}
        )
        if payload.timestamp:
            action_values["timestamp"] = payload.timestamp
        result = service.submit_action(
            RendererAction(
                **action_values,
                session_id=conversation_id,
                conversation_id=conversation_id,
                surface_id=surface_id,
            )
        )
    except (CanvasStateError, ValueError) as exc:
        raise ManaApiError(409, str(exc), error="canvas_action_rejected") from exc
    return {"ok": True, "result": result.model_dump(mode="json")}


@router.post("/conversations/{conversation_id}/canvas/surfaces/{surface_id}/close")
def close_canvas_surface(
    conversation_id: str,
    surface_id: str,
    payload: CanvasCloseRequest,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    _require_mutation_token(authorization)
    _require_conversation(conversation_id, payload.root, payload.repository_id)
    _, service = _canvas(payload.root, payload.repository_id)
    try:
        snapshot = service.delete_surface(
            session_id=conversation_id,
            conversation_id=conversation_id,
            surface_id=surface_id,
            correlation_id=payload.correlation_id,
        )
    except CanvasStateError as exc:
        raise ManaApiError(409, str(exc)) from exc
    return {"ok": True, "snapshot": snapshot.model_dump(mode="json")}


@router.get("/dashboard/live-canvas", response_class=HTMLResponse)
def dashboard_live_canvas(
    request: Request,
    conversation_id: str,
    root: str | None = None,
    repository_id: str | None = None,
    surface_id: str = "",
    height: int = 760,
) -> HTMLResponse:
    _require_conversation(conversation_id, root, repository_id)
    path, service = _canvas(root, repository_id)
    from mana_agent.dashboard.components.live_canvas import live_canvas_html

    html = live_canvas_html(
        conversation_id=conversation_id,
        root=path,
        api_base=str(request.base_url).rstrip("/"),
        surface_id=surface_id,
        height=max(360, min(int(height or 760), 1400)),
        generation_timeout_seconds=service.config.generation_timeout_seconds,
    )
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'none'; img-src https: http://localhost:* http://127.0.0.1:*; "
                "script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'; connect-src 'self' ws: wss:; "
                "form-action 'none'; frame-ancestors 'self' http://localhost:* "
                "http://127.0.0.1:*"
            ),
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )
