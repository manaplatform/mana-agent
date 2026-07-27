"""Conversation REST + chat execution API for the dashboard."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from mana_agent.api.exceptions import ManaApiError
from mana_agent.services.conversation_service import ConversationService
from mana_agent.services.execution_event_hub import get_execution_event_hub
from mana_agent.ui.streamlit_helpers import find_mana_root
from mana_agent.workspaces.paths import repository_id_for_path
from mana_agent.workspaces.service import WorkspaceService

router = APIRouter(prefix="/api/v1", tags=["conversations"])


def _require_mutation_token(authorization: str | None) -> None:
    expected = str(os.getenv("MANA_API_TOKEN") or "").strip()
    if expected and authorization != f"Bearer {expected}":
        raise ManaApiError(401, "A valid API bearer token is required.")


def _resolve_root(root: str | None = None, repository_id: str | None = None) -> tuple[Path, str]:
    if repository_id:
        try:
            repo = WorkspaceService().store.get_repository(repository_id)
            path = Path(repo.canonical_path).expanduser().resolve()
            return path, repository_id
        except FileNotFoundError as exc:
            raise ManaApiError(404, "Repository not found.") from exc
    path = find_mana_root(Path(root).expanduser().resolve() if root else None)
    return path, repository_id_for_path(path)


def _service(root: str | None = None, repository_id: str | None = None) -> ConversationService:
    path, repo_id = _resolve_root(root=root, repository_id=repository_id)
    return ConversationService(root=path, repository_id=repo_id)


class ConversationCreateRequest(BaseModel):
    title: str = ""
    root: str | None = None
    repository_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationListQuery(BaseModel):
    root: str | None = None
    repository_id: str | None = None
    limit: int = Field(default=50, ge=1, le=500)


class MessageCreateRequest(BaseModel):
    content: str = Field(min_length=1)
    client_message_id: str = Field(default="", max_length=128)
    root: str | None = None
    repository_id: str | None = None


class ComputerPermissionDecisionRequest(BaseModel):
    decision: str = Field(pattern=r"^(deny|allow_once|allow_session|always)$")
    root: str | None = None
    repository_id: str | None = None


@router.get("/conversations")
def list_conversations(
    root: str | None = None,
    repository_id: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    service = _service(root=root, repository_id=repository_id)
    rows = service.list(limit=limit)
    return {
        "ok": True,
        "repository_id": service.repository_id,
        "root": str(service.root),
        "conversations": [item.to_dict() for item in rows],
    }


@router.get("/dashboard/live-chat", response_class=HTMLResponse)
def dashboard_live_chat(
    request: Request,
    conversation_id: str,
    root: str | None = None,
    repository_id: str | None = None,
    height: int = 680,
) -> HTMLResponse:
    """Serve the same-origin dashboard reducer without a second event model."""
    service = _service(root=root, repository_id=repository_id)
    try:
        payload = service.get_full(
            conversation_id,
            message_limit=500,
            event_limit=1000,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise ManaApiError(404, "Conversation not found.") from exc
    from mana_agent.dashboard.components.live_chat import live_chat_html

    base = str(request.base_url).rstrip("/")
    html = live_chat_html(
        conversation_id=conversation_id,
        root=service.root,
        api_base=base,
        messages=payload["messages"],
        events=payload["events"],
        height=max(320, min(int(height or 680), 1200)),
    )
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'none'; script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'; connect-src 'self' ws: wss:"
            ),
            "Referrer-Policy": "no-referrer",
        },
    )


@router.post("/conversations", status_code=201)
def create_conversation(
    payload: ConversationCreateRequest,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    _require_mutation_token(authorization)
    service = _service(root=payload.root, repository_id=payload.repository_id)
    record = service.create(title=payload.title, metadata=payload.metadata)
    return {"ok": True, "conversation": record.to_dict()}


@router.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    root: str | None = None,
    repository_id: str | None = None,
    message_limit: int = 500,
    event_limit: int = 200,
) -> dict[str, Any]:
    service = _service(root=root, repository_id=repository_id)
    try:
        payload = service.get_full(conversation_id, message_limit=message_limit, event_limit=event_limit)
    except (FileNotFoundError, ValueError) as exc:
        raise ManaApiError(404, "Conversation not found.") from exc
    return {"ok": True, **payload}


@router.get("/conversations/{conversation_id}/messages")
def list_messages(
    conversation_id: str,
    root: str | None = None,
    repository_id: str | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    service = _service(root=root, repository_id=repository_id)
    try:
        messages = service.list_messages(conversation_id, limit=limit)
    except (FileNotFoundError, ValueError) as exc:
        raise ManaApiError(404, "Conversation not found.") from exc
    return {"ok": True, "conversation_id": conversation_id, "messages": [item.to_dict() for item in messages]}


@router.get("/conversations/{conversation_id}/events")
def list_events(
    conversation_id: str,
    root: str | None = None,
    repository_id: str | None = None,
    execution_id: str = "",
    limit: int = 200,
) -> dict[str, Any]:
    service = _service(root=root, repository_id=repository_id)
    try:
        events = service.list_events(conversation_id, execution_id=execution_id, limit=limit)
    except (FileNotFoundError, ValueError) as exc:
        raise ManaApiError(404, "Conversation not found.") from exc
    return {
        "ok": True,
        "conversation_id": conversation_id,
        "execution_id": execution_id or None,
        "events": events,
    }


@router.post("/conversations/{conversation_id}/messages", status_code=201)
def send_message(
    conversation_id: str,
    payload: MessageCreateRequest,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    _require_mutation_token(authorization)
    service = _service(root=payload.root, repository_id=payload.repository_id)
    try:
        result = service.send_message(
            conversation_id,
            payload.content,
            client_message_id=payload.client_message_id,
        )
    except FileNotFoundError as exc:
        raise ManaApiError(404, "Conversation not found.") from exc
    except ValueError as exc:
        raise ManaApiError(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise ManaApiError(500, "Chat execution failed.", error=str(exc)) from exc
    return {"ok": True, **result}


@router.post(
    "/conversations/{conversation_id}/computer-permissions/{permission_request_id}"
)
def decide_computer_permission_in_chat(
    conversation_id: str,
    permission_request_id: str,
    payload: ComputerPermissionDecisionRequest,
    authorization: str | None = Header(None),
) -> dict[str, Any]:
    """Apply a trusted dashboard-chat decision and resume the stored exact action."""
    _require_mutation_token(authorization)
    service = _service(root=payload.root, repository_id=payload.repository_id)
    try:
        service.get_or_raise(conversation_id)
    except (FileNotFoundError, ValueError) as exc:
        raise ManaApiError(404, "Conversation not found.") from exc

    from mana_agent.integrations.computer_control.cancellation import (
        decide_computer_permission,
        deny_computer_permission,
    )
    from mana_agent.integrations.computer_control.errors import ComputerControlError
    from mana_agent.integrations.computer_control.events import computer_event_scope
    from mana_agent.integrations.computer_control.models import PermissionDecision

    hub = get_execution_event_hub()

    def event_sink(event_type: str, title: str, **kwargs: Any) -> None:
        values = dict(kwargs)
        metadata = dict(values.pop("metadata", {}) or {})
        hub.emit(
            event_type,
            title=title,
            conversation_id=conversation_id,
            execution_id=str(values.pop("execution_id", "") or ""),
            repository_id=service.repository_id,
            message=title,
            status=str(values.pop("status", "running") or "running"),
            metadata=metadata,
        )

    if payload.decision == "deny":
        try:
            deny_computer_permission(permission_request_id, client_type="dashboard")
        except ComputerControlError as exc:
            raise ManaApiError(409, str(exc), error=exc.code) from exc
        hub.emit(
            "computer.permission_decided",
            title="Computer permission denied",
            conversation_id=conversation_id,
            repository_id=service.repository_id,
            status="cancelled",
            metadata={
                "permission_request_id": permission_request_id,
                "decision": "deny",
            },
        )
        return {"ok": True, "decision": "deny", "executed": False}

    decisions = {
        "allow_once": PermissionDecision.ALLOW_ONCE,
        "allow_session": PermissionDecision.ALLOW_SESSION,
        "always": PermissionDecision.ALWAYS_ALLOW,
    }
    try:
        with computer_event_scope(event_sink):
            result = decide_computer_permission(
                permission_request_id,
                decision=decisions[payload.decision],
                client_type="dashboard",
            )
    except ComputerControlError as exc:
        if exc.code != "confirmation_required":
            raise ManaApiError(409, str(exc), error=exc.code) from exc
        result_payload: dict[str, Any] = exc.payload()
        executed = False
    else:
        result_payload = result.model_dump(mode="json")
        executed = True
    hub.emit(
        "computer.permission_decided",
        title="Computer permission approved",
        conversation_id=conversation_id,
        execution_id=str(result_payload.get("execution_id") or ""),
        repository_id=service.repository_id,
        status="success",
        metadata={
            "permission_request_id": permission_request_id,
            "decision": payload.decision,
            "executed": executed,
        },
    )
    return {
        "ok": True,
        "decision": payload.decision,
        "executed": executed,
        "result": result_payload,
    }


@router.get("/conversations/{conversation_id}/execution")
def get_execution_state(
    conversation_id: str,
    root: str | None = None,
    repository_id: str | None = None,
    execution_id: str = "",
) -> dict[str, Any]:
    service = _service(root=root, repository_id=repository_id)
    try:
        record = service.get_or_raise(conversation_id)
    except (FileNotFoundError, ValueError) as exc:
        raise ManaApiError(404, "Conversation not found.") from exc
    exec_id = execution_id or record.last_execution_id
    events = service.list_events(conversation_id, execution_id=exec_id, limit=500) if exec_id else []
    return {
        "ok": True,
        "conversation_id": conversation_id,
        "status": record.status,
        "execution_id": exec_id or None,
        "events": events,
        "hub_history_count": len(
            get_execution_event_hub().history(
                conversation_id=conversation_id,
                execution_id=exec_id,
                repository_id=service.repository_id,
            )
        ),
    }
