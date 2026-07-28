"""HTTP surface for Teach Mode dashboard and agents."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from mana_agent.api.exceptions import ManaApiError
from mana_agent.teach import TeachError, TeachService
from mana_agent.teach.permissions import DESKTOP_GRANTS, TeachGrantScope, grant_status

router = APIRouter(prefix="/api/v1/teach", tags=["teach"])


class StartRequest(BaseModel):
    task_name: str = Field(min_length=1, max_length=240)
    permissions: list[str] = Field(default_factory=list)
    desktop: bool | None = None


class GrantRequest(BaseModel):
    scopes: list[TeachGrantScope] = Field(default_factory=lambda: list(DESKTOP_GRANTS))
    action: Literal["allow", "revoke"]


class ExplainRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ReplayRequest(BaseModel):
    version: int | None = Field(default=None, ge=1)
    mode: str = "dry_run"
    inputs: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


def _require(authorization: str | None) -> None:
    from mana_agent.api.routes.workspaces import _require_mutation_token
    _require_mutation_token(authorization)


def _call(operation):
    try:
        return operation()
    except TeachError as exc:
        raise ManaApiError(422, str(exc)) from exc


@router.get("/doctor")
def doctor() -> dict[str, Any]:
    return TeachService().doctor()


@router.get("/sessions")
def sessions() -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in TeachService().storage.list_sessions()]


@router.get("/flows")
def flows() -> list[dict[str, Any]]:
    return [item.model_dump(mode="json", by_alias=True) for item in TeachService().storage.list_flows()]


@router.get("/grants")
def grants() -> list[dict[str, Any]]:
    service = TeachService()
    return [item.model_dump(mode="json") for item in grant_status(service.grants)]


@router.post("/grants")
def update_grants(payload: GrantRequest, authorization: str | None = Header(None)) -> list[dict[str, Any]]:
    _require(authorization)
    service = TeachService()
    if payload.action == "allow":
        service.grants.grant(payload.scopes)
    else:
        service.grants.revoke(payload.scopes)
    return [item.model_dump(mode="json") for item in grant_status(service.grants)]


@router.post("/sessions")
def start(payload: StartRequest, authorization: str | None = Header(None)) -> dict[str, Any]:
    _require(authorization)
    return _call(
        lambda: TeachService().start(
            payload.task_name,
            permissions=payload.permissions,
            desktop=payload.desktop,
        )
    ).model_dump(mode="json")


@router.post("/sessions/{session_id}/pause")
def pause(session_id: str, authorization: str | None = Header(None)) -> dict[str, Any]:
    _require(authorization)
    return _call(lambda: TeachService().pause(session_id)).model_dump(mode="json")


@router.post("/sessions/{session_id}/resume")
def resume(session_id: str, authorization: str | None = Header(None)) -> dict[str, Any]:
    _require(authorization)
    return _call(lambda: TeachService().resume(session_id)).model_dump(mode="json")


@router.post("/sessions/{session_id}/explanations")
def explain(session_id: str, payload: ExplainRequest, authorization: str | None = Header(None)) -> dict[str, Any]:
    _require(authorization)
    return _call(lambda: TeachService().explain(payload.text, session_id)).model_dump(mode="json")


@router.post("/sessions/{session_id}/stop")
def stop(session_id: str, authorization: str | None = Header(None)) -> dict[str, Any]:
    _require(authorization)
    session, flow = _call(lambda: TeachService().stop(session_id))
    return {"session": session.model_dump(mode="json"), "flow": flow.model_dump(mode="json", by_alias=True)}


@router.post("/sessions/{session_id}/cancel")
def cancel(session_id: str, authorization: str | None = Header(None)) -> dict[str, Any]:
    _require(authorization)
    return _call(lambda: TeachService().cancel(session_id)).model_dump(mode="json")


@router.post("/flows/{flow_id}/replay")
def replay(flow_id: str, payload: ReplayRequest, authorization: str | None = Header(None)) -> dict[str, Any]:
    _require(authorization)
    return _call(
        lambda: TeachService().replay(
            flow_id,
            version=payload.version,
            mode=payload.mode,
            inputs=payload.inputs,
            context=payload.context,
        )
    ).model_dump(mode="json")


@router.get("/flows/{flow_id}/card")
def card(flow_id: str) -> dict[str, Any]:
    return _call(lambda: TeachService().flow_card(flow_id)).model_dump(mode="json")
