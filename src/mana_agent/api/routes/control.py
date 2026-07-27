from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, Field

from mana_agent.api.exceptions import ManaApiError
from mana_agent.background import BackgroundProcessManager
from mana_agent.chat_commands import CommandContext, CommandDispatcher, build_default_registry
from mana_agent.connectors.service import ConnectorService, TelegramConnectRequest
from mana_agent.sessions import SessionService

router = APIRouter(prefix="/api/v1", tags=["commands", "connectors", "processes"])


class CommandRequest(BaseModel):
    text: str
    session_id: str
    confirmed: bool = False


class ConnectorConnectRequest(BaseModel):
    token: str = Field(repr=False)
    settings: TelegramConnectRequest


def _require_token(authorization: str | None) -> None:
    from mana_agent.api.routes.workspaces import _require_mutation_token
    _require_mutation_token(authorization)


def _services(request: Request) -> tuple[Any | None, SessionService, BackgroundProcessManager, ConnectorService]:
    gateway = getattr(request.app.state, "chat_gateway", None)
    if gateway is not None:
        return gateway, gateway.session_service, gateway.background_processes, gateway.connector_service
    processes = BackgroundProcessManager()
    return None, SessionService(process_manager=processes), processes, ConnectorService(processes)


@router.post("/commands")
def dispatch_command(payload: CommandRequest, request: Request, authorization: str | None = Header(None)) -> dict:
    _require_token(authorization)
    gateway, sessions, processes, connectors = _services(request)
    if gateway is not None:
        result = gateway.dispatch_command(payload.text, session_id=payload.session_id, frontend="api", confirmed=payload.confirmed)
        if result is None and not payload.text.lstrip().startswith("/"):
            routed = gateway.process_turn(payload.session_id, payload.text)
            if routed.mode != "command":
                raise ManaApiError(422, "The model did not resolve this request to a registered command. No command was executed.")
            return dict((routed.payload or {}).get("command_result") or {})
    else:
        try:
            record = sessions.workspaces.store.get_session(payload.session_id)
        except FileNotFoundError as exc:
            raise ManaApiError(404, "Session not found.") from exc
        context = CommandContext(
            frontend="api", session_id=record.session_id, workspace_id=record.workspace_id,
            repository_id=record.primary_repository_id, capabilities={"chat", "sessions", "processes", "connectors"},
            sessions=sessions, processes=processes, connectors=connectors,
        )
        result = CommandDispatcher(build_default_registry()).dispatch(payload.text, context, confirmed=payload.confirmed)
    if result is None:
        raise ManaApiError(422, "Input is not a registered command; submit normal chat through the session gateway.")
    return result.model_dump(mode="json")


@router.get("/connectors")
def list_connectors(request: Request) -> list[dict[str, Any]]:
    return _services(request)[3].list()


@router.post("/connectors/telegram/connect")
def connect_telegram(payload: ConnectorConnectRequest, request: Request, authorization: str | None = Header(None)) -> dict:
    _require_token(authorization)
    return _services(request)[3].connect_telegram(payload.settings, token=payload.token).model_dump(mode="json")


@router.post("/connectors/telegram/start")
def start_telegram(request: Request, authorization: str | None = Header(None)) -> dict:
    _require_token(authorization)
    return _services(request)[3].start_telegram().model_dump(mode="json")


@router.post("/connectors/telegram/stop")
def stop_telegram(request: Request, authorization: str | None = Header(None)) -> dict:
    _require_token(authorization)
    return _services(request)[3].stop_telegram().model_dump(mode="json")


@router.get("/processes")
def list_processes(request: Request) -> list[dict]:
    return [row.model_dump(mode="json") for row in _services(request)[2].list()]


@router.get("/processes/{process_id}")
def get_process(process_id: str, request: Request) -> dict:
    try:
        return _services(request)[2].inspect(process_id).model_dump(mode="json")
    except FileNotFoundError as exc:
        raise ManaApiError(404, "Background process not found.") from exc


@router.post("/processes/{process_id}/stop")
def stop_process(process_id: str, request: Request, authorization: str | None = Header(None)) -> dict:
    _require_token(authorization)
    return _services(request)[2].stop(process_id).model_dump(mode="json")


@router.post("/processes/{process_id}/restart")
def restart_process(process_id: str, request: Request, authorization: str | None = Header(None)) -> dict:
    _require_token(authorization)
    return _services(request)[2].restart(process_id).model_dump(mode="json")
