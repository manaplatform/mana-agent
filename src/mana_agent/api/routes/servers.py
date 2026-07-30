"""Read-only server registry and audit API for dashboard clients."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request


router = APIRouter(prefix="/api/v1/servers", tags=["servers"])


def _service(request: Request):
    gateway = getattr(request.app.state, "chat_gateway", None)
    if gateway is None or not hasattr(gateway, "server_management_service"):
        raise HTTPException(status_code=503, detail="Server management is unavailable")
    return gateway.server_management_service


def _safe(server) -> dict:
    return server.model_dump(
        mode="json",
        exclude={"credential_ref", "known_hosts_file", "host_key_fingerprint"},
    )


@router.get("")
def list_servers(request: Request) -> list[dict]:
    return [_safe(server) for server in _service(request).list_servers()]


@router.get("/{server_id}")
def get_server(server_id: str, request: Request) -> dict:
    try:
        return _safe(_service(request).server(server_id))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{server_id}/audit")
def server_audit(server_id: str, request: Request, limit: int = Query(100, ge=1, le=10_000)) -> list[dict]:
    try:
        return _service(request).logs(server_id, limit=limit)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
