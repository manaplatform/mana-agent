"""Shared Fleet API used by CLI-adjacent clients and the dashboard."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Query, Request

from mana_agent.fleet.models import WorkerStatus

router = APIRouter(prefix="/api/v1/fleet", tags=["fleet"])


def _gateway(request: Request):
    gateway = getattr(request.app.state, "chat_gateway", None)
    if gateway is None or not hasattr(gateway, "fleet_service"):
        raise HTTPException(status_code=503, detail="Fleet coordinator is unavailable")
    return gateway


def _require_mutation(authorization: str | None) -> None:
    from mana_agent.api.routes.workspaces import _require_mutation_token
    _require_mutation_token(authorization)


@router.get("/workers")
def workers(request: Request) -> list[dict]:
    return [
        item.model_dump(mode="json")
        for item in _gateway(request).fleet_registry.list()
    ]


@router.get("/runs")
def runs(request: Request) -> list[dict]:
    return [
        item.model_dump(mode="json")
        for item in _gateway(request).fleet_store.list_runs()
    ]


@router.get("/events")
def events(
    request: Request,
    after_sequence: int = Query(default=0, ge=0),
    limit: int = Query(default=1000, ge=1, le=1000),
) -> list[dict]:
    return [
        item.model_dump(mode="json")
        for item in _gateway(request).fleet_store.events(
            after_sequence=after_sequence, limit=limit,
        )
    ]


@router.post("/workers/{worker_id}/drain")
def drain(worker_id: str, request: Request, authorization: str | None = Header(None)) -> dict:
    _require_mutation(authorization)
    return _gateway(request).fleet_registry.set_status(
        worker_id, WorkerStatus.DRAINING,
    ).model_dump(mode="json")


@router.post("/workers/{worker_id}/revoke")
def revoke(worker_id: str, request: Request, authorization: str | None = Header(None)) -> dict:
    _require_mutation(authorization)
    return _gateway(request).fleet_registry.set_status(
        worker_id, WorkerStatus.REVOKED,
    ).model_dump(mode="json")


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str, request: Request, authorization: str | None = Header(None)) -> dict:
    _require_mutation(authorization)
    _gateway(request).fleet_service.cancel(job_id)
    return {"job_id": job_id, "cancel_requested": True}
