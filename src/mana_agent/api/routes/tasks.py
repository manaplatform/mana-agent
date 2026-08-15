"""Authenticated resilient-execution management API."""

from __future__ import annotations

import asyncio
import hmac
import os
from typing import Any

from fastapi import APIRouter, Header, Query, Request, WebSocket
from pydantic import BaseModel, Field

from mana_agent.api.exceptions import ManaApiError
from mana_agent.execution_supervisor.errors import ExecutionSupervisorError
from mana_agent.execution_supervisor.models import ExecutionState, RecoveryDecision
from mana_agent.services.execution_event_hub import get_execution_event_hub


router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


class CancelRequest(BaseModel):
    reason: str = Field(min_length=1)
    propagate: bool = True
    attempt_id: str = ""


class RecoveryRequest(BaseModel):
    decision: RecoveryDecision


def _authorized(authorization: str | None) -> None:
    from mana_agent.api.routes.workspaces import _require_mutation_token
    _require_mutation_token(authorization)


def _supervisor(request: Request):
    supervisor = getattr(request.app.state, "execution_supervisor", None)
    if supervisor is None:
        raise ManaApiError(503, "Execution supervisor is unavailable.")
    return supervisor


def _call(operation):
    try:
        return operation()
    except ExecutionSupervisorError as exc:
        raise ManaApiError(409, str(exc)) from exc


@router.get("")
def list_tasks(
    request: Request,
    incomplete: bool = Query(default=False),
    authorization: str | None = Header(None),
) -> list[dict]:
    _authorized(authorization)
    return _call(lambda: [
        item.model_dump(mode="json")
        for item in _supervisor(request).store.list_tasks(incomplete_only=incomplete)
    ])


@router.get("/{task_id}")
def task_status(task_id: str, request: Request, authorization: str | None = Header(None)) -> dict:
    _authorized(authorization)
    supervisor = _supervisor(request)
    task = _call(lambda: supervisor.store.get_task(task_id))
    payload = task.model_dump(mode="json")
    payload["parent_progress"] = _call(
        lambda: supervisor.parent_progress(task_id)
    ).model_dump(mode="json")
    checkpoint = supervisor.store.get_checkpoint(task.checkpoint_id) if task.checkpoint_id else None
    payload["checkpoint"] = checkpoint.model_dump(mode="json") if checkpoint else None
    return payload


@router.get("/{task_id}/tree")
def task_tree(task_id: str, request: Request, authorization: str | None = Header(None)) -> list[dict]:
    _authorized(authorization)
    supervisor = _supervisor(request)
    root = _call(lambda: supervisor.store.get_task(task_id))
    rows: list[dict] = []
    pending = [(root, 0)]
    while pending:
        task, depth = pending.pop()
        rows.append({"depth": depth, **task.model_dump(mode="json")})
        pending.extend(
            (_call(lambda child_id=child: supervisor.store.get_task(child_id)), depth + 1)
            for child in reversed(task.child_task_ids)
        )
    return rows


@router.get("/{task_id}/logs")
def task_logs(
    task_id: str,
    request: Request,
    limit: int = Query(default=200, ge=1, le=5000),
    authorization: str | None = Header(None),
) -> list[dict]:
    _authorized(authorization)
    supervisor = _supervisor(request)
    _call(lambda: supervisor.store.get_task(task_id))
    return supervisor.store.events_for_task(task_id, limit=limit)


@router.get("/{task_id}/artefacts")
def task_artifacts(task_id: str, request: Request, authorization: str | None = Header(None)) -> dict:
    _authorized(authorization)
    supervisor = _supervisor(request)
    _call(lambda: supervisor.store.get_task(task_id))
    payload = supervisor.store.artifact_manifest(task_id)
    if payload is None:
        raise ManaApiError(404, "Artifact manifest was not found.")
    return payload


@router.post("/{task_id}/cancel")
def cancel_task(
    task_id: str,
    body: CancelRequest,
    request: Request,
    authorization: str | None = Header(None),
) -> dict:
    _authorized(authorization)
    supervisor = _supervisor(request)
    changed = _call(
        lambda: supervisor.cancel_attempt(
            task_id, attempt_id=body.attempt_id, reason=body.reason
        )
        if body.attempt_id
        else supervisor.cancel(task_id, reason=body.reason, propagate=body.propagate)
    )
    inbox = getattr(request.app.state, "human_inbox", None)
    if inbox is not None:
        for cancelled_task_id in changed:
            inbox.cancel_for_task(cancelled_task_id, reason=body.reason)
    return {"task_id": task_id, "cancelled": changed}


@router.post("/{task_id}/retry")
def retry_task(
    task_id: str,
    body: RecoveryRequest,
    request: Request,
    authorization: str | None = Header(None),
) -> dict:
    _authorized(authorization)
    return _call(lambda: _supervisor(request).retry(task_id, body.decision)).model_dump(mode="json")


@router.post("/{task_id}/resume")
def resume_task(
    task_id: str,
    body: RecoveryRequest,
    request: Request,
    authorization: str | None = Header(None),
) -> dict:
    _authorized(authorization)
    supervisor = _supervisor(request)
    task = _call(lambda: supervisor.store.get_task(task_id))
    if task.state not in {ExecutionState.RETRY_SCHEDULED, ExecutionState.REPLANNING}:
        task = _call(lambda: supervisor.retry(task_id, body.decision))
    resumed = _call(lambda: supervisor.release_retry(task.task_id))
    payload = resumed.model_dump(mode="json")
    checkpoint = _call(lambda: supervisor.get_resumable_checkpoint(task.task_id)) if task.checkpoint_id else None
    payload["checkpoint"] = checkpoint.model_dump(mode="json") if checkpoint else None
    return payload


@router.post("/recover")
def recover_tasks(request: Request, authorization: str | None = Header(None)) -> dict:
    _authorized(authorization)
    supervisor = _supervisor(request)
    return {
        "tree_links_repaired": supervisor.reconnect_tree(),
        **_call(supervisor.recover).model_dump(mode="json"),
    }


@router.websocket("/ws/events")
async def task_events(websocket: WebSocket, token: str | None = Query(default=None)) -> None:
    expected = str(os.getenv("MANA_API_TOKEN") or "").strip()
    authorization = str(websocket.headers.get("authorization") or "")
    bearer = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    supplied = str(token or bearer)
    if expected and not hmac.compare_digest(supplied, expected):
        await websocket.close(code=4401, reason="Authentication required.")
        return
    await websocket.accept()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=500)
    loop = asyncio.get_running_loop()

    def enqueue(payload: dict[str, Any]) -> None:
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            queue.put_nowait(payload)

    def receive(payload: dict[str, Any]) -> None:
        metadata = payload.get("metadata") or payload.get("details") or {}
        if not metadata.get("execution_supervisor"):
            return
        try:
            loop.call_soon_threadsafe(enqueue, payload)
        except RuntimeError:
            return

    unsubscribe = get_execution_event_hub().subscribe_all(receive)
    try:
        await websocket.send_json({"type": "socket.ready", "stream": "execution_supervisor"})
        while True:
            await websocket.send_json(await queue.get())
    finally:
        unsubscribe()
