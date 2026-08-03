"""Authenticated authoritative API for dashboard inbox clients."""

from __future__ import annotations

import getpass
from typing import Any

from fastapi import APIRouter, Header, Query, Request
from pydantic import BaseModel, Field

from mana_agent.api.exceptions import ManaApiError

from .models import InboxQuery, InboxRequestType, InboxStatus, ResponseOperation, ResponseSubmission


router = APIRouter(prefix="/api/v1/inbox", tags=["human-inbox"])


def _authorized(authorization: str | None) -> None:
    from mana_agent.api.routes.workspaces import _require_mutation_token
    _require_mutation_token(authorization)


def _actor(request: Request, value: str | None) -> str:
    resolver = getattr(request.app.state, "human_inbox_identity_resolver", None)
    if resolver is not None:
        resolved = str(resolver(request) or "").strip()
        if not resolved:
            raise ManaApiError(403, "Reviewer identity resolution failed.")
        return resolved
    local = getpass.getuser()
    if value and value.strip() != local:
        raise ManaApiError(403, "The standalone API cannot accept a caller-selected reviewer identity.")
    return local


def _service(request: Request):
    service = getattr(request.app.state, "human_inbox", None)
    if service is None:
        raise ManaApiError(503, "Durable human inbox is unavailable.")
    return service


def _call(operation):
    try:
        return operation()
    except (LookupError, PermissionError, ValueError, RuntimeError) as exc:
        raise ManaApiError(409, str(exc)) from exc


class TokenRequest(BaseModel):
    operation: ResponseOperation
    ttl_seconds: int = Field(default=900, ge=1, le=3600)


class WebResponse(BaseModel):
    operation: ResponseOperation
    idempotency_key: str = Field(min_length=1)
    answer: dict[str, Any] = Field(default_factory=dict)
    comment: str = ""
    signed_token: str = Field(min_length=1)
    expected_version: int | None = Field(default=None, ge=0)


@router.get("")
def list_inbox(
    request: Request,
    status: list[InboxStatus] = Query(default=[]),
    reviewer: str = Query(default=""),
    role: str = Query(default=""),
    group: str = Query(default=""),
    task: str = Query(default=""),
    branch: str = Query(default=""),
    request_type: InboxRequestType | None = Query(default=None),
    authorization: str | None = Header(None),
    reviewer_identity: str | None = Header(None, alias="X-Mana-Reviewer"),
) -> list[dict[str, Any]]:
    _authorized(authorization)
    actor_id = _actor(request, reviewer_identity)
    query = InboxQuery(
        statuses=set(status),
        reviewer_id=reviewer,
        role=role,
        group=group,
        task_id=task,
        branch_id=branch,
        request_type=request_type,
    )
    return _call(lambda: [item.card() for item in _service(request).list(query, actor_id=actor_id)])


@router.get("/metrics")
def inbox_metrics(request: Request, authorization: str | None = Header(None)) -> dict[str, Any]:
    _authorized(authorization)
    return _call(lambda: _service(request).metrics())


@router.get("/{inbox_item_id}")
def show_inbox(
    inbox_item_id: str,
    request: Request,
    authorization: str | None = Header(None),
    reviewer_identity: str | None = Header(None, alias="X-Mana-Reviewer"),
) -> dict[str, Any]:
    _authorized(authorization)
    service = _service(request)
    item = _call(lambda: service.get(inbox_item_id, actor_id=_actor(request, reviewer_identity)))
    return {
        **item.card(),
        "audit": [event.model_dump(mode="json") for event in service.repository.audit_for_item(item.inbox_item_id)],
        "delivery_attempts": [attempt.model_dump(mode="json") for attempt in service.repository.delivery_attempts(item.inbox_item_id)],
    }


@router.post("/{inbox_item_id}/response-token")
def response_token(
    inbox_item_id: str,
    body: TokenRequest,
    request: Request,
    authorization: str | None = Header(None),
    reviewer_identity: str | None = Header(None, alias="X-Mana-Reviewer"),
) -> dict[str, str]:
    _authorized(authorization)
    token, csrf = _call(lambda: _service(request).issue_response_token(
        inbox_item_id,
        actor_id=_actor(request, reviewer_identity),
        operation=body.operation,
        ttl_seconds=body.ttl_seconds,
    ))
    return {"signed_token": token, "csrf_token": csrf}


@router.post("/{inbox_item_id}/respond")
def respond(
    inbox_item_id: str,
    body: WebResponse,
    request: Request,
    authorization: str | None = Header(None),
    reviewer_identity: str | None = Header(None, alias="X-Mana-Reviewer"),
    csrf: str | None = Header(None, alias="X-Mana-CSRF"),
) -> dict[str, Any]:
    _authorized(authorization)
    service = _service(request)
    if not csrf or not service.token_signer.verify_csrf(body.signed_token, csrf):
        service.record_rejected_response(
            inbox_item_id,
            actor_id=_actor(request, reviewer_identity),
            reason="csrf_validation",
        )
        raise ManaApiError(403, "CSRF validation failed.")
    actor_id = _actor(request, reviewer_identity)
    item = service.repository.get(inbox_item_id)
    try:
        result = service.respond(ResponseSubmission(
            inbox_item_id=inbox_item_id,
            operation=body.operation,
            actor_id=actor_id,
            channel="dashboard",
            idempotency_key=body.idempotency_key,
            answer=body.answer,
            comment=body.comment,
            signed_token=body.signed_token,
            expected_version=body.expected_version,
            current_action_digest=item.action_digest,
        ))
    except (PermissionError, ValueError, RuntimeError) as exc:
        raise ManaApiError(409, str(exc)) from exc
    return result.card()


@router.post("/maintenance/reconcile")
def reconcile(request: Request, authorization: str | None = Header(None)) -> dict[str, Any]:
    _authorized(authorization)
    service = _service(request)
    expired = service.expire_due()
    reminders = service.send_due_reminders()
    report = service.reconcile()
    return {
        **report.model_dump(mode="json"),
        "expired": [item.inbox_item_id for item in expired],
        "reminders": [attempt.delivery_attempt_id for attempt in reminders],
    }
