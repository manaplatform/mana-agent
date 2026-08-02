"""Authorization-preserving operational API for memory capsules."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Header, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from mana_agent.api.exceptions import ManaApiError
from mana_agent.memory import (
    CapsuleAuthorizationError,
    CapsuleMergeConflict,
    CapsuleReadRequest,
    CapsuleScope,
    DeleteMode,
    MergeStrategy,
)
from mana_agent.memory.capsules.repository import RevisionConflict
from mana_agent.memory.errors import MemoryNotFoundError


router = APIRouter(prefix="/api/v1/memory/capsules", tags=["memory-capsules"])


class QueryRequest(BaseModel):
    query: str = ""
    allowed_scopes: set[CapsuleScope] = Field(min_length=1)
    namespaces: set[str] = Field(default_factory=set)
    max_capsules: int = Field(default=12, ge=1, le=100)
    max_tokens: int = Field(default=4000, ge=1)
    include_staged: bool = False


class StageRequest(BaseModel):
    target_scope: CapsuleScope
    strategy: MergeStrategy
    team_id: str | None = None


class MergeRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=256)
    target_capsule_id: str | None = None
    expected_target_revision: int | None = Field(default=None, ge=1)
    expected_target_hash: str | None = Field(default=None, min_length=64, max_length=64)
    strategy: MergeStrategy | None = None
    decision_reason: str = Field(default="", max_length=2000)


class DeleteRequest(BaseModel):
    mode: DeleteMode = DeleteMode.SOFT


def _authorized(authorization: str | None) -> None:
    from mana_agent.api.routes.workspaces import _require_mutation_token
    _require_mutation_token(authorization)


def _bound(request: Request):
    """Require an application-provided authenticated identity; never trust request IDs."""
    resolver = getattr(request.app.state, "capsule_identity_resolver", None)
    service = getattr(request.app.state, "capsule_service", None)
    if not callable(resolver) or service is None:
        raise ManaApiError(503, "Memory capsule identity resolution is unavailable; no fallback identity was used.")
    principal, context = resolver(request)
    return service, principal, context


def _call(operation):
    try:
        return operation()
    except MemoryNotFoundError as exc:
        raise ManaApiError(404, "Capsule was not found.") from exc
    except CapsuleAuthorizationError as exc:
        # Generic not-found prevents inaccessible capsule enumeration.
        raise ManaApiError(404, "Capsule was not found.") from exc
    except (CapsuleMergeConflict, RevisionConflict, ValueError) as exc:
        raise ManaApiError(409, str(exc)) from exc


def _public(value: Any) -> Any:
    payload = jsonable_encoder(asdict(value))
    def strip(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: strip(child) for key, child in item.items() if key != "capabilities"}
        if isinstance(item, list):
            return [strip(child) for child in item]
        return item
    return strip(payload)


@router.post("/query")
def query_capsules(body: QueryRequest, request: Request, authorization: str | None = Header(None)) -> list[dict[str, Any]]:
    _authorized(authorization)
    service, principal, context = _bound(request)
    rows = _call(lambda: service.query_capsules(CapsuleReadRequest(
        principal=principal,
        task_context=context,
        query=body.query,
        allowed_scopes=frozenset(body.allowed_scopes),
        namespaces=frozenset(body.namespaces),
        max_capsules=body.max_capsules,
        max_tokens=body.max_tokens,
        include_staged=body.include_staged,
    )))
    return [_public(item) for item in rows]


@router.get("/staged")
def list_staged(request: Request, authorization: str | None = Header(None)) -> list[dict[str, Any]]:
    _authorized(authorization)
    service, principal, context = _bound(request)
    return [_public(item) for item in _call(lambda: service.list_staged_capsules(principal=principal, context=context))]


@router.get("/{capsule_id}")
def inspect_capsule(capsule_id: str, request: Request, authorization: str | None = Header(None)) -> dict[str, Any]:
    _authorized(authorization)
    service, principal, context = _bound(request)
    return _public(_call(lambda: service.get_capsule(capsule_id, principal=principal, context=context)))


@router.get("/{capsule_id}/lineage")
def capsule_lineage(capsule_id: str, request: Request, authorization: str | None = Header(None)) -> dict[str, Any]:
    _authorized(authorization)
    service, principal, context = _bound(request)
    return _public(_call(lambda: service.get_lineage(capsule_id, principal=principal, context=context)))


@router.post("/{capsule_id}/stage")
def stage_capsule(capsule_id: str, body: StageRequest, request: Request, authorization: str | None = Header(None)) -> dict[str, Any]:
    _authorized(authorization)
    service, principal, context = _bound(request)
    capsule = _call(lambda: service.stage_capsule(
        capsule_id,
        principal=principal,
        context=context,
        target_scope=body.target_scope,
        strategy=body.strategy,
        team_id=body.team_id,
    ))
    return _public(capsule)


@router.post("/{capsule_id}/merge")
def merge_capsule(capsule_id: str, body: MergeRequest, request: Request, authorization: str | None = Header(None)) -> dict[str, Any]:
    _authorized(authorization)
    service, principal, context = _bound(request)
    record = _call(lambda: service.merge_capsule(
        capsule_id,
        principal=principal,
        context=context,
        request_id=body.request_id,
        target_capsule_id=body.target_capsule_id,
        expected_target_revision=body.expected_target_revision,
        expected_target_hash=body.expected_target_hash,
        strategy=body.strategy,
        decision_reason=body.decision_reason,
    ))
    return _public(record)


@router.post("/{capsule_id}/resolve-conflict")
def resolve_conflict(capsule_id: str, body: MergeRequest, request: Request, authorization: str | None = Header(None)) -> dict[str, Any]:
    _authorized(authorization)
    if body.strategy is None:
        raise ManaApiError(409, "Conflict resolution requires an explicit merge strategy.")
    service, principal, context = _bound(request)
    record = _call(lambda: service.resolve_conflict(
        capsule_id,
        principal=principal,
        context=context,
        request_id=body.request_id,
        target_capsule_id=body.target_capsule_id,
        expected_target_revision=body.expected_target_revision,
        expected_target_hash=body.expected_target_hash,
        strategy=body.strategy,
        decision_reason=body.decision_reason,
    ))
    return _public(record)


@router.delete("/{capsule_id}")
def delete_capsule(capsule_id: str, body: DeleteRequest, request: Request, authorization: str | None = Header(None)) -> dict[str, Any]:
    _authorized(authorization)
    service, principal, context = _bound(request)
    _call(lambda: service.delete_capsule(capsule_id, principal=principal, context=context, mode=body.mode))
    return {"deleted": True, "mode": body.mode.value}
