"""Typed, provider-neutral contracts for scoped shared-memory capsules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CapsuleScope(str, Enum):
    PRIVATE = "private"
    PARENT_CHILD = "parent_child"
    TEAM = "team"
    PROJECT = "project"
    ORGANISATION = "organisation"
    USER = "user"


class TrustState(str, Enum):
    UNTRUSTED = "untrusted"
    AGENT_GENERATED = "agent_generated"
    USER_PROVIDED = "user_provided"
    TOOL_VERIFIED = "tool_verified"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


class ReviewState(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CONFLICT = "conflict"


class MergeState(str, Enum):
    NONE = "none"
    STAGED = "staged"
    MERGED = "merged"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    CONFLICT = "conflict"


class MergeStrategy(str, Enum):
    APPEND = "append"
    REPLACE = "replace"
    PATCH = "patch"
    SUPERSEDE = "supersede"
    REJECT = "reject"


class DeleteMode(str, Enum):
    SOFT = "soft"
    PERMANENT = "permanent"
    REDACT = "redact"


@dataclass(frozen=True, slots=True)
class MemoryPrincipal:
    user_id: str | None = None
    organisation_id: str | None = None
    project_id: str | None = None
    team_ids: frozenset[str] = field(default_factory=frozenset)
    task_id: str | None = None
    parent_task_id: str | None = None
    agent_id: str | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)

    def has(self, capability: str) -> bool:
        return capability in self.capabilities

    def audit_ids(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "organisation_id": self.organisation_id,
            "project_id": self.project_id,
            "team_ids": sorted(self.team_ids),
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "agent_id": self.agent_id,
        }


@dataclass(frozen=True, slots=True)
class CapsuleTaskContext:
    user_id: str | None
    organisation_id: str | None
    project_id: str | None
    team_ids: frozenset[str]
    task_id: str
    parent_task_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    task_completed: bool = False


@dataclass(slots=True)
class MemoryCapsule:
    capsule_id: str
    schema_version: int
    scope: CapsuleScope
    namespace: str
    owner_user_id: str | None
    organisation_id: str | None
    project_id: str | None
    team_id: str | None
    task_id: str
    parent_task_id: str | None
    agent_id: str | None
    session_id: str | None
    title: str
    summary: str
    content: dict[str, Any]
    tags: list[str]
    origin_type: str
    origin_id: str
    source_capsule_ids: list[str]
    trust_state: TrustState
    review_state: ReviewState
    merge_state: MergeState
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None
    created_by: MemoryPrincipal
    updated_by: MemoryPrincipal
    content_hash: str
    revision: int
    provider: str = "mana"
    proposed_scope: CapsuleScope | None = None
    proposed_namespace: str | None = None
    supporting_evidence: list[str] = field(default_factory=list)
    requested_operation: MergeStrategy | None = None
    risk_flags: list[str] = field(default_factory=list)
    deleted_at: datetime | None = None
    superseded_by: str | None = None

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= utc_now()


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    allowed: bool
    reason_code: str
    reason: str
    matched_policy: str | None = None


@dataclass(frozen=True, slots=True)
class CapsuleReadRequest:
    principal: MemoryPrincipal
    task_context: CapsuleTaskContext
    query: str = ""
    allowed_scopes: frozenset[CapsuleScope] = field(default_factory=frozenset)
    namespaces: frozenset[str] = field(default_factory=frozenset)
    max_capsules: int = 12
    max_tokens: int = 4000
    include_staged: bool = False


@dataclass(frozen=True, slots=True)
class CapsuleProjection:
    capsule_id: str
    scope: CapsuleScope
    namespace: str
    title: str
    summary: str
    content: dict[str, Any]
    tags: tuple[str, ...]
    trust_state: TrustState
    origin_type: str
    origin_id: str
    source_capsule_ids: tuple[str, ...]
    revision: int
    content_hash: str
    provider: str
    created_at: datetime
    expires_at: datetime | None


@dataclass(frozen=True, slots=True)
class CapsuleMergeRecord:
    merge_id: str
    source_capsule_ids: tuple[str, ...]
    target_capsule_id: str | None
    strategy: MergeStrategy
    expected_target_revision: int | None
    expected_target_hash: str | None
    resulting_capsule_id: str | None
    reviewed_by: MemoryPrincipal | None
    decision_reason: str | None
    created_at: datetime
    request_id: str
    conflict: bool = False


@dataclass(frozen=True, slots=True)
class CapsuleLineage:
    capsule_id: str
    ancestors: tuple[str, ...]
    descendants: tuple[str, ...]
    merges: tuple[CapsuleMergeRecord, ...]
    consumers: tuple[dict[str, Any], ...]
    superseded_by: str | None
    approved_by: MemoryPrincipal | None
    revision_history: tuple[dict[str, Any], ...] = ()


READ_CAPABILITY = {
    scope: f"memory.capsule.read.{scope.value}" for scope in CapsuleScope
}
WRITE_CAPABILITY = {
    CapsuleScope.PRIVATE: "memory.capsule.write.private",
    CapsuleScope.PARENT_CHILD: "memory.capsule.write.parent_child",
    CapsuleScope.USER: "memory.capsule.write.user",
}
STAGE_CAPABILITY = {
    CapsuleScope.TEAM: "memory.capsule.stage.team",
    CapsuleScope.PROJECT: "memory.capsule.stage.project",
}
MERGE_CAPABILITY = {
    CapsuleScope.TEAM: "memory.capsule.merge.team",
    CapsuleScope.PROJECT: "memory.capsule.merge.project",
    CapsuleScope.ORGANISATION: "memory.capsule.merge.organisation",
}
