"""Typed data models for the Global Self-Learning Memory System."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class LearningCategory(str, Enum):
    CODING = "CODING"
    API = "API"
    TOOL = "TOOL"
    COMMAND = "COMMAND"
    REASONING = "REASONING"
    PLANNING = "PLANNING"
    EXECUTION = "EXECUTION"
    ENVIRONMENT = "ENVIRONMENT"
    GENERAL = "GENERAL"


class LearningScope(str, Enum):
    GLOBAL = "GLOBAL"
    PROJECT = "PROJECT"
    TASK = "TASK"
    ENVIRONMENT = "ENVIRONMENT"


class LearningStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    VERIFIED = "VERIFIED"
    DISPUTED = "DISPUTED"
    SUPERSEDED = "SUPERSEDED"


@dataclass(slots=True)
class LearningMemory:
    id: str
    category: LearningCategory
    scope: LearningScope
    context: str
    attempted_action: str
    failure_signature: str | None = None
    error: str = ""
    root_cause: str | None = None
    correction: str | None = None
    prevention_rule: str | None = None
    status: LearningStatus = LearningStatus.CANDIDATE
    confidence: float = 0.5
    source_execution_id: str = ""
    evidence_references: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    last_applied_at: datetime | None = None
    success_count: int = 0
    failure_count: int = 1
    project_id: str | None = None
    environment_fingerprint: str | None = None
    dependency_versions: dict[str, str] = field(default_factory=dict)
    ownership: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    expires_at: datetime | None = None
    superseded_by: str | None = None
    replaces_memory_id: str | None = None
    is_hypothesis: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        if self.status in {LearningStatus.SUPERSEDED, LearningStatus.DISPUTED}:
            return False
        if self.expires_at and self.expires_at <= utc_now():
            return False
        return True

    def is_applicable(
        self,
        project_id: str | None = None,
        environment_fingerprint: str | None = None,
    ) -> bool:
        if not self.is_active():
            return False
        if self.scope == LearningScope.GLOBAL:
            return True
        if self.scope == LearningScope.PROJECT:
            if project_id and self.project_id:
                return self.project_id == project_id
            return True
        if self.scope == LearningScope.ENVIRONMENT:
            if environment_fingerprint and self.environment_fingerprint:
                return self.environment_fingerprint == environment_fingerprint
            return True
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value if isinstance(self.category, LearningCategory) else str(self.category),
            "scope": self.scope.value if isinstance(self.scope, LearningScope) else str(self.scope),
            "context": self.context,
            "attempted_action": self.attempted_action,
            "failure_signature": self.failure_signature,
            "error": self.error,
            "root_cause": self.root_cause,
            "correction": self.correction,
            "prevention_rule": self.prevention_rule,
            "status": self.status.value if isinstance(self.status, LearningStatus) else str(self.status),
            "confidence": self.confidence,
            "source_execution_id": self.source_execution_id,
            "evidence_references": list(self.evidence_references),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_applied_at": self.last_applied_at.isoformat() if self.last_applied_at else None,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "project_id": self.project_id,
            "environment_fingerprint": self.environment_fingerprint,
            "dependency_versions": dict(self.dependency_versions),
            "ownership": self.ownership,
            "provenance": dict(self.provenance),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "superseded_by": self.superseded_by,
            "replaces_memory_id": self.replaces_memory_id,
            "is_hypothesis": self.is_hypothesis,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearningMemory:
        payload = dict(data)
        category_val = payload.get("category", LearningCategory.GENERAL.value)
        scope_val = payload.get("scope", LearningScope.GLOBAL.value)
        status_val = payload.get("status", LearningStatus.CANDIDATE.value)

        category = LearningCategory(category_val) if category_val in LearningCategory._value2member_map_ else LearningCategory.GENERAL
        scope = LearningScope(scope_val) if scope_val in LearningScope._value2member_map_ else LearningScope.GLOBAL
        status = LearningStatus(status_val) if status_val in LearningStatus._value2member_map_ else LearningStatus.CANDIDATE

        def parse_datetime(val: Any) -> datetime | None:
            if not val:
                return None
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val)
                except ValueError:
                    return None
            return None

        created = parse_datetime(payload.get("created_at")) or utc_now()
        updated = parse_datetime(payload.get("updated_at")) or utc_now()
        last_applied = parse_datetime(payload.get("last_applied_at"))
        expires = parse_datetime(payload.get("expires_at"))

        return cls(
            id=str(payload["id"]),
            category=category,
            scope=scope,
            context=str(payload.get("context", "")),
            attempted_action=str(payload.get("attempted_action", "")),
            failure_signature=str(payload.get("failure_signature")) if payload.get("failure_signature") is not None else None,
            error=str(payload.get("error", "")),
            root_cause=str(payload.get("root_cause")) if payload.get("root_cause") is not None else None,
            correction=str(payload.get("correction")) if payload.get("correction") is not None else None,
            prevention_rule=str(payload.get("prevention_rule")) if payload.get("prevention_rule") is not None else None,
            status=status,
            confidence=float(payload.get("confidence", 0.5)),
            source_execution_id=str(payload.get("source_execution_id", "")),
            evidence_references=list(payload.get("evidence_references") or []),
            created_at=created,
            updated_at=updated,
            last_applied_at=last_applied,
            success_count=int(payload.get("success_count", 0)),
            failure_count=int(payload.get("failure_count", 1)),
            project_id=str(payload.get("project_id")) if payload.get("project_id") is not None else None,
            environment_fingerprint=str(payload.get("environment_fingerprint")) if payload.get("environment_fingerprint") is not None else None,
            dependency_versions=dict(payload.get("dependency_versions") or {}),
            ownership=str(payload.get("ownership")) if payload.get("ownership") is not None else None,
            provenance=dict(payload.get("provenance") or {}),
            expires_at=expires,
            superseded_by=str(payload.get("superseded_by")) if payload.get("superseded_by") is not None else None,
            replaces_memory_id=str(payload.get("replaces_memory_id")) if payload.get("replaces_memory_id") is not None else None,
            is_hypothesis=bool(payload.get("is_hypothesis", False)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(slots=True)
class LearningHistoryEntry:
    id: str
    memory_id: str
    action: str
    timestamp: datetime = field(default_factory=utc_now)
    execution_id: str = ""
    evidence: str = ""
    previous_confidence: float = 0.0
    new_confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "memory_id": self.memory_id,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
            "execution_id": self.execution_id,
            "evidence": self.evidence,
            "previous_confidence": self.previous_confidence,
            "new_confidence": self.new_confidence,
            "details": self.details,
        }
