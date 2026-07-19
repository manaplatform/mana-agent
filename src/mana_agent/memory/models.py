"""Canonical, provider-neutral memory models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class MemoryScope:
    user_id: str = ""
    agent_id: str = ""
    session_id: str = ""
    workspace_id: str = ""
    repository_id: str = ""
    conversation_id: str = ""
    task_id: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            name: value
            for name, value in (
                ("user_id", self.user_id),
                ("agent_id", self.agent_id),
                ("session_id", self.session_id),
                ("workspace_id", self.workspace_id),
                ("repository_id", self.repository_id),
                ("conversation_id", self.conversation_id),
                ("task_id", self.task_id),
            )
            if value
        }


@dataclass(frozen=True, slots=True)
class MemoryContent:
    text: str


@dataclass(slots=True)
class MemoryRecord:
    id: str
    content: MemoryContent
    scope: MemoryScope
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    provider: str = ""
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class MemoryWriteRequest:
    content: MemoryContent
    scope: MemoryScope
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemorySearchRequest:
    query: str
    scope: MemoryScope
    limit: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryUpdateRequest:
    content: MemoryContent | None = None
    scope: MemoryScope = field(default_factory=MemoryScope)
    metadata: dict[str, Any] | None = None


class MemoryHealthStatus(str, Enum):
    HEALTHY = "healthy"
    CONFIGURATION_ERROR = "configuration_failure"
    DEPENDENCY_ERROR = "missing_optional_dependency"
    AUTHENTICATION_ERROR = "authentication_failure"
    NETWORK_ERROR = "network_failure"
    PROVIDER_ERROR = "provider_failure"
    STORAGE_ERROR = "internal_storage_failure"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class MemoryHealth:
    status: MemoryHealthStatus
    mode: str
    provider: str
    detail: str = ""

    @property
    def healthy(self) -> bool:
        return self.status is MemoryHealthStatus.HEALTHY
