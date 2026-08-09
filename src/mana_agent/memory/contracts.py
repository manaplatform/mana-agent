"""Backend contract implemented by every memory provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from mana_agent.memory.models import (
    MemoryHealth,
    MemoryRecord,
    MemoryScope,
    MemorySearchRequest,
    MemoryUpdateRequest,
    MemoryWriteRequest,
)


@dataclass(frozen=True, slots=True)
class MemoryCapabilities:
    """Declare which memory domains a service instance can satisfy.

    AI/semantic domains are provider-selected. System-state domains (run
    evidence, coding-flow checkpoints, local task continuity) remain available
    even when the selected provider is external, because they are runtime
    durable stores rather than hosted AI memory.
    """

    conversation: bool = False
    semantic_search: bool = False
    evidence: bool = True
    checkpoints: bool = True
    coding_flow: bool = True
    task_state: bool = True
    multi_agent_runtime: bool = False

    def supports(self, capability: str) -> bool:
        return bool(getattr(self, capability, False))

    def as_dict(self) -> dict[str, bool]:
        return {
            "conversation": self.conversation,
            "semantic_search": self.semantic_search,
            "evidence": self.evidence,
            "checkpoints": self.checkpoints,
            "coding_flow": self.coding_flow,
            "task_state": self.task_state,
            "multi_agent_runtime": self.multi_agent_runtime,
        }


@runtime_checkable
class MemoryBackend(Protocol):
    async def add(self, request: MemoryWriteRequest) -> MemoryRecord: ...
    async def search(self, request: MemorySearchRequest) -> list[MemoryRecord]: ...
    async def get(self, memory_id: str, scope: MemoryScope) -> MemoryRecord | None: ...
    async def update(self, memory_id: str, request: MemoryUpdateRequest) -> MemoryRecord: ...
    async def delete(self, memory_id: str, scope: MemoryScope) -> None: ...
    async def clear(self, scope: MemoryScope) -> None: ...
    async def healthcheck(self) -> MemoryHealth: ...
    async def close(self) -> None: ...


@runtime_checkable
class CapsuleBackend(Protocol):
    """Storage contract selected only behind the authorization service."""

    def create_capsule(self, capsule: Any) -> None: ...
    def get_capsule(self, capsule_id: str) -> Any | None: ...
    def query_capsules(self) -> list[Any]: ...
    def update_capsule(self, capsule: Any, *, expected_revision: int) -> None: ...
    def stage_capsule(self, capsule: Any) -> None: ...
    def list_staged_capsules(self) -> list[Any]: ...
    def merge_capsule(self, record: Any) -> None: ...
    def delete_capsule(self, capsule_id: str) -> bool: ...
    def get_lineage(self, capsule_id: str) -> Any: ...
    def record_access(self, entry: dict[str, Any]) -> None: ...
