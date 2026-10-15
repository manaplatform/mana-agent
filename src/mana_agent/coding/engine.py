"""Shared abstraction protocol for internal and external coding engines."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from mana_agent.coding.models import (
    AgentEvent,
    CodingEngineMode,
    CodingStatus,
    CodingTask,
    CodingTaskResult,
    WorkspaceContext,
)


@runtime_checkable
class CodingEngine(Protocol):
    name: str
    mode: CodingEngineMode

    async def start(self) -> None: ...

    async def execute(self, task: CodingTask, workspace: WorkspaceContext) -> CodingTaskResult: ...

    def stream(self, task: CodingTask, workspace: WorkspaceContext) -> AsyncIterator[AgentEvent]: ...

    async def cancel(self, task_id: str) -> None: ...

    async def get_status(self, task_id: str) -> CodingStatus: ...

    async def close(self) -> None: ...


__all__ = ["CodingEngine"]
