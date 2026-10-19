"""External coding engine delegating execution to configured coding providers (e.g. Codex)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, Callable

from mana_agent.coding.engine import CodingEngine
from mana_agent.coding.models import (
    AgentEvent,
    CodingEngineMode,
    CodingStatus,
    CodingTask,
    CodingTaskResult,
    WorkspaceContext,
)

logger = logging.getLogger(__name__)


class CodexProviderAdapter:
    """Provider adapter wrapping CodexCodingBackend."""

    def __init__(self, backend_factory: Callable[[], Any]) -> None:
        self._backend_factory = backend_factory
        self._backend: Any | None = None

    def _get_backend(self) -> Any:
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    async def start(self) -> None:
        backend = self._get_backend()
        if hasattr(backend, "start"):
            await backend.start()

    async def execute(self, task: CodingTask, workspace: WorkspaceContext) -> CodingTaskResult:
        backend = self._get_backend()
        return await backend.execute(task, workspace)

    def stream(self, task: CodingTask, workspace: WorkspaceContext) -> AsyncIterator[AgentEvent]:
        backend = self._get_backend()
        return backend.stream(task, workspace)

    async def cancel(self, task_id: str) -> None:
        backend = self._get_backend()
        if hasattr(backend, "cancel"):
            try:
                await backend.cancel(task_id, timeout_seconds=2.0)
            except TypeError:
                await backend.cancel(task_id)

    async def close(self) -> None:
        if self._backend is not None and hasattr(self._backend, "close"):
            try:
                await self._backend.close()
            finally:
                self._backend = None


class ExternalCodingEngine(CodingEngine):
    """External coding engine that delegates to external providers without altering their tool loop."""

    def __init__(
        self,
        provider_adapter: CodexProviderAdapter | None = None,
        *,
        backend_factory: Callable[[], Any] | None = None,
        global_memory: Any | None = None,
    ) -> None:
        self.mode: CodingEngineMode = "external"
        self.name: str = "codex"
        if provider_adapter is not None:
            self._adapter = provider_adapter
        elif backend_factory is not None:
            self._adapter = CodexProviderAdapter(backend_factory)
        else:
            raise ValueError("ExternalCodingEngine requires either provider_adapter or backend_factory")
        self._task_statuses: dict[str, CodingStatus] = {}
        if global_memory is not None:
            self.global_memory = global_memory
        else:
            try:
                from mana_agent.memory.learning import get_global_memory
                self.global_memory = get_global_memory()
            except Exception:
                self.global_memory = None

    async def start(self) -> None:
        await self._adapter.start()

    async def execute(self, task: CodingTask, workspace: WorkspaceContext) -> CodingTaskResult:
        self._task_statuses[task.task_id] = CodingStatus(
            task_id=task.task_id,
            status="running",
            backend=self.name,
        )
        effective_task = task
        if self.global_memory is not None:
            try:
                learnings = self.global_memory.retrieve_relevant(
                    action_or_query=task.user_request or task.objective,
                    limit=5,
                )
                if learnings:
                    rules_header = self.global_memory.format_retrieved_rules(learnings)
                    enriched_request = f"{rules_header}\n\n{task.user_request}"
                    effective_task = CodingTask(
                        task_id=task.task_id,
                        objective=task.objective,
                        user_request=enriched_request,
                        file_paths=task.file_paths,
                        context=task.context,
                        constraints=task.constraints,
                        expected_deliverables=task.expected_deliverables,
                        assigned_model=task.assigned_model,
                        timeout_seconds=task.timeout_seconds,
                    )
            except Exception as gm_exc:
                logger.debug("ExternalCodingEngine memory retrieval failed safely: %s", gm_exc)

        try:
            result = await self._adapter.execute(effective_task, workspace)
            self._task_statuses[task.task_id] = CodingStatus(
                task_id=task.task_id,
                status=result.status,
                backend=self.name,
                duration_ms=result.duration_breakdown.get("total_task_duration_ms"),
                changed_files=list(result.changed_files),
                error="; ".join(result.errors) if result.errors else "",
            )
            if self.global_memory is not None and result.errors:
                try:
                    self.global_memory.record_failure(
                        attempted_action=f"codex_task:{task.objective[:80]}",
                        error="; ".join(result.errors),
                        category="CODING",
                        context=f"Files: {list(task.file_paths)}",
                        execution_id=task.task_id,
                    )
                except Exception as gm_exc:
                    logger.debug("ExternalCodingEngine record failure failed safely: %s", gm_exc)
            return result
        except BaseException as exc:
            self._task_statuses[task.task_id] = CodingStatus(
                task_id=task.task_id,
                status="cancelled" if type(exc).__name__ == "CancelledError" else "failed",
                backend=self.name,
                error=str(exc),
            )
            raise

    async def stream(self, task: CodingTask, workspace: WorkspaceContext) -> AsyncIterator[AgentEvent]:
        self._task_statuses[task.task_id] = CodingStatus(
            task_id=task.task_id,
            status="running",
            backend=self.name,
        )
        async for event in self._adapter.stream(task, workspace):
            yield event

    async def cancel(self, task_id: str) -> None:
        await self._adapter.cancel(task_id)
        if task_id in self._task_statuses:
            prev = self._task_statuses[task_id]
            self._task_statuses[task_id] = CodingStatus(
                task_id=task_id,
                status="cancelled",
                backend=self.name,
                changed_files=prev.changed_files,
                error="Task was cancelled",
            )

    async def get_status(self, task_id: str) -> CodingStatus:
        if task_id in self._task_statuses:
            return self._task_statuses[task_id]
        return CodingStatus(
            task_id=task_id,
            status="queued",
            backend=self.name,
        )

    async def close(self) -> None:
        await self._adapter.close()


__all__ = ["CodexProviderAdapter", "ExternalCodingEngine"]
