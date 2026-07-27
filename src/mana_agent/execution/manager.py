"""Execution fabric orchestration, lifecycle persistence, leases, and events."""

from __future__ import annotations

import asyncio
import threading
from datetime import timedelta
from typing import Any, Callable, Coroutine, TypeVar

from mana_agent.execution.config import ExecutionConfig
from mana_agent.execution.errors import CleanupError
from mana_agent.execution.lifecycle import validate_transition
from mana_agent.execution.models import (
    ExecutionRequest,
    ExecutionResult,
    RoutingRequest,
    SandboxExecutionContext,
    SandboxHandle,
    SandboxSpec,
    SandboxState,
    SnapshotRef,
    SnapshotRequest,
    utc_now,
)
from mana_agent.execution.registry import ProviderRegistry
from mana_agent.execution.router import ExecutionRouter
from mana_agent.execution.store import SandboxStore

T = TypeVar("T")
EventSink = Callable[[str, dict[str, Any]], None]


def run_sync(awaitable: Coroutine[Any, Any, T]) -> T:
    """Run an async provider operation from synchronous tool workers."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    result: list[T] = []
    failure: list[BaseException] = []
    def runner() -> None:
        try:
            result.append(asyncio.run(awaitable))
        except BaseException as exc:
            failure.append(exc)
    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return result[0]


class ExecutionManager:
    def __init__(
        self, registry: ProviderRegistry, config: ExecutionConfig, *,
        store: SandboxStore | None = None, event_sink: EventSink | None = None,
    ) -> None:
        self.registry = registry
        self.config = config
        self.router = ExecutionRouter(registry, config)
        self.store = store or SandboxStore()
        self.event_sink = event_sink
        self._global_limit = asyncio.Semaphore(config.global_concurrency_limit)
        self._provider_limits = {
            name: asyncio.Semaphore(config.providers[name].concurrency_limit)
            for name in config.providers
        }

    def _emit(self, kind: str, handle: SandboxHandle | None = None, **details: Any) -> None:
        if self.event_sink is None:
            return
        safe = {
            "sandbox_id": handle.sandbox_id if handle else "", "provider": handle.provider if handle else "",
            "state": handle.state.value if handle else "", "task_id": handle.task_id if handle else "",
            "session_id": handle.session_id if handle else "", **details,
        }
        self.event_sink(kind, safe)

    def _transition(self, handle: SandboxHandle, target: SandboxState) -> None:
        validate_transition(handle.state, target)
        handle.state = target
        handle.updated_at = utc_now()
        handle.last_heartbeat = handle.updated_at
        self.store.save(handle)

    async def create(self, spec: SandboxSpec, routing_request: RoutingRequest) -> SandboxExecutionContext:
        decision = await self.router.route(routing_request)
        self._emit("sandbox.routing.decided", selected_provider=decision.selected_provider, decision=decision.model_dump(mode="json"))
        provider = self.registry.get(decision.selected_provider)
        handle: SandboxHandle | None = None
        async with self._global_limit, self._provider_limits[decision.selected_provider]:
            try:
                self._emit("sandbox.provisioning.started", provider=decision.selected_provider, task_id=spec.task_id)
                handle = await provider.provision(spec)
                if handle.state == SandboxState.REQUESTED:
                    self._transition(handle, SandboxState.PROVISIONING)
                else:
                    self.store.save(handle)
                handle.metadata["secrets"] = [item.model_dump(mode="json") for item in spec.secrets]
                handle.metadata["routing_decision"] = decision.model_dump(mode="json")
                handle.lease_expires_at = utc_now() + timedelta(seconds=spec.max_lifetime_seconds)
                handle = await provider.start(handle)
                if handle.state != SandboxState.READY:
                    handle.state = SandboxState.READY
                handle.updated_at = utc_now()
                self.store.save(handle)
                self._emit("sandbox.provisioned", handle)
                self._emit("sandbox.started", handle)
                return SandboxExecutionContext(handle=handle, spec=spec, routing=decision)
            except Exception as exc:
                if handle is not None:
                    handle.state = SandboxState.FAILED
                    handle.failure = str(exc)[:2000]
                    self.store.save(handle)
                    self._emit("sandbox.failed", handle, error_type=type(exc).__name__)
                raise

    async def execute(self, context: SandboxExecutionContext, request: ExecutionRequest) -> ExecutionResult:
        handle = context.handle
        provider = self.registry.get(handle.provider)
        async with self._global_limit, self._provider_limits[handle.provider]:
            self._transition(handle, SandboxState.RUNNING)
            self._emit("sandbox.command.started", handle, argv0=request.argv[0], timeout_seconds=request.timeout_seconds)
            try:
                result = await provider.execute(handle, request)
                self._transition(handle, SandboxState.READY)
                self._emit("sandbox.command.completed", handle, exit_status=result.exit_code, duration_seconds=(result.completed_at - result.started_at).total_seconds())
                return result
            except BaseException as exc:
                handle.state = SandboxState.FAILED
                handle.failure = str(exc)[:2000]
                self.store.save(handle)
                self._emit("sandbox.failed", handle, error_type=type(exc).__name__)
                raise

    async def snapshot(self, context: SandboxExecutionContext, request: SnapshotRequest) -> SnapshotRef:
        snapshot = await self.registry.get(context.handle.provider).snapshot(context.handle, request)
        context.handle.snapshot_refs.append(snapshot.snapshot_id)
        self.store.save(context.handle)
        self._emit("sandbox.snapshot.created", context.handle, snapshot_id=snapshot.snapshot_id)
        return snapshot

    async def suspend(self, context: SandboxExecutionContext) -> None:
        self._transition(context.handle, SandboxState.SUSPENDING)
        context.handle = await self.registry.get(context.handle.provider).suspend(context.handle)
        context.handle.state = SandboxState.SUSPENDED
        self.store.save(context.handle)
        self._emit("sandbox.suspended", context.handle)

    async def resume(self, context: SandboxExecutionContext) -> None:
        self._transition(context.handle, SandboxState.RESUMING)
        context.handle = await self.registry.get(context.handle.provider).resume(context.handle)
        context.handle.state = SandboxState.READY
        self.store.save(context.handle)
        self._emit("sandbox.resumed", context.handle)

    async def terminate_and_cleanup(self, context: SandboxExecutionContext) -> None:
        handle = context.handle
        provider = self.registry.get(handle.provider)
        if handle.state == SandboxState.CLEANED:
            return
        try:
            if handle.state not in {SandboxState.TERMINATED, SandboxState.CLEANING}:
                if handle.state != SandboxState.TERMINATING:
                    self._transition(handle, SandboxState.TERMINATING)
                    self._emit("sandbox.terminating", handle)
                await provider.terminate(handle)
                handle.state = SandboxState.TERMINATED
                self.store.save(handle)
            handle.state = SandboxState.CLEANING
            self.store.save(handle)
            await provider.cleanup(handle)
            handle.state = SandboxState.CLEANED
            handle.cleanup_complete = True
            handle.updated_at = utc_now()
            self.store.save(handle)
            self._emit("sandbox.cleaned", handle, cleanup_result="ok")
        except Exception as exc:
            handle.cleanup_failure = str(exc)[:2000]
            self.store.save(handle)
            self._emit("sandbox.failed", handle, error_type="CleanupError")
            raise CleanupError("sandbox cleanup failed", provider=handle.provider, diagnostics=str(exc)) from exc

    async def execute_once(self, spec: SandboxSpec, routing: RoutingRequest, request: ExecutionRequest) -> ExecutionResult:
        context = await self.create(spec, routing)
        original: BaseException | None = None
        try:
            return await self.execute(context, request)
        except BaseException as exc:
            original = exc
            raise
        finally:
            if spec.cleanup_policy != "retain":
                try:
                    await self.terminate_and_cleanup(context)
                except CleanupError:
                    if original is None:
                        raise

    def execute_once_sync(self, spec: SandboxSpec, routing: RoutingRequest, request: ExecutionRequest) -> ExecutionResult:
        return run_sync(self.execute_once(spec, routing, request))
