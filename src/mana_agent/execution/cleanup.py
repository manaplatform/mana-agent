"""Lease-based restart recovery and orphan cleanup."""

from __future__ import annotations

from mana_agent.execution.manager import ExecutionManager
from mana_agent.execution.models import RoutingDecision, SandboxExecutionContext, SandboxSpec, SandboxState, utc_now


class CleanupController:
    def __init__(self, manager: ExecutionManager) -> None:
        self.manager = manager

    async def recover_expired(self) -> list[str]:
        recovered: list[str] = []
        for handle in self.manager.store.list():
            if handle.state == SandboxState.CLEANED:
                continue
            expired = handle.lease_expires_at is not None and handle.lease_expires_at <= utc_now()
            interrupted = handle.state in {SandboxState.PROVISIONING, SandboxState.TERMINATING, SandboxState.CLEANING}
            if not expired and not interrupted:
                continue
            source = handle.workspace_path or "."
            try:
                spec = SandboxSpec(repository_source=source, task_id=handle.task_id, session_id=handle.session_id, workspace_id=handle.workspace_id)
            except ValueError:
                # Remote workspace paths need not exist locally. The provider has
                # enough persisted handle metadata to clean them.
                spec = SandboxSpec.model_construct(repository_source=source, task_id=handle.task_id, session_id=handle.session_id, workspace_id=handle.workspace_id)
            context = SandboxExecutionContext(
                handle=handle, spec=spec,
                routing=RoutingDecision(decision_id="restart-recovery", selected_provider=handle.provider, requirements_considered=["expired-lease" if expired else "interrupted-lifecycle"], policy_rule="recovery"),
            )
            await self.manager.terminate_and_cleanup(context)
            recovered.append(handle.sandbox_id)
        return recovered
