"""Pause and resume supervised tasks that depend on connector availability."""

from __future__ import annotations

import logging
from typing import Any, Callable

from .models import ConnectorHealthState, UnavailableStates

logger = logging.getLogger(__name__)


class ConnectorSupervisorBridge:
    """Translate connector health changes into durable branch checkpoints.

    When a connector becomes unavailable, dependent task branches are
    checkpointed/paused rather than retried into a known outage. Resume fires
    once when the connector returns to a usable state.
    """

    def __init__(
        self,
        *,
        supervisor: Any | None = None,
        list_dependent_tasks: Callable[[str], list[dict[str, Any]]] | None = None,
        clock=None,
    ) -> None:
        self.supervisor = supervisor
        self.list_dependent_tasks = list_dependent_tasks or (lambda _cid: [])
        self._paused: dict[str, set[str]] = {}  # connector_id -> task_ids
        self._resume_claims: set[str] = set()

    def bind_supervisor(self, supervisor: Any) -> None:
        self.supervisor = supervisor

    def on_health_change(
        self,
        connector_id: str,
        previous: ConnectorHealthState,
        current: ConnectorHealthState,
    ) -> list[str]:
        changed: list[str] = []
        became_unavailable = current in UnavailableStates or current is ConnectorHealthState.AUTH_REQUIRED
        became_available = (
            previous in UnavailableStates | {ConnectorHealthState.AUTH_REQUIRED, ConnectorHealthState.DEGRADED}
            and current is ConnectorHealthState.HEALTHY
        )
        if became_unavailable:
            changed.extend(self.pause_dependents(connector_id, state=current))
        if became_available:
            changed.extend(self.resume_dependents(connector_id))
        if current is ConnectorHealthState.DEGRADED:
            # Only pause branches that require a failed capability — callers pass
            # capability requirements via task metadata ``required_connector_capability``.
            for task_meta in self.list_dependent_tasks(connector_id):
                capability = str(task_meta.get("required_connector_capability") or "")
                if capability in {"ingress", "egress", "subscription"} and task_meta.get("capability_failed"):
                    changed.extend(
                        self.pause_dependents(
                            connector_id,
                            state=current,
                            only_task_ids={str(task_meta["task_id"])},
                        )
                    )
        return changed

    def pause_dependents(
        self,
        connector_id: str,
        *,
        state: ConnectorHealthState,
        only_task_ids: set[str] | None = None,
    ) -> list[str]:
        if self.supervisor is None:
            return []
        paused: list[str] = []
        for task_meta in self.list_dependent_tasks(connector_id):
            task_id = str(task_meta.get("task_id") or "")
            if not task_id:
                continue
            if only_task_ids is not None and task_id not in only_task_ids:
                continue
            if task_id in self._paused.get(connector_id, set()):
                continue
            checkpoint_id = str(task_meta.get("checkpoint_id") or "")
            try:
                if hasattr(self.supervisor, "suspend_for_connector"):
                    self.supervisor.suspend_for_connector(
                        task_id,
                        connector_id=connector_id,
                        checkpoint_id=checkpoint_id,
                        reason=f"connector_{state.value}",
                    )
                elif hasattr(self.supervisor, "checkpoint") and hasattr(self.supervisor, "store"):
                    # Best-effort: mark waiting_reason via connector fields when available.
                    def mark(task):
                        from mana_agent.execution_supervisor.models import ExecutionState
                        from mana_agent.execution_supervisor.state_machine import validate_transition

                        if task.state in {
                            ExecutionState.RUNNING,
                            ExecutionState.LEASED,
                            ExecutionState.QUEUED,
                            ExecutionState.CHECKPOINTING,
                        }:
                            validate_transition(task.state, ExecutionState.WAITING)
                            task.state = ExecutionState.WAITING
                        task.waiting_reason = "waiting_for_connector"
                        if hasattr(task, "waiting_connector_id"):
                            task.waiting_connector_id = connector_id
                        task.lease_owner = ""
                        task.lease_token = ""
                        task.lease_expires_at = None

                    self.supervisor.store.update_task(task_id, mark)
                self._paused.setdefault(connector_id, set()).add(task_id)
                paused.append(task_id)
                logger.info(
                    "connector.supervisor.paused task_id=%s connector_id=%s state=%s",
                    task_id,
                    connector_id,
                    state.value,
                )
            except Exception:
                logger.exception("failed to pause task %s for connector %s", task_id, connector_id)
        return paused

    def resume_dependents(self, connector_id: str) -> list[str]:
        if self.supervisor is None:
            return []
        resumed: list[str] = []
        task_ids = list(self._paused.get(connector_id, set()))
        for task_id in task_ids:
            claim = f"connector_resume:{connector_id}:{task_id}"
            if claim in self._resume_claims:
                continue
            try:
                if hasattr(self.supervisor, "resume_from_connector"):
                    self.supervisor.resume_from_connector(
                        task_id,
                        connector_id=connector_id,
                        resume_claim_id=claim,
                    )
                else:
                    def mark(task):
                        from mana_agent.execution_supervisor.models import ExecutionState
                        from mana_agent.execution_supervisor.state_machine import validate_transition

                        if task.state is ExecutionState.WAITING:
                            validate_transition(task.state, ExecutionState.QUEUED)
                            task.state = ExecutionState.QUEUED
                        if getattr(task, "waiting_reason", "") == "waiting_for_connector":
                            task.waiting_reason = ""
                        if hasattr(task, "waiting_connector_id"):
                            task.waiting_connector_id = ""

                    self.supervisor.store.update_task(task_id, mark)
                self._resume_claims.add(claim)
                self._paused.get(connector_id, set()).discard(task_id)
                resumed.append(task_id)
                logger.info(
                    "connector.supervisor.resumed task_id=%s connector_id=%s",
                    task_id,
                    connector_id,
                )
            except Exception:
                logger.exception("failed to resume task %s for connector %s", task_id, connector_id)
        return resumed
