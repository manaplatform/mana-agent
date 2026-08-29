from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Callable

from mana_agent.multi_agent.core.ids import new_task_id
from mana_agent.multi_agent.core.errors import InvalidTaskTransition
from mana_agent.multi_agent.core.types import (
    DecisionRecord,
    HandoffRecord,
    RiskLevel,
    TaskBoardItem,
    TaskStatus,
    VerificationResult,
    utc_now,
)
from mana_agent.memory import MultiAgentMemoryService, task_fingerprint
from mana_agent.memory import CapsuleTaskContext, MemoryPrincipal
from mana_agent.context_cost.artifact_store import ContextArtifactStore
from mana_agent.context_cost.compression import compress_tool_result, render_envelope
from mana_agent.context_cost.estimator import estimate_value_tokens
from mana_agent.multi_agent.taskboard.store import JsonStateStore, serialize, task_from_dict
from mana_agent.multi_agent.taskboard.validators import validate_transition


def _append_unique(target: list[str], values: list[str]) -> None:
    seen = set(target)
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            target.append(text)
            seen.add(text)


class TaskBoard:
    def __init__(
        self,
        root: str | Path = ".",
        *,
        memory_service: MultiAgentMemoryService | None = None,
        task_id_is_reserved: Callable[[str], bool] | None = None,
    ) -> None:
        self.store = JsonStateStore(root)
        self._save_lock = threading.RLock()
        self.memory_service = memory_service
        self._task_id_is_reserved = task_id_is_reserved
        self.tasks: dict[str, TaskBoardItem] = {}
        self.load()

    def _new_task_id(self) -> str:
        """Allocate an ID unused by this projection or an authoritative store."""
        task_id = new_task_id()
        while task_id in self.tasks or (
            self._task_id_is_reserved is not None
            and self._task_id_is_reserved(task_id)
        ):
            task_id = new_task_id()
        return task_id

    def set_task_id_reservation_checker(
        self, checker: Callable[[str], bool] | None
    ) -> None:
        """Bind authoritative task-ID reservations before creating new tasks."""
        self._task_id_is_reserved = checker

    def create_task(
        self,
        *,
        title: str,
        user_request: str,
        normalized_goal: str | None = None,
        priority: int = 100,
        risk_level: RiskLevel = RiskLevel.LOW,
        owner_agent_id: str | None = None,
        related_files: list[str] | None = None,
        action_type: str = "task",
        expected_output: str = "",
        workspace_id: str | None = None,
        session_id: str | None = None,
        repository_ids: list[str] | None = None,
        primary_repository_id: str | None = None,
        trigger_turn_id: str = "",
        relation_type: str = "independent",
        previous_task_id: str = "",
        wiring_required: bool = False,
        wiring_reason: str | None = None,
    ) -> TaskBoardItem:
        task_id = self._new_task_id()
        goal = normalized_goal or user_request.strip()
        duplicate_of = None
        memory_bundle_id = None
        fingerprint = task_fingerprint(
            normalized_goal=goal,
            action_type=action_type,
            target_files=related_files or [],
            expected_output=expected_output,
            root=self.store.root,
            repository_ids=repository_ids or [self.store.repository_id],
        )
        if self.memory_service is not None:
            memory_goal, fingerprint = self.memory_service.normalize_task(
                goal=goal,
                action_type=action_type,
                target_files=related_files or [],
                expected_output=expected_output,
                repository_ids=repository_ids or [self.store.repository_id],
            )
            record = self.memory_service.register_task(
                task_id=task_id,
                normalized_goal=memory_goal,
                fingerprint=fingerprint,
                assigned_agent_id=owner_agent_id or "",
                related_files=related_files or [],
                repository_ids=repository_ids or [self.store.repository_id],
            )
            duplicate_of = record.duplicate_of
            if not self.memory_service.config.capsules.enabled:
                bundle = self.memory_service.build_bundle(
                    agent_id=owner_agent_id or "agent_taskboard",
                    agent_role="taskboard",
                    task_id=task_id,
                    target_files=related_files or [],
                )
                memory_bundle_id = bundle.bundle_id
        task = TaskBoardItem(
            task_id=task_id,
            parent_task_id=None,
            root_task_id=task_id,
            title=title,
            user_request=user_request,
            normalized_goal=goal,
            # Duplicate detection is advisory. The execution supervisor decides
            # whether the durable work is resumed, reverified, reused, or superseded.
            status=TaskStatus.NEW,
            priority=priority,
            risk_level=risk_level,
            workspace_id=workspace_id or self.store.workspace_id,
            session_id=session_id or "",
            trigger_turn_id=trigger_turn_id,
            relation_type=relation_type,
            previous_task_id=previous_task_id,
            primary_repository_id=primary_repository_id or self.store.repository_id,
            repository_ids=list(repository_ids or [self.store.repository_id]),
            owner_agent_id=owner_agent_id,
            supervisor_agent_id=owner_agent_id,
            delegated_by_agent_id=owner_agent_id,
            approved_by_agent_id=owner_agent_id,
            budget_reserved_tokens=0,
            budget_remaining_tokens=0,
            budget_reserved_ms=120_000,
            blockers=[],
            wiring_required=wiring_required,
            wiring_reason=wiring_reason,
            memory_status={
                "duplicate_checked": True,
                "duplicate_of": duplicate_of,
                "cache_hits": 0,
                "file_reads_reused": 0,
                "memory_bundle_id": memory_bundle_id,
                "capsule_bundle_required": bool(self.memory_service and self.memory_service.config.capsules.enabled),
                "last_memory_check_at": utc_now(),
                "fingerprint": fingerprint,
            },
        )
        self.tasks[task_id] = task
        self._record("task.created", task)
        self.save()
        return task

    def create_child_task(
        self,
        parent_task_id: str,
        *,
        title: str,
        user_request: str = "",
        owner_agent_id: str | None = None,
        acceptance_criteria: list[str] | None = None,
        plan: list[str] | None = None,
        depends_on: list[str] | None = None,
        decomposition_local_id: str = "",
        preferred_parallelism: str = "automatic",
        trigger_turn_id: str = "",
        relation_type: str = "followup",
        previous_task_id: str = "",
        parent_memory_principal: MemoryPrincipal | None = None,
        delegated_capsule_ids: list[str] | None = None,
        integration_role: str = "",
    ) -> TaskBoardItem:
        parent = self.get_task(parent_task_id)
        task_id = self._new_task_id()
        req_text = str(user_request or parent.user_request or title).strip()
        task = TaskBoardItem(
            task_id=task_id,
            parent_task_id=parent_task_id,
            root_task_id=parent.root_task_id,
            title=title,
            user_request=req_text,
            normalized_goal=req_text,
            status=TaskStatus.NEW,
            priority=parent.priority,
            risk_level=parent.risk_level,
            workspace_id=parent.workspace_id,
            session_id=parent.session_id,
            trigger_turn_id=trigger_turn_id,
            relation_type=relation_type,
            previous_task_id=previous_task_id or parent_task_id,
            primary_repository_id=parent.primary_repository_id,
            repository_ids=list(parent.repository_ids),
            managed_workspace_id=parent.managed_workspace_id,
            managed_branch=parent.managed_branch,
            managed_worktree_path=parent.managed_worktree_path,
            workspace_status=parent.workspace_status,
            base_revision=parent.base_revision,
            execution_repo_root=parent.execution_repo_root,
            owner_agent_id=owner_agent_id,
            supervisor_agent_id=parent.owner_agent_id,
            delegated_by_agent_id=parent.owner_agent_id,
            budget_reserved_tokens=parent.budget_reserved_tokens,
            budget_remaining_tokens=parent.budget_remaining_tokens,
            budget_reserved_ms=parent.budget_reserved_ms,
            acceptance_criteria=list(acceptance_criteria or []),
            plan=list(plan or []),
            depends_on=list(depends_on or []),
            integration_role=integration_role,
            decomposition_local_id=decomposition_local_id,
            preferred_parallelism=preferred_parallelism,
            memory_status={
                "duplicate_checked": False,
                "duplicate_of": None,
                "cache_hits": 0,
                "file_reads_reused": 0,
                "memory_bundle_id": None,
                "last_memory_check_at": utc_now(),
            },
        )
        self.tasks[task_id] = task
        _append_unique(parent.child_task_ids, [task_id])
        if integration_role:
            _append_unique(parent.depends_on, [task_id])
            _append_unique(parent.required_wiring_task_ids, [task_id])
        if decomposition_local_id:
            parent.decomposition_id_map[decomposition_local_id] = task_id
        parent.updated_at = utc_now()
        if delegated_capsule_ids:
            if self.memory_service is None or parent_memory_principal is None:
                raise ValueError("Explicit capsule delegation requires a memory service and parent principal.")
            parent_context = CapsuleTaskContext(
                user_id=parent_memory_principal.user_id,
                organisation_id=parent_memory_principal.organisation_id,
                project_id=parent_memory_principal.project_id,
                team_ids=parent_memory_principal.team_ids,
                task_id=parent.task_id,
                parent_task_id=parent.parent_task_id,
                agent_id=parent_memory_principal.agent_id,
                session_id=parent.session_id,
            )
            child_context = CapsuleTaskContext(
                user_id=parent_memory_principal.user_id,
                organisation_id=parent_memory_principal.organisation_id,
                project_id=parent_memory_principal.project_id,
                team_ids=parent_memory_principal.team_ids,
                task_id=task.task_id,
                parent_task_id=parent.task_id,
                agent_id=owner_agent_id,
                session_id=task.session_id,
            )
            delegated = self.memory_service.capsules.delegate_to_child(
                delegated_capsule_ids,
                parent_principal=parent_memory_principal,
                parent_context=parent_context,
                child_context=child_context,
                correlation_id=trigger_turn_id,
            )
            task.memory_status["delegated_capsules"] = [
                {"capsule_id": item.capsule_id, "revision": item.revision}
                for item in delegated
            ]
        self._record("task.created", task)
        self.save()
        return task

    def update_orchestration(
        self,
        task_id: str,
        *,
        entry_route: str | None = None,
        owning_lane: str | None = None,
        routing_evidence: dict[str, Any] | None = None,
        result_summary: str | None = None,
        verification_status: str | None = None,
        output_artifacts: list[str] | None = None,
        approval_request_ids: list[str] | None = None,
        aggregate_progress: str | None = None,
    ) -> None:
        task = self.get_task(task_id)
        for name, value in (
            ("entry_route", entry_route),
            ("owning_lane", owning_lane),
            ("result_summary", result_summary),
            ("verification_status", verification_status),
            ("aggregate_progress", aggregate_progress),
        ):
            if value is not None:
                setattr(task, name, str(value))
        if routing_evidence is not None:
            task.routing_evidence = dict(routing_evidence)
        if output_artifacts is not None:
            task.output_artifacts = list(output_artifacts)
        if approval_request_ids is not None:
            task.approval_request_ids = list(approval_request_ids)
        task.updated_at = utc_now()
        self._record("task.orchestration_updated", {"task_id": task_id})
        self.save()

    def get_task(self, task_id: str) -> TaskBoardItem:
        return self.tasks[task_id]

    def update_status(self, task_id: str, status: TaskStatus, *, reason: str | None = None) -> None:
        task = self.get_task(task_id)
        if status == TaskStatus.DONE:
            if not task.wiring_required and task.integration_role != "wiring":
                task.wiring_outcome = "not_required"
            elif task.wiring_outcome not in {"mutation_applied", "already_integrated", "completed"}:
                task.wiring_outcome = "completed"
            self._validate_feature_completion(task)
        validate_transition(task, status, reason=reason)
        task.status = status
        task.updated_at = utc_now()
        if status == TaskStatus.DONE:
            if not task.wiring_required and task.integration_role != "wiring":
                task.wiring_outcome = "not_required"
            elif task.wiring_outcome not in {"mutation_applied", "already_integrated", "completed"}:
                task.wiring_outcome = "completed"
        elif status == TaskStatus.FAILED:
            task.wiring_outcome = "failed"
            if reason and not task.wiring_outcome_reason:
                task.wiring_outcome_reason = str(reason)
            if task.integration_role == "wiring" or (task.parent_task_id and task.parent_task_id in self.tasks):
                parent = self.tasks.get(task.parent_task_id) if task.parent_task_id else None
                if parent is not None and (task.integration_role == "wiring" or task.task_id in parent.required_wiring_task_ids):
                    parent.status = TaskStatus.FAILED
                    parent.wiring_outcome = "failed"
                    parent.wiring_outcome_reason = str(reason or task.wiring_outcome_reason or "")
                    parent.updated_at = utc_now()
                    self._record(
                        "task.updated",
                        {"task_id": parent.task_id, "status": TaskStatus.FAILED.value, "reason": parent.wiring_outcome_reason},
                    )
        elif status == TaskStatus.BLOCKED:
            if reason:
                self.add_blocker(task_id, reason, save=False)
            if task.wiring_outcome not in {"completed", "failed"}:
                task.wiring_outcome = "blocked"
        elif status in {TaskStatus.IN_PROGRESS, TaskStatus.WAITING_FOR_TOOLS, TaskStatus.NEEDS_REVIEW, TaskStatus.VERIFYING}:
            if task.wiring_outcome in {"pending", "incomplete"}:
                task.wiring_outcome = "running"
        elif status in {TaskStatus.CANCELLED, TaskStatus.SKIPPED}:
            if task.wiring_outcome in {"incomplete", "pending", "running"}:
                if not task.wiring_required and task.integration_role != "wiring":
                    task.wiring_outcome = "not_required"
                else:
                    task.wiring_outcome = "failed"
        self._record("task.updated", {"task_id": task_id, "status": status.value, "reason": reason})
        self.save()

    def record_integration_evidence(
        self,
        task_id: str,
        path: list[str],
        *,
        summary: str = "",
        source_references: list[str] | None = None,
        observable_result: str = "",
        verification_source: str = "",
        reviewer: str = "",
    ) -> None:
        task = self.get_task(task_id)
        normalized = [str(item).strip() for item in path if str(item).strip()]
        if len(normalized) < 3:
            raise ValueError("integration evidence requires an entrypoint, reachable capability, and observable result")
        sources = [str(item).strip() for item in (source_references or []) if str(item).strip()]
        if not sources:
            raise ValueError("integration evidence requires repository or tool source references")
        if not str(observable_result or summary).strip():
            raise ValueError("integration evidence requires an observable result")
        task.integration_evidence = normalized
        record = {
            "entrypoint": normalized[0],
            "evidence_path": normalized,
            "path": normalized,
            "summary": summary,
            "source_references": sources,
            "observable_result": str(observable_result or summary).strip(),
            "verification_source": verification_source,
            "reviewer": reviewer,
            "recorded_at": utc_now().isoformat(),
        }
        task.integration_evidence_records.append(record)
        task.integration_verified = True
        task.runtime_reachability_verified = True
        if summary:
            task.evidence.append(f"Integration verification: {summary}")
        task.updated_at = utc_now()
        self._record("task.integration_evidence_recorded", {"task_id": task_id, "record": record})
        self.save()

    def _validate_feature_completion(self, task: TaskBoardItem) -> None:
        """Enforce strict implementation-to-runtime completion invariants."""
        if task.integration_role == "wiring":
            if task.runtime_reachability_verified and task.integration_evidence_records:
                task.integration_verified = True
                task.implementation_verified = True
            if not task.implementation_verified:
                raise InvalidTaskTransition("INCOMPLETE_FEATURE_WIRING: wiring implementation verification is absent")
            if task.wiring_outcome not in {"mutation_applied", "already_integrated", "completed", "running"}:
                raise InvalidTaskTransition("INCOMPLETE_FEATURE_WIRING: wiring outcome is unproven")
            if not task.integration_verified or not task.runtime_reachability_verified:
                raise InvalidTaskTransition("INCOMPLETE_FEATURE_WIRING: wiring runtime reachability evidence is absent")
            if not task.verification_provenance:
                raise InvalidTaskTransition("INCOMPLETE_FEATURE_WIRING: child verification provenance is absent")
            if not task.integration_evidence_records or not all(
                record.get("source_references") and record.get("observable_result")
                for record in task.integration_evidence_records
            ):
                raise InvalidTaskTransition("INCOMPLETE_FEATURE_WIRING: wiring evidence provenance is absent")
        if task.implementation_targets and not str(task.wiring_reason or "").strip():
            raise InvalidTaskTransition(
                "INCOMPLETE_FEATURE_WIRING: planner did not explain why wiring is unnecessary"
            )
        if task.integration_role == "wiring":
            return
        if not task.wiring_required:
            task.wiring_outcome = "not_required"
            return
        if not task.required_wiring_task_ids:
            raise InvalidTaskTransition(
                "INCOMPLETE_FEATURE_WIRING: wiring is required but no integration task exists"
            )
        missing = [
            dependency_id
            for dependency_id in task.required_wiring_task_ids
            if dependency_id not in self.tasks
            or self.tasks[dependency_id].status is not TaskStatus.DONE
        ]
        if missing:
            raise InvalidTaskTransition(
                "INCOMPLETE_FEATURE_WIRING: required integration tasks are incomplete: "
                + ", ".join(missing)
            )
        if not task.implementation_verified:
            raise InvalidTaskTransition("INCOMPLETE_FEATURE_WIRING: implementation verification is absent")
        if not task.integration_verified or not task.runtime_reachability_verified:
            raise InvalidTaskTransition("INCOMPLETE_FEATURE_WIRING: runtime reachability evidence is absent")
        if not task.integration_evidence_records or not all(
            record.get("source_references") and record.get("observable_result")
            for record in task.integration_evidence_records
        ):
            raise InvalidTaskTransition("INCOMPLETE_FEATURE_WIRING: runtime evidence provenance is absent")
        task.wiring_outcome = "completed"

    def project_supervisor_completion(
        self,
        task_id: str,
        *,
        supervisor_task: Any,
        verification_evidence: dict[str, Any],
    ) -> None:
        """Project one already-persisted supervisor completion into TaskBoard."""
        task = self.get_task(task_id)
        raw_state = getattr(supervisor_task, "state", "")
        state = str(getattr(raw_state, "value", raw_state))
        raw_verification = getattr(supervisor_task, "verification_status", "")
        verification = str(getattr(raw_verification, "value", raw_verification))
        if verification in {"succeeded", "completed"}:
            verification = "passed"
        if state != "completed" or verification != "passed":
            raise ValueError("supervisor projection cannot advertise an unverified completion")
        task.supervisor_execution_id = str(
            getattr(supervisor_task, "execution_id", "")
            or getattr(supervisor_task, "task_id", "")
        )
        task.supervisor_state = state
        task.supervisor_state_version = int(getattr(supervisor_task, "state_version", 0))
        task.supervisor_verification_evidence = dict(verification_evidence)
        task.verification_status = verification
        if not task.wiring_required and task.integration_role != "wiring":
            task.wiring_outcome = "not_required"
        elif task.wiring_outcome not in {"mutation_applied", "already_integrated", "completed"}:
            task.wiring_outcome = "completed"
        # Projection repair may replace a stale terminal TaskBoard status after
        # a crash. This is not an independent task transition: the durable
        # supervisor record supplied above is authoritative.
        if task.status is not TaskStatus.VERIFYING:
            task.status = TaskStatus.VERIFYING
        self._validate_feature_completion(task)
        validate_transition(task, TaskStatus.DONE, reason="supervisor completion projected")
        task.status = TaskStatus.DONE
        if not task.wiring_required and task.integration_role != "wiring":
            task.wiring_outcome = "not_required"
        elif task.wiring_outcome not in {"mutation_applied", "already_integrated"}:
            task.wiring_outcome = "completed"
        task.updated_at = utc_now()
        self._record(
            "task.supervisor_completion_projected",
            {"task_id": task_id, "supervisor_state_version": task.supervisor_state_version},
        )
        self.save()

    def reopen(self, task_id: str, *, reason: str) -> None:
        """Requeue a stopped task so the same identity can continue incomplete work.

        FAILED, BLOCKED, and CANCELLED children of a multi-task job may reopen under
        a validated same-task recovery decision so the job can restart from its first
        incomplete step without inventing a new root identity.
        """
        task = self.get_task(task_id)
        if task.status not in {TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.CANCELLED}:
            raise ValueError(f"task {task_id} is not in a reopenable state")
        task.status = TaskStatus.QUEUED
        task.blockers = []
        task.wiring_outcome = "pending"
        task.wiring_outcome_reason = ""
        task.updated_at = utc_now()
        self._record("task.reopened", {"task_id": task_id, "reason": reason})
        self.save()

    def assign(self, task_id: str, agent_id: str) -> None:
        task = self.get_task(task_id)
        _append_unique(task.assigned_agent_ids, [agent_id])
        task.updated_at = utc_now()
        self._record("task.assigned", {"task_id": task_id, "agent_id": agent_id})
        self.save()

    def assign_subagent(self, task_id: str, subagent_id: str) -> None:
        task = self.get_task(task_id)
        _append_unique(task.assigned_subagent_ids, [subagent_id])
        task.updated_at = utc_now()
        self._record("task.subagent_assigned", {"task_id": task_id, "subagent_id": subagent_id})
        self.save()

    def add_assumption(self, task_id: str, assumption: str) -> None:
        self._add_text(task_id, "assumptions", assumption)

    def add_decision(self, task_id: str, decision: DecisionRecord | str) -> None:
        task = self.get_task(task_id)
        decision_id = decision.decision_id if isinstance(decision, DecisionRecord) else str(decision)
        _append_unique(task.decision_ids, [decision_id])
        task.updated_at = utc_now()
        self._record("decision.recorded", {"task_id": task_id, "decision": serialize(decision)})
        self.save()

    def add_evidence(self, task_id: str, evidence: str) -> None:
        self._add_text(task_id, "evidence", evidence)

    def add_blocker(self, task_id: str, blocker: str, *, save: bool = True) -> None:
        self._add_text(task_id, "blockers", blocker, save=save)

    def add_discussion(self, task_id: str, discussion_id: str) -> None:
        task = self.get_task(task_id)
        _append_unique(task.discussion_ids, [discussion_id])
        task.updated_at = utc_now()
        self._record("discussion.opened", {"task_id": task_id, "discussion_id": discussion_id})
        self.save()

    def add_files_to_inspect(self, task_id: str, files: list[str]) -> None:
        self._add_many(task_id, "files_to_inspect", files)

    def add_files_touched(self, task_id: str, files: list[str]) -> None:
        self._add_many(task_id, "files_touched", files)

    def add_queue_job(self, task_id: str, job_id: str) -> None:
        self._add_many(task_id, "queue_job_ids", [job_id])

    def add_verification_queue_job(self, task_id: str, job_id: str) -> None:
        self._add_many(task_id, "verification_queue_job_ids", [job_id])

    def record_budget(self, task_id: str, record: dict[str, Any]) -> None:
        task = self.get_task(task_id)
        payload = dict(record)
        task.budget_records.append(payload)
        agent_id = str(payload.get("agent_id") or payload.get("requested_by_agent_id") or "")
        queue_job_id = str(payload.get("queue_job_id") or "")
        reserved = int(payload.get("budget_reserved_tokens") or payload.get("budget_reserved") or 0)
        used = int(payload.get("budget_used_tokens") or payload.get("budget_used") or 0)
        if reserved:
            task.budget_reserved_tokens += reserved
            task.budget_remaining_tokens += reserved
        if used:
            task.budget_used_tokens += used
            task.budget_remaining_tokens = max(0, task.budget_reserved_tokens - task.budget_used_tokens)
        if agent_id:
            task.cost_by_agent_id[agent_id] = int(task.cost_by_agent_id.get(agent_id, 0)) + used
        if queue_job_id:
            task.cost_by_queue_job_id[queue_job_id] = int(task.cost_by_queue_job_id.get(queue_job_id, 0)) + used
        task.updated_at = utc_now()
        self._record("budget.recorded", {"task_id": task_id, "record": payload})
        self.save()

    def record_hierarchy_violation(self, task_id: str, violation: dict[str, Any]) -> None:
        task = self.get_task(task_id)
        payload = dict(violation)
        task.hierarchy_violations.append(payload)
        task.status = TaskStatus.BLOCKED
        task.updated_at = utc_now()
        self._record("hierarchy_violation", {"task_id": task_id, **payload})
        self.save()

    def record_tool_event(self, task_id: str, event: dict[str, Any]) -> None:
        task = self.get_task(task_id)
        payload = dict(event)
        event_type = str(payload.get("type") or "tool.event")
        if event_type != "tool.started":
            task.actual_tool_events.append(payload)
            worker_id = str(payload.get("agent_id") or payload.get("executed_by_worker_agent_id") or "")
            if worker_id:
                task.executed_by_worker_agent_id = worker_id
        task.updated_at = utc_now()
        self._record(event_type, {"task_id": task_id, **payload})
        self.save()

    def add_verification_result(self, task_id: str, result: VerificationResult) -> None:
        task = self.get_task(task_id)
        task.verification_results.append(result)
        task.updated_at = utc_now()
        self._record("verification.finished", {"task_id": task_id, "result": serialize(result)})
        self.save()

    def add_handoff(self, task_id: str, handoff: HandoffRecord) -> None:
        task = self.get_task(task_id)
        task.handoff_records.append(handoff)
        task.updated_at = utc_now()
        self._record("handoff.created", handoff)
        self.save()

    def compact_context(self, task_id: str, token_budget: int = 1200) -> str:
        task = self.get_task(task_id)
        lines = [
            f"Task {task.task_id}: {task.title}",
            f"Status: {task.status.value}",
            f"Goal: {task.normalized_goal}",
        ]
        for label, values in (
            ("Plan", task.plan),
            ("Acceptance", task.acceptance_criteria),
            ("Evidence", task.evidence),
            ("Blockers", task.blockers),
            ("Assumptions", task.assumptions),
        ):
            if values:
                lines.append(f"{label}:")
                lines.extend(f"- {item}" for item in values)
        text = "\n".join(lines)
        if estimate_value_tokens(text) <= max(1, int(token_budget)):
            return text
        store = ContextArtifactStore()
        envelope = compress_tool_result(
            text,
            tool_name="taskboard_context",
            store=store,
            session_id=task.session_id,
            repository_id=task.primary_repository_id,
            workspace_id=task.workspace_id,
        )
        task.memory_status["context_compaction"] = {
            "artifact_ref": envelope.artifact_ref.artifact_id,
            "artifact_hash": envelope.content_hash,
            "lossless_source_available": True,
        }
        task.updated_at = utc_now()
        self.save()
        return render_envelope(envelope)

    def save(self) -> None:
        with self._save_lock:
            self.store.save_state({
                "schema_version": 2,
                "tasks": {key: serialize(value) for key, value in self.tasks.items()},
            })

    def load(self) -> None:
        payload = self.store.load_state()
        tasks = payload.get("tasks", {}) if isinstance(payload, dict) else {}
        self.tasks = {
            task_id: task_from_dict(item)
            for task_id, item in tasks.items()
            if isinstance(item, dict)
        }

    def _add_text(self, task_id: str, field_name: str, value: str, *, save: bool = True) -> None:
        if not str(value or "").strip():
            return
        task = self.get_task(task_id)
        _append_unique(getattr(task, field_name), [value])
        task.updated_at = utc_now()
        self._record("task.updated", {"task_id": task_id, field_name: value})
        if save:
            self.save()

    def _add_many(self, task_id: str, field_name: str, values: list[str]) -> None:
        task = self.get_task(task_id)
        _append_unique(getattr(task, field_name), values)
        task.updated_at = utc_now()
        self._record("task.updated", {"task_id": task_id, field_name: values})
        self.save()

    def _record(self, event_type: str, payload: Any) -> None:
        self.store.append_history({"event_type": event_type, "payload": serialize(payload), "created_at": utc_now()})
