from __future__ import annotations

from mana_agent.multi_agent.core.errors import InvalidTaskTransition
from mana_agent.multi_agent.core.types import TaskBoardItem, TaskStatus

_TERMINAL = {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.SKIPPED}
_ALLOWED: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.NEW: {TaskStatus.PLANNING, TaskStatus.DISCUSSING, TaskStatus.ROUTED, TaskStatus.QUEUED, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.SKIPPED},
    TaskStatus.PLANNING: {TaskStatus.DISCUSSING, TaskStatus.ROUTED, TaskStatus.QUEUED, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.SKIPPED},
    TaskStatus.DISCUSSING: {TaskStatus.ROUTED, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.SKIPPED},
    TaskStatus.ROUTED: {TaskStatus.WAITING_FOR_TOOLS, TaskStatus.QUEUED, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.SKIPPED},
    TaskStatus.WAITING_FOR_TOOLS: {TaskStatus.QUEUED, TaskStatus.IN_PROGRESS, TaskStatus.NEEDS_REVIEW, TaskStatus.VERIFYING, TaskStatus.CANCELLED, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.SKIPPED},
    TaskStatus.QUEUED: {TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.SKIPPED},
    TaskStatus.IN_PROGRESS: {TaskStatus.WAITING_FOR_TOOLS, TaskStatus.NEEDS_REVIEW, TaskStatus.VERIFYING, TaskStatus.CANCELLED, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.SKIPPED},
    TaskStatus.NEEDS_REVIEW: {TaskStatus.VERIFYING, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.SKIPPED},
    TaskStatus.VERIFYING: {TaskStatus.DONE, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.FAILED, TaskStatus.SKIPPED},
    TaskStatus.BLOCKED: {TaskStatus.QUEUED, TaskStatus.IN_PROGRESS, TaskStatus.VERIFYING, TaskStatus.CANCELLED, TaskStatus.FAILED, TaskStatus.SKIPPED},
}


def validate_transition(task: TaskBoardItem, next_status: TaskStatus, *, reason: str | None = None) -> None:
    if task.status == next_status and next_status != TaskStatus.VERIFYING:
        return
    if task.status in _TERMINAL:
        raise InvalidTaskTransition(f"{task.status.value} cannot transition to {next_status.value} without reopen")
    if next_status == TaskStatus.DONE:
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
        if task.wiring_required:
            if not task.required_wiring_task_ids:
                raise InvalidTaskTransition(
                    "INCOMPLETE_FEATURE_WIRING: wiring is required but no integration task exists"
                )
            if not task.implementation_verified:
                raise InvalidTaskTransition(
                    "INCOMPLETE_FEATURE_WIRING: implementation verification is absent"
                )
            if not task.integration_verified or not task.runtime_reachability_verified:
                raise InvalidTaskTransition(
                    "INCOMPLETE_FEATURE_WIRING: runtime reachability evidence is absent"
                )
            if not task.integration_evidence_records or not all(
                record.get("source_references") and record.get("observable_result")
                for record in task.integration_evidence_records
            ):
                raise InvalidTaskTransition(
                    "INCOMPLETE_FEATURE_WIRING: runtime evidence provenance is absent"
                )
        evidence = dict(task.supervisor_verification_evidence or {})
        if not task.supervisor_execution_id:
            raise InvalidTaskTransition(
                "done status requires the authoritative supervisor execution identity"
            )
        if task.supervisor_state != "completed" or task.verification_status != "passed":
            raise InvalidTaskTransition(
                "done status requires supervisor-approved passed verification"
            )
        if not evidence.get("verification") or not evidence.get("result_id"):
            raise InvalidTaskTransition(
                "done status requires durable supervisor verification evidence"
            )
    allowed = _ALLOWED.get(task.status, set())
    if next_status not in allowed:
        raise InvalidTaskTransition(f"{task.status.value} cannot transition to {next_status.value}")
    if next_status == TaskStatus.FAILED and not str(reason or "").strip():
        raise InvalidTaskTransition("failed status requires a reason")
    if next_status == TaskStatus.BLOCKED and not str(reason or "").strip() and not task.blockers:
        raise InvalidTaskTransition("blocked status requires a blocker")
    if next_status == TaskStatus.VERIFYING:
        if (
            "Awaiting authoritative supervisor completion projection" in str(reason or "")
            and not task.supervisor_execution_id
        ):
            raise InvalidTaskTransition(
                "verifying status cannot await supervisor completion without a supervisor execution"
            )
        verification_evidence = bool(task.verification_queue_job_ids or task.verification_results)
        supervisor_evidence = bool(
            task.supervisor_execution_id
            and task.supervisor_state
            and task.supervisor_verification_evidence
        )
        if not verification_evidence and not supervisor_evidence:
            raise InvalidTaskTransition(
                "verifying status requires an executed verification result, verification queue job, "
                "or registered supervisor verification execution"
            )
