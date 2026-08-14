"""Regression test suite for checkpoint resume invariants (Scenarios A through I).

Invariants:
1. Terminal durable result > terminal task state > resumable checkpoint > generic recovery.
2. The existence of a checkpoint MUST NOT imply that an execution is resumable.
3. Checkpoints on terminal tasks are preserved for audit/diagnostics but never implicitly resumed.
4. Explicit retry creates a new attempt generation and does not mutate terminal attempts.
5. Verification boundary checkpoints require candidate results/artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from mana_agent.execution_supervisor import (
    CompletionContract,
    CompletionContractType,
    ExecutionSupervisor,
    ExecutionSupervisorConfig,
    ExecutionState,
    RecoveryDecision,
    SideEffectClassification,
    TaskRecord,
)
from mana_agent.execution_supervisor.errors import RetrySafetyError
from mana_agent.execution_supervisor.models import (
    CheckpointRecord,
    EscrowLookupStatus,
    EscrowResult,
    EscrowStatus,
    RecoveryAction,
    RetryCategory,
    TERMINAL_STATES,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def _make_supervisor(tmp_path: Path, clock: Any = None) -> ExecutionSupervisor:
    config = ExecutionSupervisorConfig(root=tmp_path / "supervisor")
    if clock is not None:
        return ExecutionSupervisor(config=config, clock=clock)
    return ExecutionSupervisor(config=config)


def _start_task(supervisor: ExecutionSupervisor, tmp_path: Path, **kwargs: Any) -> tuple[TaskRecord, str, str]:
    opts = {
        "routing_decision_id": "decision_test",
        "side_effect_classification": SideEffectClassification.READ_ONLY,
        "workspace_path": tmp_path,
        "normalized_intent": "test intent",
        "target_resources": ["workspace"],
        "important_constraints": [],
    }
    opts.update(kwargs)
    task = supervisor.create_task(**opts)
    supervisor.queue(task.task_id)
    leased, token = supervisor.acquire_lease(task.task_id, owner="worker-1")
    supervisor.start(task.task_id, attempt_id=leased.attempt_id, lease_token=token)
    return task, leased.attempt_id, token


def test_scenario_a_checkpoint_exists_task_failed_durable_failure_result_wins(tmp_path: Path) -> None:
    """Scenario A: Task failed, checkpoint exists, durable failure escrow result exists.
    
    Terminal durable result takes precedence; resume is rejected.
    """
    supervisor = _make_supervisor(tmp_path)
    task, attempt_id, token = _start_task(
        supervisor,
        tmp_path,
        normalized_intent="generate media image",
    )
    checkpoint = supervisor.checkpoint(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        resume_cursor="after_routing",
        resume_payload={"boundary": "after_routing", "mode": "route-media"},
        completed_steps=["routing"],
        pending_steps=["execute"],
    )

    # Transition to failed with error
    supervisor.transition(
        task.task_id,
        ExecutionState.FAILED,
        reason="media_image_disabled",
    )

    failed_task = supervisor.store.get_task(task.task_id)
    assert failed_task.state == ExecutionState.FAILED
    # Checkpoint is preserved for audit
    assert failed_task.checkpoint_id == checkpoint.checkpoint_id

    # Validate checkpoint resume
    eligibility = supervisor.validate_checkpoint_resume(task.task_id)
    assert eligibility.resumable is False
    assert eligibility.is_terminal is True
    assert eligibility.reason in {"terminal_execution", "terminal_result_exists"}

    # resume_checkpoint raises RetrySafetyError
    with pytest.raises(RetrySafetyError, match="checkpoint resume invalid"):
        supervisor.resume_checkpoint(task.task_id)

    # Verified result lookup returns terminal result with original failure
    lookup = supervisor.get_verified_execution_result(task.task_id)
    assert lookup.status == EscrowLookupStatus.FOUND
    assert lookup.is_terminal is True
    assert lookup.is_resumable is False
    assert lookup.result is not None
    assert lookup.result.supervisor_state == "failed"
    assert lookup.result.error_metadata.get("reason") == "media_image_disabled"


def test_scenario_b_checkpoint_exists_task_completed_completed_result_wins(tmp_path: Path) -> None:
    """Scenario B: Task completed, checkpoint exists.
    
    Completed result takes precedence; resume is rejected.
    """
    supervisor = _make_supervisor(tmp_path)
    task, attempt_id, token = _start_task(
        supervisor,
        tmp_path,
        normalized_intent="refactor auth module",
        side_effect_classification=SideEffectClassification.IDEMPOTENT,
    )
    checkpoint = supervisor.checkpoint(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        resume_cursor="after_routing",
        resume_payload={"boundary": "after_routing", "mode": "code-refactor"},
    )
    supervisor.submit_result(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        payload={"chat_result": {"answer": "Auth module refactored successfully.", "mode": "code-refactor"}},
    )

    eligibility = supervisor.validate_checkpoint_resume(task.task_id)
    assert eligibility.resumable is False
    assert eligibility.is_terminal is True

    with pytest.raises(RetrySafetyError, match="checkpoint resume invalid"):
        supervisor.resume_checkpoint(task.task_id)

    lookup = supervisor.get_verified_execution_result(task.task_id)
    assert lookup.status == EscrowLookupStatus.FOUND
    assert lookup.is_terminal is True
    assert lookup.is_resumable is False
    assert lookup.result.supervisor_state == "completed"


def test_scenario_c_checkpoint_exists_task_cancelled_cancelled_result_wins(tmp_path: Path) -> None:
    """Scenario C: Task cancelled, checkpoint exists.
    
    Cancelled state takes precedence; resume is rejected.
    """
    supervisor = _make_supervisor(tmp_path)
    task, attempt_id, token = _start_task(
        supervisor,
        tmp_path,
        normalized_intent="long running batch analysis",
    )
    supervisor.checkpoint(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        resume_cursor="after_routing",
        resume_payload={"boundary": "after_routing", "mode": "batch-analysis"},
    )
    supervisor.cancel(
        task.task_id,
        reason="user_requested_cancel",
    )

    eligibility = supervisor.validate_checkpoint_resume(task.task_id)
    assert eligibility.resumable is False
    assert eligibility.is_terminal is True

    with pytest.raises(RetrySafetyError, match="checkpoint resume invalid"):
        supervisor.resume_checkpoint(task.task_id)


def test_scenario_d_task_running_interrupted_checkpoint_valid_resumes(tmp_path: Path) -> None:
    """Scenario D: Task running / interrupted with valid checkpoint.
    
    Resume succeeds cleanly.
    """
    supervisor = _make_supervisor(tmp_path)
    task, attempt_id, token = _start_task(
        supervisor,
        tmp_path,
        normalized_intent="interactive coding session",
        side_effect_classification=SideEffectClassification.IDEMPOTENT,
    )
    checkpoint = supervisor.checkpoint(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        resume_cursor="after_step_1",
        resume_payload={"boundary": "after_step_1", "step": 1, "cursor": "line_42"},
        completed_steps=["step_1"],
        pending_steps=["step_2"],
    )

    eligibility = supervisor.validate_checkpoint_resume(task.task_id)
    assert eligibility.resumable is True
    assert eligibility.is_terminal is False
    assert eligibility.checkpoint is not None
    assert eligibility.checkpoint.checkpoint_id == checkpoint.checkpoint_id

    resumed_cp = supervisor.resume_checkpoint(task.task_id)
    assert resumed_cp.checkpoint_id == checkpoint.checkpoint_id


def test_scenario_e_task_running_checkpoint_corrupt_reports_error(tmp_path: Path) -> None:
    """Scenario E: Task running, but checkpoint is missing or corrupt.
    
    Resumability reports corrupt/missing error without crashing supervisor.
    """
    supervisor = _make_supervisor(tmp_path)
    task, attempt_id, token = _start_task(
        supervisor,
        tmp_path,
        normalized_intent="interactive coding session",
        side_effect_classification=SideEffectClassification.IDEMPOTENT,
    )
    checkpoint = supervisor.checkpoint(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        resume_cursor="after_step_1",
        resume_payload={"boundary": "after_step_1", "step": 1},
    )

    # Corrupt the checkpoint file in storage
    cp_path = supervisor.store.root / "checkpoints" / f"{checkpoint.checkpoint_id}.json"
    cp_path.write_text("NOT_VALID_JSON{", encoding="utf-8")

    eligibility = supervisor.validate_checkpoint_resume(task.task_id)
    assert eligibility.resumable is False
    assert eligibility.reason == "checkpoint_corrupt"
    assert eligibility.is_terminal is False

    with pytest.raises(RetrySafetyError, match="checkpoint resume invalid"):
        supervisor.resume_checkpoint(task.task_id)


def test_scenario_f_attempt_1_failed_explicit_retry_creates_attempt_2(tmp_path: Path) -> None:
    """Scenario F: Attempt 1 failed; explicit retry creates attempt 2 (generation 2).
    
    Attempt 1 remains terminal (failed); attempt 2 is queued with new attempt identity.
    """
    clock = FakeClock()
    supervisor = _make_supervisor(tmp_path, clock=clock)
    task, attempt_1_id, token = _start_task(
        supervisor,
        tmp_path,
        normalized_intent="process data batch",
        side_effect_classification=SideEffectClassification.IDEMPOTENT,
        idempotency_key="batch_key_1",
    )
    checkpoint = supervisor.checkpoint(
        task.task_id,
        attempt_id=attempt_1_id,
        lease_token=token,
        resume_cursor="after_routing",
        resume_payload={"boundary": "after_routing", "mode": "batch"},
    )
    supervisor.transition(
        task.task_id,
        ExecutionState.FAILED,
        reason="transient_network_error",
    )

    failed_task = supervisor.store.get_task(task.task_id)
    assert failed_task.state == ExecutionState.FAILED
    assert failed_task.attempt_generation == 1

    # Explicit retry scheduled
    decision = RecoveryDecision(
        decision_id="retry-dec-1",
        task_id=task.task_id,
        action=RecoveryAction.RETRY,
        retry_category=RetryCategory.INFRASTRUCTURE,
        reason="retry after transient failure",
        same_task_retry_authorized=True,
        safe_to_continue=True,
    )
    retried_task = supervisor.retry(task.task_id, decision)
    assert retried_task.state == ExecutionState.RETRY_SCHEDULED

    # Advance clock past retry backoff window
    clock.advance(5)

    # Release retry -> transitions to QUEUED
    released_task = supervisor.release_retry(task.task_id)
    assert released_task.state == ExecutionState.QUEUED

    # Acquire new lease for attempt 2
    task_2, token_2 = supervisor.acquire_lease(task.task_id, owner="worker-2")
    assert task_2.attempt_generation == 2
    assert task_2.attempt_id != attempt_1_id

    # Verify attempt 2 record in store
    attempt_2_rec = supervisor.store.get_attempt(task_2.attempt_id)
    assert attempt_2_rec is not None
    assert attempt_2_rec.generation == 2

    # Attempt 1 record in store remains generation 1
    attempt_1_rec = supervisor.store.get_attempt(attempt_1_id)
    assert attempt_1_rec is not None
    assert attempt_1_rec.generation == 1


def test_scenario_g_media_image_disabled_terminates_cleanly_without_before_verification_resume(tmp_path: Path) -> None:
    """Scenario G: media_image_disabled failure before image creation.
    
    The task fails cleanly without creating or attempting to resume a before_verification checkpoint.
    """
    supervisor = _make_supervisor(tmp_path)
    task, attempt_id, token = _start_task(
        supervisor,
        tmp_path,
        normalized_intent="generate avatar image",
    )
    
    # Checkpoint after routing
    supervisor.checkpoint(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        resume_cursor="after_routing",
        resume_payload={"boundary": "after_routing", "mode": "route-media"},
    )

    # Route fails with media_image_disabled -> No before_verification checkpoint is written
    supervisor.transition(
        task.task_id,
        ExecutionState.FAILED,
        reason="media_image_disabled",
    )

    # Verify task state
    task_after = supervisor.store.get_task(task.task_id)
    assert task_after.state == ExecutionState.FAILED

    # Invariant: recovery lookup returns terminal result with media_image_disabled
    lookup = supervisor.get_verified_execution_result(task.task_id)
    assert lookup.status == EscrowLookupStatus.FOUND
    assert lookup.is_terminal is True
    assert lookup.result.supervisor_state == "failed"
    assert lookup.result.error_metadata.get("reason") == "media_image_disabled"

    # Invariant: validate_checkpoint_resume rejects resume
    eligibility = supervisor.validate_checkpoint_resume(task.task_id)
    assert eligibility.resumable is False
    assert eligibility.is_terminal is True


def test_scenario_h_before_verification_checkpoint_without_candidate_results_is_rejected(tmp_path: Path) -> None:
    """Scenario H: A before_verification checkpoint lacking candidate results or files is not resumable."""
    supervisor = _make_supervisor(tmp_path)
    task, attempt_id, token = _start_task(
        supervisor,
        tmp_path,
        normalized_intent="generate report",
    )

    # Write a before_verification checkpoint that has NO generated files or candidate results and mode has an error
    checkpoint = supervisor.checkpoint(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        resume_cursor="before_verification",
        resume_payload={"boundary": "before_verification", "mode": "route-media-error", "error": "media_image_disabled"},
    )

    eligibility = supervisor.validate_checkpoint_resume(task.task_id)
    assert eligibility.resumable is False
    assert eligibility.reason == "missing_execution_result"
    assert eligibility.boundary == "before_verification"


def test_scenario_i_terminal_result_exists_while_checkpoint_is_corrupt_terminal_wins(tmp_path: Path) -> None:
    """Scenario I: Terminal result exists, but referenced checkpoint is corrupt.
    
    Terminal result wins; corrupt checkpoint does not prevent reading terminal status.
    """
    supervisor = _make_supervisor(tmp_path)
    task, attempt_id, token = _start_task(
        supervisor,
        tmp_path,
        normalized_intent="execute task",
    )
    checkpoint = supervisor.checkpoint(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        resume_cursor="after_routing",
        resume_payload={"boundary": "after_routing", "mode": "test"},
    )
    supervisor.transition(
        task.task_id,
        ExecutionState.FAILED,
        reason="service_unavailable",
    )

    # Corrupt the checkpoint file
    cp_path = supervisor.store.root / "checkpoints" / f"{checkpoint.checkpoint_id}.json"
    cp_path.write_text("CORRUPTED_JSON_DATA", encoding="utf-8")

    # Terminal state check still returns terminal eligibility (not resumable)
    eligibility = supervisor.validate_checkpoint_resume(task.task_id)
    assert eligibility.resumable is False
    assert eligibility.is_terminal is True

    # Escrow lookup succeeds with terminal result
    lookup = supervisor.get_verified_execution_result(task.task_id)
    assert lookup.status == EscrowLookupStatus.FOUND
    assert lookup.is_terminal is True
    assert lookup.result.supervisor_state == "failed"
    assert lookup.result.error_metadata.get("reason") == "service_unavailable"
