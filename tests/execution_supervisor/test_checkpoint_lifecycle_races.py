"""Regression test suite for checkpoint lifecycle races, terminal state guards,
error preservation, and explicit recovery state transitions.
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.errors import (
    InvalidTransitionError,
    LeaseConflictError,
    StaleLeaseError,
)
from mana_agent.execution_supervisor.models import (
    ExecutionState,
    RecoveryAction,
    RecoveryDecision,
    RetryCategory,
    SideEffectClassification,
    TaskRecord,
)
from mana_agent.execution_supervisor.state_machine import validate_transition
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
from mana_agent.gateway.lane_coordinator import LaneCoordinator, LaneId, LaneTaskState


class FakeClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def _make_supervisor(tmp_path: Path, clock: Any = None) -> ExecutionSupervisor:
    config = ExecutionSupervisorConfig(root=tmp_path / "supervisor", lease_seconds=30)
    if clock is not None:
        return ExecutionSupervisor(config=config, clock=clock)
    return ExecutionSupervisor(config=config)


def _start_task(
    supervisor: ExecutionSupervisor, tmp_path: Path, **kwargs: Any
) -> tuple[TaskRecord, str, str]:
    opts = {
        "routing_decision_id": "decision_test",
        "side_effect_classification": SideEffectClassification.READ_ONLY,
        "workspace_path": tmp_path,
        "normalized_intent": "test checkpoint intent",
        "target_resources": ["workspace"],
        "important_constraints": [],
    }
    opts.update(kwargs)
    task = supervisor.create_task(**opts)
    supervisor.queue(task.task_id)
    leased, token = supervisor.acquire_lease(task.task_id, owner="worker-1")
    supervisor.start(task.task_id, attempt_id=leased.attempt_id, lease_token=token)
    return task, leased.attempt_id, token


def test_1_failure_before_checkpoint_skips_and_preserves_error(tmp_path: Path) -> None:
    """1. Failure before checkpoint:
    running -> failed -> stale callback attempts checkpoint.
    Expected: checkpoint skipped, original task remains failed, original failure
    reason preserved, no exception raised.
    """
    clock = FakeClock()
    supervisor = _make_supervisor(tmp_path, clock=clock)
    task, attempt_id, token = _start_task(supervisor, tmp_path)

    # Execution fails
    original_error = "provider_error: upstream rate limit exceeded"
    supervisor.transition(task.task_id, ExecutionState.FAILED, reason=original_error)

    failed_task = supervisor.store.get_task(task.task_id)
    assert failed_task.state == ExecutionState.FAILED
    assert failed_task.failure_reason == original_error

    # can_checkpoint should return False
    assert supervisor.can_checkpoint(task.task_id) is False

    # Late/stale callback attempts checkpoint
    result = supervisor.checkpoint(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        resume_payload={"boundary": "late_post_route_hook"},
        caller="stale_callback",
    )

    # Must be safely skipped (returns None)
    assert result is None

    # Task state and failure reason must be completely preserved
    persisted_task = supervisor.store.get_task(task.task_id)
    assert persisted_task.state == ExecutionState.FAILED
    assert persisted_task.failure_reason == original_error

    # Checkpoint skipped event was emitted
    events = supervisor.store.events_for_task(task.task_id)
    event_types = [e["event_type"] for e in events]
    assert "checkpoint.skipped" in event_types
    skipped_event = next(e for e in events if e["event_type"] == "checkpoint.skipped")
    assert skipped_event["details"]["current_state"] == "failed"
    assert skipped_event["details"]["terminal_reason"] == original_error


def test_2_concurrent_failure_checkpoint_race(tmp_path: Path) -> None:
    """2. Concurrent failure/checkpoint race:
    Thread A marks task failed while Thread B requests checkpoint using stale running state.
    Expected: atomic transition protection prevents illegal checkpoint from failed state.
    """
    clock = FakeClock()
    supervisor = _make_supervisor(tmp_path, clock=clock)
    task, attempt_id, token = _start_task(supervisor, tmp_path)

    def fail_worker() -> None:
        supervisor.transition(
            task.task_id,
            ExecutionState.FAILED,
            reason="heartbeat_timeout_failure",
        )

    def checkpoint_worker() -> Any:
        return supervisor.checkpoint(
            task.task_id,
            attempt_id=attempt_id,
            lease_token=token,
            resume_payload={"boundary": "concurrent_step"},
            caller="thread_b_worker",
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_fail = pool.submit(fail_worker)
        f_cp = pool.submit(checkpoint_worker)
        f_fail.result()
        cp_result = f_cp.result()

    final_task = supervisor.store.get_task(task.task_id)
    # The task must be either failed or legitimately checkpointed then failed,
    # but never left in an invalid CHECKPOINTING or RUNNING state without failure.
    assert final_task.state == ExecutionState.FAILED
    assert final_task.failure_reason == "heartbeat_timeout_failure"


def test_3_normal_checkpoint_lifecycle(tmp_path: Path) -> None:
    """3. Normal checkpoint:
    running -> checkpointing -> checkpoint_saved -> running works continuously.
    """
    clock = FakeClock()
    supervisor = _make_supervisor(tmp_path, clock=clock)
    task, attempt_id, token = _start_task(supervisor, tmp_path)

    assert supervisor.can_checkpoint(task.task_id) is True

    checkpoint = supervisor.checkpoint(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        resume_payload={"boundary": "step_1_complete", "cursor": "line_10"},
        completed_steps=["step_1"],
        pending_steps=["step_2"],
        caller="normal_agent_loop",
    )

    assert checkpoint is not None
    assert checkpoint.task_id == task.task_id
    assert checkpoint.completed_steps == ["step_1"]

    running_task = supervisor.store.get_task(task.task_id)
    assert running_task.state == ExecutionState.RUNNING
    assert running_task.checkpoint_id == checkpoint.checkpoint_id
    assert running_task.checkpoint_count == 1

    events = supervisor.store.events_for_task(task.task_id)
    event_types = [e["event_type"] for e in events]
    assert "checkpoint.requested" in event_types
    assert "checkpoint.allowed" in event_types
    assert "task_checkpointing" in event_types
    assert "checkpoint_saved" in event_types
    assert "checkpoint.saved" in event_types


def test_4_completed_execution_late_checkpoint_callback_skips(tmp_path: Path) -> None:
    """4. Completed execution:
    completed -> late checkpoint callback.
    Must skip without mutating the completed result.
    """
    clock = FakeClock()
    supervisor = _make_supervisor(tmp_path, clock=clock)
    task, attempt_id, token = _start_task(supervisor, tmp_path)

    # Submit result and complete
    result_task = supervisor.submit_result(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        payload={"answer": "Finished work successfully."},
    )
    assert result_task.state == ExecutionState.COMPLETED

    assert supervisor.can_checkpoint(task.task_id) is False

    # Late callback
    cp = supervisor.checkpoint(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        resume_payload={"boundary": "after_completion_cleanup"},
        caller="late_cleanup_hook",
    )
    assert cp is None

    # Completed task state and result are preserved
    persisted = supervisor.store.get_task(task.task_id)
    assert persisted.state == ExecutionState.COMPLETED
    assert persisted.result_id == result_task.result_id

    escrow = supervisor.store.get_result(result_task.result_id)
    assert escrow is not None
    assert escrow.payload == {"answer": "Finished work successfully."}


def test_5_recovery_from_failed_task_requires_explicit_state_transition(tmp_path: Path) -> None:
    """5. Recovery from failed task:
    Direct failed -> checkpointing is impossible.
    Explicit recovery must transition FAILED -> RETRY_SCHEDULED -> QUEUED -> LEASED -> RUNNING -> CHECKPOINTING.
    """
    clock = FakeClock()
    supervisor = _make_supervisor(tmp_path, clock=clock)
    task, attempt_id, token = _start_task(supervisor, tmp_path)

    supervisor.transition(task.task_id, ExecutionState.FAILED, reason="transient_tool_failure")
    assert supervisor.store.get_task(task.task_id).state == ExecutionState.FAILED

    # Direct transition FAILED -> CHECKPOINTING must raise InvalidTransitionError
    with pytest.raises(InvalidTransitionError):
        validate_transition(ExecutionState.FAILED, ExecutionState.CHECKPOINTING)

    # Direct checkpoint call on FAILED task returns None
    assert supervisor.checkpoint(task.task_id, attempt_id=attempt_id, lease_token=token, resume_payload={}) is None

    # Explicit recovery via supervisor.retry
    decision = RecoveryDecision(
        decision_id="decision_recovery_test",
        task_id=task.task_id,
        action=RecoveryAction.RETRY,
        retry_category=RetryCategory.TOOL,
        reason="retry tool failure with backoff",
        safe_to_continue=True,
    )
    retried_task = supervisor.retry(task.task_id, decision)
    assert retried_task.state == ExecutionState.RETRY_SCHEDULED

    # Advance clock and queue
    clock.advance(10)
    queued_task = supervisor.queue(task.task_id)
    assert queued_task.state == ExecutionState.QUEUED

    # Lease and start new attempt
    leased_task, new_token = supervisor.acquire_lease(task.task_id, owner="worker-2")
    assert leased_task.state == ExecutionState.LEASED

    running_task = supervisor.start(task.task_id, attempt_id=leased_task.attempt_id, lease_token=new_token)
    assert running_task.state == ExecutionState.RUNNING

    # Now in RUNNING state, checkpointing succeeds
    cp = supervisor.checkpoint(
        task.task_id,
        attempt_id=leased_task.attempt_id,
        lease_token=new_token,
        resume_payload={"boundary": "after_retry_start"},
    )
    assert cp is not None
    assert cp.attempt_id == leased_task.attempt_id


def test_6_original_error_preservation_across_error_types(tmp_path: Path) -> None:
    """6. Original error preservation:
    If execution fails because of api_workflow_incomplete, provider_error, tool_error,
    verification_error, late checkpoints must not overwrite the error.
    """
    error_codes = [
        "api_workflow_incomplete",
        "provider_error: context limit exceeded",
        "tool_error: command execution exit code 1",
        "verification_error: artifact checksum mismatch",
    ]

    for error_code in error_codes:
        clock = FakeClock()
        supervisor = _make_supervisor(tmp_path / error_code.split(":")[0], clock=clock)
        task, attempt_id, token = _start_task(supervisor, tmp_path)

        supervisor.transition(task.task_id, ExecutionState.FAILED, reason=error_code)

        # Late checkpoint
        res = supervisor.checkpoint(
            task.task_id,
            attempt_id=attempt_id,
            lease_token=token,
            resume_payload={"boundary": "late_callback"},
        )
        assert res is None

        persisted = supervisor.store.get_task(task.task_id)
        assert persisted.state == ExecutionState.FAILED
        assert persisted.failure_reason == error_code
        assert "cannot checkpoint" not in persisted.failure_reason


def test_7_durable_state_reconciliation_after_restart(tmp_path: Path) -> None:
    """7. Durable state reconciliation:
    Restart/reload a failed task and trigger post-processing.
    No checkpoint attempt should occur from stale in-memory state.
    """
    clock = FakeClock()
    supervisor1 = _make_supervisor(tmp_path, clock=clock)
    task, attempt_id, token = _start_task(supervisor1, tmp_path)
    supervisor1.transition(task.task_id, ExecutionState.FAILED, reason="fatal_crash")

    # Restart supervisor with same durable store
    supervisor2 = ExecutionSupervisor(config=supervisor1.config, clock=clock)
    reloaded_task = supervisor2.store.get_task(task.task_id)
    assert reloaded_task.state == ExecutionState.FAILED

    assert supervisor2.can_checkpoint(task.task_id) is False

    # Attempt checkpoint through restarted supervisor
    cp = supervisor2.checkpoint(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        resume_payload={"boundary": "post_restart_hook"},
    )
    assert cp is None

    # State remains failed with original reason
    assert supervisor2.store.get_task(task.task_id).state == ExecutionState.FAILED
    assert supervisor2.store.get_task(task.task_id).failure_reason == "fatal_crash"


def test_8_running_task_invalid_lease_still_raises(tmp_path: Path) -> None:
    """Checkpoint failure while execution is still running must remain a real error."""
    clock = FakeClock()
    supervisor = _make_supervisor(tmp_path, clock=clock)
    task, attempt_id, token = _start_task(supervisor, tmp_path)

    # Use wrong lease token while task is RUNNING
    with pytest.raises((LeaseConflictError, StaleLeaseError), match="lease token"):
        supervisor.checkpoint(
            task.task_id,
            attempt_id=attempt_id,
            lease_token="bad_stale_token",
            resume_payload={"boundary": "step_1"},
        )


def test_9_lane_coordinator_checkpoint_skips_on_failed_task_without_crashing(tmp_path: Path) -> None:
    """LaneCoordinator checkpoint method safely handles terminal tasks and preserves errors."""
    supervisor = _make_supervisor(tmp_path)
    coordinator = LaneCoordinator(
        root=tmp_path,
        execution_supervisor=supervisor,
    )
    reservation = coordinator.reserve(
        normalized_intent="test lane intent",
        lane_id=LaneId.CODING,
        session_id="session-test-1",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
    )
    coordinator.start(reservation)
    task_id = reservation.execution.task_id

    # Finish lane as failed
    coordinator.finish(task_id, state=LaneTaskState.FAILED, error="lane_tool_failed")

    # Verify supervisor state is FAILED
    supervised = supervisor.store.get_task(task_id)
    assert supervised.state == ExecutionState.FAILED

    # can_checkpoint should return False
    assert coordinator.can_checkpoint(task_id) is False

    # Calling coordinator.checkpoint must return gracefully without throwing LeaseConflictError
    checkpoint_id = coordinator.checkpoint(task_id, boundary="late_feature_integration_hook")
    assert isinstance(checkpoint_id, str)

    # Supervisor state remains FAILED with original error
    persisted = supervisor.store.get_task(task_id)
    assert persisted.state == ExecutionState.FAILED
    assert persisted.failure_reason == "lane_tool_failed"
