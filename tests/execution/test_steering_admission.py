"""Fenced admission tests; no live-steering delivery is exposed yet."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import Barrier

import pytest

from mana_agent.execution.steering import SteeringCoordinator, SteeringKind, SteeringScope, SteeringSubmission
from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.errors import LeaseConflictError, StaleLeaseError
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor


@pytest.fixture
def running(tmp_path):
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(root=tmp_path, startup_recovery=False))
    supervisor.create_task(task_id="execution", session_id="session")
    supervisor.queue("execution")
    task, token = supervisor.acquire_lease("execution", owner="worker")
    supervisor.start("execution", attempt_id=task.attempt_id, lease_token=token)
    return supervisor, task, token


def instruction(task, request_id="request", **changes):
    values = dict(execution_id=task.task_id, session_id="session", attempt_id=task.attempt_id,
                  generation=task.attempt_generation, request_id=request_id,
                  text="Run tests first", kind=SteeringKind.PRIORITY)
    values.update(changes)
    return SteeringSubmission(**values)


def test_concurrent_admission_is_ordered_and_idempotent(running):
    supervisor, task, _ = running
    coordinator = SteeringCoordinator(supervisor)
    def submit(index):
        return coordinator.submit(instruction(task, str(index // 2)), authenticated_session_id="session")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(submit, range(40)))
    persisted = supervisor.get_task(task.task_id).steering_inputs
    assert [item["sequence"] for item in persisted] == list(range(1, 21))
    assert sum(not duplicate for _, duplicate in results) == 20
    assert all(item["status"] == "pending" for item in persisted)
    with pytest.raises(ValueError, match="different"):
        coordinator.submit(instruction(task, "0", text="Different"), authenticated_session_id="session")


def test_admission_rejects_foreign_stale_cancelled_and_unsupported(running):
    supervisor, task, _ = running
    coordinator = SteeringCoordinator(supervisor)
    with pytest.raises(PermissionError):
        coordinator.submit(instruction(task), authenticated_session_id="other")
    with pytest.raises(PermissionError):
        coordinator.submit(instruction(task, session_id="other"), authenticated_session_id="other")
    with pytest.raises(StaleLeaseError):
        coordinator.submit(instruction(task, generation=task.attempt_generation + 1), authenticated_session_id="session")
    with pytest.raises(ValueError, match="not supported"):
        coordinator.submit(instruction(task, scope=SteeringScope.TASK, target_id="child"), authenticated_session_id="session")
    with pytest.raises(ValueError, match="not supported"):
        coordinator.submit(instruction(task, kind=SteeringKind.INTERRUPT), authenticated_session_id="session")
    supervisor.cancel(task.task_id)
    with pytest.raises(LeaseConflictError):
        coordinator.submit(instruction(task), authenticated_session_id="session")
    assert supervisor.get_task(task.task_id).steering_inputs == []


def test_expired_lease_and_reload(running):
    supervisor, task, _ = running
    coordinator = SteeringCoordinator(supervisor)
    coordinator.submit(instruction(task), authenticated_session_id="session")
    reloaded = ExecutionSupervisor(supervisor.config, startup_recovery=False)
    assert reloaded.get_task(task.task_id).steering_inputs[0]["status"] == "pending"
    reloaded.clock = lambda: task.lease_expires_at + timedelta(seconds=1)
    with pytest.raises(LeaseConflictError):
        SteeringCoordinator(reloaded).submit(instruction(task, "next"), authenticated_session_id="session")


def test_cancellation_race_never_accepts_after_cancel(running):
    supervisor, task, _ = running
    coordinator = SteeringCoordinator(supervisor)
    barrier = Barrier(2)

    def submit():
        barrier.wait()
        try:
            return coordinator.submit(instruction(task), authenticated_session_id="session")
        except LeaseConflictError:
            return None

    def cancel():
        barrier.wait()
        supervisor.cancel(task.task_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        submitted = pool.submit(submit)
        cancelled = pool.submit(cancel)
        result = submitted.result()
        cancelled.result()
    assert len(supervisor.get_task(task.task_id).steering_inputs) == int(result is not None)
    with pytest.raises(LeaseConflictError):
        coordinator.submit(instruction(task, "after-cancel"), authenticated_session_id="session")


def test_path_like_execution_id_is_rejected(running):
    supervisor, task, _ = running
    with pytest.raises(ValueError, match="invalid execution ID"):
        SteeringCoordinator(supervisor).submit(
            instruction(task, execution_id="../other"), authenticated_session_id="session"
        )


def test_controller_reads_only_current_attempt_without_acknowledging(running):
    from mana_agent.execution.autonomous_controller import AutonomousExecutionController

    supervisor, task, token = running
    coordinator = SteeringCoordinator(supervisor)
    coordinator.submit(instruction(task, "first"), authenticated_session_id="session")
    coordinator.submit(instruction(task, "second"), authenticated_session_id="session")

    # Simulate an older pending record that a new generation must not consume.
    def add_retired(current):
        current.steering_inputs[0]["attempt_id"] = "retired"
        current.steering_inputs[0]["generation"] = task.attempt_generation - 1
    supervisor.store.update_task(task.task_id, add_retired)

    controller = AutonomousExecutionController(supervisor=supervisor)
    kwargs = dict(attempt_id=task.attempt_id, generation=task.attempt_generation, lease_token=token)
    result = controller.pending_steering(task.task_id, **kwargs)
    assert [entry["request_id"] for entry in result] == ["second"]
    result[0]["status"] = "applied"
    assert controller.pending_steering(task.task_id, **kwargs)[0]["status"] == "pending"

    with pytest.raises(StaleLeaseError):
        controller.pending_steering(task.task_id, **{**kwargs, "generation": task.attempt_generation + 1})
    with pytest.raises(StaleLeaseError):
        controller.pending_steering(task.task_id, **{**kwargs, "lease_token": "wrong"})
    # A stored token hash is not the worker's secret token.
    with pytest.raises(StaleLeaseError):
        controller.pending_steering(task.task_id, **{**kwargs, "lease_token": supervisor.get_task(task.task_id).lease_token})
    with pytest.raises(ValueError, match="invalid execution ID"):
        controller.pending_steering("../other", **kwargs)
    supervisor.cancel(task.task_id)
    with pytest.raises((LeaseConflictError, StaleLeaseError)):
        controller.pending_steering(task.task_id, **kwargs)


def test_checkpoint_captures_snapshot_but_not_later_admission(running):
    supervisor, task, token = running
    coordinator = SteeringCoordinator(supervisor)
    coordinator.submit(instruction(task, "first"), authenticated_session_id="session")
    checkpoint = supervisor.checkpoint(
        task.task_id, attempt_id=task.attempt_id, lease_token=token,
        resume_payload={"cursor": 1},
    )
    coordinator.submit(instruction(task, "second"), authenticated_session_id="session")

    reloaded = ExecutionSupervisor(supervisor.config, startup_recovery=False)
    saved = reloaded.store.get_checkpoint(checkpoint.checkpoint_id)
    assert [entry["request_id"] for entry in saved.steering_inputs] == ["first"]
    assert [entry["request_id"] for entry in reloaded.get_task(task.task_id).steering_inputs] == ["first", "second"]
    assert saved.steering_inputs[0]["status"] == "pending"
    assert saved.resume_payload == {"cursor": 1}


def test_pending_read_rejects_expired_lease(running):
    supervisor, task, token = running
    coordinator = SteeringCoordinator(supervisor)
    coordinator.submit(instruction(task), authenticated_session_id="session")
    supervisor.clock = lambda: task.lease_expires_at + timedelta(seconds=1)
    with pytest.raises(LeaseConflictError):
        coordinator.pending_for_attempt(
            task.task_id, attempt_id=task.attempt_id,
            generation=task.attempt_generation, lease_token=token,
        )
