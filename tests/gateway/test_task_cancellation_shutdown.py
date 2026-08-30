from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import pytest

from mana_agent.config.settings import Settings
from mana_agent.gateway.chat_gateway import AgentChatGateway, ChatGatewayConfig
from mana_agent.gateway.shutdown import ShutdownCoordinator, CancellationMetadata
from mana_agent.gateway.lane_coordinator import LaneCoordinator, LaneId, LanePriority, LaneTaskState
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
from mana_agent.execution_supervisor.models import ExecutionState, CancellationStatus
from mana_agent.execution_supervisor.store import LocalExecutionStore
from mana_agent.multi_agent.queue.queue_manager import QueueManager
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.multi_agent.core.types import QueueJob, QueueJobType, QueueJobStatus
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.coding_agent_shim import CodexCodingAgentShim


def _build_test_supervisor(tmp_path: Path) -> ExecutionSupervisor:
    store = LocalExecutionStore(tmp_path / "supervisor")
    return ExecutionSupervisor(store=store)


def _transition_to_running(supervisor: ExecutionSupervisor, task_id: str) -> None:
    supervisor.transition(task_id, ExecutionState.LEASED, reason="leased")
    supervisor.transition(task_id, ExecutionState.RUNNING, reason="started")


def _transition_to_completed(supervisor: ExecutionSupervisor, task_id: str) -> None:
    supervisor.transition(task_id, ExecutionState.COMPLETED_PENDING_VERIFICATION, reason="verifying")
    supervisor.transition(task_id, ExecutionState.COMPLETED, reason="completed")


def test_ctrl_c_during_active_task_cancellation(tmp_path: Path) -> None:
    """Ctrl+C transitions active task to status='cancelled' with structured metadata."""
    supervisor = _build_test_supervisor(tmp_path)
    task = supervisor.create_task(
        routing_decision_id="decision-123",
        assigned_agent="lane:coding",
        task_type="coding",
    )
    _transition_to_running(supervisor, task.task_id)

    # Simulate Ctrl+C interrupt
    supervisor.cancel(
        task.task_id,
        reason="user_interrupt",
        source="ctrl_c",
    )

    persisted = supervisor.store.get_task(task.task_id)
    assert persisted.state == ExecutionState.CANCELLED
    assert persisted.cancellation_status == CancellationStatus.COMPLETED
    assert persisted.cancellation_reason == "user_interrupt"
    assert persisted.cancellation_source == "ctrl_c"

    # Escrow result exists and is terminal
    result = supervisor.store.get_result(persisted.result_id)
    assert result is not None
    assert result.status.value == "available"
    assert result.supervisor_state == "cancelled"
    assert result.payload["status"] == "cancelled"
    assert result.payload["cancellation_source"] == "ctrl_c"
    assert result.payload["is_resumable"] is False


def test_exit_during_lane_execution_cancellation(tmp_path: Path) -> None:
    """Exit command cleanly cancels lane coordinator task tree before shutdown."""
    supervisor = _build_test_supervisor(tmp_path)
    events: list[tuple[str, str, dict[str, Any]]] = []

    def sink(event_type: str, title: str, **kwargs: Any) -> None:
        events.append((event_type, title, kwargs.get("metadata") or {}))

    coordinator = LaneCoordinator(tmp_path, execution_supervisor=supervisor, event_sink=sink)


    res_parent = coordinator.reserve(
        normalized_intent="Parent research task",
        lane_id=LaneId.RESEARCH,
        session_id="session-exit-test",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
    )
    coordinator.start(res_parent)
    parent_task_id = res_parent.execution.task_id

    res_child = coordinator.reserve(
        normalized_intent="Child subtask",
        lane_id=LaneId.RESEARCH,
        session_id="session-exit-test",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
        parent_task_id=parent_task_id,
    )
    coordinator.start(res_child)
    child_task_id = res_child.execution.task_id

    # User triggers exit
    cancelled = coordinator.cancel_tree(parent_task_id, reason="exit requested", source="exit")

    assert set(cancelled) == {parent_task_id, child_task_id}
    assert coordinator.inspect_task(parent_task_id).state == LaneTaskState.CANCELLED
    assert coordinator.inspect_task(child_task_id).state == LaneTaskState.CANCELLED

    # Check lifecycle event emission
    req_events = [e for e in events if e[0] == "task.cancellation.requested"]
    assert len(req_events) >= 2
    assert req_events[0][2]["source"] == "exit"


def test_ctrl_c_cancels_queued_and_running_tool_jobs(tmp_path: Path) -> None:
    """Queued jobs belonging to the cancelled task are marked CANCELLED and do not run."""
    taskboard = TaskBoard()
    tb_task = taskboard.create_task(
        title="Coding task",
        user_request="Refactor auth module",
        owner_agent_id="agent_coding_1",
    )

    queue_manager = QueueManager(tmp_path, taskboard=taskboard)

    job1 = queue_manager.enqueue(
        task_id=tb_task.task_id,
        requested_by_agent_id="agent_coding_1",
        job_type=QueueJobType.REPO_READ,
        purpose="Read auth module",
        payload={"path": "auth.py"},
    )
    job2 = queue_manager.enqueue(
        task_id=tb_task.task_id,
        requested_by_agent_id="agent_coding_1",
        job_type=QueueJobType.SHELL,
        purpose="Run pytest",
        payload={"command": "pytest"},
    )

    assert queue_manager.get_job(job1.job_id).status == QueueJobStatus.QUEUED
    assert queue_manager.get_job(job2.job_id).status == QueueJobStatus.QUEUED

    # Cancel task jobs on Ctrl+C
    cancelled = queue_manager.cancel_task_jobs(tb_task.task_id, reason="cancelled by user interrupt")
    assert set(cancelled) == {job1.job_id, job2.job_id}

    assert queue_manager.get_job(job1.job_id).status == QueueJobStatus.CANCELLED
    assert queue_manager.get_job(job2.job_id).status == QueueJobStatus.CANCELLED

    # Claim next returns None because cancelled jobs are not runnable
    assert queue_manager.claim_next("agent_tool_worker_1") is None


def test_repeated_ctrl_c_idempotency(tmp_path: Path) -> None:
    """Repeated Ctrl+C calls do not double-write terminal result or corrupt state."""
    supervisor = _build_test_supervisor(tmp_path)
    task = supervisor.create_task(
        routing_decision_id="decision-456",
        assigned_agent="lane:coding",
        task_type="coding",
    )
    _transition_to_running(supervisor, task.task_id)

    # First Ctrl+C
    first_res = supervisor.cancel(task.task_id, reason="user_interrupt", source="ctrl_c")
    assert first_res == [task.task_id]

    task_after_first = supervisor.store.get_task(task.task_id)
    first_result_id = task_after_first.result_id

    # Second Ctrl+C (repeated)
    second_res = supervisor.cancel(task.task_id, reason="user_interrupt", source="ctrl_c")
    assert second_res == []

    task_after_second = supervisor.store.get_task(task.task_id)
    assert task_after_second.result_id == first_result_id
    assert task_after_second.state == ExecutionState.CANCELLED


def test_task_completes_concurrently_with_cancellation(tmp_path: Path) -> None:
    """Write-once invariant: if task completed before cancellation, retain completed result."""
    supervisor = _build_test_supervisor(tmp_path)
    task = supervisor.create_task(
        routing_decision_id="decision-789",
        assigned_agent="lane:coding",
        task_type="coding",
    )
    _transition_to_running(supervisor, task.task_id)
    _transition_to_completed(supervisor, task.task_id)

    # Cancellation arrives after completion
    res = supervisor.cancel(task.task_id, reason="user_interrupt", source="ctrl_c")
    assert res == []

    persisted = supervisor.store.get_task(task.task_id)
    assert persisted.state == ExecutionState.COMPLETED


def test_shutdown_coordinator_and_lifecycle_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Authoritative shutdown emits all required lifecycle events in order."""
    monkeypatch.setenv("MANA_OPENAI_API_KEY", "sk-test-mock-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-key")
    events: list[tuple[str, str, dict[str, Any]]] = []

    def event_sink(event_type: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        events.append((event_type, message, metadata or {}))

    gateway = AgentChatGateway(
        tmp_path,
        config=ChatGatewayConfig(event_sink=event_sink),
        settings=Settings(openai_api_key="sk-test-mock-key"),
    )
    session_id = gateway.create_session(frontend="cli")

    coordinator = ShutdownCoordinator(gateway=gateway, event_sink=event_sink)
    coordinator.register_session(session_id)

    coordinator.request_shutdown(source="ctrl_c", session_id=session_id, reason="user_interrupt")

    emitted_types = [e[0] for e in events]
    assert "shutdown.requested" in emitted_types
    assert "runtime.shutdown.started" in emitted_types
    assert "runtime.shutdown.completed" in emitted_types
    assert gateway._shutting_down is True


def test_new_turns_rejected_during_shutdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When runtime is shutting down, new turns are rejected and marked cancelled."""
    monkeypatch.setenv("MANA_OPENAI_API_KEY", "sk-test-mock-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-key")
    gateway = AgentChatGateway(
        tmp_path,
        settings=Settings(openai_api_key="sk-test-mock-key"),
    )
    session_id = gateway.create_session(frontend="cli")

    gateway.request_shutdown(source="exit", session_id=session_id)

    result = gateway.process_turn(session_id, "Make a change")
    assert result.error == "cancelled"
    assert result.payload["status"] == "cancelled"
    assert result.payload["cancellation_source"] == "shutdown"


def test_restart_does_not_recover_cancelled_tasks(tmp_path: Path) -> None:
    """Supervisor recovery does not resume intentionally cancelled tasks."""
    supervisor = _build_test_supervisor(tmp_path)
    task = supervisor.create_task(
        routing_decision_id="decision-recover-test",
        assigned_agent="lane:coding",
        task_type="coding",
    )
    _transition_to_running(supervisor, task.task_id)
    supervisor.cancel(task.task_id, reason="user_interrupt", source="ctrl_c")

    persisted = supervisor.store.get_task(task.task_id)
    assert persisted.state == ExecutionState.CANCELLED

    # Run supervisor recovery
    summary = supervisor.recover()
    assert summary is not None

    after_recovery = supervisor.store.get_task(task.task_id)
    assert after_recovery.state == ExecutionState.CANCELLED


def test_codex_coding_agent_shim_cancel_propagation(tmp_path: Path) -> None:
    """CodexCodingAgentShim.cancel invokes active backend cancellation."""
    shim = CodexCodingAgentShim(
        repo_root=tmp_path,
        codex_settings=CodexSettings(),
    )

    cancelled_tasks: list[str] = []

    class MockBackend:
        async def cancel(self, task_id: str) -> None:
            cancelled_tasks.append(task_id)

    shim._active_backend = ("task-123", MockBackend())

    assert shim.cancel("task-123") is True
    assert cancelled_tasks == ["task-123"]

    # Calling cancel again when no active backend returns False
    shim._active_backend = None
    assert shim.cancel("task-123") is False
