"""End-to-End integration tests for Live Steer Mode, controller integration, and lease fencing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mana_agent.chat.events import (
    AssistantMessageEvent,
    CodingActivityEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from mana_agent.chat.execution_trace import ExecutionTrace
from mana_agent.chat.history import ChatHistory
from mana_agent.execution.autonomous_controller import (
    AutonomousExecutionConfig,
    AutonomousExecutionController,
)
from mana_agent.execution.continuation_decision import (
    ContinuationAction,
    ContinuationDecision,
)
from mana_agent.execution.steering import (
    SteeringCoordinator,
    SteeringKind,
    SteeringScope,
    SteeringSubmission,
)
from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.errors import LeaseConflictError, StaleLeaseError
from mana_agent.execution_supervisor.models import ExecutionState
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
from mana_agent.tui.app import ManaChatApp
from mana_agent.tui.steer_modal import SteerModal
from mana_agent.tui.widgets.chat_log import ChatLog


@pytest.fixture
def supervisor() -> ExecutionSupervisor:
    return ExecutionSupervisor(
        ExecutionSupervisorConfig(lease_seconds=30, heartbeat_seconds=5)
    )


def make_running_task(
    supervisor: ExecutionSupervisor,
    task_id: str,
    owner: str = "worker_1",
    session_id: str = "session_1",
) -> tuple[Any, str]:
    task = supervisor.create_task(task_id=task_id, task_type="coding", session_id=session_id)
    supervisor.queue(task.task_id)
    leased, token = supervisor.acquire_lease(task.task_id, owner=owner)
    started = supervisor.start(task.task_id, attempt_id=leased.attempt_id, lease_token=token)
    return started, token


def test_controller_acquires_valid_supervisor_lease_and_generation_fence(
    supervisor: ExecutionSupervisor,
) -> None:
    """Requirement 2 & 10.2: Controller acquires valid supervisor lease and fences operations."""
    task1, token1 = make_running_task(supervisor, "task_live_1", owner="controller_worker")
    assert task1.attempt_id
    assert token1
    assert task1.attempt_generation == 1

    # Valid active-owner heartbeat
    hb_task = supervisor.heartbeat(
        "task_live_1",
        attempt_id=task1.attempt_id,
        lease_token=token1,
        generation=task1.attempt_generation,
    )
    assert hb_task.state == ExecutionState.RUNNING

    # Valid checkpoint under attempt generation
    cp = supervisor.checkpoint(
        "task_live_1",
        attempt_id=task1.attempt_id,
        lease_token=token1,
        generation=task1.attempt_generation,
        resume_payload={"step": 1},
        resume_cursor="cursor_1",
    )
    assert cp.attempt_id == task1.attempt_id

    # Simulate attempt replacement (e.g. worker crash / supervisor recovery)
    supervisor.transition(
        task1.task_id,
        ExecutionState.RETRY_SCHEDULED,
        reason="worker failure",
        attempt_id=task1.attempt_id,
        generation=task1.attempt_generation,
    )
    supervisor.transition(task1.task_id, ExecutionState.QUEUED, reason="retry")
    leased2, token2 = supervisor.acquire_lease(task1.task_id, owner="replacement_worker")
    task2 = supervisor.start(task1.task_id, attempt_id=leased2.attempt_id, lease_token=token2)
    assert task2.attempt_generation == 2
    assert task2.attempt_id != task1.attempt_id

    # Old attempt heartbeat MUST be rejected with StaleLeaseError
    with pytest.raises(StaleLeaseError):
        supervisor.heartbeat(
            "task_live_1",
            attempt_id=task1.attempt_id,
            lease_token=token1,
            generation=task1.attempt_generation,
        )

    # Old attempt checkpoint MUST be rejected with StaleLeaseError
    with pytest.raises(StaleLeaseError):
        supervisor.checkpoint(
            "task_live_1",
            attempt_id=task1.attempt_id,
            lease_token=token1,
            generation=task1.attempt_generation,
            resume_payload={"step": 2},
            resume_cursor="cursor_2",
        )

    # Old attempt completion MUST be rejected and must NOT overwrite new attempt
    with pytest.raises(StaleLeaseError):
        supervisor.submit_result(
            "task_live_1",
            attempt_id=task1.attempt_id,
            lease_token=token1,
            generation=task1.attempt_generation,
            payload={"done": True},
        )

    # Active new attempt succeeds
    current = supervisor.get_task("task_live_1")
    assert current.state == ExecutionState.RUNNING
    assert current.attempt_id == task2.attempt_id
    assert current.attempt_generation == 2


def test_live_steering_delivery_at_safe_boundary(supervisor: ExecutionSupervisor) -> None:
    """Requirement 5 & 10.5-10.7: Steering delivered at safe step boundary to active controller."""
    task, token = make_running_task(supervisor, "task_steer_1", owner="controller_worker")

    coord = SteeringCoordinator(supervisor)

    # Submit live steering instruction
    sub = SteeringSubmission(
        execution_id="task_steer_1",
        session_id="session_1",
        attempt_id=task.attempt_id,
        generation=task.attempt_generation,
        request_id="steer_req_1",
        text="Prioritize security tests first",
        kind=SteeringKind.GUIDANCE,
        scope=SteeringScope.EXECUTION,
    )
    seq, dup = coord.submit(sub, authenticated_session_id="session_1", allow_scoped=True)
    assert seq == 1
    assert not dup

    # Step executions recorded
    executed_steps: list[dict[str, Any]] = []

    def mock_step_runner(step_idx: int, runner_ctx: dict[str, Any]) -> dict[str, Any]:
        executed_steps.append({
            "step": step_idx,
            "guidance": runner_ctx.get("live_steering_guidance", ""),
        })
        if step_idx == 0:
            return {
                "answer": "Ran initial tests",
                "tool_calls": [{"name": "run_tests", "args": {}}],
            }
        else:
            return {
                "answer": "Ran security tests as guided",
                "tool_calls": [{"name": "run_security_tests", "args": {}}],
                "goal_satisfied": True,
            }

    class MockEvaluator:
        def evaluate(self, **kwargs: Any) -> ContinuationDecision:
            step_out = kwargs.get("step_output", {})
            if step_out.get("answer") == "Ran security tests as guided":
                return ContinuationDecision(
                    task_id="task_steer_1",
                    action=ContinuationAction.FINALIZE,
                    goal_satisfied=True,
                    safe_to_continue=True,
                    reason="Security tests completed",
                    evidence_summary="Security tests verified",
                )
            return ContinuationDecision(
                task_id="task_steer_1",
                action=ContinuationAction.CONTINUE,
                goal_satisfied=False,
                safe_to_continue=True,
                reason="Need security tests",
                evidence_summary="Initial tests ran",
            )

    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(max_model_turns=5, max_tool_calls=10),
        supervisor=supervisor,
        evaluator=MockEvaluator(),
    )

    res = controller.execute_goal(
        task_id="task_steer_1",
        user_goal="Run test suite",
        step_runner=mock_step_runner,
        attempt_id=task.attempt_id,
        lease_token=token,
    )

    assert res.goal_satisfied
    assert res.status == "completed"
    # First step saw the steering guidance applied
    assert len(executed_steps) >= 1
    assert "Prioritize security tests first" in executed_steps[0]["guidance"]

    # Verify instruction is now marked applied in supervisor task
    updated_task = supervisor.get_task("task_steer_1")
    applied_items = [i for i in updated_task.steering_inputs if i["request_id"] == "steer_req_1"]
    assert len(applied_items) == 1
    assert applied_items[0]["status"] == "applied"


def test_scoped_multi_agent_worker_routing(supervisor: ExecutionSupervisor) -> None:
    """Requirement 6 & 10.9: Global, task-scoped, and worker-scoped steering routing."""
    task, token = make_running_task(supervisor, "task_multi_1", owner="lead_worker", session_id="sess_1")
    coord = SteeringCoordinator(supervisor)

    # 1. Global execution steering
    coord.submit(
        SteeringSubmission(
            execution_id="task_multi_1",
            session_id="sess_1",
            attempt_id=task.attempt_id,
            generation=task.attempt_generation,
            request_id="global_1",
            text="Use mock endpoints only",
            kind=SteeringKind.CONSTRAINT,
            scope=SteeringScope.EXECUTION,
        ),
        authenticated_session_id="sess_1",
        allow_scoped=True,
    )

    # 2. Worker-1 scoped steering
    coord.submit(
        SteeringSubmission(
            execution_id="task_multi_1",
            session_id="sess_1",
            attempt_id=task.attempt_id,
            generation=task.attempt_generation,
            request_id="worker_1_steer",
            text="Focus on backend auth endpoints",
            kind=SteeringKind.GUIDANCE,
            scope=SteeringScope.WORKER,
            target_id="worker_auth",
        ),
        authenticated_session_id="sess_1",
        allow_scoped=True,
    )

    # 3. Worker-2 scoped steering
    coord.submit(
        SteeringSubmission(
            execution_id="task_multi_1",
            session_id="sess_1",
            attempt_id=task.attempt_id,
            generation=task.attempt_generation,
            request_id="worker_2_steer",
            text="Focus on frontend login form",
            kind=SteeringKind.GUIDANCE,
            scope=SteeringScope.WORKER,
            target_id="worker_ui",
        ),
        authenticated_session_id="sess_1",
        allow_scoped=True,
    )

    # Query for worker_auth: should receive global + worker_auth, NOT worker_ui
    auth_pending = coord.pending_for_attempt(
        "task_multi_1",
        attempt_id=task.attempt_id,
        generation=task.attempt_generation,
        lease_token=token,
        scope="worker",
        target_id="worker_auth",
    )
    auth_req_ids = {item["request_id"] for item in auth_pending}
    assert "global_1" in auth_req_ids
    assert "worker_1_steer" in auth_req_ids
    assert "worker_2_steer" not in auth_req_ids

    # Query for worker_ui: should receive global + worker_ui, NOT worker_auth
    ui_pending = coord.pending_for_attempt(
        "task_multi_1",
        attempt_id=task.attempt_id,
        generation=task.attempt_generation,
        lease_token=token,
        scope="worker",
        target_id="worker_ui",
    )
    ui_req_ids = {item["request_id"] for item in ui_pending}
    assert "global_1" in ui_req_ids
    assert "worker_2_steer" in ui_req_ids
    assert "worker_1_steer" not in ui_req_ids


def test_plan_changing_steering_requests_approval(supervisor: ExecutionSupervisor) -> None:
    """Requirement 7 & 10.10: Steering that changes approved plan transitions to WAITING for approval."""
    task, token = make_running_task(supervisor, "task_plan_1", owner="controller_worker")
    coord = SteeringCoordinator(supervisor)

    # Submit replan steering with requires_approval flag
    sub = SteeringSubmission(
        execution_id="task_plan_1",
        session_id="session_1",
        attempt_id=task.attempt_id,
        generation=task.attempt_generation,
        request_id="steer_replan_1",
        text="Switch database architecture to Postgres",
        kind=SteeringKind.REPLAN,
        scope=SteeringScope.EXECUTION,
        requires_approval=True,
    )
    coord.submit(sub, authenticated_session_id="session_1", allow_scoped=True)

    def mock_step_runner(step_idx: int, runner_ctx: dict[str, Any]) -> dict[str, Any]:
        return {"output": "should not execute"}

    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
    )

    res = controller.execute_goal(
        task_id="task_plan_1",
        user_goal="Refactor DB",
        step_runner=mock_step_runner,
        attempt_id=task.attempt_id,
        lease_token=token,
    )

    # Execution paused for human approval
    assert res.status == "waiting"
    assert res.error == "waiting_for_approval"
    assert not res.goal_satisfied

    # Supervisor state transitioned to WAITING
    curr = supervisor.get_task("task_plan_1")
    assert curr.state == ExecutionState.WAITING


def test_failed_execution_never_displays_completed_terminal_status() -> None:
    """Requirement 9 & 10.11: FAILED execution must never show ✓ Completed."""
    trace = ExecutionTrace(turn_id="turn_fail_1")

    # Start turn
    trace.apply_event({
        "event_id": "ev_start",
        "event_type": "turn.started",
        "turn_id": "turn_fail_1",
    })

    # Fail execution with error
    trace.apply_event({
        "event_id": "ev_fail",
        "event_type": "execution.failed",
        "turn_id": "turn_fail_1",
        "error": "lease token is stale or invalid",
    })

    assert trace.is_failed
    assert not trace.is_completed

    # Subsequent spurious completion event must be rejected
    trace.apply_event({
        "event_id": "ev_comp",
        "event_type": "turn.completed",
        "turn_id": "turn_fail_1",
    })

    # Still marked as failed, never completed!
    assert trace.is_failed
    assert not trace.is_completed


def test_tui_steer_modal_and_ctrl_s_interaction() -> None:
    asyncio.run(_tui_steer_modal_and_ctrl_s_interaction())


async def _tui_steer_modal_and_ctrl_s_interaction() -> None:
    """Requirement 4 & 10.3-10.4: Ctrl+S and /steer trigger non-blocking steering in TUI."""
    mock_gateway = MagicMock()
    mock_gateway.create_session.return_value = "session_tui_1"
    mock_gateway.submit_steering.return_value = {
        "status": "pending",
        "sequence": 1,
        "duplicate": False,
    }

    history = ChatHistory()
    app = ManaChatApp(
        history=history,
        gateway=mock_gateway,
        repo_root=Path("."),
    )

    async with app.run_test() as pilot:
        # Mark turn in progress
        app._turn_in_progress = True

        # Test Ctrl+S opens SteerModal
        await pilot.press("ctrl+s")
        await pilot.pause()

        # Verify SteerModal is on screen
        assert isinstance(app.screen, SteerModal)

        # Enter instruction in SteerModal
        steer_input = app.screen.query_one("#steer-input")
        steer_input.value = "Prioritize security checks"
        await pilot.press("enter")
        await pilot.pause()

        # Modal dismissed
        assert not isinstance(app.screen, SteerModal)

        # Verify gateway.submit_steering was called
        mock_gateway.submit_steering.assert_called_with(
            session_id="session_tui_1",
            text="Prioritize security checks",
            scope="execution",
        )

        # Verify steering submitted message appeared in history
        steering_events = [
            e for e in history.get_events()
            if isinstance(e, AssistantMessageEvent) and "Live Steering submitted" in e.content
        ]
        assert len(steering_events) == 1
        assert "Prioritize security checks" in steering_events[0].content


def test_tui_slash_steer_fallback_during_turn() -> None:
    asyncio.run(_tui_slash_steer_fallback_during_turn())


async def _tui_slash_steer_fallback_during_turn() -> None:
    """Requirement 4: Fallback /steer <instruction> works while turn is in progress."""
    mock_gateway = MagicMock()
    mock_gateway.create_session.return_value = "session_tui_2"
    mock_gateway.submit_steering.return_value = {
        "status": "pending",
        "sequence": 2,
        "duplicate": False,
    }

    history = ChatHistory()
    app = ManaChatApp(
        history=history,
        gateway=mock_gateway,
        repo_root=Path("."),
    )

    async with app.run_test() as pilot:
        # Turn actively running
        app._turn_in_progress = True

        # Submit /steer command in chat input
        msg_input = app.query_one("#chat-input")
        msg_input.text = "/steer Run edge-case tests"
        from mana_agent.tui.widgets.message_input import MessageInput
        await app.on_message_input_submitted(MessageInput.Submitted(msg_input))
        await pilot.pause()

        # Verify gateway.submit_steering was called with the instruction
        mock_gateway.submit_steering.assert_called_with(
            session_id="session_tui_2",
            text="Run edge-case tests",
            scope="execution",
        )
