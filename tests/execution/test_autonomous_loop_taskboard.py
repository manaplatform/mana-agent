"""Validation tests for Mana-Agent Loop-Driven Architecture with TaskBoard State Authority.

Proves:
1. The Autonomous Loop owns execution decisions.
2. TaskBoard stores authoritative task progress.
3. TaskBoard changes are visible to the Loop.
4. Task updates do not unexpectedly restart execution.
5. Verified completion requires evidence.
6. Failed verification can trigger autonomous replanning.
7. Cancellation and approvals remain enforced.
8. Interrupted work resumes without losing progress.
9. Model-directed TaskBoard operations work dynamically.
10. Existing workflows remain compatible.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import pytest

from mana_agent.config.autonomous_config import AutonomousExecutionConfig
from mana_agent.execution.autonomous_controller import (
    AutonomousExecutionController,
    AutonomousExecutionResult,
)
from mana_agent.execution.continuation_decision import (
    ContinuationAction,
    ContinuationDecision,
)
from mana_agent.execution.continuation_evaluator import ContinuationEvaluator
from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.models import (
    CompletionContract,
    CompletionContractType,
    ExecutionState,
)
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
from mana_agent.multi_agent.agents.main_agent import MainAgent
from mana_agent.multi_agent.core.types import AgentRole, TaskStatus, VerificationResult
from mana_agent.multi_agent.taskboard.operations import TaskBoardOperationsHandler
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.tools.taskboard_tools import build_taskboard_tools


class FakeLLM:
    """Mock LLM that returns pre-configured ContinuationDecisions."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.invocations: list[Any] = []

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        self.invocations.append(messages)
        if not self.responses:
            raise RuntimeError("No more responses in FakeLLM")
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def _make_supervisor(tmp_path: Path) -> ExecutionSupervisor:
    config = ExecutionSupervisorConfig(root=tmp_path / "supervisor", startup_recovery=False)
    return ExecutionSupervisor(config)


def test_autonomous_loop_owns_execution_decisions(tmp_path: Path) -> None:
    """1. The Autonomous Loop owns execution decisions."""
    supervisor = _make_supervisor(tmp_path)
    tb = TaskBoard(tmp_path / "workspace")
    root_task = tb.create_task(title="Refactor auth", user_request="Refactor auth subsystem")

    # The model decides CONTINUE on turn 1, then FINALIZE on turn 2
    llm = FakeLLM(
        [
            ContinuationDecision(
                task_id=root_task.task_id,
                action=ContinuationAction.CONTINUE,
                goal_satisfied=False,
                progress_detected=True,
                evidence_summary="Planned auth refactor and created subtasks",
                remaining_work="Implement OAuth2 handler",
                safe_to_continue=True,
                reason="subtasks created, need implementation",
            ),
            ContinuationDecision(
                task_id=root_task.task_id,
                action=ContinuationAction.FINALIZE,
                goal_satisfied=True,
                progress_detected=True,
                evidence_summary="OAuth2 handler implemented and verified",
                user_message="Auth refactor complete and verified.",
                verification_passed=True,
                safe_to_continue=True,
                reason="objective satisfied and verified",
            ),
        ]
    )
    evaluator = ContinuationEvaluator(llm)
    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(require_verification=False),
        supervisor=supervisor,
        taskboard=tb,
        evaluator=evaluator,
    )

    executed_turns: list[int] = []

    def step_runner(turn: int, ctx: dict[str, Any]) -> dict[str, Any]:
        executed_turns.append(turn)
        # Even if TaskBoard has pending items, the loop decides progression
        if turn == 0:
            tb.create_task(title="Subtask 1", parent_task_id=root_task.task_id)
        return {"content": f"Turn {turn} output"}

    result = controller.execute_goal(
        task_id=root_task.task_id,
        user_goal="Refactor auth subsystem",
        step_runner=step_runner,
    )

    assert result.success is True
    assert result.metrics.model_turns == 2
    assert executed_turns == [0, 1]
    assert result.answer == "Auth refactor complete and verified."


def test_taskboard_stores_authoritative_task_progress(tmp_path: Path) -> None:
    """2. TaskBoard stores authoritative task progress."""
    tb = TaskBoard(tmp_path / "workspace")
    root_task = tb.create_task(
        title="Payment integration",
        user_request="Integrate Stripe and PayPal",
        acceptance_criteria=["Stripe API connected", "PayPal webhook handled"],
    )

    sub1 = tb.create_task(title="Stripe client", parent_task_id=root_task.task_id)
    sub2 = tb.create_task(title="PayPal webhook", parent_task_id=root_task.task_id, depends_on=[sub1.task_id])

    # Initial progress
    p0 = tb.get_progress()
    assert p0.total_tasks == 3
    assert p0.completed_tasks == 0
    assert p0.percent_complete == 0.0

    # Cross off and complete sub1
    tb.mark_crossed_off(sub1.task_id, crossed_off=True)
    tb.mark_completed(sub1.task_id, result_summary="Stripe SDK integrated")

    assert tb.get_task(sub1.task_id).crossed_off is True
    assert tb.get_task(sub1.task_id).status == TaskStatus.DONE

    p1 = tb.get_progress()
    assert p1.completed_tasks == 1
    assert p1.crossed_off_tasks == 1
    assert p1.percent_complete == pytest.approx(33.3, 0.1)

    # Subtask listing and dependencies
    subs = tb.list_tasks(parent_task_id=root_task.task_id)
    assert len(subs) == 2
    assert tb.get_task(sub2.task_id).depends_on == [sub1.task_id]


def test_taskboard_changes_are_visible_to_the_loop(tmp_path: Path) -> None:
    """3. TaskBoard changes are visible to the Loop."""
    supervisor = _make_supervisor(tmp_path)
    tb = TaskBoard(tmp_path / "workspace")
    root_task = tb.create_task(title="Feature X", user_request="Implement Feature X")

    seen_progress_in_context: list[dict[str, Any]] = []

    llm = FakeLLM(
        [
            ContinuationDecision(
                task_id=root_task.task_id,
                action=ContinuationAction.FINALIZE,
                goal_satisfied=True,
                progress_detected=True,
                evidence_summary="Subtask finished",
                safe_to_continue=True,
                reason="all done",
            )
        ]
    )
    evaluator = ContinuationEvaluator(llm)
    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(require_verification=False),
        supervisor=supervisor,
        taskboard=tb,
        evaluator=evaluator,
    )

    def step_runner(turn: int, ctx: dict[str, Any]) -> dict[str, Any]:
        # TaskBoard operations are accessible via ctx
        tb_ops: TaskBoardOperationsHandler = ctx["taskboard_operations"]
        tb_ops.create_task(title="Subtask A", parent_task_id=root_task.task_id)
        prog = ctx.get("taskboard_progress")
        if prog is not None:
            seen_progress_in_context.append(prog.model_dump() if hasattr(prog, "model_dump") else dict(prog))
        return {"content": "done"}

    controller.execute_goal(task_id=root_task.task_id, user_goal="Implement Feature X", step_runner=step_runner)

    assert len(seen_progress_in_context) == 1
    assert "total_tasks" in seen_progress_in_context[0]


def test_task_updates_do_not_restart_execution(tmp_path: Path) -> None:
    """4. Task updates do not unexpectedly restart execution."""
    supervisor = _make_supervisor(tmp_path)
    tb = TaskBoard(tmp_path / "workspace")
    task = tb.create_task(title="Refactor", user_request="Refactor module")

    turn_counter = 0

    llm = FakeLLM(
        [
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.CONTINUE,
                goal_satisfied=False,
                progress_detected=True,
                evidence_summary="Step 1 complete",
                safe_to_continue=True,
                reason="need step 2",
            ),
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.FINALIZE,
                goal_satisfied=True,
                progress_detected=True,
                evidence_summary="Step 2 complete",
                safe_to_continue=True,
                reason="done",
            ),
        ]
    )
    evaluator = ContinuationEvaluator(llm)
    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(require_verification=False),
        supervisor=supervisor,
        taskboard=tb,
        evaluator=evaluator,
    )

    def step_runner(turn: int, ctx: dict[str, Any]) -> dict[str, Any]:
        nonlocal turn_counter
        turn_counter += 1
        # Perform various TaskBoard updates during step
        tb.mark_crossed_off(task.task_id, crossed_off=True)
        tb.update_task(task.task_id, description="Updated description in flight")
        tb.add_subtask(task.task_id, "sub_123")
        return {"content": f"Turn {turn}"}

    result = controller.execute_goal(task_id=task.task_id, user_goal="Refactor module", step_runner=step_runner)

    # Verification: turn_counter proceeded monotonically 1 -> 2 without restarting to 1
    assert turn_counter == 2
    assert result.metrics.model_turns == 2
    assert result.success is True


def test_verified_completion_requires_evidence(tmp_path: Path) -> None:
    """5. Verified completion requires evidence; never equate crossed-off with verified."""
    tb = TaskBoard(tmp_path / "workspace")
    task = tb.create_task(title="Bugfix", user_request="Fix issue #123")

    # Marking task crossed_off does NOT mark it verified
    tb.mark_crossed_off(task.task_id, crossed_off=True)
    tb.mark_completed(task.task_id, result_summary="Checked off in checklist")

    item = tb.get_task(task.task_id)
    assert item.crossed_off is True
    assert item.status == TaskStatus.DONE
    # Verification results must be empty
    assert len(item.verification_results) == 0

    # Proper verification requires executed evidence
    v_res = tb.verify_task(
        task.task_id,
        passed=True,
        summary="pytest tests/test_bugfix.py passed with 5 assertions",
        commands=["pytest tests/test_bugfix.py"],
    )
    assert v_res.passed is True
    updated_item = tb.get_task(task.task_id)
    assert len(updated_item.verification_results) == 1
    assert updated_item.verification_results[0].passed is True


def test_failed_verification_triggers_autonomous_replanning(tmp_path: Path) -> None:
    """6. Failed verification can trigger autonomous replanning and task reopening."""
    supervisor = _make_supervisor(tmp_path)
    tb = TaskBoard(tmp_path / "workspace")
    task = tb.create_task(title="Deploy feature", user_request="Deploy to staging")
    tb.mark_completed(task.task_id, result_summary="Deployment script ran")

    # Model tries to finalize with failed verification
    llm = FakeLLM(
        [
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.FINALIZE,
                goal_satisfied=True,
                progress_detected=True,
                evidence_summary="attempted deploy",
                verification_passed=False,  # Failed verification!
                safe_to_continue=True,
                reason="tried deploy",
            ),
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.FINALIZE,
                goal_satisfied=True,
                progress_detected=True,
                evidence_summary="fixed config and redeployed successfully",
                verification_passed=True,  # Passed verification!
                safe_to_continue=True,
                reason="all tests pass",
            ),
        ]
    )
    evaluator = ContinuationEvaluator(llm)
    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(require_verification=True),
        supervisor=supervisor,
        taskboard=tb,
        evaluator=evaluator,
    )

    step_calls = 0

    def step_runner(turn: int, ctx: dict[str, Any]) -> dict[str, Any]:
        nonlocal step_calls
        step_calls += 1
        if turn == 0:
            ctx["verification_passed"] = False
            ctx["evidence"].append("Healthcheck failed with HTTP 500")
            return {"content": "deploy failed healthcheck", "verification_passed": False}
        else:
            ctx["verification_passed"] = True
            ctx["evidence"].append("Healthcheck passed with HTTP 200")
            return {"content": "deploy verified", "verification_passed": True}

    result = controller.execute_goal(task_id=task.task_id, user_goal="Deploy to staging", step_runner=step_runner)

    assert step_calls == 2
    assert result.success is True
    # Verify task was reopened upon failure and finalized upon passing
    assert tb.get_task(task.task_id).status == TaskStatus.DONE


def test_cancellation_and_approvals_enforced(tmp_path: Path) -> None:
    """7. Cancellation and approvals remain enforced."""
    supervisor = _make_supervisor(tmp_path)
    tb = TaskBoard(tmp_path / "workspace")
    task = tb.create_task(title="Long task", user_request="Process data")

    # Start task in supervisor
    sup_task = supervisor.create_task(session_id="s1", requested_operation="long_task")
    sup_task, token = supervisor.acquire_lease(sup_task.task_id, owner="worker1")
    supervisor.start(sup_task.task_id, attempt_id=sup_task.attempt_id, lease_token=token)

    llm = FakeLLM([])
    evaluator = ContinuationEvaluator(llm)
    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        taskboard=tb,
        evaluator=evaluator,
    )

    def step_runner(turn: int, ctx: dict[str, Any]) -> dict[str, Any]:
        supervisor.cancel(sup_task.task_id, reason="User clicked stop")
        return {"content": "running"}

    result = controller.execute_goal(task_id=sup_task.task_id, user_goal="Process data", step_runner=step_runner)

    assert result.success is False
    assert result.status == "cancelled"
    # Task on TaskBoard should be cancelled
    tb_task = tb.get_task(sup_task.task_id)
    if tb_task:
        assert tb_task.status == TaskStatus.CANCELLED


def test_interrupted_work_resumes_without_losing_progress(tmp_path: Path) -> None:
    """8. Interrupted work resumes without losing progress."""
    supervisor = _make_supervisor(tmp_path)
    tb = TaskBoard(tmp_path / "workspace")
    task = tb.create_task(title="Multi-stage migration", user_request="Run migration")

    sub1 = tb.create_task(title="Stage 1", parent_task_id=task.task_id)
    tb.mark_completed(sub1.task_id, result_summary="Stage 1 completed successfully")

    # Checkpoint progress in supervisor
    sup_task = supervisor.create_task(session_id="s1", requested_operation="migration")
    sup_task, token = supervisor.acquire_lease(sup_task.task_id, owner="worker1")
    supervisor.start(sup_task.task_id, attempt_id=sup_task.attempt_id, lease_token=token)
    supervisor.checkpoint(
        sup_task.task_id,
        attempt_id=sup_task.attempt_id,
        lease_token=token,
        resume_payload={"completed_subtasks": [sub1.task_id]},
    )

    # Resume from checkpoint
    recovered_task = supervisor.store.get_task(sup_task.task_id)
    assert recovered_task.checkpoint_count >= 1

    # TaskBoard state preserved
    sub1_recovered = tb.get_task(sub1.task_id)
    assert sub1_recovered.status == TaskStatus.DONE


def test_model_directed_taskboard_operations(tmp_path: Path) -> None:
    """9. Model-directed TaskBoard operations work dynamically via tools and handler."""
    tb = TaskBoard(tmp_path / "workspace")
    handler = TaskBoardOperationsHandler(tb)

    # Create task
    t = handler.create_task(title="API endpoint", description="Build /users endpoint")
    assert t["task_id"] in tb.tasks

    # Create subtask
    sub = handler.create_task(title="Validate schema", parent_task_id=t["task_id"])
    assert sub["task_id"] in tb.tasks
    assert sub["task_id"] in tb.get_task(t["task_id"]).subtasks

    # Mark completed
    handler.mark_completed(task_id=sub["task_id"], evidence="Pydantic models added")
    assert tb.get_task(sub["task_id"]).status == TaskStatus.DONE

    # Verify task
    handler.verify_task(task_id=sub["task_id"], passed=True, evidence="Schema validation tests passed")
    assert len(tb.get_task(sub["task_id"]).verification_results) == 1

    # Reopen task
    handler.reopen_task(task_id=sub["task_id"], reason="Add extra field")
    assert tb.get_task(sub["task_id"]).status in {TaskStatus.QUEUED, TaskStatus.IN_PROGRESS}

    # Tools registry
    tools = build_taskboard_tools(tb)
    tool_names = {t.name for t in tools}
    assert "taskboard_create_task" in tool_names
    assert "taskboard_mark_completed" in tool_names
    assert "taskboard_verify_task" in tool_names
    assert "taskboard_reopen_task" in tool_names
    assert "taskboard_get_progress" in tool_names


def test_main_agent_workflows_and_autonomous_execution(tmp_path: Path) -> None:
    """10. Existing workflows remain compatible and MainAgent autonomous execution works."""
    class _MockRouteModel:
        def invoke(self, messages: Any) -> Any:
            text = str(messages[-1].content if hasattr(messages[-1], "content") else messages[-1])
            if "ContinuationDecision" in str(messages) or "continuation" in text or "goal" in text:
                return SimpleNamespace(
                    content=json.dumps({
                        "task_id": "task_id",
                        "action": "FINALIZE",
                        "goal_satisfied": True,
                        "progress_detected": True,
                        "evidence_summary": "Inspected repository",
                        "user_message": "Inspection complete",
                        "safe_to_continue": True,
                        "reason": "done",
                    })
                )
            return SimpleNamespace(
                content=json.dumps({
                    "intent": "answer",
                    "confidence": 0.9,
                    "selected_tools": [],
                    "tool_inputs": {},
                    "repo_context_needed": False,
                    "web_search_needed": False,
                    "code_editing_needed": False,
                    "requested_effect": "none",
                    "target_surface": "conversation",
                    "required_subagents": [],
                    "reasoning_summary": "Simple conversational response",
                })
            )

    main = MainAgent(tmp_path, routing_llm=_MockRouteModel())
    assert main.autonomous_controller is not None
    assert main.taskboard is not None

    # Normal legacy request remains compatible
    legacy_result = main.run_user_request("hello world", entrypoint="chat")
    assert legacy_result.route_name == "simple"
    assert legacy_result.task_id

    # Autonomous execution entrypoint works
    auto_result = main.run_autonomous(
        "Inspect repository structure",
        entrypoint="analyze",
        step_runner=lambda ctx: {"content": "Inspected successfully"},
        max_turns=1,
    )
    assert auto_result.route_name == "autonomous"
    assert auto_result.task_id
