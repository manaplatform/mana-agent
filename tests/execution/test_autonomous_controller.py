"""Comprehensive tests for Universal Model-Directed Autonomous Execution Controller."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest
from pydantic import ValidationError

from mana_agent.config.autonomous_config import AutoRenewConfig, AutonomousExecutionConfig
from mana_agent.execution.autonomous_controller import (
    AutonomousExecutionController,
    AutonomousExecutionResult,
    TaskExecutionMetrics,
)
from mana_agent.execution.continuation_decision import (
    ContinuationAction,
    ContinuationDecision,
    ContinuationDecisionError,
)
from mana_agent.execution.continuation_evaluator import ContinuationEvaluator
from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.models import (
    CompletionContract,
    CompletionContractType,
    ExecutionState,
)
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor


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


def test_autonomous_config_defaults() -> None:
    config = AutonomousExecutionConfig()
    assert config.mode == "model_directed"
    assert config.max_model_turns == 120
    assert config.max_tool_calls == 200
    assert config.require_verification is True
    assert config.auto_renew.enabled is True
    assert config.auto_renew.strategy == "accounting_based"
    assert config.auto_renew.renew_model_turns == 40
    assert config.auto_renew.renew_tool_calls == 60
    assert config.auto_renew.max_auto_renewals == 3
    assert config.auto_renew.require_progress is True
    assert config.auto_renew.respect_account_budget is True
    assert config.auto_renew.require_approval_above_budget is True
    assert config.auto_renew.stop_on_no_progress is True


def test_continuation_decision_invariants() -> None:
    # 1. FINALIZE requires goal_satisfied=True
    with pytest.raises(ValidationError):
        ContinuationDecision(
            task_id="task-1",
            action=ContinuationAction.FINALIZE,
            goal_satisfied=False,  # Invalid for FINALIZE
            progress_detected=True,
            evidence_summary="done",
            safe_to_continue=True,
            reason="done",
        )

    # 2. CONTINUE requires goal_satisfied=False
    with pytest.raises(ValidationError):
        ContinuationDecision(
            task_id="task-1",
            action=ContinuationAction.CONTINUE,
            goal_satisfied=True,  # Invalid for CONTINUE
            progress_detected=True,
            evidence_summary="working",
            safe_to_continue=True,
            reason="working",
        )

    # 3. BLOCKED requires safe_to_continue=False
    with pytest.raises(ValidationError):
        ContinuationDecision(
            task_id="task-1",
            action=ContinuationAction.BLOCKED,
            goal_satisfied=False,
            progress_detected=False,
            evidence_summary="stuck",
            safe_to_continue=True,  # Invalid for BLOCKED
            reason="stuck",
        )

    # 4. renewal_requested requires renewal_reason
    with pytest.raises(ValidationError):
        ContinuationDecision(
            task_id="task-1",
            action=ContinuationAction.CONTINUE,
            goal_satisfied=False,
            progress_detected=True,
            evidence_summary="need more steps",
            safe_to_continue=True,
            renewal_requested=True,
            renewal_reason="",  # Invalid when renewal_requested=True
            reason="need steps",
        )

    # 5. Valid decision
    valid = ContinuationDecision(
        task_id="task-1",
        action=ContinuationAction.FINALIZE,
        goal_satisfied=True,
        progress_detected=True,
        evidence_summary="all requirements verified",
        user_message="The task has been completed successfully.",
        verification_passed=True,
        safe_to_continue=True,
        reason="objective satisfied with evidence",
    )
    assert valid.action == ContinuationAction.FINALIZE
    assert valid.goal_satisfied is True


def test_continuation_evaluator_strict_coercion() -> None:
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "task_id": "task-eval-1",
                    "action": "CONTINUE",
                    "goal_satisfied": False,
                    "progress_detected": True,
                    "evidence_summary": "read index file",
                    "remaining_work": "edit line 42",
                    "safe_to_continue": True,
                    "reason": "first step done, second step needed",
                }
            )
        ]
    )
    evaluator = ContinuationEvaluator(llm)
    decision = evaluator.evaluate(
        task_id="task-eval-1",
        user_goal="Update line 42 in file.py",
        step_output={"content": "file read"},
        completed_steps=[],
        metrics={"turns": 1},
    )
    assert decision.action == ContinuationAction.CONTINUE
    assert decision.progress_detected is True
    assert decision.remaining_work == "edit line 42"


def test_simple_request_stops_on_turn_one(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    task = supervisor.create_task(session_id="s1", requested_operation="answer_hello")
    task, token = supervisor.acquire_lease(task.task_id, owner="worker1")
    supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)

    llm = FakeLLM(
        [
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.FINALIZE,
                goal_satisfied=True,
                progress_detected=True,
                evidence_summary="Greeting answered directly in turn 1",
                user_message="Hello! How can I help you today?",
                safe_to_continue=True,
                reason="simple greeting fully satisfied",
            )
        ]
    )
    evaluator = ContinuationEvaluator(llm)
    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        evaluator=evaluator,
    )

    def simple_step_runner(step_idx: int, ctx: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": "Hello! How can I help you today?",
            "tool_calls": [],
            "tokens_used": 25,
            "cost": 0.0001,
        }

    result = controller.execute_goal(
        task_id=task.task_id,
        user_goal="Say hello",
        step_runner=simple_step_runner,
    )

    assert result.status == "completed"
    assert result.goal_satisfied is True
    assert result.metrics.model_turns == 1
    assert result.metrics.tool_calls == 0
    assert result.answer == "Hello! How can I help you today?"

    updated_task = supervisor.inspect_task(task.task_id)
    assert updated_task.state == ExecutionState.COMPLETED


def test_multi_turn_continuation_pursues_goal(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    task = supervisor.create_task(session_id="s1", requested_operation="multi_step_research")
    task, token = supervisor.acquire_lease(task.task_id, owner="worker1")
    supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)

    llm = FakeLLM(
        [
            # Turn 0: step 1 done, continue
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.CONTINUE,
                goal_satisfied=False,
                progress_detected=True,
                evidence_summary="Search query returned 3 hits",
                remaining_work="Fetch page content for top hit",
                safe_to_continue=True,
                reason="search complete, need read",
            ),
            # Turn 1: step 2 done, finalize
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.FINALIZE,
                goal_satisfied=True,
                progress_detected=True,
                evidence_summary="Fetched and summarized content",
                user_message="Here is the detailed summary of the research.",
                safe_to_continue=True,
                reason="goal satisfied with full evidence",
            ),
        ]
    )
    evaluator = ContinuationEvaluator(llm)
    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        evaluator=evaluator,
    )

    step_calls: list[int] = []

    def multi_step_runner(step_idx: int, ctx: dict[str, Any]) -> dict[str, Any]:
        step_calls.append(step_idx)
        if step_idx == 0:
            return {
                "answer": "Searched query",
                "tool_calls": [{"name": "web_search", "args": {"query": "mana-agent"}}],
                "tokens_used": 150,
                "cost": 0.001,
            }
        else:
            return {
                "answer": "Read content",
                "tool_calls": [{"name": "web_read", "args": {"url": "http://example.com"}}],
                "tokens_used": 300,
                "cost": 0.002,
            }

    result = controller.execute_goal(
        task_id=task.task_id,
        user_goal="Research mana-agent and summarize",
        step_runner=multi_step_runner,
    )

    assert result.status == "completed"
    assert result.goal_satisfied is True
    assert result.metrics.model_turns == 2
    assert result.metrics.tool_calls == 2
    assert step_calls == [0, 1]
    assert "summary" in result.answer


def test_missing_tool_calls_does_not_falsely_finalize(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    task = supervisor.create_task(session_id="s1", requested_operation="write_deliverable")
    task, token = supervisor.acquire_lease(task.task_id, owner="worker1")
    supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)

    # First turn returned no tool calls, but goal is NOT satisfied!
    # Evaluator returns CONTINUE. Second turn actually writes deliverable and finalizes.
    llm = FakeLLM(
        [
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.CONTINUE,
                goal_satisfied=False,
                progress_detected=False,
                evidence_summary="Model answered in prose without creating file",
                remaining_work="Invoke write_to_file tool to create report.md",
                safe_to_continue=True,
                reason="deliverable file missing",
            ),
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.FINALIZE,
                goal_satisfied=True,
                progress_detected=True,
                evidence_summary="report.md created and verified",
                user_message="report.md was successfully generated.",
                safe_to_continue=True,
                reason="file exists on disk",
            ),
        ]
    )
    evaluator = ContinuationEvaluator(llm)
    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        evaluator=evaluator,
    )

    def prose_then_write_runner(step_idx: int, ctx: dict[str, Any]) -> dict[str, Any]:
        if step_idx == 0:
            return {"answer": "I will create report.md now.", "tool_calls": [], "tokens_used": 50}
        return {
            "answer": "Created report.md",
            "tool_calls": [{"name": "write_to_file", "args": {"path": "report.md"}}],
            "changed_files": ["report.md"],
            "tokens_used": 100,
        }

    result = controller.execute_goal(
        task_id=task.task_id,
        user_goal="Generate report.md",
        step_runner=prose_then_write_runner,
    )

    assert result.status == "completed"
    assert result.metrics.model_turns == 2
    assert result.goal_satisfied is True


def test_repeated_no_progress_halts_safely(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    task = supervisor.create_task(session_id="s1", requested_operation="stuck_operation")
    task, token = supervisor.acquire_lease(task.task_id, owner="worker1")
    supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)

    # Every step returns identical output with no state changes
    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(max_no_progress_cycles=2),
        supervisor=supervisor,
        evaluator=FakeLLM([]),  # Should halt on no progress before evaluator loop runs endlessly
    )

    step_count = 0

    def stuck_runner(step_idx: int, ctx: dict[str, Any]) -> dict[str, Any]:
        nonlocal step_count
        step_count += 1
        return {
            "answer": "I cannot find the file",
            "tool_calls": [{"name": "find_file", "args": {"name": "missing.txt"}}],
            "tool_results": [{"found": False}],
            "tokens_used": 50,
        }

    # First turn sets fingerprint. Next two turns match fingerprint.
    # We provide 2 continuation decisions for steps 0 and 1:
    llm = FakeLLM(
        [
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.CONTINUE,
                goal_satisfied=False,
                progress_detected=False,
                evidence_summary="not found",
                remaining_work="retry search",
                safe_to_continue=True,
                reason="retry",
            ),
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.CONTINUE,
                goal_satisfied=False,
                progress_detected=False,
                evidence_summary="not found",
                remaining_work="retry search",
                safe_to_continue=True,
                reason="retry",
            ),
        ]
    )
    controller.evaluator = ContinuationEvaluator(llm)

    result = controller.execute_goal(
        task_id=task.task_id,
        user_goal="Find missing.txt",
        step_runner=stuck_runner,
    )

    assert result.status == "blocked"
    assert result.goal_satisfied is False
    assert "no observable progress" in result.answer
    assert supervisor.inspect_task(task.task_id).state == ExecutionState.FAILED


def test_cancellation_during_execution_stops_cleanly(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    task = supervisor.create_task(session_id="s1", requested_operation="cancellable_task")
    task, token = supervisor.acquire_lease(task.task_id, owner="worker1")
    supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)

    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        evaluator=ContinuationEvaluator(FakeLLM([])),
    )

    def cancel_after_first_step(step_idx: int, ctx: dict[str, Any]) -> dict[str, Any]:
        # Request cancellation during execution
        supervisor.cancel(task.task_id, reason="user_clicked_cancel")
        return {"answer": "step 1", "tool_calls": []}

    llm = FakeLLM(
        [
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.CONTINUE,
                goal_satisfied=False,
                progress_detected=True,
                evidence_summary="step 1 done",
                remaining_work="step 2",
                safe_to_continue=True,
                reason="continue",
            )
        ]
    )
    controller.evaluator = ContinuationEvaluator(llm)

    result = controller.execute_goal(
        task_id=task.task_id,
        user_goal="Run long task",
        step_runner=cancel_after_first_step,
    )

    assert result.status == "cancelled"
    assert result.goal_satisfied is False
    assert supervisor.inspect_task(task.task_id).state == ExecutionState.CANCELLED


def test_consequential_approval_gate_pauses_for_user(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    task = supervisor.create_task(session_id="s1", requested_operation="reboot_server")
    task, token = supervisor.acquire_lease(task.task_id, owner="worker1")
    supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)

    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        evaluator=ContinuationEvaluator(FakeLLM([])),
    )

    def approval_runner(step_idx: int, ctx: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": "Reboot requires approval.",
            "is_approval_pending": True,
            "approval_request_id": "req-srv-reboot-1",
            "action_details": {"command": "reboot"},
        }

    result = controller.execute_goal(
        task_id=task.task_id,
        user_goal="Reboot production server",
        step_runner=approval_runner,
    )

    assert result.status == "waiting"
    assert result.goal_satisfied is False
    assert "approval" in result.answer

    updated = supervisor.inspect_task(task.task_id)
    assert updated.state == ExecutionState.WAITING
    assert updated.waiting_reason == "waiting_for_approval"


def test_independent_completion_verification_enforcement(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    contract = CompletionContract(
        contract_type=CompletionContractType.FILE_EXISTS,
        path="output.txt",
    )
    task = supervisor.create_task(
        session_id="s1",
        requested_operation="create_output",
        completion_contract=[contract],
        workspace_path=str(tmp_path),
    )
    task, token = supervisor.acquire_lease(task.task_id, owner="worker1")
    supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)

    # 1. Model tries to FINALIZE, but output.txt does not exist on disk!
    # Controller must reject FINALIZE and switch to REPLAN.
    # 2. Next turn creates output.txt on disk and model finalizes.
    llm = FakeLLM(
        [
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.FINALIZE,
                goal_satisfied=True,
                progress_detected=True,
                evidence_summary="tried to claim output.txt exists",
                safe_to_continue=True,
                reason="claiming done",
            ),
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.FINALIZE,
                goal_satisfied=True,
                progress_detected=True,
                evidence_summary="output.txt created on disk",
                user_message="output.txt successfully created.",
                safe_to_continue=True,
                reason="verified on disk",
            ),
        ]
    )
    evaluator = ContinuationEvaluator(llm)
    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(require_verification=True),
        supervisor=supervisor,
        evaluator=evaluator,
    )

    def file_creation_runner(step_idx: int, ctx: dict[str, Any]) -> dict[str, Any]:
        if step_idx == 0:
            # File NOT created yet
            return {"answer": "I wrote the file"}
        # Actually create file on disk
        (tmp_path / "output.txt").write_text("Hello World", encoding="utf-8")
        return {"answer": "Created output.txt for real", "changed_files": ["output.txt"]}

    result = controller.execute_goal(
        task_id=task.task_id,
        user_goal="Create output.txt",
        step_runner=file_creation_runner,
    )

    assert result.status == "completed"
    assert result.goal_satisfied is True
    assert result.verification_passed is True
    assert (tmp_path / "output.txt").exists()
    assert supervisor.inspect_task(task.task_id).state == ExecutionState.COMPLETED
