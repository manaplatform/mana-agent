"""Tests for Accounting-Based Auto-Renewal and Budget Management in Autonomous Execution."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any
import pytest

from mana_agent.config.autonomous_config import AutoRenewConfig, AutonomousExecutionConfig
from mana_agent.context_cost.accounting import ModelTokenAccountingService
from mana_agent.context_cost.profiles import ModelIdentity, ModelTokenProfile, ModelTokenProfileResolver
from mana_agent.context_cost.store import AccountingStore
from mana_agent.execution.autonomous_controller import (
    AutonomousExecutionController,
    TaskExecutionMetrics,
)
from mana_agent.execution.continuation_decision import (
    ContinuationAction,
    ContinuationDecision,
)
from mana_agent.execution.continuation_evaluator import ContinuationEvaluator
from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.models import ExecutionState
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor


class FakeLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        if not self.responses:
            raise RuntimeError("No more responses in FakeLLM")
        return self.responses.pop(0)


def _make_supervisor(tmp_path: Path) -> ExecutionSupervisor:
    config = ExecutionSupervisorConfig(root=tmp_path / "supervisor", startup_recovery=False)
    return ExecutionSupervisor(config)


def _make_accounting(tmp_path: Path) -> ModelTokenAccountingService:
    profile = ModelTokenProfile(
        identity=ModelIdentity("openai", "gpt-4.1-mini"),
        context_window=128000,
        max_output_tokens=16384,
        input_price_per_million=Decimal("0.15"),
        output_price_per_million=Decimal("0.60"),
        tokenizer="cl100k_base",
        supports_usage_reporting=True,
        source="test",
        confidence="high",
    )
    return ModelTokenAccountingService(
        ModelTokenProfileResolver([profile]),
        store=AccountingStore(tmp_path / "accounting"),
        safety_margin_ratio=Decimal("0.05"),
    )


def test_auto_renew_granted_when_soft_limit_reached(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    accounting = _make_accounting(tmp_path)

    task = supervisor.create_task(
        session_id="s1",
        requested_operation="long_running_task",
        token_budget=50000,
    )
    task, token = supervisor.acquire_lease(task.task_id, owner="worker1")
    supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)

    # Initial limit: 2 model turns. Soft headroom: 0.
    # Turn 0: step 0 -> continue
    # Turn 1: step 1 -> soft limit reached -> auto-renew triggers -> granted!
    # Turn 2: step 2 -> finalize!
    llm = FakeLLM(
        [
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.CONTINUE,
                goal_satisfied=False,
                progress_detected=True,
                evidence_summary="part 1 done",
                remaining_work="part 2",
                safe_to_continue=True,
                reason="continue",
            ),
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.CONTINUE,
                goal_satisfied=False,
                progress_detected=True,
                evidence_summary="part 2 done",
                remaining_work="part 3",
                safe_to_continue=True,
                renewal_requested=True,
                renewal_reason="need 1 more turn for part 3",
                reason="renew needed",
            ),
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.FINALIZE,
                goal_satisfied=True,
                progress_detected=True,
                evidence_summary="part 3 done",
                user_message="All parts completed.",
                safe_to_continue=True,
                reason="all done",
            ),
        ]
    )
    evaluator = ContinuationEvaluator(llm)

    config = AutonomousExecutionConfig(
        max_model_turns=2,
        soft_limit_turns_headroom=0,
        soft_limit_tools_headroom=0,
        auto_renew=AutoRenewConfig(
            enabled=True,
            renew_model_turns=2,
            renew_tool_calls=5,
            max_auto_renewals=2,
            require_progress=True,
            respect_account_budget=True,
        ),
    )

    controller = AutonomousExecutionController(
        config=config,
        supervisor=supervisor,
        accounting_service=accounting,
        evaluator=evaluator,
    )

    def multi_part_runner(step_idx: int, ctx: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": f"Part {step_idx + 1} completed",
            "tool_calls": [{"name": "compute", "args": {"step": step_idx}}],
            "tokens_used": 100,
            "cost": 0.0005,
        }

    result = controller.execute_goal(
        task_id=task.task_id,
        user_goal="Run 3-part computation",
        step_runner=multi_part_runner,
    )

    assert result.status == "completed"
    assert result.goal_satisfied is True
    assert result.metrics.model_turns == 3
    assert result.metrics.renewals_count == 1

    # Verify that supervisor recorded the budget revision
    updated_task = supervisor.inspect_task(task.task_id)
    assert len(updated_task.budget_revisions) >= 1
    assert updated_task.budget_revisions[-1].reason == "autonomous_execution_auto_renew"


def test_auto_renew_rejected_when_max_renewals_reached(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    accounting = _make_accounting(tmp_path)

    task = supervisor.create_task(session_id="s1", requested_operation="infinite_task")
    task, token = supervisor.acquire_lease(task.task_id, owner="worker1")
    supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)

    # Max renewals = 1. Turns limit = 1.
    # Step 0: turn 1 -> renews (renewals_count becomes 1, limit becomes 2)
    # Step 1: turn 2 -> hits limit again -> renewal rejected -> budget_exhausted!
    llm = FakeLLM(
        [
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.CONTINUE,
                goal_satisfied=False,
                progress_detected=True,
                evidence_summary="step 1",
                safe_to_continue=True,
                reason="continue",
            ),
        ]
    )
    evaluator = ContinuationEvaluator(llm)

    config = AutonomousExecutionConfig(
        max_model_turns=1,
        soft_limit_turns_headroom=0,
        soft_limit_tools_headroom=0,
        auto_renew=AutoRenewConfig(
            enabled=True,
            renew_model_turns=1,
            max_auto_renewals=1,  # Only 1 renewal allowed
            require_approval_above_budget=False,
        ),
    )

    controller = AutonomousExecutionController(
        config=config,
        supervisor=supervisor,
        accounting_service=accounting,
        evaluator=evaluator,
    )

    def endless_runner(step_idx: int, ctx: dict[str, Any]) -> dict[str, Any]:
        return {"answer": f"step {step_idx}", "tool_calls": []}

    result = controller.execute_goal(
        task_id=task.task_id,
        user_goal="Run indefinitely",
        step_runner=endless_runner,
    )

    assert result.status == "budget_exhausted"
    assert result.goal_satisfied is False
    assert result.metrics.renewals_count == 1


def test_auto_renew_rejected_when_no_progress(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    accounting = _make_accounting(tmp_path)

    task = supervisor.create_task(session_id="s1", requested_operation="no_prog_task")
    task, token = supervisor.acquire_lease(task.task_id, owner="worker1")
    supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)

    llm = FakeLLM(
        [
            ContinuationDecision(
                task_id=task.task_id,
                action=ContinuationAction.CONTINUE,
                goal_satisfied=False,
                progress_detected=False,  # No progress
                evidence_summary="stuck",
                safe_to_continue=True,
                reason="stuck",
            )
        ]
    )
    evaluator = ContinuationEvaluator(llm)

    config = AutonomousExecutionConfig(
        max_model_turns=1,
        soft_limit_turns_headroom=0,
        soft_limit_tools_headroom=0,
        auto_renew=AutoRenewConfig(
            enabled=True,
            renew_model_turns=1,
            max_auto_renewals=2,
            require_progress=True,  # Progress is required
            require_approval_above_budget=False,
        ),
    )

    controller = AutonomousExecutionController(
        config=config,
        supervisor=supervisor,
        accounting_service=accounting,
        evaluator=evaluator,
    )

    # Pre-set no progress cycles on initial metrics
    metrics = TaskExecutionMetrics(task_id=task.task_id, root_task_id=task.task_id)
    metrics.consecutive_no_progress_cycles = 1
    metrics.model_turns = 1

    result = controller.execute_goal(
        task_id=task.task_id,
        user_goal="Stuck task",
        step_runner=lambda idx, ctx: {"answer": "stuck"},
        initial_metrics=metrics,
    )

    assert result.status == "budget_exhausted"
    assert "no progress detected" in result.answer


def test_auto_renew_requires_approval_above_budget(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    accounting = _make_accounting(tmp_path)

    # Task with 0 remaining token budget
    task = supervisor.create_task(
        session_id="s1",
        requested_operation="budget_capped_task",
        token_budget=100,
    )
    task, token = supervisor.acquire_lease(task.task_id, owner="worker1")
    supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)

    config = AutonomousExecutionConfig(
        max_model_turns=1,
        soft_limit_turns_headroom=0,
        soft_limit_tools_headroom=0,
        auto_renew=AutoRenewConfig(
            enabled=True,
            renew_model_turns=1,
            max_auto_renewals=2,
            respect_account_budget=True,
            require_approval_above_budget=True,
        ),
    )

    controller = AutonomousExecutionController(
        config=config,
        supervisor=supervisor,
        accounting_service=accounting,
        evaluator=ContinuationEvaluator(FakeLLM([])),
    )

    # Initial metrics with consumed tokens exceeding budget
    metrics = TaskExecutionMetrics(task_id=task.task_id, root_task_id=task.task_id)
    metrics.tokens_consumed = 200
    metrics.model_turns = 1

    result = controller.execute_goal(
        task_id=task.task_id,
        user_goal="Over budget task",
        step_runner=lambda idx, ctx: {"answer": "over budget"},
        initial_metrics=metrics,
    )

    assert result.status == "waiting"
    assert result.error == "waiting_for_approval"
    assert "user approval" in result.answer

    updated = supervisor.inspect_task(task.task_id)
    assert updated.state == ExecutionState.WAITING
    assert updated.waiting_reason == "waiting_for_approval"
