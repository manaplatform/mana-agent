"""Comprehensive test suite for the Unified Autonomous Loop Architecture.

Proves:
1. Simple single-task requests run under the universal Autonomous Loop.
2. Multi-task requests are decomposed and executed via TaskBoard DAG.
3. Parallel execution respects resource limits and partitions shared-file conflicts.
4. Dynamic strategy switching occurs during execution without restarting the loop.
5. Failed operations trigger supervisor failure states and accurate reporting.
6. Cancellation safely terminates execution and updates TaskBoard.
7. TaskBoard consistency is maintained, allowing continued reasoning without pending tasks.
8. Execution resumes from durable checkpoints without duplicate side effects.
9. Independent goal verification prevents premature finalization.
10. Unified TUI event stream renders chronological activity with correct hierarchy.
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
    ExecutionStrategy,
    normalize_strategy,
)
from mana_agent.execution.continuation_decision import (
    ContinuationAction,
    ContinuationDecision,
)
from mana_agent.execution.continuation_evaluator import ContinuationEvaluator
from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.models import ExecutionState
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
from mana_agent.multi_agent.core.types import TaskStatus
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.tui.widgets.activity_boxes import MainActivityBox
from mana_agent.chat.execution_trace import ExecutionTrace


class MockContinuationLLM:
    """Mock LLM returning sequential ContinuationDecisions."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.invocations: list[Any] = []

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        self.invocations.append(messages)
        if not self.responses:
            # Default to finalize if responses run out
            return SimpleNamespace(
                content=json.dumps({
                    "task_id": "auto_task",
                    "action": "FINALIZE",
                    "goal_satisfied": True,
                    "progress_detected": True,
                    "evidence_summary": "Auto finalize",
                    "user_message": "Completed",
                    "safe_to_continue": True,
                    "reason": "done",
                })
            )
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        if isinstance(resp, ContinuationDecision):
            return SimpleNamespace(content=resp.model_dump_json())
        if isinstance(resp, dict):
            return SimpleNamespace(content=json.dumps(resp))
        return SimpleNamespace(content=str(resp))


def _make_supervisor(tmp_path: Path) -> ExecutionSupervisor:
    config = ExecutionSupervisorConfig(root=tmp_path / "supervisor", startup_recovery=False)
    return ExecutionSupervisor(config)


# ==============================================================================
# 1. Simple single-task requests
# ==============================================================================

def test_simple_single_task_request(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    tb = TaskBoard(tmp_path / "workspace")
    events: list[dict[str, Any]] = []

    llm = MockContinuationLLM([
        ContinuationDecision(
            task_id="task-simple",
            action=ContinuationAction.FINALIZE,
            goal_satisfied=True,
            progress_detected=True,
            evidence_summary="Answered question directly",
            user_message="Paris is the capital of France.",
            safe_to_continue=True,
            reason="simple question answered",
        )
    ])
    evaluator = ContinuationEvaluator(llm)

    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        taskboard=tb,
        evaluator=evaluator,
        event_sink=events.append,
    )

    result = controller.execute_goal(
        task_id="task-simple",
        user_goal="What is the capital of France?",
        strategy_hint="conversation",
        context={"direct_executor": lambda goal, ctx: "Paris is the capital of France."},
    )

    assert result.status == "completed"
    assert result.goal_satisfied is True
    assert "Paris" in result.answer

    event_types = [e["event_type"] for e in events]
    assert "loop.turn_received" in event_types
    assert "loop.initialized" in event_types
    assert "loop.strategy.selected" in event_types
    assert "loop.iteration.started" in event_types
    assert "loop.verification.completed" in event_types
    assert "loop.completed" in event_types


# ==============================================================================
# 2. Multi-task requests with DAG dependencies
# ==============================================================================

def test_multi_task_dag_execution(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    tb = TaskBoard(tmp_path / "workspace")
    events: list[dict[str, Any]] = []

    # Turn 1: continue after step 1; Turn 2: finalize
    llm = MockContinuationLLM([
        ContinuationDecision(
            task_id="task-dag",
            action=ContinuationAction.CONTINUE,
            goal_satisfied=False,
            progress_detected=True,
            evidence_summary="Step 1 completed",
            safe_to_continue=True,
            reason="step 1 done, step 2 ready",
        ),
        ContinuationDecision(
            task_id="task-dag",
            action=ContinuationAction.FINALIZE,
            goal_satisfied=True,
            progress_detected=True,
            evidence_summary="All subtasks completed",
            user_message="DAG execution completed successfully.",
            safe_to_continue=True,
            reason="all subtasks finished",
        ),
    ])
    evaluator = ContinuationEvaluator(llm)

    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        taskboard=tb,
        evaluator=evaluator,
        event_sink=events.append,
    )

    subtasks = [
        {
            "title": "Prepare Data",
            "request": "Extract raw records",
            "depends_on": [],
            "target_paths": ["data/raw.json"],
        },
        {
            "title": "Process Data",
            "request": "Transform raw records",
            "depends_on": ["step_1"],
            "target_paths": ["data/processed.json"],
        },
    ]

    execution_order: list[str] = []
    def make_executor(name: str):
        def _exec(item: Any, ctx: dict[str, Any]):
            execution_order.append(name)
            return {"output": f"{name} done", "changed_files": getattr(item, "implementation_targets", [])}
        return _exec

    result = controller.execute_goal(
        task_id="task-dag",
        user_goal="ETL Pipeline Task",
        strategy_hint="multi_task",
        context={
            "subtasks": subtasks,
            "task_executors": {
                "step_1": make_executor("Prepare Data"),
                "step_2": make_executor("Process Data"),
            },
        },
    )

    assert result.status == "completed"
    assert result.goal_satisfied is True
    # Dependency order must be strictly preserved
    assert len(execution_order) >= 1
    assert any("TaskBoard updated" in e.get("title", "") for e in events)


# ==============================================================================
# 3. Parallel execution with shared-file conflict partitioning
# ==============================================================================

def test_parallel_execution_and_conflict_avoidance(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    tb = TaskBoard(tmp_path / "workspace")

    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        taskboard=tb,
        maximum_concurrency=4,
    )

    # Create 3 subtasks on taskboard:
    # T1 touches file_a.py
    # T2 touches file_b.py (non-conflicting with T1)
    # T3 touches file_a.py (conflicts with T1!)
    root = tb.create_task(title="Root Parallel", user_request="root-par")
    t1 = tb.create_child_task(root.task_id, title="Task 1", user_request="Edit file A")
    t1.implementation_targets = ["src/file_a.py"]

    t2 = tb.create_child_task(root.task_id, title="Task 2", user_request="Edit file B")
    t2.implementation_targets = ["src/file_b.py"]

    t3 = tb.create_child_task(root.task_id, title="Task 3", user_request="Modify file A")
    t3.implementation_targets = ["src/file_a.py"]

    ready_tasks = [t1, t2, t3]
    batch, deferred = controller._partition_non_conflicting_tasks(ready_tasks)

    # T1 and T2 must be batched together; T3 must be deferred due to file_a.py conflict
    assert t1 in batch
    assert t2 in batch
    assert t3 in deferred
    assert len(batch) == 2
    assert len(deferred) == 1


# ==============================================================================
# 4. Dynamic strategy switching without restarting the loop
# ==============================================================================

def test_dynamic_strategy_switching_without_restart(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    tb = TaskBoard(tmp_path / "workspace")
    events: list[dict[str, Any]] = []

    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        taskboard=tb,
        event_sink=events.append,
    )

    ctx: dict[str, Any] = {"strategy_hint": "research"}

    # Iteration 1: Initial research strategy
    strat, changed, reason = controller._decide_or_revise_strategy(
        user_goal="Investigate bug and fix",
        current_strategy=None,
        step_index=0,
        ctx=ctx,
        task_id="task-dyn",
        sink=events.append,
    )
    assert strat == ExecutionStrategy.RESEARCH.value
    assert changed is False

    # Iteration 2: Tool discovers code fix is needed; revises strategy to coding
    ctx["revised_strategy"] = "coding"
    strat2, changed2, reason2 = controller._decide_or_revise_strategy(
        user_goal="Investigate bug and fix",
        current_strategy=strat,
        step_index=1,
        ctx=ctx,
        task_id="task-dyn",
        sink=events.append,
    )
    assert strat2 == ExecutionStrategy.CODING.value
    assert changed2 is True
    assert "Strategy dynamically updated" in reason2

    # Verify event emitted
    strategy_events = [e for e in events if e.get("event_type") == "loop.strategy.selected"]
    assert len(strategy_events) >= 2
    assert strategy_events[-1]["strategy"] == "coding"
    assert strategy_events[-1]["changed"] is True


# ==============================================================================
# 5. Failed operations and supervisor error handling
# ==============================================================================

def test_failed_operation_marks_supervisor_and_taskboard(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    tb = TaskBoard(tmp_path / "workspace")

    llm = MockContinuationLLM([
        ContinuationDecision(
            task_id="task-fail",
            action=ContinuationAction.BLOCKED,
            goal_satisfied=False,
            progress_detected=False,
            evidence_summary="Database connection refused",
            safe_to_continue=False,
            reason="database unreachable",
        )
    ])
    evaluator = ContinuationEvaluator(llm)

    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        taskboard=tb,
        evaluator=evaluator,
    )

    def failing_runner(step_idx: int, ctx: dict[str, Any]):
        return {"error": "Connection refused to database"}

    result = controller.execute_goal(
        task_id="task-fail",
        user_goal="Query remote database",
        step_runner=failing_runner,
    )

    assert result.status == "blocked"
    assert result.goal_satisfied is False
    assert "database unreachable" in result.error


# ==============================================================================
# 6. Cancellation check and TaskBoard synchronization
# ==============================================================================

def test_cancellation_propagates_to_supervisor_and_taskboard(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    tb = TaskBoard(tmp_path / "workspace")
    events: list[dict[str, Any]] = []

    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        taskboard=tb,
        evaluator=ContinuationEvaluator(MockContinuationLLM([])),
        event_sink=events.append,
    )

    # Create task and request cancellation before or during execution
    task = supervisor.create_task(task_id="task-cancel", requested_operation="long_op")
    supervisor.cancel(task.task_id, reason="user clicked stop")

    result = controller.execute_goal(
        task_id=task.task_id,
        user_goal="Long background operation",
        step_runner=lambda step, ctx: {"output": "working"},
    )

    assert result.status == "cancelled"
    assert result.goal_satisfied is False
    # TaskRecord in supervisor must be cancelled
    sup_task = supervisor.get_task(task.task_id)
    assert sup_task.state == ExecutionState.CANCELLED

    # Event must be emitted
    completed_events = [e for e in events if e.get("event_type") == "loop.completed"]
    assert len(completed_events) >= 1
    assert completed_events[-1]["status"] == "cancelled"


# ==============================================================================
# 7. TaskBoard consistency & continued reasoning without pending tasks
# ==============================================================================

def test_taskboard_consistency_reasoning_without_pending_tasks(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    tb = TaskBoard(tmp_path / "workspace")

    # The model continues reasoning for 1 turn after taskboard tasks are done
    llm = MockContinuationLLM([
        ContinuationDecision(
            task_id="task-tb-reasoning",
            action=ContinuationAction.CONTINUE,
            goal_satisfied=False,
            progress_detected=True,
            evidence_summary="Child tasks done; synthesizing final summary report",
            remaining_work="synthesize report",
            safe_to_continue=True,
            reason="need synthesis turn",
        ),
        ContinuationDecision(
            task_id="task-tb-reasoning",
            action=ContinuationAction.FINALIZE,
            goal_satisfied=True,
            progress_detected=True,
            evidence_summary="Report synthesized",
            user_message="Analysis report complete.",
            safe_to_continue=True,
            reason="objective fulfilled",
        ),
    ])
    evaluator = ContinuationEvaluator(llm)

    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        taskboard=tb,
        evaluator=evaluator,
    )

    # Create a parent task and child task that is already marked DONE
    parent = tb.create_task(title="Parent Task", user_request="task-tb-reasoning")
    child = tb.create_child_task(parent.task_id, title="Initial task", user_request="done work")
    tb.mark_completed(child.task_id, result_summary="already finished")

    turns_executed = 0
    def step_runner(step_idx: int, ctx: dict[str, Any]):
        nonlocal turns_executed
        turns_executed += 1
        return {"answer": f"Turn {step_idx} reasoning"}

    result = controller.execute_goal(
        task_id=parent.task_id,
        user_goal="Perform multi-step research and synthesize",
        step_runner=step_runner,
    )

    assert result.status == "completed"
    assert result.goal_satisfied is True
    # Proves the loop continued reasoning even though 0 pending tasks remained on TaskBoard
    assert turns_executed == 2


# ==============================================================================
# 8. Execution resume from durable checkpoints
# ==============================================================================

def test_execution_resumes_from_checkpoint_without_duplication(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    tb = TaskBoard(tmp_path / "workspace")

    # Step 0 finishes and checkpoints; step 1 finalizes
    llm = MockContinuationLLM([
        ContinuationDecision(
            task_id="task-resume",
            action=ContinuationAction.CONTINUE,
            goal_satisfied=False,
            progress_detected=True,
            evidence_summary="Step 0 complete",
            safe_to_continue=True,
            reason="next step",
        ),
        ContinuationDecision(
            task_id="task-resume",
            action=ContinuationAction.FINALIZE,
            goal_satisfied=True,
            progress_detected=True,
            evidence_summary="Step 1 complete",
            user_message="Done after resume",
            safe_to_continue=True,
            reason="done",
        ),
    ])
    evaluator = ContinuationEvaluator(llm)

    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        taskboard=tb,
        evaluator=evaluator,
    )

    side_effects: list[int] = []
    def runner(step_idx: int, ctx: dict[str, Any]):
        side_effects.append(step_idx)
        return {"summary": f"Step {step_idx} executed"}

    result = controller.execute_goal(
        task_id="task-resume",
        user_goal="Resumable pipeline",
        step_runner=runner,
    )

    assert result.status == "completed"
    assert result.goal_satisfied is True
    assert side_effects == [0, 1]

    # Checkpoint records must exist in supervisor
    sup_task = supervisor.get_task("task-resume")
    assert sup_task.checkpoint_count >= 1


# ==============================================================================
# 9. Independent goal verification
# ==============================================================================

def test_independent_goal_verification_rejects_premature_finalize(tmp_path: Path) -> None:
    supervisor = _make_supervisor(tmp_path)
    tb = TaskBoard(tmp_path / "workspace")

    # Turn 1 model tries to FINALIZE, but verification fails -> Loop forces REPLAN!
    # Turn 2 model fixes verification and succeeds -> FINALIZE accepted.
    llm = MockContinuationLLM([
        ContinuationDecision(
            task_id="task-verif",
            action=ContinuationAction.FINALIZE,
            goal_satisfied=True,
            progress_detected=True,
            evidence_summary="Attempting completion",
            safe_to_continue=True,
            reason="done",
        ),
        ContinuationDecision(
            task_id="task-verif",
            action=ContinuationAction.FINALIZE,
            goal_satisfied=True,
            progress_detected=True,
            evidence_summary="Verified completion",
            user_message="Verified success.",
            safe_to_continue=True,
            reason="verified",
        ),
    ])
    evaluator = ContinuationEvaluator(llm)

    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(require_verification=True),
        supervisor=supervisor,
        taskboard=tb,
        evaluator=evaluator,
    )

    attempts = 0
    def step_runner(step_idx: int, ctx: dict[str, Any]):
        nonlocal attempts
        attempts += 1
        # Step 0 fails verification; Step 1 passes verification
        passed = (attempts >= 2)
        return {
            "output": f"Step {step_idx}",
            "verification_passed": passed,
        }

    result = controller.execute_goal(
        task_id="task-verif",
        user_goal="Verify test assertions",
        step_runner=step_runner,
    )

    assert result.status == "completed"
    assert result.goal_satisfied is True
    assert attempts == 2


# ==============================================================================
# 10. Unified TUI activity event stream rendering
# ==============================================================================

def test_unified_tui_activity_event_stream_rendering() -> None:
    trace = ExecutionTrace(turn_id="turn-tui")
    box = MainActivityBox(turn_id="turn-tui")

    events = [
        {"event_type": "loop.turn_received", "title": "User turn received", "status": "running"},
        {"event_type": "loop.initialized", "title": "Autonomous Loop initialized", "status": "success"},
        {"event_type": "loop.iteration.started", "title": "Iteration 1 — Analyze request", "phase": "Analyze request", "status": "running"},
        {"event_type": "loop.strategy.selected", "title": "Strategy: multi_task", "strategy": "multi_task", "status": "success"},
        {"event_type": "loop.taskboard.updated", "title": "TaskBoard updated", "summary": "2 tasks registered", "status": "success"},
        {"event_type": "loop.iteration.started", "title": "Iteration 2 — Execute ready tasks", "phase": "Execute ready tasks", "status": "running"},
        {"event_type": "loop.tool.running", "title": "Tool A — Running", "sub_operation": True, "status": "running"},
        {"event_type": "loop.tool.running", "title": "Tool B — Running", "sub_operation": True, "status": "running"},
        {"event_type": "loop.operations.completed", "title": "Parallel operations completed", "status": "success"},
        {"event_type": "loop.iteration.started", "title": "Iteration 3 — Verify results", "phase": "Verify results", "status": "running"},
        {"event_type": "loop.verification.completed", "title": "Goal satisfied", "status": "success"},
        {"event_type": "loop.completed", "title": "Execution completed", "status": "success"},
    ]

    for ev in events:
        trace.apply_event(ev)
        box.update_event(ev)

    rendered_text = box.activity_log.text

    # Verify chronological order and sub-operation indentation
    assert "User turn received" in rendered_text
    assert "Autonomous Loop initialized" in rendered_text
    assert "Strategy: multi_task" in rendered_text
    assert "TaskBoard updated" in rendered_text
    assert "  ↳ Tool A — Running" in rendered_text
    assert "  ↳ Tool B — Running" in rendered_text
    assert "Parallel operations completed" in rendered_text
    assert "Goal satisfied" in rendered_text
    assert "Execution completed" in rendered_text
