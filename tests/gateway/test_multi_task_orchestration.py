from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mana_agent.multi_agent.core.types import TaskStatus
from mana_agent.multi_agent.runtime.multi_task_orchestrator import (
    MultiTaskChildResult,
    MultiTaskError,
    MultiTaskItem,
    MultiTaskOrchestrator,
    MultiTaskPlan,
)
from mana_agent.multi_agent.taskboard.store import task_from_dict
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard


def _item(local_id: str, *, depends_on: list[str] | None = None) -> MultiTaskItem:
    return MultiTaskItem(
        local_id=local_id,
        title=f"Task {local_id}",
        request=f"Execute {local_id}",
        depends_on=depends_on or [],
        acceptance_criteria=[f"{local_id} is complete"],
        preferred_parallelism="automatic",
        reason=f"{local_id} has its own lifecycle",
    )


def _plan(*tasks: MultiTaskItem) -> MultiTaskPlan:
    return MultiTaskPlan(
        goal="Complete the compound request",
        tasks=list(tasks),
        final_acceptance_criteria=["Every runnable child is reported truthfully"],
        reason="The goals require separate execution lifecycles",
    )


def _root(board: TaskBoard):
    root = board.create_task(
        title="Compound request",
        user_request="Do A and B",
        session_id="session-1",
        workspace_id="workspace-1",
        repository_ids=["repository-1"],
        primary_repository_id="repository-1",
    )
    board.update_status(root.task_id, TaskStatus.PLANNING)
    return root


@pytest.mark.parametrize(
    "tasks,match",
    [
        ([_item("a"), _item("a")], "local IDs"),
        ([_item("a", depends_on=["missing"]), _item("b")], "unknown dependencies"),
        ([_item("a", depends_on=["b"]), _item("b", depends_on=["a"])], "cycle"),
    ],
)
def test_invalid_decomposition_graph_fails_before_execution(tasks, match) -> None:
    with pytest.raises(ValidationError, match=match):
        _plan(*tasks)


def test_invalid_model_decomposition_raises_typed_error_without_fallback(tmp_path: Path) -> None:
    board = TaskBoard(tmp_path)
    model = SimpleNamespace(invoke=lambda _messages: SimpleNamespace(content='{"goal":"only one"}'))

    with pytest.raises(MultiTaskError, match="No fallback action was executed"):
        MultiTaskOrchestrator(llm=model, taskboard=board).decompose(
            user_prompt="Do A and B", context={}
        )


def test_children_preserve_lineage_context_dependencies_and_serialization(tmp_path: Path) -> None:
    board = TaskBoard(tmp_path)
    root = _root(board)
    plan = _plan(_item("a"), _item("b", depends_on=["a"]))
    orchestrator = MultiTaskOrchestrator(llm=object(), taskboard=board)

    mapping = orchestrator.create_children(root_task_id=root.task_id, plan=plan)
    first = board.get_task(mapping["a"])
    second = board.get_task(mapping["b"])

    assert first.parent_task_id == root.task_id
    assert first.root_task_id == root.task_id
    assert first.session_id == root.session_id == "session-1"
    assert first.workspace_id == root.workspace_id == "workspace-1"
    assert first.repository_ids == root.repository_ids == ["repository-1"]
    assert second.depends_on == [first.task_id]
    assert root.decomposition_id_map == mapping
    assert root.child_task_ids == [first.task_id, second.task_id]

    reloaded = TaskBoard(tmp_path)
    assert reloaded.get_task(second.task_id).depends_on == [first.task_id]
    legacy = task_from_dict({
        "task_id": "legacy", "parent_task_id": None, "root_task_id": "legacy",
        "title": "Legacy", "user_request": "legacy", "normalized_goal": "legacy",
        "status": "new", "priority": 100, "risk_level": "low",
    })
    assert legacy.depends_on == []
    assert legacy.decomposition_id_map == {}


def test_independent_children_execute_concurrently_with_bounded_workers(tmp_path: Path) -> None:
    board = TaskBoard(tmp_path)
    root = _root(board)
    plan = _plan(_item("a"), _item("b"), _item("c"))
    orchestrator = MultiTaskOrchestrator(llm=object(), taskboard=board, maximum_concurrency=2)
    active = 0
    peak = 0
    lock = threading.Lock()

    def execute(item: MultiTaskItem, task_id: str) -> MultiTaskChildResult:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return MultiTaskChildResult(item.local_id, task_id, item.title, "repository", "completed")

    results = orchestrator.execute(root_task_id=root.task_id, plan=plan, execute_child=execute)

    assert peak == 2
    assert [item.status for item in results] == ["completed", "completed", "completed"]
    assert board.get_task(root.task_id).aggregate_progress == "3/3 completed"


def test_dependencies_run_in_order_and_failed_prerequisite_blocks_child(tmp_path: Path) -> None:
    board = TaskBoard(tmp_path)
    root = _root(board)
    plan = _plan(_item("first"), _item("second", depends_on=["first"]), _item("independent"))
    orchestrator = MultiTaskOrchestrator(llm=object(), taskboard=board, maximum_concurrency=3)
    calls: list[str] = []

    def execute(item: MultiTaskItem, task_id: str) -> MultiTaskChildResult:
        calls.append(item.local_id)
        if item.local_id == "first":
            return MultiTaskChildResult(item.local_id, task_id, item.title, "github", "failed", blocker="boom")
        return MultiTaskChildResult(item.local_id, task_id, item.title, "repository", "completed")

    results = orchestrator.execute(root_task_id=root.task_id, plan=plan, execute_child=execute)
    by_id = {item.local_id: item for item in results}

    assert "second" not in calls
    assert by_id["second"].status == "blocked"
    assert "first" in by_id["second"].blocker
    assert by_id["independent"].status == "completed"


def test_approval_wait_keeps_dependents_queued_for_safe_resume(tmp_path: Path) -> None:
    board = TaskBoard(tmp_path)
    root = _root(board)
    plan = _plan(_item("approval"), _item("dependent", depends_on=["approval"]))
    orchestrator = MultiTaskOrchestrator(llm=object(), taskboard=board)
    calls: list[str] = []

    def execute(item: MultiTaskItem, task_id: str) -> MultiTaskChildResult:
        calls.append(item.local_id)
        return MultiTaskChildResult(
            item.local_id, task_id, item.title, "computer", "awaiting_approval",
            approval_request_ids=["approval-1"],
        )

    results = orchestrator.execute(root_task_id=root.task_id, plan=plan, execute_child=execute)
    by_id = {item.local_id: item for item in results}

    assert calls == ["approval"]
    assert by_id["dependent"].status == "queued"
    assert board.get_task(by_id["dependent"].task_id).status == TaskStatus.QUEUED


def test_resume_materialization_does_not_duplicate_persisted_children(tmp_path: Path) -> None:
    board = TaskBoard(tmp_path)
    root = _root(board)
    plan = _plan(_item("a"), _item("b"))
    orchestrator = MultiTaskOrchestrator(llm=object(), taskboard=board)

    first = orchestrator.create_children(root_task_id=root.task_id, plan=plan)
    second = orchestrator.create_children(root_task_id=root.task_id, plan=plan)

    assert first == second
    assert len(board.get_task(root.task_id).child_task_ids) == 2


def test_resume_execution_does_not_rerun_completed_children(tmp_path: Path) -> None:
    board = TaskBoard(tmp_path)
    root = _root(board)
    plan = _plan(_item("done"), _item("unfinished", depends_on=["done"]))
    orchestrator = MultiTaskOrchestrator(llm=object(), taskboard=board)
    mapping = orchestrator.create_children(root_task_id=root.task_id, plan=plan)
    board.update_status(mapping["done"], TaskStatus.ROUTED)
    board.update_status(mapping["done"], TaskStatus.IN_PROGRESS)
    board.update_orchestration(mapping["done"], entry_route="repository", result_summary="already done")
    board.project_supervisor_completion(
        mapping["done"],
        supervisor_task=SimpleNamespace(
            task_id="supervisor-done",
            state=SimpleNamespace(value="completed"),
            verification_status=SimpleNamespace(value="passed"),
            state_version=1,
        ),
        verification_evidence={"verification": "passed", "result_id": "result-done"},
    )
    calls: list[str] = []

    def execute(item: MultiTaskItem, task_id: str) -> MultiTaskChildResult:
        calls.append(item.local_id)
        return MultiTaskChildResult(item.local_id, task_id, item.title, "coding", "completed")

    results = orchestrator.execute(root_task_id=root.task_id, plan=plan, execute_child=execute)

    assert calls == ["unfinished"]
    assert results[0].status == "completed"
    assert results[0].result == "already done"


def test_multi_task_capacity_estimate_ignores_depleted_session_remaining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compound children must size against model capacity, not parent planning spend."""
    from decimal import Decimal
    from types import SimpleNamespace

    from mana_agent.context_cost.accounting import (
        ModelContextLimitError,
        ModelTokenAccountingService,
        TokenEstimationRequest,
    )
    from mana_agent.context_cost.profiles import (
        ModelIdentity,
        ModelTokenProfile,
        ModelTokenProfileResolver,
    )
    from mana_agent.context_cost.store import AccountingStore
    from mana_agent.gateway.chat_gateway import AgentChatGateway
    from mana_agent.gateway.lanes import LaneId, default_lane_contracts

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    profile = ModelTokenProfile(
        ModelIdentity("fixture", "capacity-model"),
        context_window=8_192,
        max_output_tokens=1_024,
        input_price_per_million=Decimal("1"),
        output_price_per_million=Decimal("2"),
        supports_usage_reporting=True,
        source="test",
        confidence="high",
    )
    accounting = ModelTokenAccountingService(
        ModelTokenProfileResolver((profile,)),
        store=AccountingStore(tmp_path / "accounting"),
        safety_margin_ratio=Decimal("0.05"),
    )
    gateway = object.__new__(AgentChatGateway)
    gateway._lane_coordinator = SimpleNamespace(
        select_lane=lambda **_kwargs: LaneId.RESEARCH,
        contracts=default_lane_contracts(),
    )
    gateway._stack = SimpleNamespace(
        context_cost_governor=SimpleNamespace(accounting=accounting)
    )

    # Shared session remaining is intentionally too small for a normal estimate.
    with pytest.raises(ModelContextLimitError, match="effective limit is 50"):
        accounting.estimate(
            TokenEstimationRequest(
                model_identity=ModelIdentity("fixture", "capacity-model"),
                components={"user_request": "create assets " * 20},
                requested_output_tokens=256,
                task_token_remaining=50,
                session_token_remaining=50,
            )
        )

    estimate = gateway._multi_task_capacity_estimate(
        provider="fixture",
        model="capacity-model",
        request_text="create assets " * 20,
        entry_route="coding",
        expected_model_calls=1,
        requested_output_tokens=256,
    )
    assert estimate.total_tokens > 50
    assert estimate.profile.context_window == 8_192


def test_multi_task_parent_envelope_expands_for_parallel_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Root multi-task reservation must grow so sibling children can both reserve."""
    from mana_agent.gateway.chat_gateway import AgentChatGateway
    from mana_agent.gateway.lane_coordinator import LaneCoordinator
    from mana_agent.gateway.lanes import LaneId

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    root_path = tmp_path / "repo"
    root_path.mkdir()
    coordinator = LaneCoordinator(root_path)
    gateway = object.__new__(AgentChatGateway)
    gateway._lane_coordinator = coordinator
    gateway._multi_task_budget_lock = threading.Lock()

    board = coordinator.taskboard
    root_task = board.create_task(title="Compound", user_request="Do A and B")
    first_task = board.create_child_task(
        root_task.task_id,
        title="A",
        user_request="Do A",
        decomposition_local_id="a",
        acceptance_criteria=["A done"],
    )
    second_task = board.create_child_task(
        root_task.task_id,
        title="B",
        user_request="Do B",
        decomposition_local_id="b",
        acceptance_criteria=["B done"],
    )

    # Intentionally under-size the root the way the pre-fix path did (goal-only).
    root = coordinator.reserve(
        normalized_intent="Do A and B",
        lane_id=LaneId.RESEARCH,
        session_id="session-multi",
        workspace_id=board.store.workspace_id,
        repository_id=board.store.repository_id,
        requested_input_tokens=64,
        requested_output_tokens=100,
        task_type="multi_task_root",
        taskboard_task_id=root_task.task_id,
    )
    coordinator.start(root)

    with gateway._multi_task_budget_lock:
        gateway._ensure_multi_task_parent_budget(
            root.execution.task_id,
            child_input_tokens=200,
            child_output_tokens=300,
            child_estimated_cost=None,
        )
        first = coordinator.reserve(
            normalized_intent="Do A",
            lane_id=LaneId.CODING,
            session_id="session-multi",
            workspace_id=board.store.workspace_id,
            repository_id=board.store.repository_id,
            parent_task_id=root.execution.task_id,
            root_task_id=root.execution.root_task_id,
            requested_input_tokens=200,
            requested_output_tokens=300,
            task_type="multi_task_child",
            taskboard_task_id=first_task.task_id,
        )
        gateway._ensure_multi_task_parent_budget(
            root.execution.task_id,
            child_input_tokens=250,
            child_output_tokens=350,
            child_estimated_cost=None,
        )
        second = coordinator.reserve(
            normalized_intent="Do B",
            lane_id=LaneId.MEDIA,
            session_id="session-multi",
            workspace_id=board.store.workspace_id,
            repository_id=board.store.repository_id,
            parent_task_id=root.execution.task_id,
            root_task_id=root.execution.root_task_id,
            requested_input_tokens=250,
            requested_output_tokens=350,
            task_type="multi_task_child",
            taskboard_task_id=second_task.task_id,
        )

    parent = coordinator.inspect_task(root.execution.task_id)
    assert first.execution.budget.reserved_tokens == 500
    assert second.execution.budget.reserved_tokens == 600
    assert parent.budget.reserved_tokens >= 500 + 600
    assert parent.budget.revisions
    assert parent.budget.revisions[-1]["reason"] == "multi-task child budget envelope"
