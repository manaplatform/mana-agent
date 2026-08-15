from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from mana_agent.gateway import (
    AgentChatGateway,
    EntryRouteRegistry,
    EntryRouter,
    RouteAvailability,
    RouteRegistration,
)
from mana_agent.gateway.entry_routing import (
    ENTRY_ROUTER_PROMPT,
    EntryRouteContext,
    EntryRoutingDecision,
)
from mana_agent.multi_agent.core.types import TaskStatus
from mana_agent.multi_agent.runtime.multi_task_orchestrator import (
    MULTI_TASK_DECOMPOSITION_PROMPT,
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


def test_worker_threads_inherit_parent_contextvars_for_computer_client(
    tmp_path: Path,
) -> None:
    """Computer route children must see the authenticated parent-turn client.

    Multi-task children run on a ThreadPoolExecutor. Without explicit ContextVar
    propagation, computer_decision_scope fails with:
    'Computer decision scope requires an authenticated client context.'
    """
    from mana_agent.integrations.computer_control.context import (
        computer_client_scope,
        computer_decision_scope,
        current_computer_client,
    )

    board = TaskBoard(tmp_path)
    root = _root(board)
    plan = _plan(_item("directory"), _item("file", depends_on=["directory"]))
    orchestrator = MultiTaskOrchestrator(llm=object(), taskboard=board)
    observed: list[tuple[str, str | None, str | None]] = []

    def execute(item: MultiTaskItem, task_id: str) -> MultiTaskChildResult:
        client = current_computer_client()
        session_id = client.session_id if client is not None else None
        client_type = client.client_type if client is not None else None
        # Mirror the computer route entry: decision scope requires client identity.
        with computer_decision_scope(f"{item.local_id}:computer-entry-decision"):
            scoped = current_computer_client()
            observed.append(
                (
                    item.local_id,
                    session_id,
                    scoped.session_id if scoped is not None else None,
                )
            )
            assert scoped is not None
            assert scoped.client_type == "tui"
            assert f"{item.local_id}:computer-entry-decision" in scoped.allowed_decision_ids
        return MultiTaskChildResult(
            item.local_id, task_id, item.title, "computer", "completed", result="ok"
        )

    with computer_client_scope("session-1", "tui", workspace_root=str(tmp_path)):
        results = orchestrator.execute(
            root_task_id=root.task_id, plan=plan, execute_child=execute
        )
        # Child decision scopes must not leak allowed_decision_ids into the parent.
        parent_client = current_computer_client()
        assert parent_client is not None
        assert parent_client.session_id == "session-1"
        assert parent_client.client_type == "tui"
        assert parent_client.allowed_decision_ids == frozenset()

    assert [item.status for item in results] == ["completed", "completed"]
    assert observed == [
        ("directory", "session-1", "session-1"),
        ("file", "session-1", "session-1"),
    ]


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
    with pytest.raises(ModelContextLimitError, match="remaining task budget is 50"):
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


def test_execution_token_estimate_refreshes_budget_for_followup_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second/third session messages must recalculate admission budget before sizing."""
    from decimal import Decimal
    from types import SimpleNamespace

    from mana_agent.context_cost.governor import ContextCostGovernor
    from mana_agent.context_cost.profiles import (
        ModelIdentity,
        ModelTokenProfile,
        ModelTokenProfileResolver,
    )
    from mana_agent.context_cost.store import AccountingStore
    from mana_agent.gateway.chat_gateway import AgentChatGateway
    from mana_agent.gateway.lanes import LaneId, default_lane_contracts

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    settings = SimpleNamespace(
        mana_context_governor_enabled=True,
        mana_context_governor_mode="enforce",
        mana_routing_task_token_budget=500,
        mana_routing_session_cost_budget=None,
        mana_routing_verification_reserve_ratio=0.15,
        mana_context_cost_log_enabled=False,
        mana_context_cost_log_retention_days=30,
        mana_context_artifact_retention_days=30,
        mana_context_unknown_model_policy="conservative",
        mana_context_unknown_model_context_window=8_192,
        mana_context_unknown_model_max_output_tokens=1_024,
        mana_context_estimation_safety_margin_ratio=0.05,
        mana_context_default_output_ratio=0.20,
        mana_context_historical_prediction_enabled=False,
    )
    governor = ContextCostGovernor(session_id="session-multi-msg", settings=settings)
    profile = ModelTokenProfile(
        ModelIdentity("fixture", "followup-model"),
        context_window=8_192,
        max_output_tokens=1_024,
        input_price_per_million=Decimal("1"),
        output_price_per_million=Decimal("2"),
        supports_usage_reporting=True,
        source="test",
        confidence="high",
    )
    governor.profile_resolver = ModelTokenProfileResolver((profile,))
    governor.accounting = governor.accounting.__class__(
        governor.profile_resolver,
        store=AccountingStore(tmp_path / "accounting-followup"),
        safety_margin_ratio=Decimal("0.05"),
        historical_prediction_enabled=False,
    )
    # Fully spend the first-message task envelope so remaining is 0.
    governor.ledger.tokens_used = int(governor.ledger.token_limit or 0)
    assert governor._implementation_tokens_remaining() == 0

    gateway = object.__new__(AgentChatGateway)
    gateway.config = SimpleNamespace(agent_max_steps=6)
    gateway._lane_coordinator = SimpleNamespace(
        select_lane=lambda **_kwargs: LaneId.RESEARCH,
        contracts=default_lane_contracts(),
    )
    gateway._stack = SimpleNamespace(context_cost_governor=governor)

    decision = SimpleNamespace(
        provider="fixture",
        selected_model="followup-model",
        estimated_output_tokens=128,
        expected_model_calls=1,
    )
    estimate = gateway._execution_token_estimate(
        entry_route="search",
        execution_decision=decision,
        request_text="follow up: expand the previous answer with more detail",
        session_id="session-multi-msg",
    )
    assert estimate.total_tokens > 0
    assert estimate.effective_total_limit >= estimate.total_tokens
    assert governor._implementation_tokens_remaining() >= 500


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
            required_child_tokens=500,
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
            required_child_tokens=600,
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


def test_multi_task_mid_run_forecast_expands_parent_before_child_recalc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provider-call forecasts for multi-task children must grow the parent envelope."""
    from mana_agent.execution_supervisor.models import BudgetForecast
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
    root_task = board.create_task(title="Compound", user_request="assets and logo")
    child_task = board.create_child_task(
        root_task.task_id,
        title="Create project assets",
        user_request="create assets",
        decomposition_local_id="create_project_assets",
        acceptance_criteria=["assets exist"],
    )

    root = coordinator.reserve(
        normalized_intent="assets and logo",
        lane_id=LaneId.RESEARCH,
        session_id="session-recalc",
        workspace_id=board.store.workspace_id,
        repository_id=board.store.repository_id,
        requested_input_tokens=100,
        requested_output_tokens=100,
        task_type="multi_task_root",
        taskboard_task_id=root_task.task_id,
    )
    coordinator.start(root)
    # Sibling already holds most of the parent remaining capacity.
    sibling_task = board.create_child_task(
        root_task.task_id,
        title="Create project logo",
        user_request="create logo",
        decomposition_local_id="create_project_logo",
        acceptance_criteria=["logo exists"],
    )
    gateway._ensure_multi_task_parent_budget(
        root.execution.task_id,
        required_child_tokens=150,
        child_estimated_cost=None,
    )
    sibling = coordinator.reserve(
        normalized_intent="create logo",
        lane_id=LaneId.MEDIA,
        session_id="session-recalc",
        workspace_id=board.store.workspace_id,
        repository_id=board.store.repository_id,
        parent_task_id=root.execution.task_id,
        root_task_id=root.execution.root_task_id,
        requested_input_tokens=50,
        requested_output_tokens=100,
        task_type="multi_task_child",
        taskboard_task_id=sibling_task.task_id,
    )
    coordinator.start(sibling)

    gateway._ensure_multi_task_parent_budget(
        root.execution.task_id,
        required_child_tokens=120,
        child_estimated_cost=None,
    )
    child = coordinator.reserve(
        normalized_intent="create assets",
        lane_id=LaneId.CODING,
        session_id="session-recalc",
        workspace_id=board.store.workspace_id,
        repository_id=board.store.repository_id,
        parent_task_id=root.execution.task_id,
        root_task_id=root.execution.root_task_id,
        requested_input_tokens=40,
        requested_output_tokens=80,
        task_type="multi_task_child",
        taskboard_task_id=child_task.task_id,
    )
    coordinator.start(child)

    # Real Codex/coding forecast far exceeds the provisional multi-task child reserve.
    gateway._recalculate_active_lane_budget(
        BudgetForecast(
            task_id=child.execution.task_id,
            forecast_input_tokens=2_000,
            forecast_output_tokens=1_500,
            forecast_cost=0.05,
            accounting_reservation_id="reservation_coding_call",
            reason="provider-call forecast",
        )
    )

    revised_child = coordinator.inspect_task(child.execution.task_id)
    parent = coordinator.inspect_task(root.execution.task_id)
    assert revised_child.budget.reserved_tokens >= 3_500
    assert parent.budget.reserved_tokens >= (
        sibling.execution.budget.reserved_tokens + revised_child.budget.reserved_tokens
    )
    assert any(
        item.get("reason") == "multi-task child budget envelope"
        for item in parent.budget.revisions
    )


class _MultiTaskMemoryModel:
    def __init__(
        self, plan: MultiTaskPlan, *, memory_task_id: str = "task-offered"
    ) -> None:
        self.plan = plan
        self.memory_task_id = memory_task_id
        self.payloads: list[dict[str, Any]] = []

    def invoke(self, messages: list[Any], **_kwargs: Any) -> Any:
        content = str(getattr(messages[0], "content", messages[0]))
        if content == MULTI_TASK_DECOMPOSITION_PROMPT:
            return SimpleNamespace(content=self.plan.model_dump_json())
        payload = json.loads(messages[-1].content)
        self.payloads.append(payload)
        user_prompt = payload.get("user_prompt", "")
        if "who-am-i" in user_prompt:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "route": "memory",
                        "confidence": 0.99,
                        "reason": "Retrieve private memory for who-am-i",
                        "required_sources": ["memory"],
                        "memory_task_id": self.memory_task_id,
                    }
                )
            )
        return SimpleNamespace(
            content=json.dumps(
                {
                    "route": "conversation",
                    "confidence": 0.99,
                    "reason": "Report memory status",
                    "required_sources": ["none"],
                }
            )
        )


class _CapsuleMemoryService:
    user_id = "user-multi-auth"
    config = SimpleNamespace(capsules=SimpleNamespace(enabled=True))

    def __init__(self, user_id: str = "user-multi-auth", return_capsules: bool = True) -> None:
        self.user_id = user_id
        self.capsules = self
        self.reads = 0
        self.calls: list[Any] = []
        self.return_capsules = return_capsules

    def query_capsules(self, request: Any, *, correlation_id: str = "") -> list[Any]:
        self.reads += 1
        self.calls.append((request, correlation_id))
        if not self.return_capsules:
            return []
        return [
            SimpleNamespace(
                capsule_id="capsule-whoami-1",
                revision=1,
                summary="User identity capsule",
                content={"identity": "authenticated agent user", "owner": request.principal.user_id},
            )
        ]


def _build_memory_gateway(
    root: Path,
    model: Any,
    user_id: str = "user-multi-auth",
    return_capsules: bool = True,
) -> tuple[AgentChatGateway, _CapsuleMemoryService]:
    registry = EntryRouteRegistry()
    for name, description in (
        ("multi_task", "compound task orchestration"),
        ("conversation", "ordinary conversation"),
        ("coding", "Codex coding"),
        ("mcp", "MCP provider operations"),
        ("gmail", "Gmail inbox"),
        ("calendar", "calendar"),
        ("browser", "browser inspection"),
        ("search", "public search"),
        ("github", "GitHub inspection"),
        ("repository", "repository inspection"),
        ("memory", "memory retrieval"),
        ("automation", "automation"),
        ("api", "external API manager"),
        ("canvas", "Live Canvas"),
        ("artifact", "artifact operations"),
        ("unsupported", "safe stop"),
        ("capability_error", "missing capability"),
    ):
        registry.register(
            RouteRegistration(
                name,  # type: ignore[arg-type]
                description,
                lambda: RouteAvailability(True),
            )
        )
    router = EntryRouter(llm=model, registry=registry)
    ask_service = SimpleNamespace(
        ask_agent=SimpleNamespace(response=None, calls=[]),
        qna_chain=SimpleNamespace(llm=None, chat=lambda question: "chat"),
        entry_router=SimpleNamespace(llm=None),
    )
    chat_service = SimpleNamespace(
        _ask_service=ask_service,
        ask_conversation=lambda question: "status reported successfully",
        ask=lambda question, **kwargs: SimpleNamespace(
            answer="status reported successfully",
            sources=[],
            warnings=[],
            trace=[],
        ),
    )
    gateway = AgentChatGateway(
        root,
        coding_agent=False,
        agent_tools=True,
        chat_service=chat_service,
        entry_route_registry=registry,
        entry_router=router,
        memory_user_id=user_id,
    )
    memory_service = _CapsuleMemoryService(user_id=user_id, return_capsules=return_capsules)
    gateway._stack.memory_service = memory_service
    return gateway, memory_service


def test_multi_task_child_inherits_authorized_memory_routing_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multi-task child inherits parent authorized memory candidates and authenticated identity."""
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    plan = MultiTaskPlan(
        goal="Retrieve who-am-i memory and report status",
        tasks=[
            MultiTaskItem(
                local_id="who-am-i-memory",
                title="Read who am I memory",
                request="Retrieve who-am-i private memory",
                depends_on=[],
                acceptance_criteria=["Memory retrieved"],
                preferred_parallelism="automatic",
                reason="Needs memory lookup",
            ),
            MultiTaskItem(
                local_id="report-memory-status",
                title="Report memory status",
                request="Report the retrieved memory status",
                depends_on=["who-am-i-memory"],
                acceptance_criteria=["Status reported"],
                preferred_parallelism="automatic",
                reason="Depends on memory lookup",
            ),
        ],
        final_acceptance_criteria=["All child tasks completed"],
        reason="Compound memory and reporting workflow",
    )
    model = _MultiTaskMemoryModel(plan, memory_task_id="task-offered")
    gateway, memory = _build_memory_gateway(tmp_path, model, user_id="user-multi-auth")

    captured_contexts: list[tuple[str, EntryRouteContext]] = []
    original_route = gateway._entry_router.route

    def tracking_route(
        *, user_prompt: str, context: EntryRouteContext
    ) -> EntryRoutingDecision:
        captured_contexts.append((user_prompt, context))
        return original_route(user_prompt=user_prompt, context=context)

    gateway._entry_router.route = tracking_route

    parent_context = EntryRouteContext(
        session_id="session-mem-parent",
        conversation_id="conv-mem-parent",
        turn_id="turn-mem-parent",
        memory_capsules_enabled=True,
        memory_task_candidates=(
            {
                "task_id": "task-offered",
                "normalized_intent": "inspect user identity",
                "state": "completed",
            },
        ),
        authenticated_user_id="user-multi-auth",
    )

    result = gateway._recover_or_execute_multi_task(
        decision=EntryRoutingDecision(
            route="multi_task",
            confidence=0.99,
            reason="Compound memory workflow",
            required_sources=("none",),
        ),
        context=parent_context,
        text="Retrieve who-am-i private memory and report status",
        state={},
        ask_service=gateway.get_ask_service(),
        sink=None,
        options={},
        turn_id="turn-mem-parent",
        user_message_id="msg-mem-parent",
    )

    # 1. Verify child EntryRouteContext captured and inherits memory routing context & identity
    assert len(captured_contexts) >= 1
    memory_child_call = next(
        (ctx for prompt, ctx in captured_contexts if "who-am-i" in prompt), None
    )
    assert memory_child_call is not None
    assert memory_child_call.memory_capsules_enabled is True
    assert (
        memory_child_call.memory_task_candidates
        == parent_context.memory_task_candidates
    )
    assert memory_child_call.memory_task_candidates == (
        {
            "task_id": "task-offered",
            "normalized_intent": "inspect user identity",
            "state": "completed",
        },
    )
    assert memory_child_call.atomic_child is True
    assert memory_child_call.authenticated_user_id == "user-multi-auth"

    # 2. Verify authorized memory read executes with the inherited principal identity
    assert memory.reads == 1
    assert memory.calls[0][0].principal.task_id == "task-offered"
    assert memory.calls[0][0].principal.user_id == "user-multi-auth"
    assert memory.calls[0][0].task_context.user_id == "user-multi-auth"

    # 3. Verify first child succeeded and returned capsule belongs to authenticated user
    children = result.payload.get("children", [])
    by_local_id = {c["local_id"]: c for c in children}
    assert by_local_id["who-am-i-memory"]["status"] == "completed"
    assert "capsule-whoami-1" in by_local_id["who-am-i-memory"]["result"]

    # 4. Verify dependent child was subsequently eligible and completed
    assert by_local_id["report-memory-status"]["status"] == "completed"
    assert result.payload.get("overall_status") == "done"


def test_multi_task_child_rejects_unauthorized_memory_task_without_reading_private_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative case: router returns unoffered memory_task_id; rejects with zero private reads."""
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    plan = MultiTaskPlan(
        goal="Retrieve who-am-i memory and report status",
        tasks=[
            MultiTaskItem(
                local_id="who-am-i-memory",
                title="Read who am I memory",
                request="Retrieve who-am-i private memory",
                depends_on=[],
                acceptance_criteria=["Memory retrieved"],
                preferred_parallelism="automatic",
                reason="Needs memory lookup",
            ),
            MultiTaskItem(
                local_id="report-memory-status",
                title="Report memory status",
                request="Report the retrieved memory status",
                depends_on=["who-am-i-memory"],
                acceptance_criteria=["Status reported"],
                preferred_parallelism="automatic",
                reason="Depends on memory lookup",
            ),
        ],
        final_acceptance_criteria=["All child tasks completed"],
        reason="Compound memory and reporting workflow",
    )
    # The routing decision returns an unoffered foreign task ID.
    model = _MultiTaskMemoryModel(plan, memory_task_id="task-foreign-unauthorized")
    gateway, memory = _build_memory_gateway(tmp_path, model, user_id="user-multi-auth")

    parent_context = EntryRouteContext(
        session_id="session-mem-parent-neg",
        conversation_id="conv-mem-parent-neg",
        turn_id="turn-mem-parent-neg",
        memory_capsules_enabled=True,
        memory_task_candidates=(
            {
                "task_id": "task-offered",
                "normalized_intent": "inspect user identity",
                "state": "completed",
            },
        ),
        authenticated_user_id="user-multi-auth",
    )

    result = gateway._recover_or_execute_multi_task(
        decision=EntryRoutingDecision(
            route="multi_task",
            confidence=0.99,
            reason="Compound memory workflow",
            required_sources=("none",),
        ),
        context=parent_context,
        text="Retrieve who-am-i private memory and report status",
        state={},
        ask_service=gateway.get_ask_service(),
        sink=None,
        options={},
        turn_id="turn-mem-parent-neg",
        user_message_id="msg-mem-parent-neg",
    )

    # Verify zero private memory reads were performed (deny-by-default preserved)
    assert memory.reads == 0

    # Verify the unoffered memory child failed
    children = result.payload.get("children", [])
    by_local_id = {c["local_id"]: c for c in children}
    assert by_local_id["who-am-i-memory"]["status"] == "failed"
    assert (
        "not offered" in by_local_id["who-am-i-memory"]["blocker"]
        or "memory_task_id" in by_local_id["who-am-i-memory"]["blocker"]
    )

    # Verify the dependent child is blocked by prerequisite
    assert by_local_id["report-memory-status"]["status"] == "blocked"
    assert "who-am-i-memory" in by_local_id["report-memory-status"]["blocker"]
    assert result.payload.get("overall_status") != "done"


def test_multi_task_child_missing_identity_fails_with_zero_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative case: unauthenticated multi-task context fails child memory read safely with 0 reads."""
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    plan = MultiTaskPlan(
        goal="Retrieve who-am-i memory and report status",
        tasks=[
            MultiTaskItem(
                local_id="who-am-i-memory",
                title="Read who am I memory",
                request="Retrieve who-am-i private memory",
                depends_on=[],
                acceptance_criteria=["Memory retrieved"],
                preferred_parallelism="automatic",
                reason="Needs memory lookup",
            ),
            MultiTaskItem(
                local_id="report-memory-status",
                title="Report memory status",
                request="Report the retrieved memory status",
                depends_on=["who-am-i-memory"],
                acceptance_criteria=["Status reported"],
                preferred_parallelism="automatic",
                reason="Depends on memory lookup",
            ),
        ],
        final_acceptance_criteria=["All child tasks completed"],
        reason="Compound memory and reporting workflow",
    )
    model = _MultiTaskMemoryModel(plan, memory_task_id="task-offered")
    gateway, memory = _build_memory_gateway(tmp_path, model, user_id="")

    parent_context = EntryRouteContext(
        session_id="session-mem-unauth",
        conversation_id="conv-mem-unauth",
        turn_id="turn-mem-unauth",
        memory_capsules_enabled=True,
        memory_task_candidates=(
            {
                "task_id": "task-offered",
                "normalized_intent": "inspect user identity",
                "state": "completed",
            },
        ),
        authenticated_user_id="",
    )

    result = gateway._recover_or_execute_multi_task(
        decision=EntryRoutingDecision(
            route="multi_task",
            confidence=0.99,
            reason="Compound memory workflow",
            required_sources=("none",),
        ),
        context=parent_context,
        text="Retrieve who-am-i private memory and report status",
        state={},
        ask_service=gateway.get_ask_service(),
        sink=None,
        options={},
        turn_id="turn-mem-unauth",
        user_message_id="msg-mem-unauth",
    )

    # 1. Zero private memory reads performed
    assert memory.reads == 0

    # 2. Child task failed because principal identity was unavailable
    children = result.payload.get("children", [])
    by_local_id = {c["local_id"]: c for c in children}
    assert by_local_id["who-am-i-memory"]["status"] == "failed"
    assert (
        "authenticated user identity" in by_local_id["who-am-i-memory"]["blocker"]
        or "memory_principal_unavailable" in by_local_id["who-am-i-memory"]["blocker"]
    )

    # 3. Dependent child is blocked
    assert by_local_id["report-memory-status"]["status"] == "blocked"
    assert result.payload.get("overall_status") != "done"


def test_multi_task_child_empty_memory_result_fails_goal_and_blocks_dependent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero matching memory capsules fails child acceptance goal and prevents dependent execution."""
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    plan = MultiTaskPlan(
        goal="Retrieve who-am-i memory and report status",
        tasks=[
            MultiTaskItem(
                local_id="who-am-i-memory",
                title="Read who am I memory",
                request="Retrieve who-am-i private memory",
                depends_on=[],
                acceptance_criteria=["Memory retrieved"],
                preferred_parallelism="automatic",
                reason="Needs memory lookup",
            ),
            MultiTaskItem(
                local_id="report-memory-status",
                title="Report memory status",
                request="Report the retrieved memory status",
                depends_on=["who-am-i-memory"],
                acceptance_criteria=["Status reported"],
                preferred_parallelism="automatic",
                reason="Depends on memory lookup",
            ),
        ],
        final_acceptance_criteria=["All child tasks completed"],
        reason="Compound memory and reporting workflow",
    )
    # Memory query will return 0 matching capsules
    model = _MultiTaskMemoryModel(plan, memory_task_id="task-offered")
    gateway, memory = _build_memory_gateway(
        tmp_path, model, user_id="user-multi-auth", return_capsules=False
    )

    parent_context = EntryRouteContext(
        session_id="session-mem-empty",
        conversation_id="conv-mem-empty",
        turn_id="turn-mem-empty",
        memory_capsules_enabled=True,
        memory_task_candidates=(
            {
                "task_id": "task-offered",
                "normalized_intent": "inspect user identity",
                "state": "completed",
            },
        ),
        authenticated_user_id="user-multi-auth",
    )

    result = gateway._recover_or_execute_multi_task(
        decision=EntryRoutingDecision(
            route="multi_task",
            confidence=0.99,
            reason="Compound memory workflow",
            required_sources=("none",),
        ),
        context=parent_context,
        text="Retrieve who-am-i private memory and report status",
        state={},
        ask_service=gateway.get_ask_service(),
        sink=None,
        options={},
        turn_id="turn-mem-empty",
        user_message_id="msg-mem-empty",
    )

    # 1. Memory query was attempted
    assert memory.reads == 1

    # 2. Child task failed because goal was unsatisfied (0 matching records)
    children = result.payload.get("children", [])
    by_local_id = {c["local_id"]: c for c in children}
    assert by_local_id["who-am-i-memory"]["status"] == "failed"
    assert by_local_id["who-am-i-memory"]["verification_status"] == "failed"
    assert by_local_id["who-am-i-memory"]["verification_status"] != "route-memory"

    # 3. Dependent child is blocked and overall compound task is not done
    assert by_local_id["report-memory-status"]["status"] == "blocked"
    assert result.payload.get("overall_status") in {"failed", "blocked"}


def test_multi_task_two_of_two_completed_children_remains_overall_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When 2/2 children complete, overall status is done and root lane completes cleanly."""
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    plan = MultiTaskPlan(
        goal="Retrieve who-am-i memory and report status",
        tasks=[
            MultiTaskItem(
                local_id="who-am-i-memory",
                title="Read who am I memory",
                request="Retrieve who-am-i private memory",
                depends_on=[],
                acceptance_criteria=["Memory retrieved"],
                preferred_parallelism="automatic",
                reason="Needs memory lookup",
            ),
            MultiTaskItem(
                local_id="report-memory-status",
                title="Report memory status",
                request="Report the retrieved memory status",
                depends_on=["who-am-i-memory"],
                acceptance_criteria=["Status reported"],
                preferred_parallelism="automatic",
                reason="Depends on memory lookup",
            ),
        ],
        final_acceptance_criteria=["All child tasks completed"],
        reason="Compound memory and reporting workflow",
    )
    model = _MultiTaskMemoryModel(plan, memory_task_id="task-offered")
    gateway, memory = _build_memory_gateway(
        tmp_path, model, user_id="user-multi-auth", return_capsules=True
    )

    parent_context = EntryRouteContext(
        session_id="session-mem-success",
        conversation_id="conv-mem-success",
        turn_id="turn-mem-success",
        memory_capsules_enabled=True,
        memory_task_candidates=(
            {
                "task_id": "task-offered",
                "normalized_intent": "inspect user identity",
                "state": "completed",
            },
        ),
        authenticated_user_id="user-multi-auth",
    )

    result = gateway._recover_or_execute_multi_task(
        decision=EntryRoutingDecision(
            route="multi_task",
            confidence=0.99,
            reason="Compound memory workflow",
            required_sources=("none",),
        ),
        context=parent_context,
        text="Retrieve who-am-i private memory and report status",
        state={},
        ask_service=gateway.get_ask_service(),
        sink=None,
        options={},
        turn_id="turn-mem-success",
        user_message_id="msg-mem-success",
    )

    children = result.payload.get("children", [])
    by_local_id = {c["local_id"]: c for c in children}
    assert by_local_id["who-am-i-memory"]["status"] == "completed"
    assert by_local_id["who-am-i-memory"]["verification_status"] == "passed"
    assert by_local_id["report-memory-status"]["status"] == "completed"
    assert by_local_id["report-memory-status"]["verification_status"] == "passed"
    assert result.payload.get("overall_status") == "done"
    assert result.payload.get("root_lane_state") == "completed"
    assert result.payload.get("root_lane_error") == ""


def test_multi_task_actual_completion_verification_failure_returns_verification_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When root lane completion verification fails, overall status reflects verification_failed."""
    from mana_agent.gateway.lane_coordinator import LaneTaskState

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    plan = MultiTaskPlan(
        goal="Retrieve who-am-i memory and report status",
        tasks=[
            MultiTaskItem(
                local_id="who-am-i-memory",
                title="Read who am I memory",
                request="Retrieve who-am-i private memory",
                depends_on=[],
                acceptance_criteria=["Memory retrieved"],
                preferred_parallelism="automatic",
                reason="Needs memory lookup",
            ),
            MultiTaskItem(
                local_id="report-memory-status",
                title="Report memory status",
                request="Report the retrieved memory status",
                depends_on=["who-am-i-memory"],
                acceptance_criteria=["Status reported"],
                preferred_parallelism="automatic",
                reason="Depends on memory lookup",
            ),
        ],
        final_acceptance_criteria=["All child tasks completed"],
        reason="Compound memory workflow",
    )
    model = _MultiTaskMemoryModel(plan, memory_task_id="task-offered")
    gateway, memory = _build_memory_gateway(
        tmp_path, model, user_id="user-multi-auth", return_capsules=True
    )

    # Monkeypatch _finish_lane to simulate a verification failure
    orig_finish_lane = gateway._finish_lane

    def mock_finish_lane(task_id: str, **kwargs: Any) -> Any:
        finished = orig_finish_lane(task_id, **kwargs)
        # If this is the root task finish, simulate VERIFYING state
        if kwargs.get("state") != LaneTaskState.FAILED and "children" in (kwargs.get("verification_state") or {}):
            return SimpleNamespace(state=LaneTaskState.VERIFYING, error="Contract file verification failed")
        return finished

    monkeypatch.setattr(gateway, "_finish_lane", mock_finish_lane)

    parent_context = EntryRouteContext(
        session_id="session-mem-verif-fail",
        conversation_id="conv-mem-verif-fail",
        turn_id="turn-mem-verif-fail",
        memory_capsules_enabled=True,
        memory_task_candidates=(
            {
                "task_id": "task-offered",
                "normalized_intent": "inspect user identity",
                "state": "completed",
            },
        ),
        authenticated_user_id="user-multi-auth",
    )

    result = gateway._recover_or_execute_multi_task(
        decision=EntryRoutingDecision(
            route="multi_task",
            confidence=0.99,
            reason="Compound memory workflow",
            required_sources=("none",),
        ),
        context=parent_context,
        text="Retrieve who-am-i private memory",
        state={},
        ask_service=gateway.get_ask_service(),
        sink=None,
        options={},
        turn_id="turn-mem-verif-fail",
        user_message_id="msg-mem-verif-fail",
    )

    assert result.payload.get("overall_status") == "verification_failed"
    assert result.payload.get("root_lane_state") == "verifying"
    assert "Contract file verification failed" in result.payload.get("root_lane_error", "")


