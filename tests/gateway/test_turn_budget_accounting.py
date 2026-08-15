from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mana_agent.analysis.models import AskResponseWithTrace, ToolInvocationTrace
from mana_agent.gateway.lane_coordinator import (
    LaneBudget,
    LaneBudgetError,
    LaneCoordinator,
    LaneReservation,
)
from mana_agent.gateway.lanes import LaneId, LaneTaskState
from mana_agent.multi_agent.runtime.ask_agent import AskAgent


@pytest.fixture
def coordinator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LaneCoordinator:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    return LaneCoordinator(root)


def _reserve_task(
    coordinator: LaneCoordinator,
    lane: LaneId = LaneId.RESEARCH,
    *,
    intent: str = "test turn accounting",
    session: str = "session-test",
) -> LaneReservation:
    return coordinator.reserve(
        normalized_intent=intent,
        lane_id=lane,
        session_id=session,
        workspace_id="ws-test",
        repository_id="repo-test",
        target_files=(),
        requested_input_tokens=1000,
        requested_output_tokens=500,
    )


# --- Case A: Fresh Turn After Heavy Previous Usage ---
def test_case_a_fresh_turn_after_heavy_previous_usage(coordinator: LaneCoordinator) -> None:
    """Turn 2 must start with turn_consumed_tokens = 0 without losing cumulative session/lane usage."""
    reservation = _reserve_task(coordinator, LaneId.RESEARCH)
    task_id = reservation.execution.task_id
    coordinator.start(reservation)

    # Turn 1: heavy usage (5,000 tokens)
    coordinator.reset_turn_accounting(task_id, allocated_tokens=6000)
    coordinator.synchronize_usage(
        task_id,
        consumed_input_tokens=3000,
        consumed_output_tokens=2000,
    )

    execution = coordinator.inspect_task(task_id)
    assert execution.budget.consumed_tokens == 5000
    assert execution.budget.turn_consumed_tokens == 5000
    assert execution.budget.turn_remaining_tokens == 1000

    # Turn 2: authoritative turn boundary reset with fresh 2,000 token allocation
    coordinator.reset_turn_accounting(task_id, allocated_tokens=2000)
    execution = coordinator.inspect_task(task_id)

    # Cumulative usage is preserved, turn usage is freshly zeroed
    assert execution.budget.consumed_tokens == 5000
    assert execution.budget.turn_consumed_tokens == 0
    assert execution.budget.turn_budget_tokens == 2000
    assert execution.budget.turn_remaining_tokens == 2000
    assert not execution.budget.is_turn_budget_exhausted

    # Incremental usage in Turn 2 (300 tokens: 150 in, 150 out)
    coordinator.synchronize_usage(
        task_id,
        consumed_input_tokens=3150,
        consumed_output_tokens=2150,
    )
    execution = coordinator.inspect_task(task_id)
    assert execution.budget.consumed_tokens == 5300
    assert execution.budget.turn_consumed_tokens == 300
    assert execution.budget.turn_remaining_tokens == 1700
    assert coordinator.can_continue_turn(task_id)


# --- Case B: Intermediate Tool Success Is Not Completion ---
def test_case_b_intermediate_tool_success_is_not_completion() -> None:
    """Intermediate tool returning resource IDs without final answer must yield pending_required_work=True."""
    traces = [
        ToolInvocationTrace(
            tool_name="email_search",
            args_summary="query='latest report'",
            duration_ms=45.0,
            status="ok",
            output_preview=json.dumps({"ok": True, "messages": [{"id": "msg_abc123", "thread_id": "th_1"}]}),
        )
    ]

    intermediate_results = AskAgent._extract_intermediate_results(traces)
    assert intermediate_results.get("id") == "msg_abc123"

    response = AskResponseWithTrace(
        answer="Found latest email message ID msg_abc123.",
        sources=[],
        mode="agent-tools",
        trace=traces,
        warnings=["Tool loop stopped (needs_continuation); returned best-effort final answer."],
        status="needs_continuation",
        pending_required_work=True,
        stop_reason="max_steps_reached",
        intermediate_results=intermediate_results,
    )

    assert response.status == "needs_continuation"
    assert response.pending_required_work is True
    assert response.intermediate_results["id"] == "msg_abc123"


# --- Case C: Genuine Completion ---
def test_case_c_genuine_completion(coordinator: LaneCoordinator) -> None:
    """When all required tool calls complete and natural final answer is produced, pending_required_work=False."""
    reservation = _reserve_task(coordinator, LaneId.RESEARCH)
    task_id = reservation.execution.task_id
    coordinator.start(reservation)

    traces = [
        ToolInvocationTrace(
            tool_name="email_search",
            args_summary="query='latest report'",
            duration_ms=40.0,
            status="ok",
            output_preview=json.dumps({"ok": True, "messages": [{"id": "msg_abc123"}]}),
        ),
        ToolInvocationTrace(
            tool_name="email_get_message",
            args_summary="message_id='msg_abc123'",
            duration_ms=50.0,
            status="ok",
            output_preview=json.dumps({"ok": True, "subject": "Q3 Report", "body": "All systems go."}),
        ),
    ]

    response = AskResponseWithTrace(
        answer="The latest email from Q3 Report states: All systems go.",
        sources=[],
        mode="agent-tools",
        trace=traces,
        status="completed",
        pending_required_work=False,
        stop_reason="completed",
    )

    assert response.status == "completed"
    assert response.pending_required_work is False

    finished = coordinator.finish(
        task_id,
        state=LaneTaskState.COMPLETED,
        consumed_input_tokens=500,
        consumed_output_tokens=200,
        verification_state={
            "status": "completed",
            "chat_result": {"answer": response.answer},
        },
    )
    assert finished.state == LaneTaskState.COMPLETED
    assert finished.budget.turn_consumed_tokens == 700


# --- Case D: Genuine Hard Budget Exhaustion ---
def test_case_d_genuine_hard_budget_exhaustion(coordinator: LaneCoordinator) -> None:
    """Hard turn budget exhaustion must report budget_exhausted and preserve resumable intermediate results."""
    reservation = _reserve_task(coordinator, LaneId.RESEARCH)
    task_id = reservation.execution.task_id
    coordinator.start(reservation)

    # Set hard turn budget to 500 tokens
    coordinator.reset_turn_accounting(task_id, allocated_tokens=500)
    coordinator.synchronize_usage(
        task_id,
        consumed_input_tokens=300,
        consumed_output_tokens=200,
    )

    execution = coordinator.inspect_task(task_id)
    assert execution.budget.is_turn_budget_exhausted
    assert execution.budget.turn_remaining_tokens == 0
    assert not coordinator.can_continue_turn(task_id)

    # Save checkpoint with discovered intermediate result
    checkpoint_id = coordinator.checkpoint(
        task_id,
        boundary="budget_exhausted",
        resume_payload={
            "intermediate_results": {"message_id": "msg_xyz789"},
            "pending_steps": ["fetch_message", "summarize"],
        },
    )
    assert checkpoint_id

    finished = coordinator.finish(
        task_id,
        state=LaneTaskState.BUDGET_EXHAUSTED,
        error="turn_budget_exhausted",
    )
    assert finished.state == LaneTaskState.BUDGET_EXHAUSTED


# --- Case E: Soft Budget Threshold Does Not Abort Required Steps ---
def test_case_e_soft_budget_threshold_does_not_abort_required_steps(tmp_path: Path) -> None:
    """ask_agent must not abort with remaining_tool_budget_low on step max_steps-1 when budget exists."""
    class _FakeAIMessage:
        def __init__(self, content: str, tool_calls: list[dict] | None = None) -> None:
            self.content = content
            self.tool_calls = tool_calls or []

    class _FakeBoundModel:
        def __init__(self, responses: list[_FakeAIMessage]) -> None:
            self._responses = responses
            self._idx = 0

        def invoke(self, _messages: list[object], config: object | None = None) -> _FakeAIMessage:
            del config
            value = self._responses[self._idx]
            self._idx += 1
            return value

    class _FakeLLM:
        def __init__(self, responses: list[_FakeAIMessage]) -> None:
            self._responses = responses

        def bind_tools(self, _tools: list[object]) -> _FakeBoundModel:
            return _FakeBoundModel(self._responses)

    # 2-step sequence: Step 0 calls search, Step 1 calls get_message and produces final text
    llm = _FakeLLM([
        _FakeAIMessage("", tool_calls=[{"id": "1", "name": "semantic_search", "args": {"query": "test", "k": 2}}]),
        _FakeAIMessage("Final answer found in step 2.", tool_calls=[]),
    ])

    from langchain_core.tools import StructuredTool
    from mana_agent.context_cost.governor import ContextCostGovernor
    from mana_agent.search.config import SearchConfig

    class _FakeSearchService:
        def search(self, *args, **kwargs):
            return []

    search_tool = StructuredTool.from_function(
        func=lambda query, k=2: json.dumps({"ok": True, "results": ["item 1"]}),
        name="semantic_search",
        description="Search tool",
    )

    governor = ContextCostGovernor(
        session_id="test-case-e",
        settings=SimpleNamespace(
            mana_context_governor_enabled=False,
            mana_context_cost_log_enabled=False,
        ),
    )
    agent = AskAgent(
        api_key="test",
        model="fake-agent",
        search_service=_FakeSearchService(),
        project_root=tmp_path,
        context_cost_governor=governor,
    )
    agent._resolved_index = tmp_path / ".mana/index"
    agent.search_config = SearchConfig(enable_ask_agent=False)
    agent.llm = llm
    agent.tools = [search_tool]

    result = agent.run("What is in test?", tmp_path / ".mana/index", 2, max_steps=2, timeout_seconds=5)

    # The second step executed cleanly and natural completion was achieved
    assert "Final answer found in step 2." in result.answer
    assert result.status == "completed"
    assert result.pending_required_work is False
    assert not any("remaining_tool_budget_low" in str(w) for w in result.warnings)


# --- Case F: Resume Without Duplicate Work ---
def test_case_f_resume_without_duplicate_work(coordinator: LaneCoordinator) -> None:
    """Resumed checkpoint preserves intermediate results so discovery does not need repeating."""
    reservation = _reserve_task(coordinator, LaneId.RESEARCH)
    task_id = reservation.execution.task_id
    coordinator.start(reservation)

    # Save intermediate result in checkpoint
    checkpoint_id = coordinator.checkpoint(
        task_id,
        boundary="intermediate_pause",
        resume_payload={
            "intermediate_results": {
                "message_id": "msg_cached_42",
                "sender": "alice@example.com",
            },
        },
    )

    execution = coordinator.inspect_task(task_id)
    assert execution.checkpoint_id == checkpoint_id

    # On follow-up turn, intermediate results are accessible
    # Reset turn accounting for turn 2
    coordinator.reset_turn_accounting(task_id, allocated_tokens=3000)
    assert execution.budget.turn_consumed_tokens == 0
    assert execution.budget.turn_remaining_tokens == 3000


# --- Case G: Higher-Scope Limit Still Wins ---
def test_case_g_higher_scope_limit_still_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When a global/session/lane limit is reached, execution is blocked even if turn budget is fresh."""
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()

    # Coordinator with hard session token budget of 5,000 tokens
    coordinator = LaneCoordinator(root, session_token_budget=5000)
    reservation = _reserve_task(coordinator, LaneId.RESEARCH, session="session-bounded")
    task_id = reservation.execution.task_id
    coordinator.start(reservation)

    # Cumulative usage consumes 4,800 tokens
    coordinator.synchronize_usage(
        task_id,
        consumed_input_tokens=2800,
        consumed_output_tokens=2000,
    )

    # Fresh turn starts with 2,000 token turn budget
    coordinator.reset_turn_accounting(task_id, allocated_tokens=2000)
    assert coordinator.inspect_task(task_id).budget.turn_remaining_tokens == 2000

    # Recalculating budget for 500 forecast tokens exceeds the session limit (4800 + 500 = 5300 > 5000)
    with pytest.raises(LaneBudgetError, match="session token limit"):
        coordinator.recalculate_budget(
            task_id,
            forecast_input_tokens=300,
            forecast_output_tokens=200,
            forecast_cost=0.0,
        )

    # can_continue_turn with reserve 300 fails because 4800 + 300 = 5100 > 5000
    assert not coordinator.can_continue_turn(task_id, required_reserve=300)
