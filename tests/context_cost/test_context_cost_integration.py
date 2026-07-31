from __future__ import annotations

from types import SimpleNamespace

import pytest

from mana_agent.context_cost import ContextArtifactStore, ContextCostGovernor
from mana_agent.context_cost.models import GovernorMode
from mana_agent.model_routing.models import RoutingBudgets


def _settings(**overrides):
    values = {
        "mana_context_governor_enabled": True,
        "mana_context_governor_mode": "observe",
        "mana_context_warning_ratio": 0.70,
        "mana_context_compact_ratio": 0.80,
        "mana_context_max_utilization": 0.85,
        "mana_context_hard_limit_ratio": 0.95,
        "mana_context_response_reserve_ratio": 0.12,
        "mana_context_response_reserve_tokens": 0,
        "mana_context_tool_result_max_tokens": 20,
        "mana_context_history_max_tokens": 100,
        "mana_context_artifact_retention_days": 30,
        "mana_context_cost_log_enabled": False,
        "mana_context_cost_log_retention_days": 30,
        "mana_routing_task_token_budget": 1_000,
        "mana_routing_session_cost_budget": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_observe_mode_records_but_does_not_compress_tool_results() -> None:
    governor = ContextCostGovernor(session_id="s", settings=_settings())
    raw = "large result " * 1_000
    assert governor.mode is GovernorMode.OBSERVE
    assert governor.prepare_tool_result(raw, tool_name="repo_search") == raw


def test_soft_mode_compresses_large_result_and_emits_existing_event_envelope(tmp_path) -> None:
    events = []
    governor = ContextCostGovernor(
        session_id="s", repository_id="r", workspace_id="w",
        settings=_settings(mana_context_governor_mode="soft"),
        event_sink=events.append,
    )
    governor.artifacts = ContextArtifactStore(tmp_path / "artifacts")
    compact = governor.prepare_tool_result("error line\n" * 1_000, tool_name="process_log")
    assert "mana.context.compression_envelope" in compact
    assert any(event.type == "context.compacted" for event in events)
    assert governor.observability_snapshot()["compression_tokens_saved"] > 0


def test_actual_remaining_budget_is_fed_back_to_routing() -> None:
    governor = ContextCostGovernor(session_id="s", settings=_settings())
    governor.ledger.tokens_used = 400
    governor.ledger.estimated_cost = 0.25
    remaining = governor.remaining_routing_budgets(
        RoutingBudgets(task_token_limit=1_000, session_cost_remaining=0.9)
    )
    assert remaining.task_token_limit == 450  # 600 remaining minus the untouched 15% verification reserve
    assert remaining.session_cost_remaining == 0.75


def test_parallel_candidates_are_rejected_when_child_reservations_do_not_fit() -> None:
    governor = ContextCostGovernor(session_id="s", settings=_settings())
    decision = SimpleNamespace(
        decision_id="decision-1",
        task_id="task-1",
        estimated_input_tokens=150,
        estimated_output_tokens=150,
        candidate_competition=True,
        competition_candidates=("a", "b", "c"),
        multi_agent_execution_permitted=True,
        applicable_budgets=RoutingBudgets(competition_cost_limit=0.3),
    )
    with pytest.raises(ValueError, match="parent remaining"):
        governor.reserve_routing_children(decision)
