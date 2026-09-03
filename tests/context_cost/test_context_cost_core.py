from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mana_agent.context_cost import (
    ArtifactAccessError,
    CapabilityRegistry,
    ContextArtifactStore,
    ContextCostGovernor,
    calculate_cost,
    compress_tool_result,
)
from mana_agent.context_cost.estimator import estimate_value_tokens
from mana_agent.context_cost.logger import ContextCostLogger
from mana_agent.context_cost.models import ContextBudgetExceeded, ContextSegment, CostLedger, GovernorMode


def settings(**overrides):
    values = {
        "mana_context_governor_enabled": True,
        "mana_context_governor_mode": "soft",
        "mana_context_warning_ratio": 0.70,
        "mana_context_compact_ratio": 0.80,
        "mana_context_max_utilization": 0.85,
        "mana_context_hard_limit_ratio": 0.95,
        "mana_context_response_reserve_ratio": 0.12,
        "mana_context_response_reserve_tokens": 0,
        "mana_context_tool_result_max_tokens": 100,
        "mana_context_history_max_tokens": 100,
        "mana_context_artifact_retention_days": 30,
        "mana_context_cost_log_enabled": False,
        "mana_context_cost_log_retention_days": 30,
        "mana_routing_task_token_budget": 10_000,
        "mana_routing_session_cost_budget": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def register_priced_test_model(
    governor: ContextCostGovernor,
    *,
    context_window: int,
    max_output_tokens: int,
) -> None:
    governor.register_model_profiles((
        SimpleNamespace(
            provider="unknown",
            model_id="test",
            input_cost_per_million=1.0,
            output_cost_per_million=1.0,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            configuration={},
        ),
    ))


def test_token_estimation_and_exact_configured_pricing() -> None:
    assert estimate_value_tokens("hello") > 0
    profile = SimpleNamespace(input_cost_per_million=2.0, output_cost_per_million=4.0)
    cost = calculate_cost(1_000_000, 500_000, profile=profile)
    assert cost.total_cost == 4.0
    assert cost.estimated is False
    fallback = calculate_cost(100, 100)
    assert fallback.estimated is True
    assert fallback.total_cost is None


def test_parent_child_budget_allocation_and_consumption() -> None:
    parent = CostLedger("session", token_limit=1_000, cost_limit=1.0)
    child = parent.allocate_child("verifier", token_limit=300, cost_limit=0.2)
    child.record(tokens=100, input_cost=0.01, output_cost=0.02, estimated=True)
    assert child.tokens_used == 100
    assert parent.tokens_used == 100
    assert parent.total_cost == pytest.approx(0.03)
    with pytest.raises(ValueError, match="parent remaining"):
        parent.allocate_child("candidate", token_limit=800)


def test_ensure_admission_budget_refreshes_depleted_session_for_followup_message() -> None:
    """Sequential messages must not inherit an effective remaining of 0."""
    governor = ContextCostGovernor(
        session_id="session-followup",
        settings=settings(
            mana_context_governor_mode="enforce",
            mana_routing_task_token_budget=1_000,
        ),
    )
    # Simulate a completed first message that fully spent the task-sized ledger.
    governor.ledger.tokens_used = int(governor.ledger.token_limit or 0)
    assert governor._implementation_tokens_remaining() == 0

    remaining = governor.ensure_admission_budget()
    assert remaining is not None
    assert remaining >= 1_000
    assert governor._implementation_tokens_remaining() >= 1_000

    # A second follow-up estimate must not hard-fail with effective limit 0.
    estimate = governor.estimate_execution(
        provider="test-provider",
        model="test-model",
        components={"user_request": "follow up with more detail"},
        requested_output_tokens=64,
    )
    assert estimate.total_tokens > 0
    assert estimate.effective_total_limit >= estimate.total_tokens


def test_task_usage_keeps_actual_and_estimated_costs_separate() -> None:
    governor = ContextCostGovernor(session_id="s", settings=settings())
    governor.register_model_profiles((
        SimpleNamespace(
            provider="test-provider",
            model_id="test-model",
            input_cost_per_million=2.0,
            output_cost_per_million=4.0,
            context_window=16_384,
            max_output_tokens=4_096,
            configuration={},
        ),
    ))
    governor.set_execution_identity(task_id="task-1")

    governor.record_model_call(
        "actual-call",
        usage={"input_tokens": 100, "output_tokens": 50},
        provider="test-provider",
        model="test-model",
    )
    governor.record_model_call(
        "estimated-call",
        provider="test-provider",
        model="test-model",
        estimated_input="estimated prompt",
        estimated_output="estimated result",
    )

    usage = governor.task_usage("task-1")
    assert usage["consumed_input_tokens"] >= 100
    assert usage["consumed_output_tokens"] >= 50
    assert usage["actual_cost"] == pytest.approx(0.0004)
    assert usage["estimated_cost"] > 0
    assert governor.observability_snapshot()["actual_cost"] == pytest.approx(0.0004)


def test_enforce_mode_blocks_before_provider_and_protects_required_segments(tmp_path: Path) -> None:
    governor = ContextCostGovernor(
        session_id="s", repository_id="r", workspace_id="w",
        settings=settings(
            mana_context_governor_mode="enforce",
            mana_context_unknown_model_context_window=10_000,
            mana_context_unknown_model_max_output_tokens=1_000,
        ),
    )
    governor.logger = ContextCostLogger(enabled=False, root=tmp_path / "logs")
    register_priced_test_model(
        governor,
        context_window=10_000,
        max_output_tokens=1_000,
    )
    protected = ContextSegment("system", "safety", 10, protected=True, source_id="system")
    duplicate_protected = ContextSegment("system", "safety", 10, protected=True, source_id="system-copy")
    history = ContextSegment("history", "old", 10, source_id="old")
    duplicate_history = ContextSegment("history", "old", 10, source_id="old-copy")
    _, decision = governor.before_model_call(
        [protected, duplicate_protected, history, duplicate_history],
        model="test", context_window=10_000, apply_compaction=True,
    )
    assert [segment.source_id for segment in decision.segments] == ["system", "system-copy", "old"]
    huge = ContextSegment("user", "context " * 12_000, 12_000, protected=True, source_id="current-user")
    with pytest.raises(ContextBudgetExceeded) as raised:
        governor.before_model_call([protected, huge], model="test", context_window=10_000)
    assert raised.value.decision.allowed is False
    assert any(segment.source_id == "current-user" for segment in raised.value.decision.segments)


def test_observe_mode_records_task_budget_overrun_without_blocking() -> None:
    governor = ContextCostGovernor(
        session_id="observe-session",
        settings=settings(
            mana_context_governor_mode="observe",
            mana_routing_task_token_budget=100,
        ),
    )
    register_priced_test_model(
        governor,
        context_window=10_000,
        max_output_tokens=1_000,
    )

    _, decision = governor.before_model_call(
        [ContextSegment("user", "short request", 3, protected=True, source_id="user")],
        model="test",
        expected_output_tokens=100,
    )

    assert decision.allowed is True
    assert decision.action == "observe"
    assert decision.reason == "context_hard_limit"
    assert governor.observability_snapshot()["calls_blocked_token"] == 0


def test_history_selection_is_token_aware_and_chronological() -> None:
    governor = ContextCostGovernor(session_id="s", settings=settings(mana_context_history_max_tokens=20))
    messages = [{"role": "user", "content": str(index) * 20} for index in range(10)]
    selected = governor.select_history(messages)
    assert selected
    assert selected == messages[-len(selected):]
    assert sum(estimate_value_tokens(item) for item in selected) <= 20


def test_scoped_execution_identity_restores_the_prior_accounting_scope() -> None:
    governor = ContextCostGovernor(session_id="s", settings=settings())
    governor.set_execution_identity(
        turn_id="outer-turn",
        task_id="outer-task",
        execution_kind="route_execution",
    )

    with governor.scoped_execution_identity(
        turn_id="checkpoint-turn",
        step_id="checkpoint_resume:decision",
        route="conversation",
        lane="research",
        execution_kind="checkpoint_resume",
    ):
        scoped = governor._effective_identity()
        assert scoped["turn_id"] == "checkpoint-turn"
        assert scoped["task_id"] == "outer-task"
        assert scoped["execution_kind"] == "checkpoint_resume"

    restored = governor._effective_identity()
    assert restored["turn_id"] == "outer-turn"
    assert restored["task_id"] == "outer-task"
    assert restored["execution_kind"] == "route_execution"


def test_deterministic_compression_is_small_and_exactly_recoverable(tmp_path: Path) -> None:
    store = ContextArtifactStore(tmp_path / "artifacts")
    payload = [{"id": index, "status": "ok", "value": "same " * 100} for index in range(200)]
    first = compress_tool_result(payload, tool_name="repo_search", store=store, session_id="s", repository_id="r", workspace_id="w")
    second = compress_tool_result(payload, tool_name="repo_search", store=store, session_id="s", repository_id="r", workspace_id="w")
    assert first.as_dict() == second.as_dict()
    assert first.compact_token_estimate < first.original_token_estimate * 0.5
    chunks: list[str] = []
    offset = 0
    while chunk := store.read(
        first.artifact_ref,
        session_id="s",
        repository_id="r",
        workspace_id="w",
        offset=offset,
        limit=64_000,
    ):
        chunks.append(chunk)
        offset += len(chunk)
    exact = "".join(chunks)
    assert json.loads(exact) == payload
    assert store.read(first.artifact_ref, session_id="s", repository_id="r", workspace_id="w", json_path="$.[3].id") == 3


def test_artifact_scope_rejects_cross_session_and_arbitrary_references(tmp_path: Path) -> None:
    store = ContextArtifactStore(tmp_path / "artifacts")
    reference = store.put("permitted", session_id="s", repository_id="r", workspace_id="w", content_type="text")
    with pytest.raises(ArtifactAccessError, match="scope"):
        store.read(reference, session_id="other", repository_id="r", workspace_id="w")
    with pytest.raises(ArtifactAccessError, match="invalid"):
        store.read("../../etc/passwd", session_id="s", repository_id="r", workspace_id="w")


class FakeTool:
    def __init__(self, name: str, description: str = "tool") -> None:
        self.name = name
        self.description = description
        self.args_schema = None


def test_capability_manifest_load_never_widens_authorization_and_saves_schemas() -> None:
    tools = [FakeTool(f"read_{index}", "read evidence") for index in range(20)] + [FakeTool("delete_file", "delete")]
    registry = CapabilityRegistry(tools, allowed_names={"read_1", "read_2"})
    registry.initial({"read_1"}, include_core=False)
    response = registry.load(["read_2", "delete_file", "missing"])
    assert response["loaded"] == ["read_2"]
    assert response["denied"] == ["delete_file"]
    assert response["unknown"] == ["missing"]
    assert {tool.name for tool in registry.bound_tools()} == {"read_1", "read_2"}
    assert registry.active.schema_tokens < registry.schema_tokens_for(tool.name for tool in tools) * 0.4


def test_capability_pin_protects_tools_from_idle_unloading() -> None:
    tools = [
        FakeTool("read_1", "read"),
        FakeTool("api_request_preview", "preview"),
        FakeTool("api_request_execute", "execute"),
    ]
    registry = CapabilityRegistry(tools, allowed_names={"read_1", "api_request_preview", "api_request_execute"})
    registry.initial({"read_1", "api_request_preview", "api_request_execute"}, include_core=False)
    assert {tool.name for tool in registry.bound_tools()} == {"read_1", "api_request_preview", "api_request_execute"}

    # Pin preview and execute so idle unloading won't touch them
    registry.pin(["api_request_preview", "api_request_execute"], step=0)
    assert "api_request_preview" in registry.pinned
    assert "api_request_execute" in registry.pinned

    # At step 10 (far exceeding default idle_steps=3), read_1 is unloaded for being idle, but pinned tools remain
    unloaded = registry.unload_idle(step=10, idle_steps=3)
    assert "read_1" in unloaded["unloaded"]
    assert "api_request_preview" not in unloaded["unloaded"]
    assert "api_request_execute" not in unloaded["unloaded"]
    assert {tool.name for tool in registry.bound_tools()} == {"api_request_preview", "api_request_execute"}

    # Unpinning allows normal unloading
    registry.unpin(["api_request_preview"])
    assert "api_request_preview" not in registry.pinned
    unloaded2 = registry.unload_idle(step=10, idle_steps=3)
    assert "api_request_preview" in unloaded2["unloaded"]
    assert "api_request_execute" not in unloaded2["unloaded"]
    assert {tool.name for tool in registry.bound_tools()} == {"api_request_execute"}


def test_logger_redacts_and_retention_cleanup_is_best_effort(tmp_path: Path) -> None:
    logger = ContextCostLogger(root=tmp_path, enabled=True, retention_days=7)
    logger.write({"session_id": "s", "api_key": "sk-secret-value", "authorization": "Bearer abc.def", "prompt": "private prompt text"})
    line = (tmp_path / f"{date.today().isoformat()}.jsonl").read_text(encoding="utf-8")
    assert "sk-secret-value" not in line
    assert "abc.def" not in line
    assert "private prompt text" not in line
    old = tmp_path / f"{(date.today() - timedelta(days=10)).isoformat()}.jsonl"
    old.write_text("{}\n", encoding="utf-8")
    assert logger.cleanup() == 1
    assert not old.exists()


def test_sequential_model_calls_under_same_task_identity_get_fresh_call_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lane tasks pin step_id (for example after_routing) for many provider calls.

    Search routes issue a query-decision call and a later answer-synthesis call
    under that same identity. Each must get a free accounting operation id.
    """
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    governor = ContextCostGovernor(
        session_id="session-search",
        repository_id="repo",
        workspace_id="workspace",
        settings=settings(
            mana_context_governor_mode="observe",
            mana_context_unknown_model_context_window=8_000,
            mana_context_unknown_model_max_output_tokens=1_000,
        ),
    )
    register_priced_test_model(
        governor,
        context_window=8_000,
        max_output_tokens=1_000,
    )
    governor.set_execution_identity(
        turn_id="turn-1",
        task_id="task_20260807_000002",
        attempt_id="attempt-1",
        step_id="after_routing",
        route="search",
        lane="research",
        execution_kind="gateway_route",
    )
    segment = ContextSegment(
        "user",
        "check what is hermes agent.",
        8,
        protected=True,
        source_id="user:1",
    )

    first_id, _ = governor.before_model_call(
        [segment],
        model="test",
        provider="test-provider",
    )
    governor.record_model_call(
        first_id,
        usage={"input_tokens": 20, "output_tokens": 10},
        provider="test-provider",
        model="test",
    )

    second_id, _ = governor.before_model_call(
        [segment],
        model="test",
        provider="test-provider",
    )
    assert second_id != first_id
    assert second_id.startswith(first_id) or second_id.startswith(f"{first_id}:")
    governor.record_model_call(
        second_id,
        usage={"input_tokens": 30, "output_tokens": 15},
        provider="test-provider",
        model="test",
    )

    third_id, _ = governor.before_model_call(
        [segment],
        model="test",
        provider="test-provider",
    )
    assert third_id not in {first_id, second_id}


def test_released_model_call_id_is_not_reused_for_later_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    governor = ContextCostGovernor(
        session_id="session-release",
        settings=settings(
            mana_context_governor_mode="observe",
            mana_context_unknown_model_context_window=8_000,
            mana_context_unknown_model_max_output_tokens=1_000,
        ),
    )
    register_priced_test_model(
        governor,
        context_window=8_000,
        max_output_tokens=1_000,
    )
    governor.set_execution_identity(
        turn_id="turn-release",
        task_id="task-release",
        step_id="after_routing",
    )
    segment = ContextSegment("user", "prompt", 2, protected=True, source_id="user:1")
    first_id, _ = governor.before_model_call([segment], model="test", provider="test-provider")
    governor.release_reservation(first_id, reason="provider failed before usage")
    second_id, _ = governor.before_model_call([segment], model="test", provider="test-provider")
    assert second_id != first_id


def test_parallel_model_reservations_cannot_spend_the_same_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    governor = ContextCostGovernor(
        session_id="s",
        repository_id="r",
        workspace_id="w",
        settings=settings(
            mana_context_governor_mode="enforce",
            mana_routing_task_token_budget=130,
            mana_context_response_reserve_ratio=0.12,
            mana_context_unknown_model_context_window=1_000,
            mana_context_unknown_model_max_output_tokens=200,
        ),
    )
    register_priced_test_model(
        governor,
        context_window=1_000,
        max_output_tokens=200,
    )
    segment = ContextSegment("user", "constraint", 1, protected=True, source_id="user:1")
    first, _ = governor.before_model_call([segment], model="test", context_window=1_000)
    second, _ = governor.before_model_call([segment], model="test", context_window=1_000)
    with pytest.raises(ContextBudgetExceeded):
        governor.before_model_call([segment], model="test", context_window=1_000)
    snapshot = governor.observability_snapshot()
    assert snapshot["budget_reserved"]["tokens"] > 0
    assert snapshot["context_manifests"] == 3
    governor.release_reservation(first, reason="provider failed before usage")
    governor.release_reservation(second, reason="provider failed before usage")
