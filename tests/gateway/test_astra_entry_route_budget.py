"""Regression tests for Astra entry route token budgeting, compaction, and session recovery."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from mana_agent.config.model_catalog import maintained_token_limits
from mana_agent.config.settings import Settings
from mana_agent.context_cost.models import AccountingSnapshot
from mana_agent.context_cost.profiles import ModelIdentity, ModelTokenProfileResolver

from mana_agent.gateway import (
    ApprovalState,
    CompactedRoutingContext,
    ContextCompactor,
    ContextComponentBreakdown,
    EntryRouteContext,
    EntryRouteRegistry,
    EntryRouter,
    EntryRoutingError,
    ExecutionRecoveryState,
    IdentitySessionRelationship,
    ModelCandidateCapacity,
    PreviousTurnPointers,
    RouteAvailability,
    RouteRegistration,
    build_routing_execution_envelope,
)
from mana_agent.gateway.context_compactor import default_accounting_snapshot
from mana_agent.gateway.entry_routing import ENTRY_ROUTER_PROMPT
from mana_agent.model_routing.profiles import configured_profiles


class _MockAstraRoutingModel:
    """Mock routing LLM configured as Astra (GPT-6 Astra)."""

    def __init__(self, route: str = "conversation", model_name: str = "gpt-6-astra") -> None:
        self.route = route
        self.model_name = model_name
        self.model = model_name
        self.provider = "openai"
        self.invoked_messages: list[list[Any]] = []

    def with_structured_output(self, schema: Any, *, method: str = "json_schema", strict: bool = True):
        return self

    def invoke(self, messages: list[Any], **_kwargs: Any) -> Any:
        self.invoked_messages.append(messages)
        sources = ["repository"] if self.route in {"coding", "repository"} else ["none"]
        return SimpleNamespace(
            content=json.dumps({
                "route": self.route,
                "confidence": 0.98,
                "reason": "Astra routing model decision",
                "required_sources": sources,
                "target_urls": [],
                "requires_live_data": False,
                "reason_code": "ASTRA_ROUTE",
                "error_code": "",
                "reuse_active_route": False,
                "artifact_family": "",
                "automation_operation": "",
                "runtime_capability_change": False,
            })
        )


def _build_test_registry() -> EntryRouteRegistry:
    reg = EntryRouteRegistry()
    for name in ("conversation", "coding", "repository", "search", "mcp", "gmail", "artifact", "unsupported", "capability_error"):
        reg.register(RouteRegistration(name, f"{name} route", lambda: RouteAvailability(True)))  # type: ignore[arg-type]
    return reg


def test_astra_model_limits_and_profiles_resolution() -> None:
    """Verify Astra (GPT-6 Astra) context limit (1,050,000) and max output tokens (128,000) resolution."""
    # Maintained token limits catalog
    limits_astra = maintained_token_limits("openai", "gpt-6-astra")
    assert limits_astra == (1_050_000, 128_000)

    limits_gpt6 = maintained_token_limits("openai", "gpt-6")
    assert limits_gpt6 == (1_050_000, 128_000)

    limits_short = maintained_token_limits("openai", "astra")
    assert limits_short == (1_050_000, 128_000)

    # ModelTokenProfileResolver resolution
    resolver = ModelTokenProfileResolver()
    profile = resolver.resolve(ModelIdentity("openai", "gpt-6-astra"))
    assert profile.context_window == 1_050_000
    assert profile.max_output_tokens == 128_000
    assert profile.confidence == "high"

    # Configured profiles resolution
    profiles = configured_profiles([{"model_id": "gpt-6-astra", "provider": "openai", "supported_roles": ["coding"]}])
    assert len(profiles) == 1
    assert profiles[0].context_window == 1_050_000
    assert profiles[0].max_output_tokens == 128_000




def test_reproduce_36k_deficit_scenario_and_multi_pass_compaction() -> None:
    """Reproduce the ~36k deficit condition and prove multi-pass deterministic compaction resolves it."""
    compactor = ContextCompactor()

    # Create heavy candidates to construct a ~52k token payload
    oversized_candidates = [
        {
            "task_id": f"task_{i}",
            "normalized_intent": (
                f"Analyze AST structure and dependencies for module {i} with complete cross-file references "
                f"and generate typed verification reports with boundary constraints and contract validations "
            ) * 4,
            "state": "completed" if i > 2 else "running",
            "checkpoint_id": "chk_active_1" if i == 0 else None,
            "contract": {"completion_criteria": ["type checks clean", "verification commands executed"] * 6},
        }
        for i in range(50)
    ]

    lane_states = {
        f"lane_{i}": {
            "state": "running" if i == 0 else "completed",
            "active_task_id": f"task_{i}",
            "logs": ["DEBUG 2026-09-04 [tool-step]: execution trace step with tokens=512, status=ok" * 8] * 4,
        }
        for i in range(8)
    }

    envelope = build_routing_execution_envelope(
        user_request="Refactor gateway entry routing to support Astra model capabilities",
        identity=IdentitySessionRelationship(
            authenticated_user_id="user_astra",
            session_id="session_astra",
            conversation_id="session_astra",
            turn_id="turn_astra_1",
        ),
        execution_state=ExecutionRecoveryState(
            all_recovery_candidates=tuple(oversized_candidates),
            recoverable_task_candidates=tuple(oversized_candidates[:8]),
            lane_states=lane_states,
            pending_checkpoint_id="chk_active_1",
        ),
        accounting_snapshot=default_accounting_snapshot("turn_astra_1", "turn_astra_1"),
        model_candidates=(
            ModelCandidateCapacity(
                model_id="gpt-6-astra",
                provider="openai",
                context_window=1_050_000,
                max_output_tokens=128_000,
                supported_roles=("head_decision", "general_decision"),
            ),
        ),
        capabilities_and_tools=tuple(
            {"name": f"tool_{i}", "description": f"Tool description for {i} with JSON argument schema specifications " * 8}
            for i in range(25)
        ),
        approval_state=ApprovalState(
            pending_server_approvals=({"approval_id": "app_1", "action": "deploy"},),
        ),
    )

    context = EntryRouteContext(
        session_id="session_astra",
        conversation_id="session_astra",
        turn_id="turn_astra_1",
        conversation_summary="Active session working on gateway entry routing and token budgeting.",
        artifact_evidence={"references": [{"filename": f"mod_{i}.py", "family": "code"} for i in range(40)]},
        memory_task_candidates=tuple({"task_id": f"task_{i}", "intent": "memory intent"} for i in range(20)),
        envelope=envelope,
    )

    routes = _build_test_registry().snapshot()

    # 1. Verify raw breakdown calculation: non-redundant and captures heavy candidates
    breakdown = compactor.calculate_raw_breakdown(
        user_prompt="Refactor gateway entry routing to support Astra model capabilities",
        system_prompt=ENTRY_ROUTER_PROMPT,
        context=context,
        envelope=envelope,
        routes=routes,
    )

    assert breakdown.task_candidates > 10_000
    assert breakdown.total_tokens > 25_000
    assert breakdown.logs_and_traces > 0

    # 2. When measured against a constrained window (e.g. 16,384), verify compaction saves context and fits
    compacted_constrained = compactor.compact_routing_context(
        user_prompt="Refactor gateway entry routing to support Astra model capabilities",
        system_prompt=ENTRY_ROUTER_PROMPT,
        context=context,
        envelope=envelope,
        routes=routes,
        context_window=16_384,
    )

    assert compacted_constrained.compacted_context_tokens < compacted_constrained.raw_context_tokens
    assert compacted_constrained.context_tokens_saved > 5_000
    assert compacted_constrained.is_valid is True
    assert compacted_constrained.deficit == 0

    # 3. Critical execution state MUST be preserved
    bounded_es = compacted_constrained.bounded_envelope.execution_state
    assert bounded_es.pending_checkpoint_id == "chk_active_1"
    assert compacted_constrained.bounded_envelope.user_request == "Refactor gateway entry routing to support Astra model capabilities"
    assert len(compacted_constrained.bounded_envelope.approval_state.pending_server_approvals) == 1

    # 4. Under Astra's actual 1,050,000 context window, verify it fits easily
    compacted_astra = compactor.compact_routing_context(
        user_prompt="Refactor gateway entry routing to support Astra model capabilities",
        system_prompt=ENTRY_ROUTER_PROMPT,
        context=context,
        envelope=envelope,
        routes=routes,
        context_window=1_050_000,
    )

    assert compacted_astra.is_valid is True
    assert compacted_astra.deficit == 0
    assert compacted_astra.compacted_context_tokens < 1_050_000


def test_entry_router_executes_with_astra_model() -> None:
    """Verify EntryRouter uses Astra context limit and executes routing without budget block."""
    model = _MockAstraRoutingModel(route="coding", model_name="gpt-6-astra")
    registry = _build_test_registry()
    compactor = ContextCompactor()
    router = EntryRouter(llm=model, registry=registry, compactor=compactor)

    # Moderate context payload that exceeds 16k but easily fits within Astra 1.05M
    oversized_candidates = [
        {"task_id": f"t_{i}", "normalized_intent": f"Refactor module {i} " * 20, "state": "completed"}
        for i in range(30)
    ]

    envelope = build_routing_execution_envelope(
        user_request="Add unit tests for routing governor",
        identity=IdentitySessionRelationship(
            authenticated_user_id="user_1",
            session_id="session_1",
            conversation_id="session_1",
            turn_id="turn_1",
        ),
        execution_state=ExecutionRecoveryState(
            all_recovery_candidates=tuple(oversized_candidates),
        ),
        accounting_snapshot=default_accounting_snapshot("turn_1", "turn_1"),
        model_candidates=(
            ModelCandidateCapacity(
                model_id="gpt-6-astra",
                provider="openai",
                context_window=1_050_000,
                max_output_tokens=128_000,
                supported_roles=("head_decision",),
            ),
        ),
    )

    context = EntryRouteContext(
        session_id="session_1",
        conversation_id="session_1",
        turn_id="turn_1",
        envelope=envelope,
    )

    decision = router.route(
        user_prompt="Add unit tests for routing governor",
        context=context,
    )

    assert decision.route == "coding"
    assert len(model.invoked_messages) == 1  # Provider call executed successfully


def test_genuinely_impossible_context_raises_structured_diagnostics() -> None:
    """Verify impossible context raises EntryRoutingError with complete structured diagnostics."""
    model = _MockAstraRoutingModel()
    registry = _build_test_registry()
    compactor = ContextCompactor()
    router = EntryRouter(llm=model, registry=registry, compactor=compactor)

    # Prompt exceeding 1.05M context window
    impossible_prompt = "Z" * 5_000_000

    context = EntryRouteContext(
        session_id="session_impossible",
        conversation_id="session_impossible",
        turn_id="turn_impossible",
    )

    with pytest.raises(EntryRoutingError) as exc_info:
        router.route(user_prompt=impossible_prompt, context=context)

    exc = exc_info.value
    assert exc.code == "context_budget_blocked"
    assert exc.phase == "entry_route"
    assert exc.provider_call_executed is False
    assert "context_limit_deficit:" in str(exc)

    details = exc.details
    assert details["phase"] == "entry_route"
    assert details["provider_call_executed"] is False
    assert details["context_limit"] > 0
    assert details["input_tokens"] > details["context_limit"]
    assert details["remaining_deficit"] > 0
    assert "budget_equation" in details
    eq = details["budget_equation"]
    assert "model_context_limit" in eq
    assert "resulting_deficit" in eq

    # Provider call was never executed
    assert len(model.invoked_messages) == 0


def test_non_redundant_component_accounting() -> None:
    """Verify ContextComponentBreakdown does not double count logs, candidates, or envelope fields."""
    compactor = ContextCompactor()

    lane_states = {
        "lane_0": {
            "state": "running",
            "logs": ["TRACE log line %d" % i for i in range(100)],
        }
    }

    envelope = build_routing_execution_envelope(
        user_request="Check system health",
        identity=IdentitySessionRelationship(
            authenticated_user_id="user_1",
            session_id="s1",
            conversation_id="s1",
            turn_id="t1",
        ),
        execution_state=ExecutionRecoveryState(
            lane_states=lane_states,
            all_recovery_candidates=({"task_id": "t1", "normalized_intent": "Task 1", "state": "completed"},),
        ),
        accounting_snapshot=default_accounting_snapshot("t1", "t1"),
    )

    context = EntryRouteContext(
        session_id="s1",
        conversation_id="s1",
        turn_id="t1",
        memory_task_candidates=({"task_id": "m1", "intent": "Memory 1"},),
        envelope=envelope,
    )

    breakdown = compactor.calculate_raw_breakdown(
        user_prompt="Check system health",
        system_prompt=ENTRY_ROUTER_PROMPT,
        context=context,
        envelope=envelope,
        routes=_build_test_registry().snapshot(),
    )

    # Component sum matches total_tokens exactly
    component_sum = (
        breakdown.user_request
        + breakdown.system_prompt
        + breakdown.route_availability
        + breakdown.execution_state
        + breakdown.task_candidates
        + breakdown.artifact_evidence
        + breakdown.memory_candidates
        + breakdown.accounting
        + breakdown.tools_and_capabilities
        + breakdown.logs_and_traces
        + breakdown.other
    )
    assert breakdown.total_tokens == component_sum
