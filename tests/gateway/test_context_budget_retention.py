"""Regression tests for Context Budget Compaction and Bounded Routing Capsule."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mana_agent.config.settings import Settings
from mana_agent.context_cost.models import (
    AccountingSnapshot,
    BudgetSnapshot,
    ContextBreakdown,
    ContextBudget,
    ContextBudgetExceeded,
    GovernorDecision,
)
from mana_agent.execution_supervisor.models import ExecutionState, TaskRecord
from mana_agent.gateway import (
    AgentChatGateway,
    CompactedRoutingContext,
    ContextCompactor,
    ContextComponentBreakdown,
    EntryRouteContext,
    EntryRouteRegistry,
    EntryRouter,
    EntryRoutingDecision,
    EntryRoutingError,
    ExecutionRecoveryState,
    IdentitySessionRelationship,
    PreviousTurnPointers,
    RouteAvailability,
    RouteRegistration,
    RoutingExecutionEnvelope,
    build_routing_execution_envelope,
)
from mana_agent.gateway.context_compactor import default_accounting_snapshot
from mana_agent.gateway.entry_routing import ENTRY_ROUTER_PROMPT


class _MockRoutingModel:
    def __init__(self, route: str = "conversation") -> None:
        self.route = route
        self.invoked_messages: list[list[Any]] = []

    def with_structured_output(self, schema: Any, *, method: str = "json_schema", strict: bool = True):
        return self

    def invoke(self, messages: list[Any], **_kwargs: Any) -> Any:
        self.invoked_messages.append(messages)
        return SimpleNamespace(
            content=json.dumps({
                "route": self.route,
                "confidence": 0.95,
                "reason": "Test routing decision",
                "required_sources": ["none"],
                "target_urls": [],
                "requires_live_data": False,
                "reason_code": "TEST_ROUTE",
                "error_code": "",
                "reuse_active_route": False,
                "artifact_family": "",
                "automation_operation": "",
                "runtime_capability_change": False,
            })
        )


def _build_registry() -> EntryRouteRegistry:
    reg = EntryRouteRegistry()
    for name in ("conversation", "coding", "search", "mcp", "gmail", "artifact", "unsupported", "capability_error"):
        reg.register(RouteRegistration(name, f"{name} route", lambda: RouteAvailability(True)))  # type: ignore[arg-type]
    return reg


def test_reproduce_context_budget_deficit_and_token_breakdown() -> None:
    """Reproduce context_limit_deficit:13159 with breakdown of oversized components."""
    compactor = ContextCompactor()

    # Build a simulated oversized historical context (50+ tasks, large lane states, artifact evidence)
    oversized_candidates = [
        {
            "task_id": f"task_{i}",
            "normalized_intent": f"Analyze repository architectural layers and generate complete specification for module {i} with dependency graphs and verification reports " * 5,
            "state": "completed",
            "contract": {"completion_criteria": ["all files exist", "all tests pass", "type checks clean"] * 10},
            "constraints": ["no fallback routing", "strict model decision", "bounded token budget"] * 10,
        }
        for i in range(60)
    ]

    lane_states = {
        f"lane_{i}": {
            "state": "completed",
            "active_task_id": f"task_{i}",
            "logs": ["DEBUG 2026-08-30: Executed step with status=ok, token_usage=512, model=gpt-4" * 10] * 5,
        }
        for i in range(10)
    }

    envelope = build_routing_execution_envelope(
        user_request="Summarize project architecture",
        identity=IdentitySessionRelationship(
            authenticated_user_id="user_1",
            session_id="session_1",
            conversation_id="session_1",
            turn_id="turn_1",
        ),
        execution_state=ExecutionRecoveryState(
            all_recovery_candidates=tuple(oversized_candidates),
            recoverable_task_candidates=tuple(oversized_candidates[:10]),
            lane_states=lane_states,
        ),
        accounting_snapshot=default_accounting_snapshot("turn_1", "turn_1"),
        capabilities_and_tools=tuple(
            {"name": f"tool_{i}", "description": f"Tool description for tool {i} with extensive schema " * 10}
            for i in range(30)
        ),
    )

    context = EntryRouteContext(
        session_id="session_1",
        conversation_id="session_1",
        turn_id="turn_1",
        artifact_evidence={"references": [{"filename": f"file_{i}.py", "family": "code"} for i in range(50)]},
        memory_task_candidates=tuple({"task_id": f"task_{i}", "intent": "memory intent"} for i in range(30)),
        envelope=envelope,
    )

    routes = _build_registry().snapshot()

    breakdown = compactor.calculate_raw_breakdown(
        user_prompt="Summarize project architecture",
        system_prompt=ENTRY_ROUTER_PROMPT,
        context=context,
        envelope=envelope,
        routes=routes,
    )

    assert breakdown.task_candidates > 5000
    assert breakdown.total_tokens > 13000
    assert breakdown.user_request > 0
    assert breakdown.system_prompt > 0

    # Test compaction reduces the deficit
    compacted = compactor.compact_routing_context(
        user_prompt="Summarize project architecture",
        system_prompt=ENTRY_ROUTER_PROMPT,
        context=context,
        envelope=envelope,
        routes=routes,
        context_window=16384,
    )

    assert compacted.context_tokens_saved > 5000
    assert compacted.compacted_context_tokens < compacted.raw_context_tokens
    assert compacted.is_valid is True
    assert compacted.deficit == 0


def test_thousands_of_old_log_events_do_not_enter_routing_context() -> None:
    """Prove operational logs remain available for diagnostics but are excluded from model prompt."""
    compactor = ContextCompactor()

    # Raw state containing heavy logs
    lane_states = {
        "coding": {
            "state": "idle",
            "logs": ["2026-08-30 [DEBUG] Tool call execution trace step %d: result=success" % i for i in range(1000)],
        }
    }

    envelope = build_routing_execution_envelope(
        user_request="Edit main.py",
        identity=IdentitySessionRelationship(
            authenticated_user_id="user_1",
            session_id="session_1",
            conversation_id="session_1",
            turn_id="turn_1",
        ),
        execution_state=ExecutionRecoveryState(
            lane_states=lane_states,
        ),
        accounting_snapshot=default_accounting_snapshot("turn_1", "turn_1"),
    )

    context = EntryRouteContext(
        session_id="session_1",
        conversation_id="session_1",
        turn_id="turn_1",
        envelope=envelope,
    )

    compacted = compactor.compact_routing_context(
        user_prompt="Edit main.py",
        system_prompt=ENTRY_ROUTER_PROMPT,
        context=context,
        envelope=envelope,
        routes=_build_registry().snapshot(),
    )

    assert compacted.logs_excluded_tokens > 0
    # Bounded lane states must not contain the raw logs list
    bounded_lanes = compacted.bounded_envelope.execution_state.lane_states
    assert "logs" not in bounded_lanes.get("coding", {})


def test_repeated_provider_events_and_tasks_are_deduplicated() -> None:
    """Repeated task intentions and lifecycle events are deduplicated during compaction."""
    compactor = ContextCompactor()

    # Create duplicate candidate records
    duplicate_candidates = [
        {"task_id": "t1", "normalized_intent": "Fix memory leak in websocket connection", "state": "completed"},
        {"task_id": "t2", "normalized_intent": "Fix memory leak in websocket connection", "state": "completed"},
        {"task_id": "t3", "normalized_intent": "Fix memory leak in websocket connection", "state": "completed"},
        {"task_id": "t4", "normalized_intent": "Add user authentication unit test", "state": "completed"},
    ]

    compacted_candidates = compactor.compact_recovery_candidates(duplicate_candidates, max_candidates=10)

    # 3 duplicates should collapse into 1, plus the unique 1 = 2 total
    assert len(compacted_candidates) == 2
    intents = [c["normalized_intent"] for c in compacted_candidates]
    assert intents.count("Fix memory leak in websocket connection") == 1


def test_user_input_and_response_reserve_prioritized() -> None:
    """Current user input and routing response always receive reserved capacity before historical context."""
    compactor = ContextCompactor()

    user_prompt = "Refactor the authentication module to support OAuth2 providers"
    compacted = compactor.compact_routing_context(
        user_prompt=user_prompt,
        system_prompt=ENTRY_ROUTER_PROMPT,
        context=EntryRouteContext(session_id="s", conversation_id="s", turn_id="t"),
        routes=_build_registry().snapshot(),
        context_window=8192,
        response_reserve_tokens=512,
    )

    assert compacted.is_valid is True
    assert compacted.breakdown.user_request > 0
    assert compacted.compacted_context_tokens <= 8192



def test_pre_provider_routing_budget_failure_not_classified_as_provider_failure() -> None:
    """When context cannot fit even after compaction, raise pre-provider context_budget_blocked with diagnostic details."""
    model = _MockRoutingModel()
    registry = _build_registry()
    router = EntryRouter(llm=model, registry=registry)

    # Create a user prompt that exceeds context window limit (e.g. 50,000 characters for small window)
    huge_prompt = "A" * 100_000

    context = EntryRouteContext(
        session_id="session_1",
        conversation_id="session_1",
        turn_id="turn_1",
    )

    with pytest.raises(EntryRoutingError) as exc_info:
        router.route(user_prompt=huge_prompt, context=context)

    exc = exc_info.value
    assert exc.code == "context_budget_blocked"
    assert exc.phase == "entry_route"
    assert exc.provider_call_executed is False
    assert "context_limit_deficit:" in str(exc)
    assert exc.details.get("phase") == "entry_route"
    assert exc.details.get("provider_call_executed") is False
    assert len(model.invoked_messages) == 0  # Provider call was never executed


def test_context_compactor_metrics() -> None:
    """Verify all 9 metrics are accurately computed by ContextCompactor."""
    compactor = ContextCompactor()

    oversized_candidates = [
        {"task_id": f"t_{i}", "normalized_intent": f"Task intent {i} " * 20, "state": "completed"}
        for i in range(20)
    ]

    lane_states = {
        "coding": {
            "state": "running",
            "logs": ["log line %d" % i for i in range(100)],
        }
    }

    envelope = build_routing_execution_envelope(
        user_request="What is the next task?",
        identity=IdentitySessionRelationship(
            authenticated_user_id="user_1",
            session_id="session_1",
            conversation_id="session_1",
            turn_id="turn_1",
        ),
        execution_state=ExecutionRecoveryState(
            all_recovery_candidates=tuple(oversized_candidates),
            lane_states=lane_states,
        ),
        accounting_snapshot=default_accounting_snapshot("turn_1", "turn_1"),
    )

    context = EntryRouteContext(
        session_id="session_1",
        conversation_id="session_1",
        turn_id="turn_1",
        envelope=envelope,
    )

    metrics = compactor.compact_routing_context(
        user_prompt="What is the next task?",
        system_prompt=ENTRY_ROUTER_PROMPT,
        context=context,
        envelope=envelope,
        routes=_build_registry().snapshot(),
        context_window=8000,
        stale_records_pruned=3,
        workspace_records_pruned=1,
        repository_records_compacted=2,
    )

    assert metrics.raw_context_tokens > 0
    assert metrics.compacted_context_tokens > 0
    assert metrics.context_tokens_saved >= 0
    assert metrics.logs_excluded_tokens > 0
    assert metrics.stale_records_pruned == 3
    assert metrics.workspace_records_pruned == 1
    assert metrics.repository_records_compacted == 2
    assert metrics.routing_context_deficit_before_compaction >= 0
    assert metrics.routing_context_deficit_after_compaction == 0
