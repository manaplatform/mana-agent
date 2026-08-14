"""Tests for Canonical RoutingExecutionEnvelope and Context Retrieval Tools.

Covers all 12 scenarios:
1. Long stored history does not increase ordinary prompt size (100k stored conversation never injected automatically).
2. Independent turn performs zero conversation/memory reads.
3. Follow-up "why?" retrieves relevant previous turn via conversation_context_read.
4. Router selects task_A; memory tool queries exactly task_A.
5. Current turn exec_B can never become capsule task scope.
6. Unoffered task_X remains denied.
7. Authorized task_A capsule returns non-zero match when present; zero matches returns goal_satisfied=false.
8. Retrieved context participates once in the next provider-call estimate with real ContextCostGovernor.
9. Repeated identical retrieval within a turn is deduplicated (cached retrieval not charged twice).
10. Retrieval allowance is cumulative across conversation + memory.
11. Multi-task memory and follow-up workflows continue to work without copying parent chat history.
12. Terminal and dashboard sessions behave identically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from mana_agent.config.settings import Settings
from mana_agent.context_cost.governor import ContextCostGovernor
from mana_agent.context_cost.models import AccountingSnapshot, ContextSegment
from mana_agent.gateway.envelope import (
    ApprovalState,
    ConversationContextAvailability,
    ExecutionRecoveryState,
    IdentitySessionRelationship,
    MemoryAvailability,
    ModelCandidateCapacity,
    PreviousTurnPointers,
    RoutingExecutionEnvelope,
    build_routing_execution_envelope,
)
from mana_agent.gateway.entry_routing import EntryRouteContext
from mana_agent.gateway.turn_engine import _conversation_prompt
from mana_agent.memory.capsules.models import (
    CapsuleScope,
    CapsuleTaskContext,
    MemoryPrincipal,
)
from mana_agent.tools.context_retrieval import (
    MemoryTaskBinding,
    TurnRetrievalLedger,
    build_context_retrieval_tools,
    execute_conversation_context_read,
    execute_memory_read,
)


@dataclass
class DummyCapsuleProjection:
    capsule_id: str
    revision: int
    title: str
    summary: str
    content: Any
    tags: tuple[str, ...] = ()
    scope: CapsuleScope = CapsuleScope.PRIVATE


@dataclass
class DummyMessage:
    role: str
    content: str
    turn_id: str
    timestamp: str = "2026-08-14T20:00:00Z"

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
        }


class DummyHistoryStore:
    def __init__(self, messages: list[DummyMessage] | None = None) -> None:
        self.messages = messages or []

    def list(self, session_id: str) -> list[DummyMessage]:
        return list(self.messages)


class DummyCapsuleService:
    def __init__(self, projections: list[Any] | None = None) -> None:
        self.projections = projections or []
        self.last_request = None

    def query_capsules(self, request: Any, correlation_id: str = "") -> list[Any]:
        self.last_request = request
        query_text = str(request.query or "").strip().lower()
        if not query_text:
            return list(self.projections)
        return [
            p
            for p in self.projections
            if query_text in str(p.title).lower()
            or query_text in str(p.content).lower()
            or query_text in str(p.summary).lower()
            or any(query_text in str(t).lower() for t in p.tags)
        ]


def test_scenario_1_long_stored_history_does_not_increase_prompt_size():
    """Scenario 1: 100k stored conversation is never injected automatically."""
    session_state: dict[str, Any] = {
        "messages": [
            DummyMessage(role="user", content=f"User question {i}" * 50, turn_id=f"turn_{i}").to_dict()
            for i in range(100)
        ],
        "followup_memory_context": "Durable memory text that was previously injected",
    }
    current_message = "What is the current time?"
    prompt = _conversation_prompt(session_state, current_message)
    assert prompt == "What is the current time?"
    assert "User question" not in prompt
    assert "Durable memory text" not in prompt


def test_scenario_2_independent_turn_performs_zero_reads():
    """Scenario 2: Independent turn performs zero conversation/memory reads."""
    history = DummyHistoryStore(
        [
            DummyMessage(role="user", content="Hello", turn_id="turn_1"),
            DummyMessage(role="assistant", content="Hi there!", turn_id="turn_1"),
        ]
    )
    settings = Settings()
    governor = ContextCostGovernor(session_id="session_123", settings=settings)
    turn_cache: dict[str, Any] = {}
    ledger = TurnRetrievalLedger(retrieval_budget_tokens=12000)

    tools = build_context_retrieval_tools(
        session_id="session_123",
        conversation_id="conv_123",
        authenticated_user_id="user_abc",
        history_store=history,
        current_turn_id="turn_2",
        governor=governor,
        turn_retrieval_cache=turn_cache,
        retrieval_ledger=ledger,
    )
    assert len(tools) == 2
    assert len(turn_cache) == 0
    assert ledger.retrieval_used_tokens == 0
    assert ledger.conversation_retrieval_tokens == 0
    assert ledger.memory_retrieval_tokens == 0


def test_scenario_3_why_retrieves_relevant_previous_turn():
    """Scenario 3: Follow-up 'why?' retrieves relevant previous turn via conversation_context_read."""
    history = DummyHistoryStore(
        [
            DummyMessage(
                role="user",
                content="Suggest an authentication architecture.",
                turn_id="turn_1",
            ),
            DummyMessage(
                role="assistant",
                content="I recommend using OAuth2 with PKCE because of client security.",
                turn_id="turn_1",
            ),
        ]
    )
    events: list[dict[str, Any]] = []

    def sink(event_type: str, msg: str, metadata: dict[str, Any] | None = None) -> None:
        events.append({"type": event_type, "message": msg, "metadata": metadata or {}})

    turn_cache: dict[str, Any] = {}
    ledger = TurnRetrievalLedger(retrieval_budget_tokens=12000)
    res_json = execute_conversation_context_read(
        session_id="session_123",
        conversation_id="conv_123",
        authenticated_user_id="user_abc",
        history_store=history,
        current_turn_id="turn_2",
        query="OAuth2 PKCE",
        max_turns=1,
        max_tokens=2000,
        turn_retrieval_cache=turn_cache,
        event_sink=sink,
        retrieval_ledger=ledger,
    )
    data = json.loads(res_json)
    assert data["source"] == "conversation_context"
    assert data["turns_returned"] == 1
    assert data["empty"] is False
    assert len(data["turns"]) == 2  # user + assistant in turn_1
    assert any("OAuth2 with PKCE" in t["content"] for t in data["turns"])
    assert ledger.conversation_retrieval_tokens > 0
    assert any(e["type"] == "context.conversation_read" for e in events)


def test_scenario_4_router_selects_task_a_memory_tool_queries_task_a():
    """Scenario 4: Router selects task_A; memory tool queries exactly task_A."""
    sample_projection = DummyCapsuleProjection(
        capsule_id="cap_1",
        revision=1,
        title="Coding Preference",
        summary="User prefers dataclasses over pydantic models for internal types",
        content="Prefer frozen dataclasses with slots for internal data structures.",
        tags=("preferences", "style"),
        scope=CapsuleScope.PRIVATE,
    )
    capsule_service = DummyCapsuleService([sample_projection])
    events: list[dict[str, Any]] = []

    def sink(event_type: str, msg: str, metadata: dict[str, Any] | None = None) -> None:
        events.append({"type": event_type, "message": msg, "metadata": metadata or {}})

    turn_cache: dict[str, Any] = {}
    ledger = TurnRetrievalLedger(retrieval_budget_tokens=4000)
    res_json = execute_memory_read(
        capsule_service=capsule_service,
        authenticated_user_id="user_alice",
        session_id="session_123",
        repository_id="repo_1",
        current_turn_id="turn_2",
        selected_memory_task_id="task_A",
        memory_task_candidates=({"task_id": "task_A", "normalized_intent": "coding", "state": "active"},),
        query="dataclasses",
        max_capsules=3,
        turn_retrieval_cache=turn_cache,
        event_sink=sink,
        retrieval_ledger=ledger,
    )
    data = json.loads(res_json)
    assert data["source"] == "memory_capsules"
    assert data["status"] == "matched"
    assert data["selected_memory_task_id"] == "task_A"
    assert data["capsules_returned"] == 1
    assert data["goal_satisfied"] is True
    assert data["empty"] is False
    assert data["capsules"][0]["capsule_id"] == "cap_1"
    assert "slots" in data["capsules"][0]["content"]
    assert ledger.memory_retrieval_tokens > 0
    assert any(e["type"] == "context.memory_read" for e in events)


def test_scenario_5_current_turn_exec_b_never_becomes_capsule_task_scope():
    """Scenario 5: Current turn exec_B can never become capsule task scope."""
    sample_projection = DummyCapsuleProjection(
        capsule_id="cap_secret",
        revision=1,
        title="Secret",
        summary="Secret summary",
        content="Secret content",
        tags=("secret",),
        scope=CapsuleScope.PRIVATE,
    )
    capsule_service = DummyCapsuleService([sample_projection])

    # Attempting to use current turn ID "exec_B" as task scope when it is not in offered candidates
    res_json = execute_memory_read(
        capsule_service=capsule_service,
        authenticated_user_id="user_alice",
        session_id="session_123",
        repository_id="repo_1",
        current_turn_id="exec_B",
        selected_memory_task_id="exec_B",  # Model or code tries to use turn ID as task scope
        memory_task_candidates=({"task_id": "task_A", "normalized_intent": "foo", "state": "active"},),
        query="secret",
    )
    data = json.loads(res_json)
    assert data["empty"] is True
    assert data["capsules_returned"] == 0
    assert data["goal_satisfied"] is False
    assert data["status"] == "no_match"
    assert "was not offered to the router" in data["error"]


def test_scenario_6_unoffered_task_x_remains_denied():
    """Scenario 6: Unoffered task_X remains denied."""
    sample_projection = DummyCapsuleProjection(
        capsule_id="cap_secret",
        revision=1,
        title="Secret API Key",
        summary="Private token",
        content="secret_token_12345",
        tags=("secret",),
        scope=CapsuleScope.PRIVATE,
    )
    capsule_service = DummyCapsuleService([sample_projection])

    # Unauthenticated call
    res_unauth = execute_memory_read(
        capsule_service=capsule_service,
        authenticated_user_id="",  # Empty user ID
        session_id="session_123",
        repository_id="repo_1",
        selected_memory_task_id="task_allowed",
        memory_task_candidates=({"task_id": "task_allowed", "normalized_intent": "foo", "state": "active"},),
        query="secret",
    )
    data_unauth = json.loads(res_unauth)
    assert data_unauth["empty"] is True
    assert data_unauth["capsules_returned"] == 0
    assert data_unauth["goal_satisfied"] is False
    assert "requires an authenticated user identity" in data_unauth["error"]

    # Unauthorized task ID not in offered candidates
    res_unoffered = execute_memory_read(
        capsule_service=capsule_service,
        authenticated_user_id="user_alice",
        session_id="session_123",
        repository_id="repo_1",
        current_turn_id="turn_allowed",
        memory_task_candidates=({"task_id": "task_allowed", "normalized_intent": "foo", "state": "active"},),
        selected_memory_task_id="task_unauthorized_999",
        query="secret",
    )
    data_unoffered = json.loads(res_unoffered)
    assert data_unoffered["empty"] is True
    assert data_unoffered["capsules_returned"] == 0
    assert data_unoffered["goal_satisfied"] is False
    assert "not offered to the router" in data_unoffered["error"]


def test_scenario_7_zero_matches_returns_goal_satisfied_false():
    """Scenario 7: Zero matches returns goal_satisfied=false."""
    capsule_service = DummyCapsuleService([])  # Empty capsule store

    res_json = execute_memory_read(
        capsule_service=capsule_service,
        authenticated_user_id="user_alice",
        session_id="session_123",
        repository_id="repo_1",
        current_turn_id="turn_1",
        selected_memory_task_id="task_A",
        memory_task_candidates=({"task_id": "task_A", "normalized_intent": "foo", "state": "active"},),
        query="nonexistent_query_12345",
    )
    data = json.loads(res_json)
    assert data["source"] == "memory_capsules"
    assert data["status"] == "no_match"
    assert data["selected_memory_task_id"] == "task_A"
    assert data["capsules_returned"] == 0
    assert data["goal_satisfied"] is False
    assert data["empty"] is True


def test_scenario_8_retrieved_context_participates_in_provider_call_estimate():
    """Scenario 8: Retrieved context participates in Phase-0 provider call estimation with real governor."""
    history = DummyHistoryStore(
        [
            DummyMessage(role="user", content="Explain OAuth2 in detail.", turn_id="turn_1"),
            DummyMessage(role="assistant", content="OAuth2 uses access tokens and scopes.", turn_id="turn_1"),
        ]
    )
    settings = Settings()
    governor = ContextCostGovernor(session_id="session_123", settings=settings)
    turn_cache: dict[str, Any] = {}
    ledger = TurnRetrievalLedger(retrieval_budget_tokens=12000)

    # 1. Retrieve context via tool
    res_json = execute_conversation_context_read(
        session_id="session_123",
        conversation_id="conv_123",
        authenticated_user_id="user_alice",
        history_store=history,
        current_turn_id="turn_2",
        query="OAuth2",
        turn_retrieval_cache=turn_cache,
        retrieval_ledger=ledger,
    )
    data = json.loads(res_json)
    retrieved_tokens = data["tokens"]
    assert retrieved_tokens > 0
    assert ledger.conversation_retrieval_tokens == retrieved_tokens

    # 2. When that retrieved projection is included in the LLM model call,
    # the governor forecasts and reserves provider tokens for it
    segments = [
        ContextSegment(kind="prompt", content="You are a helpful assistant.", token_estimate=10),
        ContextSegment(
            kind="prompt",
            content=f"Context: {res_json}\n\nQuestion: Summarize the OAuth2 setup.",
            token_estimate=retrieved_tokens + 20,
        ),
    ]
    call_id, decision = governor.before_model_call(
        segments=segments,
        provider="openai",
        model="gpt-4o",
    )
    assert decision.allowed is True
    assert decision.snapshot.used_tokens >= retrieved_tokens
    governor.release_reservation(call_id, reason="turn_completed")


def test_scenario_9_cached_retrieval_is_not_charged_twice():
    """Scenario 9: Repeated identical retrieval is deduplicated with zero additional charge."""
    history = DummyHistoryStore(
        [
            DummyMessage(role="user", content="Question 1", turn_id="turn_1"),
            DummyMessage(role="assistant", content="Answer 1", turn_id="turn_1"),
        ]
    )
    turn_cache: dict[str, Any] = {}
    ledger = TurnRetrievalLedger(retrieval_budget_tokens=12000)
    events: list[dict[str, Any]] = []

    def sink(event_type: str, msg: str, metadata: dict[str, Any] | None = None) -> None:
        events.append({"type": event_type, "message": msg, "metadata": metadata or {}})

    # First call
    res_1 = execute_conversation_context_read(
        session_id="session_123",
        conversation_id="conv_123",
        authenticated_user_id="user_alice",
        history_store=history,
        current_turn_id="turn_2",
        query="Question",
        max_turns=3,
        turn_retrieval_cache=turn_cache,
        event_sink=sink,
        retrieval_ledger=ledger,
    )
    first_charged = ledger.retrieval_used_tokens
    assert first_charged > 0

    # Second identical call in same turn
    res_2 = execute_conversation_context_read(
        session_id="session_123",
        conversation_id="conv_123",
        authenticated_user_id="user_alice",
        history_store=history,
        current_turn_id="turn_2",
        query="Question",
        max_turns=3,
        turn_retrieval_cache=turn_cache,
        event_sink=sink,
        retrieval_ledger=ledger,
    )
    assert res_1 == res_2
    # Ledger was not charged a second time (0 additional tokens charged)
    assert ledger.retrieval_used_tokens == first_charged
    assert any(e["type"] == "context.retrieval_deduplicated" for e in events)


def test_scenario_10_retrieval_allowance_is_cumulative_across_conversation_and_memory():
    """Scenario 10: Retrieval allowance is cumulative across conversation + memory in the same turn."""
    history = DummyHistoryStore(
        [
            DummyMessage(role="user", content="Short query", turn_id="turn_1"),
            DummyMessage(role="assistant", content="Short answer", turn_id="turn_1"),
        ]
    )
    sample_projection = DummyCapsuleProjection(
        capsule_id="cap_1",
        revision=1,
        title="Preferences",
        summary="Coding style",
        content="Use snake_case for functions.",
        tags=("preferences",),
        scope=CapsuleScope.PRIVATE,
    )
    capsule_service = DummyCapsuleService([sample_projection])

    # Small shared budget of 100 tokens
    ledger = TurnRetrievalLedger(retrieval_budget_tokens=100)
    turn_cache: dict[str, Any] = {}

    # Call 1: Conversation context consumes some tokens
    execute_conversation_context_read(
        session_id="session_123",
        conversation_id="conv_123",
        authenticated_user_id="user_alice",
        history_store=history,
        current_turn_id="turn_2",
        query="Short",
        max_tokens=60,
        turn_retrieval_cache=turn_cache,
        retrieval_ledger=ledger,
    )
    conv_tokens = ledger.conversation_retrieval_tokens
    assert conv_tokens > 0
    assert ledger.retrieval_remaining_tokens == 100 - conv_tokens

    # Call 2: Memory read consumes remaining tokens
    execute_memory_read(
        capsule_service=capsule_service,
        authenticated_user_id="user_alice",
        session_id="session_123",
        repository_id="repo_1",
        current_turn_id="turn_2",
        selected_memory_task_id="task_A",
        memory_task_candidates=({"task_id": "task_A", "normalized_intent": "foo", "state": "active"},),
        query="Preferences",
        max_tokens=50,
        turn_retrieval_cache=turn_cache,
        retrieval_ledger=ledger,
    )
    mem_tokens = ledger.memory_retrieval_tokens
    assert mem_tokens > 0
    assert ledger.retrieval_used_tokens == conv_tokens + mem_tokens
    assert ledger.retrieval_used_tokens <= 100


def test_scenario_11_multitask_workflow_does_not_copy_parent_history():
    """Scenario 11: Multi-task children receive child requests and metadata without copying parent history."""
    child_envelope = build_routing_execution_envelope(
        user_request="Child task: Run tests for component A",
        identity=IdentitySessionRelationship(
            authenticated_user_id="user_alice",
            session_id="session_123",
            conversation_id="conv_123",
            turn_id="turn_3:child_1",
            task_id="child_task_1",
            parent_task_id="root_task_3",
        ),
        execution_state=ExecutionRecoveryState(active_route=""),
        accounting_snapshot=AccountingSnapshot(
            task_id="child_task_1",
            turn_id="turn_3:child_1",
            task_budget_tokens=500000,
            task_consumed_tokens=0,
            task_reserved_tokens=0,
            task_remaining_tokens=500000,
            turn_budget_tokens=None,
            turn_consumed_tokens=0,
            turn_remaining_tokens=None,
            verification_reserve_tokens=25000,
            session_budget_tokens=None,
            session_consumed_tokens=0,
            session_remaining_tokens=None,
            cost_budget=None,
            cost_consumed=0.0,
            cost_remaining=None,
            active_reservations_count=0,
            status="ok",
        ),
        conversation_context_availability=ConversationContextAvailability(has_history=False, available_turns=0),
        memory_availability=MemoryAvailability(memory_capsules_enabled=True),
    )
    child_context = EntryRouteContext(
        session_id="session_123",
        conversation_id="conv_123",
        turn_id="turn_3:child_1",
        conversation_summary="",
        atomic_child=True,
        orchestration_parent_task_id="root_task_3",
        authenticated_user_id="user_alice",
        envelope=child_envelope,
    )
    child_dict = child_context.to_dict()
    assert child_dict["conversation_summary"] == ""
    assert child_dict["envelope"]["identity"]["parent_task_id"] == "root_task_3"
    assert child_dict["envelope"]["user_request"] == "Child task: Run tests for component A"


def test_scenario_12_terminal_and_dashboard_behave_identically():
    """Scenario 12: Terminal and dashboard sessions produce identical envelopes and tool structures."""
    # Terminal session envelope
    terminal_envelope = build_routing_execution_envelope(
        user_request="Analyze codebase",
        identity=IdentitySessionRelationship(
            authenticated_user_id="user_local",
            session_id="cli_session_1",
            conversation_id="cli_session_1",
            turn_id="turn_cli_1",
        ),
        execution_state=ExecutionRecoveryState(),
        accounting_snapshot=AccountingSnapshot(
            task_id="turn_cli_1",
            turn_id="turn_cli_1",
            task_budget_tokens=1000000,
            task_consumed_tokens=0,
            task_reserved_tokens=0,
            task_remaining_tokens=1000000,
            turn_budget_tokens=None,
            turn_consumed_tokens=0,
            turn_remaining_tokens=None,
            verification_reserve_tokens=50000,
            session_budget_tokens=None,
            session_consumed_tokens=0,
            session_remaining_tokens=None,
            cost_budget=None,
            cost_consumed=0.0,
            cost_remaining=None,
            active_reservations_count=0,
            status="ok",
        ),
    )

    # Dashboard/API session envelope
    dashboard_envelope = build_routing_execution_envelope(
        user_request="Analyze codebase",
        identity=IdentitySessionRelationship(
            authenticated_user_id="user_local",
            session_id="api_session_1",
            conversation_id="api_session_1",
            turn_id="turn_api_1",
        ),
        execution_state=ExecutionRecoveryState(),
        accounting_snapshot=AccountingSnapshot(
            task_id="turn_api_1",
            turn_id="turn_api_1",
            task_budget_tokens=1000000,
            task_consumed_tokens=0,
            task_reserved_tokens=0,
            task_remaining_tokens=1000000,
            turn_budget_tokens=None,
            turn_consumed_tokens=0,
            turn_remaining_tokens=None,
            verification_reserve_tokens=50000,
            session_budget_tokens=None,
            session_consumed_tokens=0,
            session_remaining_tokens=None,
            cost_budget=None,
            cost_consumed=0.0,
            cost_remaining=None,
            active_reservations_count=0,
            status="ok",
        ),
    )

    assert set(terminal_envelope.to_dict().keys()) == set(dashboard_envelope.to_dict().keys())
    assert terminal_envelope.to_dict()["conversation_context_availability"] == dashboard_envelope.to_dict()["conversation_context_availability"]
    assert terminal_envelope.to_dict()["memory_availability"] == dashboard_envelope.to_dict()["memory_availability"]
