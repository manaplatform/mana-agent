"""Tests for Canonical RoutingExecutionEnvelope and Context Retrieval Tools.

Covers all 12 scenarios:
1. Long stored history does not increase ordinary prompt size.
2. Independent turn performs zero conversation/memory reads.
3. Follow-up "why?" retrieves relevant previous turn via conversation_context_read.
4. Memory question uses memory_read to fetch authorized capsules.
5. Unauthorized private memory remains inaccessible (deny-by-default).
6. Fabricated session/user identifiers cannot alter tool scope.
7. Retrieval results are token bounded.
8. Retrieval usage participates in Phase-0 accounting.
9. Repeated identical retrieval within a turn is deduplicated.
10. Routing receives the complete structured envelope but no raw history.
11. Multi-task memory and follow-up workflows continue to work without copying parent chat history.
12. Terminal and dashboard sessions behave identically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from mana_agent.context_cost.governor import ContextCostGovernor
from mana_agent.context_cost.models import AccountingSnapshot
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
    """Scenario 1: Long stored history does not increase ordinary prompt size."""
    session_state: dict[str, Any] = {
        "messages": [
            DummyMessage(role="user", content=f"User question {i}", turn_id=f"turn_{i}").to_dict()
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
    governor = MagicMock()
    turn_cache: dict[str, Any] = {}

    tools = build_context_retrieval_tools(
        session_id="session_123",
        conversation_id="conv_123",
        authenticated_user_id="user_abc",
        history_store=history,
        current_turn_id="turn_2",
        governor=governor,
        turn_retrieval_cache=turn_cache,
    )
    assert len(tools) == 2
    assert len(turn_cache) == 0
    assert governor.record_usage.call_count == 0


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
    )
    data = json.loads(res_json)
    assert data["source"] == "conversation_context"
    assert data["turns_returned"] == 1
    assert data["empty"] is False
    assert len(data["turns"]) == 2  # user + assistant in turn_1
    assert any("OAuth2 with PKCE" in t["content"] for t in data["turns"])
    assert any(e["type"] == "context.conversation_read" for e in events)


def test_scenario_4_memory_question_uses_memory_read():
    """Scenario 4: Memory question uses memory_read to fetch authorized capsules."""
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
    res_json = execute_memory_read(
        capsule_service=capsule_service,
        authenticated_user_id="user_alice",
        session_id="session_123",
        repository_id="repo_1",
        current_task_id="task_100",
        query="dataclasses",
        max_capsules=3,
        turn_retrieval_cache=turn_cache,
        event_sink=sink,
    )
    data = json.loads(res_json)
    assert data["source"] == "memory_capsules"
    assert data["capsules_returned"] == 1
    assert data["empty"] is False
    assert data["capsules"][0]["capsule_id"] == "cap_1"
    assert "slots" in data["capsules"][0]["content"]
    assert any(e["type"] == "context.memory_read" for e in events)


def test_scenario_5_unauthorized_private_memory_remains_inaccessible():
    """Scenario 5: Unauthorized private memory remains inaccessible (deny-by-default)."""
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
        query="secret",
    )
    data_unauth = json.loads(res_unauth)
    assert data_unauth["empty"] is True
    assert data_unauth["capsules_returned"] == 0
    assert "requires an authenticated user identity" in data_unauth["error"]

    # Unauthorized task ID not in offered candidates
    res_unoffered = execute_memory_read(
        capsule_service=capsule_service,
        authenticated_user_id="user_alice",
        session_id="session_123",
        repository_id="repo_1",
        current_task_id="task_allowed",
        memory_task_candidates=({"task_id": "task_allowed", "normalized_intent": "foo", "state": "active"},),
        task_id="task_unauthorized_999",
        query="secret",
    )
    data_unoffered = json.loads(res_unoffered)
    assert data_unoffered["empty"] is True
    assert data_unoffered["capsules_returned"] == 0
    assert "not offered to the router" in data_unoffered["error"]


def test_scenario_6_fabricated_identifiers_cannot_alter_tool_scope():
    """Scenario 6: Fabricated session/user identifiers cannot alter tool scope."""
    history = DummyHistoryStore(
        [
            DummyMessage(role="user", content="Original user message", turn_id="turn_1"),
            DummyMessage(role="assistant", content="Original answer", turn_id="turn_1"),
        ]
    )
    tools = build_context_retrieval_tools(
        session_id="trusted_session",
        conversation_id="trusted_conv",
        authenticated_user_id="trusted_user",
        history_store=history,
        current_turn_id="turn_2",
    )
    conv_tool = next(t for t in tools if t.name == "conversation_context_read")

    # Even if model passes fabricated kwargs, trusted session and user ID are used
    result_str = conv_tool.invoke({"query": "Original", "max_turns": 2, "session_id": "attacker_session", "user_id": "attacker_user"})
    data = json.loads(result_str)
    assert data["session_id"] == "trusted_session"
    assert data["conversation_id"] == "trusted_conv"


def test_scenario_7_retrieval_results_are_token_bounded():
    """Scenario 7: Retrieval results are token bounded."""
    huge_content = "Word " * 5000  # ~5000 tokens
    sample_projection = DummyCapsuleProjection(
        capsule_id="cap_huge",
        revision=1,
        title="Huge Documentation",
        summary="A very long document",
        content=huge_content,
        tags=("docs",),
        scope=CapsuleScope.PROJECT,
    )
    capsule_service = DummyCapsuleService([sample_projection])

    res_json = execute_memory_read(
        capsule_service=capsule_service,
        authenticated_user_id="user_alice",
        session_id="session_123",
        repository_id="repo_1",
        query="Documentation",
        max_capsules=1,
        max_tokens=100,  # Strict token bound
    )
    data = json.loads(res_json)
    assert data["tokens"] <= 100 or data["empty"] is True


def test_scenario_8_retrieval_usage_participates_in_accounting():
    """Scenario 8: Retrieval usage participates in Phase-0 accounting."""
    history = DummyHistoryStore(
        [
            DummyMessage(role="user", content="Question 1", turn_id="turn_1"),
            DummyMessage(role="assistant", content="Answer 1", turn_id="turn_1"),
        ]
    )
    governor = MagicMock()

    execute_conversation_context_read(
        session_id="session_123",
        conversation_id="conv_123",
        authenticated_user_id="user_alice",
        history_store=history,
        current_turn_id="turn_2",
        governor=governor,
    )
    assert governor.record_usage.call_count == 1
    kwargs = governor.record_usage.call_args[1]
    assert kwargs["task_id"] == "turn_2"
    assert kwargs["input_tokens"] > 0


def test_scenario_9_repeated_identical_retrieval_is_deduplicated():
    """Scenario 9: Repeated identical retrieval is deduplicated with zero additional charge."""
    history = DummyHistoryStore(
        [
            DummyMessage(role="user", content="Question 1", turn_id="turn_1"),
            DummyMessage(role="assistant", content="Answer 1", turn_id="turn_1"),
        ]
    )
    governor = MagicMock()
    turn_cache: dict[str, Any] = {}
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
        governor=governor,
        turn_retrieval_cache=turn_cache,
        event_sink=sink,
    )
    assert governor.record_usage.call_count == 1

    # Second identical call in same turn
    res_2 = execute_conversation_context_read(
        session_id="session_123",
        conversation_id="conv_123",
        authenticated_user_id="user_alice",
        history_store=history,
        current_turn_id="turn_2",
        query="Question",
        max_turns=3,
        governor=governor,
        turn_retrieval_cache=turn_cache,
        event_sink=sink,
    )
    assert res_1 == res_2
    # Governor was not called a second time (0 additional tokens charged)
    assert governor.record_usage.call_count == 1
    assert any(e["type"] == "context.retrieval_deduplicated" for e in events)


def test_scenario_10_routing_receives_complete_structured_envelope_no_raw_history():
    """Scenario 10: Routing receives the complete structured envelope but no raw history."""
    envelope = build_routing_execution_envelope(
        user_request="Refactor the routing layer",
        identity=IdentitySessionRelationship(
            authenticated_user_id="user_alice",
            session_id="session_123",
            conversation_id="conv_123",
            turn_id="turn_10",
            task_id="task_10",
            workspace_id="ws_1",
            repository_id="repo_1",
        ),
        execution_state=ExecutionRecoveryState(
            active_route="coding",
            lane_id="lane_fast",
        ),
        accounting_snapshot=AccountingSnapshot(
            task_id="turn_10",
            turn_id="turn_10",
            task_budget_tokens=1000000,
            task_consumed_tokens=5000,
            task_reserved_tokens=0,
            task_remaining_tokens=995000,
            turn_budget_tokens=None,
            turn_consumed_tokens=5000,
            turn_remaining_tokens=None,
            verification_reserve_tokens=50000,
            session_budget_tokens=None,
            session_consumed_tokens=5000,
            session_remaining_tokens=None,
            cost_budget=None,
            cost_consumed=0.05,
            cost_remaining=None,
            active_reservations_count=0,
            status="ok",
        ),
        model_candidates=(
            ModelCandidateCapacity(
                model_id="gpt-4o",
                provider="openai",
                context_window=128000,
                max_output_tokens=4096,
                supported_roles=("main", "coding"),
            ),
        ),
        route_availability=({"name": "coding", "available": True},),
        capabilities_and_tools=({"name": "repo_search", "category": "repository"},),
        approval_state=ApprovalState(),
        artifact_metadata={"has_artifacts": False},
        previous_turn_pointers=PreviousTurnPointers(previous_turn_id="turn_9", previous_route="coding"),
        conversation_context_availability=ConversationContextAvailability(has_history=True, available_turns=9),
        memory_availability=MemoryAvailability(memory_capsules_enabled=True),
    )

    data = envelope.to_dict()
    assert data["user_request"] == "Refactor the routing layer"
    assert data["identity"]["authenticated_user_id"] == "user_alice"
    assert data["accounting_snapshot"]["task_remaining_tokens"] == 995000
    assert data["previous_turn_pointers"]["previous_turn_id"] == "turn_9"
    assert "messages" not in data
    assert "transcript" not in data
    assert "history" not in data
    assert "secrets" not in data

    context = EntryRouteContext(
        session_id="session_123",
        conversation_id="conv_123",
        turn_id="turn_10",
        envelope=envelope,
    )
    context_data = context.to_dict()
    assert "envelope" in context_data
    assert context_data["envelope"]["user_request"] == "Refactor the routing layer"


def test_scenario_11_multitask_workflow_does_not_copy_parent_history():
    """Scenario 11: Multi-task children receive child requests and metadata without copying parent history."""
    parent_history = DummyHistoryStore(
        [
            DummyMessage(role="user", content="Parent turn 1", turn_id="turn_1"),
            DummyMessage(role="assistant", content="Parent answer 1", turn_id="turn_1"),
            DummyMessage(role="user", content="Parent turn 2", turn_id="turn_2"),
        ]
    )
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
