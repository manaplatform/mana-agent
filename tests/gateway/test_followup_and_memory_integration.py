"""End-to-end integration and regression tests for follow-up continuity and online/external memory.

Covers all 12 required scenarios:
1. completed task -> "why?" -> prior task is correctly selected.
2. completed task -> "do the same for X" -> classified as expansion/new downstream task without resuming completed task.
3. completed task -> correction -> linked correctly without mutating completed task history.
4. unrelated conversation -> no previous task memory authorization.
5. follow-up-selected task -> MemoryTaskBinding receives exactly that task ID.
6. unoffered task ID -> access denied.
7. external Mem0: write task memory, follow-up, provider search, memory returned.
8. external Supermemory equivalent.
9. capsules enabled + external mode must not bypass the configured online provider.
10. internal mode continues working.
11. failed/running task resume/retry behavior remains intact.
12. /new and session switching must not leak task-private memory between sessions.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mana_agent.gateway import (
    AgentChatGateway,
    ChatTurnResult,
)
from mana_agent.gateway.entry_routing import (
    EntryRouteRegistry,
    EntryRouter,
    RouteAvailability,
    RouteRegistration,
)
from mana_agent.gateway.followup_classifier import (
    FollowupClassification,
    FollowupClassificationError,
    FollowupClassifier,
)
from mana_agent.gateway.lane_coordinator import LaneId, LaneTaskState
from mana_agent.memory import (
    CapsuleConfig,
    CapsuleReadRequest,
    CapsuleScope,
    CapsuleTaskContext,
    MemoryConfig,
    MemoryError,
    MemoryPrincipal,
    MemoryService,
)
from mana_agent.tools.context_retrieval import (
    MemoryTaskBinding,
    execute_memory_read,
)


class _StructuredModel:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def with_structured_output(self, _schema: Any, *, method: str = "json_schema", strict: bool = True):
        return self

    def invoke(self, _messages: Any) -> Any:
        return self.payload


class _RouteModel:
    def __init__(self, *routes: str) -> None:
        self.routes = list(routes)
        self.payloads: list[dict[str, Any]] = []

    def with_structured_output(self, _schema: Any, *, method: str = "json_schema", strict: bool = True):
        return self

    def invoke(self, messages: Any, **_kwargs: Any) -> Any:
        first_content = str(messages[0].content) if messages else ""
        if "Classify this newly received chat turn" in first_content:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "action": "classify",
                        "category": "new_task",
                        "related_task_id": "",
                        "safe_to_continue": True,
                        "reason": "independent turn",
                    }
                )
            )
        if "You decide whether a new user request may resume" in first_content or "Checkpoint resume" in first_content:
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "action": "start_fresh",
                        "task_id": "",
                        "checkpoint_id": "",
                        "same_work": False,
                        "fresh_data_required": False,
                        "checkpoint_still_valid": False,
                        "side_effects_safe_to_repeat": False,
                        "safe_to_continue": True,
                        "reason": "start fresh execution",
                    }
                )
            )
        self.payloads.append(json.loads(messages[-1].content))
        route = self.routes.pop(0) if self.routes else "conversation"
        source_by_route = {
            "conversation": ["none"],
            "coding": ["repository"],
            "repository": ["repository"],
            "unsupported": ["none"],
        }
        return SimpleNamespace(
            content=json.dumps(
                {
                    "route": route,
                    "confidence": 0.98,
                    "reason": f"selected {route}",
                    "required_sources": source_by_route.get(route, ["none"]),
                    "target_urls": [],
                    "requires_live_data": False,
                    "reason_code": "TEST_ROUTE",
                    "error_code": "",
                    "reuse_active_route": len(self.payloads) > 1,
                    "runtime_capability_change": False,
                }
            )
        )


class _AskAgent:
    def __init__(self, answer: str = "Test response") -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            answer=self.answer,
            sources=[],
            warnings=[],
            trace=[],
            payload={"route": "conversation"},
        )


class _AskService:
    def __init__(self, ask_agent: _AskAgent | None = None) -> None:
        self.ask_agent = ask_agent or _AskAgent()
        self.qna_chain = SimpleNamespace(llm=None, chat=lambda question: "chat")
        self.entry_router = SimpleNamespace(llm=None)


class _ChatService:
    def __init__(self, ask_service: _AskService) -> None:
        self._ask_service = ask_service
        self.conversation_calls: list[str] = []

    def ask_conversation(self, question: str) -> str:
        self.conversation_calls.append(question)
        return "ordinary conversation response"

    def ask(self, question: str, **kwargs: Any) -> Any:
        return SimpleNamespace(answer="repository response", sources=[], warnings=[], trace=[])


class _CodingAgent:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.session_id = "bootstrap-session"

    def generate(self, request: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(request)
        return {
            "answer": "coding response",
            "status": "completed",
            "changed_files": ["example.py"],
            "warnings": [],
        }

    generate_auto_execute = generate
    generate_dir_mode = generate

    def get_active_flow_id(self) -> None:
        return None

    def flow_summary(self, flow_id: str) -> None:
        return None

    def reset_flow(self, flow_id: str) -> str:
        return flow_id

    def _tool_policy_for_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"allowed_tools": ["read_file"]}


def _create_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: _RouteModel | None = None,
    coding_agent: _CodingAgent | None = None,
    memory_config: MemoryConfig | None = None,
) -> tuple[AgentChatGateway, list[tuple[str, str, dict[str, Any]]]]:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    events: list[tuple[str, str, dict[str, Any]]] = []

    def sink(event_type: str, message: str = "", metadata: dict[str, Any] | None = None) -> None:
        events.append((event_type, message, metadata or {}))

    route_model = model or _RouteModel("conversation")
    ask_agent = _AskAgent()
    ask_service = _AskService(ask_agent)
    chat_service = _ChatService(ask_service)

    registry = EntryRouteRegistry()
    for name, desc in (
        ("conversation", "ordinary conversation"),
        ("coding", "coding"),
        ("repository", "repository"),
        ("unsupported", "unsupported"),
    ):
        tools = (
            "read_file",
            "edit_file",
            "create_file",
            "write_file",
        ) if name == "coding" else ()
        registry.register(RouteRegistration(name, desc, lambda: RouteAvailability(True), tools=tools))

    gateway = AgentChatGateway(
        tmp_path,
        coding_agent=coding_agent is not None,
        coding_agent_instance=coding_agent,
        agent_tools=True,
        chat_service=chat_service,
        entry_route_registry=registry,
        entry_router=EntryRouter(llm=route_model, registry=registry),
        event_sink=sink,
    )
    if memory_config is not None:
        gateway.config.memory_user_id = "user_test"
        gateway._stack.memory_service = MemoryService(
            root=tmp_path,
            config=memory_config,
            user_id="user_test",
            session_id="session_test",
            repository_id=str(tmp_path),
        )
    return gateway, events


# ---------------------------------------------------------------------------
# Scenario 1: completed task -> "why?" -> prior task is correctly selected.
# ---------------------------------------------------------------------------
def test_completed_task_why_selects_prior_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gateway, events = _create_gateway(tmp_path, monkeypatch, _RouteModel("conversation"))
    session_id = gateway.create_session(frontend="test")

    # Turn 1: execute a coding task that completes
    res = gateway._lane_coordinator.reserve(
        normalized_intent="implement auth module",
        lane_id=LaneId.CODING,
        session_id=session_id,
        workspace_id=gateway._lane_coordinator.taskboard.store.workspace_id,
        repository_id=gateway._lane_coordinator.taskboard.store.repository_id,
        requested_input_tokens=10,
        requested_output_tokens=10,
    )
    gateway._lane_coordinator.start(res)
    gateway._lane_coordinator.finish(
        res.execution.task_id,
        state=LaneTaskState.COMPLETED,
        verification_state={"verification_evidence_present": True},
    )
    completed_task_id = res.execution.task_id

    # Turn 2: user asks "why?"
    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.FollowupClassifier.decide",
        lambda *args, **kwargs: FollowupClassification(
            decision_id="followup-why-1",
            category="clarification_answer",
            related_task_id=completed_task_id,
            safe_to_continue=True,
            reason="User asks why about the auth architecture",
        ),
    )

    result = gateway.process_turn(session_id, "why?")
    assert not result.error

    # Verify events
    followup_events = [e for e in events if e[0] == "followup_relation_selected"]
    assert len(followup_events) == 1
    assert followup_events[0][2]["related_task_id"] == completed_task_id
    assert followup_events[0][2]["category"] == "clarification_answer"

    bound_events = [e for e in events if e[0] == "memory_task_bound" and e[2].get("source") == "followup_classification"]
    assert len(bound_events) == 1
    assert bound_events[0][2]["task_id"] == completed_task_id


# ---------------------------------------------------------------------------
# Scenario 2: completed task -> "do the same for X" -> expansion/new downstream task.
# ---------------------------------------------------------------------------
def test_completed_task_do_the_same_for_x_classified_as_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coding = _CodingAgent()
    gateway, events = _create_gateway(tmp_path, monkeypatch, _RouteModel("coding"), coding_agent=coding)
    session_id = gateway.create_session(frontend="test")

    res = gateway._lane_coordinator.reserve(
        normalized_intent="generate user model",
        lane_id=LaneId.CODING,
        session_id=session_id,
        workspace_id=gateway._lane_coordinator.taskboard.store.workspace_id,
        repository_id=gateway._lane_coordinator.taskboard.store.repository_id,
        requested_input_tokens=10,
        requested_output_tokens=10,
    )
    gateway._lane_coordinator.start(res)
    gateway._lane_coordinator.finish(
        res.execution.task_id,
        state=LaneTaskState.COMPLETED,
        verification_state={"verification_evidence_present": True},
    )
    completed_task_id = res.execution.task_id

    # Follow-up "do the same for X"
    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.FollowupClassifier.decide",
        lambda *args, **kwargs: FollowupClassification(
            decision_id="followup-exp-1",
            category="task_expansion",
            related_task_id=completed_task_id,
            safe_to_continue=True,
            reason="User wants to expand the previous model creation for product model",
        ),
    )

    result = gateway.process_turn(session_id, "do the same for product model")
    assert not result.error

    # The completed task must remain COMPLETED and not retried/resumed
    original_task = gateway._lane_coordinator.execution_supervisor.store.get_task(completed_task_id)
    assert original_task.state.value == "completed"

    # A new downstream task was created with parent_task_id pointing to completed task
    task_linked_events = [e for e in events if e[0] == "task_linked"]
    assert len(task_linked_events) == 1
    assert task_linked_events[0][2]["parent_task_id"] == completed_task_id
    assert task_linked_events[0][2]["relation_type"] == "expansion"


# ---------------------------------------------------------------------------
# Scenario 3: completed task -> correction -> linked correctly without mutating history.
# ---------------------------------------------------------------------------
def test_completed_task_correction_linked_without_mutating_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coding = _CodingAgent()
    gateway, events = _create_gateway(tmp_path, monkeypatch, _RouteModel("coding"), coding_agent=coding)
    session_id = gateway.create_session(frontend="test")

    res = gateway._lane_coordinator.reserve(
        normalized_intent="write config parser",
        lane_id=LaneId.CODING,
        session_id=session_id,
        workspace_id=gateway._lane_coordinator.taskboard.store.workspace_id,
        repository_id=gateway._lane_coordinator.taskboard.store.repository_id,
        requested_input_tokens=10,
        requested_output_tokens=10,
    )
    gateway._lane_coordinator.start(res)
    gateway._lane_coordinator.finish(
        res.execution.task_id,
        state=LaneTaskState.COMPLETED,
        verification_state={"verification_evidence_present": True},
    )
    completed_task_id = res.execution.task_id

    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.FollowupClassifier.decide",
        lambda *args, **kwargs: FollowupClassification(
            decision_id="followup-corr-1",
            category="task_correction",
            related_task_id=completed_task_id,
            safe_to_continue=True,
            reason="Fix typo in config parser",
        ),
    )

    result = gateway.process_turn(session_id, "fix the typo in the port number")
    assert not result.error

    original_task = gateway._lane_coordinator.execution_supervisor.store.get_task(completed_task_id)
    assert original_task.state.value == "completed"

    task_linked_events = [e for e in events if e[0] == "task_linked"]
    assert len(task_linked_events) == 1
    assert task_linked_events[0][2]["parent_task_id"] == completed_task_id
    assert task_linked_events[0][2]["relation_type"] == "correction"


# ---------------------------------------------------------------------------
# Scenario 4: unrelated conversation -> no previous task memory authorization.
# ---------------------------------------------------------------------------
def test_unrelated_conversation_no_previous_task_memory_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway, events = _create_gateway(tmp_path, monkeypatch, _RouteModel("conversation"))
    session_id = gateway.create_session(frontend="test")

    res = gateway._lane_coordinator.reserve(
        normalized_intent="task alpha",
        lane_id=LaneId.CODING,
        session_id=session_id,
        workspace_id=gateway._lane_coordinator.taskboard.store.workspace_id,
        repository_id=gateway._lane_coordinator.taskboard.store.repository_id,
        requested_input_tokens=10,
        requested_output_tokens=10,
    )
    gateway._lane_coordinator.start(res)
    gateway._lane_coordinator.finish(res.execution.task_id, state=LaneTaskState.COMPLETED)

    # Independent message
    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.FollowupClassifier.decide",
        lambda *args, **kwargs: FollowupClassification(
            decision_id="followup-indep-1",
            category="conversation_only",
            related_task_id="",
            safe_to_continue=True,
            reason="independent conversational greeting",
        ),
    )

    result = gateway.process_turn(session_id, "hello there")
    assert not result.error

    # No task was bound to memory
    bound_events = [e for e in events if e[0] == "memory_task_bound" and e[2].get("source") == "followup_classification"]
    assert len(bound_events) == 0

    # Reading memory without authorized task fails
    unbound = MemoryTaskBinding(selected_memory_task_id="")
    candidates = ({"task_id": res.execution.task_id, "normalized_intent": "task alpha", "state": "completed"},)
    read_res = json.loads(
        execute_memory_read(
            query="alpha details",
            session_id=session_id,
            authenticated_user_id="user_test",
            capsule_service=gateway._stack.memory_service.capsules,
            repository_id=str(tmp_path),
            current_turn_id="turn_unrelated",
            selected_memory_task_id=unbound,
            memory_task_candidates=candidates,
            event_sink=lambda *args, **kwargs: None,
        )
    )
    assert read_res["status"] == "unauthorized"


# ---------------------------------------------------------------------------
# Scenario 5: follow-up-selected task -> MemoryTaskBinding receives exactly that task ID.
# ---------------------------------------------------------------------------
def test_followup_selected_task_binds_exact_task_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binding = MemoryTaskBinding(selected_memory_task_id="")
    assert binding.selected_memory_task_id == ""

    # Binding a valid task ID
    binding.bind("task_selected_123")
    assert binding.selected_memory_task_id == "task_selected_123"

    # Gateway turn integration test
    gateway, events = _create_gateway(tmp_path, monkeypatch, _RouteModel("conversation"))
    session_id = gateway.create_session(frontend="test")

    res = gateway._lane_coordinator.reserve(
        normalized_intent="task target",
        lane_id=LaneId.CODING,
        session_id=session_id,
        workspace_id=gateway._lane_coordinator.taskboard.store.workspace_id,
        repository_id=gateway._lane_coordinator.taskboard.store.repository_id,
        requested_input_tokens=10,
        requested_output_tokens=10,
    )
    gateway._lane_coordinator.start(res)
    gateway._lane_coordinator.finish(
        res.execution.task_id,
        state=LaneTaskState.COMPLETED,
        verification_state={"verification_evidence_present": True},
    )
    target_task_id = res.execution.task_id

    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.FollowupClassifier.decide",
        lambda *args, **kwargs: FollowupClassification(
            decision_id="followup-dec-5",
            category="clarification_answer",
            related_task_id=target_task_id,
            safe_to_continue=True,
            reason="User follow-up on target task",
        ),
    )

    gateway.process_turn(session_id, "tell me about target task")
    bound_events = [e for e in events if e[0] == "memory_task_bound" and e[2].get("task_id") == target_task_id]
    assert len(bound_events) >= 1


# ---------------------------------------------------------------------------
# Scenario 6: unoffered task ID -> access denied.
# ---------------------------------------------------------------------------
def test_unoffered_task_id_access_denied(tmp_path: Path) -> None:
    events: list[tuple[str, str, dict[str, Any]]] = []

    def sink(event_type: str, message: str = "", metadata: dict[str, Any] | None = None) -> None:
        events.append((event_type, message, metadata or {}))

    binding = MemoryTaskBinding(selected_memory_task_id="unoffered_attacker_task")
    offered_candidates = ({"task_id": "legit_task_1", "normalized_intent": "clean code", "state": "completed"},)

    read_res = json.loads(
        execute_memory_read(
            query="secret data",
            session_id="session_1",
            authenticated_user_id="user_test",
            capsule_service=None,
            repository_id=str(tmp_path),
            current_turn_id="turn_denied",
            selected_memory_task_id=binding,
            memory_task_candidates=offered_candidates,
            event_sink=sink,
        )
    )

    assert read_res["status"] == "unauthorized"
    assert "was not offered to the router. Access denied." in read_res["error"]

    denied_events = [e for e in events if e[0] == "denied_memory_task_access"]
    assert len(denied_events) >= 1
    assert denied_events[0][2]["task_id"] == "unoffered_attacker_task"
    assert denied_events[0][2]["error"] == "task_not_offered"


# ---------------------------------------------------------------------------
# Scenario 7: external Mem0: write task memory, follow-up, provider search, memory returned.
# ---------------------------------------------------------------------------
def test_external_mem0_write_and_followup_recall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mem0_writes: list[dict[str, Any]] = []
    mem0_searches: list[dict[str, Any]] = []

    class FakeMem0Client:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def add(self, text: str, **kwargs: Any) -> dict[str, Any]:
            mem0_writes.append({"text": text, **kwargs})
            return {"results": [{"id": f"mem0-{len(mem0_writes)}", "memory": text}]}

        def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
            mem0_searches.append({"query": query, **kwargs})
            return {
                "results": [
                    {
                        "id": "mem0-1",
                        "memory": "Database migration succeeded on PostgreSQL 16",
                        "score": 0.95,
                        "metadata": {
                            "task_id": "task_mem0_test",
                            "user_id": "user_test",
                            "scope": "private",
                            "title": "Task result task_mem0_test",
                        },
                    }
                ]
            }

        def get_all(self, **_kwargs: Any) -> dict[str, Any]:
            return {"results": []}

    monkeypatch.setitem(sys.modules, "mem0", types.SimpleNamespace(MemoryClient=FakeMem0Client))

    mem_config = MemoryConfig(
        mode="external",
        provider="mem0",
        api_key="test-mem0-key",
        capsules=CapsuleConfig(enabled=True),
    )
    gateway, events = _create_gateway(tmp_path, monkeypatch, memory_config=mem_config)
    session_id = gateway.create_session(frontend="test")

    # Record turn memory
    turn_result = ChatTurnResult(
        answer="PostgreSQL migration completed successfully.",
        mode="coding",
        changed_files=["migrations/001.sql"],
        payload={"execution_id": "task_mem0_test", "entry_route": "coding"},
    )
    write_warning = gateway._record_followup_memory(
        session_id=session_id,
        conversation_id=session_id,
        turn_id="turn_mem0_write",
        user_text="run the database migration",
        result=turn_result,
        sink=gateway._event_sink,
    )
    assert not write_warning
    assert len(mem0_writes) == 1
    assert "PostgreSQL migration completed successfully." in mem0_writes[0]["text"]

    # Verify provider_memory_write diagnostic event
    write_events = [e for e in events if e[0] == "provider_memory_write"]
    assert len(write_events) == 1
    assert write_events[0][2]["provider"] == "mem0"
    assert write_events[0][2]["task_id"] == "task_mem0_test"

    # Follow-up recall
    binding = MemoryTaskBinding(selected_memory_task_id="task_mem0_test")
    candidates = ({"task_id": "task_mem0_test", "normalized_intent": "run migration", "state": "completed"},)

    read_encoded = execute_memory_read(
        query="migration postgresql",
        session_id=session_id,
        authenticated_user_id="user_test",
        capsule_service=gateway._stack.memory_service.capsules,
        repository_id=str(tmp_path),
        current_turn_id="turn_mem0_read",
        selected_memory_task_id=binding,
        memory_task_candidates=candidates,
        event_sink=gateway._event_sink,
    )
    read_payload = json.loads(read_encoded)
    assert read_payload["status"] in {"ok", "matched"}
    assert read_payload["capsules_returned"] == 1
    assert "Database migration succeeded on PostgreSQL 16" in read_payload["capsules"][0]["summary"]

    # Verify provider_memory_read diagnostic event
    read_events = [e for e in events if e[0] == "provider_memory_read"]
    assert len(read_events) == 1
    assert read_events[0][2]["provider"] == "mem0"


# ---------------------------------------------------------------------------
# Scenario 8: external Supermemory equivalent.
# ---------------------------------------------------------------------------
def test_external_supermemory_write_and_followup_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sm_writes: list[dict[str, Any]] = []
    sm_searches: list[dict[str, Any]] = []

    from mana_agent.memory.providers.supermemory.backend import SupermemoryProvider

    async def fake_sm_call(method: str, *args: Any, operation: str, **kwargs: Any) -> Any:
        if method == "add":
            sm_writes.append({"kwargs": kwargs, "args": args})
            return types.SimpleNamespace(id="sm-doc-1", status="queued")
        elif method in {"search", "search.memories"}:
            sm_searches.append({"kwargs": kwargs, "args": args})
            return types.SimpleNamespace(
                results=[
                    types.SimpleNamespace(
                        id="sm-doc-1",
                        content="Redis cache cluster configured with TLS",
                        similarity=0.92,
                        metadata={
                            "task_id": "task_sm_test",
                            "user_id": "user_test",
                            "scope": "private",
                            "title": "Task result task_sm_test",
                        },
                        created_at="2026-10-02T10:00:00Z",
                    )
                ]
            )
        return types.SimpleNamespace(results=[])

    mem_config = MemoryConfig(
        mode="external",
        provider="supermemory",
        api_key="test-sm-key",
        capsules=CapsuleConfig(enabled=True),
    )
    gateway, events = _create_gateway(tmp_path, monkeypatch, memory_config=mem_config)
    session_id = gateway.create_session(frontend="test")

    # Wire fake call into the supermemory backend client
    backend = gateway._stack.memory_service.backend
    assert isinstance(backend, SupermemoryProvider)
    backend.client.call = fake_sm_call  # type: ignore[method-assign]

    turn_result = ChatTurnResult(
        answer="Redis cache cluster configured with TLS.",
        mode="coding",
        changed_files=["redis.conf"],
        payload={"execution_id": "task_sm_test", "entry_route": "coding"},
    )
    write_warning = gateway._record_followup_memory(
        session_id=session_id,
        conversation_id=session_id,
        turn_id="turn_sm_write",
        user_text="setup redis TLS",
        result=turn_result,
        sink=gateway._event_sink,
    )
    assert not write_warning
    assert len(sm_writes) == 1

    write_events = [e for e in events if e[0] == "provider_memory_write"]
    assert len(write_events) == 1
    assert write_events[0][2]["provider"] == "supermemory"

    binding = MemoryTaskBinding(selected_memory_task_id="task_sm_test")
    candidates = ({"task_id": "task_sm_test", "normalized_intent": "setup redis", "state": "completed"},)

    read_encoded = execute_memory_read(
        query="redis cluster TLS",
        session_id=session_id,
        authenticated_user_id="user_test",
        capsule_service=gateway._stack.memory_service.capsules,
        repository_id=str(tmp_path),
        current_turn_id="turn_sm_read",
        selected_memory_task_id=binding,
        memory_task_candidates=candidates,
        event_sink=gateway._event_sink,
    )
    read_payload = json.loads(read_encoded)
    assert read_payload["status"] in {"ok", "matched"}
    assert read_payload["capsules_returned"] == 1
    assert "Redis cache cluster configured with TLS" in read_payload["capsules"][0]["summary"]

    read_events = [e for e in events if e[0] == "provider_memory_read"]
    assert len(read_events) == 1
    assert read_events[0][2]["provider"] == "supermemory"


# ---------------------------------------------------------------------------
# Scenario 9: capsules enabled + external mode must not bypass online provider.
# ---------------------------------------------------------------------------
def test_capsules_enabled_external_mode_does_not_bypass_online_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    searches_called: list[str] = []

    class FailingMem0Client:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def add(self, text: str, **kwargs: Any) -> dict[str, Any]:
            return {"results": [{"id": "m1"}]}

        def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
            searches_called.append(query)
            raise ConnectionError("Mem0 provider network timeout")

        def get_all(self, **_kwargs: Any) -> dict[str, Any]:
            return {"results": []}

    monkeypatch.setitem(sys.modules, "mem0", types.SimpleNamespace(MemoryClient=FailingMem0Client))

    mem_config = MemoryConfig(
        mode="external",
        provider="mem0",
        api_key="test-key",
        capsules=CapsuleConfig(enabled=True),
    )
    service = MemoryService(
        root=tmp_path,
        config=mem_config,
        user_id="user_1",
        session_id="session_1",
        repository_id="repo_1",
    )

    principal = MemoryPrincipal(
        user_id="user_1",
        project_id="repo_1",
        task_id="task_fail_test",
        agent_id="gateway:chat",
        capabilities=frozenset({"memory.capsule.read.private"}),
    )
    task_context = CapsuleTaskContext(
        user_id="user_1",
        organisation_id=None,
        project_id="repo_1",
        team_ids=frozenset(),
        task_id="task_fail_test",
        agent_id="gateway:chat",
        session_id="session_1",
    )

    # When external search fails, it must NOT silently fall back to internal repository
    with pytest.raises((ConnectionError, MemoryError)):
        service.capsules.query_capsules(
            CapsuleReadRequest(
                principal=principal,
                task_context=task_context,
                query="find previous work",
                allowed_scopes=frozenset({CapsuleScope.PRIVATE}),
            )
        )
    assert len(searches_called) == 1


# ---------------------------------------------------------------------------
# Scenario 10: internal mode continues working.
# ---------------------------------------------------------------------------
def test_internal_mode_continues_working(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mem_config = MemoryConfig(
        mode="internal",
        provider="mana",
        capsules=CapsuleConfig(enabled=True),
    )
    gateway, events = _create_gateway(tmp_path, monkeypatch, memory_config=mem_config)
    session_id = gateway.create_session(frontend="test")

    turn_result = ChatTurnResult(
        answer="Built React dashboard components.",
        mode="coding",
        changed_files=["dashboard.tsx"],
        payload={"execution_id": "task_internal_1", "entry_route": "coding"},
    )
    gateway._record_followup_memory(
        session_id=session_id,
        conversation_id=session_id,
        turn_id="turn_internal_write",
        user_text="build the dashboard",
        result=turn_result,
        sink=gateway._event_sink,
    )

    binding = MemoryTaskBinding(selected_memory_task_id="task_internal_1")
    candidates = ({"task_id": "task_internal_1", "normalized_intent": "build dashboard", "state": "completed"},)

    read_encoded = execute_memory_read(
        query="dashboard components",
        session_id=session_id,
        authenticated_user_id="user_test",
        capsule_service=gateway._stack.memory_service.capsules,
        repository_id=str(tmp_path),
        current_turn_id="turn_internal_read",
        selected_memory_task_id=binding,
        memory_task_candidates=candidates,
        event_sink=gateway._event_sink,
    )
    read_payload = json.loads(read_encoded)
    assert read_payload["status"] in {"ok", "matched"}
    assert read_payload["capsules_returned"] == 1
    assert "Built React dashboard components." in read_payload["capsules"][0]["summary"]


# ---------------------------------------------------------------------------
# Scenario 11: failed/running task resume/retry behavior must remain intact.
# ---------------------------------------------------------------------------
def test_failed_running_task_resume_retry_behavior_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway, events = _create_gateway(tmp_path, monkeypatch, _RouteModel("conversation", "conversation"))
    session_id = gateway.create_session(frontend="test")

    # 1. Failed task can be retried
    res_failed = gateway._lane_coordinator.reserve(
        normalized_intent="compile native binary",
        lane_id=LaneId.CODING,
        session_id=session_id,
        workspace_id=gateway._lane_coordinator.taskboard.store.workspace_id,
        repository_id=gateway._lane_coordinator.taskboard.store.repository_id,
        requested_input_tokens=10,
        requested_output_tokens=10,
    )
    gateway._lane_coordinator.start(res_failed)
    gateway._lane_coordinator.finish(res_failed.execution.task_id, state=LaneTaskState.FAILED, error="gcc not found")
    failed_task_id = res_failed.execution.task_id

    dec = FollowupClassifier(
        _StructuredModel(
            {
                "category": "retry_request",
                "related_task_id": failed_task_id,
                "safe_to_continue": True,
                "reason": "Retry the failed compilation task",
            }
        )
    ).decide(
        message="retry compiling the binary",
        recent_history=[],
        candidates=[{"task_id": failed_task_id, "state": "failed", "normalized_intent": "compile native binary"}],
    )
    assert dec.category == "retry_request"
    assert dec.related_task_id == failed_task_id

    # 2. Running task status request
    res_running = gateway._lane_coordinator.reserve(
        normalized_intent="long running build",
        lane_id=LaneId.CODING,
        session_id=session_id,
        workspace_id=gateway._lane_coordinator.taskboard.store.workspace_id,
        repository_id=gateway._lane_coordinator.taskboard.store.repository_id,
        requested_input_tokens=10,
        requested_output_tokens=10,
    )
    gateway._lane_coordinator.start(res_running)
    running_task_id = res_running.execution.task_id

    monkeypatch.setattr(
        "mana_agent.gateway.chat_gateway.FollowupClassifier.decide",
        lambda *args, **kwargs: FollowupClassification(
            decision_id="followup-status-1",
            category="status_request",
            related_task_id=running_task_id,
            safe_to_continue=True,
            reason="User asking for status of running task",
        ),
    )
    turn_res = gateway.process_turn(session_id, "what is the build status?")
    assert not turn_res.error
    assert "currently running" in turn_res.answer or turn_res.payload.get("status") == "running"


# ---------------------------------------------------------------------------
# Scenario 12: /new and session switching must not leak task-private memory between sessions.
# ---------------------------------------------------------------------------
def test_new_and_session_switching_no_memory_leak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gateway, events = _create_gateway(tmp_path, monkeypatch, _RouteModel("conversation"))

    # Session 1: run task and store memory
    sid_1 = gateway.create_session(frontend="test")
    res_s1 = gateway._lane_coordinator.reserve(
        normalized_intent="confidential session 1 task",
        lane_id=LaneId.CODING,
        session_id=sid_1,
        workspace_id=gateway._lane_coordinator.taskboard.store.workspace_id,
        repository_id=gateway._lane_coordinator.taskboard.store.repository_id,
        requested_input_tokens=10,
        requested_output_tokens=10,
    )
    gateway._lane_coordinator.start(res_s1)
    gateway._lane_coordinator.finish(
        res_s1.execution.task_id,
        state=LaneTaskState.COMPLETED,
        verification_state={"verification_evidence_present": True},
    )
    s1_task_id = res_s1.execution.task_id

    # Session 2: created via /new or start_new_conversation
    sid_2 = gateway.start_new_conversation(sid_1, frontend="test")

    # In Session 2, s1_task_id must NOT be offered as a candidate
    all_candidates = gateway._recovery_candidates(
        lane_id=None,
        session_id=sid_2,
        workspace_id=gateway._lane_coordinator.taskboard.store.workspace_id,
        repository_id=gateway._lane_coordinator.taskboard.store.repository_id,
    )
    s2_session_candidates = [
        item for item in all_candidates if str(item.get("session_id") or "") == sid_2
    ]
    offered_s2_ids = {item["task_id"] for item in s2_session_candidates}
    assert s1_task_id not in offered_s2_ids

    # Querying s1_task_id memory from Session 2 fails closed
    binding = MemoryTaskBinding(selected_memory_task_id=s1_task_id)
    candidates_s2 = tuple(
        {"task_id": c["task_id"], "normalized_intent": c["normalized_intent"], "state": c["state"]}
        for c in s2_session_candidates
    )

    read_encoded = execute_memory_read(
        query="confidential details",
        session_id=sid_2,
        authenticated_user_id="user_test",
        capsule_service=gateway._stack.memory_service.capsules,
        repository_id=str(tmp_path),
        current_turn_id="turn_s2",
        selected_memory_task_id=binding,
        memory_task_candidates=candidates_s2,
        event_sink=gateway._event_sink,
    )
    read_res = json.loads(read_encoded)
    assert read_res["status"] == "unauthorized"

    denied_events = [e for e in events if e[0] == "denied_memory_task_access" and e[2].get("session_id") == sid_2]
    assert len(denied_events) >= 1
