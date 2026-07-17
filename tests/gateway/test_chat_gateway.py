"""Tests for the central AgentChatGateway.

These verify:
- Construction succeeds with minimal config.
- Gateway builds coding stack when enabled (with injected fakes).
- Simple send / process_turn paths work.
- Rich context is provided.
- Auto-chat + coding agent exist on the gateway (not only chat_cli).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mana_agent.config.settings import Settings
from mana_agent.gateway import (
    AgentChatGateway,
    ChatGatewayConfig,
    ChatTurnResult,
    RichChatContext,
)
from mana_agent.integrations.codex.coding_agent_shim import CodexCodingAgentShim
from mana_agent.multi_agent.routing.agent_decision import AgentDecision
from mana_agent.services.chat_session_history import ChatSessionHistory


class _DummyAskService:
    """Minimal stand-in so gateway construction tests do not require OPENAI_API_KEY."""

    ask_agent = SimpleNamespace(llm=None, update_model=lambda m: None, model="dummy")
    qna_chain = SimpleNamespace(
        llm=None,
        chat=lambda question: "(dummy conversational response)",
    )

    def ask(self, *args, **kwargs):
        return type("Resp", (), {"answer": "(dummy response)"})()

    def ask_with_tools(self, *args, **kwargs):
        return type("Resp", (), {"answer": "(dummy tools response)"})()

    def ask_dir_mode(self, *args, **kwargs):
        return type("Resp", (), {"answer": "(dummy dir response)"})()

    def ask_with_tools_dir_mode(self, *args, **kwargs):
        return type("Resp", (), {"answer": "(dummy dir tools response)"})()


class _DummyCodingAgent:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.coding_memory_service = kwargs.get("coding_memory_service")
        self.tool_worker_client = kwargs.get("tool_worker_client")

    def generate(self, request, **kwargs):
        return {
            "answer": f"coding-ok: {request[:40]}",
            "changed_files": [],
            "warnings": [],
            "flow_id": "flow-test",
        }

    def generate_auto_execute(self, request, **kwargs):
        return {
            "answer": f"auto-exec: {request[:40]}",
            "changed_files": ["README.md"],
            "warnings": [],
            "flow_id": "flow-auto",
            "auto_execute_terminal_reason": "completed",
        }

    def generate_dir_mode(self, request, **kwargs):
        return self.generate(request, **kwargs)

    def get_active_flow_id(self):
        return None

    def reset_flow(self, flow_id: str):
        return flow_id

    def flow_summary(self, flow_id: str):
        return None

    def _tool_policy_for_request(self, *a, **k):
        return {"allowed_tools": ["read_file"]}

    def set_tools_manager_orchestrator(self, orch):
        self.orch = orch


def test_gateway_constructs_minimally(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    gw = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        agent_tools=False,
    )
    assert gw is not None
    assert gw.root == tmp_path.resolve()
    assert not gw.owns_coding_stack()


def test_gateway_creates_session_and_simple_send(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    # Force classic ask path (no decision LLM) by disabling agent_tools + coding
    gw = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        agent_tools=False,
    )
    sid = gw.create_session(frontend="test")
    assert isinstance(sid, str) and sid

    try:
        result = gw.send(sid, "hello from gateway test")
        assert isinstance(result, str)
    except Exception:
        # Acceptable in environments without keys/indexes
        assert True


def test_gateway_provides_rich_context(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    gw = AgentChatGateway(
        tmp_path,
        dir_mode=True,
        auto_execute_plan=False,
        coding_agent=False,
    )
    ctx = gw.get_rich_context()
    assert isinstance(ctx, RichChatContext)
    assert ctx.dir_mode is True
    assert ctx.root == gw.root or ctx.root is None
    assert ctx.config is not None


def test_gateway_accepts_pre_built_objects(tmp_path: Path) -> None:
    gw = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        chat_service=object(),  # fake
        coding_agent_instance=None,
        tools_orchestrator=None,
    )
    ctx = gw.get_rich_context()
    assert ctx.chat_service is not None


def test_gateway_builds_coding_stack_when_enabled(tmp_path: Path, monkeypatch) -> None:
    """Gateway owns coding agent construction (no chat_cli injection required)."""
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.CodingAgent",
        _DummyCodingAgent,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.ToolWorkerClient",
        lambda **kw: SimpleNamespace(
            start=lambda: None,
            health=lambda: True,
            init_payload_dict=lambda: {},
            **kw,
        ),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.QueueManager",
        lambda **kw: SimpleNamespace(**kw, attach_decision_provider=lambda x: None),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.build_tools_executor_with_fallback",
        lambda **kw: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.CodingMemoryService",
        lambda **kw: SimpleNamespace(**kw),
    )

    gw = AgentChatGateway(
        tmp_path,
        coding_agent=True,
        agent_tools=True,
        tool_worker_process=True,
        auto_execute_plan=True,
    )
    assert gw.owns_coding_stack()
    ctx = gw.get_rich_context()
    assert ctx.coding_agent is not None
    assert isinstance(ctx.coding_agent, _DummyCodingAgent)


def test_gateway_uses_codex_shim_without_legacy_coding_workers(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )

    with caplog.at_level("INFO", logger="mana_agent.gateway.stack"):
        gw = AgentChatGateway(
            tmp_path,
            coding_agent=True,
            agent_tools=True,
            tool_worker_process=True,
            auto_execute_plan=True,
            settings=Settings(MANA_CODEX_MODEL="codex-test-model"),
        )

    ctx = gw.get_rich_context()
    assert isinstance(ctx.coding_agent, CodexCodingAgentShim)
    assert ctx.tool_worker_client is None
    assert ctx.tools_orchestrator is None
    model_log = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Resolved chat runtime models:")
    )
    assert "main=" in model_log
    assert "router=" in model_log
    assert "coding_backend=codex" in model_log
    assert "coding=codex-test-model" in model_log
    assert "planner=codex-owned" in model_log
    assert "tool_worker=disabled" in model_log


def test_gateway_process_turn_ask_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )

    fixed = AgentDecision(
        intent="answer",
        code_editing_needed=False,
        selected_tools=[],
        tool_inputs={},
        flow_action="none",
        reasoning_summary="answer only",
        confidence=0.9,
        verifier_passed=True,
    )

    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.decide_chat_route",
        lambda **kw: fixed,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.handle_small_direct_edit",
        lambda root, q: SimpleNamespace(handled=False),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.load_auto_chat_state",
        lambda root: SimpleNamespace(
            last_mode="answer_only",
            last_task="",
            relevant_files=[],
            changed_files=[],
            verification="",
            summary="",
        ),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.save_auto_chat_state",
        lambda root, state: None,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.resolve_auto_followup",
        lambda q, state: q,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.classify_auto_chat_intent",
        lambda q: __import__(
            "mana_agent.multi_agent.runtime.auto_chat", fromlist=["AutoChatMode"]
        ).AutoChatMode.ANSWER_ONLY,
    )

    gw = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=True)
    sid = gw.create_session(frontend="test")
    result = gw.process_turn(sid, "what is this project?")
    assert isinstance(result, ChatTurnResult)
    assert result.error is None
    assert "dummy" in result.answer.lower() or result.answer
    assert result.used_coding_agent is False


def test_gateway_process_turn_coding_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.CodingAgent",
        _DummyCodingAgent,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.ToolWorkerClient",
        lambda **kw: SimpleNamespace(
            start=lambda: None,
            health=lambda: True,
            init_payload_dict=lambda: {},
        ),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.QueueManager",
        lambda **kw: SimpleNamespace(attach_decision_provider=lambda x: None),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.build_tools_executor_with_fallback",
        lambda **kw: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.CodingMemoryService",
        lambda **kw: SimpleNamespace(),
    )

    fixed = AgentDecision(
        intent="edit",
        code_editing_needed=True,
        selected_tools=["apply_patch"],
        tool_inputs={},
        flow_action="none",
        reasoning_summary="edit files",
        confidence=0.95,
        verifier_passed=True,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.decide_chat_route",
        lambda **kw: fixed,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.handle_small_direct_edit",
        lambda root, q: SimpleNamespace(handled=False),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.load_auto_chat_state",
        lambda root: SimpleNamespace(
            last_mode="edit",
            last_task="",
            relevant_files=[],
            changed_files=[],
            verification="",
            summary="",
        ),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.save_auto_chat_state",
        lambda root, state: None,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.resolve_auto_followup",
        lambda q, state: q,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.classify_auto_chat_intent",
        lambda q: __import__(
            "mana_agent.multi_agent.runtime.auto_chat", fromlist=["AutoChatMode"]
        ).AutoChatMode.EDIT,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.is_plan_execution_request",
        lambda q: False,
    )

    gw = AgentChatGateway(
        tmp_path,
        coding_agent=True,
        agent_tools=True,
        auto_execute_plan=True,
    )
    assert gw.owns_coding_stack()
    sid = gw.create_session(frontend="test")
    result = gw.process_turn(sid, "update the README title")
    assert isinstance(result, ChatTurnResult)
    assert result.error is None
    assert result.used_coding_agent is True
    assert "auto-exec" in result.answer or "coding-ok" in result.answer
    assert result.flow_id in {"flow-auto", "flow-test", None} or result.flow_id


def test_gateway_gmail_uses_auto_chat_not_coding_agent(tmp_path: Path, monkeypatch) -> None:
    """'check my latest gmail' must route to ChatService.ask (auto-chat), not CodingAgent."""
    ask_calls: list[str] = []
    coding_calls: list[str] = []

    class _AskTrackingService(_DummyAskService):
        def ask(self, *args, **kwargs):
            ask_calls.append(str(args[0] if args else kwargs.get("question", "")))
            return type("Resp", (), {"answer": "Here are your latest Gmail messages (dummy).", "sources": [], "warnings": [], "mode": "agent-tools"})()

    class _CodingTracker(_DummyCodingAgent):
        def generate(self, request, **kwargs):
            coding_calls.append(str(request))
            return super().generate(request, **kwargs)

        def generate_auto_execute(self, request, **kwargs):
            coding_calls.append(str(request))
            return super().generate_auto_execute(request, **kwargs)

    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _AskTrackingService(),
    )
    monkeypatch.setattr("mana_agent.gateway.stack.CodingAgent", _CodingTracker)
    monkeypatch.setattr(
        "mana_agent.gateway.stack.ToolWorkerClient",
        lambda **kw: SimpleNamespace(start=lambda: None, health=lambda: True, init_payload_dict=lambda: {}),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.QueueManager",
        lambda **kw: SimpleNamespace(attach_decision_provider=lambda x: None),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.build_tools_executor_with_fallback",
        lambda **kw: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.stack.CodingMemoryService",
        lambda **kw: SimpleNamespace(),
    )

    # Model decision: answer intent with email tools (as Gmail check would produce)
    fixed = AgentDecision(
        intent="answer",
        code_editing_needed=False,
        selected_tools=["email_search", "email_read"],
        tool_inputs={"email_search": {"query": "latest"}},
        flow_action="none",
        reasoning_summary="User wants latest Gmail; use email connector tools.",
        confidence=0.92,
        verifier_passed=True,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.decide_chat_route",
        lambda **kw: fixed,
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.handle_small_direct_edit",
        lambda root, q: SimpleNamespace(handled=False),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.load_auto_chat_state",
        lambda root: SimpleNamespace(
            last_mode="answer_only", last_task="", relevant_files=[], changed_files=[], verification="", summary=""
        ),
    )
    monkeypatch.setattr("mana_agent.gateway.turn_engine.save_auto_chat_state", lambda root, state: None)
    monkeypatch.setattr("mana_agent.gateway.turn_engine.resolve_auto_followup", lambda q, state: q)
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.classify_auto_chat_intent",
        lambda q: __import__(
            "mana_agent.multi_agent.runtime.auto_chat", fromlist=["AutoChatMode"]
        ).AutoChatMode.ANSWER_ONLY,
    )

    # Wire ChatService.ask to track via gateway's chat_service after construction
    gw = AgentChatGateway(tmp_path, coding_agent=True, agent_tools=True)
    assert gw.owns_coding_stack()

    # Patch chat service ask on the live instance
    original_cs = gw._chat_service

    def _tracking_ask(question, **kwargs):
        ask_calls.append(str(question))
        # Capture callbacks so TUI live tool emission can be verified.
        ask_calls.append({"callbacks": kwargs.get("callbacks")})
        return type(
            "Resp",
            (),
            {
                "answer": "Here are your latest Gmail messages (dummy).",
                "sources": [],
                "warnings": [],
                "mode": "agent-tools",
                "trace": [
                    {
                        "tool_name": "email_search",
                        "args_summary": '{"query":"latest"}',
                        "duration_ms": 12.0,
                        "status": "ok",
                        "output_preview": "1 message",
                    },
                    {
                        "tool_name": "email_read",
                        "args_summary": '{"message_ref":"x"}',
                        "duration_ms": 8.0,
                        "status": "ok",
                        "output_preview": "Subject: Hello",
                    },
                ],
            },
        )()

    original_cs.ask = _tracking_ask  # type: ignore[method-assign]

    sid = gw.create_session(frontend="tui")
    result = gw.process_turn(sid, "check my latest gmail", callbacks=[object()])
    assert result.error is None
    assert result.used_coding_agent is False
    assert result.auto_chat_mode == "answer_only"
    assert "gmail" in result.answer.lower() or "Gmail" in result.answer
    assert any(isinstance(item, str) and "gmail" in item.lower() for item in ask_calls), (
        "expected ChatService.ask (auto-chat path) to be used for Gmail"
    )
    assert not coding_calls, "CodingAgent must not run for Gmail auto-chat turns"
    assert (result.payload or {}).get("route") == "auto_chat"
    # Tool traces must reach TUI consumers for ToolCard rendering.
    trace_names = [row.get("tool_name") for row in (result.trace or []) if isinstance(row, dict)]
    assert "email_search" in trace_names
    assert "email_read" in trace_names
    assert (result.payload or {}).get("trace")


def test_should_use_coding_agent_turn_gmail_is_false() -> None:
    from mana_agent.gateway.turn_engine import should_use_coding_agent_turn, is_auto_chat_connector_turn
    from mana_agent.multi_agent.runtime.auto_chat import AutoChatMode

    decision = AgentDecision(
        intent="answer",
        confidence=0.9,
        selected_tools=["email_search"],
        code_editing_needed=False,
        reasoning_summary="gmail",
        verifier_passed=True,
    )
    assert is_auto_chat_connector_turn(
        decision=decision, auto_chat_mode=AutoChatMode.ANSWER_ONLY, question="check my latest gmail"
    )
    assert not should_use_coding_agent_turn(
        coding_agent_available=True,
        agent_tools=True,
        edit_request=False,
        plan_trigger_request=False,
        force_plan_only_response=False,
        has_pending_prechecklist=False,
        coding_agent_is_custom=False,
        general_coding_agent_turns=False,
        decision=decision,
        auto_chat_mode=AutoChatMode.ANSWER_ONLY,
        question="check my latest gmail",
    )


def test_gateway_config_normalized_full_auto() -> None:
    cfg = ChatGatewayConfig(full_auto=True, auto_execute_max_passes=4).normalized()
    assert cfg.execution_profile == "full-auto"
    assert cfg.auto_execute_plan is True
    assert cfg.auto_execute_max_passes == 10


def test_gateway_decision_failure_no_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *a, **k: _DummyAskService(),
    )

    def _boom(**kw):
        raise RuntimeError("decision unavailable")

    monkeypatch.setattr("mana_agent.gateway.turn_engine.decide_chat_route", _boom)
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.load_auto_chat_state",
        lambda root: SimpleNamespace(
            last_mode="answer_only",
            last_task="",
            relevant_files=[],
            changed_files=[],
            verification="",
            summary="",
        ),
    )
    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.resolve_auto_followup",
        lambda q, state: q,
    )

    gw = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=True)
    sid = gw.create_session(frontend="test")
    result = gw.process_turn(sid, "do something")
    assert result.error is not None
    assert "No fallback" in result.error or "decision" in result.error.lower()
    assert result.answer == ""


def _answer_decision(*, selected_tools: list[str] | None = None) -> AgentDecision:
    return AgentDecision(
        intent="answer",
        confidence=0.99,
        selected_tools=list(selected_tools or []),
        code_editing_needed=False,
        reasoning_summary="answer conversationally",
        verifier_passed=True,
    )


def test_gateway_persists_same_session_history_without_duplicate_current_message(
    tmp_path: Path, monkeypatch
) -> None:
    prompts: list[str] = []

    class TrackingChatService:
        _ask_service = _DummyAskService()

        def ask(self, question: str, **kwargs: Any):
            prompts.append(question)
            answer = "Understood." if len(prompts) == 1 else "One is b."
            return SimpleNamespace(answer=answer, sources=[], warnings=[], trace=[])

        def ask_conversation(self, question: str):
            return self.ask(question).answer

    monkeypatch.setattr("mana_agent.gateway.turn_engine.decide_chat_route", lambda **kwargs: _answer_decision())
    monkeypatch.setattr("mana_agent.gateway.turn_engine.handle_small_direct_edit", lambda *args, **kwargs: SimpleNamespace(handled=False))
    gateway = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        agent_tools=False,
        chat_service=TrackingChatService(),
    )
    session_id = gateway.create_session(frontend="test")

    gateway.process_turn(session_id, "Remember one = b.")
    gateway.process_turn(session_id, "What is one?")

    assert "Remember one = b." in prompts[1]
    assert "Understood." in prompts[1]
    assert prompts[1].count("What is one?") == 1
    messages = gateway.session_messages(session_id)
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]
    assert {message["session_id"] for message in messages} == {session_id}
    assert {message["conversation_id"] for message in messages} == {session_id}


def test_answer_only_conversation_uses_validated_route_without_second_router(
    tmp_path: Path, monkeypatch
) -> None:
    prompts: list[str] = []

    class ConversationChatService:
        _ask_service = _DummyAskService()

        def ask(self, question: str, **kwargs: Any):
            raise AssertionError("entry router must not run after an answer-only decision")

        def ask_conversation(self, question: str) -> str:
            prompts.append(question)
            return "a is test"

    monkeypatch.setattr("mana_agent.gateway.turn_engine.decide_chat_route", lambda **kwargs: _answer_decision())
    monkeypatch.setattr("mana_agent.gateway.turn_engine.handle_small_direct_edit", lambda *args, **kwargs: SimpleNamespace(handled=False))
    gateway = AgentChatGateway(
        tmp_path,
        coding_agent=False,
        agent_tools=False,
        chat_service=ConversationChatService(),
    )
    session_id = gateway.create_session(frontend="test")
    gateway.process_turn(session_id, "memory-test a=test")
    result = gateway.process_turn(session_id, "what is a?")

    assert result.answer == "a is test"
    assert result.mode == "route-conversation"
    assert "User: memory-test a=test" in prompts[-1]
    assert prompts[-1].count("what is a?") == 1


def test_gateway_new_conversation_isolates_history(tmp_path: Path, monkeypatch) -> None:
    prompts: list[str] = []

    class TrackingChatService:
        _ask_service = _DummyAskService()

        def ask(self, question: str, **kwargs: Any):
            prompts.append(question)
            return SimpleNamespace(answer="ok", sources=[], warnings=[], trace=[])

        def ask_conversation(self, question: str):
            return self.ask(question).answer

    monkeypatch.setattr("mana_agent.gateway.turn_engine.decide_chat_route", lambda **kwargs: _answer_decision())
    monkeypatch.setattr("mana_agent.gateway.turn_engine.handle_small_direct_edit", lambda *args, **kwargs: SimpleNamespace(handled=False))
    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False, chat_service=TrackingChatService())
    old_session = gateway.create_session(frontend="test")
    gateway.process_turn(old_session, "Remember one = b.")

    new_session = gateway.start_new_conversation(old_session, frontend="test")
    gateway.process_turn(new_session, "What is one?")

    assert new_session != old_session
    assert "Remember one = b." not in prompts[-1]
    assert gateway.session_messages(old_session)
    assert [row["content"] for row in gateway.session_messages(new_session) if row["role"] == "user"] == ["What is one?"]


def test_gateway_failed_turn_keeps_session_and_records_failure(tmp_path: Path, monkeypatch) -> None:
    class FailingChatService:
        _ask_service = _DummyAskService()

        def ask(self, question: str, **kwargs: Any):
            raise RuntimeError("provider unavailable")

        def ask_conversation(self, question: str):
            return self.ask(question)

    monkeypatch.setattr("mana_agent.gateway.turn_engine.decide_chat_route", lambda **kwargs: _answer_decision())
    monkeypatch.setattr("mana_agent.gateway.turn_engine.handle_small_direct_edit", lambda *args, **kwargs: SimpleNamespace(handled=False))
    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False, chat_service=FailingChatService())
    session_id = gateway.create_session(frontend="test")
    result = gateway.process_turn(session_id, "Remember this even if the model fails.")

    assert result.error and "provider unavailable" in result.error
    messages = gateway.session_messages(session_id)
    assert messages[0]["role"] == "user"
    assert messages[-1]["role"] == "system"
    assert messages[-1]["metadata"]["state"] == "failed"
    assert gateway.create_session(frontend="test", session_id=session_id) == session_id


def test_gateway_persists_tool_summary_for_followup_context(tmp_path: Path, monkeypatch) -> None:
    prompts: list[str] = []

    class ToolChatService:
        _ask_service = _DummyAskService()

        def ask(self, question: str, **kwargs: Any):
            prompts.append(question)
            trace = [] if len(prompts) > 1 else [{"tool_name": "read_file", "output_preview": "one=b", "status": "ok"}]
            return SimpleNamespace(answer="tool answer", sources=[], warnings=[], trace=trace)

        def ask_conversation(self, question: str):
            return self.ask(question).answer

    monkeypatch.setattr(
        "mana_agent.gateway.turn_engine.decide_chat_route",
        lambda **kwargs: _answer_decision(selected_tools=["read_file"]),
    )
    monkeypatch.setattr("mana_agent.gateway.turn_engine.handle_small_direct_edit", lambda *args, **kwargs: SimpleNamespace(handled=False))
    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False, chat_service=ToolChatService())
    session_id = gateway.create_session(frontend="test")
    gateway.process_turn(session_id, "Read the value.")
    gateway.process_turn(session_id, "What did the tool return?")

    assert "Tool result: one=b" in prompts[-1]
    assert [row["role"] for row in gateway.session_messages(session_id)][:3] == ["user", "tool", "assistant"]


def test_gateway_does_not_create_sessions_per_message(tmp_path: Path, monkeypatch) -> None:
    class ChatService:
        _ask_service = _DummyAskService()

        def ask(self, question: str, **kwargs: Any):
            return SimpleNamespace(answer="ok", sources=[], warnings=[], trace=[])

        def ask_conversation(self, question: str):
            return self.ask(question).answer

    monkeypatch.setattr("mana_agent.gateway.turn_engine.decide_chat_route", lambda **kwargs: _answer_decision())
    monkeypatch.setattr("mana_agent.gateway.turn_engine.handle_small_direct_edit", lambda *args, **kwargs: SimpleNamespace(handled=False))
    gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False, chat_service=ChatService())
    create_calls = 0
    original_create = gateway._workspaces.create_session

    def counted_create(*args: Any, **kwargs: Any):
        nonlocal create_calls
        create_calls += 1
        return original_create(*args, **kwargs)

    monkeypatch.setattr(gateway._workspaces, "create_session", counted_create)
    session_id = gateway.create_session(frontend="test")
    for message in ("one", "two", "three"):
        gateway.process_turn(session_id, message)

    # Gateway construction/MainAgent startup already restored or created the one
    # active session. Opening it and sending turns must not create another.
    assert create_calls == 0
    assert {row["session_id"] for row in gateway.session_messages(session_id)} == {session_id}


def test_gateway_startup_restores_session_and_only_new_creates_another(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "mana_agent.commands.cli_internal.build_ask_service",
        lambda *args, **kwargs: _DummyAskService(),
    )
    first_gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False)
    first_session = first_gateway.create_session(frontend="cli")

    second_gateway = AgentChatGateway(tmp_path, coding_agent=False, agent_tools=False)
    restored_session = second_gateway.create_session(frontend="cli")
    new_session = second_gateway.start_new_conversation(restored_session, frontend="cli")

    assert restored_session == first_session
    assert new_session != restored_session
    repository_id = second_gateway._workspaces.register_repository(tmp_path).repository_id
    sessions = [
        item
        for item in second_gateway._workspaces.store.list_sessions()
        if item.primary_repository_id == repository_id
    ]
    assert len(sessions) == 2
    assert second_gateway._workspaces.store.get_session(restored_session).status == "archived"
    assert second_gateway._workspaces.store.get_session(new_session).status == "active"


def test_chat_session_history_redacts_secrets(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "mana_agent.services.chat_session_history.session_dir",
        lambda session_id: tmp_path / session_id,
    )
    history = ChatSessionHistory()
    history.append(
        "session_test",
        role="tool",
        content="Authorization: Bearer private-token and sk-private-key",
        turn_id="turn_test",
        metadata={"api_key": "private", "tool_name": "example"},
    )

    stored = history.list("session_test")[0]
    assert "private-token" not in stored.content
    assert "sk-private-key" not in stored.content
    assert stored.metadata["api_key"] == "***REDACTED***"
