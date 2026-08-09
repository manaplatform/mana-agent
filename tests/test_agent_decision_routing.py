from __future__ import annotations

import json
from pathlib import Path

import pytest

from mana_agent.commands.chat_cli import _decide_chat_route
from mana_agent.multi_agent.routing.agent_decision import AgentDecision, AgentDecisionEngine, verify_agent_decision
from mana_agent.multi_agent.routing.router import Router, RoutingDecisionError


def _semantic_contract(intent: str) -> dict[str, str]:
    return {
        "answer": {"requested_effect": "none", "target_surface": "conversation"},
        "repo_search": {"requested_effect": "read", "target_surface": "repository"},
        "analyze": {"requested_effect": "read", "target_surface": "repository"},
        "review": {"requested_effect": "read", "target_surface": "repository"},
        "edit": {"requested_effect": "write", "target_surface": "repository"},
        "web_research": {"requested_effect": "read", "target_surface": "web"},
        "tool": {"requested_effect": "execute", "target_surface": "local_system"},
        "verify": {"requested_effect": "execute", "target_surface": "local_system"},
        "high_risk_tool": {"requested_effect": "execute", "target_surface": "local_system"},
        "plan": {"requested_effect": "none", "target_surface": "conversation"},
    }[intent]


class _DecisionModel:
    def __init__(self, payloads: dict[str, dict]) -> None:
        self.payloads = payloads

    def invoke(self, messages):  # noqa: ANN001
        body = json.loads(messages[-1].content)
        request = body["user_request"]
        payload = dict(self.payloads[request])
        payload.update({key: value for key, value in _semantic_contract(payload["intent"]).items() if key not in payload})
        payload.setdefault("required_subagents", [])
        return type("Msg", (), {"content": json.dumps(payload)})()


def _engine() -> AgentDecisionEngine:
    return AgentDecisionEngine(
        llm=_DecisionModel(
            {
                "search internet and give me description about openclaw": {
                    "intent": "web_research",
                    "confidence": 0.91,
                    "selected_tools": ["web_search"],
                    "tool_inputs": {"web_search": {"query": "openclaw description"}},
                    "repo_context_needed": False,
                    "web_search_needed": True,
                    "code_editing_needed": False,
                    "reasoning_summary": "The user asks for public information about an unknown topic.",
                },
                "what is openclaw?": {
                    "intent": "web_research",
                    "confidence": 0.84,
                    "selected_tools": ["web_search"],
                    "tool_inputs": {"web_search": {"query": "openclaw"}},
                    "repo_context_needed": False,
                    "web_search_needed": True,
                    "code_editing_needed": False,
                    "reasoning_summary": "The term is not tied to the local repository.",
                },
                "search repo for openclaw": {
                    "intent": "repo_search",
                    "confidence": 0.88,
                    "selected_tools": ["repo_search"],
                    "tool_inputs": {"repo_search": {"query": "openclaw"}},
                    "repo_context_needed": True,
                    "web_search_needed": False,
                    "code_editing_needed": False,
                    "reasoning_summary": "The user explicitly wants local repository matches.",
                },
                "find ToolManager in this repo": {
                    "intent": "repo_search",
                    "confidence": 0.89,
                    "selected_tools": ["repo_search"],
                    "tool_inputs": {"repo_search": {"query": "ToolManager"}},
                    "repo_context_needed": True,
                    "web_search_needed": False,
                    "code_editing_needed": False,
                    "reasoning_summary": "The request asks for a symbol in this repository.",
                },
                "fix the search router": {
                    "intent": "edit",
                    "confidence": 0.92,
                    "selected_tools": ["repo_search", "read_file", "apply_patch"],
                    "tool_inputs": {"repo_search": {"query": "search router"}},
                    "repo_context_needed": True,
                    "web_search_needed": False,
                    "code_editing_needed": True,
                    "reasoning_summary": "The user wants a local code change.",
                },
                "latest OpenAI API docs": {
                    "intent": "web_research",
                    "confidence": 0.93,
                    "selected_tools": ["web_search"],
                    "tool_inputs": {"web_search": {"query": "latest OpenAI API docs"}},
                    "repo_context_needed": False,
                    "web_search_needed": True,
                    "code_editing_needed": False,
                    "reasoning_summary": "Latest public documentation requires web research.",
                },
                "search internet & github for hermes-agent and describe to me.": {
                    "intent": "web_research",
                    "confidence": 0.94,
                    "selected_tools": ["web_search", "github_search"],
                    "tool_inputs": {
                        "web_search": {"query": "hermes-agent"},
                        "github_search": {"query": "hermes-agent", "github_kind": "repositories"},
                    },
                    "repo_context_needed": False,
                    "web_search_needed": True,
                    "code_editing_needed": False,
                    "reasoning_summary": "The user explicitly asks for internet and GitHub research.",
                },
            }
        )
    )


def test_agent_decision_routes_internet_description_to_web_search() -> None:
    decision = _engine().decide(user_request="search internet and give me description about openclaw")
    assert decision.intent == "web_research"
    assert decision.selected_tools == ["web_search"]
    assert decision.web_search_needed is True
    assert decision.repo_context_needed is False
    assert decision.verifier_passed is True


def test_agent_decision_routes_unknown_what_is_question_to_web_search() -> None:
    decision = _engine().decide(user_request="what is openclaw?")
    assert decision.selected_tools == ["web_search"]
    assert decision.web_search_needed is True


def test_agent_decision_routes_explicit_repo_search_to_repo_search() -> None:
    decision = _engine().decide(user_request="search repo for openclaw")
    assert decision.selected_tools == ["repo_search"]
    assert decision.repo_context_needed is True
    assert decision.web_search_needed is False


def test_agent_decision_routes_repo_symbol_find_to_repo_search() -> None:
    decision = _engine().decide(user_request="find ToolManager in this repo")
    assert decision.selected_tools == ["repo_search"]
    assert decision.tool_inputs["repo_search"]["query"] == "ToolManager"


def test_agent_decision_routes_fix_request_to_repo_read_and_patch_tools() -> None:
    decision = _engine().decide(user_request="fix the search router")
    assert decision.selected_tools == ["repo_search", "read_file", "apply_patch"]
    assert decision.repo_context_needed is True
    assert decision.code_editing_needed is True


def test_agent_decision_selects_new_flow_for_distinct_repository_work() -> None:
    payload = {
        "intent": "edit",
        "confidence": 0.94,
        "selected_tools": ["repo_search", "read_file", "apply_patch"],
        "tool_inputs": {"repo_search": {"query": "telegram connector"}},
        "repo_context_needed": True,
        "web_search_needed": False,
        "code_editing_needed": True,
        "flow_action": "new",
        "reasoning_summary": "This is distinct from the active documentation flow.",
    }
    engine = AgentDecisionEngine(llm=_DecisionModel({"fix telegram polling": payload}))

    decision = engine.decide(
        user_request="fix telegram polling",
        memory_context='Active coding flow: {"objective":"update README"}',
    )

    assert decision.flow_action == "new"
    assert decision.verifier_passed is True


def test_agent_decision_rejects_missing_flow_action_for_active_edit() -> None:
    payload = {
        "intent": "edit",
        "confidence": 0.94,
        "selected_tools": ["repo_search", "read_file", "apply_patch"],
        "tool_inputs": {"repo_search": {"query": "telegram connector"}},
        "repo_context_needed": True,
        "web_search_needed": False,
        "code_editing_needed": True,
        "reasoning_summary": "Repository edit selected without a flow decision.",
    }
    engine = AgentDecisionEngine(llm=_DecisionModel({"fix telegram polling": payload}))

    decision = engine.decide(
        user_request="fix telegram polling",
        memory_context='Active coding flow: {"objective":"update README"}',
    )

    assert decision.flow_action == "none"
    assert decision.verifier_passed is False
    assert "requires flow_action" in decision.verifier_summary


def test_agent_decision_routes_latest_docs_to_web_search() -> None:
    decision = _engine().decide(user_request="latest OpenAI API docs")
    assert decision.selected_tools == ["web_search"]
    assert decision.web_search_needed is True


def test_agent_decision_routes_internet_and_github_to_both_external_tools() -> None:
    decision = _engine().decide(user_request="search internet & github for hermes-agent and describe to me.")
    assert decision.intent == "web_research"
    assert decision.selected_tools == ["web_search", "github_search"]
    assert decision.web_search_needed is True
    assert decision.verifier_passed is True


def test_agent_decision_without_model_stops_without_static_route() -> None:
    decision = AgentDecisionEngine(llm=None).decide(user_request="fix the search router")

    assert decision.intent == "answer"
    assert decision.confidence == 0.0
    assert decision.selected_tools == []
    assert decision.source == "model_unavailable"
    assert decision.verifier_passed is False
    assert "fallback" not in decision.reasoning_summary.lower()


def test_router_uses_agent_decision_for_research_route() -> None:
    route = Router(decision_engine=_engine()).route(
        task_id="task_1",
        user_request="search internet and give me description about openclaw",
    )
    assert route.route_name == "research"
    assert route.required_capabilities == ["web_search", "summarization"]


def test_router_preserves_github_search_capability() -> None:
    route = Router(decision_engine=_engine()).route(
        task_id="task_1",
        user_request="search internet & github for hermes-agent and describe to me.",
    )
    assert route.route_name == "research"
    assert route.required_capabilities == ["web_search", "github_search", "summarization"]


def test_router_keeps_entrypoint_metadata_out_of_semantic_user_text() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.payloads: list[dict] = []

        def invoke(self, messages):  # noqa: ANN001
            payload = json.loads(messages[-1].content)
            self.payloads.append(payload)
            return type(
                "Msg",
                (),
                {"content": json.dumps({
                    "intent": "answer",
                    "confidence": 0.9,
                    "selected_tools": [],
                    "tool_inputs": {},
                    "repo_context_needed": False,
                    "web_search_needed": False,
                    "code_editing_needed": False,
                    "requested_effect": "none",
                    "target_surface": "conversation",
                    "required_subagents": [],
                    "reasoning_summary": "Plain conversation.",
                })},
            )()

    model = RecordingModel()
    router = Router(decision_engine=AgentDecisionEngine(llm=model))
    for entrypoint, request in (
        ("chat", "chat command"),
        ("cli", "cli command is broken"),
        ("api", "api routing bug"),
    ):
        assert router.route(task_id=f"task-{entrypoint}", user_request=request, command_hint=entrypoint).route_name == "simple"
    decision_payloads = [payload for payload in model.payloads if "tools" in payload]
    assert [(payload["user_request"], payload["command_hint"]) for payload in decision_payloads] == [
        ("chat command", "chat"),
        ("cli command is broken", "cli"),
        ("api routing bug", "api"),
    ]


def test_router_keeps_coding_request_semantic_text_when_entered_through_chat() -> None:
    class CodingModel:
        def __init__(self) -> None:
            self.decision_payload: dict | None = None

        def invoke(self, messages):  # noqa: ANN001
            payload = json.loads(messages[-1].content)
            if "tools" in payload:
                self.decision_payload = payload
            return type(
                "Msg",
                (),
                {"content": json.dumps({
                    "intent": "edit",
                    "confidence": 0.95,
                    "selected_tools": ["repo_search", "read_file", "apply_patch"],
                    "tool_inputs": {"repo_search": {"query": "chat command routing"}},
                    "repo_context_needed": True,
                    "web_search_needed": False,
                    "code_editing_needed": True,
                    "requested_effect": "write",
                    "target_surface": "repository",
                    "required_subagents": [],
                    "reasoning_summary": "Repository routing code needs a mutation.",
                })},
            )()

    model = CodingModel()
    route = Router(decision_engine=AgentDecisionEngine(llm=model)).route(
        task_id="task-coding",
        user_request="fix the chat command routing bug",
        command_hint="chat",
    )

    assert route.route_name == "coding"
    assert model.decision_payload is not None
    assert model.decision_payload["user_request"] == "fix the chat command routing bug"
    assert model.decision_payload["command_hint"] == "chat"


def test_router_rejects_invalid_semantic_decision_instead_of_using_simple_route() -> None:
    decision = AgentDecision(
        intent="answer",
        confidence=0.9,
        selected_tools=[],
        requested_effect="write",
        target_surface="repository",
    )
    assert verify_agent_decision(decision, user_request="fix the routing bug").passed is False

    class InvalidDecisionEngine:
        def decide(self, **_kwargs):  # noqa: ANN003
            decision.verifier_passed = False
            decision.verifier_summary = "write+repository requires intent=edit and code_editing_needed=true"
            return decision

    with pytest.raises(RoutingDecisionError, match="No route was executed"):
        Router(decision_engine=InvalidDecisionEngine()).route(
            task_id="task-invalid",
            user_request="fix the routing bug",
            command_hint="chat",
        )


def test_router_blocks_when_model_omits_required_semantic_contract() -> None:
    class IncompleteModel:
        def invoke(self, _messages):  # noqa: ANN001
            return type(
                "Msg",
                (),
                {"content": json.dumps({
                    "intent": "answer",
                    "confidence": 0.9,
                    "selected_tools": [],
                    "tool_inputs": {},
                    "repo_context_needed": False,
                    "web_search_needed": False,
                    "code_editing_needed": False,
                    "reasoning_summary": "Missing the semantic contract.",
                })},
            )()

    with pytest.raises(RoutingDecisionError, match="routing model decision is unavailable"):
        Router(decision_engine=AgentDecisionEngine(llm=IncompleteModel())).route(
            task_id="task-incomplete",
            user_request="hello",
            command_hint="chat",
        )


def test_chat_route_uses_direct_agent_decision_without_search_repair() -> None:
    class AnswerModel:
        def invoke(self, _messages):  # noqa: ANN001
            return type(
                "Msg",
                (),
                {
                    "content": json.dumps(
                        {
                            "intent": "answer",
                            "confidence": 0.7,
                            "selected_tools": [],
                            "tool_inputs": {},
                            "repo_context_needed": False,
                            "web_search_needed": False,
                            "code_editing_needed": False,
                            "requested_effect": "none",
                            "target_surface": "conversation",
                            "required_subagents": [],
                            "reasoning_summary": "Answered through the standard chat service path.",
                        }
                    )
                },
            )()

    ask_service = type("AskService", (), {"ask_agent": type("AskAgent", (), {"llm": AnswerModel()})()})()
    decision = _decide_chat_route(
        ask_service=ask_service,
        question="search internet and give me description about hermes-agent",
        root=Path("/repo"),
    )
    assert decision.intent == "answer"
    assert decision.selected_tools == []
    assert decision.source == "model"
