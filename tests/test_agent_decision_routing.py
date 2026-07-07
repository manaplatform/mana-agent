from __future__ import annotations

import json

from mana_agent.multi_agent.routing.agent_decision import AgentDecisionEngine
from mana_agent.multi_agent.routing.router import Router


class _DecisionModel:
    def __init__(self, payloads: dict[str, dict]) -> None:
        self.payloads = payloads

    def invoke(self, messages):  # noqa: ANN001
        body = json.loads(messages[-1].content)
        request = body["user_request"]
        return type("Msg", (), {"content": json.dumps(self.payloads[request])})()


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


def test_agent_decision_routes_latest_docs_to_web_search() -> None:
    decision = _engine().decide(user_request="latest OpenAI API docs")
    assert decision.selected_tools == ["web_search"]
    assert decision.web_search_needed is True


def test_router_uses_agent_decision_for_research_route() -> None:
    route = Router(decision_engine=_engine()).route(
        task_id="task_1",
        user_request="search internet and give me description about openclaw",
    )
    assert route.route_name == "research"
    assert route.required_capabilities == ["web_search", "summarization"]
