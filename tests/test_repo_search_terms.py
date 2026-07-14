from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from mana_agent.analysis.models import AskResponseWithTrace
from mana_agent.multi_agent.routing.agent_decision import AgentDecision, verify_agent_decision
from mana_agent.multi_agent.routing.repo_search_terms import (
    RepoSearchTermsDecision,
    RepoSearchTermsDecisionEngine,
    RepoSearchTermsDecisionError,
    project_search_with_terms,
    resolve_repo_search_terms,
    validate_repo_search_terms_decision,
)
from mana_agent.multi_agent.runtime.entry_router import RouteDecision
from mana_agent.multi_agent.runtime.route_executor import RouteExecutionContext, RouteExecutor


class _FakeQnA:
    def run(self, question: str, context: str) -> str:
        return f"answer for {question} using {context[:40]}"


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list = []

    def invoke(self, messages):
        self.messages.append(messages)
        return AIMessage(content=self.content)


class _StaticRouter:
    def __init__(self) -> None:
        self.llm = None
        self.router_model = "fake-router"


def _write_socket_repo(tmp_path: Path) -> Path:
    (tmp_path / "net.py").write_text(
        "import socket\nfrom websockets import WebSocket\n",
        encoding="utf-8",
    )
    return tmp_path


def test_validate_rejects_full_user_message_as_term() -> None:
    with pytest.raises(ValueError, match="must not equal the full user message"):
        validate_repo_search_terms_decision(
            {
                "terms": ["this project use socket?"],
                "confidence": 0.9,
                "reason": "bad whole message",
            },
            user_request="this project use socket?",
        )


def test_validate_allows_compact_single_token_equal_to_request() -> None:
    decision = validate_repo_search_terms_decision(
        {"terms": ["socket"], "confidence": 0.8, "reason": "compact token"},
        user_request="socket",
    )
    assert decision.terms == ["socket"]


def test_model_terms_engine_returns_compact_needles() -> None:
    llm = _FakeLLM(
        json.dumps(
            {
                "terms": ["socket", "websocket", "WebSocket"],
                "confidence": 0.93,
                "reason": "user asks about socket usage",
                "fixed_strings": True,
            }
        )
    )
    decision = RepoSearchTermsDecisionEngine(llm=llm).decide(
        user_request="this project use socket?"
    )
    assert decision.terms == ["socket", "websocket", "WebSocket"]
    assert decision.source == "model"


def test_model_terms_engine_fails_without_llm() -> None:
    with pytest.raises(RepoSearchTermsDecisionError, match="term decision model is unavailable"):
        RepoSearchTermsDecisionEngine(llm=None).decide(user_request="socket?")


def test_resolve_prefers_tool_inputs_and_never_uses_whole_message(tmp_path: Path) -> None:
    decision = resolve_repo_search_terms(
        user_request="does mana-agent use socket?",
        tool_inputs={"repo_search": {"terms": ["socket", "websocket"]}},
        llm=None,
    )
    assert decision.terms == ["socket", "websocket"]
    assert decision.source == "tool_inputs"

    with pytest.raises(RepoSearchTermsDecisionError):
        resolve_repo_search_terms(
            user_request="does mana-agent use socket?",
            tool_inputs={"repo_search": {"query": "does mana-agent use socket?"}},
            llm=None,
        )


def test_project_search_with_terms_finds_socket_code(tmp_path: Path) -> None:
    root = _write_socket_repo(tmp_path)
    result = project_search_with_terms(["socket", "websocket"], root, max_results=20)
    assert result.matches
    assert any("socket" in match.line_text.lower() for match in result.matches)


def test_route_executor_repo_search_uses_model_terms_not_full_question(tmp_path: Path) -> None:
    root = _write_socket_repo(tmp_path)

    class _TermsEngine:
        def decide(self, *, user_request: str):
            assert "socket" in user_request
            # Must not search the full sentence.
            assert user_request != "socket"
            return RepoSearchTermsDecision(
                terms=["socket", "websocket"],
                confidence=0.95,
                reason="model extracted socket-related terms",
                source="test",
            )

    executor = RouteExecutor(
        router=_StaticRouter(),  # type: ignore[arg-type]
        store=None,
        qna_chain=_FakeQnA(),
        project_root=root,
        repo_search_terms_engine=_TermsEngine(),  # type: ignore[arg-type]
    )
    response = executor._repo_search(
        RouteExecutionContext(
            question="this project use socket?",
            index_dir=None,
            project_root=root,
            k=5,
        )
    )
    assert isinstance(response, AskResponseWithTrace)
    assert response.mode == "route-repo_search"
    assert response.sources
    assert "No direct project matches" not in response.answer
    assert any(item.tool_name == "repo_search_terms" for item in response.trace)


def test_route_executor_repo_search_stops_when_terms_decision_missing(tmp_path: Path) -> None:
    root = _write_socket_repo(tmp_path)
    executor = RouteExecutor(
        router=_StaticRouter(),  # type: ignore[arg-type]
        store=None,
        qna_chain=_FakeQnA(),
        project_root=root,
    )
    response = executor._repo_search(
        RouteExecutionContext(
            question="this project use socket?",
            index_dir=None,
            project_root=root,
            k=5,
        )
    )
    assert "Model decision failed: repo_search_terms" in response.answer
    assert response.sources == []
    assert any("no repository search was executed" in warning for warning in response.warnings)


def test_agent_decision_requires_compact_repo_search_terms() -> None:
    missing = AgentDecision(
        intent="repo_search",
        confidence=0.9,
        selected_tools=["repo_search"],
        tool_inputs={},
        repo_context_needed=True,
    )
    missing_result = verify_agent_decision(missing, user_request="this project use socket?")
    assert missing_result.passed is False
    assert any("compact model-selected query/terms" in item for item in missing_result.warnings)

    whole = AgentDecision(
        intent="repo_search",
        confidence=0.9,
        selected_tools=["repo_search"],
        tool_inputs={"repo_search": {"query": "this project use socket?"}},
        repo_context_needed=True,
    )
    whole_result = verify_agent_decision(whole, user_request="this project use socket?")
    assert whole_result.passed is False
    assert any("invalid" in item for item in whole_result.warnings)

    good = AgentDecision(
        intent="repo_search",
        confidence=0.9,
        selected_tools=["repo_search"],
        tool_inputs={"repo_search": {"terms": ["socket", "websocket"]}},
        repo_context_needed=True,
    )
    good_result = verify_agent_decision(good, user_request="this project use socket?")
    assert good_result.passed is True


def test_entry_route_decision_object_still_supports_repo_search_kind() -> None:
    decision = RouteDecision(kind="repo_search", confidence=0.8, reason="inspect repo files")
    assert decision.kind == "repo_search"
