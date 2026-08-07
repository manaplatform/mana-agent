from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mana_agent.gateway.turn_engine import (
    SearchOperationDecisionError,
    decide_search_operation,
    is_valid_search_operation_decision,
    run_web_research_answer,
)
from mana_agent.multi_agent.routing.agent_decision import AgentDecision


def test_web_research_executes_only_the_model_selected_compact_query(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    captured: dict[str, object] = {}

    class Router:
        def __init__(self, **_kwargs) -> None:  # noqa: ANN001
            pass

        def run(self, **kwargs):  # noqa: ANN003
            captured["decision"] = kwargs["decision_override"]
            return SimpleNamespace(
                context_block=lambda **_kwargs: "External Search Context: result",
                results=[],
                memory_hits=[],
                warnings=[],
            )

    monkeypatch.setattr("mana_agent.gateway.turn_engine.SearchRouter", Router)
    decision = AgentDecision(
        intent="web_research",
        confidence=1.0,
        selected_tools=["web_search"],
        tool_inputs={"web_search": {"query": "latest OpenAI model"}},
        web_search_needed=True,
    )

    answer, _sources, _trace = run_web_research_answer(
        ask_service=SimpleNamespace(qna_chain=SimpleNamespace(run=lambda **_kwargs: "answer")),
        question="Active conversation history that must not be sent to Tavily",
        root=tmp_path,
        decision=decision,
    )

    assert answer == "answer"
    assert captured["decision"].queries[0].query == "latest OpenAI model"


def test_web_research_stops_when_the_model_omits_a_query(tmp_path) -> None:
    decision = AgentDecision(
        intent="web_research",
        confidence=1.0,
        selected_tools=["web_search"],
        web_search_needed=True,
    )

    with pytest.raises(SearchOperationDecisionError, match="web_search.query"):
        run_web_research_answer(
            ask_service=SimpleNamespace(),
            question="A long conversation transcript is not a valid search query.",
            root=tmp_path,
            decision=decision,
        )


def test_search_operation_uses_a_dedicated_query_decision(tmp_path) -> None:
    class Model:
        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        def invoke(self, messages):  # noqa: ANN001
            payload = json.loads(messages[-1].content)
            self.payloads.append(payload)
            assert payload["required_tool"] == "web_search"
            assert "max_query_chars" in payload
            system = str(messages[0].content)
            assert "already selected the external search tool" in system
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "query": "latest OpenAI API docs",
                        "reasoning_summary": "Current documentation requires public research.",
                    }
                )
            )

    model = Model()
    decision = decide_search_operation(
        ask_service=SimpleNamespace(ask_agent=SimpleNamespace(llm=model)),
        question="What are the latest OpenAI API docs?",
        root=tmp_path,
        required_tool="web_search",
    )

    assert decision.verifier_passed is True
    assert decision.selected_tools == ["web_search"]
    assert decision.tool_inputs == {"web_search": {"query": "latest OpenAI API docs"}}
    assert is_valid_search_operation_decision(
        decision,
        required_tool="web_search",
    )
    assert len(model.payloads) == 1


def test_search_operation_accepts_nested_tool_input_shape(tmp_path) -> None:
    class Model:
        def invoke(self, _messages):  # noqa: ANN001
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "intent": "web_research",
                        "selected_tools": ["web_search"],
                        "tool_inputs": {"web_search": {"query": "hermes agent"}},
                        "reasoning_summary": "Need public description.",
                    }
                )
            )

    decision = decide_search_operation(
        ask_service=SimpleNamespace(ask_agent=SimpleNamespace(llm=Model())),
        question="check what is hermes agent.",
        root=tmp_path,
        required_tool="web_search",
    )

    assert decision.tool_inputs == {"web_search": {"query": "hermes agent"}}
    assert is_valid_search_operation_decision(decision, required_tool="web_search")


def test_search_operation_accepts_string_tool_payload(tmp_path) -> None:
    class Model:
        def invoke(self, _messages):  # noqa: ANN001
            return SimpleNamespace(
                content=json.dumps({"web_search": "what is hermes agent"})
            )

    decision = decide_search_operation(
        ask_service=SimpleNamespace(ask_agent=SimpleNamespace(llm=Model())),
        question="check what is hermes agent.",
        root=tmp_path,
        required_tool="web_search",
    )

    assert decision.tool_inputs["web_search"]["query"] == "what is hermes agent"


def test_search_operation_fails_closed_when_query_missing(tmp_path) -> None:
    class Model:
        def invoke(self, _messages):  # noqa: ANN001
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "reasoning_summary": "I selected web research without a query.",
                        "selected_tools": ["web_search"],
                    }
                )
            )

    with pytest.raises(SearchOperationDecisionError, match="did not include a query"):
        decide_search_operation(
            ask_service=SimpleNamespace(ask_agent=SimpleNamespace(llm=Model())),
            question="check what is hermes agent.",
            root=tmp_path,
            required_tool="web_search",
        )


def test_search_operation_fails_closed_when_model_unavailable(tmp_path) -> None:
    with pytest.raises(SearchOperationDecisionError, match="model was unavailable"):
        decide_search_operation(
            ask_service=SimpleNamespace(),
            question="check what is hermes agent.",
            root=tmp_path,
            required_tool="web_search",
        )


def test_search_operation_fails_closed_when_query_too_long(tmp_path) -> None:
    class Model:
        def invoke(self, _messages):  # noqa: ANN001
            return SimpleNamespace(content=json.dumps({"query": "x" * 401}))

    with pytest.raises(SearchOperationDecisionError, match="400-character limit"):
        decide_search_operation(
            ask_service=SimpleNamespace(ask_agent=SimpleNamespace(llm=Model())),
            question="check what is hermes agent.",
            root=tmp_path,
            required_tool="web_search",
        )


def test_search_operation_github_includes_kind_and_repo(tmp_path) -> None:
    class Model:
        def invoke(self, messages):  # noqa: ANN001
            payload = json.loads(messages[-1].content)
            assert payload["required_tool"] == "github_search"
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "query": "hermes-agent",
                        "github_kind": "repositories",
                        "repo": "",
                        "reasoning_summary": "Find public repositories.",
                    }
                )
            )

    decision = decide_search_operation(
        ask_service=SimpleNamespace(ask_agent=SimpleNamespace(llm=Model())),
        question="search github for hermes-agent",
        root=tmp_path,
        required_tool="github_search",
    )

    assert decision.selected_tools == ["github_search"]
    assert decision.tool_inputs == {
        "github_search": {
            "query": "hermes-agent",
            "github_kind": "repositories",
        }
    }
    assert is_valid_search_operation_decision(decision, required_tool="github_search")
