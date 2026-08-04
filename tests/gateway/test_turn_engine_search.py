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


def test_search_operation_uses_a_constrained_model_decision(tmp_path) -> None:
    class Model:
        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        def invoke(self, messages):  # noqa: ANN001
            payload = json.loads(messages[-1].content)
            self.payloads.append(payload)
            tools = payload.get("tools") or payload.get("available_tools")
            assert [tool["name"] for tool in tools] == ["web_search"]
            assert 'Select only "web_search"' in payload["operation_constraint"]
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "intent": "web_research",
                        "confidence": 0.9,
                        "selected_tools": ["web_search"],
                        "tool_inputs": {"web_search": {"query": "latest OpenAI API docs"}},
                        "repo_context_needed": False,
                        "web_search_needed": True,
                        "code_editing_needed": False,
                        "flow_action": "none",
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
    assert len(model.payloads) == 2
