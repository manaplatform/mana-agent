from __future__ import annotations

from types import SimpleNamespace

import pytest

from mana_agent.gateway.turn_engine import SearchOperationDecisionError, run_web_research_answer
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
