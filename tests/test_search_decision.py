from __future__ import annotations

import json
import pytest

from mana_agent.search.config import SearchConfig
from mana_agent.search.decision import SearchDecisionEngine


class _ModelRouter:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def generate_sync(self, _messages, **_native_options):  # noqa: ANN001
        return type("Msg", (), {"content": json.dumps(self.payload)})()

    def decide_sync(self, messages, schema, **kwargs):
        import json
        raw = self.generate_sync(messages)
        if isinstance(schema, dict):
            from mana_agent.multi_agent.tools.operation_policy import validate_arguments
            value = raw if isinstance(raw, dict) else json.loads(getattr(raw, "content", raw))
            validate_arguments(value, schema)
            return value
        if isinstance(raw, schema):
            return raw
        return schema.model_validate(raw, strict=True) if isinstance(raw, dict) else schema.model_validate_json(getattr(raw, "content", raw), strict=True)


def _config() -> SearchConfig:
    return SearchConfig(enable_web=True, enable_github=True, enable_ask_agent=True)


def test_no_search_for_simple_local_repo_task() -> None:
    engine = SearchDecisionEngine(
        llm=_ModelRouter({"mode": "none", "reason": "local", "confidence": 0.9, "queries": []}),
        config=_config(),
    )
    decision = engine.decide(user_query="update README.md with this local version")
    assert decision.mode == "none"
    assert decision.needs_search is False


def test_web_search_for_latest_docs_model_decision() -> None:
    engine = SearchDecisionEngine(
        llm=_ModelRouter(
            {
                "mode": "web",
                "reason": "current docs needed",
                "confidence": 0.86,
                "queries": [{"target": "web", "query": "Pydantic latest model config docs"}],
            }
        ),
        config=_config(),
    )
    decision = engine.decide(user_query="check latest Pydantic model config docs")
    assert decision.mode == "web"
    assert decision.targets == ["web"]
    assert decision.queries[0].query == "Pydantic latest model config docs"


def test_guardrail_does_not_special_case_search_internet_without_model() -> None:
    engine = SearchDecisionEngine(llm=None, config=_config())
    with pytest.raises(RuntimeError, match="decision is unavailable"):
        engine.decide(user_query="search internet about hermes-agent")


def test_github_search_for_like_repo_request() -> None:
    engine = SearchDecisionEngine(
        llm=_ModelRouter(
            {
                "mode": "github",
                "reason": "repo example requested",
                "confidence": 0.82,
                "queries": [{"target": "github", "query": '"tool call"', "repo": "openai/codex"}],
            }
        ),
        config=_config(),
    )
    decision = engine.decide(user_query='implement this like openai/codex "tool call"')
    assert decision.mode == "github"
    assert decision.queries[0].repo == "openai/codex"


def test_both_search_for_latest_docs_and_production_examples() -> None:
    engine = SearchDecisionEngine(
        llm=_ModelRouter(
            {
                "mode": "both",
                "reason": "needs docs and examples",
                "confidence": 0.91,
                "queries": [
                    {"target": "web", "query": "latest OpenAI structured outputs docs"},
                    {"target": "github", "query": "structured outputs", "language": "Python"},
                ],
            }
        ),
        config=_config(),
    )
    decision = engine.decide(user_query="find production examples and explain latest docs")
    assert decision.mode == "both"
    assert set(decision.targets) == {"web", "github"}


def test_private_local_code_is_not_sent_to_external_search() -> None:
    engine = SearchDecisionEngine(
        llm=_ModelRouter(
            {
                "mode": "web",
                "reason": "bad model decision",
                "confidence": 0.99,
                "queries": [{"target": "web", "query": "leaky"}],
            }
        ),
        config=_config(),
    )
    decision = engine.decide(user_query="search web for /Users/ah/project/.env token=secret")
    assert decision.mode == "none"
    assert decision.queries == []
