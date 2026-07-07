from __future__ import annotations

import json

from mana_agent.search.config import SearchConfig
from mana_agent.search.memory import SearchMemoryStore
from mana_agent.search.models import SearchDecision, SearchQuery, SearchResult
from mana_agent.search.providers import SearchProviders
from mana_agent.search.router import SearchRouter


class _RouterModel:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def invoke(self, _messages):  # noqa: ANN001
        return type("Msg", (), {"content": json.dumps(self.payload)})()


class _WebProvider:
    def __init__(self) -> None:
        self.calls = 0

    def search_sync(self, query: str, *, max_results: int):  # noqa: ANN001
        self.calls += 1
        assert query == "latest docs"
        assert max_results == 4
        return [
            SearchResult(
                source_type="web",
                title="Docs",
                url="https://docs.example.dev/a",
                summary="Official docs explain the current supported behavior.",
                source_domain="docs.example.dev",
                confidence=0.85,
            ),
            SearchResult(
                source_type="web",
                title="Docs duplicate",
                url="https://docs.example.dev/a#section",
                summary="Duplicate official docs explain the behavior.",
                source_domain="docs.example.dev",
                confidence=0.8,
            ),
        ]


class _GitHubProvider:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, query, *, max_results: int):  # noqa: ANN001
        self.calls += 1
        return [
            SearchResult(
                source_type="github",
                title="openai/codex:src/tool.py",
                url="https://github.com/openai/codex/blob/main/src/tool.py",
                summary="Production code shows a compact tool routing pattern.",
                source_domain="github.com",
                repo="openai/codex",
                confidence=0.82,
            )
        ]


def _config() -> SearchConfig:
    return SearchConfig(
        enable_web=True,
        enable_github=True,
        max_results=4,
        web_provider="custom",
        max_injected_results=5,
        enable_ask_agent=True,
    )


def test_router_deduplicates_results_and_stores_memory(tmp_path) -> None:  # noqa: ANN001
    web = _WebProvider()
    router = SearchRouter(
        root=str(tmp_path),
        llm=_RouterModel(
            {
                "mode": "web",
                "reason": "current docs",
                "confidence": 0.9,
                "queries": [{"target": "web", "query": "latest docs"}],
                "max_results": 4,
            }
        ),
        config=_config(),
        providers=SearchProviders(web=web, github=None),
    )
    result = router.run(user_query="latest docs")
    assert web.calls == 1
    assert len(result.results) == 1
    assert "External Search Context" in result.context_block()
    assert (tmp_path / ".mana" / "search_memory.jsonl").exists()


def test_router_memory_hit_skips_provider(tmp_path) -> None:  # noqa: ANN001
    memory = SearchMemoryStore(root=tmp_path)
    memory.store_results(
        original_query="latest docs",
        source_type="web",
        results=[
            SearchResult(
                source_type="web",
                title="Cached docs",
                url="https://docs.example.dev/cached",
                summary="Cached official docs explain the current behavior.",
                source_domain="docs.example.dev",
                confidence=0.9,
            )
        ],
    )
    web = _WebProvider()
    router = SearchRouter(
        root=str(tmp_path),
        llm=_RouterModel(
            {
                "mode": "web",
                "reason": "current docs",
                "confidence": 0.9,
                "queries": [{"target": "web", "query": "latest docs"}],
                "max_results": 4,
            }
        ),
        config=_config(),
        providers=SearchProviders(web=web, github=None),
        memory_store=memory,
    )
    result = router.run(user_query="latest docs")
    assert web.calls == 0
    assert result.decision.mode == "memory_only"
    assert result.memory_hits[0].title == "Cached docs"


def test_router_can_run_both_web_and_github(tmp_path) -> None:  # noqa: ANN001
    web = _WebProvider()
    github = _GitHubProvider()
    router = SearchRouter(
        root=str(tmp_path),
        llm=_RouterModel(
            {
                "mode": "both",
                "reason": "docs and examples",
                "confidence": 0.9,
                "queries": [
                    {"target": "web", "query": "latest docs"},
                    {"target": "github", "query": "tool routing", "repo": "openai/codex"},
                ],
                "max_results": 4,
            }
        ),
        config=_config(),
        providers=SearchProviders(web=web, github=github),
    )
    result = router.run(user_query="find production examples and latest docs")
    assert web.calls == 1
    assert github.calls == 1
    assert {item.source_type for item in result.results} == {"web", "github"}


def test_router_decision_override_runs_selected_external_tools(tmp_path) -> None:  # noqa: ANN001
    web = _WebProvider()
    github = _GitHubProvider()
    router = SearchRouter(
        root=str(tmp_path),
        llm=_RouterModel({"mode": "none", "reason": "should not be used", "confidence": 0.1, "queries": []}),
        config=_config(),
        providers=SearchProviders(web=web, github=github),
    )
    decision = SearchDecision(
        needs_search=True,
        targets=["web", "github"],
        reason="forced by AgentDecision",
        confidence=0.9,
        queries=[
            SearchQuery(target="web", query="latest docs"),
            SearchQuery(target="github", query="hermes-agent", github_kind="repositories"),
        ],
        max_results=4,
        mode="both",
    )
    result = router.run(user_query="search internet & github", decision_override=decision)
    assert web.calls == 1
    assert github.calls == 1
    assert {item.source_type for item in result.results} == {"web", "github"}
