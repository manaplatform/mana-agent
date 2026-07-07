from __future__ import annotations

from mana_agent.search.github_provider import build_github_query
from mana_agent.search.models import SearchQuery


def test_github_query_builder_adds_structured_qualifiers() -> None:
    query = SearchQuery(
        target="github",
        query="agent memory",
        github_kind="code",
        repo="openai/codex",
        org="openai",
        language="Python",
        path="src/",
        exact_phrases=["tool call"],
        exclude_paths=["tests"],
    )
    built = build_github_query(query)
    assert "agent memory" in built
    assert '"tool call"' in built
    assert "repo:openai/codex" in built
    assert "org:openai" in built
    assert "language:Python" in built
    assert "path:src/" in built
    assert "NOT path:tests" in built
