from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

from mana_agent.search.memory import SearchMemoryStore, search_fingerprint
from mana_agent.search.models import SearchResult


def test_search_memory_hit_prevents_duplicate_search(tmp_path) -> None:  # noqa: ANN001
    store = SearchMemoryStore(root=tmp_path)
    result = SearchResult(
        source_type="web",
        title="Official docs",
        url="https://docs.example.dev/page",
        summary="Official documentation explains the stable feature behavior.",
        source_domain="docs.example.dev",
        confidence=0.88,
    )
    store.store_results(original_query="latest feature docs", source_type="web", results=[result])
    hits = store.find(query="latest feature docs", source_type="web")
    assert len(hits) == 1
    assert hits[0].title == "Official docs"


def test_stale_search_memory_does_not_match(tmp_path) -> None:  # noqa: ANN001
    store = SearchMemoryStore(root=tmp_path)
    store.path.parent.mkdir(parents=True)
    expired = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    store.path.write_text(
        json.dumps(
            {
                "id": "old",
                "task_id": None,
                "query_fingerprint": search_fingerprint("latest feature docs", "web"),
                "original_query": "latest feature docs",
                "normalized_query": "latest feature docs",
                "source_type": "web",
                "title": "Old docs",
                "url": "https://docs.example.dev/old",
                "repo": None,
                "source_domain": "docs.example.dev",
                "summary": "Old result",
                "key_findings": [],
                "confidence": 0.9,
                "fetched_at": expired,
                "expires_at": expired,
                "tags": [],
                "used_in_answer": False,
                "negative": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert store.find(query="latest feature docs", source_type="web") == []


def test_low_quality_result_writes_negative_marker(tmp_path) -> None:  # noqa: ANN001
    store = SearchMemoryStore(root=tmp_path)
    result = SearchResult(source_type="github", title="Noisy", url="https://github.com/x/y", summary="short", confidence=0.2)
    saved = store.store_results(original_query="bad query", source_type="github", results=[result])
    assert saved == []
    assert store.has_negative(query="bad query", source_type="github") is True
