from __future__ import annotations

from urllib.parse import urlparse

from mana_agent.search.models import SearchResult


OFFICIAL_DOMAINS = {
    "docs.github.com",
    "github.com",
    "docs.python.org",
    "developer.mozilla.org",
    "platform.openai.com",
    "docs.pydantic.dev",
}


def dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for result in results:
        key = result.canonical_url()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(result)
    return out


def rank_results(results: list[SearchResult], *, query: str = "") -> list[SearchResult]:
    terms = {item for item in query.lower().split() if len(item) > 2}

    def score(result: SearchResult) -> tuple[float, str]:
        text = f"{result.title} {result.snippet} {result.summary} {result.repo or ''}".lower()
        domain = result.source_domain or urlparse(result.url).netloc.lower()
        value = float(result.confidence)
        if domain in OFFICIAL_DOMAINS or domain.endswith(".gov"):
            value += 0.25
        if result.repo and result.metadata.get("fork") is False:
            value += 0.1
        if result.stars:
            value += min(0.2, result.stars / 50000)
        if result.updated_at:
            value += 0.05
        if terms:
            value += min(0.25, sum(1 for term in terms if term in text) * 0.04)
        return value, result.title.lower()

    return sorted(dedupe_results(results), key=score, reverse=True)
