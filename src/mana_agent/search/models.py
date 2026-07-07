from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


SearchTarget = Literal["web", "github"]
SearchMode = Literal["none", "web", "github", "both", "memory_only"]
GitHubSearchKind = Literal["repositories", "code", "issues"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


@dataclass(slots=True)
class SearchQuery:
    query: str
    target: SearchTarget
    github_kind: GitHubSearchKind = "code"
    repo: str | None = None
    org: str | None = None
    user: str | None = None
    language: str | None = None
    path: str | None = None
    exact_phrases: list[str] = field(default_factory=list)
    exclude_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchDecision:
    needs_search: bool
    targets: list[SearchTarget]
    reason: str
    confidence: float
    queries: list[SearchQuery]
    reuse_memory_first: bool = True
    max_results: int = 8
    mode: SearchMode = "none"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["queries"] = [item.to_dict() for item in self.queries]
        return payload


@dataclass(slots=True)
class SearchResult:
    source_type: SearchTarget
    title: str
    url: str
    snippet: str = ""
    summary: str = ""
    source_domain: str | None = None
    repo: str | None = None
    language: str | None = None
    path: str | None = None
    stars: int | None = None
    updated_at: str | None = None
    published_at: str | None = None
    fetched_at: str = field(default_factory=utc_iso)
    confidence: float = 0.5
    key_findings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def canonical_url(self) -> str:
        return self.url.split("#", 1)[0].rstrip("/")

    def compact_summary(self, *, max_words: int = 80) -> str:
        text = " ".join((self.summary or self.snippet or "").split())
        if not text:
            return ""
        words = text.split()
        return " ".join(words[:max_words])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchMemoryRecord:
    id: str
    task_id: str | None
    query_fingerprint: str
    original_query: str
    normalized_query: str
    source_type: SearchTarget
    title: str
    url: str
    repo: str | None
    source_domain: str | None
    summary: str
    key_findings: list[str]
    confidence: float
    fetched_at: str
    expires_at: str | None
    tags: list[str] = field(default_factory=list)
    used_in_answer: bool = False
    negative: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
