from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from mana_agent.search.config import SearchConfig
from mana_agent.search.decision import SearchDecisionEngine
from mana_agent.search.github_provider import GitHubSearchError
from mana_agent.search.memory import SearchMemoryStore
from mana_agent.search.models import SearchDecision, SearchMemoryRecord, SearchQuery, SearchResult
from mana_agent.search.providers import SearchProviders, build_search_providers
from mana_agent.search.ranker import rank_results

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SearchRouterResult:
    decision: SearchDecision
    results: list[SearchResult] = field(default_factory=list)
    memory_hits: list[SearchMemoryRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def context_block(self, *, max_results: int = 5, max_words: int = 80) -> str:
        rows: list[str] = []
        for hit in self.memory_hits[:max_results]:
            rows.append(
                "\n".join(
                    [
                        f"- Source type: {hit.source_type}",
                        f"  Query: {hit.original_query}",
                        f"  Result title: {hit.title}",
                        f"  URL/repo: {hit.repo or hit.url}",
                        f"  Summary: {_limit_words(hit.summary, max_words)}",
                        f"  Key findings: {'; '.join(hit.key_findings[:3])}",
                        f"  Confidence: {hit.confidence:.2f}",
                        f"  Fetched at: {hit.fetched_at}",
                    ]
                )
            )
        remaining = max(0, max_results - len(rows))
        for result in self.results[:remaining]:
            rows.append(
                "\n".join(
                    [
                        f"- Source type: {result.source_type}",
                        f"  Query: {self._query_for_result(result)}",
                        f"  Result title: {result.title}",
                        f"  URL/repo: {result.repo or result.url}",
                        f"  Summary: {result.compact_summary(max_words=max_words)}",
                        f"  Key findings: {'; '.join(result.key_findings[:3])}",
                        f"  Confidence: {result.confidence:.2f}",
                        f"  Fetched at: {result.fetched_at}",
                    ]
                )
            )
        if not rows:
            return ""
        return "External Search Context:\n" + "\n".join(rows)

    def _query_for_result(self, result: SearchResult) -> str:
        for query in self.decision.queries:
            if query.target == result.source_type:
                return query.query
        return ""


class SearchRouter:
    def __init__(
        self,
        *,
        root: str,
        llm: Any | None = None,
        config: SearchConfig | None = None,
        providers: SearchProviders | None = None,
        memory_store: SearchMemoryStore | None = None,
    ) -> None:
        self.config = config or SearchConfig.from_env()
        self.decision_engine = SearchDecisionEngine(llm=llm, config=self.config)
        self.providers = providers or build_search_providers(self.config)
        self.memory = memory_store or SearchMemoryStore(root=root)

    def run(
        self,
        *,
        user_query: str,
        repo_context: str = "",
        memory_context: str = "",
        task_id: str | None = None,
        decision_override: SearchDecision | None = None,
    ) -> SearchRouterResult:
        decision = decision_override or self.decision_engine.decide(
            user_query=user_query,
            repo_context=repo_context,
            memory_context=memory_context,
            max_results=self.config.max_results,
        )
        if decision.mode == "none":
            return SearchRouterResult(decision=decision)
        memory_hits: list[SearchMemoryRecord] = []
        if decision.reuse_memory_first:
            for query in decision.queries:
                memory_hits.extend(
                    self.memory.find(query=query.query, source_type=query.target, limit=self.config.max_injected_results)
                )
        if decision.mode == "memory_only" or self._memory_satisfies(decision, memory_hits):
            decision.mode = "memory_only"
            decision.needs_search = False
            return SearchRouterResult(decision=decision, memory_hits=memory_hits)

        results: list[SearchResult] = []
        warnings: list[str] = []
        for query in decision.queries:
            if self.memory.has_negative(query=query.query, source_type=query.target):
                warnings.append(f"negative search memory reused for {query.target}:{query.query}")
                continue
            try:
                found = self._search(query, max_results=decision.max_results)
            except Exception as exc:
                warnings.append(str(exc))
                logger.debug("external search failed", exc_info=True)
                continue
            ranked = rank_results(found, query=query.query)
            results.extend(ranked)
            self.memory.store_results(
                original_query=query.query,
                source_type=query.target,
                results=ranked,
                task_id=task_id,
                ttl_days=self.config.memory_ttl_days,
            )
        return SearchRouterResult(
            decision=decision,
            results=rank_results(results, query=user_query)[: self.config.max_injected_results],
            memory_hits=memory_hits,
            warnings=warnings,
        )

    def _search(self, query: SearchQuery, *, max_results: int) -> list[SearchResult]:
        if query.target == "github":
            if self.providers.github is None:
                raise RuntimeError("GitHub search is disabled")
            try:
                return self.providers.github.search(query, max_results=max_results)
            except GitHubSearchError:
                raise
        if self.providers.web is None:
            raise RuntimeError("web search is disabled")
        searcher = getattr(self.providers.web, "search_sync", None)
        if callable(searcher):
            return list(searcher(query.query, max_results=max_results))
        return list(asyncio.run(self.providers.web.search(query.query, max_results=max_results)))

    @staticmethod
    def _memory_satisfies(decision: SearchDecision, hits: list[SearchMemoryRecord]) -> bool:
        if not decision.targets:
            return False
        hit_targets = {hit.source_type for hit in hits if hit.confidence >= 0.75}
        return set(decision.targets).issubset(hit_targets)


def _limit_words(text: str, max_words: int) -> str:
    words = " ".join(str(text or "").split()).split()
    return " ".join(words[:max_words])
