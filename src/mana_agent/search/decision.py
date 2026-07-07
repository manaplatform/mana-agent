from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mana_agent.search.config import SearchConfig
from mana_agent.search.models import SearchDecision, SearchMode, SearchQuery, SearchTarget
from mana_agent.search.prompts import SEARCH_ROUTER_PROMPT


EXPLICIT_NO_SEARCH = re.compile(r"\b(do not|don't|dont|without|no)\s+(web\s+)?search\b", re.I)
WEB_HINTS = re.compile(
    r"\b(web search|search web|latest|current|today|new version|official docs|documentation|best practice|unknown api|unknown package|pricing|release)\b",
    re.I,
)
GITHUB_HINTS = re.compile(
    r"\b(search github|github search|find repo|like [\w.-]+/[\w.-]+|repo:[\w.-]+/[\w.-]+|open-source|production examples?|how do other repos|implementation in git)\b",
    re.I,
)
LOCAL_ONLY_HINTS = re.compile(
    r"\b(rewrite|branch name|git command|local repo|this repo|update|edit|fix|add tests?|run tests?|explain this file)\b",
    re.I,
)
PRIVATE_SNIPPET_HINTS = re.compile(
    r"(```|BEGIN (RSA|OPENSSH|PRIVATE)|api[_-]?key|token\s*=|password\s*=|/Users/|\.env|customer|private url)",
    re.I,
)
REPO_REF_RE = re.compile(r"\b(?:repo:)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\b")
ORG_RE = re.compile(r"\borg:([A-Za-z0-9_.-]+)\b")
USER_RE = re.compile(r"\buser:([A-Za-z0-9_.-]+)\b")
LANG_RE = re.compile(r"\b(?:language|lang):([A-Za-z0-9_+#.-]+)\b", re.I)
PATH_RE = re.compile(r"\bpath:([^\s]+)\b", re.I)


class SearchDecisionEngine:
    def __init__(self, *, llm: Any | None = None, config: SearchConfig | None = None) -> None:
        self.llm = llm
        self.config = config or SearchConfig.from_env()

    def decide(
        self,
        *,
        user_query: str,
        repo_context: str = "",
        memory_context: str = "",
        max_results: int | None = None,
    ) -> SearchDecision:
        text = str(user_query or "").strip()
        if not text or EXPLICIT_NO_SEARCH.search(text) or PRIVATE_SNIPPET_HINTS.search(text):
            return self._none("External search is disabled by user intent or privacy guard.")
        model_decision = self._model_decision(
            text,
            repo_context=repo_context,
            memory_context=memory_context,
            max_results=max_results,
        )
        decision = model_decision or self._guardrail_decision(text, max_results=max_results)
        if not self.config.enable_web:
            decision.targets = [target for target in decision.targets if target != "web"]
            decision.queries = [query for query in decision.queries if query.target != "web"]
        if not self.config.enable_github:
            decision.targets = [target for target in decision.targets if target != "github"]
            decision.queries = [query for query in decision.queries if query.target != "github"]
        decision.needs_search = bool(decision.targets) and decision.mode != "none"
        decision.mode = self._mode_from_targets(decision.targets) if decision.needs_search else "none"
        return decision

    def _model_decision(
        self,
        user_query: str,
        *,
        repo_context: str,
        memory_context: str,
        max_results: int | None,
    ) -> SearchDecision | None:
        if self.llm is None:
            return None
        payload = {
            "user_query": user_query,
            "repo_context": repo_context[:1200],
            "memory_context": memory_context[:1200],
            "max_results": max_results or self.config.max_results,
        }
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=SEARCH_ROUTER_PROMPT),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
                ]
            )
            content = getattr(response, "content", response)
            if isinstance(content, list):
                content = " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
            data = _extract_json(str(content))
            return self._decision_from_payload(data, fallback_query=user_query)
        except Exception:
            return None

    def _guardrail_decision(self, text: str, *, max_results: int | None) -> SearchDecision:
        wants_web = bool(WEB_HINTS.search(text))
        wants_github = bool(GITHUB_HINTS.search(text))
        if not wants_web and not wants_github and LOCAL_ONLY_HINTS.search(text):
            return self._none("Task appears answerable from local repository context.")
        if wants_web and wants_github:
            targets: list[SearchTarget] = ["web", "github"]
        elif wants_github:
            targets = ["github"]
        elif wants_web:
            targets = ["web"]
        else:
            targets = []
        if not targets:
            return self._none("No clear external search need was detected.")
        return SearchDecision(
            needs_search=True,
            targets=targets,
            reason="External search hints were present in the user task.",
            confidence=0.65,
            queries=[self._query_for_target(text, target) for target in targets],
            reuse_memory_first=True,
            max_results=max_results or self.config.max_results,
            mode=self._mode_from_targets(targets),
        )

    def _merge_decisions(
        self,
        model_decision: SearchDecision | None,
        guardrail: SearchDecision,
        text: str,
    ) -> SearchDecision:
        if model_decision is None:
            return guardrail
        if model_decision.mode == "memory_only":
            return model_decision
        if not model_decision.needs_search and guardrail.needs_search and guardrail.confidence >= 0.7:
            return guardrail
        if model_decision.needs_search and not guardrail.needs_search and model_decision.confidence < 0.7 and LOCAL_ONLY_HINTS.search(text):
            return self._none("Model confidence was low and the task appears local-only.")
        if model_decision.needs_search and not model_decision.queries:
            model_decision.queries = [self._query_for_target(text, target) for target in model_decision.targets]
        return model_decision

    def _decision_from_payload(self, data: dict[str, Any], *, fallback_query: str) -> SearchDecision:
        mode = str(data.get("mode") or "none").strip().lower()
        if mode not in {"none", "web", "github", "both", "memory_only"}:
            mode = "none"
        targets = self._targets_from_mode(mode)  # type: ignore[arg-type]
        queries: list[SearchQuery] = []
        for item in list(data.get("queries") or []):
            if not isinstance(item, dict):
                continue
            target = str(item.get("target") or "").strip().lower()
            if target not in {"web", "github"}:
                continue
            queries.append(
                SearchQuery(
                    query=str(item.get("query") or fallback_query).strip(),
                    target=target,  # type: ignore[arg-type]
                    github_kind=str(item.get("github_kind") or "code") if target == "github" else "code",  # type: ignore[arg-type]
                    repo=_optional_str(item.get("repo")),
                    org=_optional_str(item.get("org")),
                    user=_optional_str(item.get("user")),
                    language=_optional_str(item.get("language")),
                    path=_optional_str(item.get("path")),
                    exact_phrases=[str(x) for x in item.get("exact_phrases") or [] if str(x).strip()],
                    exclude_paths=[str(x) for x in item.get("exclude_paths") or [] if str(x).strip()],
                )
            )
        if targets and not queries:
            queries = [self._query_for_target(fallback_query, target) for target in targets]
        return SearchDecision(
            needs_search=bool(targets),
            targets=targets,
            reason=str(data.get("reason") or "Model-routed search decision.")[:500],
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5) or 0.5))),
            queries=queries,
            reuse_memory_first=bool(data.get("reuse_memory_first", True)),
            max_results=max(1, min(25, int(data.get("max_results", self.config.max_results) or self.config.max_results))),
            mode=mode,  # type: ignore[arg-type]
        )

    def _query_for_target(self, text: str, target: SearchTarget) -> SearchQuery:
        repo = _first_match(REPO_REF_RE, text)
        org = _first_match(ORG_RE, text)
        user = _first_match(USER_RE, text)
        language = _first_match(LANG_RE, text)
        path = _first_match(PATH_RE, text)
        cleaned = _sanitize_public_query(text)
        if target == "github":
            return SearchQuery(
                query=cleaned,
                target="github",
                github_kind="code" if repo or path or language else "repositories",
                repo=repo,
                org=org,
                user=user,
                language=language,
                path=path,
            )
        return SearchQuery(query=cleaned, target="web")

    @staticmethod
    def _targets_from_mode(mode: SearchMode) -> list[SearchTarget]:
        if mode == "both":
            return ["web", "github"]
        if mode in {"web", "github"}:
            return [mode]
        return []

    @staticmethod
    def _mode_from_targets(targets: list[SearchTarget]) -> SearchMode:
        unique = set(targets)
        if unique == {"web", "github"}:
            return "both"
        if unique == {"web"}:
            return "web"
        if unique == {"github"}:
            return "github"
        return "none"

    def _none(self, reason: str) -> SearchDecision:
        return SearchDecision(
            needs_search=False,
            targets=[],
            reason=reason,
            confidence=0.8,
            queries=[],
            reuse_memory_first=True,
            max_results=self.config.max_results,
            mode="none",
        )


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    data = json.loads(stripped)
    return data if isinstance(data, dict) else {}


def _sanitize_public_query(text: str) -> str:
    cleaned = re.sub(r"```.*?```", " ", text, flags=re.S)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"(/Users/|/home/|[A-Za-z]:\\)\S+", " ", cleaned)
    cleaned = re.sub(r"\b(api[_-]?key|token|password)\s*[:=]\s*\S+", " ", cleaned, flags=re.I)
    return re.sub(r"\s+", " ", cleaned).strip()[:240]


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
