from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from mana_agent.search.models import SearchQuery, SearchResult


@dataclass(slots=True)
class GitHubRateLimit:
    remaining: int | None = None
    reset_at: int | None = None
    retry_after: int | None = None


class GitHubSearchError(RuntimeError):
    def __init__(self, message: str, *, rate_limit: GitHubRateLimit | None = None) -> None:
        super().__init__(message)
        self.rate_limit = rate_limit or GitHubRateLimit()


def build_github_query(query: SearchQuery) -> str:
    parts: list[str] = []
    base = str(query.query or "").strip()
    if base:
        parts.append(base)
    for phrase in query.exact_phrases:
        clean = str(phrase or "").strip().strip('"')
        if clean:
            parts.append(f'"{clean}"')
    if query.repo:
        parts.append(f"repo:{query.repo}")
    if query.org:
        parts.append(f"org:{query.org}")
    if query.user:
        parts.append(f"user:{query.user}")
    if query.language:
        parts.append(f"language:{query.language}")
    if query.path:
        parts.append(f"path:{query.path}")
    for excluded in query.exclude_paths:
        clean = str(excluded or "").strip()
        if clean:
            parts.append(f"NOT path:{clean}")
    return " ".join(dict.fromkeys(parts))


class GitHubSearchProvider:
    def __init__(self, *, token: str = "", credential_source: str = "token", timeout_seconds: int = 15) -> None:
        self.token = token
        self.credential_source = credential_source
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.last_rate_limit = GitHubRateLimit()

    def search(self, query: SearchQuery, *, max_results: int = 8) -> list[SearchResult]:
        kind = query.github_kind or "code"
        endpoint = {
            "repositories": "https://api.github.com/search/repositories",
            "issues": "https://api.github.com/search/issues",
            "code": "https://api.github.com/search/code",
        }.get(kind, "https://api.github.com/search/code")
        built = build_github_query(query)
        params = {"q": built, "per_page": str(max(1, min(int(max_results or 8), 25)))}
        if kind == "repositories":
            params["sort"] = "updated"
            params["order"] = "desc"
        url = f"{endpoint}?{urllib.parse.urlencode(params)}"
        payload = self._get_json(url)
        items = list(payload.get("items") or [])
        return [self._to_result(item, kind=kind) for item in items]

    async def search_async(self, query: SearchQuery, *, max_results: int = 8) -> list[SearchResult]:
        return self.search(query, max_results=max_results)

    def _get_json(self, url: str) -> dict[str, Any]:
        if self.credential_source == "gh_cli":
            return self._get_json_with_gh(url)
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "mana-agent-search",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                self.last_rate_limit = GitHubRateLimit(
                    remaining=_int_header(response.headers.get("x-ratelimit-remaining")),
                    reset_at=_int_header(response.headers.get("x-ratelimit-reset")),
                    retry_after=_int_header(response.headers.get("retry-after")),
                )
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            rate = GitHubRateLimit(
                remaining=_int_header(exc.headers.get("x-ratelimit-remaining")),
                reset_at=_int_header(exc.headers.get("x-ratelimit-reset")),
                retry_after=_int_header(exc.headers.get("retry-after")),
            )
            self.last_rate_limit = rate
            if exc.code == 429 or self._is_rate_limit_error(exc, rate):
                delay = rate.retry_after
                if delay is None and rate.reset_at:
                    delay = max(0, int(rate.reset_at - time.time()))
                raise GitHubSearchError(f"GitHub rate limit reached; retry after {delay or 'unknown'} seconds", rate_limit=rate) from exc
            raise GitHubSearchError(f"GitHub search failed with HTTP {exc.code}", rate_limit=rate) from exc
        except urllib.error.URLError as exc:
            raise GitHubSearchError(f"GitHub search failed: {exc.reason}") from exc

    def _get_json_with_gh(self, url: str) -> dict[str, Any]:
        """Use GitHub CLI authentication by reference; never extract its token."""
        try:
            completed = subprocess.run(
                ["gh", "api", url, "--method", "GET"],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GitHubSearchError("GitHub CLI authentication is unavailable.") from exc
        if completed.returncode != 0:
            raise GitHubSearchError("GitHub CLI request failed; run `gh auth status` for details.")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise GitHubSearchError("GitHub CLI returned an invalid response.") from exc
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _is_rate_limit_error(error: urllib.error.HTTPError, rate: GitHubRateLimit) -> bool:
        if rate.remaining == 0:
            return True
        try:
            body = error.read().decode("utf-8", "replace").lower()
        except OSError:
            body = ""
        return "rate limit" in body or "abuse detection" in body

    def _to_result(self, item: dict[str, Any], *, kind: str) -> SearchResult:
        if kind == "repositories":
            repo = item.get("full_name")
            url = item.get("html_url") or ""
            title = repo or item.get("name") or "GitHub repository"
            snippet = item.get("description") or ""
            stars = item.get("stargazers_count")
            updated_at = item.get("updated_at")
            fork = bool(item.get("fork", False))
        elif kind == "issues":
            # Issue-search responses have no repository object and use title/body,
            # not code-search name/text_matches. Retain real upstream provenance.
            url = str(item.get("html_url") or "")
            segments = urllib.parse.urlparse(url).path.strip("/").split("/")
            repo = "/".join(segments[:2]) if len(segments) >= 2 else None
            title = item.get("title") or "GitHub issue"
            snippet = item.get("body") or ""
            stars = None
            updated_at = item.get("updated_at")
            fork = False
        else:
            repo_obj = item.get("repository") or {}
            repo = repo_obj.get("full_name")
            url = item.get("html_url") or ""
            path = item.get("path")
            title = f"{repo}:{path}" if repo and path else item.get("name") or "GitHub result"
            snippet = item.get("text_matches") or item.get("name") or ""
            stars = repo_obj.get("stargazers_count")
            updated_at = repo_obj.get("updated_at")
            fork = bool(repo_obj.get("fork", False))
        confidence = 0.72
        if fork:
            confidence -= 0.12
        return SearchResult(
            source_type="github",
            title=str(title),
            url=str(url),
            snippet=str(snippet)[:500],
            summary=str(snippet)[:500],
            source_domain="github.com",
            repo=str(repo) if repo else None,
            path=str(item.get("path") or "") or None,
            stars=int(stars) if isinstance(stars, int) else None,
            updated_at=str(updated_at) if updated_at else None,
            confidence=confidence,
            published_at=str(item.get("created_at") or "") or None,
            metadata={"fork": fork, "kind": kind, **({
                "issue_body": str(item.get("body") or "")[:12000],
                "issue_state": item.get("state"),
                "is_pull_request": bool(item.get("pull_request")),
                "author_association": item.get("author_association"),
            } if kind == "issues" else {})},
        )


def _int_header(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None
