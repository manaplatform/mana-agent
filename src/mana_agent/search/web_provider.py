from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol
from urllib.parse import urlparse

from mana_agent.search.models import SearchResult


class WebSearchProvider(Protocol):
    async def search(self, query: str, *, max_results: int) -> list[SearchResult]:
        ...


class WebSearchError(RuntimeError):
    pass


class ConfiguredWebSearchProvider:
    def __init__(
        self,
        *,
        provider: str,
        api_key: str,
        endpoint: str = "",
        engine_id: str = "",
        timeout_seconds: int = 15,
    ) -> None:
        self.provider = provider.strip().lower()
        self.api_key = api_key
        self.endpoint = endpoint
        self.engine_id = engine_id
        self.timeout_seconds = max(1, int(timeout_seconds))

    async def search(self, query: str, *, max_results: int = 8) -> list[SearchResult]:
        return self.search_sync(query, max_results=max_results)

    def search_sync(self, query: str, *, max_results: int = 8) -> list[SearchResult]:
        if not self.provider:
            raise WebSearchError("web search provider is not configured")
        if self.provider == "tavily":
            return self._tavily(query, max_results=max_results)
        if self.provider == "brave":
            return self._brave(query, max_results=max_results)
        if self.provider == "serpapi":
            return self._serpapi(query, max_results=max_results)
        if self.provider == "exa":
            return self._exa(query, max_results=max_results)
        if self.provider in {"google_cse", "google", "google-cse"}:
            return self._google_cse(query, max_results=max_results)
        if self.provider in {"bing", "bing-compatible"}:
            return self._bing(query, max_results=max_results)
        if self.provider == "custom":
            return self._custom(query, max_results=max_results)
        raise WebSearchError(f"unsupported web search provider: {self.provider}")

    def _tavily(self, query: str, *, max_results: int) -> list[SearchResult]:
        if not self.api_key:
            raise WebSearchError("MANA_WEB_SEARCH_API_KEY is required for Tavily")
        payload = json.dumps({"api_key": self.api_key, "query": query, "max_results": max_results}).encode("utf-8")
        data = self._request_json(
            "https://api.tavily.com/search",
            method="POST",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        return [self._generic_result(item, query=query) for item in data.get("results", [])]

    def _brave(self, query: str, *, max_results: int) -> list[SearchResult]:
        if not self.api_key:
            raise WebSearchError("MANA_WEB_SEARCH_API_KEY is required for Brave Search")
        url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(
            {"q": query, "count": max(1, min(max_results, 20))}
        )
        data = self._request_json(url, headers={"X-Subscription-Token": self.api_key})
        return [self._generic_result(item, query=query) for item in (data.get("web") or {}).get("results", [])]

    def _serpapi(self, query: str, *, max_results: int) -> list[SearchResult]:
        if not self.api_key:
            raise WebSearchError("MANA_WEB_SEARCH_API_KEY is required for SerpAPI")
        url = "https://serpapi.com/search.json?" + urllib.parse.urlencode(
            {"q": query, "api_key": self.api_key, "num": max_results}
        )
        data = self._request_json(url)
        return [self._generic_result(item, query=query) for item in data.get("organic_results", [])]

    def _exa(self, query: str, *, max_results: int) -> list[SearchResult]:
        if not self.api_key:
            raise WebSearchError("MANA_WEB_SEARCH_API_KEY is required for Exa")
        payload = json.dumps({"query": query, "numResults": max_results}).encode("utf-8")
        data = self._request_json(
            self.endpoint or "https://api.exa.ai/search",
            method="POST",
            data=payload,
            headers={"Content-Type": "application/json", "x-api-key": self.api_key},
        )
        return [self._generic_result(item, query=query) for item in data.get("results", [])]

    def _google_cse(self, query: str, *, max_results: int) -> list[SearchResult]:
        if not self.api_key:
            raise WebSearchError("MANA_WEB_SEARCH_API_KEY is required for Google CSE")
        if not self.engine_id:
            raise WebSearchError("MANA_WEB_SEARCH_ENGINE_ID is required for Google CSE")
        endpoint = self.endpoint or "https://www.googleapis.com/customsearch/v1"
        url = endpoint + "?" + urllib.parse.urlencode(
            {
                "key": self.api_key,
                "cx": self.engine_id,
                "q": query,
                "num": max(1, min(max_results, 10)),
            }
        )
        data = self._request_json(url)
        return [self._generic_result(item, query=query) for item in data.get("items", [])]

    def _bing(self, query: str, *, max_results: int) -> list[SearchResult]:
        if not self.api_key:
            raise WebSearchError("MANA_WEB_SEARCH_API_KEY is required for Bing-compatible search")
        endpoint = self.endpoint or "https://api.bing.microsoft.com/v7.0/search"
        url = endpoint + "?" + urllib.parse.urlencode({"q": query, "count": max_results})
        data = self._request_json(url, headers={"Ocp-Apim-Subscription-Key": self.api_key})
        return [self._generic_result(item, query=query) for item in (data.get("webPages") or {}).get("value", [])]

    def _custom(self, query: str, *, max_results: int) -> list[SearchResult]:
        if not self.endpoint:
            raise WebSearchError("MANA_WEB_SEARCH_ENDPOINT is required for custom search")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = self.endpoint + ("&" if "?" in self.endpoint else "?") + urllib.parse.urlencode(
            {"q": query, "max_results": max_results}
        )
        data = self._request_json(url, headers=headers)
        items = data.get("results") or data.get("items") or []
        return [self._generic_result(item, query=query) for item in items]

    def _request_json(
        self,
        url: str,
        *,
        method: str = "GET",
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        req = urllib.request.Request(url, method=method, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise WebSearchError(f"web search failed with HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise WebSearchError(f"web search failed: {exc.reason}") from exc

    @staticmethod
    def _generic_result(item: dict[str, Any], *, query: str) -> SearchResult:
        title = str(item.get("title") or item.get("name") or "Search result")
        url = str(item.get("url") or item.get("link") or "")
        snippet = str(item.get("snippet") or item.get("content") or item.get("description") or "")
        domain = urlparse(url).netloc.lower() or None
        confidence = 0.65
        if domain and ("docs." in domain or domain.endswith(".gov")):
            confidence += 0.15
        if query.lower() in f"{title} {snippet}".lower():
            confidence += 0.05
        return SearchResult(
            source_type="web",
            title=title,
            url=url,
            snippet=snippet[:1000],
            summary=snippet[:1000],
            source_domain=domain,
            published_at=str(item.get("published_date") or item.get("date") or "") or None,
            confidence=min(confidence, 0.95),
        )
