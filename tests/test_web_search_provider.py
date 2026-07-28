from __future__ import annotations

import json
from typing import Any

from mana_agent.search.web_provider import ConfiguredWebSearchProvider


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_tavily_uses_bearer_authentication_header(monkeypatch) -> None:  # noqa: ANN001
    captured: dict[str, Any] = {}

    def open_request(request, *, timeout: int):  # noqa: ANN001
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response({"results": []})

    monkeypatch.setattr("mana_agent.search.web_provider.urllib.request.urlopen", open_request)

    provider = ConfiguredWebSearchProvider(provider="tavily", api_key="test-key", timeout_seconds=12)
    assert provider.search_sync("bitcoin market trend", max_results=5) == []

    assert captured == {
        "url": "https://api.tavily.com/search",
        "method": "POST",
        "headers": {"Authorization": "Bearer test-key", "Content-type": "application/json"},
        "body": {"query": "bitcoin market trend", "max_results": 5},
        "timeout": 12,
    }
