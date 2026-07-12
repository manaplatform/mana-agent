from __future__ import annotations

import urllib.error

import pytest

from mana_agent.search.github_provider import GitHubSearchError, GitHubSearchProvider


class _Headers(dict):
    def get(self, key, default=None):  # noqa: ANN001
        return super().get(key.lower(), default)


def test_github_provider_rate_limit_error_includes_headers(monkeypatch) -> None:  # noqa: ANN001
    provider = GitHubSearchProvider(timeout_seconds=1)
    headers = _Headers({"x-ratelimit-remaining": "0", "x-ratelimit-reset": "123", "retry-after": "5"})

    def _raise(_request, timeout):  # noqa: ANN001
        raise urllib.error.HTTPError("https://api.github.com/search/code", 403, "rate limit", headers, None)

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    with pytest.raises(GitHubSearchError) as exc:
        provider._get_json("https://api.github.com/search/code?q=x")

    assert "rate limit" in str(exc.value).lower()
    assert exc.value.rate_limit.remaining == 0
    assert exc.value.rate_limit.retry_after == 5


def test_github_provider_does_not_mislabel_non_rate_limit_403(monkeypatch) -> None:  # noqa: ANN001
    provider = GitHubSearchProvider(timeout_seconds=1)
    headers = _Headers({"x-ratelimit-remaining": "42"})

    def _raise(_request, timeout):  # noqa: ANN001
        raise urllib.error.HTTPError("https://api.github.com/search/code", 403, "forbidden", headers, None)

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    with pytest.raises(GitHubSearchError, match="HTTP 403") as exc:
        provider._get_json("https://api.github.com/search/code?q=x")

    assert exc.value.rate_limit.remaining == 42
