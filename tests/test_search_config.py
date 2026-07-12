from __future__ import annotations

from types import SimpleNamespace

from mana_agent.search.config import SearchConfig


def test_search_config_uses_user_settings_not_environment(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("MANA_WEB_SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("MANA_WEB_SEARCH_API_KEY", "environment-key")
    monkeypatch.setattr(
        "mana_agent.search.config._settings",
        lambda: SimpleNamespace(
            mana_github_token="",
            mana_search_enable_web=True,
            mana_search_enable_github=True,
            mana_search_max_results=8,
            mana_search_timeout_seconds=15,
            mana_search_memory_ttl_days=14,
            mana_web_search_provider="tavily",
            mana_web_search_api_key="user-key",
            mana_web_search_endpoint="https://search.example.test",
            mana_web_search_base_url="",
            mana_web_search_engine_id="",
            mana_web_search_max_results=8,
            mana_search_max_injected_results=5,
            mana_search_max_summary_words=80,
            mana_search_enable_ask_agent=True,
        ),
    )

    config = SearchConfig.from_env()

    assert config.enable_web is True
    assert config.web_provider == "tavily"
    assert config.web_api_key == "user-key"
    assert config.web_endpoint == "https://search.example.test"
