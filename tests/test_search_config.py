from __future__ import annotations

from mana_agent.search.config import SearchConfig


def test_search_config_loads_web_provider_from_dotenv(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MANA_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("MANA_WEB_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("MANA_WEB_SEARCH_ENDPOINT", raising=False)
    monkeypatch.delenv("MANA_SEARCH_ENABLE_WEB", raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                'OPENAI_API_KEY="test-key"',
                'MANA_WEB_SEARCH_PROVIDER="tavily"',
                'MANA_WEB_SEARCH_API_KEY="search-key"',
                'MANA_WEB_SEARCH_ENDPOINT="https://search.example.test"',
                "MANA_SEARCH_ENABLE_WEB=true",
            ]
        ),
        encoding="utf-8",
    )

    config = SearchConfig.from_env()

    assert config.enable_web is True
    assert config.web_provider == "tavily"
    assert config.web_api_key == "search-key"
    assert config.web_endpoint == "https://search.example.test"


def test_search_config_env_overrides_dotenv(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                'OPENAI_API_KEY="test-key"',
                'MANA_WEB_SEARCH_PROVIDER="tavily"',
                'MANA_WEB_SEARCH_API_KEY="dotenv-key"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MANA_WEB_SEARCH_PROVIDER", "brave")
    monkeypatch.setenv("MANA_WEB_SEARCH_API_KEY", "env-key")

    config = SearchConfig.from_env()

    assert config.web_provider == "brave"
    assert config.web_api_key == "env-key"
