from __future__ import annotations

from dataclasses import dataclass
import os

from mana_agent.config.settings import Settings


def _settings() -> Settings | None:
    try:
        return Settings()
    except Exception:
        return None


def _str_config(settings: Settings | None, attr: str, default: str = "") -> str:
    if settings is not None:
        return str(getattr(settings, attr, default) or "")
    return default


def _bool_config(settings: Settings | None, attr: str, default: bool) -> bool:
    if settings is not None:
        return bool(getattr(settings, attr, default))
    return default


def _int_config(
    settings: Settings | None,
    attr: str,
    default: int,
    *,
    min_value: int = 1,
    max_value: int = 1000,
) -> int:
    if settings is not None:
        try:
            value = int(getattr(settings, attr, default))
        except (TypeError, ValueError):
            value = default
    else:
        value = default
    return max(min_value, min(max_value, value))


@dataclass(frozen=True, slots=True)
class SearchConfig:
    github_token: str = ""
    github_credential_source: str = "disabled"
    github_secret_ref: str = ""
    enable_web: bool = True
    enable_github: bool = True
    max_results: int = 8
    timeout_seconds: int = 15
    memory_ttl_days: int = 14
    web_provider: str = ""
    web_api_key: str = ""
    web_endpoint: str = ""
    web_base_url: str = ""
    web_engine_id: str = ""
    web_max_results: int = 8
    max_injected_results: int = 5
    max_summary_words: int = 80
    enable_ask_agent: bool = True

    @property
    def web_search_configuration_error(self) -> str:
        """Return the actionable reason public-web search cannot run, if any.

        Route availability and execution validation share this contract so the
        entry router never advertises a search capability that the provider
        layer will reject later.
        """
        if not self.enable_web:
            return "Public web search is disabled for this session."
        provider = self.web_provider.strip().lower()
        if not provider:
            return "Public web search has no configured provider."
        supported = {
            "tavily",
            "brave",
            "serpapi",
            "exa",
            "google_cse",
            "google",
            "google-cse",
            "bing",
            "bing-compatible",
            "custom",
        }
        if provider not in supported:
            return f"Public web search provider '{provider}' is not supported."
        if provider != "custom" and not self.web_api_key:
            return "Public web search provider credentials are not configured."
        if provider in {"google_cse", "google", "google-cse"} and not self.web_engine_id:
            return "Google Custom Search requires MANA_WEB_SEARCH_ENGINE_ID."
        if provider == "custom" and not self.web_endpoint:
            return "Custom web search requires MANA_WEB_SEARCH_ENDPOINT."
        return ""

    @property
    def web_search_available(self) -> bool:
        return not self.web_search_configuration_error

    @classmethod
    def from_env(cls) -> "SearchConfig":
        settings = _settings()
        github_source = _str_config(settings, "mana_github_credential_source", "disabled")
        github_ref = _str_config(settings, "mana_github_secret_ref")
        github_token = _str_config(settings, "mana_github_token")
        if github_source == "environment" and github_ref:
            github_token = str(os.getenv(github_ref) or "")
        return cls(
            github_token=github_token,
            github_credential_source=github_source,
            github_secret_ref=github_ref,
            enable_web=_bool_config(settings, "mana_search_enable_web", True),
            enable_github=_bool_config(settings, "mana_search_enable_github", True),
            max_results=_int_config(settings, "mana_search_max_results", 8, min_value=1, max_value=25),
            timeout_seconds=_int_config(settings, "mana_search_timeout_seconds", 15, min_value=1, max_value=60),
            memory_ttl_days=_int_config(settings, "mana_search_memory_ttl_days", 14, min_value=1, max_value=365),
            web_provider=_str_config(settings, "mana_web_search_provider").strip().lower(),
            web_api_key=_str_config(settings, "mana_web_search_api_key"),
            web_endpoint=(
                _str_config(settings, "mana_web_search_endpoint")
                or _str_config(settings, "mana_web_search_base_url")
            ),
            web_base_url=_str_config(settings, "mana_web_search_base_url"),
            web_engine_id=_str_config(settings, "mana_web_search_engine_id"),
            web_max_results=_int_config(settings, "mana_web_search_max_results", 8, min_value=1, max_value=25),
            max_injected_results=_int_config(settings, "mana_search_max_injected_results", 5, min_value=1, max_value=20),
            max_summary_words=_int_config(settings, "mana_search_max_summary_words", 80, min_value=20, max_value=200),
            enable_ask_agent=_bool_config(settings, "mana_search_enable_ask_agent", True),
        )
