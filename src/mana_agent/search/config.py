from __future__ import annotations

import os
from dataclasses import dataclass


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _int_env(name: str, default: int, *, min_value: int = 1, max_value: int = 1000) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(min_value, min(max_value, value))


@dataclass(frozen=True, slots=True)
class SearchConfig:
    github_token: str = ""
    enable_web: bool = True
    enable_github: bool = True
    max_results: int = 8
    timeout_seconds: int = 15
    memory_ttl_days: int = 14
    web_provider: str = ""
    web_api_key: str = ""
    web_endpoint: str = ""
    web_max_results: int = 8
    max_injected_results: int = 5
    max_summary_words: int = 80
    enable_ask_agent: bool = True

    @classmethod
    def from_env(cls) -> "SearchConfig":
        return cls(
            github_token=os.getenv("MANA_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN") or "",
            enable_web=_bool_env("MANA_SEARCH_ENABLE_WEB", True),
            enable_github=_bool_env("MANA_SEARCH_ENABLE_GITHUB", True),
            max_results=_int_env("MANA_SEARCH_MAX_RESULTS", 8, min_value=1, max_value=25),
            timeout_seconds=_int_env("MANA_SEARCH_TIMEOUT_SECONDS", 15, min_value=1, max_value=60),
            memory_ttl_days=_int_env("MANA_SEARCH_MEMORY_TTL_DAYS", 14, min_value=1, max_value=365),
            web_provider=(os.getenv("MANA_WEB_SEARCH_PROVIDER") or "").strip().lower(),
            web_api_key=os.getenv("MANA_WEB_SEARCH_API_KEY") or "",
            web_endpoint=os.getenv("MANA_WEB_SEARCH_ENDPOINT") or "",
            web_max_results=_int_env("MANA_WEB_SEARCH_MAX_RESULTS", 8, min_value=1, max_value=25),
            max_injected_results=_int_env("MANA_SEARCH_MAX_INJECTED_RESULTS", 5, min_value=1, max_value=20),
            max_summary_words=_int_env("MANA_SEARCH_MAX_SUMMARY_WORDS", 80, min_value=20, max_value=200),
            enable_ask_agent=_bool_env("MANA_SEARCH_ENABLE_ASK_AGENT", True),
        )
