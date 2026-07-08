from __future__ import annotations

from dataclasses import dataclass

from mana_agent.search.config import SearchConfig
from mana_agent.search.github_provider import GitHubSearchProvider
from mana_agent.search.web_provider import ConfiguredWebSearchProvider


@dataclass(slots=True)
class SearchProviders:
    web: ConfiguredWebSearchProvider | None
    github: GitHubSearchProvider | None


def build_search_providers(config: SearchConfig) -> SearchProviders:
    web = (
        ConfiguredWebSearchProvider(
            provider=config.web_provider,
            api_key=config.web_api_key,
            endpoint=config.web_endpoint,
            engine_id=config.web_engine_id,
            timeout_seconds=config.timeout_seconds,
        )
        if config.enable_web
        else None
    )
    github = (
        GitHubSearchProvider(token=config.github_token, timeout_seconds=config.timeout_seconds)
        if config.enable_github
        else None
    )
    return SearchProviders(web=web, github=github)
