from __future__ import annotations

from mana_agent.config.user_config import validate_positive_int
from mana_agent.tui.forms import secret_input, text_input
from mana_agent.tui.menu import MenuOption, select_option


def configure_search_provider(current: dict[str, object]) -> dict[str, object]:
    provider = select_option(
        title="Search provider",
        text="Select the web search provider.",
        options=[
            MenuOption("disabled", "Disabled"),
            MenuOption("tavily", "Tavily"),
            MenuOption("brave", "Brave Search API"),
            MenuOption("exa", "Exa"),
            MenuOption("serpapi", "SerpAPI"),
            MenuOption("google_cse", "Google Programmable Search / Custom Search JSON API"),
            MenuOption("custom", "Custom HTTP provider"),
        ],
        default=str(current.get("MANA_WEB_SEARCH_PROVIDER") or "disabled"),
    )
    github = select_option(
        title="GitHub search",
        text="Enable GitHub search?",
        options=[MenuOption("true", "Enabled"), MenuOption("false", "Disabled")],
        default="true" if current.get("MANA_SEARCH_ENABLE_GITHUB", True) else "false",
    )
    values: dict[str, object] = {
        "MANA_SEARCH_ENABLE_GITHUB": github == "true",
        "MANA_GITHUB_TOKEN": str(current.get("MANA_GITHUB_TOKEN") or ""),
    }
    token = secret_input("GitHub token", "Optional GitHub token. Leave empty for unauthenticated search:")
    if token:
        values["MANA_GITHUB_TOKEN"] = token
    if provider == "disabled":
        values.update(
            {
                "MANA_SEARCH_ENABLE_WEB": False,
                "MANA_WEB_SEARCH_PROVIDER": "",
                "MANA_WEB_SEARCH_API_KEY": "",
            }
        )
        return values
    max_results = text_input(
        "Search results",
        "Maximum web search results:",
        default=str(current.get("MANA_WEB_SEARCH_MAX_RESULTS") or current.get("MANA_SEARCH_MAX_RESULTS") or 8),
    )
    values.update(
        {
            "MANA_SEARCH_ENABLE_WEB": True,
            "MANA_WEB_SEARCH_PROVIDER": provider,
            "MANA_WEB_SEARCH_API_KEY": secret_input("Search API key", f"API key for {provider}:"),
            "MANA_WEB_SEARCH_MAX_RESULTS": validate_positive_int("MANA_WEB_SEARCH_MAX_RESULTS", max_results, minimum=1, maximum=25),
        }
    )
    if provider == "google_cse":
        values["MANA_WEB_SEARCH_ENGINE_ID"] = text_input(
            "Google CSE",
            "Search engine ID / cx:",
            default=str(current.get("MANA_WEB_SEARCH_ENGINE_ID") or ""),
        )
    if provider == "custom":
        base_url = text_input(
            "Custom search",
            "Custom search endpoint/base URL:",
            default=str(current.get("MANA_WEB_SEARCH_BASE_URL") or current.get("MANA_WEB_SEARCH_ENDPOINT") or ""),
        )
        query_param = text_input(
            "Custom search",
            "Query parameter name:",
            default=str(current.get("MANA_WEB_SEARCH_QUERY_PARAM") or "q"),
        )
        values["MANA_WEB_SEARCH_BASE_URL"] = base_url
        values["MANA_WEB_SEARCH_ENDPOINT"] = base_url
        values["MANA_WEB_SEARCH_QUERY_PARAM"] = query_param or "q"
    return values
