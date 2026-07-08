from __future__ import annotations

import sys
import os
from typing import Callable

from rich.console import Console

from mana_agent.config.user_config import (
    DEFAULT_USER_CONFIG,
    UserConfigError,
    has_user_config,
    is_user_config_valid,
    load_effective_settings,
    masked_config_summary,
    save_effective_user_config,
    validate_base_url,
    validate_config_values,
    validate_positive_int,
)
from mana_agent.tui.forms import confirm, secret_input, text_input
from mana_agent.tui.menu import MenuOption, NonInteractivePromptError, select_option
from mana_agent.tui.model_picker import choose_models, load_or_fetch_models
from mana_agent.tui.search_provider_picker import configure_search_provider
from mana_agent.tui.status import config_table, error, info, success


PROVIDER_DEFAULTS = {
    "openai": ("OpenAI", "https://api.openai.com/v1"),
    "custom": ("OpenAI-compatible custom endpoint", ""),
    "nvidia": ("NVIDIA OpenAI-compatible endpoint", "https://integrate.api.nvidia.com/v1"),
    "manual": ("Manual / skip for now", ""),
}

MODEL_ROLE_ENV = {
    "main": "MANA_MODEL_MAIN",
    "head_decision": "MANA_MODEL_HEAD_DECISION",
    "planner": "MANA_MODEL_PLANNER",
    "coding": "MANA_MODEL_CODING",
    "verifier": "MANA_MODEL_VERIFIER",
    "reviewer": "MANA_MODEL_REVIEWER",
    "tool": "MANA_MODEL_TOOL",
    "summarizer": "MANA_MODEL_SUMMARIZER",
}

MODEL_LEVEL_OPTIONS = [
    MenuOption("MODEL_LEVEL_1_FAST_TOOL", "MODEL_LEVEL_1_FAST_TOOL"),
    MenuOption("MODEL_LEVEL_2_CODING", "MODEL_LEVEL_2_CODING"),
    MenuOption("MODEL_LEVEL_3_HIGH_REASONING", "MODEL_LEVEL_3_HIGH_REASONING"),
]


def _timeout(current: dict[str, object]) -> int:
    return validate_positive_int(
        "MANA_SEARCH_TIMEOUT_SECONDS",
        current.get("MANA_SEARCH_TIMEOUT_SECONDS", 15),
        minimum=1,
        maximum=60,
    )


def configure_model_provider(current: dict[str, object], *, force_refresh: bool = False) -> dict[str, object]:
    provider = select_option(
        title="Model provider",
        text="Select the model provider.",
        options=[
            MenuOption("openai", PROVIDER_DEFAULTS["openai"][0]),
            MenuOption("custom", PROVIDER_DEFAULTS["custom"][0]),
            MenuOption("nvidia", PROVIDER_DEFAULTS["nvidia"][0]),
            MenuOption("manual", PROVIDER_DEFAULTS["manual"][0]),
        ],
        default="openai",
    )
    if provider == "manual":
        return {
            "OPENAI_API_KEY": str(current.get("OPENAI_API_KEY") or ""),
            "OPENAI_BASE_URL": str(current.get("OPENAI_BASE_URL") or ""),
            "OPENAI_CHAT_MODEL": text_input(
                "Main model",
                "Enter main chat model ID:",
                default=str(current.get("OPENAI_CHAT_MODEL") or DEFAULT_USER_CONFIG["OPENAI_CHAT_MODEL"]),
            ),
        }
    _, default_url = PROVIDER_DEFAULTS[provider]
    base_url = text_input(
        "Base URL",
        "OpenAI-compatible API base URL:",
        default=str(current.get("OPENAI_BASE_URL") or default_url or "https://api.openai.com/v1"),
    )
    api_key = secret_input("API key", "Enter API key. It will not be printed back:")
    values: dict[str, object] = {
        "OPENAI_API_KEY": api_key or str(current.get("OPENAI_API_KEY") or ""),
        "OPENAI_BASE_URL": validate_base_url(base_url),
    }
    if confirm("Validate provider", "Fetch model list now?", default=True):
        values.update(
            choose_models(
                provider=provider,
                base_url=str(values["OPENAI_BASE_URL"]),
                api_key=str(values["OPENAI_API_KEY"]),
                timeout_seconds=_timeout(current),
                current={**current, **values},
                force_refresh=force_refresh,
            )
        )
    else:
        main = text_input(
            "Main model",
            "Enter main chat model ID:",
            default=str(current.get("OPENAI_CHAT_MODEL") or DEFAULT_USER_CONFIG["OPENAI_CHAT_MODEL"]),
        )
        values.update({"OPENAI_CHAT_MODEL": main, "LLM_MODEL": main})
    return values


def configure_model_roles(current: dict[str, object]) -> dict[str, object]:
    values: dict[str, object] = {}
    for label, env_name in MODEL_ROLE_ENV.items():
        values[env_name] = select_option(
            title="Model role levels",
            text=f"Select model level for {label}.",
            options=MODEL_LEVEL_OPTIONS,
            default=str(current.get(env_name) or DEFAULT_USER_CONFIG.get(env_name) or MODEL_LEVEL_OPTIONS[0].value),
        )
    return values


def configure_numeric_defaults(current: dict[str, object]) -> dict[str, object]:
    return {
        "DEFAULT_TOP_K": validate_positive_int(
            "DEFAULT_TOP_K",
            text_input("Retrieval", "Default top-k:", default=str(current.get("DEFAULT_TOP_K") or 8)),
            minimum=1,
            maximum=1000,
        ),
        "MANA_SEARCH_MAX_RESULTS": validate_positive_int(
            "MANA_SEARCH_MAX_RESULTS",
            text_input("Search", "Maximum combined search results:", default=str(current.get("MANA_SEARCH_MAX_RESULTS") or 8)),
            minimum=1,
            maximum=25,
        ),
        "MANA_SEARCH_TIMEOUT_SECONDS": validate_positive_int(
            "MANA_SEARCH_TIMEOUT_SECONDS",
            text_input("Search", "Search/model fetch timeout seconds:", default=str(current.get("MANA_SEARCH_TIMEOUT_SECONDS") or 15)),
            minimum=1,
            maximum=60,
        ),
        "MANA_SEARCH_MEMORY_TTL_DAYS": validate_positive_int(
            "MANA_SEARCH_MEMORY_TTL_DAYS",
            text_input("Search memory", "Search memory TTL days:", default=str(current.get("MANA_SEARCH_MEMORY_TTL_DAYS") or 14)),
            minimum=1,
            maximum=365,
        ),
        "MANA_LLM_LOG_FILE": text_input(
            "LLM logging",
            "Optional LLM log file path:",
            default=str(current.get("MANA_LLM_LOG_FILE") or ""),
        ),
    }


def run_setup_wizard(*, console: Console | None = None) -> None:
    target = console or Console()
    info("First-run setup will save configuration under ~/.mana. Secrets are stored separately and masked in summaries.", console=target)
    current = load_effective_settings(include_env=True)
    values: dict[str, object] = {}
    values.update(configure_model_provider(current))
    current = {**current, **values}
    values.update(configure_model_roles(current))
    current = {**current, **values}
    values.update(configure_numeric_defaults(current))
    current = {**current, **values}
    values.update(configure_search_provider(current))
    cleaned = validate_config_values(values)
    save_effective_user_config(cleaned, merge=True)
    success("Mana Agent configuration saved under ~/.mana.", console=target)


def ensure_setup(*, no_interactive: bool = False, command_needs_llm: bool = True, console: Console | None = None) -> None:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return
    if has_user_config() or is_user_config_valid():
        return
    if no_interactive or not (sys.stdin.isatty() and sys.stdout.isatty()):
        if command_needs_llm:
            raise NonInteractivePromptError(
                "Mana Agent configuration is missing. Set OPENAI_API_KEY or run `mana-agent` in an interactive terminal to complete first-run setup."
            )
        return
    run_setup_wizard(console=console)


def refresh_model_list(*, console: Console | None = None) -> None:
    target = console or Console()
    current = load_effective_settings(include_env=True)
    base_url = validate_base_url(str(current.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"))
    api_key = str(current.get("OPENAI_API_KEY") or "")
    try:
        models = load_or_fetch_models(
            provider="openai-compatible",
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=_timeout(current),
            force_refresh=True,
        )
    except Exception as exc:  # noqa: BLE001 - displayed as TUI status
        error(str(exc), console=target)
        return
    success(f"Cached {len(models)} model(s) from {base_url}.", console=target)


def settings_menu(*, console: Console | None = None) -> None:
    target = console or Console()
    actions: dict[str, Callable[[], None]] = {
        "provider": lambda: save_effective_user_config(
            validate_config_values(configure_model_provider(load_effective_settings(include_env=True), force_refresh=True)),
            merge=True,
        ),
        "refresh": lambda: refresh_model_list(console=target),
        "models": lambda: save_effective_user_config(
            validate_config_values(
                choose_models(
                    provider="openai-compatible",
                    base_url=str(load_effective_settings(include_env=True).get("OPENAI_BASE_URL") or "https://api.openai.com/v1"),
                    api_key=str(load_effective_settings(include_env=True).get("OPENAI_API_KEY") or ""),
                    timeout_seconds=_timeout(load_effective_settings(include_env=True)),
                    current=load_effective_settings(include_env=True),
                    force_refresh=False,
                )
            ),
            merge=True,
        ),
        "roles": lambda: save_effective_user_config(validate_config_values(configure_model_roles(load_effective_settings(include_env=True))), merge=True),
        "search": lambda: save_effective_user_config(validate_config_values(configure_search_provider(load_effective_settings(include_env=True))), merge=True),
        "summary": lambda: config_table(masked_config_summary(), console=target),
    }
    while True:
        try:
            choice = select_option(
                title="Settings",
                text="Choose a settings action.",
                options=[
                    MenuOption("provider", "Change model provider/API key"),
                    MenuOption("refresh", "Refresh model list"),
                    MenuOption("models", "Change selected models"),
                    MenuOption("roles", "Change model role levels"),
                    MenuOption("search", "Configure search providers"),
                    MenuOption("summary", "Show current config summary"),
                    MenuOption("back", "Back"),
                ],
            )
            if choice == "back":
                return
            actions[choice]()
            if choice != "summary":
                success("Settings saved.", console=target)
        except (KeyboardInterrupt, EOFError):
            return
        except (UserConfigError, NonInteractivePromptError) as exc:
            error(str(exc), console=target)
