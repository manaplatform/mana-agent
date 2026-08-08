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
from mana_agent.tui.configuration_app import run_configuration_tui
from mana_agent.config.provider_registry import PROVIDERS


PROVIDER_DEFAULTS = {
    item.id: (item.display_name, item.default_base_url)
    for item in PROVIDERS.all()
}
PROVIDER_DEFAULTS["manual"] = ("Manual / skip for now", "")

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
    from mana_agent.config.provider_registry import provider_credential_env_names

    provider = select_option(
        title="Model provider",
        text="Select the model provider.",
        options=[
            *(MenuOption(item.id, item.display_name) for item in PROVIDERS.all()),
            MenuOption("manual", PROVIDER_DEFAULTS["manual"][0]),
        ],
        default=str(current.get("MANA_AI_PROVIDER") or "openai"),
    )
    if provider == "manual":
        main = text_input(
            "Main model",
            "Enter main chat model ID:",
            default=str(current.get("OPENAI_CHAT_MODEL") or DEFAULT_USER_CONFIG["OPENAI_CHAT_MODEL"]),
        )
        return {
            "MANA_AI_PROVIDER": "custom",
            "OPENAI_API_KEY": str(current.get("OPENAI_API_KEY") or ""),
            "OPENAI_BASE_URL": str(current.get("OPENAI_BASE_URL") or ""),
            "OPENAI_CHAT_MODEL": main,
            "LLM_MODEL": main,
            "MODEL_LEVEL_3_HIGH_REASONING": main,
        }
    definition = PROVIDERS.get(provider)
    key_env, base_env = provider_credential_env_names(provider)
    _, default_url = PROVIDER_DEFAULTS[provider]
    base_url = text_input(
        f"{definition.display_name} base URL",
        f"{definition.display_name} OpenAI-compatible API base URL:",
        default=str(current.get(base_env) or default_url or definition.default_base_url),
    )
    api_key = secret_input(
        f"{definition.display_name} API key",
        f"Enter {definition.api_key_env}. It will not be printed back:",
    )
    values: dict[str, object] = {
        "MANA_AI_PROVIDER": provider,
        key_env: api_key or str(current.get(key_env) or ""),
        base_env: validate_base_url(base_url) if base_url else definition.default_base_url,
    }
    resolved_key = str(values[key_env])
    resolved_base = str(values[base_env])
    if confirm("Validate provider", "Fetch model list now?", default=True):
        values.update(
            choose_models(
                provider=provider,
                base_url=resolved_base,
                api_key=resolved_key,
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
        values.update({"OPENAI_CHAT_MODEL": main, "LLM_MODEL": main, "MODEL_LEVEL_3_HIGH_REASONING": main})
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


def run_setup_wizard(*, console: Console | None = None) -> bool:
    """Launch the Textual configuration application.

    ``console`` remains accepted for API compatibility; the full-screen app
    owns rendering so credentials never pass through plain terminal prompts.
    """
    _ = console
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise NonInteractivePromptError(
            "Configuration requires an interactive terminal. Run `mana-agent --configure` from a TTY."
        )
    return run_configuration_tui()


def ensure_setup(*, no_interactive: bool = False, command_needs_llm: bool = True, console: Console | None = None) -> bool:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return True
    if has_user_config() or is_user_config_valid():
        return
    if no_interactive or not (sys.stdin.isatty() and sys.stdout.isatty()):
        if command_needs_llm:
            raise NonInteractivePromptError(
                "Mana Agent configuration is missing. Run `mana-agent --configure` from an interactive terminal."
            )
        return False
    saved = run_setup_wizard(console=console)
    if not saved or not is_user_config_valid():
        raise NonInteractivePromptError("Configuration was not completed; chat was not started.")
    return True


def refresh_model_list(*, console: Console | None = None) -> None:
    from mana_agent.config.inference_provider import credentials_from_mapping

    target = console or Console()
    current = load_effective_settings(include_env=True)
    provider = str(current.get("MANA_AI_PROVIDER") or "openai")
    api_key, base_url = credentials_from_mapping(current, provider=provider)
    if not base_url:
        base_url = PROVIDERS.get(provider).default_base_url or "https://api.openai.com/v1"
    base_url = validate_base_url(base_url)
    try:
        models = load_or_fetch_models(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=_timeout(current),
            force_refresh=True,
        )
    except Exception as exc:  # noqa: BLE001 - displayed as TUI status
        error(str(exc), console=target)
        return
    success(f"Cached {len(models)} model(s) from {base_url}.", console=target)


def _save_selected_models() -> None:
    from mana_agent.config.inference_provider import credentials_from_mapping

    current = load_effective_settings(include_env=True)
    provider = str(current.get("MANA_AI_PROVIDER") or "openai")
    api_key, base_url = credentials_from_mapping(current, provider=provider)
    if not base_url:
        base_url = PROVIDERS.get(provider).default_base_url or "https://api.openai.com/v1"
    save_effective_user_config(
        validate_config_values(
            choose_models(
                provider=provider,
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=_timeout(current),
                current=current,
                force_refresh=False,
            )
        ),
        merge=True,
    )


def settings_menu(*, console: Console | None = None) -> None:
    target = console or Console()
    actions: dict[str, Callable[[], None]] = {
        "provider": lambda: save_effective_user_config(
            validate_config_values(configure_model_provider(load_effective_settings(include_env=True), force_refresh=True)),
            merge=True,
        ),
        "refresh": lambda: refresh_model_list(console=target),
        "models": lambda: _save_selected_models(),
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
