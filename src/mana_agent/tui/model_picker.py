from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from mana_agent.config.user_config import load_model_cache, save_model_cache
from mana_agent.tui.forms import text_input
from mana_agent.tui.menu import MenuOption, select_option
from mana_agent.tui.status import error, info


class ModelFetchError(RuntimeError):
    pass


def parse_model_ids(payload: dict[str, Any]) -> list[str]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    ids = [str(item.get("id", "")).strip() for item in data if isinstance(item, dict)]
    return sorted(dict.fromkeys(model_id for model_id in ids if model_id))


def fetch_openai_compatible_models(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: int = 15,
) -> list[str]:
    if not api_key.strip():
        raise ModelFetchError("API key is required to fetch models.")
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout_seconds))) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ModelFetchError(f"Model fetch failed with HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise ModelFetchError(f"Model fetch failed: {exc.reason}.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelFetchError(f"Model fetch failed: {exc}.") from exc
    models = parse_model_ids(payload)
    if not models:
        raise ModelFetchError("Model fetch succeeded, but no model IDs were returned.")
    return models


def load_or_fetch_models(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    timeout_seconds: int,
    force_refresh: bool = False,
) -> list[str]:
    if not force_refresh:
        cached = load_model_cache(provider, base_url)
        if cached and cached.models:
            return cached.models
    models = fetch_openai_compatible_models(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    save_model_cache(provider, base_url, models)
    return models


def select_model(
    *,
    title: str,
    role_label: str,
    models: list[str],
    current: str = "",
    allow_same_as_main: bool = False,
    allow_manual: bool = True,
) -> str:
    options: list[MenuOption] = []
    if allow_same_as_main:
        options.append(MenuOption("same_as_main", "Same as main model"))
    options.extend(MenuOption(model, model) for model in models)
    if allow_manual:
        options.append(MenuOption("manual", "Manual model ID"))
    selected = select_option(
        title=title,
        text=f"Select {role_label}.",
        options=options,
        default=current if current in {option.value for option in options} else (options[0].value if options else None),
    )
    if selected == "manual":
        return text_input("Manual model", f"Enter model ID for {role_label}:", default=current)
    return selected


def choose_models(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    timeout_seconds: int,
    current: dict[str, object],
    force_refresh: bool = False,
) -> dict[str, str]:
    models: list[str] = []
    try:
        models = load_or_fetch_models(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            force_refresh=force_refresh,
        )
        info(f"Loaded {len(models)} model(s) from {base_url}.")
    except ModelFetchError as exc:
        error(f"{exc}\nManual model entry is available.")
    main = select_model(
        title="Main model",
        role_label="the main chat model",
        models=models,
        current=str(current.get("OPENAI_CHAT_MODEL") or ""),
        allow_manual=True,
    )
    tool = select_model(
        title="Tool worker model",
        role_label="the tool worker model",
        models=models,
        current=str(current.get("OPENAI_TOOL_WORKER_MODEL") or ""),
        allow_same_as_main=True,
        allow_manual=True,
    )
    planner = select_model(
        title="Coding planner model",
        role_label="the coding planner model",
        models=models,
        current=str(current.get("OPENAI_CODING_PLANNER_MODEL") or ""),
        allow_same_as_main=True,
        allow_manual=True,
    )
    embed = select_model(
        title="Embedding model",
        role_label="the embedding model",
        models=models,
        current=str(current.get("OPENAI_EMBED_MODEL") or ""),
        allow_manual=True,
    )
    return {
        "OPENAI_CHAT_MODEL": main,
        "LLM_MODEL": main,
        "OPENAI_TOOL_WORKER_MODEL": main if tool == "same_as_main" else tool,
        "OPENAI_CODING_PLANNER_MODEL": main if planner == "same_as_main" else planner,
        "OPENAI_EMBED_MODEL": embed,
    }
