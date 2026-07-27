from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from mana_agent.config.user_config import load_model_cache, save_model_cache
from mana_agent.config.model_catalog import ModelCapability, ModelPurpose, descriptors_from_catalog, filter_models
from mana_agent.config.provider_registry import PROVIDERS
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


def parse_openrouter_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve OpenRouter's canonical IDs and useful catalog metadata."""
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    records: dict[str, dict[str, Any]] = {}
    for raw in data:
        if not isinstance(raw, dict) or not str(raw.get("id") or "").strip():
            continue
        item = dict(raw)
        model_id = str(item["id"]).strip()
        architecture = item.get("architecture") if isinstance(item.get("architecture"), dict) else {}
        supported = item.get("supported_parameters") if isinstance(item.get("supported_parameters"), list) else []
        modalities = architecture.get("input_modalities") if isinstance(architecture, dict) else []
        capabilities: set[str] = {ModelCapability.TEXT_GENERATION.value}
        if any(str(value).lower() in {"tools", "tool_choice", "parallel_tool_calls"} for value in supported):
            capabilities.add(ModelCapability.TOOL_CALLING.value)
        if any("structured" in str(value).lower() or "response_format" in str(value).lower() for value in supported):
            capabilities.add(ModelCapability.STRUCTURED_OUTPUT.value)
        if any(str(value).lower() in {"image", "image_url"} for value in modalities):
            capabilities.add(ModelCapability.IMAGE_INPUT.value)
        if any("reasoning" in str(value).lower() for value in supported):
            capabilities.add(ModelCapability.REASONING.value)
        item["capabilities"] = sorted(capabilities)
        item["input_modalities"] = modalities
        records[model_id] = item
    return [records[key] for key in sorted(records)]


def fetch_openai_compatible_models(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: int = 15,
) -> list[str | dict[str, Any]]:
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


def fetch_provider_models(*, provider: str, base_url: str, api_key: str, timeout_seconds: int = 15) -> list[str | dict[str, Any]]:
    """Fetch one provider catalog without converting OpenRouter into an alias."""
    if not api_key.strip():
        raise ModelFetchError("API key is required to fetch models.")
    try:
        definition = PROVIDERS.get(provider)
    except KeyError as exc:
        raise ModelFetchError(str(exc)) from exc
    url = (base_url or definition.default_base_url).rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}", **dict(definition.default_headers)}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout_seconds))) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ModelFetchError(f"{definition.display_name} model fetch failed with HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise ModelFetchError(f"{definition.display_name} model fetch failed: {exc.reason}.") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelFetchError(f"{definition.display_name} model fetch failed: {exc}.") from exc
    models = parse_openrouter_models(payload) if provider == "openrouter" else parse_model_ids(payload)
    if not models:
        raise ModelFetchError(f"{definition.display_name} model fetch succeeded, but no model IDs were returned.")
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
    models = fetch_provider_models(
        provider=provider,
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
    models: list[str | dict[str, Any]] = []
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
    descriptors = descriptors_from_catalog(provider, models)
    text_models = [item.id for item in filter_models(descriptors, ModelPurpose.AGENT)]
    embedding_models = [item.id for item in filter_models(descriptors, ModelPurpose.EMBEDDING)]
    main = select_model(
        title="Main model",
        role_label="the main chat model",
        models=text_models,
        current=str(current.get("OPENAI_CHAT_MODEL") or ""),
        allow_manual=True,
    )
    tool = select_model(
        title="Tool worker model",
        role_label="the tool worker model",
        models=text_models,
        current=str(current.get("OPENAI_TOOL_WORKER_MODEL") or ""),
        allow_same_as_main=True,
        allow_manual=True,
    )
    planner = select_model(
        title="Coding planner model",
        role_label="the coding planner model",
        models=text_models,
        current=str(current.get("OPENAI_CODING_PLANNER_MODEL") or ""),
        allow_same_as_main=True,
        allow_manual=True,
    )
    embed = select_model(
        title="Embedding model",
        role_label="the embedding model",
        models=embedding_models,
        current=str(current.get("OPENAI_EMBED_MODEL") or ""),
        allow_manual=True,
    )
    resolved_tool = main if tool == "same_as_main" else tool
    resolved_planner = main if planner == "same_as_main" else planner
    return {
        "OPENAI_CHAT_MODEL": main,
        "LLM_MODEL": main,
        "OPENAI_TOOL_WORKER_MODEL": resolved_tool,
        "OPENAI_CODING_PLANNER_MODEL": resolved_planner,
        "OPENAI_EMBED_MODEL": embed,
        "MODEL_LEVEL_3_HIGH_REASONING": main,
        "MODEL_LEVEL_2_CODING": resolved_planner,
        "MODEL_LEVEL_1_FAST_TOOL": resolved_tool,
    }
