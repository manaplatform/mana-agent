from __future__ import annotations

import json
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation
from typing import Any

from mana_agent.config.user_config import load_model_cache, save_model_cache
from mana_agent.config.catalog_service import (
    ModelFetchError,
    ModelListFetchFailedError,
    ProviderAuthenticationFailedError,
    ProviderConnectionFailedError,
    ProviderValidationError,
)
from mana_agent.config.model_catalog import ModelCapability, ModelPurpose, descriptors_from_catalog, filter_models
from mana_agent.config.provider_registry import PROVIDERS
from mana_agent.tui.forms import text_input
from mana_agent.tui.menu import MenuOption, select_option
from mana_agent.tui.status import error, info


def parse_model_ids(payload: dict[str, Any]) -> list[str]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    ids = [str(item.get("id", "")).strip() for item in data if isinstance(item, dict)]
    return sorted(dict.fromkeys(model_id for model_id in ids if model_id))


def parse_openai_model_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve model IDs and lifecycle/deprecation metadata from OpenAI models API."""
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    records: dict[str, dict[str, Any]] = {}
    for raw in data:
        if isinstance(raw, str) and raw.strip():
            model_id = raw.strip()
            records[model_id] = {"id": model_id}
        elif isinstance(raw, dict) and str(raw.get("id") or "").strip():
            item = dict(raw)
            model_id = str(item["id"]).strip()
            item["id"] = model_id
            records[model_id] = item
    return [records[key] for key in sorted(records)]


def parse_openai_compatible_model_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve canonical upstream model IDs and any capability metadata present.

    Used for NVIDIA Build / NIM and other OpenAI-compatible catalogs where the
    ``id`` field is authoritative (including nested org namespaces such as
    ``deepseek-ai/deepseek-v4-flash`` or ``nvidia/nemotron-...``). IDs are never
    rewritten or stripped.
    """
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    records: dict[str, dict[str, Any]] = {}
    for raw in data:
        if not isinstance(raw, dict) or not str(raw.get("id") or "").strip():
            continue
        item = dict(raw)
        model_id = str(item["id"]).strip()
        # Preserve only capability hints the catalog actually supplies.
        capabilities: set[str] = set()
        supplied = item.get("capabilities")
        if isinstance(supplied, list):
            for value in supplied:
                text = str(value or "").strip().lower().replace("-", "_")
                if text:
                    capabilities.add(text)
        supported = item.get("supported_parameters") if isinstance(item.get("supported_parameters"), list) else []
        if any(str(value).lower() in {"tools", "tool_choice", "parallel_tool_calls"} for value in supported):
            capabilities.add(ModelCapability.TOOL_CALLING.value)
        if any(
            "structured" in str(value).lower() or "response_format" in str(value).lower()
            for value in supported
        ):
            capabilities.add(ModelCapability.STRUCTURED_OUTPUT.value)
        if any("reasoning" in str(value).lower() for value in supported):
            capabilities.add(ModelCapability.REASONING.value)
        architecture = item.get("architecture") if isinstance(item.get("architecture"), dict) else {}
        modalities = architecture.get("input_modalities") if isinstance(architecture, dict) else []
        if any(str(value).lower() in {"image", "image_url"} for value in modalities or []):
            capabilities.add(ModelCapability.IMAGE_INPUT.value)
        lowered = model_id.lower()
        if any(marker in lowered for marker in ("embed", "embedding")):
            capabilities.add(ModelCapability.EMBEDDING.value)
        elif capabilities or item.get("object") == "model":
            # Basic OpenAI-style model entries with no capability metadata remain
            # capability-empty so Advanced/manual selection can still use them.
            pass
        if capabilities:
            item["capabilities"] = sorted(capabilities)
        item["id"] = model_id
        records[model_id] = item
    return [records[key] for key in sorted(records)]


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
        capabilities: set[str] = set()
        lowered_id = model_id.lower()
        endpoint = item.get("_mana_endpoint", "")
        out_mods = architecture.get("output_modalities") if isinstance(architecture, dict) else []
        if not isinstance(out_mods, list):
            out_mods = []
        out_mods_lower = [str(m).lower() for m in out_mods]

        if endpoint == "/embeddings/models" or "embeddings" in out_mods_lower or any(marker in lowered_id for marker in ("embed", "embedding", "embedder")):
            capabilities.add(ModelCapability.EMBEDDING.value)
        elif endpoint == "/images/models" or "image" in out_mods_lower or any(marker in lowered_id for marker in ("dall-e", "image-gen", "image_generation", "flux", "stable-diffusion", "midjourney")):
            capabilities.add(ModelCapability.IMAGE_GENERATION.value)
        elif endpoint == "/videos/models" or "video" in out_mods_lower or isinstance(item.get("supported_durations"), list) or any(marker in lowered_id for marker in ("sora", "video-gen", "video_generation", "runway", "luma", "haiper", "minimax/video", "kling")):
            capabilities.add(ModelCapability.VIDEO_GENERATION.value)
        elif "audio" in out_mods_lower or "voice" in out_mods_lower or any(marker in lowered_id for marker in ("tts", "voice", "audio", "speech", "elevenlabs")):
            capabilities.add(ModelCapability.TEXT_TO_SPEECH.value)
        else:
            capabilities.add(ModelCapability.TEXT_GENERATION.value)

        if any(str(value).lower() in {"tools", "tool_choice", "parallel_tool_calls"} for value in supported):
            capabilities.add(ModelCapability.TOOL_CALLING.value)
        if any("structured" in str(value).lower() or "response_format" in str(value).lower() for value in supported):
            capabilities.add(ModelCapability.STRUCTURED_OUTPUT.value)
        if any(str(value).lower() in {"image", "image_url"} for value in modalities or []):
            capabilities.add(ModelCapability.IMAGE_INPUT.value)
        if any("reasoning" in str(value).lower() for value in supported):
            capabilities.add(ModelCapability.REASONING.value)
        item["capabilities"] = sorted(capabilities)
        item["input_modalities"] = modalities
        top_provider = item.get("top_provider") if isinstance(item.get("top_provider"), dict) else {}
        if item.get("max_output_tokens") is None and top_provider.get("max_completion_tokens") is not None:
            item["max_output_tokens"] = top_provider["max_completion_tokens"]
        pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
        for source_key, target_key in (
            ("prompt", "input_price_per_million"),
            ("completion", "output_price_per_million"),
            ("input_cache_read", "cached_input_price_per_million"),
        ):
            if pricing.get(source_key) not in (None, ""):
                try:
                    item[target_key] = str(Decimal(str(pricing[source_key])) * Decimal(1_000_000))
                except (InvalidOperation, ValueError):
                    pass
        records[model_id] = item
    return [records[key] for key in sorted(records)]


def _handle_http_error(provider_label: str, exc: urllib.error.HTTPError) -> None:
    code = int(exc.code)
    if code in {401, 403}:
        raise ProviderAuthenticationFailedError(
            f"{provider_label} authentication failed (HTTP {code}). Check the API key and account access."
        ) from exc
    if code == 404:
        raise ModelListFetchFailedError(
            f"{provider_label} model-list fetch failed: endpoint was not found (HTTP 404). "
            "Verify the base URL ends with /v1 for OpenAI-compatible hosts."
        ) from exc
    if code == 429:
        raise ModelListFetchFailedError(
            f"{provider_label} model-list fetch failed: rate limit or quota was exceeded (HTTP 429)."
        ) from exc
    if code >= 500:
        raise ModelListFetchFailedError(
            f"{provider_label} model-list fetch failed: service failure (HTTP {code})."
        ) from exc
    raise ModelListFetchFailedError(f"{provider_label} model-list fetch failed with HTTP {code}.") from exc


def _http_error_message(provider_label: str, exc: urllib.error.HTTPError) -> str:
    code = int(exc.code)
    if code in {401, 403}:
        return (
            f"{provider_label} authentication failed (HTTP {code}). "
            "Check the API key and account access."
        )
    if code == 404:
        return (
            f"{provider_label} model-list fetch failed: endpoint was not found (HTTP 404). "
            "Verify the base URL ends with /v1 for OpenAI-compatible hosts."
        )
    if code == 429:
        return f"{provider_label} model-list fetch failed: rate limit or quota was exceeded (HTTP 429)."
    if code >= 500:
        return f"{provider_label} model-list fetch failed: service failure (HTTP {code})."
    return f"{provider_label} model-list fetch failed with HTTP {code}."


def fetch_openai_compatible_models(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: int = 15,
) -> list[str | dict[str, Any]]:
    if not api_key.strip():
        raise ProviderAuthenticationFailedError("API key is required to fetch models.")
    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=max(1, int(timeout_seconds))) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        _handle_http_error("Provider", exc)
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if "timed out" in str(reason).lower() or "timeout" in str(reason).lower():
            raise ProviderConnectionFailedError(
                f"Provider provider connection failed: timed out: {reason}."
            ) from exc
        raise ProviderConnectionFailedError(
            f"Provider provider connection failed: Model fetch failed: {reason}."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelListFetchFailedError(
            f"Provider model-list fetch failed: Model fetch failed: {exc}."
        ) from exc
    models = parse_openai_model_records(payload)
    if not models:
        raise ModelListFetchFailedError("Provider model-list fetch failed: no model IDs were returned.")
    return models


def fetch_provider_models(*, provider: str, base_url: str, api_key: str, timeout_seconds: int = 15) -> list[str | dict[str, Any]]:
    """Fetch one provider catalog without converting multi-tenant hosts into aliases."""
    if not api_key.strip():
        raise ProviderAuthenticationFailedError("API key is required to fetch models.")
    try:
        definition = PROVIDERS.get(provider)
    except KeyError as exc:
        raise ModelListFetchFailedError(str(exc)) from exc

    headers = {"Authorization": f"Bearer {api_key}", **dict(definition.default_headers)}
    base = (base_url or definition.default_base_url).rstrip("/")

    def _fetch(endpoint: str) -> dict[str, Any]:
        request = urllib.request.Request(base + endpoint, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=max(1, int(timeout_seconds))) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            _handle_http_error(definition.display_name, exc)
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if "timed out" in str(reason).lower() or "timeout" in str(reason).lower():
                raise ProviderConnectionFailedError(
                    f"{definition.display_name} provider connection failed: timed out: {reason}."
                ) from exc
            raise ProviderConnectionFailedError(
                f"{definition.display_name} provider connection failed: Model fetch failed: {reason}."
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelListFetchFailedError(
                f"{definition.display_name} model-list fetch failed: Model fetch failed: {exc}."
            ) from exc

    if provider == "openrouter":
        all_data: list[Any] = []
        seen: set[str] = set()
        for endpoint in ("/models", "/embeddings/models", "/images/models", "/videos/models"):
            try:
                payload = _fetch(endpoint)
                if isinstance(payload.get("data"), list):
                    for item in payload["data"]:
                        if isinstance(item, dict):
                            item_id = str(item.get("id", ""))
                            if item_id and item_id not in seen:
                                seen.add(item_id)
                                item["_mana_endpoint"] = endpoint
                                all_data.append(item)
            except ModelFetchError:
                if endpoint == "/models":
                    raise
        models: list[str | dict[str, Any]] = parse_openrouter_models({"data": all_data})
    elif provider == "nvidia":
        payload = _fetch("/models")
        models = parse_openai_compatible_model_records(payload)
    else:
        payload = _fetch("/models")
        models = parse_openai_model_records(payload)

    if not models:
        raise ModelListFetchFailedError(
            f"{definition.display_name} model-list fetch failed: no model IDs were returned."
        )
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
