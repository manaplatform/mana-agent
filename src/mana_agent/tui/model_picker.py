from __future__ import annotations

import json
import urllib.error
import urllib.parse
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
from mana_agent.config.model_catalog import (
    ModelCapability,
    ModelPurpose,
    descriptors_from_catalog,
    extract_capabilities_from_record,
    filter_models,
)
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
            architecture = item.get("architecture") if isinstance(item.get("architecture"), dict) else {}
            modalities = architecture.get("input_modalities") if isinstance(architecture, dict) else item.get("input_modalities")
            if modalities and not item.get("input_modalities"):
                item["input_modalities"] = modalities
            caps = extract_capabilities_from_record(item)
            if caps:
                item["capabilities"] = sorted(str(c.value) for c in caps)
            records[model_id] = item
    return [records[key] for key in sorted(records)]


def parse_openai_compatible_model_records(
    payload: dict[str, Any],
    *,
    provider: str = "",
) -> list[dict[str, Any]]:
    """Preserve canonical upstream model IDs and any capability metadata present.

    Used for NVIDIA Build / NIM, xAI, and other OpenAI-compatible catalogs where the
    ``id`` field is authoritative (including nested org namespaces such as
    ``deepseek-ai/deepseek-v4-flash`` or ``nvidia/nemotron-...``). IDs are never
    rewritten or stripped.
    """
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    norm_provider = str(provider or "").strip().lower()
    if norm_provider in {"x-ai", "grok"}:
        norm_provider = "xai"
    records: dict[str, dict[str, Any]] = {}
    for raw in data:
        if not isinstance(raw, dict) or not str(raw.get("id") or "").strip():
            continue
        item = dict(raw)
        model_id = str(item["id"]).strip()
        caps = extract_capabilities_from_record(item)
        if caps:
            item["capabilities"] = sorted(str(c.value) for c in caps)
        item["id"] = model_id
        records[model_id] = item
    return [records[key] for key in sorted(records)]


def parse_openrouter_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return parse_openai_compatible_model_records(payload, provider="openrouter")


def _handle_http_error(provider_label: str, exc: urllib.error.HTTPError) -> None:
    code = int(exc.code)
    if code in {401, 403}:
        raise ProviderAuthenticationFailedError(
            f"{provider_label} authentication failed (HTTP {code}). Check the API key and account access."
        ) from exc
    if code == 404:
        if provider_label.casefold() == "anthropic":
            raise ModelListFetchFailedError(
                "Anthropic model-list fetch failed: API endpoint was not found (HTTP 404). "
                "Check the Anthropic API base URL; model listing uses /v1/models."
            ) from exc
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
        if provider_label.casefold() == "anthropic":
            return (
                "Anthropic model-list fetch failed: API endpoint was not found (HTTP 404). "
                "Check the Anthropic API base URL; model listing uses /v1/models."
            )
        return (
            f"{provider_label} model-list fetch failed: endpoint was not found (HTTP 404). "
            "Verify the base URL ends with /v1 for OpenAI-compatible hosts."
        )
    if code == 429:
        return f"{provider_label} model-list fetch failed: rate limit or quota was exceeded (HTTP 429)."
    if code >= 500:
        return f"{provider_label} model-list fetch failed: service failure (HTTP {code})."
    return f"{provider_label} model-list fetch failed with HTTP {code}."


def _provider_url(provider: str, base_url: str, endpoint: str) -> str:
    """Build a catalog URL using the provider's API path conventions.

    Anthropic's API is rooted at ``/v1`` while its default base URL is the host
    root. Accept base URLs both with and without that version prefix to avoid
    producing either ``/models`` or ``/v1/v1/models``.
    """
    base = base_url.rstrip("/")
    if provider.casefold() != "anthropic":
        return base + endpoint

    parsed = urllib.parse.urlsplit(base)
    path = parsed.path.rstrip("/")
    if path == "/v1" or path.endswith("/v1"):
        path += endpoint
    else:
        path += "/v1" + endpoint
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
    )


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
    from mana_agent.llm.discovery import ModelDiscoveryService
    profiles = ModelDiscoveryService().discover_models(provider=provider, base_url=base_url, api_key=api_key, timeout_seconds=timeout_seconds)
    return [{"id": profile.model_id, **profile.metadata} for profile in profiles]


def fetch_provider_model_detail(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model_id: str,
    timeout_seconds: int = 10,
) -> dict[str, Any] | None:
    from mana_agent.llm.discovery import ModelDiscoveryService
    profile = ModelDiscoveryService().get_model(model_id, provider=provider, base_url=base_url, api_key=api_key, timeout_seconds=timeout_seconds)
    return profile.metadata if profile is not None else None


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
    for model in models:
        label = model
        try:
            from mana_agent.config.model_discovery import GLOBAL_CAPABILITY_REGISTRY

            prof = GLOBAL_CAPABILITY_REGISTRY.get("openai", model)
            if prof:
                b_badge = "Batch: Yes" if prof.supports_batch else ("Batch: No" if prof.supports_batch is False else "Batch: ?")
                ctx_str = f"{prof.context_window // 1000}k ctx" if prof.context_window else ""
                parts = [p for p in (b_badge, ctx_str) if p]
                if parts:
                    label = f"{model} ({' · '.join(parts)})"
        except Exception:
            pass
        options.append(MenuOption(model, label))
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


def choose_models(*, provider: str, base_url: str, api_key: str, timeout_seconds: int,
                  current: dict[str, object], force_refresh: bool = False) -> dict[str, str]:
    models = load_or_fetch_models(provider=provider, base_url=base_url, api_key=api_key,
        timeout_seconds=timeout_seconds, force_refresh=force_refresh)
    choices = [item.id for item in descriptors_from_catalog(provider, models)]
    selected = {}
    for key, title in (("OPENAI_CHAT_MODEL", "Main model"), ("OPENAI_CODING_PLANNER_MODEL", "Coding model"), ("OPENAI_TOOL_WORKER_MODEL", "Fast model")):
        selected[key] = select_model(models=choices, current=str(current.get(key) or ""), title=title,
            role_label=title, allow_manual=True)
    return selected
