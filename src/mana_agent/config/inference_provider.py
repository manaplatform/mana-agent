"""Provider-neutral inference connection resolution.

This is deliberately the only place that maps persisted provider credentials
to a runtime transport.  Callers retain the selected provider alongside the
model instead of guessing it from a model ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mana_agent.config.provider_registry import PROVIDERS, provider_credential_env_names


@dataclass(frozen=True, slots=True)
class InferenceConnection:
    provider: str
    display_name: str
    api_key: str
    base_url: str
    headers: dict[str, str]
    env_headers: dict[str, str]
    query_params: dict[str, str]
    supports_responses_api: bool


class ProviderConfigurationError(ValueError):
    """A selected provider has no usable local configuration."""


def credentials_from_mapping(
    values: Mapping[str, Any],
    *,
    provider: str | None = None,
) -> tuple[str, str]:
    """Return ``(api_key, base_url)`` for ``provider`` from a settings mapping.

    Credentials are isolated per provider. NVIDIA never falls back to
    ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``, and OpenRouter never falls back
    to OpenAI credentials either.
    """
    provider_id = str(provider or values.get("MANA_AI_PROVIDER") or "openai").strip().lower() or "openai"
    definition = PROVIDERS.get(provider_id)
    key_env, base_env = provider_credential_env_names(provider_id)
    api_key = str(values.get(key_env) or "")
    base_url = str(values.get(base_env) or definition.default_base_url or "")
    return api_key, base_url.rstrip("/")


def resolve_inference_connection(
    settings: Any,
    *,
    provider: str | None = None,
    require_api_key: bool = True,
) -> InferenceConnection:
    provider_id = str(
        provider or getattr(settings, "mana_ai_provider", "openai") or "openai"
    ).strip().lower()
    definition = PROVIDERS.get(provider_id)
    if provider_id == "openrouter":
        api_key = str(getattr(settings, "openrouter_api_key", "") or "")
        base_url = str(
            getattr(settings, "openrouter_base_url", "") or definition.default_base_url
        )
        headers = {
            key: value
            for key, value in {
                "HTTP-Referer": str(getattr(settings, "openrouter_http_referer", "") or ""),
                "X-OpenRouter-Title": str(getattr(settings, "openrouter_title", "") or ""),
            }.items()
            if value
        }
    elif provider_id == "nvidia":
        api_key = str(getattr(settings, "nvidia_api_key", "") or "")
        base_url = str(
            getattr(settings, "nvidia_base_url", "") or definition.default_base_url
        )
        headers = dict(definition.default_headers)
    else:
        api_key = str(getattr(settings, "openai_api_key", "") or "")
        base_url = str(
            getattr(settings, "openai_base_url", "") or definition.default_base_url
        )
        headers = dict(definition.default_headers)
    if require_api_key and not api_key:
        raise ProviderConfigurationError(
            f"{definition.display_name} authentication is not configured. "
            f"Set {definition.api_key_env}."
        )
    return InferenceConnection(
        provider=provider_id,
        display_name=definition.display_name,
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        headers=headers,
        env_headers=dict(definition.default_env_headers),
        query_params=dict(definition.default_query_params),
        supports_responses_api=definition.supports_responses_api,
    )
