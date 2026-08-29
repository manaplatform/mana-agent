from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AuthenticationMethod(str, Enum):
    API_KEY = "api_key"
    NONE = "none"


class CodexTransport(str, Enum):
    """How the official Codex app-server can reach this provider.

    Native Responses capability and Codex usability are separate concepts.
    Chat Completions-only hosts can still run Codex through Mana's local
    Responses compatibility bridge without claiming native Responses support.
    """

    DIRECT_RESPONSES = "direct_responses"
    RESPONSES_BRIDGE = "responses_bridge"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class ProviderDefinition:
    id: str
    display_name: str
    auth_method: AuthenticationMethod
    default_base_url: str
    api_key_env: str
    default_headers: tuple[tuple[str, str], ...] = ()
    default_env_headers: tuple[tuple[str, str], ...] = ()
    default_query_params: tuple[tuple[str, str], ...] = ()
    supports_model_refresh: bool = True
    supports_validation: bool = True
    supports_responses_api: bool = False
    codex_transport: CodexTransport = CodexTransport.UNSUPPORTED
    custom: bool = False


class ProviderRegistry:
    """Canonical inventory of providers backed by working runtime adapters.

    Mana-Agent currently executes all configured inference through its
    OpenAI-compatible adapter.  Keeping that fact here prevents the CLI and
    TUI from advertising providers that the runtime cannot actually call.
    """

    def __init__(self, providers: tuple[ProviderDefinition, ...] | None = None) -> None:
        self._providers = providers or (
            ProviderDefinition(
                id="openai",
                display_name="OpenAI",
                auth_method=AuthenticationMethod.API_KEY,
                default_base_url="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY",
                supports_responses_api=True,
                codex_transport=CodexTransport.DIRECT_RESPONSES,
            ),
            ProviderDefinition(
                id="openrouter",
                display_name="OpenRouter",
                auth_method=AuthenticationMethod.API_KEY,
                default_base_url="https://openrouter.ai/api/v1",
                api_key_env="OPENROUTER_API_KEY",
                default_headers=(
                    ("HTTP-Referer", "https://github.com/manaplatform/mana-agent"),
                    ("X-OpenRouter-Title", "Mana-Agent"),
                ),
                supports_responses_api=True,
                codex_transport=CodexTransport.DIRECT_RESPONSES,
            ),
            ProviderDefinition(
                id="nvidia",
                display_name="NVIDIA",
                auth_method=AuthenticationMethod.API_KEY,
                default_base_url="https://integrate.api.nvidia.com/v1",
                api_key_env="NVIDIA_API_KEY",
                # NVIDIA Build / NIM is OpenAI Chat Completions compatible.
                # Do not claim Responses API support; Codex uses the bridge.
                supports_responses_api=False,
                codex_transport=CodexTransport.RESPONSES_BRIDGE,
            ),
            ProviderDefinition(
                id="custom",
                display_name="OpenAI-compatible provider",
                auth_method=AuthenticationMethod.API_KEY,
                default_base_url="",
                api_key_env="OPENAI_API_KEY",
                custom=True,
                codex_transport=CodexTransport.RESPONSES_BRIDGE,
            ),
        )

    def all(self) -> tuple[ProviderDefinition, ...]:
        return self._providers

    def get(self, provider_id: str) -> ProviderDefinition:
        normalized = str(provider_id or "").strip().lower()
        for provider in self._providers:
            if provider.id == normalized:
                return provider
        raise KeyError(f"Unsupported inference provider: {provider_id}")


PROVIDERS = ProviderRegistry()

# Providers whose upstream catalog IDs are usually bare model names (not org/model).
_SIMPLE_UPSTREAM_PROVIDERS = frozenset({"openai", "custom", "openrouter"})


def _known_provider_ids() -> set[str]:
    return {item.id for item in PROVIDERS.all()}


def qualify_model_id(provider: str, model_id: str) -> str:
    """Build a Mana-qualified model identity ``provider/upstream_id``.

    Upstream NVIDIA Build IDs may themselves contain slashes and may even
    begin with ``nvidia/`` (for models published under the NVIDIA org). Those
    must round-trip as::

        nvidia/nemotron-x  →  nvidia/nvidia/nemotron-x
        deepseek-ai/x      →  nvidia/deepseek-ai/x

    Already-qualified Mana IDs remain idempotent.
    """
    provider_id = str(provider or "").strip().lower()
    model = str(model_id or "").strip()
    if not provider_id or not model:
        raise ValueError("Provider and model ID are required.")
    if model.startswith(f"{provider_id}/"):
        split_provider, split_model = split_qualified_model_id(
            model, default_provider=provider_id
        )
        if split_provider == provider_id and f"{provider_id}/{split_model}" == model:
            return model
        # Bare upstream that happens to start with the provider name
        # (e.g. nvidia/nemotron-...) still needs a Mana provider prefix.
        return f"{provider_id}/{model}"
    return f"{provider_id}/{model}"


def split_qualified_model_id(
    value: str, *, default_provider: str = "openai"
) -> tuple[str, str]:
    """Split a Mana-qualified or bare upstream model ID.

    A leading known Mana provider is authoritative for fully qualified IDs
    such as ``openrouter/anthropic/claude-sonnet`` or
    ``nvidia/deepseek-ai/deepseek-v4-flash``.

    Multi-tenant hosts (OpenRouter, NVIDIA) may also store bare upstream IDs
    whose first segment collides with a Mana provider id (e.g. OpenRouter
    hosting ``openai/gpt-4.1-mini``). When ``default_provider`` is one of those
    hosts and the first segment is a simple single-namespace provider
    (``openai`` / ``custom``) with no further slash, the whole string is treated
    as the upstream model under the default provider.

    When the first segment equals the default provider and the remainder has
    no further slash, providers that publish under their own org name
    (NVIDIA) keep the full string as the upstream model ID so
    ``nvidia/nemotron-x`` is not reduced to ``nemotron-x``.
    """
    text = str(value or "").strip()
    default = str(default_provider or "openai").strip().lower() or "openai"
    if not text:
        return default, ""
    if "/" not in text:
        return default, text
    head, rest = text.split("/", 1)
    if not rest:
        return default, text
    known = _known_provider_ids()
    if head not in known:
        # org/model under the active provider (deepseek-ai/..., anthropic/...).
        return default, text
    if head == default:
        if "/" in rest:
            return head, rest
        if head in _SIMPLE_UPSTREAM_PROVIDERS:
            return head, rest
        # Nested catalog namespaces that share the provider name (nvidia/*).
        return head, text
    # Different known Mana provider prefix.
    # provider/org/model is always a fully qualified Mana identity.
    if "/" in rest:
        return head, rest
    # openai/gpt-4.1-mini stored as a bare OpenRouter/NVIDIA upstream ID.
    if default in {"openrouter", "nvidia"} and head in {"openai", "custom"}:
        return default, text
    return head, rest


def provider_credential_env_names(provider: str) -> tuple[str, str]:
    """Return ``(api_key_env, base_url_env)`` config keys for a provider."""
    provider_id = str(provider or "openai").strip().lower() or "openai"
    if provider_id == "openrouter":
        return "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"
    if provider_id == "nvidia":
        return "NVIDIA_API_KEY", "NVIDIA_BASE_URL"
    return "OPENAI_API_KEY", "OPENAI_BASE_URL"
