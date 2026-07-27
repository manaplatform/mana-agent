from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AuthenticationMethod(str, Enum):
    API_KEY = "api_key"
    NONE = "none"


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
            ),
            ProviderDefinition(
                id="openrouter",
                display_name="OpenRouter",
                auth_method=AuthenticationMethod.API_KEY,
                default_base_url="https://openrouter.ai/api/v1",
                api_key_env="OPENROUTER_API_KEY",
                default_headers=(("HTTP-Referer", "https://github.com/mana-agent/mana-agent"), ("X-OpenRouter-Title", "Mana-Agent")),
                supports_responses_api=True,
            ),
            ProviderDefinition(
                id="nvidia",
                display_name="NVIDIA NIM",
                auth_method=AuthenticationMethod.API_KEY,
                default_base_url="https://integrate.api.nvidia.com/v1",
                api_key_env="OPENAI_API_KEY",
            ),
            ProviderDefinition(
                id="custom",
                display_name="OpenAI-compatible provider",
                auth_method=AuthenticationMethod.API_KEY,
                default_base_url="",
                api_key_env="OPENAI_API_KEY",
                custom=True,
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


def qualify_model_id(provider: str, model_id: str) -> str:
    provider_id = str(provider or "").strip().lower()
    model = str(model_id or "").strip()
    if not provider_id or not model:
        raise ValueError("Provider and model ID are required.")
    if model.startswith(f"{provider_id}/"):
        return model
    return f"{provider_id}/{model}"


def split_qualified_model_id(value: str, *, default_provider: str = "openai") -> tuple[str, str]:
    text = str(value or "").strip()
    if "/" not in text:
        return default_provider, text
    provider, model = text.split("/", 1)
    if provider in {item.id for item in PROVIDERS.all()} and model:
        return provider, model
    return default_provider, text
