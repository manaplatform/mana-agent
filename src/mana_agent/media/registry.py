from __future__ import annotations

from collections.abc import Callable

from mana_agent.media.config import MediaModalityConfig
from mana_agent.media.errors import MediaCapabilityError
from mana_agent.media.providers.base import MediaProvider
from mana_agent.media.providers.openai import OpenAIMediaProvider
from mana_agent.media.providers.openrouter import OpenRouterMediaProvider


ProviderFactory = Callable[[MediaModalityConfig, str], MediaProvider]


def _openai_factory(config: MediaModalityConfig, api_key: str) -> MediaProvider:
    return OpenAIMediaProvider(
        api_key=api_key,
        base_url=config.base_url or "https://api.openai.com/v1",
        timeout_seconds=config.timeout_seconds,
    )


def _openrouter_factory(config: MediaModalityConfig, api_key: str) -> MediaProvider:
    from mana_agent.config.user_config import load_effective_settings

    settings = load_effective_settings(include_env=False)
    referer = str(settings.get("OPENROUTER_HTTP_REFERER") or "https://github.com/manaplatform/mana-agent")
    title = str(settings.get("OPENROUTER_TITLE") or "Mana-Agent")
    return OpenRouterMediaProvider(
        api_key=api_key,
        base_url=config.base_url or "https://openrouter.ai/api/v1",
        timeout_seconds=config.timeout_seconds,
        http_referer=referer,
        title=title,
    )


class MediaProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {
            "openai": _openai_factory,
            "custom": _openai_factory,
            "openrouter": _openrouter_factory,
        }

    def register(self, provider_id: str, factory: ProviderFactory) -> None:
        normalized = str(provider_id or "").strip().lower()
        if not normalized:
            raise ValueError("media provider ID is required")
        self._factories[normalized] = factory

    def create(self, config: MediaModalityConfig, api_key: str) -> MediaProvider:
        try:
            factory = self._factories[config.provider]
        except KeyError as exc:
            raise MediaCapabilityError(
                "media_provider_unsupported",
                f"Media provider {config.provider!r} is not supported.",
            ) from exc
        return factory(config, api_key)

    def provider_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


MEDIA_PROVIDERS = MediaProviderRegistry()
