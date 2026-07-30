from __future__ import annotations

from collections.abc import Callable

from mana_agent.media.config import MediaModalityConfig
from mana_agent.media.errors import MediaCapabilityError
from mana_agent.media.providers.base import MediaProvider
from mana_agent.media.providers.openai import OpenAIMediaProvider


ProviderFactory = Callable[[MediaModalityConfig, str], MediaProvider]


def _openai_factory(config: MediaModalityConfig, api_key: str) -> MediaProvider:
    return OpenAIMediaProvider(
        api_key=api_key,
        base_url=config.base_url or "https://api.openai.com/v1",
        timeout_seconds=config.timeout_seconds,
    )


class MediaProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {
            "openai": _openai_factory,
            "custom": _openai_factory,
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
