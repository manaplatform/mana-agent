from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mana_agent.config.model_catalog import ModelDescriptor, descriptors_from_catalog
from mana_agent.config.user_config import load_model_cache, save_model_cache


class ProviderValidationError(RuntimeError):
    pass


@dataclass(slots=True)
class ModelCatalogService:
    """Provider-layer model discovery used by both configuration and chat UI."""

    fetcher: Callable[..., list[str | dict[str, Any]]] | None = None

    def refresh(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        timeout_seconds: int = 15,
    ) -> list[ModelDescriptor]:
        if not api_key.strip():
            raise ProviderValidationError("Provider authentication is not configured.")
        try:
            fetch = self.fetcher
            if fetch is None:
                from mana_agent.tui.model_picker import fetch_provider_models

                fetch = fetch_provider_models
            model_ids = fetch(provider=provider, base_url=base_url, api_key=api_key, timeout_seconds=timeout_seconds)
        except Exception as exc:
            raise ProviderValidationError(str(exc)) from exc
        save_model_cache(provider, base_url, model_ids)
        return descriptors_from_catalog(provider, model_ids, source="discovered")

    def cached(self, *, provider: str, base_url: str) -> list[ModelDescriptor]:
        cached = load_model_cache(provider, base_url)
        if cached is None:
            return []
        return descriptors_from_catalog(provider, cached.models, source="cached")

    def validate(self, **kwargs: object) -> int:
        return len(self.refresh(**kwargs))  # type: ignore[arg-type]
