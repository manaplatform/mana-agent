from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mana_agent.config.model_catalog import (
    ModelDescriptor,
    descriptors_from_catalog,
    extract_capabilities_from_record,
    normalize_capabilities,
)
from mana_agent.config.user_config import load_model_cache, save_model_cache


class ProviderValidationError(RuntimeError):
    pass


class ModelFetchError(ProviderValidationError):
    pass


class ProviderConnectionFailedError(ModelFetchError):
    pass


class ProviderAuthenticationFailedError(ModelFetchError):
    pass


class ModelListFetchFailedError(ModelFetchError):
    pass


@dataclass
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
        except ProviderValidationError:
            raise
        except Exception as exc:
            raise ProviderValidationError(str(exc)) from exc
        save_model_cache(provider, base_url, model_ids)
        try:
            from mana_agent.config.model_capabilities import clear_capability_cache

            clear_capability_cache()
        except Exception:
            pass
        return descriptors_from_catalog(provider, model_ids, source="discovered")

    def cached(self, *, provider: str, base_url: str) -> list[ModelDescriptor]:
        cached = load_model_cache(provider, base_url)
        if cached is None:
            return []
        return descriptors_from_catalog(provider, cached.models, source="cached")

    def get_descriptor(
        self,
        provider: str,
        model_id: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        allow_refresh: bool = True,
        timeout_seconds: int = 10,
    ) -> ModelDescriptor | None:
        """Look up model descriptor from catalog cache or query provider API.

        Dynamic retrieval of model capabilities from live provider catalog responses,
        avoiding hardcoded model family and capability lists.
        """
        target = str(model_id or "").strip()
        if not target:
            return None
        p_lower = str(provider or "").strip().lower()

        # Resolve credentials if not provided
        eff_base_url = base_url
        eff_api_key = api_key
        if eff_base_url is None or eff_api_key is None:
            try:
                from mana_agent.config.inference_provider import resolve_inference_connection
                from mana_agent.config.settings import Settings

                conn = resolve_inference_connection(Settings(), provider=p_lower, require_api_key=False)
                if eff_base_url is None:
                    eff_base_url = conn.base_url
                if eff_api_key is None:
                    eff_api_key = conn.api_key
            except Exception:
                pass

        # 1. Search cached descriptors
        cached_models = self.cached(provider=p_lower, base_url=eff_base_url or "")
        for desc in cached_models:
            if desc.id == target or desc.qualified_id == target:
                if desc.capabilities:
                    return desc
            if target.endswith(f"/{desc.id}") or desc.id.endswith(f"/{target}"):
                if desc.capabilities:
                    return desc

        # 2. Try fetching single model detail directly from provider API endpoint
        if allow_refresh and eff_api_key and eff_base_url:
            try:
                from mana_agent.tui.model_picker import fetch_provider_model_detail

                detail = fetch_provider_model_detail(
                    provider=p_lower,
                    base_url=eff_base_url,
                    api_key=eff_api_key,
                    model_id=target,
                    timeout_seconds=timeout_seconds,
                )
                if detail and isinstance(detail, dict):
                    extracted = extract_capabilities_from_record(detail)
                    caps = normalize_capabilities(p_lower, target, extracted or detail.get("capabilities"))
                    if caps:
                        return ModelDescriptor(
                            provider=p_lower,
                            id=target,
                            capabilities=caps,
                            source="api_detail",
                            metadata=detail,
                        )
            except Exception:
                pass

        # 3. If not found in cache and refresh is allowed, fetch from provider API catalog
        if allow_refresh and eff_api_key and eff_base_url:
            try:
                refreshed = self.refresh(
                    provider=p_lower,
                    base_url=eff_base_url,
                    api_key=eff_api_key,
                    timeout_seconds=timeout_seconds,
                )
                for desc in refreshed:
                    if desc.id == target or desc.qualified_id == target:
                        return desc
                    if target.endswith(f"/{desc.id}") or desc.id.endswith(f"/{target}"):
                        return desc
            except Exception:
                pass

        # 3. Baseline descriptor via normalize_capabilities if known
        caps = normalize_capabilities(p_lower, target)
        if caps:
            return ModelDescriptor(
                provider=p_lower,
                id=target,
                capabilities=caps,
                source="inferred",
            )

        return None

    def validate(self, **kwargs: object) -> int:
        return len(self.refresh(**kwargs))  # type: ignore[arg-type]
