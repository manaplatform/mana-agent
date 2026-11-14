from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import httpx

from mana_agent.config.model_catalog import (
    ModelDescriptor,
    descriptors_from_catalog,
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
        prov_norm = str(provider or "").strip().lower()
        if prov_norm in {"x-ai", "grok"}:
            prov_norm = "xai"
        try:
            if self.fetcher is not None:
                model_ids = self.fetcher(provider=prov_norm, base_url=base_url, api_key=api_key, timeout_seconds=timeout_seconds)
                from mana_agent.llm.discovery import ModelMetadataResolver
                resolver = ModelMetadataResolver()
                for record in model_ids:
                    resolver.resolve(record["id"] if isinstance(record, dict) else record, prov_norm, record if isinstance(record, dict) else None)
                save_model_cache(prov_norm, base_url, model_ids)
            else:
                from mana_agent.llm.discovery import ModelDiscoveryService
                profiles = ModelDiscoveryService().discover_models(provider=prov_norm, base_url=base_url, api_key=api_key, timeout_seconds=timeout_seconds)
                model_ids = [{"id": profile.model_id, **profile.metadata} for profile in profiles]
        except ProviderValidationError:
            raise
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code in {401, 403}:
                raise ProviderAuthenticationFailedError(f"Provider rejected model discovery credentials (HTTP {code}).") from exc
            raise ModelListFetchFailedError(f"Provider model discovery returned HTTP {code}.") from exc
        except httpx.RequestError as exc:
            raise ProviderConnectionFailedError(f"Provider model discovery failed ({type(exc).__name__}).") from exc
        except Exception as exc:
            raise ProviderValidationError(str(exc)) from exc
        return descriptors_from_catalog(prov_norm, model_ids, source="discovered")

    def cached(self, *, provider: str, base_url: str) -> list[ModelDescriptor]:
        prov_norm = str(provider or "").strip().lower()
        if prov_norm in {"x-ai", "grok"}:
            prov_norm = "xai"
        eff_base_url = str(base_url or "").strip()
        if not eff_base_url:
            try:
                from mana_agent.config.provider_registry import PROVIDERS

                eff_base_url = PROVIDERS.get(prov_norm).default_base_url
            except Exception:
                eff_base_url = ""
        cached = load_model_cache(prov_norm, eff_base_url)
        if cached is None and eff_base_url != base_url:
            cached = load_model_cache(prov_norm, base_url)
        if cached is None:
            return []
        return descriptors_from_catalog(prov_norm, cached.models, source="cached")

    def cached_profiles(self, *, provider: str, base_url: str = "") -> list[ModelDescriptor]:
        return self.cached(provider=provider, base_url=base_url)

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
        if p_lower in {"x-ai", "grok"}:
            p_lower = "xai"

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

        # Discovery walks the catalog and uses supported model detail endpoints in one pass.
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
            except Exception:
                pass

        # A configured model may have evidence without a provider catalog entry.
        caps = normalize_capabilities(p_lower, target)
        if caps:
            return ModelDescriptor(
                provider=p_lower,
                id=target,
                capabilities=caps,
                source="registry",
            )

        return None

    def validate(self, **kwargs: object) -> int:
        return len(self.refresh(**kwargs))  # type: ignore[arg-type]
