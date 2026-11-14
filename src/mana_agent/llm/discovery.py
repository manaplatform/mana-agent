"""Read-only provider discovery and models.dev enrichment; never capability probes."""
from __future__ import annotations

import asyncio
import json
import hashlib
from typing import Any
from urllib.parse import quote
import httpx
from mana_agent.llm.capabilities import ModelProfile, metadata_facts, observation
from mana_agent.llm.registry import GLOBAL_CAPABILITY_REGISTRY, ModelCapabilityRegistry, metadata_binding


class ModelMetadataResolver:
    def __init__(self, registry: ModelCapabilityRegistry | None = None):
        self.registry = registry or GLOBAL_CAPABILITY_REGISTRY

    def resolve(self, model_id: str, provider: str, raw_metadata: dict[str, Any] | None = None, *,
                binding: str = "", endpoint: str = "") -> ModelProfile:
        profile = self.registry.get(provider, model_id, binding) or ModelProfile(
            provider=provider, model_id=model_id, endpoint=endpoint, configuration_fingerprint=binding)
        if raw_metadata:
            facts = metadata_facts(raw_metadata, source="provider", scope=binding)
            for name, observations in profile.observations.items():
                if name not in facts and any(fact.source == "provider" for fact in observations):
                    facts[name] = observation(None, source="provider", scope=binding, method="omitted_in_refreshed_metadata")
            profile.metadata = dict(raw_metadata)
            profile = self.registry.merge(profile, facts)
        return profile


class ModelDiscoveryService:
    def __init__(self, resolver: ModelMetadataResolver | None = None, registry: ModelCapabilityRegistry | None = None,
                 client: httpx.AsyncClient | None = None):
        self.registry = registry or GLOBAL_CAPABILITY_REGISTRY
        self.resolver = resolver or ModelMetadataResolver(self.registry)
        self.client = client

    async def fetch_records(self, *, provider: str, base_url: str, api_key: str, timeout_seconds: float = 15,
                            default_headers: dict | None = None, query_params: dict | None = None) -> list[dict[str, Any]]:
        from mana_agent.config.provider_registry import PROVIDERS
        from mana_agent.config.inference_provider import resolve_inference_connection
        from mana_agent.config.settings import Settings
        definition = PROVIDERS.get(provider)
        if not api_key and definition.auth_method.value != "none":
            raise ValueError(f"{definition.display_name} credentials are required for model discovery")
        connection = resolve_inference_connection(Settings(), provider=provider, require_api_key=False)
        headers = dict(connection.headers if default_headers is None else default_headers)
        query = connection.query_params if query_params is None else query_params
        if definition.inference_protocol == "anthropic_messages":
            headers.setdefault("anthropic-version", "2023-06-01")
            headers["x-api-key"] = api_key
        elif api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        base = base_url.rstrip("/")
        from urllib.parse import urlsplit
        parsed = urlsplit(base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Configure an HTTP(S) provider base URL without credentials, query fields, or a fragment")
        if definition.inference_protocol == "anthropic_messages" and not base.endswith("/v1"):
            base += "/v1"
        client = self.client or httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=False)
        records: dict[str, dict[str, Any]] = {}
        try:
            for path_index, path in enumerate(definition.model_catalog_paths):
                params = dict(query)
                cursors = set()
                for _ in range(100):
                    response = await client.get(base + path, headers=headers, params=params)
                    if path_index and response.status_code in {404, 405}:
                        break
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload.get("data"), list):
                        raise ValueError("Provider catalog has no model records")
                    for record in payload["data"]:
                        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
                            raise ValueError("Provider returned an invalid model record")
                        records[record["id"]] = {**records.get(record["id"], {}), **record}
                    cursor = payload.get("next_cursor")
                    if not payload.get("has_more") and not cursor:
                        break
                    cursor = cursor or payload.get("last_id") or (payload["data"][-1]["id"] if payload["data"] else None)
                    if not cursor or cursor in cursors:
                        raise ValueError("Provider model pagination did not advance")
                    cursors.add(cursor)
                    params["cursor" if payload.get("next_cursor") else "after_id" if definition.inference_protocol == "anthropic_messages" else "after"] = cursor
                else:
                    raise ValueError("Provider model catalog exceeds the bounded pagination limit")
            if definition.model_detail_path:
                semaphore = asyncio.Semaphore(2)
                async def detail(record):
                    async with semaphore:
                        response = await client.get(base + definition.model_detail_path.format(model=quote(record["id"], safe="")), headers=headers, params=query)
                        if response.status_code in {404, 405}:
                            return record
                        response.raise_for_status()
                        value = response.json()
                        value = value.get("data", value)
                        if not isinstance(value, dict) or value.get("id") != record["id"]:
                            raise ValueError("Provider detail identity does not match the requested model")
                        return {**record, **value}
                detailed = await asyncio.gather(*(detail(record) for record in records.values()))
                records = {record["id"]: record for record in detailed}
            return list(records.values())
        finally:
            if self.client is None:
                await client.aclose()

    async def catalog(self) -> dict[str, Any]:
        """Public catalog fetch never receives provider credentials or prompt data."""
        from datetime import datetime, timezone, timedelta
        from mana_agent.workspaces.paths import mana_home
        from mana_agent.workspaces.store import atomic_write_json
        path = mana_home() / "cache" / "models-dev.json"
        saved = json.loads(path.read_text()) if path.exists() else None
        if saved and datetime.fromisoformat(saved["fetched_at"]) + timedelta(days=1) > datetime.now(timezone.utc):
            self.catalog_observed_at = datetime.fromisoformat(saved["fetched_at"])
            return saved["catalog"]
        # Reuse an injected transport for tests, but construct the public request
        # independently of its defaults, including auth, cookies and headers.
        client = self.client or httpx.AsyncClient(timeout=15, follow_redirects=False)
        try:
            headers = {"If-None-Match": saved["etag"]} if saved and saved.get("etag") else {}
            request = httpx.Request("GET", "https://models.dev/api.json?type=all", headers=headers,
                                    extensions={"timeout": dict.fromkeys(("connect", "read", "write", "pool"), 15)})
            response = await client.send(request, stream=True, auth=None, follow_redirects=False)
            try:
                if response.status_code == 304 and saved:
                    saved["fetched_at"] = datetime.now(timezone.utc).isoformat()
                    atomic_write_json(path, saved)
                    self.catalog_observed_at = datetime.fromisoformat(saved["fetched_at"])
                    return saved["catalog"]
                response.raise_for_status()
                pieces, size = [], 0
                async for piece in response.aiter_bytes():
                    size += len(piece)
                    if size > 32 * 1024 * 1024:
                        raise ValueError("models.dev catalog exceeds the 32 MiB input bound")
                    pieces.append(piece)
                catalog = json.loads(b"".join(pieces))
                if not isinstance(catalog, dict):
                    raise ValueError("Invalid models.dev catalog")
                self.catalog_observed_at = datetime.now(timezone.utc)
                atomic_write_json(path, {"fetched_at": self.catalog_observed_at.isoformat(), "etag": response.headers.get("etag"), "catalog": catalog})
                return catalog
            finally:
                await response.aclose()
        finally:
            if self.client is None:
                await client.aclose()

    async def discover(self, *, provider: str, base_url: str, api_key: str, timeout_seconds: float = 15,
                       default_headers: dict | None = None, query_params: dict | None = None) -> list[ModelProfile]:
        from mana_agent.config.user_config import save_model_cache
        from mana_agent.evals.recorder import record_current
        from mana_agent.config.inference_provider import resolve_inference_connection
        from mana_agent.config.settings import Settings
        from mana_agent.config.provider_registry import PROVIDERS
        provider = PROVIDERS.get(provider).id
        connection = resolve_inference_connection(Settings(), provider=provider, require_api_key=False)
        headers = connection.headers if default_headers is None else default_headers
        query = connection.query_params if query_params is None else query_params
        binding = metadata_binding(provider, base_url, hashlib.sha256(api_key.encode()).hexdigest(),
            hashlib.sha256(json.dumps(headers, sort_keys=True).encode()).hexdigest(),
            hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest())
        records = await self.fetch_records(provider=provider, base_url=base_url, api_key=api_key, timeout_seconds=timeout_seconds,
            default_headers=default_headers, query_params=query_params)
        from mana_agent.utils.redaction import redact_secrets
        records = [{**redact_secrets(record), "id": record["id"], "_mana_catalog_binding": binding} for record in records]
        save_model_cache(provider, base_url, records)
        try:
            catalog = await self.catalog()
        except (httpx.HTTPError, ValueError, OSError) as exc:
            # Previously observed metadata retains its original expiry on transport failure.
            record_current("model.catalog.failed", {"source": "models.dev", "error_type": type(exc).__name__})
            catalog = None
        provider_models = ((catalog or {}).get(provider) or {}).get("models") or {}
        profiles = []
        for record in records:
            profile = self.resolver.resolve(record["id"], provider, record, binding=binding, endpoint=base_url)
            external = provider_models.get(record["id"])
            if catalog is not None:
                from datetime import timedelta
                facts = metadata_facts(external, source="models.dev", scope=binding) if isinstance(external, dict) else {}
                for name, previous in profile.observations.items():
                    if name not in facts and any(fact.source == "models.dev" for fact in previous):
                        facts[name] = observation(None, source="models.dev", scope=binding, method="omitted_in_refreshed_metadata")
                if getattr(self, "catalog_observed_at", None):
                    facts = {name: fact.model_copy(update={"observed_at": self.catalog_observed_at, "expires_at": self.catalog_observed_at + timedelta(days=1)}) for name, fact in facts.items()}
                profile = self.registry.merge(profile, facts)
            profile.endpoint = base_url
            if isinstance(external, dict) and isinstance(external.get("cost"), dict):
                from datetime import datetime, timezone, timedelta
                observed = getattr(self, "catalog_observed_at", datetime.now(timezone.utc))
                profile.metadata["catalog_pricing"] = external["cost"]
                profile.metadata["catalog_pricing_expires_at"] = (observed + timedelta(days=1)).isoformat()
            elif catalog is not None:
                profile.metadata.pop("catalog_pricing", None)
                profile.metadata.pop("catalog_pricing_expires_at", None)
            profile.metadata["availability_endpoint"] = base_url
            profile.metadata.setdefault("available", True)
            self.registry.register(profile)
            profiles.append(profile)
        self.registry.save()
        return profiles

    def discover_models(self, *, provider: str, base_url: str, api_key: str = "", timeout_seconds: float = 15) -> list[ModelProfile]:
        return asyncio.run(self.discover(provider=provider, base_url=base_url, api_key=api_key, timeout_seconds=timeout_seconds))

    async def discover_configured(self, settings: Any) -> dict[str, Any]:
        from mana_agent.config.inference_provider import resolve_inference_connection
        ai_provider = getattr(settings, "mana_ai_provider", "openai") or "openai"
        configured_providers = getattr(settings, "mana_configured_providers", ["openai"]) or ["openai"]
        providers = list(dict.fromkeys([ai_provider, *configured_providers]))
        semaphore = asyncio.Semaphore(2)
        async def fetch(provider):
            async with semaphore:
                connection = resolve_inference_connection(settings, provider=provider)
                return await self.discover(provider=connection.provider, base_url=connection.base_url, api_key=connection.api_key,
                    default_headers=connection.headers, query_params=connection.query_params)
        results = await asyncio.gather(*(fetch(provider) for provider in providers), return_exceptions=True)
        return dict(zip(providers, results))

    def get_model(self, model_id: str, *, provider: str, base_url: str, api_key: str = "", timeout_seconds: float = 10) -> ModelProfile | None:
        # Catalog membership does not require a provider to expose a detail endpoint.
        profiles = self.discover_models(provider=provider, base_url=base_url, api_key=api_key, timeout_seconds=timeout_seconds)
        return next((profile for profile in profiles if profile.model_id == model_id), None)
