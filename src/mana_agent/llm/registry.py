"""One registry for native capability evidence, overrides and persisted probes."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any
from mana_agent.llm.capabilities import CapabilityFact, CapabilityState, ModelProfile, metadata_facts, observation

_PRIORITY = {"unknown": 0, "models.dev": 1, "provider": 2, "probe": 3, "override": 4}


def _winner(observations: list[CapabilityFact], scope: str) -> CapabilityFact:
    candidates = {}
    for fact in observations:
        if fact.scope not in {"", scope}:
            continue
        stamp = fact.observed_at.isoformat() if fact.observed_at else ""
        prior = candidates.get(fact.source)
        if prior is None or stamp >= prior[0]:
            candidates[fact.source] = (stamp, fact)
    applicable = [(stamp, fact) for stamp, fact in candidates.values()
        if fact.current() and fact.state is not CapabilityState.UNKNOWN]
    if not applicable:
        return CapabilityFact()
    return max(applicable, key=lambda item: (_PRIORITY.get(item[1].source, 0), item[0]))[1]


def scope_key(*, provider: str, model: str, endpoint: str, api_version: str = "", transport: str = "",
              configuration: dict[str, Any] | None = None, credential_revision: str = "", deployment: str = "") -> str:
    value = {"provider": provider, "model": model, "endpoint": endpoint.rstrip("/"), "api_version": api_version,
        "transport": transport, "configuration": configuration or {}, "credential_revision": credential_revision,
        "deployment": deployment, "probe_version": 1}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def metadata_binding(provider: str, endpoint: str, credential_revision: str, headers_revision: str, query_revision: str) -> str:
    return scope_key(provider=provider, model="", endpoint=endpoint,
        credential_revision=credential_revision, configuration={"headers_revision": headers_revision, "query_revision": query_revision})


class ModelCapabilityRegistry:
    def __init__(self, cache_path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._profiles: dict[tuple[str, str, str], ModelProfile] = {}
        self._cache_path = cache_path
        self._loaded = False

    @property
    def cache_path(self) -> Path:
        if self._cache_path is None:
            from mana_agent.workspaces.paths import mana_home
            return mana_home() / "cache" / "model-capabilities.json"
        return self._cache_path

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            if self.cache_path.exists():
                data = json.loads(self.cache_path.read_text())
                if data.get("schema_version") != 1:
                    raise ValueError("Unsupported model capability cache version; refresh model settings")
                for raw in data.get("profiles", []):
                    profile = ModelProfile.model_validate(raw)
                    self._profiles[(profile.provider, profile.model_id, profile.configuration_fingerprint)] = profile
            self._loaded = True

    def save(self) -> None:
        from mana_agent.workspaces.store import atomic_write_json
        with self._lock:
            profiles = []
            for profile in self._profiles.values():
                saved = profile.model_copy(deep=True)
                for name in saved.observations:
                    saved.observations[name] = [fact for fact in saved.observations[name] if fact.source != "override"]
                saved = self._refresh(saved)
                profiles.append(saved.model_dump(mode="json"))
            atomic_write_json(self.cache_path, {"schema_version": 1, "profiles": profiles})
        self.cache_path.chmod(0o600)

    def register(self, profile: ModelProfile, *, persist: bool = False) -> None:
        self._load()
        with self._lock:
            key = (profile.provider, profile.model_id, profile.configuration_fingerprint)
            self._profiles[key] = profile
            if persist:
                self.save()

    def merge(self, profile: ModelProfile, facts: dict[str, CapabilityFact], *, persist: bool = False) -> ModelProfile:
        self._load()
        with self._lock:
            current = self._profiles.get((profile.provider, profile.model_id, profile.configuration_fingerprint), profile).model_copy(deep=True)
            current.metadata.update(profile.metadata)
            if profile.endpoint:
                current.endpoint = profile.endpoint
            for name, fact in facts.items():
                if name in {"context_window", "max_output_tokens", "max_input_tokens"} and fact.state is CapabilityState.SUPPORTED and (type(fact.value) is not int or fact.value <= 0):
                    raise ValueError(f"{name} evidence must contain a positive integer")
                observations = current.observations.setdefault(name, [])
                observations[:] = [old for old in observations if not (old.source == fact.source and old.scope == fact.scope
                    and old.state == fact.state and old.value == fact.value)]
                observations.append(fact)
                if len(observations) > 48:
                    del observations[:-48]
                current.facts[name] = _winner(observations, current.configuration_fingerprint)
            self.register(current, persist=persist)
            return current

    def get(self, provider: str, model_id: str, scope: str = "") -> ModelProfile | None:
        self._load()
        with self._lock:
            profile = self._profiles.get((provider, model_id, scope))
            if profile is None:
                return None
            return self._refresh(profile)

    def _refresh(self, profile: ModelProfile) -> ModelProfile:
        refreshed = profile.model_copy(deep=True)
        for name, facts in profile.observations.items():
            refreshed.facts[name] = _winner(facts, profile.configuration_fingerprint)
        return refreshed

    def list_profiles(self, provider: str | None = None) -> list[ModelProfile]:
        self._load()
        with self._lock:
            return [self._refresh(p) for p in self._profiles.values() if provider is None or p.provider == provider]

    def clear_cache(self) -> None:
        with self._lock:
            self._profiles.clear()
            self._loaded = False

    def clear_overrides(self) -> None:
        self._load()
        with self._lock:
            for key, profile in list(self._profiles.items()):
                updated = profile.model_copy(deep=True)
                for name in updated.observations:
                    updated.observations[name] = [f for f in updated.observations[name] if f.source != "override"]
                self._profiles[key] = self._refresh(updated)

    def register_override(self, profile: ModelProfile) -> None:
        self.merge(profile, {name: fact.model_copy(update={"source": "override", "method": "user_configuration", "expires_at": None})
                            for name, fact in profile.facts.items()})


GLOBAL_CAPABILITY_REGISTRY = ModelCapabilityRegistry()


def resolve_profile(provider: str, model: str, *, endpoint: str = "", api_version: str = "", transport: str = "",
                    configuration: dict[str, Any] | None = None, credential_revision: str = "",
                    catalog_records: Any = None) -> ModelProfile:
    from mana_agent.config.provider_registry import PROVIDERS, model_id_for_provider
    from mana_agent.config.user_config import get_setting, load_model_cache
    try:
        definition = PROVIDERS.get(provider)
        provider = definition.id
    except KeyError:
        return ModelProfile(provider=provider, model_id=model, endpoint=endpoint, api_version=api_version, transport=transport)
    model = model_id_for_provider(provider, model)
    from mana_agent.config.inference_provider import resolve_inference_connection
    from mana_agent.config.settings import Settings
    connection = resolve_inference_connection(Settings(), provider=provider, require_api_key=False)
    endpoint = endpoint or connection.base_url
    credential_revision = credential_revision or hashlib.sha256(connection.api_key.encode()).hexdigest()
    api_version = api_version or connection.headers.get("anthropic-version", "") or connection.query_params.get("api-version", "")
    mode = str(get_setting("MANA_LLM_API_MODE", "auto") or "auto")
    transport = transport or (definition.inference_protocol if mode == "auto" or definition.inference_protocol == "anthropic_messages" else mode)
    if configuration is None:
        effort = get_setting("MANA_LLM_REASONING_EFFORT", "")
        configuration = {"reasoning_effort": effort} if effort else {}
    if "headers_revision" not in configuration:
        configuration = {"parameters": configuration,
            "headers_revision": hashlib.sha256(json.dumps(connection.headers, sort_keys=True).encode()).hexdigest(),
            "query_revision": hashlib.sha256(json.dumps(connection.query_params, sort_keys=True).encode()).hexdigest()}
    deployment = str(configuration.get("deployment") or configuration.get("parameters", {}).get("deployment_id") or "")
    scope = scope_key(provider=provider, model=model, endpoint=endpoint, api_version=api_version,
        transport=transport, configuration=configuration, credential_revision=credential_revision, deployment=deployment)
    binding = metadata_binding(provider, endpoint, credential_revision, configuration.get("headers_revision", ""), configuration.get("query_revision", ""))
    profile = GLOBAL_CAPABILITY_REGISTRY.get(provider, model, scope) or ModelProfile(provider=provider,
        model_id=model, endpoint=endpoint, api_version=api_version, transport=transport, configuration_fingerprint=scope)
    records = catalog_records
    if records is None:
        cached = load_model_cache(provider, endpoint)
        records = cached.models if cached else []
    for record in records:
        if not isinstance(record, dict) or record.get("id") != model:
            continue
        if catalog_records is None and record.get("_mana_catalog_binding") != binding:
            continue
        facts = metadata_facts(record, source="provider", scope=scope)
        if catalog_records is None and cached is not None:
            from datetime import datetime, timedelta
            try:
                observed = datetime.fromisoformat(cached.created_at)
            except ValueError:
                continue
            if observed.tzinfo is None:
                continue
            facts = {name: fact.model_copy(update={"observed_at": observed, "expires_at": observed + timedelta(days=1)}) for name, fact in facts.items()}
        profile = GLOBAL_CAPABILITY_REGISTRY.merge(profile, facts)
    # Source profiles are reusable metadata, but never reusable endpoint-specific probes/overrides.
    generic = GLOBAL_CAPABILITY_REGISTRY.get(provider, model, binding)
    if generic is not None:
        profile.metadata.update(generic.metadata)
        for name, observations in generic.observations.items():
            for fact in observations:
                if fact.source in {"models.dev", "provider"}:
                    profile = GLOBAL_CAPABILITY_REGISTRY.merge(profile, {name: fact.model_copy(update={"scope": scope})})
    if not profile.is_known:
        from mana_agent.config.model_catalog import normalize_capabilities, maintained_token_limits, ModelCapability
        caps = normalize_capabilities(provider, model)
        if caps:
            mapping = {
                ModelCapability.TOOL_CALLING: "tool_calling",
                ModelCapability.STRUCTURED_OUTPUT: "native_structured_output",
                ModelCapability.TEXT_GENERATION: "output_text",
                ModelCapability.REASONING: "reasoning",
                ModelCapability.IMAGE_INPUT: "input_image",
                ModelCapability.AUDIO_INPUT: "input_audio",
                ModelCapability.VIDEO_INPUT: "input_video",
                ModelCapability.EMBEDDING: "embedding",
                ModelCapability.REALTIME: "realtime",
                ModelCapability.CODE: "code",
            }
            facts = {}
            for cap, name in mapping.items():
                if cap in caps:
                    facts[name] = observation(True, source="provider", scope=scope)
            if ModelCapability.TEXT_GENERATION in caps:
                facts["input_text"] = observation(True, source="provider", scope=scope)
                facts["output_text"] = observation(True, source="provider", scope=scope)
            if ModelCapability.TOOL_CALLING in caps and provider in {"openai", "anthropic"}:
                facts["native_structured_output"] = observation(True, source="provider", scope=scope)
            if provider in {"openai", "azure_openai", "azure"} and caps & {ModelCapability.TEXT_GENERATION, ModelCapability.EMBEDDING, ModelCapability.CODE}:
                if not (ModelCapability.REALTIME in caps and "realtime" in model.lower()) and "realtime" not in model.lower():
                    facts["batch"] = observation(True, source="provider", scope=scope)
                    facts["batch_endpoints"] = observation(["/v1/responses", "/v1/chat/completions", "/v1/embeddings"], source="provider", scope=scope)
                    facts["batch_completion_windows"] = observation(["24h"], source="provider", scope=scope)
            limits = maintained_token_limits(provider, model)
            if limits:
                facts["context_window"] = observation(limits[0], source="provider", scope=scope)
                facts["max_output_tokens"] = observation(limits[1], source="provider", scope=scope)
            if facts:
                profile = GLOBAL_CAPABILITY_REGISTRY.merge(profile, facts)
    overrides = get_setting("MANA_MODEL_CAPABILITY_OVERRIDES", []) or []
    if isinstance(overrides, str):
        try:
            overrides = json.loads(overrides)
        except Exception:
            overrides = []
    if not isinstance(overrides, (list, tuple)):
        overrides = []
    overlay: dict[str, CapabilityFact] = {}
    for item in overrides:
        if not isinstance(item, dict):
            continue
        if item.get("provider") != provider or item.get("model_id") != model or item.get("endpoint", endpoint).rstrip("/") != endpoint.rstrip("/"):
            continue
        if item.get("scope") and item["scope"] != scope:
            continue
        for name, value in item.get("capabilities", {}).items():
            fact = CapabilityFact.model_validate(value) if isinstance(value, dict) else observation(value, source="override", scope=scope, method="user_configuration", confidence=1)
            overlay[name] = fact.model_copy(update={"source": "override", "scope": scope, "expires_at": None})
    # Settings are authoritative on every read. An edited or deleted override
    # must not survive in the registry after its configuration entry is removed.
    if overlay:
        profile = profile.model_copy(deep=True)
        for name, fact in overlay.items():
            observations = profile.observations.setdefault(name, [])
            observations.append(fact)
            profile.facts[name] = _winner(observations, scope)
    return profile
