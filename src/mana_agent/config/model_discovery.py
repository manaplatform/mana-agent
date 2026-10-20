"""Unified model discovery, capability resolution, and capability registry for Mana-Agent.

Provides evidence-backed model profiles, tool support resolution, and Batch API compatibility
by inspecting official OpenAI endpoints (GET /v1/models, GET /v1/models/{model_id}) and
consulting a maintained, versioned capability registry.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from mana_agent.config.model_catalog import (
    ModelCapability,
    maintained_token_limits,
    normalize_capabilities,
)
from mana_agent._version import get_version

logger = logging.getLogger(__name__)

# Registry version
REGISTRY_VERSION = "2026.10.1"


@dataclass
class ModelCapabilityProfile:
    """Evidence-backed capability profile for a model candidate."""

    # 1. Model specifications
    model_id: str
    provider: str = "openai"
    context_window: int | None = None
    max_output_tokens: int | None = None
    input_modalities: list[str] = field(default_factory=lambda: ["text"])
    output_modalities: list[str] = field(default_factory=lambda: ["text"])
    reasoning_effort_levels: list[str] = field(default_factory=list)
    streaming_supported: bool = True
    structured_output_supported: bool = True
    function_calling_supported: bool = True
    parallel_function_calling_supported: bool = True
    prompt_caching_supported: bool = True
    supported_api_endpoints: list[str] = field(
        default_factory=lambda: ["/v1/responses", "/v1/chat/completions"]
    )

    # 2. Tool support
    web_search: bool = True
    file_search: bool = True
    code_interpreter: bool = True
    shell_and_local_shell: bool = True
    apply_patch: bool = True
    computer_use: bool = False
    image_generation: bool = False
    function_calling: bool = True
    mcp: bool = True
    tool_search: bool = True
    programmatic_tool_calling: bool = True
    skills: bool = True

    # 3. Batch support
    supports_batch: bool | None = True  # True (yes), False (no), None (unknown)
    supported_batch_endpoints: list[str] = field(
        default_factory=lambda: ["/v1/responses", "/v1/chat/completions", "/v1/embeddings"]
    )
    supported_input_modalities_batch: list[str] = field(default_factory=lambda: ["text"])
    batch_restrictions: list[str] = field(
        default_factory=lambda: [
            "Batch API does not execute arbitrary local shell tools during asynchronous processing.",
            "Batch requests execute independent model turns server-side; dependent tool loops resume in Mana-Agent.",
        ]
    )
    batch_pricing: dict[str, Any] = field(
        default_factory=lambda: {
            "discount_factor": 0.5,
            "discount_percent": 50,
            "cost_notes": "50% token cost reduction compared to synchronous API.",
        }
    )
    batch_rate_limits: dict[str, Any] = field(
        default_factory=lambda: {
            "max_requests_per_batch": 50000,
            "max_file_size_mb": 200,
        }
    )
    supported_completion_windows: list[str] = field(default_factory=lambda: ["24h"])

    # 4. Verification provenance
    verification_source: str = "maintained_registry"  # "openai_api", "catalog", "maintained_registry", "model_card", "unknown"
    verified_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    confidence: str = "high"  # "high", "medium", "low", "unknown"
    raw_metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def supports_batch_display(self) -> str:
        """Formatted batch support indicator: yes / no / unknown."""
        if self.supports_batch is True:
            return "yes"
        if self.supports_batch is False:
            return "no"
        return "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelCapabilityRegistry:
    """Thread-safe, versioned capability registry maintaining authoritative profiles."""

    version: str = REGISTRY_VERSION

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._profiles: dict[tuple[str, str], ModelCapabilityProfile] = {}
        self._populate_maintained_profiles()

    def _normalize_key(self, provider: str, model_id: str) -> tuple[str, str]:
        p = str(provider or "openai").strip().lower()
        m = str(model_id or "").strip()
        if p == "openai" and m.startswith("openai/"):
            m = m[len("openai/"):]
        return p, m

    def _populate_maintained_profiles(self) -> None:
        """Populate initial registry with tested model profiles."""
        now_iso = datetime.now(timezone.utc).isoformat()

        # Luna and Astra flagship models
        for m_id in ("gpt-6-luna", "luna"):
            self._profiles[("openai", m_id)] = ModelCapabilityProfile(
                model_id=m_id,
                provider="openai",
                context_window=1_050_000,
                max_output_tokens=128_000,
                input_modalities=["text", "image"],
                reasoning_effort_levels=["low", "medium", "high"],
                supports_batch=True,
                supported_batch_endpoints=["/v1/responses", "/v1/chat/completions"],
                verification_source="maintained_registry",
                verified_at=now_iso,
                confidence="high",
            )

        for m_id in ("gpt-6", "gpt-6-astra", "astra"):
            self._profiles[("openai", m_id)] = ModelCapabilityProfile(
                model_id=m_id,
                provider="openai",
                context_window=1_050_000,
                max_output_tokens=128_000,
                input_modalities=["text", "image"],
                reasoning_effort_levels=["low", "medium", "high"],
                supports_batch=True,
                supported_batch_endpoints=["/v1/responses", "/v1/chat/completions"],
                verification_source="maintained_registry",
                verified_at=now_iso,
                confidence="high",
            )

        # GPT-5.6 family
        for m_id in ("gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"):
            self._profiles[("openai", m_id)] = ModelCapabilityProfile(
                model_id=m_id,
                provider="openai",
                context_window=400_000,
                max_output_tokens=128_000,
                input_modalities=["text", "image"],
                reasoning_effort_levels=["low", "medium", "high"],
                supports_batch=True,
                supported_batch_endpoints=["/v1/responses", "/v1/chat/completions"],
                verification_source="maintained_registry",
                verified_at=now_iso,
                confidence="high",
            )

        # GPT-5 family
        for m_id in ("gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.5", "gpt-5-mini", "gpt-5-nano"):
            self._profiles[("openai", m_id)] = ModelCapabilityProfile(
                model_id=m_id,
                provider="openai",
                context_window=400_000,
                max_output_tokens=128_000,
                input_modalities=["text", "image"],
                reasoning_effort_levels=["low", "medium", "high"] if "mini" not in m_id else [],
                supports_batch=True,
                supported_batch_endpoints=["/v1/responses", "/v1/chat/completions"],
                verification_source="maintained_registry",
                verified_at=now_iso,
                confidence="high",
            )

        # GPT-4o family
        for m_id in ("gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"):
            self._profiles[("openai", m_id)] = ModelCapabilityProfile(
                model_id=m_id,
                provider="openai",
                context_window=128_000,
                max_output_tokens=16_384,
                input_modalities=["text", "image"] if "3.5" not in m_id else ["text"],
                supports_batch=True,
                supported_batch_endpoints=["/v1/responses", "/v1/chat/completions"],
                verification_source="maintained_registry",
                verified_at=now_iso,
                confidence="high",
            )

        # GPT-4.1 family
        for m_id in ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"):
            self._profiles[("openai", m_id)] = ModelCapabilityProfile(
                model_id=m_id,
                provider="openai",
                context_window=1_047_576,
                max_output_tokens=32_768,
                input_modalities=["text", "image"],
                supports_batch=True,
                supported_batch_endpoints=["/v1/responses", "/v1/chat/completions"],
                verification_source="maintained_registry",
                verified_at=now_iso,
                confidence="high",
            )

        # Reasoning models (o1, o3, o4)
        for m_id in ("o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"):
            self._profiles[("openai", m_id)] = ModelCapabilityProfile(
                model_id=m_id,
                provider="openai",
                context_window=200_000,
                max_output_tokens=100_000,
                input_modalities=["text", "image"] if "mini" not in m_id else ["text"],
                reasoning_effort_levels=["low", "medium", "high"],
                supports_batch=True,
                supported_batch_endpoints=["/v1/responses", "/v1/chat/completions"],
                verification_source="maintained_registry",
                verified_at=now_iso,
                confidence="high",
            )

        # Embeddings
        for m_id in ("text-embedding-3-small", "text-embedding-3-large"):
            self._profiles[("openai", m_id)] = ModelCapabilityProfile(
                model_id=m_id,
                provider="openai",
                context_window=8_191,
                max_output_tokens=0,
                input_modalities=["text"],
                output_modalities=["embedding"],
                supports_batch=True,
                supported_batch_endpoints=["/v1/embeddings"],
                web_search=False,
                file_search=False,
                code_interpreter=False,
                shell_and_local_shell=False,
                apply_patch=False,
                verification_source="maintained_registry",
                verified_at=now_iso,
                confidence="high",
            )

        # Explicitly unsupported batch models
        for m_id in ("gpt-4o-realtime-preview", "gpt-live", "gpt-live-1"):
            self._profiles[("openai", m_id)] = ModelCapabilityProfile(
                model_id=m_id,
                provider="openai",
                context_window=128_000,
                max_output_tokens=4_096,
                input_modalities=["text", "audio"],
                output_modalities=["text", "audio"],
                supports_batch=False,
                supported_batch_endpoints=[],
                batch_restrictions=["Realtime audio streaming models are not supported on OpenAI Batch API."],
                verification_source="maintained_registry",
                verified_at=now_iso,
                confidence="high",
            )

        for m_id in ("dall-e-2", "dall-e-3"):
            self._profiles[("openai", m_id)] = ModelCapabilityProfile(
                model_id=m_id,
                provider="openai",
                input_modalities=["text"],
                output_modalities=["image"],
                image_generation=True,
                supports_batch=False,
                supported_batch_endpoints=[],
                verification_source="maintained_registry",
                verified_at=now_iso,
                confidence="high",
            )

        for m_id in ("whisper-1", "tts-1", "tts-1-hd"):
            self._profiles[("openai", m_id)] = ModelCapabilityProfile(
                model_id=m_id,
                provider="openai",
                supports_batch=False,
                supported_batch_endpoints=[],
                verification_source="maintained_registry",
                verified_at=now_iso,
                confidence="high",
            )

    def register(self, profile: ModelCapabilityProfile) -> None:
        key = self._normalize_key(profile.provider, profile.model_id)
        with self._lock:
            self._profiles[key] = profile

    def get(self, provider: str, model_id: str) -> ModelCapabilityProfile | None:
        key = self._normalize_key(provider, model_id)
        with self._lock:
            direct = self._profiles.get(key)
            if direct is not None:
                return direct
            # Prefix search for dated/suffixed IDs (e.g. gpt-4o-2024-08-06)
            p_val, m_val = key
            for (p_cand, m_cand), prof in self._profiles.items():
                if p_cand == p_val and m_val.startswith(m_cand):
                    return prof
        return None

    def supports_batch(self, provider: str, model_id: str) -> bool:
        prof = self.get(provider, model_id)
        if prof is not None and prof.supports_batch is not None:
            return prof.supports_batch
        # Fallback to catalog normalization check
        from mana_agent.batch.router import is_model_supported_for_batch

        return is_model_supported_for_batch(model_id, provider=provider)

    def list_profiles(self, provider: str | None = None) -> list[ModelCapabilityProfile]:
        with self._lock:
            if provider is None:
                return list(self._profiles.values())
            p_clean = str(provider).strip().lower()
            return [prof for (p, _), prof in self._profiles.items() if p == p_clean]


# Global shared registry
GLOBAL_CAPABILITY_REGISTRY = ModelCapabilityRegistry()


class ModelMetadataResolver:
    """Resolves specifications, tool support, and Batch API capabilities from API data and registry."""

    def __init__(self, registry: ModelCapabilityRegistry | None = None) -> None:
        self.registry = registry or GLOBAL_CAPABILITY_REGISTRY

    def resolve(
        self,
        model_id: str,
        provider: str = "openai",
        raw_metadata: dict[str, Any] | None = None,
    ) -> ModelCapabilityProfile:
        model_id = str(model_id or "").strip()
        provider = str(provider or "openai").strip().lower()
        raw = dict(raw_metadata or {})

        # Check existing maintained profile
        base_profile = self.registry.get(provider, model_id)
        if base_profile is not None:
            # Augment with any live metadata fields
            if raw:
                updated_raw = {**base_profile.raw_metadata, **raw}
                return ModelCapabilityProfile(
                    model_id=base_profile.model_id,
                    provider=base_profile.provider,
                    context_window=raw.get("context_window") or base_profile.context_window,
                    max_output_tokens=raw.get("max_output_tokens") or base_profile.max_output_tokens,
                    input_modalities=base_profile.input_modalities,
                    output_modalities=base_profile.output_modalities,
                    reasoning_effort_levels=base_profile.reasoning_effort_levels,
                    streaming_supported=base_profile.streaming_supported,
                    structured_output_supported=base_profile.structured_output_supported,
                    function_calling_supported=base_profile.function_calling_supported,
                    parallel_function_calling_supported=base_profile.parallel_function_calling_supported,
                    prompt_caching_supported=base_profile.prompt_caching_supported,
                    supported_api_endpoints=base_profile.supported_api_endpoints,
                    web_search=base_profile.web_search,
                    file_search=base_profile.file_search,
                    code_interpreter=base_profile.code_interpreter,
                    shell_and_local_shell=base_profile.shell_and_local_shell,
                    apply_patch=base_patch if (base_patch := getattr(base_profile, "apply_patch", None)) is not None else True,
                    computer_use=base_profile.computer_use,
                    image_generation=base_profile.image_generation,
                    function_calling=base_profile.function_calling,
                    mcp=base_profile.mcp,
                    tool_search=base_profile.tool_search,
                    programmatic_tool_calling=base_profile.programmatic_tool_calling,
                    skills=base_profile.skills,
                    supports_batch=base_profile.supports_batch,
                    supported_batch_endpoints=base_profile.supported_batch_endpoints,
                    supported_input_modalities_batch=base_profile.supported_input_modalities_batch,
                    batch_restrictions=base_profile.batch_restrictions,
                    batch_pricing=base_profile.batch_pricing,
                    batch_rate_limits=base_profile.batch_rate_limits,
                    supported_completion_windows=base_profile.supported_completion_windows,
                    verification_source="openai_api" if raw.get("created") else base_profile.verification_source,
                    verified_at=datetime.now(timezone.utc).isoformat(),
                    confidence=base_profile.confidence,
                    raw_metadata=updated_raw,
                )
            return base_profile

        # Dynamic resolution for discovered or custom models
        lowered = model_id.lower()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Token limits
        limits = maintained_token_limits(provider, model_id)
        ctx_win = limits[0] if limits else raw.get("context_window", 128_000)
        max_out = limits[1] if limits else raw.get("max_output_tokens", 16_384)

        # Batch compatibility detection
        supports_batch: bool | None = None
        from mana_agent.batch.router import is_model_supported_for_batch, parse_batch_support_from_metadata

        meta_check = parse_batch_support_from_metadata(raw)
        if meta_check is not None:
            supports_batch = meta_check
        else:
            supports_batch = is_model_supported_for_batch(model_id, metadata=raw, provider=provider)

        # Tool capabilities
        caps = normalize_capabilities(provider, model_id)
        has_tools = ModelCapability.TOOL_CALLING in caps or ("gpt-" in lowered or "luna" in lowered or "o1" in lowered or "o3" in lowered or "o4" in lowered)
        has_reasoning = ModelCapability.REASONING in caps or any(k in lowered for k in ("o1", "o3", "o4", "reasoning", "luna"))

        profile = ModelCapabilityProfile(
            model_id=model_id,
            provider=provider,
            context_window=ctx_win,
            max_output_tokens=max_out,
            input_modalities=["text", "image"] if ModelCapability.IMAGE_INPUT in caps else ["text"],
            reasoning_effort_levels=["low", "medium", "high"] if has_reasoning else [],
            streaming_supported=True,
            structured_output_supported=ModelCapability.STRUCTURED_OUTPUT in caps or has_tools,
            function_calling_supported=has_tools,
            parallel_function_calling_supported=has_tools,
            prompt_caching_supported=True,
            supported_api_endpoints=["/v1/responses", "/v1/chat/completions"],
            web_search=True,
            file_search=True,
            code_interpreter=True,
            shell_and_local_shell=True,
            apply_patch=True,
            computer_use=False,
            image_generation=ModelCapability.IMAGE_GENERATION in caps,
            function_calling=has_tools,
            mcp=True,
            tool_search=True,
            programmatic_tool_calling=has_tools,
            skills=True,
            supports_batch=supports_batch,
            supported_batch_endpoints=["/v1/responses", "/v1/chat/completions"] if supports_batch else [],
            verification_source="openai_api" if raw.get("created") else "catalog",
            verified_at=now_iso,
            confidence="medium" if supports_batch is not None else "low",
            raw_metadata=raw,
        )
        return profile


class ModelDiscoveryService:
    """Discovers models via official OpenAI endpoints (GET /v1/models, GET /v1/models/{model_id})."""

    def __init__(
        self,
        resolver: ModelMetadataResolver | None = None,
        registry: ModelCapabilityRegistry | None = None,
    ) -> None:
        self.registry = registry or GLOBAL_CAPABILITY_REGISTRY
        self.resolver = resolver or ModelMetadataResolver(self.registry)

    def discover_models(
        self,
        *,
        provider: str = "openai",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        timeout_seconds: float = 15.0,
    ) -> list[ModelCapabilityProfile]:
        """Fetch available models from GET /v1/models and resolve capability profiles."""
        p_clean = str(provider or "openai").strip().lower()
        if not api_key:
            try:
                from mana_agent.config.inference_provider import resolve_inference_connection
                from mana_agent.config.settings import Settings

                settings = Settings()
                if p_clean == "openai" and getattr(settings, "openai_api_key", None):
                    api_key = str(settings.openai_api_key).strip()
                if not api_key:
                    conn = resolve_inference_connection(settings, provider=p_clean, require_api_key=False)
                    api_key = conn.api_key or ""
                    if not base_url or base_url == "https://api.openai.com/v1":
                        base_url = conn.base_url or base_url
            except Exception:
                pass

        if not api_key:
            # Offline or unconfigured: return maintained profiles
            return self.registry.list_profiles(provider=p_clean)

        url = base_url.rstrip("/") + "/models"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": f"Mana-Agent/{get_version()}",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("Failed to discover models from %s: %s. Using maintained registry.", url, exc)
            return self.registry.list_profiles(provider=p_clean)

        raw_list = data.get("data", [])
        profiles: list[ModelCapabilityProfile] = []

        for item in raw_list:
            if not isinstance(item, dict):
                continue
            m_id = str(item.get("id") or "").strip()
            if not m_id:
                continue

            profile = self.resolver.resolve(
                model_id=m_id,
                provider=p_clean,
                raw_metadata=item,
            )
            self.registry.register(profile)
            profiles.append(profile)

        return profiles

    def get_model(
        self,
        model_id: str,
        *,
        provider: str = "openai",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        timeout_seconds: float = 10.0,
    ) -> ModelCapabilityProfile | None:
        """Fetch model metadata from GET /v1/models/{model_id} and resolve profile."""
        clean_id = str(model_id or "").strip()
        if not clean_id:
            return None
        p_clean = str(provider or "openai").strip().lower()

        if api_key:
            url = f"{base_url.rstrip('/')}/models/{urllib.parse.quote(clean_id, safe='')}"
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": f"Mana-Agent/{get_version()}",
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                    raw_data = json.loads(resp.read().decode("utf-8"))
                    profile = self.resolver.resolve(
                        model_id=clean_id,
                        provider=p_clean,
                        raw_metadata=raw_data,
                    )
                    self.registry.register(profile)
                    return profile
            except Exception as exc:
                logger.debug("GET /v1/models/%s failed (%s); checking capability registry", clean_id, exc)

        profile = self.resolver.resolve(clean_id, provider=p_clean)
        self.registry.register(profile)
        return profile


__all__ = [
    "GLOBAL_CAPABILITY_REGISTRY",
    "ModelCapabilityProfile",
    "ModelCapabilityRegistry",
    "ModelDiscoveryService",
    "ModelMetadataResolver",
    "REGISTRY_VERSION",
]
