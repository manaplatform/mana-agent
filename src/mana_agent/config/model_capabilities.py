from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, Iterable

from mana_agent.config.provider_registry import (
    CodexTransport,
    PROVIDERS,
)
from mana_agent.evals.recorder import record_current


@dataclass(frozen=True, slots=True)
class ModelCapabilityDescriptor:
    """Authoritative descriptor of transport-level capabilities for a model candidate."""

    provider: str
    model: str
    transport: str
    supports_tool_calls: bool = False
    supports_repository_read: bool = False
    supports_repository_write: bool = False
    supports_shell: bool = False
    supports_structured_output: bool = False
    supports_streaming: bool = False
    supports_parallel_tools: bool = False
    capability_confidence: str = "unknown"  # "high", "medium", "low", "unknown"
    capability_source: str = "unknown"  # "catalog", "provider_metadata", "maintained", "override", "probing", "unknown"
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def is_known(self) -> bool:
        return self.capability_confidence != "unknown" and self.capability_source != "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "transport": self.transport,
            "supports_tool_calls": self.supports_tool_calls,
            "supports_repository_read": self.supports_repository_read,
            "supports_repository_write": self.supports_repository_write,
            "supports_shell": self.supports_shell,
            "supports_structured_output": self.supports_structured_output,
            "supports_streaming": self.supports_streaming,
            "supports_parallel_tools": self.supports_parallel_tools,
            "capability_confidence": self.capability_confidence,
            "capability_source": self.capability_source,
        }


def normalize_model_lookup_id(provider: str, model_id: str) -> str:
    """Normalize model ID for capability lookup without stripping org prefixes.

    OpenRouter model IDs are multi-segment (e.g. deepseek/deepseek-v4-flash,
    anthropic/claude-3.7-sonnet). They must retain their org namespace and not
    be stripped to bare names or treated as OpenAI native models.
    """
    provider_id = str(provider or "").strip().lower()
    raw = str(model_id or "").strip()
    if not raw:
        return ""
    if provider_id == "openrouter":
        if raw.lower().startswith("openrouter/"):
            raw = raw[len("openrouter/") :].strip()
        return raw
    if raw.lower().startswith(f"{provider_id}/"):
        raw = raw[len(f"{provider_id}/") :].strip()
    return raw


def resolve_transport_name(provider: str, transport: str | CodexTransport | None = None) -> str:
    """Resolve transport name for a provider."""
    if transport is not None:
        if isinstance(transport, CodexTransport):
            return transport.value
        val = str(transport).strip().lower()
        if val:
            return val
    provider_id = str(provider or "").strip().lower()
    try:
        defn = PROVIDERS.get(provider_id)
        return defn.codex_transport.value
    except KeyError:
        return CodexTransport.DIRECT_RESPONSES.value


# Explicit maintained descriptors for tested provider/model/transport configurations
_MAINTAINED_DESCRIPTORS: dict[tuple[str, str, str], ModelCapabilityDescriptor] = {
    # OpenAI native models via direct_responses
    ("openai", "gpt-4.1", "direct_responses"): ModelCapabilityDescriptor(
        provider="openai",
        model="gpt-4.1",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    ("openai", "gpt-4.1-mini", "direct_responses"): ModelCapabilityDescriptor(
        provider="openai",
        model="gpt-4.1-mini",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    ("openai", "gpt-4.1-nano", "direct_responses"): ModelCapabilityDescriptor(
        provider="openai",
        model="gpt-4.1-nano",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    ("openai", "gpt-4o", "direct_responses"): ModelCapabilityDescriptor(
        provider="openai",
        model="gpt-4o",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    ("openai", "gpt-4o-mini", "direct_responses"): ModelCapabilityDescriptor(
        provider="openai",
        model="gpt-4o-mini",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    ("openai", "gpt-5", "direct_responses"): ModelCapabilityDescriptor(
        provider="openai",
        model="gpt-5",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    ("openai", "gpt-5-mini", "direct_responses"): ModelCapabilityDescriptor(
        provider="openai",
        model="gpt-5-mini",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    ("openai", "o3", "direct_responses"): ModelCapabilityDescriptor(
        provider="openai",
        model="o3",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    ("openai", "o3-mini", "direct_responses"): ModelCapabilityDescriptor(
        provider="openai",
        model="o3-mini",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    ("openai", "o4-mini", "direct_responses"): ModelCapabilityDescriptor(
        provider="openai",
        model="o4-mini",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    # NVIDIA NIM hosted models via responses_bridge
    ("nvidia", "deepseek-ai/deepseek-v4-flash", "responses_bridge"): ModelCapabilityDescriptor(
        provider="nvidia",
        model="deepseek-ai/deepseek-v4-flash",
        transport="responses_bridge",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    ("nvidia", "deepseek-ai/deepseek-v4-pro", "responses_bridge"): ModelCapabilityDescriptor(
        provider="nvidia",
        model="deepseek-ai/deepseek-v4-pro",
        transport="responses_bridge",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    ("nvidia", "nvidia/nemotron-3-nano-30b-a3b", "responses_bridge"): ModelCapabilityDescriptor(
        provider="nvidia",
        model="nvidia/nemotron-3-nano-30b-a3b",
        transport="responses_bridge",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    # OpenRouter maintained entries
    ("openrouter", "anthropic/claude-3.7-sonnet", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="anthropic/claude-3.7-sonnet",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    ("openrouter", "anthropic/claude-3.5-sonnet", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    ("openrouter", "openai/gpt-4.1", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="openai/gpt-4.1",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    ("openrouter", "openai/gpt-4.1-mini", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="openai/gpt-4.1-mini",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    ("openrouter", "openai/gpt-4o", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="openai/gpt-4o",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
    ("openrouter", "openai/gpt-4o-mini", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="openai/gpt-4o-mini",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        capability_confidence="high",
        capability_source="maintained",
    ),
}


class ModelCapabilityRegistry:
    """Thread-safe registry and cache for model capabilities."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str, str], ModelCapabilityDescriptor] = {}
        self._overrides: dict[tuple[str, str, str], ModelCapabilityDescriptor] = {}

    def clear_cache(self) -> None:
        """Invalidate the capability cache."""
        with self._lock:
            self._cache.clear()

    def register_override(self, descriptor: ModelCapabilityDescriptor) -> None:
        """Register an explicit test or configuration override."""
        key = (
            descriptor.provider.strip().lower(),
            normalize_model_lookup_id(descriptor.provider, descriptor.model),
            descriptor.transport.strip().lower(),
        )
        with self._lock:
            self._overrides[key] = descriptor
            self._cache[key] = descriptor

    def clear_overrides(self) -> None:
        """Clear registered overrides."""
        with self._lock:
            self._overrides.clear()
            self._cache.clear()

    def resolve(
        self,
        provider: str,
        model: str,
        transport: str | CodexTransport | None = None,
        *,
        catalog_records: Iterable[dict[str, Any] | Any] | None = None,
    ) -> ModelCapabilityDescriptor:
        """Resolve authoritative capability descriptor for (provider, model, transport)."""
        provider_id = str(provider or "").strip().lower()
        normalized_model = normalize_model_lookup_id(provider_id, model)
        transport_id = resolve_transport_name(provider_id, transport)
        cache_key = (provider_id, normalized_model, transport_id)

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        descriptor = self._resolve_uncached(
            provider_id,
            normalized_model,
            transport_id,
            catalog_records=catalog_records,
        )

        with self._lock:
            self._cache[cache_key] = descriptor

        # Record diagnostic events
        if descriptor.is_known:
            record_current(
                "model.capability.resolved",
                {
                    "provider": descriptor.provider,
                    "model": descriptor.model,
                    "transport": descriptor.transport,
                    "required_capabilities": [],
                    "resolved_capabilities": descriptor.to_dict(),
                    "capability_source": descriptor.capability_source,
                    "capability_confidence": descriptor.capability_confidence,
                },
            )
        else:
            record_current(
                "model.capability.unknown",
                {
                    "provider": descriptor.provider,
                    "model": descriptor.model,
                    "transport": descriptor.transport,
                    "required_capabilities": [],
                    "resolved_capabilities": descriptor.to_dict(),
                    "capability_source": "unknown",
                    "rejection_reason": "model capabilities are unknown",
                },
            )

        return descriptor

    def _resolve_uncached(
        self,
        provider: str,
        model: str,
        transport: str,
        *,
        catalog_records: Iterable[dict[str, Any] | Any] | None = None,
    ) -> ModelCapabilityDescriptor:
        key = (provider, model, transport)
        # 1. Overrides
        with self._lock:
            if key in self._overrides:
                return self._overrides[key]

        # 2. Maintained entries (exact match or family prefix match)
        if key in _MAINTAINED_DESCRIPTORS:
            return _MAINTAINED_DESCRIPTORS[key]

        # Check family prefix for dated/suffixed IDs in maintained descriptors
        for (m_prov, m_model, m_trans), m_desc in _MAINTAINED_DESCRIPTORS.items():
            if m_prov == provider and m_trans == transport:
                if model.startswith(m_model) and (
                    len(model) == len(m_model) or model[len(m_model)] in "-._/:"
                ):
                    return ModelCapabilityDescriptor(
                        provider=provider,
                        model=model,
                        transport=transport,
                        supports_tool_calls=m_desc.supports_tool_calls,
                        supports_repository_read=m_desc.supports_repository_read,
                        supports_repository_write=m_desc.supports_repository_write,
                        supports_shell=m_desc.supports_shell,
                        supports_structured_output=m_desc.supports_structured_output,
                        supports_streaming=m_desc.supports_streaming,
                        supports_parallel_tools=m_desc.supports_parallel_tools,
                        capability_confidence=m_desc.capability_confidence,
                        capability_source="maintained",
                    )

        # 3. Catalog records (supplied or from cached catalog)
        records = catalog_records
        if records is None:
            records = self._load_cached_catalog_records(provider)

        if records:
            for record in records:
                rec_id = ""
                rec_dict: dict[str, Any] = {}
                if isinstance(record, dict):
                    rec_id = str(record.get("id") or "").strip()
                    rec_dict = record
                elif hasattr(record, "id"):
                    rec_id = str(getattr(record, "id") or "").strip()
                    rec_dict = getattr(record, "metadata", {}) or {}
                if not rec_id:
                    continue
                normalized_rec_id = normalize_model_lookup_id(provider, rec_id)
                if normalized_rec_id == model or rec_id == model:
                    return self._descriptor_from_catalog_record(
                        provider, model, transport, rec_dict
                    )

        # 4. Unknown fail-closed descriptor
        return ModelCapabilityDescriptor(
            provider=provider,
            model=model,
            transport=transport,
            supports_tool_calls=False,
            supports_repository_read=False,
            supports_repository_write=False,
            supports_shell=False,
            supports_structured_output=False,
            supports_streaming=False,
            supports_parallel_tools=False,
            capability_confidence="unknown",
            capability_source="unknown",
        )

    def _descriptor_from_catalog_record(
        self,
        provider: str,
        model: str,
        transport: str,
        record: dict[str, Any],
    ) -> ModelCapabilityDescriptor:
        supported_params = record.get("supported_parameters")
        if not isinstance(supported_params, list):
            supported_params = []
        supported_lower = {str(p).lower() for p in supported_params}

        # Check explicit tool calling capability from provider metadata
        supports_tools = bool(
            supported_lower & {"tools", "tool_choice", "parallel_tool_calls"}
        )
        supports_parallel = "parallel_tool_calls" in supported_lower
        supports_structured = bool(
            supported_lower & {"response_format", "structured_outputs"}
            or any("structured" in str(p).lower() for p in supported_lower)
        )

        caps = record.get("capabilities")
        if isinstance(caps, list):
            caps_lower = {str(c).lower() for c in caps}
            if "tool_calling" in caps_lower or "tools" in caps_lower:
                supports_tools = True
            if "structured_output" in caps_lower:
                supports_structured = True

        architecture = record.get("architecture") if isinstance(record.get("architecture"), dict) else {}
        out_mods = architecture.get("output_modalities") if isinstance(architecture, dict) else []
        if isinstance(out_mods, list) and any(m in {"image", "video", "audio"} for m in out_mods):
            # Non-text model
            return ModelCapabilityDescriptor(
                provider=provider,
                model=model,
                transport=transport,
                supports_tool_calls=False,
                supports_repository_read=False,
                supports_repository_write=False,
                supports_shell=False,
                supports_structured_output=False,
                supports_streaming=False,
                supports_parallel_tools=False,
                capability_confidence="high",
                capability_source="catalog",
                metadata=record,
            )

        # If tools or parameters were explicitly reported in the catalog:
        if supported_params or record.get("capabilities"):
            return ModelCapabilityDescriptor(
                provider=provider,
                model=model,
                transport=transport,
                supports_tool_calls=supports_tools,
                supports_repository_read=supports_tools,
                supports_repository_write=supports_tools,
                supports_shell=supports_tools,
                supports_structured_output=supports_structured,
                supports_streaming=True,
                supports_parallel_tools=supports_parallel,
                capability_confidence="high",
                capability_source="catalog",
                metadata=record,
            )

        # Basic catalog entry with no capability metadata -> unknown
        return ModelCapabilityDescriptor(
            provider=provider,
            model=model,
            transport=transport,
            supports_tool_calls=False,
            supports_repository_read=False,
            supports_repository_write=False,
            supports_shell=False,
            supports_structured_output=False,
            supports_streaming=False,
            supports_parallel_tools=False,
            capability_confidence="unknown",
            capability_source="unknown",
            metadata=record,
        )

    def _load_cached_catalog_records(self, provider: str) -> list[dict[str, Any]]:
        try:
            from mana_agent.config.inference_provider import resolve_inference_connection
            from mana_agent.config.settings import Settings
            from mana_agent.config.user_config import load_model_cache

            connection = resolve_inference_connection(Settings(), provider=provider, require_api_key=False)
            cached = load_model_cache(connection.provider, connection.base_url)
            if cached is not None and isinstance(cached.models, list):
                result: list[dict[str, Any]] = []
                for item in cached.models:
                    if isinstance(item, dict):
                        result.append(item)
                    elif isinstance(item, str):
                        result.append({"id": item})
                return result
        except Exception:
            pass
        return []


_CAPABILITY_REGISTRY = ModelCapabilityRegistry()


def resolve_model_capability(
    provider: str,
    model: str,
    transport: str | CodexTransport | None = None,
    *,
    catalog_records: Iterable[dict[str, Any] | Any] | None = None,
) -> ModelCapabilityDescriptor:
    """Public helper to resolve model capability."""
    return _CAPABILITY_REGISTRY.resolve(
        provider, model, transport, catalog_records=catalog_records
    )


def clear_capability_cache() -> None:
    """Clear cached capabilities."""
    _CAPABILITY_REGISTRY.clear_cache()


def register_model_capability_override(descriptor: ModelCapabilityDescriptor) -> None:
    """Register an explicit capability override."""
    _CAPABILITY_REGISTRY.register_override(descriptor)


def clear_model_capability_overrides() -> None:
    """Clear explicit capability overrides."""
    _CAPABILITY_REGISTRY.clear_overrides()


__all__ = [
    "ModelCapabilityDescriptor",
    "ModelCapabilityRegistry",
    "clear_capability_cache",
    "clear_model_capability_overrides",
    "normalize_model_lookup_id",
    "register_model_capability_override",
    "resolve_model_capability",
    "resolve_transport_name",
]
