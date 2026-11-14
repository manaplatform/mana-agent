"""Native capability access for Mana configuration, coding and routing domains."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable
from mana_agent.llm.capabilities import ModelProfile, CapabilityState
from mana_agent.llm.registry import GLOBAL_CAPABILITY_REGISTRY, ModelCapabilityRegistry, resolve_profile
from mana_agent.config.provider_registry import PROVIDERS, CodexTransport, split_qualified_model_id
from mana_agent.evals.recorder import record_current

class ReasoningEffortPolicy(str, Enum):
    DISABLED = "disabled"
    OPTIONAL = "optional"
    REQUIRED = "required"
    REQUIRED_UNCONFIGURABLE = "required_unconfigurable"
    UNKNOWN = "unknown"

def normalize_model_lookup_id(provider: str, model_id: str) -> str:
    return split_qualified_model_id(model_id, default_provider=provider)[1]

def resolve_transport_name(provider: str, transport: str | CodexTransport | None = None) -> str:
    value = transport.value if isinstance(transport, CodexTransport) else transport
    if value is None:
        from mana_agent.config.user_config import get_setting
        configured = str(get_setting("MANA_LLM_API_MODE", "auto") or "auto")
        if PROVIDERS.get(provider).inference_protocol != "anthropic_messages" and configured != "auto":
            value = configured
    return {"direct_responses": "responses", "responses_bridge": "chat_completions"}.get(value, value) or PROVIDERS.get(provider).inference_protocol

def resolve_model_capability(provider: str, model: str, transport: str | CodexTransport | None = None, *, catalog_records=None) -> ModelProfile:
    return resolve_profile(provider, model, transport=resolve_transport_name(provider, transport), catalog_records=catalog_records)

def clear_capability_cache() -> None:
    GLOBAL_CAPABILITY_REGISTRY.clear_cache()

def register_model_capability_override(profile: ModelProfile) -> None:
    GLOBAL_CAPABILITY_REGISTRY.register_override(profile)

def clear_model_capability_overrides() -> None:
    GLOBAL_CAPABILITY_REGISTRY.clear_overrides()

@dataclass(frozen=True, slots=True)
class ModelRequestPolicy:
    """Normalized request compatibility policy for (provider, model, transport)."""

    provider: str
    model: str
    transport: str
    reasoning_policy: str
    reasoning_required: bool
    reasoning_can_disable: bool
    reasoning_effort_configurable: bool
    supports_tool_calls: bool
    supports_structured_output: bool
    supports_repository_write: bool
    supports_server_tools: bool = False
    capability_source: str = "unknown"
    capability_confidence: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "transport": self.transport,
            "reasoning_policy": self.reasoning_policy,
            "reasoning_required": self.reasoning_required,
            "reasoning_can_disable": self.reasoning_can_disable,
            "reasoning_effort_configurable": self.reasoning_effort_configurable,
            "supports_tool_calls": self.supports_tool_calls,
            "supports_structured_output": self.supports_structured_output,
            "supports_repository_write": self.supports_repository_write,
            "supports_server_tools": self.supports_server_tools,
            "capability_source": self.capability_source,
            "capability_confidence": self.capability_confidence,
        }


def resolve_model_request_policy(
    provider: str,
    model: str,
    transport: str | CodexTransport | None = None,
    *,
    catalog_records: Iterable[dict[str, Any] | Any] | None = None,
) -> ModelRequestPolicy:
    """Resolve request compatibility policy for (provider, model, transport)."""
    desc = resolve_model_capability(
        provider, model, transport, catalog_records=catalog_records
    )
    policy = ModelRequestPolicy(
        provider=desc.provider,
        model=desc.model,
        transport=desc.transport,
        reasoning_policy=desc.reasoning_policy,
        reasoning_required=desc.reasoning_required,
        reasoning_can_disable=desc.reasoning_can_disable,
        reasoning_effort_configurable=desc.reasoning_effort_configurable,
        supports_tool_calls=desc.supports_tool_calls,
        supports_structured_output=desc.supports_structured_output,
        supports_repository_write=desc.supports_repository_write,
        supports_server_tools=desc.supports_server_tools,
        capability_source=desc.capability_source,
        capability_confidence=desc.capability_confidence,
        metadata=dict(desc.metadata),
    )
    record_current(
        "model.reasoning_policy.resolved",
        {
            "provider": policy.provider,
            "model": policy.model,
            "transport": policy.transport,
            "reasoning_required": policy.reasoning_required,
            "reasoning_configurable": policy.reasoning_effort_configurable,
            "requested_reasoning": "",
            "effective_reasoning": policy.reasoning_policy,
            "metadata_source": policy.capability_source,
        },
    )
    return policy


def normalize_reasoning_request_overrides(provider: str, model: str, transport=None, overrides=None) -> dict[str, Any]:
    normalized = dict(overrides or {})
    profile = resolve_model_capability(provider, model, transport)
    effort = normalized.get("reasoning_effort")
    if isinstance(normalized.get("reasoning"), dict):
        effort = normalized["reasoning"].get("effort", effort)
    if effort is not None:
        profile.require("reasoning", "reasoning_effort")
        if not profile.supports_value("reasoning_effort", effort):
            raise ValueError("Reasoning effort is not supported by the selected model profile")
    return normalized
