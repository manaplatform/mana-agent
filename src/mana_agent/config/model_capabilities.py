from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import threading
from typing import Any, Iterable

from mana_agent.config.provider_registry import (
    CodexTransport,
    PROVIDERS,
)
from mana_agent.evals.recorder import record_current


class ReasoningEffortPolicy(str, Enum):
    """Reasoning configuration policy for a model candidate."""

    DISABLED = "disabled"
    OPTIONAL = "optional"
    REQUIRED = "required"
    REQUIRED_UNCONFIGURABLE = "required_unconfigurable"


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
    supports_server_tools: bool = False
    capability_confidence: str = "unknown"  # "high", "medium", "low", "unknown"
    capability_source: str = "unknown"  # "catalog", "provider_metadata", "maintained", "override", "probing", "unknown"
    reasoning_policy: str = "disabled"  # "disabled", "optional", "required", "required_unconfigurable"
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def is_known(self) -> bool:
        return self.capability_confidence != "unknown" and self.capability_source != "unknown"

    @property
    def supports_reasoning(self) -> bool:
        return self.reasoning_policy != ReasoningEffortPolicy.DISABLED.value

    @property
    def reasoning_required(self) -> bool:
        return self.reasoning_policy in {
            ReasoningEffortPolicy.REQUIRED.value,
            ReasoningEffortPolicy.REQUIRED_UNCONFIGURABLE.value,
        }

    @property
    def reasoning_can_disable(self) -> bool:
        return self.reasoning_policy in {
            ReasoningEffortPolicy.DISABLED.value,
            ReasoningEffortPolicy.OPTIONAL.value,
        }

    @property
    def reasoning_effort_configurable(self) -> bool:
        return self.reasoning_policy in {
            ReasoningEffortPolicy.OPTIONAL.value,
            ReasoningEffortPolicy.REQUIRED.value,
        }

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
            "supports_server_tools": self.supports_server_tools,
            "capability_confidence": self.capability_confidence,
            "capability_source": self.capability_source,
            "reasoning_policy": self.reasoning_policy,
            "supports_reasoning": self.supports_reasoning,
            "reasoning_required": self.reasoning_required,
            "reasoning_can_disable": self.reasoning_can_disable,
            "reasoning_effort_configurable": self.reasoning_effort_configurable,
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
        supports_server_tools=True,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
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
        supports_server_tools=True,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
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
        supports_server_tools=True,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
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
        supports_server_tools=True,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
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
        supports_server_tools=True,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
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
        supports_server_tools=True,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
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
        supports_server_tools=True,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openai", "gpt-6", "direct_responses"): ModelCapabilityDescriptor(
        provider="openai",
        model="gpt-6",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=True,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openai", "gpt-6-astra", "direct_responses"): ModelCapabilityDescriptor(
        provider="openai",
        model="gpt-6-astra",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=True,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openai", "astra", "direct_responses"): ModelCapabilityDescriptor(
        provider="openai",
        model="astra",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=True,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openai", "gpt-6", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openai",
        model="gpt-6",
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
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openai", "gpt-6-astra", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openai",
        model="gpt-6-astra",
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
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openai", "astra", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openai",
        model="astra",
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
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
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
        supports_server_tools=True,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.REQUIRED.value,
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
        supports_server_tools=True,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.REQUIRED.value,
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
        supports_server_tools=True,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.REQUIRED.value,
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
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
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
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
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
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    # OpenRouter maintained entries
    ("openrouter", "x-ai/grok-4.6", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="x-ai/grok-4.6",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.REQUIRED_UNCONFIGURABLE.value,
    ),
    ("openrouter", "x-ai/grok-4.6", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="x-ai/grok-4.6",
        transport="responses_bridge",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.REQUIRED_UNCONFIGURABLE.value,
    ),
    ("openrouter", "x-ai/grok-2-1212", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="x-ai/grok-2-1212",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "x-ai/grok-2-1212", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="x-ai/grok-2-1212",
        transport="responses_bridge",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "x-ai/grok-beta", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="x-ai/grok-beta",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "x-ai/grok-beta", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="x-ai/grok-beta",
        transport="responses_bridge",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "x-ai/grok-vision-beta", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="x-ai/grok-vision-beta",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "x-ai/grok-vision-beta", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="x-ai/grok-vision-beta",
        transport="responses_bridge",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "x-ai/grok-3", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="x-ai/grok-3",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openrouter", "x-ai/grok-3", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="x-ai/grok-3",
        transport="responses_bridge",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openrouter", "x-ai/grok-3-mini", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="x-ai/grok-3-mini",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.REQUIRED.value,
    ),
    ("openrouter", "x-ai/grok-3-mini", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="x-ai/grok-3-mini",
        transport="responses_bridge",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.REQUIRED.value,
    ),
    ("openrouter", "z-ai/glm-4.5", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="z-ai/glm-4.5",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openrouter", "z-ai/glm-4.5", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="z-ai/glm-4.5",
        transport="responses_bridge",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openrouter", "z-ai/glm-4.5-air", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="z-ai/glm-4.5-air",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openrouter", "z-ai/glm-4.5-air", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="z-ai/glm-4.5-air",
        transport="responses_bridge",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openrouter", "z-ai/glm-4.5v", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="z-ai/glm-4.5v",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "z-ai/glm-4.5v", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="z-ai/glm-4.5v",
        transport="responses_bridge",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "z-ai/glm-4-plus", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="z-ai/glm-4-plus",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "z-ai/glm-4-plus", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="z-ai/glm-4-plus",
        transport="responses_bridge",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "z-ai/glm-4-9b-chat", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="z-ai/glm-4-9b-chat",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "z-ai/glm-4-9b-chat", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="z-ai/glm-4-9b-chat",
        transport="responses_bridge",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "z-ai/glm-4-voice", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="z-ai/glm-4-voice",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "z-ai/glm-4-voice", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="z-ai/glm-4-voice",
        transport="responses_bridge",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "z-ai/glm-4", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="z-ai/glm-4",
        transport="direct_responses",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "z-ai/glm-4", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="z-ai/glm-4",
        transport="responses_bridge",
        supports_tool_calls=True,
        supports_repository_read=True,
        supports_repository_write=True,
        supports_shell=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_parallel_tools=True,
        supports_server_tools=False,
        capability_confidence="high",
        capability_source="maintained",
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
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
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openrouter", "anthropic/claude-3.7-sonnet", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="anthropic/claude-3.7-sonnet",
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
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
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
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "anthropic/claude-3.5-sonnet", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
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
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "deepseek/deepseek-v4-flash", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="deepseek/deepseek-v4-flash",
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
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openrouter", "deepseek/deepseek-v4-flash", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="deepseek/deepseek-v4-flash",
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
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openrouter", "deepseek/deepseek-r1", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="deepseek/deepseek-r1",
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
        reasoning_policy=ReasoningEffortPolicy.REQUIRED_UNCONFIGURABLE.value,
    ),
    ("openrouter", "deepseek/deepseek-r1", "responses_bridge"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="deepseek/deepseek-r1",
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
        reasoning_policy=ReasoningEffortPolicy.REQUIRED_UNCONFIGURABLE.value,
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
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
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
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
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
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
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
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "google/gemini-2.5-pro", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="google/gemini-2.5-pro",
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
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openrouter", "google/gemini-2.5-flash", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="google/gemini-2.5-flash",
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
        reasoning_policy=ReasoningEffortPolicy.OPTIONAL.value,
    ),
    ("openrouter", "qwen/qwen-2.5-coder-32b-instruct", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="qwen/qwen-2.5-coder-32b-instruct",
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
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "mistralai/mistral-large-2411", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="mistralai/mistral-large-2411",
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
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
    ),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct", "direct_responses"): ModelCapabilityDescriptor(
        provider="openrouter",
        model="meta-llama/llama-3.3-70b-instruct",
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
        reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
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
            if catalog_records is None and cache_key in self._cache:
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

        # 2. Explicitly supplied catalog records take precedence
        if catalog_records is not None:
            for record in catalog_records:
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

        # 3. Maintained entries (exact match or family prefix match)
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
                        supports_server_tools=m_desc.supports_server_tools,
                        capability_confidence=m_desc.capability_confidence,
                        capability_source="maintained",
                        reasoning_policy=m_desc.reasoning_policy,
                    )

        # 4. Cached catalog records from disk
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

        # 5. Unknown fail-closed descriptor (minimum safe capabilities)
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
            supports_server_tools=False,
            capability_confidence="unknown",
            capability_source="unknown",
            reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
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
        supports_server_tools = bool(
            supports_tools and provider == "openai" and transport == "direct_responses"
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
                supports_server_tools=False,
                capability_confidence="high",
                capability_source="catalog",
                reasoning_policy=ReasoningEffortPolicy.DISABLED.value,
                metadata=record,
            )

        # Classify reasoning capability / policy from catalog record & model identifiers
        has_reasoning = bool(
            supported_lower & {"reasoning", "reasoning_effort", "thinking", "include_reasoning", "max_thinking_tokens"}
            or (isinstance(caps, list) and any("reasoning" in str(c).lower() or "thinking" in str(c).lower() for c in caps))
        )
        lowered_model = model.lower()
        if any(marker in lowered_model for marker in ("grok-4.6", "deepseek-r1", "o1", "o3")) and not any(marker in lowered_model for marker in ("o3-mini", "o4-mini")):
            reasoning_policy = ReasoningEffortPolicy.REQUIRED_UNCONFIGURABLE.value
        elif any(marker in lowered_model for marker in ("o3-mini", "o4-mini")):
            reasoning_policy = ReasoningEffortPolicy.REQUIRED.value
        elif has_reasoning or any(marker in lowered_model for marker in ("claude-3.7", "deepseek-v4", "gpt-5", "gpt-6", "astra")):
            reasoning_policy = ReasoningEffortPolicy.OPTIONAL.value
        else:
            reasoning_policy = ReasoningEffortPolicy.DISABLED.value

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
                supports_server_tools=supports_server_tools,
                capability_confidence="high",
                capability_source="catalog",
                reasoning_policy=reasoning_policy,
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
            supports_server_tools=False,
            capability_confidence="unknown",
            capability_source="unknown",
            reasoning_policy=reasoning_policy,
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
                    elif hasattr(item, "id"):
                        meta = getattr(item, "metadata", {}) or {}
                        if isinstance(meta, dict):
                            result.append({"id": getattr(item, "id"), **meta})
                        else:
                            result.append({"id": getattr(item, "id")})
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


def normalize_reasoning_request_overrides(
    provider: str,
    model: str,
    transport: str | CodexTransport | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and normalize request overrides against authoritative reasoning policy."""
    if not isinstance(overrides, dict) or not overrides:
        return {}
    policy = resolve_model_request_policy(provider, model, transport)
    normalized = dict(overrides)
    if not policy.reasoning_required:
        return normalized

    # For reasoning-required models, remove disable flags and unconfigurable effort
    removed: list[tuple[str, Any]] = []

    # Check top-level reasoning_effort
    if "reasoning_effort" in normalized:
        effort_val = str(normalized.get("reasoning_effort") or "").strip().lower()
        if effort_val in {"none", "off", "0", "false", "disabled"} or not policy.reasoning_effort_configurable:
            val = normalized.pop("reasoning_effort")
            removed.append(("reasoning_effort", val))

    # Check top-level reasoning dict or boolean
    if "reasoning" in normalized:
        r_val = normalized.get("reasoning")
        if r_val is False or r_val == "disabled":
            val = normalized.pop("reasoning")
            removed.append(("reasoning", val))
        elif isinstance(r_val, dict):
            if r_val.get("enabled") is False or str(r_val.get("effort") or "").strip().lower() in {"none", "off", "0", "false", "disabled"}:
                val = normalized.pop("reasoning")
                removed.append(("reasoning", val))
            elif not policy.reasoning_effort_configurable:
                val = normalized.pop("reasoning")
                removed.append(("reasoning", val))

    # Check enable_thinking / thinking
    for key in ("enable_thinking", "thinking"):
        if key in normalized:
            val = normalized.pop(key)
            removed.append((key, val))

    # Check nested chat_template_kwargs
    if "chat_template_kwargs" in normalized and isinstance(normalized["chat_template_kwargs"], dict):
        ctk = dict(normalized["chat_template_kwargs"])
        ctk_changed = False
        if ctk.get("thinking") is False:
            ctk.pop("thinking", None)
            ctk_changed = True
            removed.append(("chat_template_kwargs.thinking", False))
        if str(ctk.get("reasoning_effort") or "").strip().lower() in {"none", "off", "0"}:
            val = ctk.pop("reasoning_effort", None)
            ctk_changed = True
            removed.append(("chat_template_kwargs.reasoning_effort", val))
        if ctk_changed:
            if ctk:
                normalized["chat_template_kwargs"] = ctk
            else:
                normalized.pop("chat_template_kwargs", None)

    # Check nested extra_body
    if "extra_body" in normalized and isinstance(normalized["extra_body"], dict):
        extra = dict(normalized["extra_body"])
        cleaned_extra = normalize_reasoning_request_overrides(
            provider, model, transport, extra
        )
        if cleaned_extra != extra:
            normalized["extra_body"] = cleaned_extra

    if removed:
        for r_key, r_val in removed:
            record_current(
                "codex.request.override_removed",
                {
                    "provider": policy.provider,
                    "model": policy.model,
                    "transport": policy.transport,
                    "reasoning_required": policy.reasoning_required,
                    "reasoning_configurable": policy.reasoning_effort_configurable,
                    "requested_reasoning": f"{r_key}={r_val!r}",
                    "effective_reasoning": "omitted" if not policy.reasoning_effort_configurable else policy.reasoning_policy,
                    "metadata_source": policy.capability_source,
                },
            )
            record_current(
                "codex.request.reasoning_normalized",
                {
                    "provider": policy.provider,
                    "model": policy.model,
                    "transport": policy.transport,
                    "reasoning_required": policy.reasoning_required,
                    "reasoning_configurable": policy.reasoning_effort_configurable,
                    "requested_reasoning": f"{r_key}={r_val!r}",
                    "effective_reasoning": "default",
                    "metadata_source": policy.capability_source,
                },
            )
    return normalized


__all__ = [
    "ModelCapabilityDescriptor",
    "ModelCapabilityRegistry",
    "ModelRequestPolicy",
    "ReasoningEffortPolicy",
    "clear_capability_cache",
    "clear_model_capability_overrides",
    "normalize_model_lookup_id",
    "normalize_reasoning_request_overrides",
    "register_model_capability_override",
    "resolve_model_capability",
    "resolve_model_request_policy",
    "resolve_transport_name",
]
