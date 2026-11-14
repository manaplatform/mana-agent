"""Native model profiles with field-level, scoped capability evidence."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class CapabilityState(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNKNOWN = "UNKNOWN"


class CapabilityFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: CapabilityState = CapabilityState.UNKNOWN
    value: Any = None
    source: str = "unknown"
    method: str = "unobserved"
    observed_at: datetime | None = None
    confidence: float = Field(default=0, ge=0, le=1)
    scope: str = ""
    expires_at: datetime | None = None

    def current(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return self.expires_at is None or self.expires_at > now


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    provider: str
    model_id: str
    endpoint: str = ""
    api_version: str = ""
    transport: str = ""
    configuration_fingerprint: str = ""
    facts: dict[str, CapabilityFact] = Field(default_factory=dict)
    observations: dict[str, list[CapabilityFact]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def model(self) -> str:
        return self.model_id

    def fact(self, capability: str) -> CapabilityFact:
        value = self.facts.get(capability, CapabilityFact())
        return value if value.current() else CapabilityFact(source=value.source, method="expired")

    def supports(self, capability: str) -> bool:
        return self.fact(capability).state is CapabilityState.SUPPORTED

    def supports_value(self, capability: str, value: str) -> bool:
        choices = []
        parents = self.observations.get(capability, [self.fact(capability)])
        specifics = self.observations.get(f"{capability}.{value}", [self.fact(f"{capability}.{value}")])
        for parent in parents:
            if not parent.current() or parent.state is CapabilityState.UNKNOWN or parent.scope not in {"", self.configuration_fingerprint}:
                continue
            if parent.state is CapabilityState.UNSUPPORTED:
                choices.append((parent, False))
            elif isinstance(parent.value, list):
                if value in parent.value or parent.source != "probe":
                    choices.append((parent, value in parent.value))
        for specific in specifics:
            if specific.current() and specific.state is not CapabilityState.UNKNOWN and specific.scope in {"", self.configuration_fingerprint}:
                choices.append((specific, specific.state is CapabilityState.SUPPORTED))
        priority = {"unknown": 0, "models.dev": 1, "provider": 2, "probe": 3, "override": 4}
        choices.sort(key=lambda item: (priority.get(item[0].source, 0), item[0].observed_at.isoformat() if item[0].observed_at else ""), reverse=True)
        return choices[0][1] if choices else False

    def require(self, *capabilities: str) -> None:
        from mana_agent.llm.errors import CapabilityError
        rejected = [f"{name}={self.fact(name).state.value}" for name in capabilities if not self.supports(name)]
        if rejected:
            raise CapabilityError(f"{self.provider}/{self.model_id}: {', '.join(rejected)}. Open model settings to refresh metadata, configure an override, or initiate a capability test.")

    def value(self, name: str, default: Any = None) -> Any:
        return self.fact(name).value if self.supports(name) else default

    @property
    def is_known(self) -> bool:
        return any(self.fact(name).state is not CapabilityState.UNKNOWN for name in self.facts)

    @property
    def capability_source(self) -> str:
        return ",".join(sorted({f.source for f in self.facts.values() if f.current() and f.source != "unknown"})) or "unknown"

    @property
    def capability_confidence(self) -> str:
        scores = [f.confidence for f in self.facts.values() if f.current() and f.state is not CapabilityState.UNKNOWN]
        return "unknown" if not scores else "high" if min(scores) >= .8 else "medium" if min(scores) >= .5 else "low"

    @property
    def context_window(self) -> int | None:
        value = self.value("context_window")
        return value if type(value) is int and value > 0 else None

    @property
    def max_output_tokens(self) -> int | None:
        value = self.value("max_output_tokens")
        return value if type(value) is int and value > 0 else None

    @property
    def input_modalities(self) -> list[str]:
        return self.value("input_modalities", [])

    @property
    def output_modalities(self) -> list[str]:
        return self.value("output_modalities", [])

    @property
    def reasoning_effort_levels(self) -> list[str]:
        value = self.value("reasoning_effort", [])
        return value if isinstance(value, list) else []

    @property
    def reasoning_policy(self) -> str:
        return self.value("reasoning_policy", "unknown")

    @property
    def reasoning_required(self) -> bool:
        return self.reasoning_policy in {"required", "required_unconfigurable"}

    @property
    def reasoning_can_disable(self) -> bool:
        return self.reasoning_policy in {"disabled", "optional"}

    @property
    def reasoning_effort_configurable(self) -> bool:
        return self.supports("reasoning_effort")

    @property
    def supports_batch(self) -> bool | None:
        state = self.fact("batch").state
        return True if state is CapabilityState.SUPPORTED else False if state is CapabilityState.UNSUPPORTED else None

    @property
    def supports_batch_display(self) -> str:
        return "unknown" if self.supports_batch is None else "yes" if self.supports_batch else "no"

    @property
    def supported_batch_endpoints(self) -> list[str]:
        return self.value("batch_endpoints", [])

    @property
    def supported_api_endpoints(self) -> list[str]:
        return self.value("api_endpoints", [])

    @property
    def batch_pricing(self) -> dict[str, Any]:
        return self.metadata.get("batch_pricing", {})

    @property
    def batch_rate_limits(self) -> dict[str, Any]:
        return self.metadata.get("batch_rate_limits", {})

    @property
    def batch_restrictions(self) -> list[str]:
        return self.metadata.get("batch_restrictions", [])

    @property
    def supported_completion_windows(self) -> list[str]:
        return self.value("batch_completion_windows", [])

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @property
    def raw_metadata(self) -> dict[str, Any]:
        return self.metadata

    @property
    def catalog_pricing(self) -> dict[str, Any]:
        timestamp = self.metadata.get("catalog_pricing_expires_at")
        if not isinstance(timestamp, str):
            return {}
        try:
            expiration = datetime.fromisoformat(timestamp)
        except ValueError:
            return {}
        if expiration.tzinfo is None or expiration <= datetime.now(timezone.utc):
            return {}
        pricing = self.metadata.get("catalog_pricing")
        return pricing if isinstance(pricing, dict) else {}


# These views expose the canonical facts to existing Mana domain contracts.
# Client tool support describes inference, never workspace permissions.
_VIEWS = {
    "supports_tool_calls": "tool_calling", "function_calling": "tool_calling",
    "function_calling_supported": "tool_calling", "supports_repository_read": "tool_calling",
    "supports_repository_write": "tool_calling", "supports_shell": "tool_calling",
    "supports_structured_output": "native_structured_output", "structured_output_supported": "native_structured_output",
    "supports_streaming": "streaming", "streaming_supported": "streaming",
    "supports_reasoning": "reasoning", "supports_public_reasoning_summaries": "public_reasoning_summaries",
    "supports_parallel_tools": "parallel_tool_calls", "parallel_function_calling_supported": "parallel_tool_calls",
    "supports_server_tools": "server_tools", "programmatic_tool_calling": "programmatic_tool_calling",
    "computer_use": "computer_use", "image_generation": "image_generation",
    "prompt_caching_supported": "prompt_caching", "web_search": "web_search", "file_search": "file_search",
    "code_interpreter": "code_interpreter", "shell_and_local_shell": "tool_calling",
    "apply_patch": "tool_calling", "mcp": "tool_calling", "tool_search": "tool_search", "skills": "tool_calling",
}
for _attribute, _capability in _VIEWS.items():
    setattr(ModelProfile, _attribute, property(lambda self, name=_capability: self.supports(name)))


def observation(value: Any, *, source: str, scope: str = "", method: str = "metadata", confidence: float = .8) -> CapabilityFact:
    now = datetime.now(timezone.utc)
    state = CapabilityState.UNKNOWN if value is None else CapabilityState.UNSUPPORTED if value is False else CapabilityState.SUPPORTED
    return CapabilityFact(state=state, value=value, source=source, method=method,
        observed_at=now, confidence=confidence, scope=scope,
        expires_at=None if source == "override" else now + timedelta(days=7 if source == "probe" else 1))


def metadata_facts(record: dict[str, Any], *, source: str, scope: str = "") -> dict[str, CapabilityFact]:
    """Only explicit fields grant capabilities. Model names are never inspected."""
    values: dict[str, Any] = {}
    names = {"image_request_format": "image_request_format", "image_flexible_dimensions": "image_flexible_dimensions",
        "image_quality_levels": "image_quality_levels", "image_input_fidelity": "image_input_fidelity", "embedding": "embedding", "tool_call": "tool_calling", "tool_calling": "tool_calling", "tools": "tool_calling",
        "tool_choice": "tool_choice", "reasoning": "reasoning", "reasoning_effort": "reasoning_effort",
        "thinking_control": "thinking_control",
        "structured_output": "native_structured_output", "native_structured_output": "native_structured_output",
        "tool_call_streaming": "tool_call_streaming", "streaming": "streaming",
        "parallel_tool_calls": "parallel_tool_calls", "context_window": "context_window",
        "context_length": "context_window", "max_output_tokens": "max_output_tokens",
        "max_completion_tokens": "max_output_tokens", "max_input_tokens": "max_input_tokens",
        "reasoning_policy": "reasoning_policy", "programmatic_tool_calling": "programmatic_tool_calling",
        "supports_batch": "batch", "decisions_api": "decisions_api", "supported_batch_endpoints": "batch_endpoints",
        "supported_api_endpoints": "api_endpoints", "public_reasoning_summaries": "public_reasoning_summaries"}
    for external, native in names.items():
        if external in record:
            values[native] = record[external]
    capabilities = record.get("capabilities")
    if isinstance(capabilities, dict):
        for key, value in capabilities.items():
            if key in names:
                values[names[key]] = value
    elif isinstance(capabilities, list):
        for key in capabilities:
            if key in names:
                values[names[key]] = True
    parameters = record.get("supported_parameters")
    if isinstance(parameters, list):
        for key, native in {"tools": "tool_calling", "tool_choice": "tool_choice", "reasoning": "reasoning", "reasoning_effort": "reasoning_effort", "structured_outputs": "native_structured_output", "parallel_tool_calls": "parallel_tool_calls"}.items():
            if key in parameters:
                values.setdefault(native, True)
    limits = record.get("limit")
    if not isinstance(limits, dict):
        limits = {}
    for key, native in {"context": "context_window", "input": "max_input_tokens", "output": "max_output_tokens"}.items():
        if key in limits:
            values[native] = limits[key] if type(limits[key]) is int and limits[key] > 0 else None
    top_provider = record.get("top_provider")
    if isinstance(top_provider, dict):
        values.setdefault("context_window", top_provider.get("context_length"))
        values.setdefault("max_output_tokens", top_provider.get("max_completion_tokens"))
    for name in ("input_text", "input_image", "input_audio", "input_video", "input_pdf", "output_text", "output_image", "output_audio", "output_video", "image_generation", "image_editing", "speech_to_text", "text_to_speech", "realtime", "batch", "batch_endpoints", "batch_completion_windows"):
        if name in record:
            values[name] = record[name]
    modalities = record.get("modalities") or record.get("architecture") or {key: record.get(key) for key in ("input_modalities", "output_modalities")}
    if not isinstance(modalities, dict):
        modalities = {}
    for key in ("input", "output"):
        value = modalities.get(key, modalities.get(f"{key}_modalities"))
        if isinstance(value, list):
            values[f"{key}_modalities"] = value
            for modality in value:
                if isinstance(modality, str):
                    values.setdefault(f"{key}_{modality}", True)
    for option in record.get("reasoning_options") or []:
        if isinstance(option, dict) and option.get("type") == "effort":
            values["reasoning_effort"] = option.get("values")
    for name in ("context_window", "max_input_tokens", "max_output_tokens"):
        if name in values and (type(values[name]) is not int or values[name] <= 0):
            values[name] = None
    return {name: observation(value, source=source, scope=scope) for name, value in values.items()}
