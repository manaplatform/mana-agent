"""Configured routing candidates projected from the canonical native registry."""
from __future__ import annotations
import json
from typing import Any
from mana_agent.config.provider_registry import split_qualified_model_id
from mana_agent.config.user_config import get_setting
from mana_agent.config.model_capabilities import resolve_model_capability
from mana_agent.llm.capabilities import observation
from mana_agent.llm.registry import GLOBAL_CAPABILITY_REGISTRY
from mana_agent.model_routing.models import LatencyClass, ModelProfile, sanitize_configuration
from mana_agent.multi_agent.core.types import AgentRole

class ProfileValidationError(ValueError):
    pass

def configured_profiles(value: list[dict[str, Any]] | str) -> tuple[ModelProfile, ...]:
    try:
        parsed = json.loads(value) if isinstance(value, str) and value.strip() else ([] if isinstance(value, str) else value or [])
    except json.JSONDecodeError as exc:
        raise ProfileValidationError("MANA_MODEL_PROFILES must be valid JSON") from exc
    if not isinstance(parsed, list):
        raise ProfileValidationError("MANA_MODEL_PROFILES must be a list")
    candidates = []
    for raw in parsed:
        if not isinstance(raw, dict):
            raise ProfileValidationError("Each model profile must be an object")
        provider = str(raw.get("provider") or "").strip()
        model_id = str(raw.get("model_id") or raw.get("model") or "").strip()
        roles = frozenset(raw.get("supported_roles") or [role.value for role in AgentRole])
        if not provider or not model_id:
            raise ProfileValidationError("Routing candidates require explicit provider and model IDs")
        desc = resolve_model_capability(provider, model_id)
        provider, model_id = desc.provider, desc.model_id
        overrides = {}
        for key, native in {"can_tool_call": "tool_calling", "can_structured_output": "native_structured_output",
                            "context_window": "context_window", "max_output_tokens": "max_output_tokens"}.items():
            if key in raw:
                overrides[native] = observation(raw[key], source="override", scope=desc.configuration_fingerprint, method="configured_model_profile", confidence=1)
        if overrides:
            desc = GLOBAL_CAPABILITY_REGISTRY.merge(desc, overrides)
        config = sanitize_configuration(dict(raw.get("configuration") or {}))
        pricing = desc.catalog_pricing
        pricing_known = all(raw.get(setting, pricing.get(field)) is not None for setting, field in
            (("input_cost_per_million", "input"), ("output_cost_per_million", "output")))
        config.update(capability_source=desc.capability_source, capability_confidence=desc.capability_confidence,
                      capability_descriptor=desc.to_dict(), pricing_known=pricing_known)
        candidates.append(ModelProfile(provider=provider, model_id=model_id, supported_roles=roles,
            supported_tools=frozenset(raw.get("supported_tools") or (["*"] if desc.supports_tool_calls else [])),
            reasoning_settings=frozenset(desc.reasoning_effort_levels if isinstance(desc.reasoning_effort_levels, list) else []),
            context_window=int(desc.context_window or 0), max_output_tokens=int(desc.max_output_tokens or 0),
            input_cost_per_million=float(raw.get("input_cost_per_million", pricing.get("input", 0))),
            output_cost_per_million=float(raw.get("output_cost_per_million", pricing.get("output", 0))),
            cached_input_cost_per_million=raw.get("cached_input_cost_per_million", pricing.get("cache_read")),
            reasoning_cost_per_million=raw.get("reasoning_cost_per_million", pricing.get("reasoning")),
            latency_class=LatencyClass(raw.get("latency_class", "interactive")),
            reliability_score=float(raw.get("reliability_score", 0)), benchmark_scores=dict(raw.get("benchmark_scores") or {}),
            can_patch=desc.supports_tool_calls and raw.get("can_patch") is not False, can_tool_call=desc.supports_tool_calls,
            can_structured_output=desc.supports_structured_output,
            can_verify=(desc.supports_tool_calls or desc.supports_structured_output) and raw.get("can_verify") is not False,
            available=desc.metadata.get("available") is not False,
            configuration=config, capability_descriptor=desc, source_level=str(raw.get("source_level") or "configured")))
    return tuple(candidates)

def profiles_for_pinned_models(models: list[str] | tuple[str, ...], *, default_provider: str = "openai",
                               context_window: int | None = None, max_output_tokens: int | None = None) -> tuple[ModelProfile, ...]:
    rows = []
    identities = set()
    for value in dict.fromkeys(models):
        if not value:
            continue
        provider, model = split_qualified_model_id(value, default_provider=default_provider)
        if (provider, model) in identities:
            continue
        identities.add((provider, model))
        row = {"provider": provider, "model_id": model, "source_level": "pinned"}
        # Explicit supplied limits are operator configuration, never inferred model defaults.
        if context_window is not None:
            row["context_window"] = context_window
        if max_output_tokens is not None:
            row["max_output_tokens"] = max_output_tokens
        rows.append(row)
    return configured_profiles(rows)

def profiles_from_legacy_configuration(*, global_model: str = "", default_provider: str = "openai",
                                       context_window: int | None = None, max_output_tokens: int | None = None) -> tuple[ModelProfile, ...]:
    models = [str(get_setting(key, "") or "") for key in ("MODEL_LEVEL_1_FAST_TOOL", "MODEL_LEVEL_2_CODING", "MODEL_LEVEL_3_HIGH_REASONING")]
    if global_model:
        models.append(global_model)
    return profiles_for_pinned_models(models, default_provider=default_provider)
