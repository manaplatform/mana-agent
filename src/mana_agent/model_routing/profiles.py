from __future__ import annotations

from collections import defaultdict
import json
from typing import Any

from mana_agent.config.provider_registry import split_qualified_model_id
from mana_agent.config.user_config import get_setting
from mana_agent.model_routing.models import LatencyClass, ModelProfile, sanitize_configuration
from mana_agent.multi_agent.core.types import AgentRole


_LEVELS = (
    ("MODEL_LEVEL_1_FAST_TOOL", ""),
    ("MODEL_LEVEL_2_CODING", ""),
    ("MODEL_LEVEL_3_HIGH_REASONING", ""),
)
_ALL_ROLES = frozenset(role.value for role in AgentRole)
_LEVEL_METADATA = {
    "MODEL_LEVEL_1_FAST_TOOL": (0.78, 1.0, LatencyClass.INTERACTIVE, frozenset({"none"})),
    "MODEL_LEVEL_2_CODING": (0.87, 3.0, LatencyClass.STANDARD, frozenset({"medium", "high"})),
    "MODEL_LEVEL_3_HIGH_REASONING": (0.93, 6.0, LatencyClass.STANDARD, frozenset({"low", "medium", "high"})),
}
_LEVEL_BENCHMARKS = {
    "MODEL_LEVEL_1_FAST_TOOL": {"routine": 0.92, "tool": 0.92, "summarization": 0.90, "research": 0.84},
    "MODEL_LEVEL_2_CODING": {"coding": 0.96, "verification": 0.94, "routine": 0.82},
    "MODEL_LEVEL_3_HIGH_REASONING": {"routing": 0.97, "planning": 0.97, "review": 0.96, "coding": 0.88},
}


class ProfileValidationError(ValueError):
    pass


def configured_profiles(value: list[dict[str, Any]] | str) -> tuple[ModelProfile, ...]:
    if isinstance(value, str):
        if not value.strip():
            return ()
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ProfileValidationError(f"MANA_MODEL_PROFILES is not valid JSON: {exc}") from exc
    else:
        parsed = value
    if not isinstance(parsed, list):
        raise ProfileValidationError("MANA_MODEL_PROFILES must be a list")
    profiles: list[ModelProfile] = []
    errors: list[str] = []
    for index, raw in enumerate(parsed):
        if not isinstance(raw, dict):
            errors.append(f"profile {index} must be an object")
            continue
        provider = str(raw.get("provider") or "").strip()
        model_id = str(raw.get("model_id") or raw.get("model") or "").strip()
        roles = frozenset(str(item).strip() for item in raw.get("supported_roles") or [] if str(item).strip())
        if not provider or not model_id or not roles:
            errors.append(f"profile {index} requires provider, model_id, and supported_roles")
            continue
        try:
            profiles.append(ModelProfile(
                provider=provider,
                model_id=model_id,
                supported_roles=roles,
                supported_tools=frozenset(str(item) for item in raw.get("supported_tools") or []),
                reasoning_settings=frozenset(str(item) for item in raw.get("reasoning_settings") or ["none"]),
                context_window=int(raw.get("context_window", 128_000)),
                latency_class=LatencyClass(str(raw.get("latency_class") or "standard")),
                input_cost_per_million=float(raw.get("input_cost_per_million") or 0.0),
                output_cost_per_million=float(raw.get("output_cost_per_million") or 0.0),
                logical_cost_per_1k_tokens=float(raw.get("logical_cost_per_1k_tokens", 1.0)),
                reliability_score=float(raw.get("reliability_score", 0.8)),
                supported_languages=frozenset(str(item).lower() for item in raw.get("supported_languages") or []),
                benchmark_scores={str(key): float(score) for key, score in dict(raw.get("benchmark_scores") or {}).items()},
                can_patch=bool(raw.get("can_patch", True)),
                can_structured_output=bool(raw.get("can_structured_output", True)),
                can_tool_call=bool(raw.get("can_tool_call", True)),
                can_verify=bool(raw.get("can_verify", True)),
                available=bool(raw.get("available", True)),
                configuration=sanitize_configuration(dict(raw.get("configuration") or {})),
                source_level=str(raw.get("source_level") or "configured"),
            ))
        except (TypeError, ValueError) as exc:
            errors.append(f"profile {index}: {exc}")
    if errors:
        raise ProfileValidationError("; ".join(errors))
    keys = [item.key for item in profiles]
    if len(set(keys)) != len(keys):
        raise ProfileValidationError("MANA_MODEL_PROFILES contains duplicate provider/model IDs")
    return tuple(profiles)


def profiles_from_legacy_configuration(*, global_model: str = "", default_provider: str = "openai") -> tuple[ModelProfile, ...]:
    """Migrate logical levels into candidate hints without preserving role locks."""
    configured: list[tuple[str, str]] = []
    for level, _default in _LEVELS:
        value = str(get_setting(level, "") or "").strip()
        if value:
            configured.append((level, value))
    if global_model:
        configured.append(("MODEL_LEVEL_1_FAST_TOOL", global_model))
    if not configured:
        return ()

    levels_by_model: dict[tuple[str, str], list[str]] = defaultdict(list)
    for level, value in configured:
        provider, model_id = split_qualified_model_id(value, default_provider=default_provider)
        if model_id:
            levels_by_model[(provider, model_id)].append(level)

    profiles: list[ModelProfile] = []
    for (provider, model_id), levels in sorted(levels_by_model.items()):
        strongest = max(levels, key=lambda item: _LEVELS_INDEX[item])
        reliability, logical_cost, latency, reasoning = _LEVEL_METADATA[strongest]
        # One selected model can intentionally serve several logical levels.
        # Keep the strongest quality/cost evidence, but retain its explicit
        # fast-level assignment for interactive lanes. Otherwise a model used
        # for both high-reasoning and tool work is incorrectly rejected before
        # routing can make a provider/model decision.
        if "MODEL_LEVEL_1_FAST_TOOL" in levels:
            latency = LatencyClass.INTERACTIVE
        profiles.append(ModelProfile(
            provider=provider,
            model_id=model_id,
            supported_roles=_ALL_ROLES,
            supported_tools=frozenset({"*"}),
            reasoning_settings=reasoning,
            latency_class=latency,
            logical_cost_per_1k_tokens=logical_cost,
            reliability_score=reliability,
            benchmark_scores=dict(_LEVEL_BENCHMARKS[strongest]),
            source_level=strongest,
            configuration={"source_levels": tuple(sorted(levels))},
        ))
    return tuple(profiles)


_LEVELS_INDEX = {name: index for index, (name, _) in enumerate(_LEVELS)}
