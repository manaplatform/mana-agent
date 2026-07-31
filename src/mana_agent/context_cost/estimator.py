"""Deterministic token estimates and context component classification."""

from __future__ import annotations

import json
from typing import Any, Iterable

from mana_agent.context_cost.models import ContextBreakdown, ContextSegment
from mana_agent.telemetry.tokens import estimate_tokens


def estimate_value_tokens(value: Any) -> int:
    if isinstance(value, str):
        return estimate_tokens(value)
    try:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = str(value)
    return estimate_tokens(rendered)


def estimate_tool_schema_tokens(tool: Any) -> int:
    payload = {
        "name": getattr(tool, "name", type(tool).__name__),
        "description": getattr(tool, "description", ""),
        "args_schema": _schema(getattr(tool, "args_schema", None)),
    }
    return estimate_value_tokens(payload)


def _schema(value: Any) -> Any:
    if value is None:
        return {}
    for method in ("model_json_schema", "schema"):
        candidate = getattr(value, method, None)
        if callable(candidate):
            try:
                return candidate()
            except (TypeError, ValueError):
                continue
    return str(value)


def breakdown_for_segments(segments: Iterable[ContextSegment]) -> ContextBreakdown:
    totals = {name: 0 for name in ContextBreakdown.__dataclass_fields__}
    mapping = {
        "system": "system_tokens", "safety": "system_tokens", "user": "user_tokens",
        "history": "history_tokens", "memory": "memory_tokens", "retrieval": "evidence_tokens",
        "repository": "evidence_tokens", "document": "evidence_tokens", "schema": "schema_tokens",
        "tool_result": "tool_result_tokens",
    }
    for segment in segments:
        key = mapping.get(segment.kind, "other_tokens")
        totals[key] += max(0, int(segment.token_estimate))
    return ContextBreakdown(**totals)


__all__ = ["breakdown_for_segments", "estimate_tool_schema_tokens", "estimate_value_tokens"]
