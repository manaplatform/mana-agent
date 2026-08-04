"""Provider-neutral actual token usage normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    result: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            item = getattr(value, name)
        except Exception:
            continue
        if isinstance(item, (str, int, float, bool, type(None), Mapping, list)):
            result[name] = item
    return result


def _first(raw: Mapping[str, Any], *names: str) -> int:
    for name in names:
        value = raw.get(name)
        if value not in (None, ""):
            return max(0, int(value))
    return 0


@dataclass(frozen=True, slots=True)
class ActualTokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    actual_cost: Decimal | None = None
    status: str = "reported"
    provider_fields: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if min(self.input_tokens, self.cached_input_tokens, self.output_tokens, self.reasoning_tokens, self.total_tokens) < 0:
            raise ValueError("token usage cannot be negative")


def normalize_provider_usage(usage: Any) -> ActualTokenUsage | None:
    if usage is None:
        return None
    raw = _mapping(usage)
    if not raw:
        return None
    input_details = _mapping(raw.get("input_token_details") or raw.get("prompt_tokens_details"))
    output_details = _mapping(raw.get("output_token_details") or raw.get("completion_tokens_details"))
    input_tokens = _first(
        raw, "input_tokens", "prompt_tokens", "inputTokens", "prompt_token_count",
        "promptTokenCount", "inputTokenCount",
    )
    output_tokens = _first(
        raw, "output_tokens", "completion_tokens", "outputTokens",
        "candidates_token_count", "candidatesTokenCount", "outputTokenCount",
    )
    cached = _first(
        raw, "cached_input_tokens", "cached_tokens", "cachedInputTokens",
        "cachedTokens", "cache_read_input_tokens",
    ) or _first(input_details, "cached_tokens", "cached_input_tokens")
    reasoning = _first(raw, "reasoning_tokens", "reasoningTokens") or _first(output_details, "reasoning_tokens")
    total = _first(raw, "total_tokens", "totalTokens", "total_token_count", "totalTokenCount")
    if total <= 0:
        total = input_tokens + output_tokens
        if reasoning and reasoning > output_tokens:
            total += reasoning
    if total <= 0:
        return None
    safe_fields = {
        key: raw[key]
        for key in (
            "input_tokens", "prompt_tokens", "output_tokens", "completion_tokens",
            "total_tokens", "inputTokens", "outputTokens", "totalTokens",
            "prompt_token_count", "candidates_token_count", "total_token_count",
        )
        if key in raw and isinstance(raw[key], (int, float, str))
    }
    return ActualTokenUsage(input_tokens, cached, output_tokens, reasoning, total, status="reported", provider_fields=safe_fields)


__all__ = ["ActualTokenUsage", "normalize_provider_usage"]
