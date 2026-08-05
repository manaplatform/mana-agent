"""Exact model-aware price calculation; unknown pricing remains unknown."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True, slots=True)
class CostEstimate:
    input_cost: Decimal | None
    output_cost: Decimal | None
    cached_input_cost: Decimal | None = None
    reasoning_cost: Decimal | None = None
    estimated: bool = False

    @property
    def total_cost(self) -> Decimal | None:
        values = (self.input_cost, self.output_cost, self.cached_input_cost, self.reasoning_cost)
        return None if any(value is None for value in values[:2]) else sum((value or Decimal("0") for value in values), Decimal("0"))


def _rate(profile: Any, *names: str) -> Decimal | None:
    for name in names:
        value = getattr(profile, name, None)
        if value not in (None, ""):
            parsed = Decimal(str(value))
            if parsed > 0:
                return parsed
    return None


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    *,
    cached_input_tokens: int = 0,
    reasoning_tokens: int = 0,
    profile: Any | None = None,
    **_obsolete_fallback_rates: Any,
) -> CostEstimate:
    if profile is None:
        return CostEstimate(None, None, estimated=True)
    input_rate = _rate(profile, "input_price_per_million", "input_cost_per_million")
    output_rate = _rate(profile, "output_price_per_million", "output_cost_per_million")
    if input_rate is None or output_rate is None:
        return CostEstimate(None, None, estimated=True)
    cached_rate = _rate(profile, "cached_input_price_per_million", "cached_input_cost_per_million") or input_rate
    reasoning_rate = _rate(profile, "reasoning_price_per_million", "reasoning_cost_per_million")
    million = Decimal(1_000_000)
    return CostEstimate(
        Decimal(max(0, input_tokens)) * input_rate / million,
        Decimal(max(0, output_tokens)) * output_rate / million,
        Decimal(max(0, cached_input_tokens)) * cached_rate / million,
        (Decimal(max(0, reasoning_tokens)) * reasoning_rate / million if reasoning_rate is not None else Decimal("0")),
        estimated=False,
    )


__all__ = ["CostEstimate", "calculate_cost"]
