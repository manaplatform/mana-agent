"""Pricing adapters that distinguish configured prices from estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CostEstimate:
    input_cost: float
    output_cost: float
    estimated: bool

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    *,
    profile: Any | None = None,
    fallback_input_per_million: float = 1.0,
    fallback_output_per_million: float = 3.0,
) -> CostEstimate:
    configured_input = float(getattr(profile, "input_cost_per_million", 0.0) or 0.0)
    configured_output = float(getattr(profile, "output_cost_per_million", 0.0) or 0.0)
    exact = profile is not None and configured_input > 0 and configured_output > 0
    input_rate = configured_input if configured_input > 0 else fallback_input_per_million
    output_rate = configured_output if configured_output > 0 else fallback_output_per_million
    return CostEstimate(
        input_cost=max(0, input_tokens) * input_rate / 1_000_000,
        output_cost=max(0, output_tokens) * output_rate / 1_000_000,
        estimated=not exact,
    )


__all__ = ["CostEstimate", "calculate_cost"]
