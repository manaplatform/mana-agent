from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Iterable

from .models import UsageRecord

PRICING_VERSION = "2026-07-20"


@dataclass(frozen=True, slots=True)
class Price:
    provider: str
    model: str
    effective_date: date
    input_per_million: Decimal
    cached_input_per_million: Decimal | None
    output_per_million: Decimal
    reasoning_per_million: Decimal | None = None


class PricingRegistry:
    def __init__(self, prices: Iterable[Price] = (), *, version: str = PRICING_VERSION) -> None:
        self.version = version
        self._prices = {(item.provider, item.model): item for item in prices}

    def add(self, price: Price) -> None:
        self._prices[(price.provider, price.model)] = price

    def calculate(self, *, provider: str, model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int, reasoning_tokens: int) -> float | None:
        price = self._prices.get((provider, model))
        if price is None:
            return None
        total = Decimal(input_tokens) * price.input_per_million
        total += Decimal(output_tokens) * price.output_per_million
        if price.cached_input_per_million is not None:
            total += Decimal(cached_input_tokens) * price.cached_input_per_million
        elif cached_input_tokens:
            total += Decimal(cached_input_tokens) * price.input_per_million
        if price.reasoning_per_million is not None:
            total += Decimal(reasoning_tokens) * price.reasoning_per_million
        return float(total / Decimal(1_000_000))

    def usage_record(self, *, provider: str, model: str, input_tokens: int = 0, cached_input_tokens: int = 0, output_tokens: int = 0, reasoning_tokens: int = 0, latency_seconds: float = 0.0, lane: str = "") -> UsageRecord:
        return UsageRecord(
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=input_tokens + cached_input_tokens + output_tokens + reasoning_tokens,
            latency_seconds=latency_seconds,
            provider=provider,
            model=model,
            calculated_cost=self.calculate(
                provider=provider, model=model, input_tokens=input_tokens,
                cached_input_tokens=cached_input_tokens, output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
            ),
            pricing_table_version=self.version,
            lane=lane,
        )


DEFAULT_PRICING = PricingRegistry()
