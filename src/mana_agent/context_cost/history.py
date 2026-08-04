"""Non-sensitive robust historical token prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class PredictionKey:
    provider: str
    model: str
    route: str
    lane: str
    execution_kind: str
    tool_count_bucket: int
    step_count_bucket: int


@dataclass(frozen=True, slots=True)
class HistoricalSample:
    key: PredictionKey
    input_tokens: int
    output_tokens: int


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(max(0, int(item)) for item in values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


class HistoricalTokenPredictor:
    def __init__(self, samples: Iterable[HistoricalSample] = (), *, percentile: float = 0.80, max_samples: int = 100) -> None:
        self.samples = list(samples)[-max(1, int(max_samples)):]
        self.percentile = min(0.99, max(0.50, float(percentile)))

    def record(self, sample: HistoricalSample) -> None:
        self.samples.append(sample)

    def predict(self, key: PredictionKey) -> tuple[int | None, int | None, int]:
        rows = [item for item in self.samples if item.key == key][-100:]
        return (
            _percentile([item.input_tokens for item in rows], self.percentile),
            _percentile([item.output_tokens for item in rows], self.percentile),
            len(rows),
        )


__all__ = ["HistoricalSample", "HistoricalTokenPredictor", "PredictionKey"]
