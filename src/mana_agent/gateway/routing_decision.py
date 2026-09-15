"""Routing decision representation for gateway routing execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    route: str
    confidence: float = 1.0
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
