"""Typed contracts for session-scoped context and cost governance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class GovernorMode(str, Enum):
    OBSERVE = "observe"
    SOFT = "soft"
    ENFORCE = "enforce"


@dataclass(frozen=True, slots=True)
class ContextSegment:
    kind: str
    content: Any
    token_estimate: int
    protected: bool = False
    source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True, slots=True)
class ContextBreakdown:
    system_tokens: int = 0
    user_tokens: int = 0
    history_tokens: int = 0
    memory_tokens: int = 0
    evidence_tokens: int = 0
    schema_tokens: int = 0
    tool_result_tokens: int = 0
    other_tokens: int = 0

    @property
    def input_tokens(self) -> int:
        return sum(asdict(self).values())

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "input_tokens": self.input_tokens}


@dataclass(frozen=True, slots=True)
class ContextBudget:
    context_window: int
    task_token_limit: int | None = None
    session_token_limit: int | None = None
    monetary_limit: float | None = None
    response_reserve_tokens: int = 0
    reasoning_reserve_tokens: int = 0
    safety_margin_tokens: int = 0
    warning_ratio: float = 0.70
    compact_ratio: float = 0.80
    max_utilization: float = 0.85
    hard_limit_ratio: float = 0.95


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    breakdown: ContextBreakdown
    budget: ContextBudget
    used_tokens: int
    remaining_tokens: int
    utilization_ratio: float
    cumulative_tokens: int
    remaining_task_tokens: int | None
    cumulative_cost: float
    remaining_cost: float | None
    estimated: bool
    status: str = "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "breakdown": self.breakdown.as_dict(),
            "context_window": self.budget.context_window,
            "limits": {
                "task_token_limit": self.budget.task_token_limit,
                "session_token_limit": self.budget.session_token_limit,
                "monetary_limit": self.budget.monetary_limit,
                "response_reserve_tokens": self.budget.response_reserve_tokens,
                "reasoning_reserve_tokens": self.budget.reasoning_reserve_tokens,
                "safety_margin_tokens": self.budget.safety_margin_tokens,
                "warning_ratio": self.budget.warning_ratio,
                "compact_ratio": self.budget.compact_ratio,
                "max_utilization": self.budget.max_utilization,
                "hard_limit_ratio": self.budget.hard_limit_ratio,
            },
            "used_tokens": self.used_tokens,
            "remaining_tokens": self.remaining_tokens,
            "utilization_ratio": self.utilization_ratio,
            "cumulative_tokens": self.cumulative_tokens,
            "remaining_task_tokens": self.remaining_task_tokens,
            "cumulative_cost": self.cumulative_cost,
            "remaining_cost": self.remaining_cost,
            "estimated": self.estimated,
            "status": self.status,
        }


@dataclass(slots=True)
class CostLedger:
    ledger_id: str
    parent_id: str | None = None
    token_limit: int | None = None
    cost_limit: float | None = None
    tokens_used: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    children: dict[str, "CostLedger"] = field(default_factory=dict)
    _parent: "CostLedger | None" = field(default=None, repr=False)

    @property
    def total_cost(self) -> float:
        return self.actual_cost + self.estimated_cost

    @property
    def remaining_tokens(self) -> int | None:
        return None if self.token_limit is None else max(0, self.token_limit - self.tokens_used)

    @property
    def remaining_cost(self) -> float | None:
        return None if self.cost_limit is None else max(0.0, self.cost_limit - self.total_cost)

    def allocate_child(
        self,
        ledger_id: str,
        *,
        token_limit: int | None = None,
        cost_limit: float | None = None,
        allow_parent_override: bool = False,
    ) -> "CostLedger":
        if ledger_id in self.children:
            return self.children[ledger_id]
        if not allow_parent_override:
            reserved_tokens = sum(child.token_limit or 0 for child in self.children.values())
            reserved_cost = sum(child.cost_limit or 0.0 for child in self.children.values())
            if self.remaining_tokens is not None and token_limit is not None and token_limit > max(0, self.remaining_tokens - reserved_tokens):
                raise ValueError("child token allocation exceeds parent remaining budget")
            if self.remaining_cost is not None and cost_limit is not None and cost_limit > max(0.0, self.remaining_cost - reserved_cost):
                raise ValueError("child cost allocation exceeds parent remaining budget")
        child = CostLedger(
            ledger_id=ledger_id,
            parent_id=self.ledger_id,
            token_limit=token_limit,
            cost_limit=cost_limit,
            _parent=self,
        )
        self.children[ledger_id] = child
        return child

    def record(self, *, tokens: int, input_cost: float, output_cost: float, estimated: bool) -> None:
        self.tokens_used += max(0, int(tokens))
        self.input_cost += max(0.0, float(input_cost))
        self.output_cost += max(0.0, float(output_cost))
        total = max(0.0, float(input_cost)) + max(0.0, float(output_cost))
        if estimated:
            self.estimated_cost += total
        else:
            self.actual_cost += total
        if self._parent is not None:
            self._parent.record(tokens=tokens, input_cost=input_cost, output_cost=output_cost, estimated=estimated)


@dataclass(frozen=True, slots=True)
class CapabilityManifestEntry:
    name: str
    category: str
    description: str
    risk_class: str
    permission_requirements: tuple[str, ...]
    factory_key: str
    estimated_schema_tokens: int
    aliases: tuple[str, ...] = ()


@dataclass(slots=True)
class ActiveCapabilitySet:
    loaded: set[str] = field(default_factory=set)
    last_used_step: dict[str, int] = field(default_factory=dict)
    schema_tokens: int = 0
    revision: int = 0


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    artifact_id: str
    content_hash: str
    session_id: str
    repository_id: str
    workspace_id: str
    content_type: str
    byte_length: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CompressionEnvelope:
    artifact_ref: ArtifactReference
    tool_name: str
    content_type: str
    summary: str
    important_items: tuple[Any, ...]
    omitted_counts: dict[str, int]
    original_token_estimate: int
    compact_token_estimate: int
    compression_ratio: float
    content_hash: str
    lossless_source_available: bool = True

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["artifact_ref"] = self.artifact_ref.artifact_id
        result["artifact_scope"] = self.artifact_ref.as_dict()
        result["type"] = "mana.context.compression_envelope"
        return result


@dataclass(frozen=True, slots=True)
class GovernorDecision:
    action: str
    reason: str
    allowed: bool
    snapshot: BudgetSnapshot
    segments: tuple[ContextSegment, ...] = ()
    tokens_saved: int = 0
    threshold: float | None = None


class ContextBudgetExceeded(RuntimeError):
    """Raised only when a validated enforce-mode budget blocks execution."""

    def __init__(self, decision: GovernorDecision) -> None:
        super().__init__(f"Context budget blocked: {decision.reason}. No provider call was executed.")
        self.decision = decision


__all__ = [
    "ActiveCapabilitySet", "ArtifactReference", "BudgetSnapshot", "CapabilityManifestEntry",
    "CompressionEnvelope", "ContextBreakdown", "ContextBudget", "ContextBudgetExceeded",
    "ContextSegment", "CostLedger", "GovernorDecision", "GovernorMode",
]
