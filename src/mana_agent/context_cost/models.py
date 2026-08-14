"""Typed contracts for session-scoped context and cost governance."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping
from datetime import datetime, timezone


class GovernorMode(str, Enum):
    OBSERVE = "observe"
    SOFT = "soft"
    ENFORCE = "enforce"


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    reservation_id: str
    operation_type: str
    operation_id: str
    tokens: int
    cost: float
    cost_known: bool = True
    verification: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass(frozen=True, slots=True)
class ContextManifest:
    manifest_id: str
    model_call_id: str
    execution_id: str
    attempt_id: str
    included_messages: tuple[str, ...]
    included_files: tuple[str, ...]
    included_memories: tuple[str, ...]
    included_skills: tuple[str, ...]
    included_tool_schemas: tuple[str, ...]
    included_artifacts: tuple[str, ...]
    token_estimate: int
    reasons: tuple[str, ...]
    compression_references: tuple[str, ...]
    artifact_reference: str = ""
    current_turn_refs: tuple[str, ...] = ()
    current_turn_tokens: int = 0
    conversation_refs: tuple[str, ...] = ()
    conversation_tokens: int = 0
    memory_refs: tuple[str, ...] = ()
    memory_tokens: int = 0
    tool_refs: tuple[str, ...] = ()
    tool_tokens: int = 0
    artifact_refs: tuple[str, ...] = ()
    artifact_tokens: int = 0
    dependency_refs: tuple[str, ...] = ()
    dependency_tokens: int = 0
    skill_refs: tuple[str, ...] = ()
    skill_tokens: int = 0
    component_token_breakdown: Mapping[str, int] = field(default_factory=dict)


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
class ToolResultEnvelope:
    tool_name: str
    tool_call_id: str
    status: str
    artifact_ref: str
    content_hash: str
    original_tokens: int
    projection_tokens: int
    inline_projection: Any
    truncated: bool
    more_available: bool
    source_refs: tuple[str, ...] = ()
    content_type: str = "text"
    replayable: bool = True
    sensitive: bool = False

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["type"] = "mana.context.tool_result_envelope"
        result["compression_envelope"] = "mana.context.compression_envelope"
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


@dataclass(frozen=True, slots=True)
class ProviderCallForecast:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    safety_margin_tokens: int
    context_window: int
    max_output_tokens: int
    estimated_cost: Decimal | None
    confidence: str
    components: Mapping[str, int]
    assumptions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "context_window": self.context_window,
            "max_output_tokens": self.max_output_tokens,
            "estimated_cost": None if self.estimated_cost is None else format(self.estimated_cost, "f"),
            "confidence": self.confidence,
            "components": dict(self.components),
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True, slots=True)
class TaskExecutionForecast:
    task_id: str
    expected_calls: int
    expected_tool_steps: int
    forecast_input_tokens: int
    forecast_output_tokens: int
    forecast_total_tokens: int
    forecast_cost: float | None
    verification_reserve_tokens: int
    task_budget_tokens: int
    remaining_task_tokens: int
    feasible: bool
    rejection_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "expected_calls": self.expected_calls,
            "expected_tool_steps": self.expected_tool_steps,
            "forecast_input_tokens": self.forecast_input_tokens,
            "forecast_output_tokens": self.forecast_output_tokens,
            "forecast_total_tokens": self.forecast_total_tokens,
            "forecast_cost": self.forecast_cost,
            "verification_reserve_tokens": self.verification_reserve_tokens,
            "task_budget_tokens": self.task_budget_tokens,
            "remaining_task_tokens": self.remaining_task_tokens,
            "feasible": self.feasible,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True, slots=True)
class AccountingSnapshot:
    task_id: str
    turn_id: str
    task_budget_tokens: int
    task_consumed_tokens: int
    task_reserved_tokens: int
    task_remaining_tokens: int
    turn_budget_tokens: int | None
    turn_consumed_tokens: int
    turn_remaining_tokens: int | None
    verification_reserve_tokens: int
    session_budget_tokens: int | None
    session_consumed_tokens: int
    session_remaining_tokens: int | None
    cost_budget: float | None
    cost_consumed: float
    cost_remaining: float | None
    active_reservations_count: int
    status: str = "ok"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "AccountingSnapshot", "ActiveCapabilitySet", "ArtifactReference", "BudgetReservation", "BudgetSnapshot", "CapabilityManifestEntry",
    "CompressionEnvelope", "ContextBreakdown", "ContextBudget", "ContextBudgetExceeded",
    "ContextManifest", "ContextSegment", "CostLedger", "GovernorDecision", "GovernorMode",
    "ProviderCallForecast", "TaskExecutionForecast", "ToolResultEnvelope",
]
