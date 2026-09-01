"""Domain dataclasses and models for specialist lanes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mana_agent.gateway.lanes import LaneId, LanePriority, LaneTaskState, LockMode


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class LaneBudget:
    reserved_input_tokens: int = 0
    reserved_output_tokens: int = 0
    consumed_input_tokens: int = 0
    consumed_output_tokens: int = 0
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    estimated_cost_known: bool = False
    actual_cost_known: bool = False
    model_context_window: int = 0
    model_max_output_tokens: int = 0
    estimate_confidence: str = ""
    estimate_source: str = ""
    revisions: list[dict[str, Any]] = field(default_factory=list)

    turn_budget_tokens: int = 0
    turn_consumed_tokens: int = 0
    turn_reserved_tokens: int = 0

    def start_new_turn(self, allocated_tokens: int) -> None:
        self.turn_budget_tokens = allocated_tokens
        self.turn_consumed_tokens = 0
        self.turn_reserved_tokens = 0

    @property
    def is_turn_budget_exhausted(self) -> bool:
        return self.turn_budget_tokens > 0 and (
            self.turn_consumed_tokens + self.turn_reserved_tokens >= self.turn_budget_tokens
        )

    @property
    def turn_remaining_tokens(self) -> int:
        if self.turn_budget_tokens <= 0:
            return 1_000_000_000
        return max(0, self.turn_budget_tokens - self.turn_consumed_tokens - self.turn_reserved_tokens)

    @property
    def reserved_tokens(self) -> int:
        return self.reserved_input_tokens + self.reserved_output_tokens

    @property
    def consumed_tokens(self) -> int:
        return self.consumed_input_tokens + self.consumed_output_tokens


@dataclass(slots=True)
class LaneHandoff:
    source_lane: LaneId
    target_lane: LaneId
    task_id: str
    reason: str
    artifacts: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    remaining_work: list[str] = field(default_factory=list)
    verification_state: dict[str, Any] = field(default_factory=dict)
    budget_consumed: LaneBudget = field(default_factory=LaneBudget)
    created_at: str = field(default_factory=_iso)


@dataclass(slots=True)
class LaneExecution:
    task_id: str
    root_task_id: str
    parent_task_id: str | None = None
    owning_lane: LaneId = LaneId.CODING
    state: LaneTaskState = LaneTaskState.CREATED
    normalized_intent: str = ""
    repository_id: str = ""
    workspace_id: str = ""
    session_id: str = ""
    target_files: list[str] = field(default_factory=list)
    priority: LanePriority = LanePriority.NORMAL
    budget: LaneBudget = field(default_factory=LaneBudget)
    taskboard_task_id: str = ""
    worker_id: str = ""
    model: str = ""
    provider: str = ""
    routing_decision_id: str = ""
    accounting_reservation_ids: list[str] = field(default_factory=list)
    task_type: str = "single"
    capabilities: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    verification_state: dict[str, Any] = field(default_factory=dict)
    lane_history: list[dict[str, Any]] = field(default_factory=list)
    handoffs: list[LaneHandoff] = field(default_factory=list)
    duplicate_of: str | None = None
    last_heartbeat: str = field(default_factory=_iso)
    created_at: str = field(default_factory=_iso)
    updated_at: str = field(default_factory=_iso)
    task_created_at: str = ""
    scheduled_at: str = ""
    worker_claimed_at: str = ""
    provider_started_at: str = ""
    provider_completed_at: str = ""
    task_completed_at: str = ""
    duration_breakdown: dict[str, int] = field(default_factory=dict)
    error: str = ""
    heartbeat_failure: str = ""
    progress_summary: str = ""
    current_tool_activity: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    cancellation_state: dict[str, Any] = field(default_factory=dict)
    final_result: dict[str, Any] = field(default_factory=dict)
    supervisor_attempt_id: str = ""
    supervisor_lease_token: str = ""
    checkpoint_id: str = ""
    trigger_turn_id: str = ""
    relation_type: str = "independent"
    previous_task_id: str = ""
    user_message_id: str = ""


@dataclass(slots=True)
class LockLease:
    lease_id: str
    task_id: str
    mode: LockMode
    workspace_id: str
    repository_id: str
    paths: list[str]
    owner_pid: int
    acquired_at: str
    expires_at: str


@dataclass(slots=True)
class LaneReservation:
    execution: LaneExecution
    duplicate: bool = False
