"""Typed contracts for resilient task execution."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stable_id(prefix: str) -> str:
    return f"{prefix}_{uuid4()}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ExecutionState(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    WAITING = "waiting"
    RETRY_SCHEDULED = "retry_scheduled"
    REPLANNING = "replanning"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    COMPLETED_PENDING_VERIFICATION = "completed_pending_verification"
    COMPLETED = "completed"


TERMINAL_STATES = frozenset(
    {ExecutionState.CANCELLED, ExecutionState.FAILED, ExecutionState.BUDGET_EXHAUSTED, ExecutionState.COMPLETED}
)


class SideEffectClassification(str, Enum):
    READ_ONLY = "read_only"
    IDEMPOTENT = "idempotent"
    CONDITIONALLY_IDEMPOTENT = "conditionally_idempotent"
    DEDUPLICATED = "deduplicated"
    COMPENSATABLE = "compensatable"
    NON_IDEMPOTENT = "non_idempotent"
    UNKNOWN = "unknown"


class ActionRequestState(str, Enum):
    PREPARED = "prepared"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    OUTCOME_UNKNOWN = "outcome_unknown"
    RECONCILED = "reconciled"


class ActionRecord(StrictModel):
    action_id: str = Field(default_factory=lambda: stable_id("action"))
    execution_id: str
    attempt_id: str
    attempt_generation: int = Field(ge=1)
    tool_name: str = Field(min_length=1)
    action_fingerprint: str = Field(min_length=1)
    idempotency_key: str = ""
    classification: SideEffectClassification
    request_state: ActionRequestState = ActionRequestState.PREPARED
    external_receipt: str = ""
    result_reference: str = ""
    verification_state: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class RetryCategory(str, Enum):
    INFRASTRUCTURE = "infrastructure"
    MODEL = "model"
    TOOL = "tool"
    VERIFICATION = "verification"
    LEASE_LOSS = "lease_loss"
    REPLAN = "replan"


class CompletionContractType(str, Enum):
    FILE_EXISTS = "file_exists"
    DIRECTORY_EXISTS = "directory_exists"
    GIT_DIFF_PRESENT = "git_diff_present"
    GIT_COMMIT_EXISTS = "git_commit_exists"
    COMMAND_SUCCEEDED = "command_succeeded"
    STRUCTURED_RESULT_VALID = "structured_result_valid"
    REMOTE_RESOURCE_CONFIRMED = "remote_resource_confirmed"
    CUSTOM_VERIFIER = "custom_verifier"


class VerificationStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    NOT_SUPPORTED = "not_supported"


class EscrowStatus(str, Enum):
    PRODUCED = "produced"
    STORED = "stored"
    AVAILABLE = "available"
    DELIVERY_PENDING = "delivery_pending"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    REJECTED = "rejected"


class CancellationStatus(str, Enum):
    REQUESTED = "requested"
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"
    FORCED = "forced"
    PARTIALLY_COMPLETED = "partially_completed"
    BLOCKED_BY_SIDE_EFFECT = "blocked_by_side_effect"


class WaitPolicy(str, Enum):
    FAIL_FAST = "fail_fast"
    WAIT_ALL = "wait_all"
    BEST_EFFORT = "best_effort"
    MINIMUM_SUCCESS_COUNT = "minimum_success_count"
    DEPENDENCY_GRAPH = "dependency_graph"


class RetryBudget(StrictModel):
    infrastructure: int = Field(default=3, ge=0)
    model: int = Field(default=2, ge=0)
    tool: int = Field(default=2, ge=0)
    verification: int = Field(default=1, ge=0)
    lease_loss: int = Field(default=3, ge=0)
    replan: int = Field(default=2, ge=0)

    def remaining(self, category: RetryCategory, used: dict[str, int]) -> int:
        return max(0, int(getattr(self, category.value)) - int(used.get(category.value, 0)))


class CompletionContract(StrictModel):
    contract_type: CompletionContractType
    path: str = ""
    expected_sha256: str = ""
    minimum_size: int = Field(default=0, ge=0)
    expected_kind: Literal["file", "directory", "any"] = "any"
    require_attempt_change: bool = False
    commit: str = ""
    verifier_name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompletionArtifact(StrictModel):
    artifact_type: str
    path: str = ""
    exists: bool = False
    size: int | None = Field(default=None, ge=0)
    sha256: str = ""
    verified_at: datetime | None = None
    produced_by_attempt: bool | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class VerificationReport(StrictModel):
    status: VerificationStatus
    checks: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[CompletionArtifact] = Field(default_factory=list)
    verified_at: datetime = Field(default_factory=utc_now)


class AttemptRecord(StrictModel):
    attempt_id: str = Field(default_factory=lambda: stable_id("attempt"))
    task_id: str
    number: int = Field(ge=1)
    generation: int = Field(default=1, ge=1)
    state: str = "created"
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    lease_owner: str = ""
    lease_token: str = ""
    lease_expires_at: datetime | None = None
    failure_reason: str = ""
    recovery_reason: str = ""
    checkpoint_id: str = ""
    token_usage: int = Field(default=0, ge=0)
    estimated_cost: float = Field(default=0.0, ge=0)
    actual_cost: float = Field(default=0.0, ge=0)


class CheckpointRecord(StrictModel):
    schema_version: int = Field(default=2, ge=1)
    checkpoint_id: str = Field(default_factory=lambda: stable_id("checkpoint"))
    task_id: str
    attempt_id: str
    state_version: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    resume_payload: dict[str, Any] = Field(default_factory=dict)
    completed_steps: list[str] = Field(default_factory=list)
    pending_steps: list[str] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    workspace_reference: str = ""
    git_reference: str = ""
    generated_files: list[str] = Field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.PENDING
    plan_version: int = Field(default=0, ge=0)
    child_execution_ids: list[str] = Field(default_factory=list)
    result_escrow_references: list[str] = Field(default_factory=list)
    artifact_references: list[str] = Field(default_factory=list)
    context_manifest_id: str = ""
    budget_snapshot: dict[str, Any] = Field(default_factory=dict)
    retry_state: dict[str, Any] = Field(default_factory=dict)
    idempotency_records: list[str] = Field(default_factory=list)
    external_action_receipts: list[str] = Field(default_factory=list)
    resume_cursor: str = ""


class EscrowResult(StrictModel):
    result_id: str = Field(default_factory=lambda: stable_id("result"))
    task_id: str
    parent_task_id: str | None = None
    attempt_id: str
    attempt_generation: int = Field(default=1, ge=1)
    lease_token_hash: str
    created_at: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[CompletionArtifact] = Field(default_factory=list)
    status: EscrowStatus = EscrowStatus.PRODUCED
    delivery_count: int = Field(default=0, ge=0)
    delivered_at: datetime | None = None
    rejected_reason: str = ""
    acknowledged_at: datetime | None = None
    acknowledged_by: str = ""


class ExecutionEvent(StrictModel):
    event_id: str = Field(default_factory=lambda: stable_id("event"))
    event_type: str
    task_id: str
    parent_task_id: str | None = None
    root_task_id: str
    attempt_id: str = ""
    state: ExecutionState
    created_at: datetime = Field(default_factory=utc_now)
    details: dict[str, Any] = Field(default_factory=dict)


class TaskRecord(StrictModel):
    schema_version: int = Field(default=3, ge=1)
    state_version: int = Field(default=0, ge=0)
    task_id: str = Field(default_factory=lambda: stable_id("task"))
    parent_task_id: str | None = None
    root_task_id: str = ""
    attempt_id: str = ""
    attempt_ids: list[str] = Field(default_factory=list)
    attempt_generation: int = Field(default=0, ge=0)
    child_task_ids: list[str] = Field(default_factory=list)
    dependency_task_ids: list[str] = Field(default_factory=list)
    task_type: str = "task"
    state: ExecutionState = ExecutionState.CREATED
    assigned_agent: str = ""
    assigned_model: str = ""
    assigned_worker: str = ""
    runtime_provider: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    heartbeat_at: datetime | None = None
    lease_owner: str = ""
    lease_token: str = ""
    lease_expires_at: datetime | None = None
    retry_count: int = Field(default=0, ge=0)
    retry_budget: RetryBudget = Field(default_factory=RetryBudget)
    retry_usage: dict[str, int] = Field(default_factory=dict)
    retry_not_before: datetime | None = None
    checkpoint_id: str = ""
    checkpoint_count: int = Field(default=0, ge=0)
    idempotency_key: str = ""
    compensation_strategy: str = ""
    side_effect_classification: SideEffectClassification = SideEffectClassification.UNKNOWN
    irreversible_side_effect_started: bool = False
    completion_contract: list[CompletionContract] = Field(default_factory=list)
    completion_artefacts: list[CompletionArtifact] = Field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.PENDING
    failure_reason: str = ""
    recovery_reason: str = ""
    cancellation_status: CancellationStatus | None = None
    cancellation_reason: str = ""
    token_usage: int = Field(default=0, ge=0)
    token_budget: int | None = Field(default=None, ge=0)
    estimated_cost: float = Field(default=0.0, ge=0)
    actual_cost: float = Field(default=0.0, ge=0)
    monetary_budget: float | None = Field(default=None, ge=0)
    deadline_at: datetime | None = None
    wait_policy: WaitPolicy = WaitPolicy.WAIT_ALL
    minimum_success_count: int | None = Field(default=None, ge=1)
    max_child_depth: int = Field(default=5, ge=0)
    max_children: int = Field(default=20, ge=0)
    max_total_subtasks: int = Field(default=100, ge=0)
    max_concurrent_children: int = Field(default=4, ge=1)
    routing_decision_id: str = ""
    workspace_path: str = ""
    result_id: str = ""
    execution_fingerprint: str = ""
    session_id: str = ""
    workspace_id: str = ""
    repository_id: str = ""
    normalized_intent: str = ""
    requested_operation: str = ""
    target_resources: list[str] = Field(default_factory=list)
    expected_output: str = ""
    important_constraints: list[str] = Field(default_factory=list)
    supersedes_execution_id: str = ""
    derived_from_execution_id: str = ""
    previous_execution_id: str = ""
    trigger_turn_id: str = ""
    relation_type: str = "independent"
    previous_task_id: str = ""

    @model_validator(mode="after")
    def normalize_root(self) -> "TaskRecord":
        if not self.root_task_id:
            object.__setattr__(self, "root_task_id", self.task_id)
        if self.parent_task_id == self.task_id:
            raise ValueError("a task cannot be its own parent")
        for field_name in (
            "created_at", "updated_at", "started_at", "finished_at",
            "heartbeat_at", "lease_expires_at", "retry_not_before", "deadline_at",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{field_name} must be timezone-aware")
            object.__setattr__(self, field_name, value.astimezone(timezone.utc))
        return self


class RecoveryAction(str, Enum):
    CONTINUE = "continue"
    RETRY = "retry"
    RESUME_CHECKPOINT = "resume_checkpoint"
    REASSIGN = "reassign"
    REPLAN = "replan"
    WAIT_FOR_DEPENDENCY = "wait_for_dependency"
    REQUIRE_REVERIFICATION = "require_reverification"
    PAUSE_FOR_USER = "pause_for_user"
    MARK_BUDGET_EXHAUSTED = "mark_budget_exhausted"
    FAIL = "fail"
    REQUIRE_INTERVENTION = "require_intervention"


class RecoveryDecision(StrictModel):
    decision_id: str = Field(min_length=1)
    task_id: str
    action: RecoveryAction
    retry_category: RetryCategory = RetryCategory.LEASE_LOSS
    reason: str = Field(min_length=1)
    selected_agent: str = ""
    selected_worker: str = ""
    selected_model: str = ""
    resume_checkpoint_id: str = ""
    same_task_retry_authorized: bool = False
    safe_to_continue: bool


class RecoverySummary(StrictModel):
    scanned: int = 0
    recovered: list[str] = Field(default_factory=list)
    retry_scheduled: list[str] = Field(default_factory=list)
    intervention_required: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)


class ParentProgress(StrictModel):
    task_id: str
    policy: WaitPolicy
    total_children: int
    completed: int
    failed: int
    cancelled: int
    active: int
    timed_out: bool = False
    satisfied: bool = False
    blocking_task_ids: list[str] = Field(default_factory=list)
    blocking_dependencies: dict[str, list[str]] = Field(default_factory=dict)
