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
    PENDING_BUDGET_DECISION = "pending_budget_decision"
    RECOVERY_REVIEW_REQUIRED = "recovery_review_required"
    COMPLETED_PENDING_VERIFICATION = "completed_pending_verification"
    COMPLETED = "completed"


TERMINAL_STATES = frozenset(
    {
        ExecutionState.CANCELLED,
        ExecutionState.FAILED,
        ExecutionState.BUDGET_EXHAUSTED,
        ExecutionState.RECOVERY_REVIEW_REQUIRED,
        ExecutionState.COMPLETED,
    }
)


class SideEffectClassification(str, Enum):
    READ_ONLY = "read_only"
    IDEMPOTENT = "idempotent"
    CONDITIONALLY_IDEMPOTENT = "conditionally_idempotent"
    DEDUPLICATED = "deduplicated"
    COMPENSATABLE = "compensatable"
    NON_IDEMPOTENT = "non_idempotent"
    UNKNOWN = "unknown"


class ActionEffectScope(str, Enum):
    UNKNOWN = "UNKNOWN"
    LOCAL_REPOSITORY = "LOCAL_REPOSITORY"
    LOCAL_PROCESS = "LOCAL_PROCESS"
    REMOTE_REVERSIBLE = "REMOTE_REVERSIBLE"
    EXTERNAL_CONSEQUENTIAL = "EXTERNAL_CONSEQUENTIAL"


class LostLeaseOutcome(str, Enum):
    SAFE_AUTOMATIC_RECOVERY = "SAFE_AUTOMATIC_RECOVERY"
    LOCAL_RECONCILIATION_REQUIRED = "LOCAL_RECONCILIATION_REQUIRED"
    DURABLE_RESULT_AVAILABLE = "DURABLE_RESULT_AVAILABLE"
    RETRY_BUDGET_EXHAUSTED = "RETRY_BUDGET_EXHAUSTED"
    DEADLINE_EXPIRED = "DEADLINE_EXPIRED"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    UNKNOWN_EXTERNAL_OUTCOME = "UNKNOWN_EXTERNAL_OUTCOME"


class ReconciliationOutcome(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    PARTIALLY_APPLIED = "PARTIALLY_APPLIED"
    ALREADY_APPLIED = "ALREADY_APPLIED"
    UNKNOWN = "UNKNOWN"


class ActionRequestState(str, Enum):
    PREPARED = "prepared"
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    OUTCOME_UNKNOWN = "outcome_unknown"
    ACTION_RECONCILED = "action_reconciled"
    RECONCILED = "action_reconciled"  # backwards-compatible enum name


LOCAL_REPOSITORY_TOOLS = frozenset(
    {
        "apply_patch",
        "apply_patch_batch",
        "edit_file",
        "multi_edit_file",
        "write_file",
        "create_file",
        "delete_file",
        "write_to_file",
        "replace_file_content",
        "patch",
        "git_apply",
    }
)


class ActionRecord(StrictModel):
    action_id: str = Field(default_factory=lambda: stable_id("action"))
    execution_id: str
    attempt_id: str
    attempt_generation: int = Field(ge=1)
    tool_name: str = Field(min_length=1)
    action_fingerprint: str = Field(min_length=1)
    idempotency_key: str = ""
    classification: SideEffectClassification
    effect_scope: ActionEffectScope = ActionEffectScope.UNKNOWN
    request_state: ActionRequestState = ActionRequestState.PREPARED
    external_receipt: str = ""
    result_reference: str = ""
    verification_state: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def infer_effect_scope(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "verification_state" in data and data["verification_state"]:
            from mana_agent.utils.tool_results import json_safe_tool_payload

            data["verification_state"] = json_safe_tool_payload(data["verification_state"])
        if "effect_scope" not in data or not data["effect_scope"]:
            tool = str(data.get("tool_name") or "")
            classification = data.get("classification")
            if tool in LOCAL_REPOSITORY_TOOLS:
                data["effect_scope"] = ActionEffectScope.LOCAL_REPOSITORY.value
            elif tool in {
                "read_file",
                "view_file",
                "grep_search",
                "find_by_name",
                "list_dir",
                "directory_list",
                "file_read",
            } or classification in {
                SideEffectClassification.READ_ONLY.value,
                SideEffectClassification.READ_ONLY,
            }:
                data["effect_scope"] = ActionEffectScope.LOCAL_PROCESS.value
            elif tool in {
                "cloud_deploy",
                "send_payment",
                "send_email",
                "webhook_trigger",
                "external_api_call",
            } or classification in {
                SideEffectClassification.NON_IDEMPOTENT.value,
                SideEffectClassification.NON_IDEMPOTENT,
            }:
                data["effect_scope"] = ActionEffectScope.EXTERNAL_CONSEQUENTIAL.value
            else:
                data["effect_scope"] = ActionEffectScope.UNKNOWN.value
        return data


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
    estimated_cost_known: bool = False
    actual_cost_known: bool = False
    accounting_reservation_ids: list[str] = Field(default_factory=list)


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
    capsule_revisions: dict[str, int] = Field(default_factory=dict)
    budget_snapshot: dict[str, Any] = Field(default_factory=dict)
    retry_state: dict[str, Any] = Field(default_factory=dict)
    idempotency_records: list[str] = Field(default_factory=list)
    external_action_receipts: list[str] = Field(default_factory=list)
    resume_cursor: str = ""
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointResumeEligibility(StrictModel):
    resumable: bool
    reason: str = ""
    error_code: str = ""
    error_message: str = ""
    task_id: str = ""
    checkpoint_id: str = ""
    boundary: str = ""
    state: str = ""
    is_terminal: bool = False
    checkpoint: CheckpointRecord | None = None


class ResultAcknowledgement(StrictModel):
    acknowledgement_id: str = Field(default_factory=lambda: stable_id("ack"))
    result_id: str
    execution_id: str = ""
    consumer_execution_id: str = ""
    consumer_turn_id: str = ""
    acknowledged_at: datetime = Field(default_factory=utc_now)
    acknowledged_by: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EscrowLookupStatus(str, Enum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    NOT_YET_TERMINAL = "NOT_YET_TERMINAL"
    EXECUTION_STILL_RUNNING = "EXECUTION_STILL_RUNNING"
    UNVERIFIED = "UNVERIFIED"
    CORRUPT = "CORRUPT"
    INCOMPATIBLE_VERSION = "INCOMPATIBLE_VERSION"
    ESCROW_NOT_CONFIGURED = "ESCROW_NOT_CONFIGURED"


class EscrowResult(StrictModel):
    schema_version: int = Field(default=2, ge=1)
    result_id: str = Field(default_factory=lambda: stable_id("result"))
    execution_id: str = ""
    task_id: str
    root_task_id: str = ""
    parent_task_id: str | None = None
    trigger_turn_id: str = ""
    session_id: str = ""
    lane_id: str = ""
    owning_lane: str = ""
    attempt_id: str = ""
    attempt_generation: int = Field(default=1, ge=0)
    lease_token_hash: str = ""
    status: EscrowStatus = EscrowStatus.PRODUCED
    supervisor_state: str = ""
    verification_status: VerificationStatus = VerificationStatus.PENDING
    result_kind: str = "chat_result"
    payload: dict[str, Any] = Field(default_factory=dict)
    result_reference: str = ""
    verification_evidence: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[CompletionArtifact] = Field(default_factory=list)
    artifact_references: list[str] = Field(default_factory=list)
    capsule_revisions: dict[str, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    verified_at: datetime | None = None
    resume_checkpoint: str = ""
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    error_metadata: dict[str, Any] = Field(default_factory=dict)
    delivery_count: int = Field(default=0, ge=0)
    delivered_at: datetime | None = None
    rejected_reason: str = ""
    acknowledged_at: datetime | None = None
    acknowledged_by: str = ""

    @model_validator(mode="before")
    @classmethod
    def migrate_and_validate_schema(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        version = data.get("schema_version", 1)
        if isinstance(version, int) and version > 2:
            raise ValueError(f"unsupported escrow schema_version {version}; max supported is 2")
        if not data.get("schema_version") or version < 2:
            data = dict(data)
            data["schema_version"] = 2
            task_id = str(data.get("task_id") or "")
            if not data.get("execution_id"):
                data["execution_id"] = task_id
            if not data.get("root_task_id"):
                data["root_task_id"] = task_id
            if not data.get("lane_id") and data.get("owning_lane"):
                data["lane_id"] = str(data.get("owning_lane"))
            if not data.get("owning_lane") and data.get("lane_id"):
                data["owning_lane"] = str(data.get("lane_id"))
            if not data.get("supervisor_state"):
                st = data.get("status")
                if st in {"available", "delivered", "acknowledged"}:
                    data["supervisor_state"] = ExecutionState.COMPLETED.value
                else:
                    data["supervisor_state"] = ExecutionState.RUNNING.value
            if not data.get("verification_status"):
                if data.get("artifacts") or data.get("status") in {"available", "delivered", "acknowledged"}:
                    data["verification_status"] = VerificationStatus.PASSED.value
                else:
                    data["verification_status"] = VerificationStatus.PENDING.value
            if not data.get("result_kind"):
                data["result_kind"] = "chat_result"
        from mana_agent.utils.tool_results import json_safe_tool_payload

        for key in ("payload", "verification_evidence", "provider_metadata", "error_metadata"):
            if key in data and data[key]:
                data[key] = json_safe_tool_payload(data[key])
        return data


class VerifiedExecutionResultLookup(StrictModel):
    status: EscrowLookupStatus
    execution_id: str
    result: EscrowResult | None = None
    task: TaskRecord | None = None
    acknowledgement: ResultAcknowledgement | None = None
    error_code: str = ""
    error_message: str = ""
    is_terminal: bool = False
    is_resumable: bool = False
    is_verified: bool = False
    requires_action: bool = False



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


class RecoveryInterventionReason(str, Enum):
    AMBIGUOUS_LOST_LEASE = "AMBIGUOUS_LOST_LEASE"


class HumanRecoveryDecisionAction(str, Enum):
    RESUME_WITHOUT_REPLAY = "RESUME_WITHOUT_REPLAY"
    RETRY_ACTION = "RETRY_ACTION"
    MARK_ACTION_ALREADY_COMPLETED = "MARK_ACTION_ALREADY_COMPLETED"
    ABORT_EXECUTION = "ABORT_EXECUTION"


class RecoveryInterventionRecord(StrictModel):
    """Durable evidence for recovery that is blocked pending human review."""

    intervention_id: str = Field(default_factory=lambda: stable_id("recovery_intervention"))
    task_id: str
    execution_id: str
    attempt_id: str = ""
    action_id: str = ""
    checkpoint_id: str = ""
    integration_stage: str = ""
    target_resources: list[str] = Field(default_factory=list)
    receipt_lookup_state: str = ""
    reason_details: str = ""
    inbox_item_id: str = ""
    side_effect_classification: str = ""
    execution_state: Literal["interrupted"] = "interrupted"
    status: Literal["blocked"] = "blocked"
    reason: RecoveryInterventionReason
    action: Literal["human_review_required"] = "human_review_required"
    last_lease_owner: str = ""
    lease_expiry: datetime | None = None
    terminal_state: ExecutionState = ExecutionState.RECOVERY_REVIEW_REQUIRED
    external_side_effects_possible: bool
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def normalize_timestamps(self) -> "RecoveryInterventionRecord":
        if self.terminal_state not in TERMINAL_STATES:
            raise ValueError("terminal_state must be terminal")
        if self.lease_expiry is not None:
            if self.lease_expiry.tzinfo is None or self.lease_expiry.utcoffset() is None:
                raise ValueError("lease_expiry must be timezone-aware")
            object.__setattr__(self, "lease_expiry", self.lease_expiry.astimezone(timezone.utc))
        return self


class TaskRecord(StrictModel):
    schema_version: int = Field(default=8, ge=1)
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
    task_created_at: datetime | None = None
    scheduled_at: datetime | None = None
    worker_claimed_at: datetime | None = None
    provider_started_at: datetime | None = None
    provider_completed_at: datetime | None = None
    task_completed_at: datetime | None = None
    duration_breakdown: dict[str, int] = Field(default_factory=dict)
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
    cancellation_source: str = ""
    token_usage: int = Field(default=0, ge=0)
    token_budget: int | None = Field(default=None, ge=0)
    estimated_cost: float = Field(default=0.0, ge=0)
    actual_cost: float = Field(default=0.0, ge=0)
    estimated_cost_known: bool = False
    actual_cost_known: bool = False
    model_context_window: int = Field(default=0, ge=0)
    model_max_output_tokens: int = Field(default=0, ge=0)
    token_estimate_confidence: str = ""
    token_estimate_source: str = ""
    accounting_reservation_ids: list[str] = Field(default_factory=list)
    monetary_budget: float | None = Field(default=None, ge=0)
    budget_revisions: list[BudgetRevision] = Field(default_factory=list)
    budget_overrun: dict[str, Any] = Field(default_factory=dict)
    budget_finalization_decision_id: str = ""
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
    # Values that cannot exist until a provider call or verifier runs must stay
    # visibly unknown.  This avoids treating an empty string/list as evidence
    # that a value was deliberately selected or observed.
    field_provenance: dict[str, str] = Field(default_factory=dict)
    supervision_contract_decision_id: str = ""
    supersedes_execution_id: str = ""
    derived_from_execution_id: str = ""
    previous_execution_id: str = ""
    trigger_turn_id: str = ""
    relation_type: str = "independent"
    previous_task_id: str = ""
    delegated_capsule_revisions: dict[str, int] = Field(default_factory=dict)
    result_capsule_revisions: dict[str, int] = Field(default_factory=dict)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    waiting_inbox_item_id: str = ""
    waiting_kind: str = ""
    wake_up_source: str = ""
    wake_up_reference: str = ""
    resume_checkpoint_id: str = ""
    resume_operation: str = ""
    waiting_reason: Literal[
        "",
        "waiting_for_approval",
        "waiting_for_clarification",
        "waiting_for_connector",
        "ambiguous_lost_lease",
    ] = ""
    waiting_connector_id: str = ""
    required_connector_ids: list[str] = Field(default_factory=list)
    human_inputs: list[dict[str, Any]] = Field(default_factory=list)
    human_resume_claim_ids: list[str] = Field(default_factory=list)
    human_wait_started_at: datetime | None = None
    recovery_intervention_id: str = ""

    def wall_clock_deadline_exceeded(self, now: datetime | None = None) -> bool:
        """Return True when the absolute wall-clock deadline has already elapsed.

        Deadline-dead tasks must not be requeued. Callers create a new task with a
        fresh deadline instead of retrying or resuming the expired identity.
        """
        if self.deadline_at is None:
            return False
        clock = now or utc_now()
        deadline = self.deadline_at
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        if clock.tzinfo is None:
            clock = clock.replace(tzinfo=timezone.utc)
        return deadline <= clock

    @model_validator(mode="after")
    def normalize_root(self) -> "TaskRecord":
        if self.schema_version < 8:
            object.__setattr__(self, "schema_version", 8)
        provenance_defaults = {
            "side_effect_classification": "legacy_or_unspecified",
            "completion_contract": (
                "model_selected" if self.completion_contract else "pending_runtime_evidence"
            ),
            "target_resources": (
                "model_selected" if self.target_resources else "not_applicable_or_not_selected"
            ),
            "important_constraints": (
                "model_selected" if self.important_constraints else "not_applicable_or_not_selected"
            ),
            "estimated_cost": (
                "provider_estimate" if self.estimated_cost_known else "unknown_provider_pricing"
            ),
            "actual_cost": (
                "runtime_accounting" if self.actual_cost_known else "pending_runtime_accounting"
            ),
            "completion_artefacts": (
                "verified" if self.completion_artefacts else "pending_completion_verification"
            ),
        }
        for field_name, status in provenance_defaults.items():
            self.field_provenance.setdefault(field_name, status)
        if not self.root_task_id:
            object.__setattr__(self, "root_task_id", self.task_id)
        if self.parent_task_id == self.task_id:
            raise ValueError("a task cannot be its own parent")
        for field_name in (
            "created_at", "updated_at", "started_at", "finished_at",
            "task_created_at", "scheduled_at", "worker_claimed_at",
            "provider_started_at", "provider_completed_at", "task_completed_at",
            "heartbeat_at", "lease_expires_at", "retry_not_before", "deadline_at",
            "human_wait_started_at",
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


class BudgetOverrunAction(str, Enum):
    ACCEPT_WITH_OVERRUN = "accept_with_overrun"
    REQUIRE_REVIEW = "require_review"
    RETRY_OR_REPLAN = "retry_or_replan"


class BudgetRevision(StrictModel):
    revision_id: str = Field(default_factory=lambda: stable_id("budget_revision"))
    reason: str = Field(min_length=1)
    previous_token_budget: int | None = Field(default=None, ge=0)
    revised_token_budget: int | None = Field(default=None, ge=0)
    previous_estimated_cost: float = Field(default=0.0, ge=0)
    revised_estimated_cost: float = Field(default=0.0, ge=0)
    evidence: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class BudgetForecast(StrictModel):
    task_id: str = Field(min_length=1)
    forecast_input_tokens: int = Field(ge=0)
    forecast_output_tokens: int = Field(ge=0)
    forecast_cost: float | None = Field(default=None, ge=0)
    accounting_reservation_id: str = ""
    reason: str = Field(min_length=1)


class BudgetOverrunFinalizationDecision(StrictModel):
    """Fresh model decision required before an over-budget result can progress."""

    decision_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    result_id: str = Field(min_length=1)
    result_evidence_hash: str = Field(min_length=1)
    action: BudgetOverrunAction
    reason: str = Field(min_length=1)
    safe_to_continue: bool
    recovery_decision: RecoveryDecision | None = None

    @model_validator(mode="after")
    def validate_recovery(self) -> "BudgetOverrunFinalizationDecision":
        if not self.safe_to_continue:
            raise ValueError("budget-overrun finalization decision is not safe to continue")
        if self.action is BudgetOverrunAction.RETRY_OR_REPLAN:
            if self.recovery_decision is None:
                raise ValueError("retry_or_replan requires a recovery decision")
            if self.recovery_decision.task_id != self.task_id:
                raise ValueError("recovery decision task does not match budget-overrun task")
            if self.recovery_decision.action not in {RecoveryAction.RETRY, RecoveryAction.REPLAN}:
                raise ValueError("budget-overrun recovery must select retry or replan")
            if not self.recovery_decision.safe_to_continue:
                raise ValueError("budget-overrun recovery decision is not safe to continue")
        elif self.recovery_decision is not None:
            raise ValueError("only retry_or_replan may include a recovery decision")
        return self


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


BudgetOverrunFinalizationDecision.model_rebuild()
TaskRecord.model_rebuild()


class RecoverySummary(StrictModel):
    scanned: int = 0
    recovered: list[str] = Field(default_factory=list)
    retry_scheduled: list[str] = Field(default_factory=list)
    intervention_required: list[str] = Field(default_factory=list)
    intervention_records: list[RecoveryInterventionRecord] = Field(default_factory=list)
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
