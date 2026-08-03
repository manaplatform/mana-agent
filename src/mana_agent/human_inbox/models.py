"""Typed contracts for the durable human-in-the-loop inbox."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mana_agent.utils.redaction import redact_secrets


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class InboxRequestType(str, Enum):
    APPROVAL = "approval"
    CLARIFICATION = "clarification"
    NOTICE = "notice"


class InboxStatus(str, Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    APPROVED = "approved"
    DENIED = "denied"
    ANSWERED = "answered"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    DELEGATED = "delegated"
    SUPERSEDED = "superseded"
    RECORDED = "recorded"


UNRESOLVED_STATUSES = frozenset({InboxStatus.PENDING, InboxStatus.DELIVERED})
TERMINAL_STATUSES = frozenset(set(InboxStatus) - set(UNRESOLVED_STATUSES))


class ReviewerType(str, Enum):
    PERSON = "person"
    GROUP = "group"
    ROLE = "role"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ResponseOperation(str, Enum):
    APPROVE = "approve"
    DENY = "deny"
    ANSWER = "answer"


class ExpiryBehavior(str, Enum):
    REMAIN_BLOCKED = "remain_blocked"
    CANCEL_BRANCH = "cancel_branch"
    DENY_BY_DEFAULT = "deny_by_default"
    REQUEST_REPLANNING = "request_replanning"
    ESCALATE = "escalate"


class ExpectedResponseType(str, Enum):
    TEXT = "text"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    CHOICE = "choice"
    MULTI_CHOICE = "multi_choice"
    OBJECT = "object"


class ReviewerAssignment(StrictModel):
    reviewer_type: ReviewerType
    reviewer_id: str = Field(min_length=1)


class ClarificationValidation(StrictModel):
    minimum: float | None = None
    maximum: float | None = None
    min_length: int | None = Field(default=None, ge=0)
    max_length: int | None = Field(default=None, ge=0)
    pattern: str = ""

    @model_validator(mode="after")
    def validate_ranges(self) -> "ClarificationValidation":
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("clarification minimum cannot exceed maximum")
        if self.min_length is not None and self.max_length is not None and self.min_length > self.max_length:
            raise ValueError("clarification min_length cannot exceed max_length")
        if self.pattern:
            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise ValueError("clarification pattern is invalid") from exc
        return self


class ClarificationField(StrictModel):
    field_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    expected_type: ExpectedResponseType = ExpectedResponseType.TEXT
    choices: list[str] = Field(default_factory=list)
    validation: ClarificationValidation = Field(default_factory=ClarificationValidation)
    allow_free_form: bool = True
    sensitive: bool = False
    required: bool = True

    @model_validator(mode="after")
    def validate_choices(self) -> "ClarificationField":
        if self.expected_type in {ExpectedResponseType.CHOICE, ExpectedResponseType.MULTI_CHOICE} and not self.choices:
            raise ValueError("choice clarification fields require explicit choices")
        if not self.allow_free_form and not self.choices:
            raise ValueError("non-free-form clarification fields require explicit choices")
        return self


class EscalationPolicy(StrictModel):
    expiry_behavior: ExpiryBehavior = ExpiryBehavior.REMAIN_BLOCKED
    target: ReviewerAssignment | None = None
    max_escalations: int = Field(default=0, ge=0)
    escalation_ttl_seconds: int = Field(default=3600, ge=60)

    @model_validator(mode="after")
    def escalation_target_is_explicit(self) -> "EscalationPolicy":
        if self.expiry_behavior is ExpiryBehavior.ESCALATE and self.target is None:
            raise ValueError("escalation requires an explicit target")
        if self.expiry_behavior is ExpiryBehavior.ESCALATE and self.max_escalations < 1:
            raise ValueError("escalation requires a positive escalation limit")
        return self


class ReminderPolicy(StrictModel):
    interval_seconds: int = Field(default=3600, ge=60)
    max_reminders: int = Field(default=0, ge=0)


class HumanResponse(StrictModel):
    operation: ResponseOperation
    answer: dict[str, Any] = Field(default_factory=dict)
    comment: str = ""
    submitted_at: datetime = Field(default_factory=utc_now)


class InboxItem(StrictModel):
    schema_version: int = Field(default=1, ge=1)
    inbox_item_id: str = Field(default_factory=lambda: f"inbox_{uuid4().hex}")
    request_type: InboxRequestType
    status: InboxStatus = InboxStatus.PENDING
    tenant_id: str = "local"
    project_id: str = "local"
    task_id: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    parent_task_id: str | None = None
    checkpoint_id: str = ""
    execution_attempt_id: str = ""
    policy_decision_id: str = ""
    permission_request_id: str = ""
    action_intent_id: str = ""
    action_digest: str = ""
    requested_by_agent_id: str = Field(min_length=1)
    assigned_reviewer_type: ReviewerType
    assigned_reviewer_id: str = Field(min_length=1)
    eligible_reviewer_ids: list[str] = Field(default_factory=list)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    requested_fields: list[ClarificationField] = Field(default_factory=list)
    allowed_responses: list[ResponseOperation] = Field(default_factory=list)
    editable_parameters: list[str] = Field(default_factory=list)
    minimal_context: dict[str, Any] = Field(default_factory=dict)
    protected_context_ref: str = ""
    disclosed_fields: list[str] = Field(default_factory=list)
    reversibility: str = "unknown"
    other_work_continues: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    delivered_at: datetime | None = None
    responded_at: datetime | None = None
    expires_at: datetime
    escalation_policy: EscalationPolicy = Field(default_factory=EscalationPolicy)
    reminder_policy: ReminderPolicy = Field(default_factory=ReminderPolicy)
    reminder_count: int = Field(default=0, ge=0)
    last_reminded_at: datetime | None = None
    response: HumanResponse | None = None
    response_actor_id: str = ""
    response_channel: str = ""
    response_signature: str = ""
    version: int = Field(default=0, ge=0)
    idempotency_key: str = Field(min_length=1)
    deduplication_key: str = Field(min_length=1)
    response_idempotency_keys: list[str] = Field(default_factory=list)
    response_idempotency_digests: dict[str, str] = Field(default_factory=dict)
    token_nonce_hashes: list[str] = Field(default_factory=list)
    delegated_from_item_id: str = ""
    delegated_to_item_id: str = ""
    superseded_by_item_id: str = ""
    configuration_error: str = ""
    resume_claim_id: str = ""
    resume_claimed_at: datetime | None = None
    resume_completed_at: datetime | None = None
    execution_claim_id: str = ""
    execution_claimed_at: datetime | None = None
    execution_completed_at: datetime | None = None
    execution_result_digest: str = ""

    @model_validator(mode="after")
    def validate_contract(self) -> "InboxItem":
        if self.expires_at <= self.created_at:
            raise ValueError("inbox item expiration must follow creation")
        if self.request_type is InboxRequestType.APPROVAL:
            if self.requested_fields:
                raise ValueError("binary approvals cannot request clarification fields")
            if set(self.allowed_responses) - {ResponseOperation.APPROVE, ResponseOperation.DENY}:
                raise ValueError("approval requests allow only approve or deny")
            if not {ResponseOperation.APPROVE, ResponseOperation.DENY}.issubset(self.allowed_responses):
                raise ValueError("binary approvals must allow both approve and deny")
        elif self.request_type is InboxRequestType.CLARIFICATION:
            if self.allowed_responses != [ResponseOperation.ANSWER]:
                raise ValueError("clarification requests must allow only answer")
            if not self.requested_fields:
                raise ValueError("clarification requests require at least one typed field")
        else:
            if self.status is not InboxStatus.RECORDED:
                raise ValueError("notice requests must be terminal recorded items")
            if self.allowed_responses or self.requested_fields:
                raise ValueError("notice requests cannot accept human responses")
        field_ids = [field.field_id for field in self.requested_fields]
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("clarification field IDs must be unique")
        if self.status in TERMINAL_STATUSES and self.status not in {
            InboxStatus.EXPIRED, InboxStatus.CANCELLED, InboxStatus.DELEGATED, InboxStatus.SUPERSEDED,
            InboxStatus.RECORDED,
        } and self.response is None:
            raise ValueError("responded inbox items require a structured response")
        for name in (
            "created_at", "delivered_at", "responded_at", "expires_at",
            "last_reminded_at", "resume_claimed_at", "resume_completed_at",
            "execution_claimed_at", "execution_completed_at",
        ):
            value = getattr(self, name)
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{name} must be timezone-aware")
        return self

    def card(self) -> dict[str, Any]:
        """Return the minimal-disclosure projection safe for UI and notifications."""
        return redact_secrets({
            "inbox_item_id": self.inbox_item_id,
            "request_type": self.request_type.value,
            "status": self.status.value,
            "task_id": self.task_id,
            "branch_id": self.branch_id,
            "project_id": self.project_id,
            "parent_task_id": self.parent_task_id,
            "title": self.title,
            "summary": self.summary,
            "risk_level": self.risk_level.value,
            "requested_fields": [field.model_dump(mode="json") for field in self.requested_fields],
            "allowed_responses": [item.value for item in self.allowed_responses],
            "editable_parameters": self.editable_parameters,
            "minimal_context": self.minimal_context,
            "disclosed_fields": self.disclosed_fields,
            "reversibility": self.reversibility,
            "other_work_continues": self.other_work_continues,
            "created_at": self.created_at.isoformat(),
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "responded_at": self.responded_at.isoformat() if self.responded_at else None,
            "expires_at": self.expires_at.isoformat(),
            "assigned_reviewer_type": self.assigned_reviewer_type.value,
            "assigned_reviewer_id": self.assigned_reviewer_id,
            "delivery_status": "delivered" if self.delivered_at else "pending",
            "execution_status": (
                "completed"
                if self.execution_completed_at
                else "claimed"
                if self.execution_claim_id
                else "not_started"
            ),
            "configuration_error": self.configuration_error,
            "version": self.version,
        })


class InboxRequest(StrictModel):
    request_type: InboxRequestType
    tenant_id: str = "local"
    project_id: str = "local"
    task_id: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    parent_task_id: str | None = None
    checkpoint_id: str = ""
    execution_attempt_id: str = ""
    policy_decision_id: str = ""
    permission_request_id: str = ""
    action_intent_id: str = ""
    action_digest: str = ""
    requested_by_agent_id: str = Field(min_length=1)
    reviewer: ReviewerAssignment
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    requested_fields: list[ClarificationField] = Field(default_factory=list)
    allowed_responses: list[ResponseOperation] = Field(default_factory=list)
    editable_parameters: list[str] = Field(default_factory=list)
    minimal_context: dict[str, Any] = Field(default_factory=dict)
    protected_context: dict[str, Any] = Field(default_factory=dict)
    disclosed_fields: list[str] = Field(default_factory=list)
    reversibility: str = "unknown"
    other_work_continues: bool = True
    expires_at: datetime
    escalation_policy: EscalationPolicy = Field(default_factory=EscalationPolicy)
    reminder_policy: ReminderPolicy = Field(default_factory=ReminderPolicy)
    idempotency_key: str = Field(min_length=1)
    deduplication_key: str = Field(min_length=1)


class ResponseSubmission(StrictModel):
    inbox_item_id: str = Field(min_length=1)
    operation: ResponseOperation
    actor_id: str = Field(min_length=1)
    channel: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    answer: dict[str, Any] = Field(default_factory=dict)
    comment: str = ""
    signed_token: str = ""
    expected_version: int | None = Field(default=None, ge=0)
    current_action_digest: str = ""


class DeliveryAttempt(StrictModel):
    delivery_attempt_id: str = Field(default_factory=lambda: f"delivery_{uuid4().hex}")
    inbox_item_id: str
    adapter: str
    destination: str
    attempt_number: int = Field(ge=1)
    status: Literal["attempted", "delivered", "failed"]
    error: str = ""
    timestamp: datetime = Field(default_factory=utc_now)
    external_message_id: str = ""


class InboxAuditEvent(StrictModel):
    audit_event_id: str = Field(default_factory=lambda: f"inbox_event_{uuid4().hex}")
    sequence: int = Field(default=0, ge=0)
    event_type: str
    inbox_item_id: str
    task_id: str
    branch_id: str
    policy_decision_id: str = ""
    action_intent_id: str = ""
    reviewer_id: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    details: dict[str, Any] = Field(default_factory=dict)


class InboxQuery(StrictModel):
    statuses: set[InboxStatus] = Field(default_factory=set)
    reviewer_id: str = ""
    role: str = ""
    group: str = ""
    task_id: str = ""
    branch_id: str = ""
    request_type: InboxRequestType | None = None
    tenant_id: str = ""
    project_id: str = ""


class ReconciliationReport(StrictModel):
    scanned: int = 0
    waiting_restored: list[str] = Field(default_factory=list)
    resumed: list[str] = Field(default_factory=list)
    cancelled: list[str] = Field(default_factory=list)
    superseded: list[str] = Field(default_factory=list)
    orphaned_items: list[str] = Field(default_factory=list)
    waiting_without_item: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class AgentInboxObservation(StrictModel):
    """Narrow task-scoped projection available to the requesting agent."""

    inbox_item_id: str
    task_id: str
    branch_id: str
    status: InboxStatus
    request_type: InboxRequestType
    response: HumanResponse | None = None
    resume_completed: bool = False


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(redact_secrets(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
