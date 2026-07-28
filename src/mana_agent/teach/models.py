"""Versioned contracts for local Teach Mode recordings and flows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from mana_agent.compat import StrEnum


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TeachError(RuntimeError):
    """Base class for actionable Teach Mode failures."""


class SessionState(StrEnum):
    CREATED = "created"
    RECORDING = "recording"
    PAUSED = "paused"
    COMPILING = "compiling"
    AWAITING_REVIEW = "awaiting_review"
    REPLAYING = "replaying"
    REPAIRING = "repairing"
    VERIFIED = "verified"
    SAVED = "saved"
    CANCELLED = "cancelled"
    FAILED = "failed"


ALLOWED_TRANSITIONS: dict[SessionState, set[SessionState]] = {
    SessionState.CREATED: {SessionState.RECORDING, SessionState.CANCELLED},
    SessionState.RECORDING: {SessionState.PAUSED, SessionState.COMPILING, SessionState.CANCELLED, SessionState.FAILED},
    SessionState.PAUSED: {SessionState.RECORDING, SessionState.COMPILING, SessionState.CANCELLED, SessionState.FAILED},
    SessionState.COMPILING: {SessionState.AWAITING_REVIEW, SessionState.FAILED},
    SessionState.AWAITING_REVIEW: {SessionState.REPLAYING, SessionState.SAVED, SessionState.CANCELLED, SessionState.FAILED},
    SessionState.REPLAYING: {SessionState.REPAIRING, SessionState.VERIFIED, SessionState.AWAITING_REVIEW, SessionState.FAILED},
    SessionState.REPAIRING: {SessionState.REPLAYING, SessionState.AWAITING_REVIEW, SessionState.FAILED},
    SessionState.VERIFIED: {SessionState.SAVED, SessionState.REPLAYING},
    SessionState.SAVED: {SessionState.REPLAYING},
    SessionState.CANCELLED: set(),
    SessionState.FAILED: {SessionState.AWAITING_REVIEW, SessionState.CANCELLED},
}


class EventSource(StrEnum):
    ACCESSIBILITY = "accessibility"
    BROWSER = "browser"
    KEYBOARD = "keyboard"
    POINTER = "pointer"
    FILESYSTEM = "filesystem"
    APPLICATION = "application"
    VOICE = "voice"


class SelectorCandidate(BaseModel):
    type: str = Field(min_length=1, max_length=64)
    value: Any
    confidence: float = Field(ge=0, le=1)
    successes: int = Field(default=0, ge=0)
    failures: int = Field(default=0, ge=0)
    last_verified_at: datetime | None = None


class EventApplication(BaseModel):
    id: str = ""
    name: str = ""
    version: str | None = None


class EventTarget(BaseModel):
    role: str | None = None
    name: str | None = None
    label: str | None = None
    automation_id: str | None = None
    hierarchy: list[str] = Field(default_factory=list)
    selectors: list[SelectorCandidate] = Field(default_factory=list)


class RelativePosition(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class RecordedEvent(BaseModel):
    schema_version: int = 1
    event_id: str = Field(default_factory=lambda: f"evt_{uuid4().hex}")
    session_id: str
    timestamp: datetime = Field(default_factory=utc_now)
    source: EventSource
    action: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    application: EventApplication = Field(default_factory=EventApplication)
    target: EventTarget = Field(default_factory=EventTarget)
    context: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    fallback_position: RelativePosition | None = None
    sensitive: bool = False
    redactions: list[str] = Field(default_factory=list)


class Explanation(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    text: str = Field(min_length=1, max_length=4000)
    source: Literal["typed", "voice"] = "typed"


class AuditEntry(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    action: str
    detail: str = ""


class ReplayAttempt(BaseModel):
    run_id: str = Field(default_factory=lambda: f"teachrun_{uuid4().hex}")
    mode: Literal["dry_run", "guided", "normal"]
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    status: Literal["running", "verified", "unverified", "failed"] = "running"
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class TeachSession(BaseModel):
    schema_version: int = 1
    id: str = Field(default_factory=lambda: f"teach_{uuid4().hex}")
    task_name: str = Field(min_length=1, max_length=240)
    state: SessionState = SessionState.CREATED
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    paused_at: datetime | None = None
    resumed_at: datetime | None = None
    stopped_at: datetime | None = None
    cancelled_at: datetime | None = None
    active_applications: list[str] = Field(default_factory=list)
    active_windows: list[str] = Field(default_factory=list)
    explanations: list[Explanation] = Field(default_factory=list)
    sensitive_fields: list[str] = Field(default_factory=list)
    detected_inputs: list[str] = Field(default_factory=list)
    proposed_verification_rules: list[dict[str, Any]] = Field(default_factory=list)
    permission_grants: list[str] = Field(default_factory=list)
    recorder_capabilities: list[str] = Field(default_factory=list)
    platform_metadata: dict[str, Any] = Field(default_factory=dict)
    compilation_status: str = "not_started"
    replay_attempts: list[ReplayAttempt] = Field(default_factory=list)
    generated_flow_id: str | None = None
    audit_trail: list[AuditEntry] = Field(default_factory=list)
    raw_event_count: int = 0
    normalized_event_count: int = 0
    monitor_pid: int | None = None

    def transition(self, target: SessionState, detail: str = "") -> None:
        if target not in ALLOWED_TRANSITIONS[self.state]:
            raise TeachError(f"Invalid Teach Mode transition: {self.state.value} -> {target.value}.")
        self.state = target
        now = utc_now()
        if target == SessionState.RECORDING:
            if self.started_at is None:
                self.started_at = now
            else:
                self.resumed_at = now
        elif target == SessionState.PAUSED:
            self.paused_at = now
        elif target == SessionState.COMPILING:
            self.stopped_at = now
        elif target == SessionState.CANCELLED:
            self.cancelled_at = now
        self.audit_trail.append(AuditEntry(action=f"state.{target.value}", detail=detail))


class FlowInput(BaseModel):
    type: Literal["string", "date", "email", "path", "secret", "integer", "boolean"] = "string"
    required: bool = True
    default: Any | None = None
    secret: bool = False
    share: bool = True
    description: str = ""

    @model_validator(mode="after")
    def protect_secrets(self) -> "FlowInput":
        if self.secret:
            self.type = "secret"
            self.default = None
            self.share = False
        return self


class FlowStep(BaseModel):
    id: str
    action: str
    with_: dict[str, Any] = Field(default_factory=dict, alias="with")
    depends_on: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1, ge=0, le=1)
    requires_review: bool = False
    requires_confirmation: bool = False
    selectors: list[SelectorCandidate] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)
    checkpoint: bool = False

    model_config = {"populate_by_name": True}


class VerificationRule(BaseModel):
    id: str
    type: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    step_id: str | None = None
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    required: bool = True


class FlowStatistics(BaseModel):
    successful_replays: int = 0
    verified_replays: int = 0
    failed_replays: int = 0
    repair_count: int = 0
    last_verified_at: datetime | None = None
    total_replay_seconds: float = 0


class ManaFlow(BaseModel):
    schema_version: int = 1
    id: str
    version: int = Field(default=1, ge=1)
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    source_session_id: str | None = None
    inputs: dict[str, FlowInput] = Field(default_factory=dict)
    permissions: list[str] = Field(default_factory=list)
    supported_platforms: list[str] = Field(default_factory=list)
    required_applications: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    steps: list[FlowStep] = Field(default_factory=list)
    verify: list[VerificationRule] = Field(default_factory=list)
    status: Literal["draft", "verified", "active", "imported_pending"] = "draft"
    statistics: FlowStatistics = Field(default_factory=FlowStatistics)

    @model_validator(mode="after")
    def validate_graph(self) -> "ManaFlow":
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Flow step IDs must be unique.")
        known: set[str] = set()
        for step in self.steps:
            if any(dependency not in known for dependency in step.depends_on):
                raise ValueError(f"Step {step.id} has a missing or forward dependency.")
            known.add(step.id)
        return self


class StepResult(BaseModel):
    step_id: str
    status: Literal["planned", "completed", "waiting_confirmation", "failed", "skipped"]
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class ReplayResult(BaseModel):
    run_id: str = Field(default_factory=lambda: f"teachrun_{uuid4().hex}")
    flow_id: str
    flow_version: int
    mode: Literal["dry_run", "guided", "normal"]
    verification_status: Literal["verified", "unverified", "failed"]
    steps: list[StepResult] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)


class FlowCard(BaseModel):
    title: str
    demonstration_duration_seconds: int
    application_count: int
    action_count: int
    verified_replays: int
    successful_replays: int
    success_rate: float
    estimated_minutes_saved_per_run: int
    estimated_runs_per_week: float = 1
    visibility: Literal["private", "export_ready"] = "private"
    share_copy: str
