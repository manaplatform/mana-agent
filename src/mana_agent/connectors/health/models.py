"""Typed contracts for connector health, probes, receipts, and incidents."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stable_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ConnectorHealthState(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    OFFLINE = "offline"
    AUTH_REQUIRED = "auth_required"
    RATE_LIMITED = "rate_limited"
    DISABLED = "disabled"


class HealthReasonCode(str, Enum):
    NONE = "NONE"
    AUTH_EXPIRED = "AUTH_EXPIRED"
    AUTH_REVOKED = "AUTH_REVOKED"
    TOKEN_REFRESH_FAILED = "TOKEN_REFRESH_FAILED"
    CONNECTION_REFUSED = "CONNECTION_REFUSED"
    CONNECTION_TIMEOUT = "CONNECTION_TIMEOUT"
    INGRESS_STALLED = "INGRESS_STALLED"
    EGRESS_FAILED = "EGRESS_FAILED"
    ACK_TIMEOUT = "ACK_TIMEOUT"
    SUBSCRIPTION_MISSING = "SUBSCRIPTION_MISSING"
    SUBSCRIPTION_EXPIRED = "SUBSCRIPTION_EXPIRED"
    WEBHOOK_UNREACHABLE = "WEBHOOK_UNREACHABLE"
    RATE_LIMITED = "RATE_LIMITED"
    REMOTE_SERVICE_DEGRADED = "REMOTE_SERVICE_DEGRADED"
    RECONNECT_FAILED = "RECONNECT_FAILED"
    PROBE_FAILED = "PROBE_FAILED"
    PROCESS_ONLY_ALIVE = "PROCESS_ONLY_ALIVE"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    DISABLED = "DISABLED"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    STARTUP_PENDING = "STARTUP_PENDING"
    RECOVERY_IN_PROGRESS = "RECOVERY_IN_PROGRESS"
    PARTIAL_CAPABILITY_FAILURE = "PARTIAL_CAPABILITY_FAILURE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ProbeCategory(str, Enum):
    AUTH = "auth"
    CONNECTIVITY = "connectivity"
    INGRESS = "ingress"
    EGRESS = "egress"
    SUBSCRIPTION = "subscription"
    ACKNOWLEDGEMENT = "acknowledgement"


class SyntheticProbeMode(str, Enum):
    PASSIVE = "passive"
    SAFE_ENDPOINT = "safe_endpoint"
    TEST_CHANNEL = "test_channel"
    ACTIVE = "active"


class ProbeOutcome(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RATE_LIMITED = "rate_limited"
    UNSUPPORTED = "unsupported"


class CapabilitySignal(str, Enum):
    UNKNOWN = "unknown"
    OK = "ok"
    FAILED = "failed"
    DEGRADED = "degraded"
    NOT_APPLICABLE = "not_applicable"


class DeliveryState(str, Enum):
    QUEUED = "queued"
    SUBMITTED = "submitted"
    PROVIDER_ACCEPTED = "provider_accepted"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    UNKNOWN = "unknown"


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RecoveryActionKind(str, Enum):
    TRANSPORT_RECONNECT = "transport_reconnect"
    WEBSOCKET_RECREATE = "websocket_recreate"
    SUBSCRIPTION_RENEW = "subscription_renew"
    WEBHOOK_REREGISTER = "webhook_reregister"
    TOKEN_REFRESH = "token_refresh"
    CLIENT_RECREATE = "client_recreate"
    POLLER_RESTART = "poller_restart"
    CONSUMER_RESTART = "consumer_restart"
    CONNECTION_POOL_RESET = "connection_pool_reset"
    NONE = "none"


class ConnectorHealthCapabilities(StrictModel):
    auth: bool = True
    connectivity: bool = True
    ingress: bool = False
    egress: bool = False
    subscriptions: bool = False
    acknowledgements: bool = False


class PathSignals(StrictModel):
    """Independent path signals; process aliveness never implies connector health."""

    runtime_alive: bool = False
    transport_connected: bool = False
    authenticated: CapabilitySignal = CapabilitySignal.UNKNOWN
    ingress_operational: CapabilitySignal = CapabilitySignal.UNKNOWN
    egress_operational: CapabilitySignal = CapabilitySignal.UNKNOWN
    subscription_operational: CapabilitySignal = CapabilitySignal.UNKNOWN
    acknowledgements_operational: CapabilitySignal = CapabilitySignal.UNKNOWN


class ProbeResult(StrictModel):
    category: ProbeCategory
    outcome: ProbeOutcome
    reason_code: HealthReasonCode = HealthReasonCode.NONE
    latency_ms: float | None = Field(default=None, ge=0)
    message: str = ""
    checked_at: datetime = Field(default_factory=utc_now)
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("details")
    @classmethod
    def no_secret_keys(cls, value: dict[str, Any]) -> dict[str, Any]:
        from mana_agent.utils.redaction import redact_secrets

        return redact_secrets(value) if value else {}


class DeliveryReceipt(StrictModel):
    message_id: str
    connector_id: str
    provider_message_id: str = ""
    state: DeliveryState = DeliveryState.UNKNOWN
    submitted_at: datetime | None = None
    acknowledged_at: datetime | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    failure_reason: str = ""
    reason_code: HealthReasonCode = HealthReasonCode.NONE


class RecoveryAttempt(StrictModel):
    recovery_attempt_id: str = Field(default_factory=lambda: stable_id("recovery"))
    connector_id: str
    action: RecoveryActionKind
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    success: bool | None = None
    attempt_number: int = Field(default=1, ge=1)
    reason_code: HealthReasonCode = HealthReasonCode.NONE
    message: str = ""
    requires_transactional_policy: bool = False
    requires_human: bool = False


class IncidentEvent(StrictModel):
    event_id: str = Field(default_factory=lambda: stable_id("incident_event"))
    incident_id: str
    connector_id: str
    event_type: str
    occurred_at: datetime = Field(default_factory=utc_now)
    reason_code: HealthReasonCode = HealthReasonCode.NONE
    recovery_attempt_id: str = ""
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class ConnectorIncident(StrictModel):
    incident_id: str = Field(default_factory=lambda: stable_id("incident"))
    connector_id: str
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    opening_state: ConnectorHealthState
    closing_state: ConnectorHealthState | None = None
    opening_reason: HealthReasonCode = HealthReasonCode.NONE
    recovered: bool = False
    events: list[IncidentEvent] = Field(default_factory=list)
    recovery_attempt_ids: list[str] = Field(default_factory=list)

    @property
    def open(self) -> bool:
        return self.ended_at is None


class ConnectorSLOMetrics(StrictModel):
    connector_id: str
    availability: float | None = Field(default=None, ge=0, le=1)
    successful_probe_ratio: float | None = Field(default=None, ge=0, le=1)
    delivery_success_rate: float | None = Field(default=None, ge=0, le=1)
    ack_latency_ms: float | None = Field(default=None, ge=0)
    reconnect_count: int = Field(default=0, ge=0)
    mean_recovery_seconds: float | None = Field(default=None, ge=0)
    failure_count: int = Field(default=0, ge=0)
    degraded_duration_seconds: float = Field(default=0, ge=0)
    offline_duration_seconds: float = Field(default=0, ge=0)
    observation_count: int = Field(default=0, ge=0)
    insufficient_data: bool = True


class ConnectorHealthReport(StrictModel):
    connector_id: str
    connector_type: str
    state: ConnectorHealthState = ConnectorHealthState.UNKNOWN
    checked_at: datetime = Field(default_factory=utc_now)
    last_healthy_at: datetime | None = None
    last_failure_at: datetime | None = None
    consecutive_failures: int = Field(default=0, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    signals: PathSignals = Field(default_factory=PathSignals)
    auth: CapabilitySignal = CapabilitySignal.UNKNOWN
    ingress: CapabilitySignal = CapabilitySignal.UNKNOWN
    egress: CapabilitySignal = CapabilitySignal.UNKNOWN
    subscriptions: CapabilitySignal = CapabilitySignal.UNKNOWN
    acknowledgements: CapabilitySignal = CapabilitySignal.UNKNOWN
    reconnect: CapabilitySignal = CapabilitySignal.UNKNOWN
    recovery_attempt: RecoveryAttempt | None = None
    current_incident_id: str = ""
    circuit_state: CircuitState = CircuitState.CLOSED
    reason_code: HealthReasonCode = HealthReasonCode.NONE
    message: str = ""
    synthetic_probe_mode: SyntheticProbeMode = SyntheticProbeMode.PASSIVE
    probe_results: list[ProbeResult] = Field(default_factory=list)
    metrics: ConnectorSLOMetrics | None = None
    disabled: bool = False

    def display_state(self) -> str:
        return self.state.value


class ConnectorHealthChanged(StrictModel):
    event_id: str = Field(default_factory=lambda: stable_id("health_event"))
    connector_id: str
    previous_state: ConnectorHealthState
    state: ConnectorHealthState
    reason_code: HealthReasonCode = HealthReasonCode.NONE
    incident_id: str = ""
    severity: AlertSeverity = AlertSeverity.INFO
    occurred_at: datetime = Field(default_factory=utc_now)
    message: str = ""
    dedupe_key: str = ""


class ConnectorHealthSnapshot(StrictModel):
    """Persisted connector health state for crash/restart recovery."""

    connector_id: str
    connector_type: str
    report: ConnectorHealthReport
    circuit_state: CircuitState = CircuitState.CLOSED
    circuit_opened_at: datetime | None = None
    circuit_failure_count: int = Field(default=0, ge=0)
    recovery_attempts: int = Field(default=0, ge=0)
    next_probe_at: datetime | None = None
    next_recovery_at: datetime | None = None
    rate_limited_until: datetime | None = None
    shutting_down: bool = False
    updated_at: datetime = Field(default_factory=utc_now)


class HealthCheckPlan(StrictModel):
    connector_id: str
    categories: list[ProbeCategory] = Field(default_factory=list)
    synthetic_mode: SyntheticProbeMode = SyntheticProbeMode.PASSIVE
    force: bool = False


# Compatibility aliases used by recovery decision trees
TerminalAuthStates = frozenset(
    {ConnectorHealthState.AUTH_REQUIRED, ConnectorHealthState.DISABLED}
)
UnavailableStates = frozenset(
    {
        ConnectorHealthState.OFFLINE,
        ConnectorHealthState.RECOVERING,
        ConnectorHealthState.AUTH_REQUIRED,
        ConnectorHealthState.UNKNOWN,
    }
)
DegradedCapableStates = frozenset(
    {ConnectorHealthState.HEALTHY, ConnectorHealthState.DEGRADED, ConnectorHealthState.RATE_LIMITED}
)
"""States that may still execute work when the required capability remains OK."""

AlertSeverityByState: dict[ConnectorHealthState, AlertSeverity] = {
    ConnectorHealthState.HEALTHY: AlertSeverity.INFO,
    ConnectorHealthState.UNKNOWN: AlertSeverity.INFO,
    ConnectorHealthState.DISABLED: AlertSeverity.INFO,
    ConnectorHealthState.DEGRADED: AlertSeverity.WARNING,
    ConnectorHealthState.RECOVERING: AlertSeverity.WARNING,
    ConnectorHealthState.RATE_LIMITED: AlertSeverity.WARNING,
    ConnectorHealthState.OFFLINE: AlertSeverity.ERROR,
    ConnectorHealthState.AUTH_REQUIRED: AlertSeverity.CRITICAL,
}
