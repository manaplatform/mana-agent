"""Connector Health and Self-Healing subsystem.

Health is independent of process/gateway aliveness. A running connector process
never implies HEALTHY — authentication, transport, ingress, egress,
subscriptions, and acknowledgements are probed separately.
"""

from __future__ import annotations

from typing import Any

from .adapters import (
    GmailHealthAdapter,
    TelegramHealthAdapter,
    discover_gmail_adapters,
    discover_telegram_adapter,
)
from .circuit_breaker import CircuitBreaker
from .config import ConnectorHealthConfig, load_connector_health_config
from .contracts import HealthProbeable
from .events import HealthEventRouter
from .hitl_bridge import ConnectorHitlBridge, build_auth_intervention_request
from .manager import ConnectorHealthManager, get_health_manager, reset_health_manager
from .models import (
    AlertSeverity,
    CapabilitySignal,
    CircuitState,
    ConnectorHealthCapabilities,
    ConnectorHealthChanged,
    ConnectorHealthReport,
    ConnectorHealthState,
    ConnectorIncident,
    ConnectorSLOMetrics,
    DeliveryReceipt,
    DeliveryState,
    HealthReasonCode,
    PathSignals,
    ProbeCategory,
    ProbeOutcome,
    ProbeResult,
    RecoveryActionKind,
    RecoveryAttempt,
    SyntheticProbeMode,
)
from .supervisor_bridge import ConnectorSupervisorBridge
from .transactional_bridge import ConnectorTransactionalBridge


def bootstrap_health_manager(
    *,
    manager: ConnectorHealthManager | None = None,
    include_gmail: bool = True,
    include_telegram: bool = True,
    hitl: ConnectorHitlBridge | None = None,
    supervisor_bridge: ConnectorSupervisorBridge | None = None,
    transactional: ConnectorTransactionalBridge | None = None,
    runtime_alive: bool = True,
) -> ConnectorHealthManager:
    """Create/start the health manager and register real connectors."""
    health = manager or get_health_manager()
    if hitl is not None:
        health.hitl_callback = lambda connector_id, reason, message: hitl.request_auth_intervention(
            connector_id=connector_id,
            connector_type=connector_id.split(":", 1)[0],
            reason_code=reason,
            message=message,
        )
    if transactional is not None:
        health.transactional_callback = transactional.authorize
    if supervisor_bridge is not None:
        health.supervisor_callback = supervisor_bridge.on_health_change
    if include_gmail:
        for adapter in discover_gmail_adapters(runtime_alive=runtime_alive):
            health.register(adapter)
    if include_telegram:
        adapter = discover_telegram_adapter()
        if adapter is not None:
            health.register(adapter)
    return health


def format_status_report(report: ConnectorHealthReport) -> str:
    """Human-readable CLI status block."""
    lines = [
        report.connector_type.title() if report.connector_type != "gmail" else f"Gmail ({report.connector_id})",
        f"State: {report.state.value.upper()}",
        f"Authentication: {_signal_label(report.auth)}",
        f"Ingress: {_signal_label(report.ingress)}",
        f"Egress: {_signal_label(report.egress)}",
    ]
    if report.subscriptions is not CapabilitySignal.NOT_APPLICABLE:
        lines.append(f"Subscriptions: {_signal_label(report.subscriptions)}")
    signals = report.signals
    lines.append(f"Transport: {'connected' if signals.transport_connected else 'disconnected'}")
    lines.append(f"Runtime: {'alive' if signals.runtime_alive else 'stopped'}")
    if report.reason_code is not HealthReasonCode.NONE:
        lines.append(f"Reason: {report.reason_code.value} — {report.message}")
    if report.checked_at:
        lines.append(f"Last probe: {report.checked_at.isoformat()}")
    if report.latency_ms is not None:
        lines.append(f"Latency: {report.latency_ms:.0f}ms")
    if report.recovery_attempt is not None:
        attempt = report.recovery_attempt
        lines.append(
            f"Recovery: {attempt.action.value} attempt={attempt.attempt_number} "
            f"success={attempt.success}"
        )
    if report.current_incident_id:
        lines.append(f"Incident: {report.current_incident_id}")
    if report.circuit_state is not CircuitState.CLOSED:
        lines.append(f"Circuit: {report.circuit_state.value}")
    return "\n".join(lines)


def _signal_label(signal: CapabilitySignal) -> str:
    if signal is CapabilitySignal.OK:
        return "OK"
    if signal is CapabilitySignal.FAILED:
        return "FAILED"
    if signal is CapabilitySignal.DEGRADED:
        return "DEGRADED"
    if signal is CapabilitySignal.NOT_APPLICABLE:
        return "N/A"
    return "UNKNOWN"


__all__ = [
    "AlertSeverity",
    "CapabilitySignal",
    "CircuitBreaker",
    "CircuitState",
    "ConnectorHealthCapabilities",
    "ConnectorHealthChanged",
    "ConnectorHealthConfig",
    "ConnectorHealthManager",
    "ConnectorHealthReport",
    "ConnectorHealthState",
    "ConnectorHitlBridge",
    "ConnectorIncident",
    "ConnectorSLOMetrics",
    "ConnectorSupervisorBridge",
    "ConnectorTransactionalBridge",
    "DeliveryReceipt",
    "DeliveryState",
    "GmailHealthAdapter",
    "HealthEventRouter",
    "HealthProbeable",
    "HealthReasonCode",
    "PathSignals",
    "ProbeCategory",
    "ProbeOutcome",
    "ProbeResult",
    "RecoveryActionKind",
    "RecoveryAttempt",
    "SyntheticProbeMode",
    "TelegramHealthAdapter",
    "bootstrap_health_manager",
    "build_auth_intervention_request",
    "discover_gmail_adapters",
    "discover_telegram_adapter",
    "format_status_report",
    "get_health_manager",
    "load_connector_health_config",
    "reset_health_manager",
]
