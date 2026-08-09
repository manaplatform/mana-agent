"""Deterministic derivation of connector health state from path signals."""

from __future__ import annotations

from .models import (
    CapabilitySignal,
    ConnectorHealthState,
    HealthReasonCode,
    PathSignals,
)


def _failed(signal: CapabilitySignal) -> bool:
    return signal is CapabilitySignal.FAILED


def _degraded(signal: CapabilitySignal) -> bool:
    return signal is CapabilitySignal.DEGRADED


def derive_state(
    signals: PathSignals,
    *,
    disabled: bool = False,
    rate_limited: bool = False,
    circuit_open: bool = False,
    recovering: bool = False,
    consecutive_failures: int = 0,
    failure_threshold: int = 3,
    shutting_down: bool = False,
) -> tuple[ConnectorHealthState, HealthReasonCode, str]:
    """Derive health state from independent path signals.

    A running process/gateway alone never yields HEALTHY.
    """
    if shutting_down:
        return ConnectorHealthState.DISABLED, HealthReasonCode.SHUTTING_DOWN, "Connector is shutting down"
    if disabled:
        return ConnectorHealthState.DISABLED, HealthReasonCode.DISABLED, "Connector is disabled"

    auth = signals.authenticated
    if auth is CapabilitySignal.FAILED:
        return (
            ConnectorHealthState.AUTH_REQUIRED,
            HealthReasonCode.AUTH_EXPIRED,
            "Authentication failed; human reauthorization may be required",
        )

    if rate_limited:
        return (
            ConnectorHealthState.RATE_LIMITED,
            HealthReasonCode.RATE_LIMITED,
            "Provider rate limit is active",
        )

    if circuit_open and not recovering:
        return (
            ConnectorHealthState.OFFLINE,
            HealthReasonCode.CIRCUIT_OPEN,
            "Circuit breaker is open; aggressive probes are paused",
        )

    if recovering:
        return (
            ConnectorHealthState.RECOVERING,
            HealthReasonCode.RECOVERY_IN_PROGRESS,
            "Automatic recovery is in progress",
        )

    operational_failures = [
        (signals.ingress_operational, HealthReasonCode.INGRESS_STALLED, "Ingress path failed"),
        (signals.egress_operational, HealthReasonCode.EGRESS_FAILED, "Egress path failed"),
        (signals.subscription_operational, HealthReasonCode.SUBSCRIPTION_MISSING, "Subscription path failed"),
        (signals.acknowledgements_operational, HealthReasonCode.ACK_TIMEOUT, "Acknowledgement path failed"),
    ]
    hard_failures = [(s, code, msg) for s, code, msg in operational_failures if _failed(s)]
    soft_failures = [(s, code, msg) for s, code, msg in operational_failures if _degraded(s)]

    if not signals.transport_connected:
        if consecutive_failures >= failure_threshold or not signals.runtime_alive:
            return (
                ConnectorHealthState.OFFLINE,
                HealthReasonCode.CONNECTION_REFUSED,
                "Transport is disconnected",
            )
        return (
            ConnectorHealthState.RECOVERING,
            HealthReasonCode.CONNECTION_TIMEOUT,
            "Transport disconnected; reconnection expected",
        )

    if hard_failures:
        # Runtime may still be alive while a path is broken — never report healthy.
        _, code, message = hard_failures[0]
        if signals.runtime_alive and not signals.transport_connected:
            return ConnectorHealthState.OFFLINE, code, message
        if consecutive_failures >= failure_threshold:
            return ConnectorHealthState.OFFLINE, code, message
        return ConnectorHealthState.DEGRADED, code, message

    if soft_failures:
        _, code, message = soft_failures[0]
        return ConnectorHealthState.DEGRADED, code, message

    if auth is CapabilitySignal.UNKNOWN and not signals.transport_connected:
        return (
            ConnectorHealthState.UNKNOWN,
            HealthReasonCode.STARTUP_PENDING,
            "Initial health verification has not completed",
        )

    if signals.runtime_alive and not signals.transport_connected and auth is not CapabilitySignal.OK:
        return (
            ConnectorHealthState.UNKNOWN,
            HealthReasonCode.PROCESS_ONLY_ALIVE,
            "Runtime is alive but connector path is not verified",
        )

    if auth is CapabilitySignal.OK and signals.transport_connected:
        # All applicable non-OK signals already handled above.
        return ConnectorHealthState.HEALTHY, HealthReasonCode.NONE, "Connector path is healthy"

    if auth is CapabilitySignal.UNKNOWN:
        return (
            ConnectorHealthState.UNKNOWN,
            HealthReasonCode.STARTUP_PENDING,
            "Awaiting first successful health probe",
        )

    return (
        ConnectorHealthState.DEGRADED,
        HealthReasonCode.PARTIAL_CAPABILITY_FAILURE,
        "Connector path is only partially verified",
    )


def next_state_after_probe(
    current: ConnectorHealthState,
    derived: ConnectorHealthState,
    *,
    consecutive_failures: int,
    failure_threshold: int,
    recovery_enabled: bool,
) -> ConnectorHealthState:
    """Apply lifecycle transitions after a probe-derived candidate state."""
    if derived in {
        ConnectorHealthState.AUTH_REQUIRED,
        ConnectorHealthState.DISABLED,
        ConnectorHealthState.RATE_LIMITED,
    }:
        return derived

    if derived is ConnectorHealthState.HEALTHY:
        return ConnectorHealthState.HEALTHY

    if current is ConnectorHealthState.HEALTHY and derived is ConnectorHealthState.DEGRADED:
        return ConnectorHealthState.DEGRADED

    if derived is ConnectorHealthState.DEGRADED:
        if consecutive_failures >= failure_threshold and recovery_enabled:
            return ConnectorHealthState.RECOVERING
        return ConnectorHealthState.DEGRADED

    if derived is ConnectorHealthState.RECOVERING:
        return ConnectorHealthState.RECOVERING

    if derived is ConnectorHealthState.OFFLINE:
        if current is ConnectorHealthState.RECOVERING:
            return ConnectorHealthState.OFFLINE
        if consecutive_failures >= failure_threshold:
            return ConnectorHealthState.OFFLINE
        if recovery_enabled and current in {
            ConnectorHealthState.HEALTHY,
            ConnectorHealthState.DEGRADED,
            ConnectorHealthState.UNKNOWN,
        }:
            return ConnectorHealthState.RECOVERING
        return ConnectorHealthState.OFFLINE

    if current is ConnectorHealthState.OFFLINE and derived is ConnectorHealthState.RECOVERING:
        return ConnectorHealthState.RECOVERING

    if current is ConnectorHealthState.OFFLINE and recovery_enabled and derived is ConnectorHealthState.DEGRADED:
        return ConnectorHealthState.RECOVERING

    return derived
