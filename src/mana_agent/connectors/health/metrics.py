"""Connector reliability metrics; never invent SLO values without data."""

from __future__ import annotations

from datetime import datetime, timedelta

from .models import (
    ConnectorHealthState,
    ConnectorIncident,
    ConnectorSLOMetrics,
    DeliveryState,
    ProbeOutcome,
    ProbeResult,
    utc_now,
)
from .storage import ConnectorHealthStore


MIN_OBSERVATIONS = 3


def compute_slo_metrics(
    connector_id: str,
    *,
    store: ConnectorHealthStore,
    window: timedelta | None = None,
    now: datetime | None = None,
) -> ConnectorSLOMetrics:
    clock = now or utc_now()
    lookback = window or timedelta(hours=24)
    cutoff = clock - lookback

    probes = [
        item
        for item in store.load_probe_results(connector_id, limit=500)
        if item.checked_at >= cutoff
    ]
    receipts = [
        item
        for item in store.list_receipts(connector_id, limit=200)
        if (item.submitted_at or clock) >= cutoff
    ]
    incidents = [
        item
        for item in store.list_incidents(connector_id=connector_id, limit=100)
        if item.started_at >= cutoff or (item.ended_at is None)
    ]

    observation_count = len(probes)
    if observation_count < MIN_OBSERVATIONS:
        return ConnectorSLOMetrics(
            connector_id=connector_id,
            observation_count=observation_count,
            insufficient_data=True,
            failure_count=sum(1 for p in probes if p.outcome is ProbeOutcome.FAILED),
            reconnect_count=_reconnect_count(incidents),
        )

    successes = sum(1 for p in probes if p.outcome is ProbeOutcome.PASSED)
    failures = sum(1 for p in probes if p.outcome is ProbeOutcome.FAILED)
    total_scored = successes + failures
    successful_probe_ratio = (successes / total_scored) if total_scored else None

    delivered = [
        r
        for r in receipts
        if r.state in {DeliveryState.DELIVERED, DeliveryState.ACKNOWLEDGED, DeliveryState.PROVIDER_ACCEPTED}
    ]
    failed_delivery = [r for r in receipts if r.state is DeliveryState.FAILED]
    delivery_total = len(delivered) + len(failed_delivery)
    delivery_success_rate = (len(delivered) / delivery_total) if delivery_total else None

    ack_latencies = [
        float(r.latency_ms)
        for r in receipts
        if r.latency_ms is not None and r.state is DeliveryState.ACKNOWLEDGED
    ]
    ack_latency_ms = (sum(ack_latencies) / len(ack_latencies)) if ack_latencies else None

    degraded_seconds, offline_seconds, recovery_seconds = _duration_stats(incidents, clock)
    mean_recovery = (
        sum(recovery_seconds) / len(recovery_seconds) if recovery_seconds else None
    )

    availability = successful_probe_ratio

    return ConnectorSLOMetrics(
        connector_id=connector_id,
        availability=availability,
        successful_probe_ratio=successful_probe_ratio,
        delivery_success_rate=delivery_success_rate,
        ack_latency_ms=ack_latency_ms,
        reconnect_count=_reconnect_count(incidents),
        mean_recovery_seconds=mean_recovery,
        failure_count=failures,
        degraded_duration_seconds=degraded_seconds,
        offline_duration_seconds=offline_seconds,
        observation_count=observation_count,
        insufficient_data=False,
    )


def _reconnect_count(incidents: list[ConnectorIncident]) -> int:
    count = 0
    for incident in incidents:
        for event in incident.events:
            if event.event_type in {
                "reconnect_attempt_started",
                "connector.recovery.started",
            }:
                count += 1
    return count


def _duration_stats(
    incidents: list[ConnectorIncident],
    now: datetime,
) -> tuple[float, float, list[float]]:
    degraded = 0.0
    offline = 0.0
    recoveries: list[float] = []
    for incident in incidents:
        end = incident.ended_at or now
        duration = max(0.0, (end - incident.started_at).total_seconds())
        if incident.opening_state is ConnectorHealthState.DEGRADED:
            degraded += duration
        if incident.opening_state in {
            ConnectorHealthState.OFFLINE,
            ConnectorHealthState.RECOVERING,
            ConnectorHealthState.AUTH_REQUIRED,
        }:
            offline += duration
        if incident.recovered and incident.ended_at is not None:
            recoveries.append(max(0.0, (incident.ended_at - incident.started_at).total_seconds()))
    return degraded, offline, recoveries
