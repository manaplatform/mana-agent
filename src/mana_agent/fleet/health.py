"""Deterministic fleet health evaluation."""

from __future__ import annotations

from datetime import timedelta

from .models import FleetWorker, WorkerStatus, utc_now


def effective_status(
    worker: FleetWorker, *,
    heartbeat_timeout_seconds: int,
    capability_ttl_seconds: int,
    now=None,
) -> WorkerStatus:
    if worker.health.status is WorkerStatus.REVOKED:
        return WorkerStatus.REVOKED
    if worker.health.status is WorkerStatus.DRAINING:
        return WorkerStatus.DRAINING
    if worker.health.status in {
        WorkerStatus.OFFLINE, WorkerStatus.DEGRADED, WorkerStatus.UNKNOWN,
    }:
        return worker.health.status
    current = now or utc_now()
    if (
        worker.health.last_heartbeat is None
        or current - worker.health.last_heartbeat > timedelta(seconds=heartbeat_timeout_seconds)
    ):
        return WorkerStatus.OFFLINE
    if current - worker.capabilities.last_probe_at > timedelta(seconds=capability_ttl_seconds):
        return WorkerStatus.DEGRADED
    if not worker.identity.authenticated or worker.health.identity_status != "valid":
        return WorkerStatus.DEGRADED
    if worker.health.active_job_count >= worker.health.concurrency_limit:
        return WorkerStatus.BUSY
    return WorkerStatus.CONNECTED
