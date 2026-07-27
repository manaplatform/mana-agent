"""Persistent trusted-worker registry."""

from __future__ import annotations

import json
from datetime import timedelta

from .config import FleetConfig
from .errors import FleetStateError
from .health import effective_status
from .models import (
    FleetWorker, WorkerCapabilities, WorkerHealth, WorkerIdentity, WorkerStatus, utc_now,
)
from .store import FleetStore


class FleetRegistry:
    def __init__(self, store: FleetStore, config: FleetConfig) -> None:
        self.store = store
        self.config = config
        self._workers = {item.worker_id: item for item in store.load_workers()}

    def accept_capabilities(
        self, capabilities: WorkerCapabilities, identity: WorkerIdentity, *,
        display_name: str = "",
    ) -> tuple[FleetWorker, bool]:
        if capabilities.worker_id != identity.worker_id or not identity.authenticated:
            raise FleetStateError("authenticated worker identity is required for capability updates")
        encoded = json.dumps(capabilities.model_dump(mode="json"), separators=(",", ":")).encode()
        if len(encoded) > 64 * 1024:
            raise FleetStateError("worker capability inventory exceeds the 64 KiB limit")
        age = utc_now() - capabilities.last_probe_at
        if age < timedelta(minutes=-5):
            raise FleetStateError("worker capability inventory timestamp is in the future")
        if age > timedelta(seconds=self.config.capability_ttl_seconds):
            raise FleetStateError("worker capability inventory is stale")
        previous = self._workers.get(identity.worker_id)
        if previous and previous.health.status is WorkerStatus.REVOKED:
            raise FleetStateError("revoked workers cannot update capabilities")
        changed = previous is None or previous.capability_fingerprint != capabilities.fingerprint
        health = previous.health if previous else WorkerHealth()
        health.last_capability_update = utc_now()
        health.concurrency_limit = capabilities.max_concurrency
        health.identity_status = identity.credential_status
        worker = FleetWorker(
            identity=identity, display_name=display_name or (previous.display_name if previous else ""),
            capabilities=capabilities, capability_fingerprint=capabilities.fingerprint,
            health=health, registered_at=previous.registered_at if previous else utc_now(),
            updated_at=utc_now(),
        )
        self._workers[worker.worker_id] = worker
        self.store.save_worker(worker)
        return worker, changed

    def heartbeat(self, worker_id: str, *, transport_status: str = "connected") -> FleetWorker:
        worker = self.require(worker_id)
        if worker.health.status is WorkerStatus.REVOKED:
            raise FleetStateError("revoked workers cannot heartbeat")
        worker.health.last_heartbeat = utc_now()
        worker.health.transport_status = transport_status
        if worker.health.status in {WorkerStatus.UNKNOWN, WorkerStatus.OFFLINE, WorkerStatus.DEGRADED}:
            worker.health.status = WorkerStatus.CONNECTED
        worker.updated_at = utc_now()
        self.store.save_worker(worker)
        return worker

    def require(self, worker_id: str) -> FleetWorker:
        try:
            return self._workers[worker_id]
        except KeyError as exc:
            raise FleetStateError(f"fleet worker not found: {worker_id}") from exc

    def list(self) -> list[FleetWorker]:
        return [self._workers[key] for key in sorted(self._workers)]

    def eligible(self) -> list[FleetWorker]:
        return [
            worker for worker in self.list()
            if effective_status(
                worker,
                heartbeat_timeout_seconds=self.config.heartbeat_timeout_seconds,
                capability_ttl_seconds=self.config.capability_ttl_seconds,
            ) is WorkerStatus.CONNECTED
        ]

    def set_status(self, worker_id: str, status: WorkerStatus) -> FleetWorker:
        worker = self.require(worker_id)
        if worker.health.status is WorkerStatus.REVOKED and status is not WorkerStatus.REVOKED:
            raise FleetStateError("revoked workers require fresh enrollment and cannot be re-enabled")
        worker.health.status = status
        worker.updated_at = utc_now()
        self.store.save_worker(worker)
        return worker

    def reserve(self, worker_id: str) -> None:
        worker = self.require(worker_id)
        status = effective_status(
            worker,
            heartbeat_timeout_seconds=self.config.heartbeat_timeout_seconds,
            capability_ttl_seconds=self.config.capability_ttl_seconds,
        )
        if status is not WorkerStatus.CONNECTED:
            raise FleetStateError(f"worker {worker_id} is not dispatchable: {status.value}")
        worker.health.active_job_count += 1
        self.store.save_worker(worker)

    def release(self, worker_id: str, *, success: bool, failure: str = "") -> None:
        worker = self.require(worker_id)
        worker.health.active_job_count = max(0, worker.health.active_job_count - 1)
        if success:
            worker.health.last_successful_job = utc_now()
        elif failure:
            worker.health.recent_failures = (*worker.health.recent_failures[-19:], failure[:500])
        self.store.save_worker(worker)
