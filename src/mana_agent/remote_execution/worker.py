"""Coordinator-side registry for authenticated reverse-connected workers.

The transport is deliberately a small protocol boundary: a WebSocket or HTTPS
long-poll adapter calls :meth:`connect` after authenticating a worker, then
pulls jobs and pushes events.  The coordinator never opens a connection to it.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from mana_agent.remote_execution.models import RemoteExecutionRequest, WorkerRegistration


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass
class _Enrolment:
    worker_id: str
    digest: str
    expires_at: datetime


@dataclass
class ConnectedWorker:
    registration: WorkerRegistration
    credential_digest: str
    last_heartbeat: datetime = field(default_factory=_now)
    status: str = "offline"
    submit: Callable[[RemoteExecutionRequest], None] | None = None


class WorkerRegistry:
    """Owns credentials and reverse worker identities without retaining secrets."""

    def __init__(self) -> None:
        self._enrolments: dict[str, _Enrolment] = {}
        self._workers: dict[str, ConnectedWorker] = {}

    def issue_enrolment_token(self, worker_id: str, *, ttl_seconds: int = 600) -> str:
        token = secrets.token_urlsafe(32)
        self._enrolments[_digest(token)] = _Enrolment(worker_id, _digest(token), _now() + timedelta(seconds=ttl_seconds))
        return token

    def enrol(self, token: str, registration: WorkerRegistration) -> str:
        record = self._enrolments.pop(_digest(token), None)  # one use, including invalid registration attempts
        if record is None or record.expires_at <= _now() or record.worker_id != registration.worker_id:
            raise PermissionError("invalid, expired, or replayed worker enrolment token")
        credential = secrets.token_urlsafe(48)
        self._workers[registration.worker_id] = ConnectedWorker(registration, _digest(credential))
        return credential

    def connect(self, worker_id: str, credential: str, submit: Callable[[RemoteExecutionRequest], None]) -> None:
        worker = self._workers.get(worker_id)
        if worker is None or not secrets.compare_digest(worker.credential_digest, _digest(credential)):
            raise PermissionError("worker authentication failed")
        worker.status, worker.last_heartbeat, worker.submit = "online", _now(), submit

    def heartbeat(self, worker_id: str) -> None:
        worker = self._workers.get(worker_id)
        if worker is None or worker.status != "online":
            raise PermissionError("worker is not connected")
        worker.last_heartbeat = _now()

    def worker(self, worker_id: str) -> ConnectedWorker:
        worker = self._workers.get(worker_id)
        if worker is None or worker.status != "online" or worker.submit is None:
            raise LookupError("selected worker is not connected")
        return worker

    def select_connected_worker(self) -> ConnectedWorker:
        connected = [worker for worker in self._workers.values() if worker.status == "online" and worker.submit is not None]
        if len(connected) != 1:
            raise LookupError("Automatic worker selection requires exactly one trusted connected worker.")
        return connected[0]

    def disconnect(self, worker_id: str) -> None:
        if worker := self._workers.get(worker_id):
            worker.status, worker.submit = "offline", None

    def registration(self, worker_id: str) -> WorkerRegistration | None:
        worker = self._workers.get(worker_id)
        return worker.registration if worker is not None else None

    def revoke(self, worker_id: str) -> None:
        self._workers.pop(worker_id, None)


def store_worker_credential(path: Path, credential: str) -> None:
    """Persist only the opaque worker credential with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(credential)
    os.chmod(path, 0o600)
