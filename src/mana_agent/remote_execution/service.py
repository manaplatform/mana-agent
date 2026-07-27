"""Provider selection, exact permissions, event de-duplication, and jobs."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field
from typing import Callable

from mana_agent.remote_execution.models import RemoteExecutionEvent, RemoteExecutionRequest, RemoteJobState
from mana_agent.remote_execution.providers.local_ssh import LocalSSHProvider
from mana_agent.remote_execution.permissions import required_permission
from mana_agent.remote_execution.target_policy import TargetPolicy
from mana_agent.remote_execution.worker import WorkerRegistry


@dataclass
class RemoteJob:
    request: RemoteExecutionRequest
    state: RemoteJobState = RemoteJobState.QUEUED
    events: list[RemoteExecutionEvent] = field(default_factory=list)
    cancel: asyncio.Event = field(default_factory=asyncio.Event)


class RemoteExecutionService:
    def __init__(self, *, workers: WorkerRegistry | None = None, target_policy: TargetPolicy | None = None, event_sink: Callable[[RemoteExecutionEvent], None] | None = None, outbound_tcp_available: bool = True) -> None:
        self.workers = workers or WorkerRegistry()
        self.target_policy = target_policy or TargetPolicy()
        self.event_sink = event_sink
        self.outbound_tcp_available = outbound_tcp_available
        self.jobs: dict[str, RemoteJob] = {}
        self._permission_requests: dict[str, str] = {}
        self._seen_events: set[tuple[str, int, str, tuple[tuple[str, object], ...]]] = set()

    def _emit(self, job: RemoteJob, event: RemoteExecutionEvent) -> None:
        identity = (event.job_id, event.sequence, event.kind, tuple(sorted(event.data.items())))
        if identity in self._seen_events:
            return
        self._seen_events.add(identity)
        job.events.append(event)
        if self.event_sink:
            self.event_sink(event)

    def submit(self, request: RemoteExecutionRequest) -> RemoteJob:
        if request.job_id in self.jobs:
            raise ValueError("duplicate remote execution job ID")
        job = RemoteJob(request=request)
        self.jobs[request.job_id] = job
        # The exact risk-derived category is surfaced to the existing gateway
        # permission layer rather than trusting a client-supplied read_only flag.
        category = required_permission(request).value
        if self.target_policy.requires_approval(request):
            job.state = RemoteJobState.AWAITING_PERMISSION
            permission_request_id = f"remote_permission_{secrets.token_urlsafe(18)}"
            self._permission_requests[permission_request_id] = request.job_id
            self._emit(job, RemoteExecutionEvent(job_id=request.job_id, session_id=request.session_id, kind="permission_requested", data={"permission_request_id": permission_request_id, "action_key": request.exact_action_key(), "permission_category": category}))
        return job

    def approve(self, job_id: str) -> None:
        job = self.jobs[job_id]
        if job.state is not RemoteJobState.AWAITING_PERMISSION:
            raise ValueError("job is not awaiting permission")
        self.target_policy.approve_action(job.request)
        job.state = RemoteJobState.QUEUED

    def pending_permissions(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for request_id, job_id in self._permission_requests.items():
            job = self.jobs.get(job_id)
            if job is not None and job.state is RemoteJobState.AWAITING_PERMISSION:
                rows.append({"permission_request_id": request_id, "job_id": job_id, "session_id": job.request.session_id, "worker_id": job.request.worker_id, "target": f"{job.request.target.user}@{job.request.target.host}:{job.request.target.port}", "command": " ".join(job.request.command.argv)})
        return rows

    def approve_permission(self, permission_request_id: str) -> RemoteJob:
        job_id = self._permission_requests.pop(permission_request_id, None)
        if not job_id:
            raise ValueError("No remote SSH job is waiting for that permission request.")
        self.approve(job_id)
        return self.jobs[job_id]

    async def execute(self, job_id: str) -> RemoteJob:
        job = self.jobs[job_id]
        if job.state is RemoteJobState.AWAITING_PERMISSION:
            return job
        request = job.request
        if request.provider in {"reverse-worker", "external_worker"}:
            return self._dispatch_worker(job)
        # Legacy/auto requests may be routed to an already-connected worker when
        # the host process cannot open TCP. Explicit `remote-ssh` never changes
        # route, especially for state-changing work.
        if not self.outbound_tcp_available and request.provider == "local_ssh":
            return self._dispatch_worker(job)
        if not self.outbound_tcp_available:
            job.state = RemoteJobState.FAILED
            raise RuntimeError(
                "The local Mana-Agent process is not permitted to open the SSH connection. "
                "This is a host-process sandbox restriction, not an SSH-key or route error."
            )
        job.state = RemoteJobState.RUNNING
        try:
            code, _out, err = await LocalSSHProvider().execute(request, lambda event: self._emit(job, event), job.cancel)
            self._emit(job, RemoteExecutionEvent(job_id=request.job_id, session_id=request.session_id, kind="exit_code", data={"code": code}))
            if code == 0:
                job.state = RemoteJobState.SUCCEEDED
            else:
                # An explicit direct-SSH request is never silently rerouted to a worker.
                job.state = RemoteJobState.FAILED
        except TimeoutError:
            job.state = RemoteJobState.TIMED_OUT
            self._emit(job, RemoteExecutionEvent(job_id=request.job_id, session_id=request.session_id, kind="timeout"))
        except asyncio.CancelledError:
            job.state = RemoteJobState.CANCELLED
            self._emit(job, RemoteExecutionEvent(job_id=request.job_id, session_id=request.session_id, kind="cancelled"))
        return job

    def _dispatch_worker(self, job: RemoteJob) -> RemoteJob:
        try:
            worker = self.workers.worker(job.request.worker_id)
        except LookupError:
            job.state = RemoteJobState.FAILED
            raise RuntimeError(
                "No trusted external SSH worker is connected. "
                "No local SSH fallback was attempted."
            ) from None
        job.state = RemoteJobState.ASSIGNED
        self._emit(job, RemoteExecutionEvent(job_id=job.request.job_id, session_id=job.request.session_id, kind="worker_selected", data={"worker_id": worker.registration.worker_id}))
        worker.submit(job.request)
        return job

    def worker_disconnected(self, worker_id: str) -> None:
        self.workers.disconnect(worker_id)
        for job in self.jobs.values():
            if job.request.worker_id == worker_id and job.state in {RemoteJobState.ASSIGNED, RemoteJobState.RUNNING}:
                job.state = RemoteJobState.WORKER_DISCONNECTED
                self._emit(job, RemoteExecutionEvent(job_id=job.request.job_id, session_id=job.request.session_id, kind="worker_disconnected", data={"worker_id": worker_id}))

    def cancel(self, job_id: str) -> None:
        self.jobs[job_id].cancel.set()
