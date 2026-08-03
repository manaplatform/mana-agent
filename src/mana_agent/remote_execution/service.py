"""Provider selection, exact permissions, event de-duplication, and jobs."""

from __future__ import annotations

import asyncio
import getpass
import secrets
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable

from mana_agent.human_inbox.models import (
    InboxQuery,
    InboxRequest,
    InboxRequestType,
    InboxStatus,
    ResponseOperation,
    ResponseSubmission,
    ReviewerAssignment,
    ReviewerType,
    RiskLevel,
    UNRESOLVED_STATUSES,
)
from mana_agent.human_inbox.service import HumanInboxService
from mana_agent.utils.redaction import redact_secrets
from mana_agent.remote_execution.models import (
    RemoteExecutionEvent,
    RemoteExecutionRequest,
    RemoteJobState,
)
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
    def __init__(
        self,
        *,
        workers: WorkerRegistry | None = None,
        target_policy: TargetPolicy | None = None,
        event_sink: Callable[[RemoteExecutionEvent], None] | None = None,
        outbound_tcp_available: bool = True,
        inbox_service: HumanInboxService | None = None,
    ) -> None:
        self.workers = workers or WorkerRegistry()
        self.target_policy = target_policy or TargetPolicy()
        self.event_sink = event_sink
        self.outbound_tcp_available = outbound_tcp_available
        self.inbox_service = inbox_service
        self.jobs: dict[str, RemoteJob] = {}
        self._permission_requests: dict[str, str] = {}
        self._seen_events: set[tuple[str, int, str, tuple[tuple[str, object], ...]]] = set()
        if self.inbox_service is not None:
            self._restore_durable_permissions()

    def attach_inbox(self, inbox_service: HumanInboxService) -> None:
        """Attach the coordinator's authoritative inbox and recover pending jobs."""
        if self.inbox_service is not None and self.inbox_service is not inbox_service:
            current_root = getattr(self.inbox_service.repository, "root", None)
            requested_root = getattr(inbox_service.repository, "root", None)
            if current_root != requested_root:
                raise RuntimeError("remote execution already has a different inbox authority")
            self._restore_durable_permissions()
            return
        self.inbox_service = inbox_service
        self._restore_durable_permissions()

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
            if self.inbox_service is not None:
                category_risk = {
                    "read_only": RiskLevel.LOW,
                    "remote_write": RiskLevel.HIGH,
                    "privileged_or_destructive": RiskLevel.CRITICAL,
                    "interactive_shell": RiskLevel.HIGH,
                }.get(category, RiskLevel.MEDIUM)
                effect_labels: dict[str, bool | None] = {
                    "reversible": None,
                    "compensatable": None,
                    "irreversible": None,
                    "externally_visible": True,
                    "data_disclosing": True if request.environment else None,
                    "potentially_billable": None,
                }
                item = self.inbox_service.create(InboxRequest(
                    request_type=InboxRequestType.APPROVAL,
                    task_id=request.job_id,
                    branch_id=request.job_id,
                    policy_decision_id=f"remote-target-policy:{request.exact_action_key()}",
                    permission_request_id=permission_request_id,
                    action_intent_id=f"remote:{request.job_id}",
                    action_digest=request.exact_action_key(),
                    requested_by_agent_id="remote_execution",
                    reviewer=ReviewerAssignment(
                        reviewer_type=ReviewerType.PERSON,
                        reviewer_id=getpass.getuser(),
                    ),
                    title="Approve remote SSH execution",
                    summary=f"Review an exact {category.replace('_', ' ')} remote operation.",
                    risk_level=category_risk,
                    allowed_responses=[ResponseOperation.APPROVE, ResponseOperation.DENY],
                    minimal_context={
                        "permission_category": category,
                        "action_count": 1,
                        "argument_count": len(request.command.argv),
                        "resource_count": 1,
                        "effect_labels": effect_labels,
                    },
                    protected_context={
                        "remote_request": request.model_dump(mode="json"),
                        "effect_labels": effect_labels,
                    },
                    disclosed_fields=[
                        "permission_category",
                        "action_count",
                        "argument_count",
                        "resource_count",
                        "effect_labels",
                    ],
                    reversibility="unknown",
                    expires_at=self.inbox_service.clock() + timedelta(minutes=15),
                    idempotency_key=f"remote-permission:{request.job_id}:{request.exact_action_key()}",
                    deduplication_key=f"remote-permission:{request.job_id}:{request.exact_action_key()}",
                ))
                permission_request_id = item.permission_request_id
            job.request = request.model_copy(
                update={"permission_request_id": permission_request_id}
            )
            self._permission_requests[permission_request_id] = request.job_id
            self._emit(job, RemoteExecutionEvent(
                job_id=request.job_id,
                session_id=request.session_id,
                kind="permission_requested",
                data={
                    "permission_request_id": permission_request_id,
                    "action_key": request.exact_action_key(),
                    "permission_category": category,
                },
            ))
        return job

    def approve(self, job_id: str) -> None:
        job = self.jobs[job_id]
        if job.state is not RemoteJobState.AWAITING_PERMISSION:
            raise ValueError("job is not awaiting permission")
        self.target_policy.approve_action(job.request)
        job.state = RemoteJobState.QUEUED

    def pending_permissions(self) -> list[dict[str, str]]:
        self._synchronize_durable_permission_states()
        rows: list[dict[str, str]] = []
        for request_id, job_id in self._permission_requests.items():
            job = self.jobs.get(job_id)
            if job is not None and job.state is RemoteJobState.AWAITING_PERMISSION:
                rows.append({
                    "permission_request_id": request_id,
                    "job_id": job_id,
                    "session_id": job.request.session_id,
                    "worker_id": job.request.worker_id,
                    "target": f"{job.request.target.user}@{job.request.target.host}:{job.request.target.port}",
                    "command": str(redact_secrets(" ".join(job.request.command.argv))),
                })
        return rows

    def approve_permission(
        self,
        permission_request_id: str,
        *,
        reviewer_id: str | None = None,
    ) -> RemoteJob:
        job_id = self._permission_requests.get(permission_request_id)
        if not job_id:
            raise ValueError("No remote SSH job is waiting for that permission request.")
        if self.inbox_service is not None:
            item = self._permission_item(permission_request_id)
            if item.status in UNRESOLVED_STATUSES:
                item = self.inbox_service.respond(ResponseSubmission(
                    inbox_item_id=item.inbox_item_id,
                    operation=ResponseOperation.APPROVE,
                    actor_id=reviewer_id or getpass.getuser(),
                    channel="remote_execution_legacy",
                    idempotency_key=f"remote-approve:{permission_request_id}",
                    current_action_digest=self.jobs[job_id].request.exact_action_key(),
                ))
            if item.status is not InboxStatus.APPROVED:
                raise PermissionError(f"remote permission is {item.status.value}")
            if item.expires_at <= self.inbox_service.clock():
                raise PermissionError("remote permission expired before execution")
            self.inbox_service.assert_response_actor_is_currently_authorized(item)
        if self.jobs[job_id].state is RemoteJobState.AWAITING_PERMISSION:
            self.approve(job_id)
        elif self.jobs[job_id].state is not RemoteJobState.QUEUED:
            raise ValueError("job is not awaiting permission")
        self._permission_requests.pop(permission_request_id, None)
        return self.jobs[job_id]

    def deny_permission(
        self,
        permission_request_id: str,
        *,
        reviewer_id: str | None = None,
    ) -> RemoteJob:
        job_id = self._permission_requests.get(permission_request_id)
        if not job_id:
            raise ValueError("No remote SSH job is waiting for that permission request.")
        job = self.jobs[job_id]
        if self.inbox_service is not None:
            item = self._permission_item(permission_request_id)
            if item.status in UNRESOLVED_STATUSES:
                item = self.inbox_service.respond(ResponseSubmission(
                    inbox_item_id=item.inbox_item_id,
                    operation=ResponseOperation.DENY,
                    actor_id=reviewer_id or getpass.getuser(),
                    channel="remote_execution_legacy",
                    idempotency_key=f"remote-deny:{permission_request_id}",
                    current_action_digest=job.request.exact_action_key(),
                ))
            if item.status is not InboxStatus.DENIED:
                raise PermissionError(f"remote permission is {item.status.value}")
        job.state = RemoteJobState.CANCELLED
        self._permission_requests.pop(permission_request_id, None)
        return job

    def _permission_item(self, permission_request_id: str):
        assert self.inbox_service is not None
        matches = [
            item
            for item in self.inbox_service.repository.list()
            if item.permission_request_id == permission_request_id
            and item.action_intent_id.startswith("remote:")
        ]
        if not matches:
            raise LookupError("durable remote permission record was not found")
        return matches[0]

    def _restore_durable_permissions(self) -> None:
        assert self.inbox_service is not None
        for item in self.inbox_service.repository.list(InboxQuery(
            statuses={InboxStatus.PENDING, InboxStatus.DELIVERED, InboxStatus.APPROVED},
        )):
            if not item.action_intent_id.startswith("remote:") or not item.protected_context_ref:
                continue
            context = self.inbox_service.repository.read_protected_context(item.protected_context_ref)
            request = RemoteExecutionRequest.model_validate(context.get("remote_request")).model_copy(
                update={"permission_request_id": item.permission_request_id}
            )
            if request.exact_action_key() != item.action_digest:
                continue
            state = (
                RemoteJobState.QUEUED
                if item.status is InboxStatus.APPROVED
                else RemoteJobState.AWAITING_PERMISSION
            )
            self.jobs.setdefault(request.job_id, RemoteJob(request=request, state=state))
            self._permission_requests[item.permission_request_id] = request.job_id

    def _synchronize_durable_permission_states(self, *, job_id: str = "") -> None:
        if self.inbox_service is None:
            return
        for permission_request_id, mapped_job_id in list(self._permission_requests.items()):
            if job_id and mapped_job_id != job_id:
                continue
            item = self._permission_item(permission_request_id)
            job = self.jobs.get(mapped_job_id)
            if job is None:
                continue
            if item.status is InboxStatus.APPROVED:
                if item.expires_at <= self.inbox_service.clock():
                    if job.state in {RemoteJobState.AWAITING_PERMISSION, RemoteJobState.QUEUED}:
                        job.state = RemoteJobState.CANCELLED
                    self._permission_requests.pop(permission_request_id, None)
                    continue
                self.inbox_service.assert_response_actor_is_currently_authorized(item)
                if job.state is RemoteJobState.AWAITING_PERMISSION:
                    self.target_policy.approve_action(job.request)
                    job.state = RemoteJobState.QUEUED
            elif item.status not in UNRESOLVED_STATUSES and item.status is not InboxStatus.APPROVED:
                if job.state is RemoteJobState.AWAITING_PERMISSION:
                    job.state = RemoteJobState.CANCELLED
                self._permission_requests.pop(permission_request_id, None)

    async def execute(self, job_id: str) -> RemoteJob:
        self._synchronize_durable_permission_states(job_id=job_id)
        job = self.jobs[job_id]
        if job.state is RemoteJobState.AWAITING_PERMISSION:
            return job
        if job.state is not RemoteJobState.QUEUED:
            # A durable job ID is a one-execution fence. Assigned, running, and
            # terminal jobs are observed, never dispatched a second time.
            return job
        request = job.request
        execution_claim_id = ""
        execution_item_id = ""
        if self.inbox_service is not None and request.permission_request_id:
            item = self._permission_item(request.permission_request_id)
            if item.status is not InboxStatus.APPROVED:
                raise PermissionError(f"remote permission is {item.status.value}")
            if item.expires_at <= self.inbox_service.clock():
                raise PermissionError("remote permission expired before execution")
            self.inbox_service.assert_response_actor_is_currently_authorized(item)
            execution_item_id = item.inbox_item_id
            execution_claim_id = self.inbox_service.claim_action_execution(
                execution_item_id
            )
        if request.provider in {"reverse-worker", "external_worker"}:
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
        if self.inbox_service is not None and execution_claim_id:
            self.inbox_service.complete_action_execution(
                execution_item_id,
                execution_claim_id=execution_claim_id,
                result_digest=f"remote:{request.job_id}:{job.state.value}",
            )
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

    def record_worker_terminal(self, job_id: str) -> None:
        """Commit the durable execution fence after a worker terminal result."""
        job = self.jobs[job_id]
        permission_request_id = job.request.permission_request_id
        if self.inbox_service is None or not permission_request_id:
            return
        item = self._permission_item(permission_request_id)
        if not item.execution_claim_id or item.execution_completed_at is not None:
            return
        self.inbox_service.complete_action_execution(
            item.inbox_item_id,
            execution_claim_id=item.execution_claim_id,
            result_digest=f"remote:{job_id}:{job.state.value}",
        )

    def cancel(self, job_id: str) -> None:
        self.jobs[job_id].cancel.set()
