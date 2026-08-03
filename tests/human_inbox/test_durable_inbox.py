from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Lock
import getpass

import pytest
from typer.testing import CliRunner

from mana_agent.execution_supervisor import ExecutionSupervisor, ExecutionSupervisorConfig
from mana_agent.execution_supervisor.errors import LeaseConflictError
from mana_agent.execution_supervisor.models import ExecutionState, SideEffectClassification
from mana_agent.human_inbox.identity import ReviewerIdentity, StaticIdentityDirectory
from mana_agent.human_inbox import cli as inbox_cli
from mana_agent.human_inbox.models import (
    ClarificationField,
    ExpectedResponseType,
    ExpiryBehavior,
    InboxQuery,
    InboxRequest,
    InboxRequestType,
    InboxStatus,
    ReminderPolicy,
    ResponseOperation,
    ResponseSubmission,
    ReviewerAssignment,
    ReviewerType,
)
from mana_agent.human_inbox.notifications import NotificationResult
from mana_agent.human_inbox.repository import InboxConcurrentUpdateError, LocalInboxRepository
from mana_agent.human_inbox.service import HumanInboxService
from mana_agent.human_inbox import tokens as inbox_tokens
from mana_agent.human_inbox.tokens import ResponseTokenSigner
from mana_agent.transactional_actions.adapters import ActionInvalidatedError, FileActionAdapter
from mana_agent.transactional_actions.approvals import ApprovalRegistry
from mana_agent.transactional_actions.gateway import ActionGateway, ApprovalRequired
from mana_agent.transactional_actions.policy import ActionPolicy, PolicyConfig
from mana_agent.transactional_actions.store import ActionStore
from mana_agent.remote_execution.models import (
    RemoteCommand,
    RemoteExecutionRequest,
    SSHAuthentication,
    SSHTarget,
)
from mana_agent.remote_execution.service import RemoteExecutionService
from mana_agent.remote_execution.target_policy import TargetPolicy


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


def service(tmp_path: Path, clock: Clock, *, supervisor=None, adapters=None, identities=None) -> HumanInboxService:
    root = tmp_path / "inbox"
    return HumanInboxService(
        repository=LocalInboxRepository(root),
        identities=StaticIdentityDirectory(
            [ReviewerIdentity(identity_id="reviewer-1", roles={"security"})]
            if identities is None
            else identities
        ),
        token_signer=ResponseTokenSigner(root / "signing.key", clock=clock),
        branch_controller=supervisor,
        notification_adapters=adapters or [],
        clock=clock,
    )


def request(clock: Clock, **changes) -> InboxRequest:
    values = {
        "request_type": InboxRequestType.APPROVAL,
        "task_id": "task-1",
        "branch_id": "task-1",
        "requested_by_agent_id": "agent-1",
        "reviewer": ReviewerAssignment(reviewer_type=ReviewerType.PERSON, reviewer_id="reviewer-1"),
        "title": "Approve exact action",
        "summary": "Delete one generated file.",
        "allowed_responses": [ResponseOperation.APPROVE, ResponseOperation.DENY],
        "minimal_context": {"resource_count": 1},
        "disclosed_fields": ["resource_count"],
        "expires_at": clock() + timedelta(hours=1),
        "idempotency_key": "request-1",
        "deduplication_key": "dedupe-1",
    }
    values.update(changes)
    return InboxRequest(**values)


def test_terminal_notice_is_persisted_without_response_or_delivery(tmp_path: Path) -> None:
    clock = Clock()
    inbox = service(tmp_path, clock)
    notice = inbox.create(request(
        clock,
        request_type=InboxRequestType.NOTICE,
        title="Computer request recorded without execution",
        summary="The selected action was unavailable.",
        allowed_responses=[],
        requested_fields=[],
        idempotency_key="notice-1",
        deduplication_key="notice-dedupe-1",
    ))
    assert notice.status is InboxStatus.RECORDED
    assert notice.allowed_responses == []
    assert inbox.repository.delivery_attempts(notice.inbox_item_id) == []
    audit = inbox.repository.audit_for_item(notice.inbox_item_id)
    assert [event.event_type for event in audit] == [
        "request_created",
        "reviewer_resolved",
    ]
    assert [event.sequence for event in audit] == [1, 2]


def test_cli_rejects_terminal_notice_without_traceback(tmp_path: Path, monkeypatch) -> None:
    clock = Clock()
    inbox = service(tmp_path, clock)
    notice = inbox.create(request(
        clock,
        request_type=InboxRequestType.NOTICE,
        title="Computer request recorded without execution",
        summary="The selected action was unavailable.",
        allowed_responses=[],
        requested_fields=[],
        idempotency_key="notice-cli-1",
        deduplication_key="notice-cli-dedupe-1",
    ))
    monkeypatch.setattr(inbox_cli, "_service", lambda: inbox)
    monkeypatch.setattr(inbox_cli, "_actor", lambda _actor: "reviewer-1")

    result = CliRunner().invoke(inbox_cli.inbox_app, ["approve", notice.inbox_item_id])

    assert result.exit_code == 2
    assert "recorded terminal notice" in result.output
    assert "Traceback" not in result.output
    assert any(
        event.event_type == "response_rejected"
        for event in inbox.repository.audit_for_item(notice.inbox_item_id)
    )


def respond(item, **changes) -> ResponseSubmission:
    values = {
        "inbox_item_id": item.inbox_item_id,
        "operation": ResponseOperation.APPROVE,
        "actor_id": "reviewer-1",
        "channel": "test",
        "idempotency_key": "response-1",
        "expected_version": item.version,
        "current_action_digest": item.action_digest,
    }
    values.update(changes)
    return ResponseSubmission(**values)


def test_approval_and_clarification_survive_restart(tmp_path: Path) -> None:
    clock = Clock()
    first = service(tmp_path, clock)
    approval = first.create(request(clock))
    clarification = first.create(request(
        clock,
        request_type=InboxRequestType.CLARIFICATION,
        requested_fields=[ClarificationField(field_id="region", prompt="Which region?", expected_type=ExpectedResponseType.CHOICE, choices=["eu", "us"], allow_free_form=False)],
        allowed_responses=[ResponseOperation.ANSWER],
        idempotency_key="request-2",
        deduplication_key="dedupe-2",
    ))
    restarted = service(tmp_path, clock)
    assert restarted.repository.get(approval.inbox_item_id).status is InboxStatus.PENDING
    assert restarted.repository.get(clarification.inbox_item_id).requested_fields[0].choices == ["eu", "us"]
    assert {
        item.inbox_item_id
        for item in restarted.list(InboxQuery(), actor_id="reviewer-1")
    } == {approval.inbox_item_id, clarification.inbox_item_id}


def test_only_checkpointed_branch_waits_and_resumes_once(tmp_path: Path) -> None:
    clock = Clock()
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(
        root=tmp_path / "execution", startup_recovery=False, lease_seconds=30, heartbeat_seconds=5,
        max_concurrent_children=1,
    ), clock=clock)
    parent = supervisor.create_task(routing_decision_id="route-parent", side_effect_classification=SideEffectClassification.READ_ONLY, workspace_path=tmp_path)
    left = supervisor.create_task(parent_task_id=parent.task_id, routing_decision_id="route-left", side_effect_classification=SideEffectClassification.READ_ONLY, workspace_path=tmp_path)
    right = supervisor.create_task(parent_task_id=parent.task_id, routing_decision_id="route-right", side_effect_classification=SideEffectClassification.READ_ONLY, workspace_path=tmp_path)
    for task in (left, right):
        supervisor.queue(task.task_id)
    leased_left, token_left = supervisor.acquire_lease(left.task_id, owner="worker-left")
    supervisor.start(left.task_id, attempt_id=leased_left.attempt_id, lease_token=token_left)
    checkpoint = supervisor.checkpoint(left.task_id, attempt_id=leased_left.attempt_id, lease_token=token_left, resume_payload={"cursor": 2})
    inbox = service(tmp_path, clock, supervisor=supervisor)
    item = inbox.create(request(
        clock,
        task_id=left.task_id,
        branch_id=left.task_id,
        parent_task_id=parent.task_id,
        checkpoint_id=checkpoint.checkpoint_id,
        execution_attempt_id=leased_left.attempt_id,
    ))
    leased_right, token_right = supervisor.acquire_lease(right.task_id, owner="worker-right")
    supervisor.start(right.task_id, attempt_id=leased_right.attempt_id, lease_token=token_right)
    assert supervisor.store.get_task(left.task_id).state is ExecutionState.WAITING
    assert supervisor.store.get_task(left.task_id).lease_owner == ""
    assert supervisor.store.get_task(right.task_id).state is ExecutionState.RUNNING
    with pytest.raises(LeaseConflictError, match="durable inbox claim"):
        supervisor.queue(left.task_id)
    resolved = inbox.respond(respond(item))
    resumed = supervisor.store.get_task(left.task_id)
    assert resumed.state is ExecutionState.QUEUED
    assert resumed.human_inputs[0]["inbox_item_id"] == item.inbox_item_id
    assert resolved.resume_completed_at is not None
    assert supervisor.store.get_task(right.task_id).state is ExecutionState.RUNNING
    assert inbox.respond(respond(resolved, expected_version=None)).resume_claim_id == resolved.resume_claim_id


def test_denial_resumes_only_its_branch_as_structured_input(tmp_path: Path) -> None:
    clock = Clock()
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(
        root=tmp_path / "execution", startup_recovery=False,
        lease_seconds=30, heartbeat_seconds=5,
    ), clock=clock)
    task = supervisor.create_task(
        routing_decision_id="route-denial",
        side_effect_classification=SideEffectClassification.READ_ONLY,
        workspace_path=tmp_path,
    )
    supervisor.queue(task.task_id)
    leased, token = supervisor.acquire_lease(task.task_id, owner="worker")
    supervisor.start(task.task_id, attempt_id=leased.attempt_id, lease_token=token)
    checkpoint = supervisor.checkpoint(
        task.task_id,
        attempt_id=leased.attempt_id,
        lease_token=token,
        resume_payload={"cursor": "approval"},
    )
    inbox = service(tmp_path, clock, supervisor=supervisor)
    item = inbox.create(request(
        clock,
        task_id=task.task_id,
        branch_id=task.task_id,
        checkpoint_id=checkpoint.checkpoint_id,
        execution_attempt_id=leased.attempt_id,
    ))
    denied = inbox.respond(respond(item, operation=ResponseOperation.DENY))
    resumed = supervisor.store.get_task(task.task_id)
    assert denied.status is InboxStatus.DENIED
    assert resumed.state is ExecutionState.QUEUED
    assert resumed.human_inputs[-1]["response"]["operation"] == "deny"


def test_concurrent_approve_and_deny_accepts_one_terminal_response(tmp_path: Path) -> None:
    clock = Clock()
    inbox = service(tmp_path, clock)
    item = inbox.create(request(clock))

    def submit(operation: ResponseOperation) -> str:
        try:
            return inbox.respond(respond(item, operation=operation, idempotency_key=operation.value)).status.value
        except InboxConcurrentUpdateError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, [ResponseOperation.APPROVE, ResponseOperation.DENY]))
    assert results.count("conflict") == 1
    assert inbox.repository.get(item.inbox_item_id).status in {InboxStatus.APPROVED, InboxStatus.DENIED}


def test_concurrent_signers_publish_only_complete_key_files(tmp_path: Path, monkeypatch) -> None:
    key_path = tmp_path / "inbox" / "signing.key"
    first = ResponseTokenSigner(key_path)
    second = ResponseTokenSigner(key_path)
    first_write_started = Event()
    release_first_write = Event()
    write_lock = Lock()
    write_count = 0
    original_write = inbox_tokens.os.write

    def delayed_first_write(descriptor: int, value: bytes) -> int:
        nonlocal write_count
        with write_lock:
            write_count += 1
            is_first_write = write_count == 1
        if is_first_write:
            first_write_started.set()
            assert release_first_write.wait(timeout=5)
        return original_write(descriptor, value)

    monkeypatch.setattr(inbox_tokens.os, "write", delayed_first_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_digest = pool.submit(first.protected_digest, {"response": "value"})
        assert first_write_started.wait(timeout=5)
        second_digest = pool.submit(second.protected_digest, {"response": "value"})
        try:
            second_result = second_digest.result(timeout=5)
        finally:
            release_first_write.set()
        first_result = first_digest.result(timeout=5)

    assert first_result == second_result
    assert len(key_path.read_bytes()) >= 32


def test_response_idempotency_key_cannot_be_rebound(tmp_path: Path) -> None:
    clock = Clock()
    inbox = service(tmp_path, clock)
    item = inbox.create(request(clock))
    approved = inbox.respond(respond(item, idempotency_key="same-key"))
    assert inbox.respond(respond(approved, idempotency_key="same-key", expected_version=None)).status is InboxStatus.APPROVED
    with pytest.raises(InboxConcurrentUpdateError, match="different response content"):
        inbox.respond(respond(
            approved,
            operation=ResponseOperation.DENY,
            idempotency_key="same-key",
            expected_version=None,
        ))


def test_signed_tokens_expire_reject_replay_and_require_current_authorization(tmp_path: Path) -> None:
    clock = Clock()
    inbox = service(tmp_path, clock)
    item = inbox.create(request(clock))
    token, csrf = inbox.issue_response_token(item.inbox_item_id, actor_id="reviewer-1", operation=ResponseOperation.APPROVE, ttl_seconds=5)
    assert inbox.token_signer.verify_csrf(token, csrf)
    assert not inbox.token_signer.verify_csrf(token, csrf + "x")
    current = inbox.repository.get(item.inbox_item_id)
    with pytest.raises(PermissionError, match="currently authorized"):
        inbox.respond(respond(current, actor_id="intruder", signed_token=token))
    with pytest.raises(PermissionError, match="signature is invalid"):
        inbox.respond(respond(
            current,
            signed_token=token[:-1] + ("A" if token[-1] != "A" else "B"),
        ))
    clock.advance(6)
    with pytest.raises(PermissionError, match="expired"):
        inbox.respond(respond(current, signed_token=token))

    replay_item = inbox.create(request(
        clock,
        idempotency_key="token-replay-request",
        deduplication_key="token-replay-dedupe",
        expires_at=clock() + timedelta(hours=1),
    ))
    replay_token, _ = inbox.issue_response_token(
        replay_item.inbox_item_id,
        actor_id="reviewer-1",
        operation=ResponseOperation.APPROVE,
    )
    accepted = inbox.respond(respond(
        inbox.repository.get(replay_item.inbox_item_id),
        idempotency_key="token-replay-response",
        signed_token=replay_token,
    ))
    with pytest.raises(PermissionError, match="invalid after terminal"):
        inbox.respond(respond(
            accepted,
            idempotency_key="token-replay-response",
            signed_token=replay_token,
            expected_version=None,
        ))


def test_delegation_preserves_history_and_revokes_original_item(tmp_path: Path) -> None:
    clock = Clock()
    identities = [ReviewerIdentity(identity_id="reviewer-1"), ReviewerIdentity(identity_id="reviewer-2")]
    inbox = service(tmp_path, clock, identities=identities)
    original = inbox.create(request(clock))
    delegated = inbox.delegate(
        original.inbox_item_id,
        actor_id="reviewer-1",
        target=ReviewerAssignment(reviewer_type=ReviewerType.PERSON, reviewer_id="reviewer-2"),
    )
    assert inbox.repository.get(original.inbox_item_id).status is InboxStatus.DELEGATED
    assert delegated.delegated_from_item_id == original.inbox_item_id
    with pytest.raises(PermissionError):
        inbox.respond(respond(delegated, actor_id="reviewer-1"))
    assert inbox.respond(respond(delegated, actor_id="reviewer-2")).status is InboxStatus.APPROVED


def test_notification_failure_keeps_request_pending_and_records_attempt(tmp_path: Path) -> None:
    class BrokenAdapter:
        name = "broken"
        def deliver(self, notification):  # noqa: ANN001
            return NotificationResult(delivered=False, error="offline")

    clock = Clock()
    inbox = service(tmp_path, clock, adapters=[BrokenAdapter()])
    item = inbox.create(request(clock))
    assert item.status is InboxStatus.PENDING
    attempts = inbox.repository.delivery_attempts(item.inbox_item_id)
    assert attempts[0].status == "failed"
    assert attempts[0].error == "offline"


def test_expiry_rejects_old_response_and_never_approves(tmp_path: Path) -> None:
    clock = Clock()
    inbox = service(tmp_path, clock)
    item = inbox.create(request(clock, expires_at=clock() + timedelta(seconds=5)))
    clock.advance(6)
    expired = inbox.expire_due()
    assert expired[0].status is InboxStatus.EXPIRED
    with pytest.raises(InboxConcurrentUpdateError):
        inbox.respond(respond(item))


def test_sensitive_clarification_answer_uses_protected_reference(tmp_path: Path) -> None:
    clock = Clock()
    inbox = service(tmp_path, clock)
    item = inbox.create(request(
        clock,
        request_type=InboxRequestType.CLARIFICATION,
        requested_fields=[ClarificationField(field_id="credential", prompt="Credential", sensitive=True)],
        allowed_responses=[ResponseOperation.ANSWER],
    ))
    answered = inbox.respond(respond(
        item,
        operation=ResponseOperation.ANSWER,
        answer={"credential": "private-value"},
    ))
    assert answered.response is not None
    assert answered.response.answer["protected_response_ref"].startswith("protected-response:")
    assert answered.response_idempotency_digests["response-1"].startswith("hmac-sha256:")
    item_text = next((tmp_path / "inbox" / "items").glob("*.json")).read_text(encoding="utf-8")
    assert "private-value" not in item_text
    with pytest.raises(InboxConcurrentUpdateError, match="different response content"):
        inbox.respond(respond(
            answered,
            operation=ResponseOperation.ANSWER,
            answer={"credential": "different-private-value"},
            expected_version=None,
        ))
    consumed = inbox.consume_for_agent(
        item.inbox_item_id,
        requesting_agent_id="agent-1",
        task_id="task-1",
    )
    assert consumed.answer == {"credential": "private-value"}
    with pytest.raises(PermissionError):
        inbox.consume_for_agent(
            item.inbox_item_id,
            requesting_agent_id="another-agent",
            task_id="task-1",
        )


def test_recovery_resumes_response_persisted_before_controller_restart(tmp_path: Path) -> None:
    clock = Clock()
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(
        root=tmp_path / "execution", startup_recovery=False, lease_seconds=30, heartbeat_seconds=5,
    ), clock=clock)
    task = supervisor.create_task(routing_decision_id="route", side_effect_classification=SideEffectClassification.READ_ONLY, workspace_path=tmp_path)
    supervisor.queue(task.task_id)
    leased, token = supervisor.acquire_lease(task.task_id, owner="worker")
    supervisor.start(task.task_id, attempt_id=leased.attempt_id, lease_token=token)
    checkpoint = supervisor.checkpoint(task.task_id, attempt_id=leased.attempt_id, lease_token=token, resume_payload={})
    active = service(tmp_path, clock, supervisor=supervisor)
    item = active.create(request(clock, task_id=task.task_id, branch_id=task.task_id, checkpoint_id=checkpoint.checkpoint_id))
    interrupted = service(tmp_path, clock)
    interrupted.respond(respond(interrupted.repository.get(item.inbox_item_id)))
    recovered = service(tmp_path, clock, supervisor=supervisor)
    report = recovered.reconcile()
    assert task.task_id in report.resumed
    assert supervisor.store.get_task(task.task_id).state is ExecutionState.QUEUED


def test_role_without_members_stays_pending_with_configuration_error(tmp_path: Path) -> None:
    clock = Clock()
    inbox = service(tmp_path, clock, identities=[])
    item = inbox.create(request(
        clock,
        reviewer=ReviewerAssignment(reviewer_type=ReviewerType.ROLE, reviewer_id="finance"),
    ))
    assert item.status is InboxStatus.PENDING
    assert "No eligible reviewer" in item.configuration_error


def test_reviewer_assignment_respects_tenant_and_project_scope(tmp_path: Path) -> None:
    clock = Clock()
    scoped = ReviewerIdentity(
        identity_id="reviewer-1",
        tenant_ids={"tenant-a"},
        project_ids={"project-a"},
    )
    inbox = service(tmp_path, clock, identities=[scoped])
    item = inbox.create(request(
        clock,
        tenant_id="tenant-b",
        project_id="project-b",
    ))
    assert item.status is InboxStatus.PENDING
    assert item.eligible_reviewer_ids == []
    with pytest.raises(PermissionError):
        inbox.get(item.inbox_item_id, actor_id="reviewer-1")


def test_transactional_permission_event_maps_to_authoritative_inbox(tmp_path: Path) -> None:
    clock = Clock()
    inbox = service(tmp_path, clock)
    state = tmp_path / "actions"
    gateway = ActionGateway(
        store=ActionStore(state),
        policy=ActionPolicy(PolicyConfig(
            workspace_roots=(tmp_path,),
            approval_reviewer_id="reviewer-1",
        )),
        approvals=ApprovalRegistry(state / "approvals"),
        inbox_service=inbox,
    )
    target = tmp_path / "delete-me.txt"
    target.write_text("recoverable", encoding="utf-8")
    adapter = FileActionAdapter(
        workspace_root=tmp_path,
        operation="delete",
        path=target.name,
        content="",
        parent_task_id="legacy-task",
        actor="user",
        originating_agent="agent-1",
        idempotency_key="durable-delete-key",
        snapshot_root=tmp_path / "snapshots",
    )
    with pytest.raises(ApprovalRequired) as pending:
        gateway.execute(adapter)
    item = inbox.repository.find_for_action(pending.value.action.action_id)[0]
    assert pending.value.inbox_item_id == item.inbox_item_id
    assert item.permission_request_id == pending.value.action.action_id
    assert item.minimal_context["effect_labels"] == {
        "reversible": True,
        "compensatable": False,
        "irreversible": False,
        "externally_visible": False,
        "data_disclosing": False,
        "potentially_billable": False,
    }
    protected = inbox.repository.read_protected_context(item.protected_context_ref)
    assert protected["effect_labels"] == item.minimal_context["effect_labels"]
    approval_id = gateway.approve(
        pending.value.action.action_id,
        approved_by="trusted-test-channel",
        reviewer_id="reviewer-1",
    )
    assert inbox.repository.get(item.inbox_item_id).status is InboxStatus.APPROVED
    assert gateway.execute(adapter, approval_id=approval_id).action.state.value == "committed"


def test_resource_change_after_approval_supersedes_inbox_item(tmp_path: Path) -> None:
    clock = Clock()
    inbox = service(tmp_path, clock)
    state = tmp_path / "actions"
    gateway = ActionGateway(
        store=ActionStore(state),
        policy=ActionPolicy(PolicyConfig(
            workspace_roots=(tmp_path,),
            approval_reviewer_id="reviewer-1",
        )),
        approvals=ApprovalRegistry(state / "approvals"),
        inbox_service=inbox,
    )
    target = tmp_path / "change-before-execute.txt"
    target.write_text("previewed", encoding="utf-8")
    adapter = FileActionAdapter(
        workspace_root=tmp_path,
        operation="delete",
        path=target.name,
        content="",
        parent_task_id="legacy-task",
        actor="user",
        originating_agent="agent-1",
        idempotency_key="changed-resource-key",
        snapshot_root=tmp_path / "snapshots",
    )
    with pytest.raises(ApprovalRequired) as pending:
        gateway.execute(adapter)
    approval_id = gateway.approve(
        pending.value.action.action_id,
        approved_by="trusted-test-channel",
        reviewer_id="reviewer-1",
    )
    target.write_text("changed", encoding="utf-8")
    with pytest.raises(ActionInvalidatedError):
        gateway.execute(adapter, approval_id=approval_id)
    item = inbox.repository.find_for_action(pending.value.action.action_id)[0]
    assert item.status is InboxStatus.SUPERSEDED


def test_remote_permission_legacy_id_recovers_from_durable_inbox(tmp_path: Path) -> None:
    clock = Clock()
    first_inbox = service(tmp_path, clock)
    remote_request = RemoteExecutionRequest(
        job_id="remote-job-1",
        session_id="session-1",
        provider="remote-ssh",
        target=SSHTarget(host="example.test", user="operator"),
        authentication=SSHAuthentication(mode="key_path", key_path="/keys/test"),
        command=RemoteCommand(argv=["tail", "-n", "5", "/var/log/app.log"]),
    )
    first = RemoteExecutionService(
        target_policy=TargetPolicy(),
        inbox_service=first_inbox,
    )
    first.submit(remote_request)
    permission_id = first.pending_permissions()[0]["permission_request_id"]

    restarted_inbox = service(tmp_path, clock)
    restarted = RemoteExecutionService(
        target_policy=TargetPolicy(),
        inbox_service=restarted_inbox,
    )
    assert restarted.pending_permissions()[0]["permission_request_id"] == permission_id
    approved = restarted.approve_permission(
        permission_id,
        reviewer_id=getpass.getuser(),
    )
    assert approved.state.value == "queued"
    item = next(
        item
        for item in restarted_inbox.repository.list()
        if item.permission_request_id == permission_id
    )
    assert item.status is InboxStatus.APPROVED
    assert item.minimal_context["effect_labels"]["externally_visible"] is True
    assert item.minimal_context["effect_labels"]["potentially_billable"] is None


def test_task_cancellation_and_replanning_close_stale_requests(tmp_path: Path) -> None:
    clock = Clock()
    inbox = service(tmp_path, clock)
    cancelled = inbox.create(request(clock))
    changed = inbox.cancel_for_task(cancelled.task_id, reason="task cancelled")
    assert changed[0].status is InboxStatus.CANCELLED

    stale = inbox.create(request(
        clock,
        task_id="task-2",
        branch_id="task-2",
        action_intent_id="act_stale",
        idempotency_key="request-stale",
        deduplication_key="dedupe-stale",
    ))
    inbox.supersede_for_action(stale.action_intent_id)
    assert inbox.repository.get(stale.inbox_item_id).status is InboxStatus.SUPERSEDED


def test_material_action_change_supersedes_prior_approval(tmp_path: Path) -> None:
    clock = Clock()
    digest = {"value": "before"}
    root = tmp_path / "inbox"
    inbox = HumanInboxService(
        repository=LocalInboxRepository(root),
        identities=StaticIdentityDirectory([ReviewerIdentity(identity_id="reviewer-1")]),
        token_signer=ResponseTokenSigner(root / "signing.key", clock=clock),
        action_digest_resolver=lambda _action_id: digest["value"],
        clock=clock,
    )
    item = inbox.create(request(
        clock,
        action_intent_id="act_changed",
        action_digest="before",
    ))
    digest["value"] = "after"
    with pytest.raises(PermissionError, match="material changed"):
        inbox.respond(respond(item, current_action_digest="before"))
    assert inbox.repository.get(item.inbox_item_id).status is InboxStatus.SUPERSEDED


def test_notifications_and_audit_expose_only_disclosed_snapshot(tmp_path: Path) -> None:
    class CapturingAdapter:
        name = "capture"
        def __init__(self) -> None:
            self.notifications = []
        def deliver(self, notification):  # noqa: ANN001
            self.notifications.append(notification)
            return NotificationResult(delivered=True)

    clock = Clock()
    adapter = CapturingAdapter()
    inbox = service(tmp_path, clock, adapters=[adapter])
    item = inbox.create(request(
        clock,
        protected_context={"credential": "private-value"},
    ))
    assert "private-value" not in adapter.notifications[0].model_dump_json()
    created = next(
        event
        for event in inbox.repository.audit_for_item(item.inbox_item_id)
        if event.event_type == "request_created"
    )
    assert created.details["disclosed_fields"] == ["resource_count"]
    assert "protected_context_ref" not in created.details["disclosed_snapshot"]


def test_concurrent_reminder_scans_claim_only_one_delivery(tmp_path: Path) -> None:
    class CapturingAdapter:
        name = "capture"
        def deliver(self, _notification):  # noqa: ANN001
            return NotificationResult(delivered=True)

    clock = Clock()
    inbox = service(tmp_path, clock, adapters=[CapturingAdapter()])
    item = inbox.create(request(
        clock,
        reminder_policy=ReminderPolicy(interval_seconds=60, max_reminders=1),
    ))
    clock.advance(61)
    with ThreadPoolExecutor(max_workers=2) as pool:
        scans = list(pool.map(lambda _value: inbox.send_due_reminders(), range(2)))
    assert sum(len(attempts) for attempts in scans) == 1
    assert inbox.repository.get(item.inbox_item_id).reminder_count == 1


def test_approved_legacy_action_has_one_durable_execution_claim(tmp_path: Path) -> None:
    clock = Clock()
    inbox = service(tmp_path, clock)
    item = inbox.create(request(clock))
    approved = inbox.respond(respond(item))
    claim_id = inbox.claim_action_execution(approved.inbox_item_id)
    with pytest.raises(InboxConcurrentUpdateError, match="already claimed"):
        inbox.claim_action_execution(approved.inbox_item_id)
    completed = inbox.complete_action_execution(
        approved.inbox_item_id,
        execution_claim_id=claim_id,
        result_digest="sha256:result",
    )
    assert completed.execution_completed_at is not None
    with pytest.raises(InboxConcurrentUpdateError, match="already completed"):
        inbox.claim_action_execution(approved.inbox_item_id)
