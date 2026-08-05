"""Lifecycle service for durable approval and clarification requests."""

from __future__ import annotations

from datetime import datetime, timedelta
import re
from typing import Any, Callable, Protocol
from uuid import uuid4

from mana_agent.utils.redaction import redact_secrets

from .identity import IdentityDirectory
from .models import (
    AgentInboxObservation,
    DeliveryAttempt,
    ExpiryBehavior,
    ExpectedResponseType,
    HumanResponse,
    InboxAuditEvent,
    InboxItem,
    InboxQuery,
    InboxRequest,
    InboxRequestType,
    InboxStatus,
    ReconciliationReport,
    ResponseOperation,
    ResponseSubmission,
    ReviewerAssignment,
    TERMINAL_STATUSES,
    UNRESOLVED_STATUSES,
    canonical_digest,
    utc_now,
)
from .notifications import InboxNotification, NotificationAdapter
from .repository import InboxConcurrentUpdateError, InboxRepository
from .tokens import ResponseTokenSigner


class BranchController(Protocol):
    store: Any
    def suspend_for_human_input(self, task_id: str, *, inbox_item_id: str, checkpoint_id: str, request_type: str) -> Any: ...
    def resume_from_human_input(self, task_id: str, *, inbox_item_id: str, checkpoint_id: str, resume_claim_id: str, structured_response: dict[str, Any]) -> Any: ...
    def restore_human_wait(self, task_id: str, *, inbox_item_id: str, checkpoint_id: str, request_type: str) -> Any: ...
    def cancel(self, task_id: str, *, reason: str, propagate: bool = True) -> list[str]: ...


EventSink = Callable[[dict[str, Any]], None]


class _IdempotentResponseReplay(Exception):
    pass


class HumanInboxService:
    def __init__(
        self,
        *,
        repository: InboxRepository,
        identities: IdentityDirectory,
        token_signer: ResponseTokenSigner,
        branch_controller: BranchController | None = None,
        notification_adapters: list[NotificationAdapter] | None = None,
        event_sink: EventSink | None = None,
        terminal_response_handler: Callable[[InboxItem], None] | None = None,
        action_digest_resolver: Callable[[str], str | None] | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.repository = repository
        self.identities = identities
        self.token_signer = token_signer
        self.branch_controller = branch_controller
        self.notification_adapters = list(notification_adapters or [])
        self.event_sink = event_sink
        self.terminal_response_handler = terminal_response_handler
        self.action_digest_resolver = action_digest_resolver
        self.clock = clock

    def create(self, request: InboxRequest) -> InboxItem:
        if redact_secrets(request.minimal_context) != request.minimal_context:
            raise ValueError("minimal inbox context contains secret material")
        if redact_secrets(request.title) != request.title or redact_secrets(request.summary) != request.summary:
            raise ValueError("inbox title or summary contains secret material")
        eligible = [
            identity
            for identity in self.identities.resolve(request.reviewer)
            if request.tenant_id in identity.tenant_ids
            and (not identity.project_ids or request.project_id in identity.project_ids)
        ]
        if any(
            identity.identity_id == request.requested_by_agent_id
            for identity in eligible
        ):
            raise PermissionError("the requesting agent cannot be its own human reviewer")
        configuration_error = ""
        if not eligible:
            configuration_error = (
                f"No eligible reviewer is configured for {request.reviewer.reviewer_type.value} "
                f"{request.reviewer.reviewer_id}."
            )
        item = InboxItem(
            request_type=request.request_type,
            status=(InboxStatus.RECORDED if request.request_type is InboxRequestType.NOTICE else InboxStatus.PENDING),
            tenant_id=request.tenant_id,
            project_id=request.project_id,
            task_id=request.task_id,
            branch_id=request.branch_id,
            parent_task_id=request.parent_task_id,
            checkpoint_id=request.checkpoint_id,
            execution_attempt_id=request.execution_attempt_id,
            policy_decision_id=request.policy_decision_id,
            permission_request_id=request.permission_request_id,
            action_intent_id=request.action_intent_id,
            action_digest=request.action_digest,
            requested_by_agent_id=request.requested_by_agent_id,
            assigned_reviewer_type=request.reviewer.reviewer_type,
            assigned_reviewer_id=request.reviewer.reviewer_id,
            eligible_reviewer_ids=[identity.identity_id for identity in eligible],
            title=request.title,
            summary=request.summary,
            risk_level=request.risk_level,
            requested_fields=request.requested_fields,
            allowed_responses=request.allowed_responses,
            editable_parameters=request.editable_parameters,
            minimal_context=request.minimal_context,
            disclosed_fields=request.disclosed_fields,
            reversibility=request.reversibility,
            other_work_continues=request.other_work_continues,
            created_at=self.clock(),
            expires_at=request.expires_at,
            escalation_policy=request.escalation_policy,
            reminder_policy=request.reminder_policy,
            idempotency_key=request.idempotency_key,
            deduplication_key=request.deduplication_key,
            configuration_error=configuration_error,
        )
        item, created = self.repository.create(item, protected_context=request.protected_context)
        if not created:
            return item
        self._audit(
            "request_created",
            item,
            disclosed_fields=item.disclosed_fields,
            disclosed_snapshot=item.card(),
            disclosed_snapshot_digest=canonical_digest(item.card()),
        )
        self._audit(
            "reviewer_resolved",
            item,
            eligible_reviewer_ids=item.eligible_reviewer_ids,
            configuration_error=item.configuration_error,
        )
        if item.request_type is not InboxRequestType.NOTICE and self.branch_controller is not None and item.checkpoint_id:
            self.branch_controller.suspend_for_human_input(
                item.task_id,
                inbox_item_id=item.inbox_item_id,
                checkpoint_id=item.checkpoint_id,
                request_type=item.request_type.value,
            )
            self._audit("branch_suspended", item, checkpoint_id=item.checkpoint_id)
        if item.request_type is InboxRequestType.NOTICE:
            self._emit("inbox.item.recorded", item)
        elif item.eligible_reviewer_ids:
            self._deliver(item)
        else:
            self._emit("inbox.item.configuration_error", item)
        return self.repository.get(item.inbox_item_id)

    def get(self, inbox_item_id: str, *, actor_id: str) -> InboxItem:
        item = self.repository.get(inbox_item_id)
        self._authorize(item, actor_id)
        self._audit("request_viewed", item, reviewer_id=actor_id)
        return item

    def list(self, query: InboxQuery, *, actor_id: str) -> list[InboxItem]:
        identity = self.identities.get(actor_id)
        rows = self.repository.list(query)
        authorized: list[InboxItem] = []
        for item in rows:
            assignment = ReviewerAssignment(
                reviewer_type=item.assigned_reviewer_type,
                reviewer_id=item.assigned_reviewer_id,
            )
            eligible_now = {
                reviewer.identity_id for reviewer in self.identities.resolve(assignment)
            }
            scope_allowed = (
                item.tenant_id == "local"
                if identity is None
                else item.tenant_id in identity.tenant_ids
                and (not identity.project_ids or item.project_id in identity.project_ids)
            )
            if not scope_allowed:
                continue
            if actor_id in eligible_now:
                authorized.append(item)
        return authorized

    def issue_response_token(
        self,
        inbox_item_id: str,
        *,
        actor_id: str,
        operation: ResponseOperation,
        ttl_seconds: int = 900,
    ) -> tuple[str, str]:
        item = self.repository.get(inbox_item_id)
        self._authorize(item, actor_id)
        if item.status not in UNRESOLVED_STATUSES or operation not in item.allowed_responses:
            raise PermissionError("inbox item no longer accepts this response")
        if item.expires_at <= self.clock():
            self._expire_one(item)
            raise PermissionError("inbox item expired before a response token was issued")
        token, nonce_hash = self.token_signer.issue(
            inbox_item_id=item.inbox_item_id,
            reviewer_scope=actor_id,
            operation=operation,
            expires_at=item.expires_at,
            ttl_seconds=ttl_seconds,
        )

        def register(current: InboxItem) -> None:
            if current.status not in UNRESOLVED_STATUSES:
                raise PermissionError("inbox item is terminal")
            current.token_nonce_hashes.append(nonce_hash)

        self.repository.update(item.inbox_item_id, register)
        return token, self.token_signer.csrf_token(token)

    def observe_for_agent(
        self,
        inbox_item_id: str,
        *,
        requesting_agent_id: str,
        task_id: str,
    ) -> AgentInboxObservation:
        """Allow only the originating task agent to observe durable status/input."""
        item = self.repository.get(inbox_item_id)
        if (
            item.requested_by_agent_id != requesting_agent_id
            or item.task_id != task_id
        ):
            raise PermissionError("agent is not authorized for this inbox task input")
        self._audit(
            "request_observed_by_agent",
            item,
            requesting_agent_id=requesting_agent_id,
        )
        return AgentInboxObservation(
            inbox_item_id=item.inbox_item_id,
            task_id=item.task_id,
            branch_id=item.branch_id,
            status=item.status,
            request_type=item.request_type,
            response=item.response,
            resume_completed=item.resume_completed_at is not None,
        )

    def consume_for_agent(
        self,
        inbox_item_id: str,
        *,
        requesting_agent_id: str,
        task_id: str,
    ) -> HumanResponse:
        """Resolve one terminal structured input for its exact originating task."""
        self.observe_for_agent(
            inbox_item_id,
            requesting_agent_id=requesting_agent_id,
            task_id=task_id,
        )
        item = self.repository.get(inbox_item_id)
        if item.response is None:
            raise ValueError("inbox item does not contain a human response")
        response = item.response.model_copy(deep=True)
        reference = str(response.answer.get("protected_response_ref") or "")
        if reference:
            response.answer = self.repository.read_protected_response(reference)
            self._audit(
                "protected_response_consumed",
                item,
                requesting_agent_id=requesting_agent_id,
                protected_response_ref=reference,
            )
        return response

    def respond(self, submission: ResponseSubmission) -> InboxItem:
        initial = self.repository.get(submission.inbox_item_id)
        response_digest = self.token_signer.protected_digest({
            "operation": submission.operation.value,
            "actor_id": submission.actor_id,
            "channel": submission.channel,
            "answer": submission.answer,
            "comment": submission.comment,
        })
        if submission.idempotency_key in initial.response_idempotency_keys:
            if submission.signed_token:
                self._audit(
                    "response_rejected",
                    initial,
                    reviewer_id=submission.actor_id,
                    reason="token_replay",
                )
                raise PermissionError("response token is invalid after terminal response")
            recorded = initial.response_idempotency_digests.get(submission.idempotency_key)
            if recorded and recorded != response_digest:
                raise InboxConcurrentUpdateError(
                    "response idempotency key is bound to different response content"
                )
            self._audit(
                "response_idempotent_replay",
                initial,
                reviewer_id=submission.actor_id,
            )
            self._resume_terminal_if_needed(initial)
            return self.repository.get(initial.inbox_item_id)
        if initial.status in TERMINAL_STATUSES:
            self._audit("response_rejected", initial, reviewer_id=submission.actor_id, reason="terminal_state")
            raise InboxConcurrentUpdateError(f"inbox item is already terminal: {initial.status.value}")
        if initial.expires_at <= self.clock():
            self._expire_one(initial)
            raise PermissionError("inbox item expired before the response was accepted")
        self._authorize(initial, submission.actor_id)
        if submission.operation not in initial.allowed_responses:
            self._audit(
                "response_rejected",
                initial,
                reviewer_id=submission.actor_id,
                reason="operation_not_allowed",
            )
            raise PermissionError("response operation is not allowed for this inbox item")
        if initial.action_digest:
            resolved_digest = (
                self.action_digest_resolver(initial.action_intent_id)
                if self.action_digest_resolver is not None
                else None
            )
            current_digest = (
                submission.current_action_digest
                if resolved_digest is None
                else resolved_digest
            )
            if current_digest != initial.action_digest:
                self.supersede_for_action(initial.action_intent_id)
                self._audit(
                    "response_rejected",
                    initial,
                    reviewer_id=submission.actor_id,
                    reason="action_material_changed",
                )
                raise PermissionError("action material changed; the prior approval was superseded")

        signature = ""
        nonce_hash = ""
        if submission.signed_token:
            try:
                claims = self.token_signer.verify(submission.signed_token)
            except PermissionError:
                self._audit(
                    "response_rejected",
                    initial,
                    reviewer_id=submission.actor_id,
                    reason="token_validation",
                )
                raise
            nonce_hash = self.token_signer.nonce_hash(claims.nonce)
            if (
                claims.inbox_item_id != initial.inbox_item_id
                or claims.operation is not submission.operation
                or claims.reviewer_scope != submission.actor_id
                or nonce_hash not in initial.token_nonce_hashes
            ):
                self._audit("response_rejected", initial, reviewer_id=submission.actor_id, reason="token_scope")
                raise PermissionError("response token does not authorize this operation")
            signature = self.token_signer.response_signature(submission.signed_token)
        else:
            signature = "local:" + canonical_digest({
                "item": initial.inbox_item_id,
                "actor": submission.actor_id,
                "operation": submission.operation.value,
                "idempotency_key": submission.idempotency_key,
            })

        answer = self._validate_answer(initial, submission)
        sensitive = any(field.sensitive for field in initial.requested_fields)
        if sensitive and answer:
            reference = self.repository.save_protected_response(initial.inbox_item_id, answer)
            answer = {"protected_response_ref": reference}
        target_status = {
            ResponseOperation.APPROVE: InboxStatus.APPROVED,
            ResponseOperation.DENY: InboxStatus.DENIED,
            ResponseOperation.ANSWER: InboxStatus.ANSWERED,
        }[submission.operation]

        def accept(item: InboxItem) -> None:
            if submission.expected_version is not None and item.version != submission.expected_version:
                raise InboxConcurrentUpdateError("inbox version changed before response submission")
            if submission.idempotency_key in item.response_idempotency_keys:
                recorded = item.response_idempotency_digests.get(submission.idempotency_key)
                if recorded and recorded != response_digest:
                    raise InboxConcurrentUpdateError(
                        "response idempotency key is bound to different response content"
                    )
                raise _IdempotentResponseReplay
            if item.status not in UNRESOLVED_STATUSES:
                raise InboxConcurrentUpdateError(f"inbox item is already terminal: {item.status.value}")
            if item.expires_at <= self.clock():
                raise PermissionError("inbox item expired before response commit")
            if nonce_hash and nonce_hash not in item.token_nonce_hashes:
                raise PermissionError("response token was already consumed")
            item.response = HumanResponse(
                operation=submission.operation,
                answer=answer,
                comment=submission.comment,
                submitted_at=self.clock(),
            )
            # The model requires response evidence for responded terminal states;
            # assign it before the status so assignment validation stays atomic.
            item.status = target_status
            item.response_actor_id = submission.actor_id
            item.response_channel = submission.channel
            item.response_signature = signature
            item.responded_at = self.clock()
            item.response_idempotency_keys.append(submission.idempotency_key)
            item.response_idempotency_digests[submission.idempotency_key] = response_digest
            item.token_nonce_hashes.clear()

        try:
            item, _ = self.repository.update(initial.inbox_item_id, accept)
        except _IdempotentResponseReplay:
            replayed = self.repository.get(initial.inbox_item_id)
            self._audit(
                "response_idempotent_replay",
                replayed,
                reviewer_id=submission.actor_id,
            )
            self._resume_terminal_if_needed(replayed)
            return self.repository.get(replayed.inbox_item_id)
        except InboxConcurrentUpdateError:
            latest = self.repository.get(initial.inbox_item_id)
            self._audit(
                "response_rejected",
                latest,
                reviewer_id=submission.actor_id,
                reason=(
                    "terminal_state"
                    if latest.status in TERMINAL_STATUSES
                    else "concurrent_update"
                ),
            )
            raise
        self._audit(
            "response_submitted",
            item,
            reviewer_id=submission.actor_id,
            operation=submission.operation.value,
            channel=submission.channel,
            disclosed_fields=item.disclosed_fields,
            disclosed_snapshot=item.card(),
            disclosed_snapshot_digest=canonical_digest(item.card()),
        )
        self._emit("inbox.item.responded", item)
        if self.terminal_response_handler is not None:
            self.terminal_response_handler(item)
        self._resume_terminal_if_needed(item)
        return self.repository.get(item.inbox_item_id)

    def expire_due(self) -> list[InboxItem]:
        expired: list[InboxItem] = []
        for item in self.repository.due_for_expiration(self.clock()):
            try:
                expired.append(self._expire_one(item))
            except InboxConcurrentUpdateError:
                continue
        return expired

    def _expire_one(self, initial: InboxItem) -> InboxItem:
        def expire(item: InboxItem) -> None:
            if item.status not in UNRESOLVED_STATUSES:
                raise InboxConcurrentUpdateError("inbox item resolved during expiration")
            item.status = InboxStatus.EXPIRED
            item.responded_at = self.clock()
            item.token_nonce_hashes.clear()

        item, _ = self.repository.update(initial.inbox_item_id, expire)
        self._audit("request_expired", item)
        behavior = item.escalation_policy.expiry_behavior
        should_escalate = behavior is ExpiryBehavior.ESCALATE and not item.action_intent_id
        if self.terminal_response_handler is not None and not should_escalate:
            self.terminal_response_handler(item)
        if behavior is ExpiryBehavior.CANCEL_BRANCH or behavior is ExpiryBehavior.DENY_BY_DEFAULT:
            if self.branch_controller is not None and item.checkpoint_id:
                self.branch_controller.cancel(item.task_id, reason=f"human inbox item expired: {item.inbox_item_id}", propagate=False)
        elif behavior is ExpiryBehavior.REQUEST_REPLANNING:
            self._audit("expiry_action_required", item, behavior=behavior.value)
        elif behavior is ExpiryBehavior.ESCALATE:
            if item.action_intent_id:
                self._audit(
                    "expiry_action_required",
                    item,
                    behavior="escalate_requires_new_policy_decision",
                )
            else:
                self._escalate_expired(item)
        self._emit("inbox.item.expired", item)
        return item

    def _escalate_expired(self, source: InboxItem) -> InboxItem:
        policy = source.escalation_policy
        assert policy.target is not None
        remaining = policy.max_escalations - 1
        next_policy = policy.model_copy(update={
            "max_escalations": remaining,
            "expiry_behavior": ExpiryBehavior.ESCALATE if remaining else ExpiryBehavior.REMAIN_BLOCKED,
            "target": policy.target if remaining else None,
        })
        protected_context = (
            self.repository.read_protected_context(source.protected_context_ref)
            if source.protected_context_ref
            else {}
        )
        escalated = self.create(InboxRequest(
            request_type=source.request_type,
            tenant_id=source.tenant_id,
            project_id=source.project_id,
            task_id=source.task_id,
            branch_id=source.branch_id,
            parent_task_id=source.parent_task_id,
            checkpoint_id=source.checkpoint_id,
            execution_attempt_id=source.execution_attempt_id,
            policy_decision_id=source.policy_decision_id,
            permission_request_id=source.permission_request_id,
            action_intent_id=source.action_intent_id,
            action_digest=source.action_digest,
            requested_by_agent_id=source.requested_by_agent_id,
            reviewer=policy.target,
            title=source.title,
            summary=source.summary,
            risk_level=source.risk_level,
            requested_fields=source.requested_fields,
            allowed_responses=source.allowed_responses,
            editable_parameters=source.editable_parameters,
            minimal_context=source.minimal_context,
            protected_context=protected_context,
            disclosed_fields=source.disclosed_fields,
            reversibility=source.reversibility,
            other_work_continues=source.other_work_continues,
            expires_at=self.clock() + timedelta(seconds=policy.escalation_ttl_seconds),
            escalation_policy=next_policy,
            reminder_policy=source.reminder_policy,
            idempotency_key=f"escalate:{source.inbox_item_id}:{policy.target.reviewer_type.value}:{policy.target.reviewer_id}",
            deduplication_key=f"escalate:{source.deduplication_key}:{policy.target.reviewer_type.value}:{policy.target.reviewer_id}",
        ))
        def link_original(current: InboxItem) -> None:
            current.delegated_to_item_id = escalated.inbox_item_id
        self.repository.update(source.inbox_item_id, link_original)
        def link_new(current: InboxItem) -> None:
            current.delegated_from_item_id = source.inbox_item_id
        escalated, _ = self.repository.update(escalated.inbox_item_id, link_new)
        self._audit("request_delegated", source, delegated_to_item_id=escalated.inbox_item_id, reason="expiry_escalation")
        return escalated

    def send_due_reminders(self) -> list[DeliveryAttempt]:
        attempts: list[DeliveryAttempt] = []
        if not self.notification_adapters:
            return attempts
        for candidate in self.repository.due_for_reminder(self.clock()):
            def claim_reminder(current: InboxItem) -> None:
                policy = current.reminder_policy
                baseline = current.last_reminded_at or current.delivered_at or current.created_at
                if (
                    current.status not in UNRESOLVED_STATUSES
                    or current.reminder_count >= policy.max_reminders
                    or (self.clock() - baseline).total_seconds() < policy.interval_seconds
                ):
                    raise InboxConcurrentUpdateError("reminder is no longer due")
                # Claim before the external delivery side effect. A crash can
                # delay a reminder, but concurrent schedulers cannot spam it.
                current.reminder_count += 1
                current.last_reminded_at = self.clock()

            try:
                item, _ = self.repository.update(candidate.inbox_item_id, claim_reminder)
            except InboxConcurrentUpdateError:
                continue
            attempts.extend(self._deliver(item, reminder=True, reminder_claimed=True))
        return attempts

    def cancel_for_task(self, task_id: str, *, reason: str) -> list[InboxItem]:
        changed: list[InboxItem] = []
        for item in self.repository.list(InboxQuery(task_id=task_id, statuses=set(UNRESOLVED_STATUSES))):
            def cancel(current: InboxItem) -> None:
                if current.status not in UNRESOLVED_STATUSES:
                    raise InboxConcurrentUpdateError("inbox item resolved during cancellation")
                current.status = InboxStatus.CANCELLED
                current.responded_at = self.clock()
                current.token_nonce_hashes.clear()
            try:
                cancelled, _ = self.repository.update(item.inbox_item_id, cancel)
            except InboxConcurrentUpdateError:
                continue
            self._audit("request_cancelled", cancelled, reason=reason)
            if self.terminal_response_handler is not None:
                self.terminal_response_handler(cancelled)
            changed.append(cancelled)
        return changed

    def supersede_for_action(self, action_intent_id: str, *, replacement_item_id: str = "") -> list[InboxItem]:
        changed: list[InboxItem] = []
        candidates = [
            item
            for item in self.repository.find_for_action(action_intent_id)
            if item.status in UNRESOLVED_STATUSES or item.status is InboxStatus.APPROVED
        ]
        for item in candidates:
            def supersede(current: InboxItem) -> None:
                if current.status not in UNRESOLVED_STATUSES and current.status is not InboxStatus.APPROVED:
                    raise InboxConcurrentUpdateError("inbox item resolved during supersession")
                current.status = InboxStatus.SUPERSEDED
                current.superseded_by_item_id = replacement_item_id
                current.token_nonce_hashes.clear()
            try:
                superseded, _ = self.repository.update(item.inbox_item_id, supersede)
            except InboxConcurrentUpdateError:
                continue
            self._audit("request_superseded", superseded, replacement_item_id=replacement_item_id)
            if self.terminal_response_handler is not None:
                self.terminal_response_handler(superseded)
            changed.append(superseded)
        return changed

    def delegate(self, inbox_item_id: str, *, actor_id: str, target: ReviewerAssignment) -> InboxItem:
        source = self.repository.get(inbox_item_id)
        self._authorize(source, actor_id)
        if source.status not in UNRESOLVED_STATUSES:
            raise InboxConcurrentUpdateError("only a pending inbox item may be delegated")
        if source.expires_at <= self.clock():
            self._expire_one(source)
            raise PermissionError("expired inbox items cannot be delegated")
        protected_context = (
            self.repository.read_protected_context(source.protected_context_ref)
            if source.protected_context_ref
            else {}
        )
        request = InboxRequest(
            request_type=source.request_type,
            tenant_id=source.tenant_id,
            project_id=source.project_id,
            task_id=source.task_id,
            branch_id=source.branch_id,
            parent_task_id=source.parent_task_id,
            checkpoint_id=source.checkpoint_id,
            execution_attempt_id=source.execution_attempt_id,
            policy_decision_id=source.policy_decision_id,
            permission_request_id=source.permission_request_id,
            action_intent_id=source.action_intent_id,
            action_digest=source.action_digest,
            requested_by_agent_id=source.requested_by_agent_id,
            reviewer=target,
            title=source.title,
            summary=source.summary,
            risk_level=source.risk_level,
            requested_fields=source.requested_fields,
            allowed_responses=source.allowed_responses,
            editable_parameters=source.editable_parameters,
            minimal_context=source.minimal_context,
            protected_context=protected_context,
            disclosed_fields=source.disclosed_fields,
            reversibility=source.reversibility,
            other_work_continues=source.other_work_continues,
            expires_at=source.expires_at,
            escalation_policy=source.escalation_policy,
            reminder_policy=source.reminder_policy,
            idempotency_key=f"delegate:{source.inbox_item_id}:{target.reviewer_type.value}:{target.reviewer_id}",
            deduplication_key=f"delegate:{source.deduplication_key}:{target.reviewer_type.value}:{target.reviewer_id}",
        )
        delegated = self.create(request)
        def mark(current: InboxItem) -> None:
            if current.status not in UNRESOLVED_STATUSES:
                raise InboxConcurrentUpdateError("source item resolved during delegation")
            current.status = InboxStatus.DELEGATED
            current.delegated_to_item_id = delegated.inbox_item_id
            current.token_nonce_hashes.clear()
        source, _ = self.repository.update(source.inbox_item_id, mark)
        def link(current: InboxItem) -> None:
            current.delegated_from_item_id = source.inbox_item_id
        delegated, _ = self.repository.update(delegated.inbox_item_id, link)
        self._audit("request_delegated", source, reviewer_id=actor_id, delegated_to_item_id=delegated.inbox_item_id)
        return delegated

    def reconcile(self) -> ReconciliationReport:
        report = ReconciliationReport()
        for item in self.repository.list(InboxQuery(statuses=set(UNRESOLVED_STATUSES))):
            assignment = ReviewerAssignment(
                reviewer_type=item.assigned_reviewer_type,
                reviewer_id=item.assigned_reviewer_id,
            )
            eligible = [
                identity
                for identity in self.identities.resolve(assignment)
                if item.tenant_id in identity.tenant_ids
                and (not identity.project_ids or item.project_id in identity.project_ids)
            ]
            eligible_ids = [identity.identity_id for identity in eligible]
            if eligible_ids != item.eligible_reviewer_ids or (
                eligible_ids and item.configuration_error
            ):
                def refresh_reviewer(current: InboxItem) -> None:
                    if current.status not in UNRESOLVED_STATUSES:
                        raise InboxConcurrentUpdateError("reviewer assignment resolved after terminal response")
                    current.eligible_reviewer_ids = eligible_ids
                    current.configuration_error = "" if eligible_ids else (
                        f"No eligible reviewer is configured for "
                        f"{current.assigned_reviewer_type.value} {current.assigned_reviewer_id}."
                    )

                try:
                    item, _ = self.repository.update(item.inbox_item_id, refresh_reviewer)
                except InboxConcurrentUpdateError:
                    continue
                self._audit(
                    "reviewer_resolved",
                    item,
                    eligible_reviewer_ids=eligible_ids,
                    configuration_error=item.configuration_error,
                    recovered=True,
                )
            if (
                eligible_ids
                and item.delivered_at is None
                and not self.repository.delivery_attempts(item.inbox_item_id)
                and not any(
                    event.event_type == "delivery_attempted"
                    for event in self.repository.audit_for_item(item.inbox_item_id)
                )
            ):
                self._deliver(item)
        if self.branch_controller is None:
            return report
        all_items = self.repository.list()
        item_ids = {item.inbox_item_id for item in all_items}
        for item in all_items:
            report.scanned += 1
            task = self.branch_controller.store.get_task_or_none(item.task_id)
            if task is None:
                # Unsupervised policy-gated actions have no execution-supervisor
                # checkpoint by design. Only checkpoint-bound requests can be
                # diagnosed as orphaned from the branch store.
                if item.checkpoint_id:
                    report.orphaned_items.append(item.inbox_item_id)
                continue
            try:
                if item.status in UNRESOLVED_STATUSES and task.state.value == "cancelled":
                    self.cancel_for_task(item.task_id, reason="originating task is cancelled")
                    report.cancelled.append(item.inbox_item_id)
                elif item.status in UNRESOLVED_STATUSES and task.state.value == "replanning":
                    self.supersede_for_action(item.action_intent_id) if item.action_intent_id else self._supersede_item(item)
                    report.superseded.append(item.inbox_item_id)
                elif item.status in UNRESOLVED_STATUSES and item.checkpoint_id and (
                    task.state.value != "waiting" or task.waiting_inbox_item_id != item.inbox_item_id
                ):
                    self.branch_controller.restore_human_wait(
                        item.task_id,
                        inbox_item_id=item.inbox_item_id,
                        checkpoint_id=item.checkpoint_id,
                        request_type=item.request_type.value,
                    )
                    report.waiting_restored.append(item.branch_id)
                elif item.status in {
                    InboxStatus.APPROVED,
                    InboxStatus.DENIED,
                    InboxStatus.ANSWERED,
                } and item.checkpoint_id and not item.resume_completed_at:
                    if self.terminal_response_handler is not None:
                        self.terminal_response_handler(item)
                    self._resume(item)
                    report.resumed.append(item.branch_id)
            except Exception as exc:
                report.errors.append(f"{item.inbox_item_id}: {type(exc).__name__}: {exc}")
        for task in self.branch_controller.store.list_tasks(incomplete_only=True):
            if task.waiting_inbox_item_id and task.waiting_inbox_item_id not in item_ids:
                report.waiting_without_item.append(task.task_id)
        return report

    def _supersede_item(self, item: InboxItem) -> InboxItem:
        def supersede(current: InboxItem) -> None:
            if current.status not in UNRESOLVED_STATUSES:
                raise InboxConcurrentUpdateError("inbox item resolved during supersession")
            current.status = InboxStatus.SUPERSEDED
            current.token_nonce_hashes.clear()
        changed, _ = self.repository.update(item.inbox_item_id, supersede)
        self._audit("request_superseded", changed, reason="originating branch replanned")
        if self.terminal_response_handler is not None:
            self.terminal_response_handler(changed)
        return changed

    def metrics(self) -> dict[str, Any]:
        rows = self.repository.list()
        pending = [item for item in rows if item.status in UNRESOLVED_STATUSES]
        responded = [item for item in rows if item.responded_at is not None and item.response is not None]
        latencies = [
            (item.responded_at - item.created_at).total_seconds()
            for item in responded
            if item.responded_at is not None
        ]
        resume_latencies = [
            (item.resume_completed_at - item.responded_at).total_seconds()
            for item in rows
            if item.resume_completed_at is not None and item.responded_at is not None
        ]
        statuses: dict[str, int] = {}
        risks: dict[str, int] = {}
        delivery_failures: dict[str, int] = {}
        deduplication_counts: dict[str, int] = {}
        for item in rows:
            statuses[item.status.value] = statuses.get(item.status.value, 0) + 1
            risks[item.risk_level.value] = risks.get(item.risk_level.value, 0) + 1
            deduplication_counts[item.deduplication_key] = deduplication_counts.get(item.deduplication_key, 0) + 1
            for attempt in self.repository.delivery_attempts(item.inbox_item_id):
                if attempt.status == "failed":
                    delivery_failures[attempt.adapter] = delivery_failures.get(attempt.adapter, 0) + 1
        oldest = min((item.created_at for item in pending), default=None)
        now = self.clock()
        terminal_count = sum(item.status in TERMINAL_STATUSES for item in rows)
        return {
            "pending_item_count": len(pending),
            "oldest_pending_age_seconds": (now - oldest).total_seconds() if oldest else 0,
            "average_response_latency_seconds": sum(latencies) / len(latencies) if latencies else 0,
            "average_resume_latency_seconds": (
                sum(resume_latencies) / len(resume_latencies)
                if resume_latencies
                else 0
            ),
            "status_counts": statuses,
            "risk_counts": risks,
            "approval_count": statuses.get(InboxStatus.APPROVED.value, 0),
            "denial_count": statuses.get(InboxStatus.DENIED.value, 0),
            "expiration_count": statuses.get(InboxStatus.EXPIRED.value, 0),
            "approval_rate": (
                statuses.get(InboxStatus.APPROVED.value, 0) / terminal_count
                if terminal_count
                else 0
            ),
            "denial_rate": (
                statuses.get(InboxStatus.DENIED.value, 0) / terminal_count
                if terminal_count
                else 0
            ),
            "expiration_rate": (
                statuses.get(InboxStatus.EXPIRED.value, 0) / terminal_count
                if terminal_count
                else 0
            ),
            "delivery_failures_by_adapter": delivery_failures,
            "repeated_equivalent_requests": sum(count - 1 for count in deduplication_counts.values() if count > 1),
            "duplicate_response_attempts": sum(
                event.event_type == "response_idempotent_replay"
                or (
                    event.event_type == "response_rejected"
                    and event.details.get("reason") in {"terminal_state", "token_replay"}
                )
                for item in rows
                for event in self.repository.audit_for_item(item.inbox_item_id)
            ),
            "unauthorized_response_attempts": sum(
                event.event_type == "response_rejected" and event.details.get("reason") == "unauthorized_reviewer"
                for item in rows
                for event in self.repository.audit_for_item(item.inbox_item_id)
            ),
        }

    def record_action_event(self, action_intent_id: str, event_type: str, details: dict[str, Any]) -> None:
        items = self.repository.find_for_action(action_intent_id)
        if items:
            self._audit(
                "transactional_" + event_type.replace(".", "_"),
                items[0],
                **details,
            )

    def record_execution_event(
        self,
        inbox_item_id: str,
        *,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append correlated evidence for a non-transactional legacy adapter."""
        allowed = {
            "policy_revalidated",
            "action_executed",
            "action_verification_completed",
        }
        if event_type not in allowed:
            raise ValueError("unsupported inbox execution audit event")
        self._audit(
            event_type,
            self.repository.get(inbox_item_id),
            **(details or {}),
        )

    def claim_action_execution(self, inbox_item_id: str) -> str:
        """Fence one legacy action attempt before any external side effect."""
        def claim(item: InboxItem) -> str:
            if item.status is not InboxStatus.APPROVED:
                raise PermissionError("only an approved inbox item may claim execution")
            if item.execution_completed_at is not None:
                raise InboxConcurrentUpdateError("approved action already completed execution")
            if item.execution_claim_id:
                raise InboxConcurrentUpdateError(
                    "approved action execution is already claimed; reconcile its outcome"
                )
            item.execution_claim_id = f"execute_{uuid4().hex}"
            item.execution_claimed_at = self.clock()
            return item.execution_claim_id

        item, claim_id = self.repository.update(inbox_item_id, claim)
        self._audit("action_execution_claimed", item, execution_claim_id=claim_id)
        return claim_id

    def complete_action_execution(
        self,
        inbox_item_id: str,
        *,
        execution_claim_id: str,
        result_digest: str,
    ) -> InboxItem:
        def complete(item: InboxItem) -> None:
            if item.execution_claim_id != execution_claim_id:
                raise InboxConcurrentUpdateError("action execution claim changed")
            item.execution_completed_at = item.execution_completed_at or self.clock()
            if item.execution_result_digest and item.execution_result_digest != result_digest:
                raise InboxConcurrentUpdateError("action execution result changed")
            item.execution_result_digest = result_digest

        item, _ = self.repository.update(inbox_item_id, complete)
        self._audit(
            "action_execution_completed",
            item,
            execution_claim_id=execution_claim_id,
            result_digest=result_digest,
        )
        return item

    def record_rejected_response(self, inbox_item_id: str, *, actor_id: str, reason: str) -> None:
        item = self.repository.get(inbox_item_id)
        self._audit("response_rejected", item, reviewer_id=actor_id, reason=reason)

    def assert_response_actor_is_currently_authorized(self, item: InboxItem) -> None:
        if not item.response_actor_id:
            raise PermissionError("inbox item has no attributed reviewer response")
        self._authorize(item, item.response_actor_id)

    def _resume(self, item: InboxItem) -> None:
        if self.branch_controller is None or not item.checkpoint_id:
            return

        def claim(current: InboxItem) -> str:
            if current.resume_completed_at is not None:
                return current.resume_claim_id
            if not current.resume_claim_id:
                current.resume_claim_id = f"resume_{uuid4().hex}"
                current.resume_claimed_at = self.clock()
            return current.resume_claim_id

        claimed, claim_id = self.repository.update(item.inbox_item_id, claim)
        self._audit("branch_resume_claimed", claimed, resume_claim_id=claim_id)
        assert claimed.response is not None
        self.branch_controller.resume_from_human_input(
            claimed.task_id,
            inbox_item_id=claimed.inbox_item_id,
            checkpoint_id=claimed.checkpoint_id,
            resume_claim_id=claim_id,
            structured_response=claimed.response.model_dump(mode="json"),
        )

        def complete(current: InboxItem) -> None:
            if current.resume_claim_id != claim_id:
                raise InboxConcurrentUpdateError("resume claim changed concurrently")
            current.resume_completed_at = current.resume_completed_at or self.clock()

        completed, _ = self.repository.update(claimed.inbox_item_id, complete)
        self._audit("branch_resumed", completed, resume_claim_id=claim_id)

    def _resume_terminal_if_needed(self, item: InboxItem) -> None:
        if (
            item.status
            in {InboxStatus.APPROVED, InboxStatus.DENIED, InboxStatus.ANSWERED}
            and item.checkpoint_id
            and item.resume_completed_at is None
        ):
            self._resume(item)

    def _validate_answer(self, item: InboxItem, submission: ResponseSubmission) -> dict[str, Any]:
        if item.request_type is InboxRequestType.APPROVAL:
            if submission.answer:
                raise ValueError("binary approval responses cannot supply clarification answers")
            return {}
        fields = {field.field_id: field for field in item.requested_fields}
        unknown = set(submission.answer) - set(fields)
        if unknown:
            raise ValueError(f"clarification response contains unknown fields: {sorted(unknown)}")
        missing = [field.field_id for field in fields.values() if field.required and field.field_id not in submission.answer]
        if missing:
            raise ValueError(f"clarification response is missing required fields: {missing}")
        for field_id, value in submission.answer.items():
            field = fields[field_id]
            valid_type = {
                ExpectedResponseType.TEXT: isinstance(value, str),
                ExpectedResponseType.BOOLEAN: isinstance(value, bool),
                ExpectedResponseType.INTEGER: isinstance(value, int) and not isinstance(value, bool),
                ExpectedResponseType.NUMBER: isinstance(value, (int, float)) and not isinstance(value, bool),
                ExpectedResponseType.CHOICE: isinstance(value, str),
                ExpectedResponseType.MULTI_CHOICE: isinstance(value, list),
                ExpectedResponseType.OBJECT: isinstance(value, dict),
            }[field.expected_type]
            if not valid_type:
                raise ValueError(f"clarification field {field_id} has the wrong response type")
            if field.choices:
                values = value if isinstance(value, list) else [value]
                if any(str(candidate) not in field.choices for candidate in values):
                    raise ValueError(f"clarification field {field_id} contains a value outside its choices")
            if not field.allow_free_form and not field.choices:
                raise ValueError(f"clarification field {field_id} has no permitted response form")
            constraints = field.validation
            if constraints.minimum is not None and isinstance(value, (int, float)) and value < constraints.minimum:
                raise ValueError(f"clarification field {field_id} is below its minimum")
            if constraints.maximum is not None and isinstance(value, (int, float)) and value > constraints.maximum:
                raise ValueError(f"clarification field {field_id} exceeds its maximum")
            if constraints.min_length is not None and hasattr(value, "__len__") and len(value) < constraints.min_length:
                raise ValueError(f"clarification field {field_id} is shorter than its minimum length")
            if constraints.max_length is not None and hasattr(value, "__len__") and len(value) > constraints.max_length:
                raise ValueError(f"clarification field {field_id} exceeds its maximum length")
            if constraints.pattern and isinstance(value, str) and re.fullmatch(constraints.pattern, value) is None:
                raise ValueError(f"clarification field {field_id} does not match its required pattern")
        return dict(submission.answer)

    def _authorize(self, item: InboxItem, actor_id: str) -> None:
        assignment = ReviewerAssignment(
            reviewer_type=item.assigned_reviewer_type,
            reviewer_id=item.assigned_reviewer_id,
        )
        eligible_now = {identity.identity_id for identity in self.identities.resolve(assignment)}
        identity = self.identities.get(actor_id)
        scope_allowed = (
            item.tenant_id == "local"
            if identity is None
            else item.tenant_id in identity.tenant_ids
            and (not identity.project_ids or item.project_id in identity.project_ids)
        )
        if actor_id not in eligible_now or not scope_allowed:
            self._audit("response_rejected", item, reviewer_id=actor_id, reason="unauthorized_reviewer")
            raise PermissionError("reviewer is not currently authorized for this inbox item")

    def _deliver(
        self,
        item: InboxItem,
        *,
        reminder: bool = False,
        reminder_claimed: bool = False,
    ) -> list[DeliveryAttempt]:
        attempts: list[DeliveryAttempt] = []
        for adapter in self.notification_adapters:
            prior = [attempt for attempt in self.repository.delivery_attempts(item.inbox_item_id) if attempt.adapter == adapter.name]
            attempt_number = len(prior) + 1
            notification = InboxNotification(
                inbox_item_id=item.inbox_item_id,
                destination=f"{item.assigned_reviewer_type.value}:{item.assigned_reviewer_id}",
                title=item.title,
                summary=item.summary,
                risk_level=item.risk_level.value,
                request_type=item.request_type.value,
                expires_at=item.expires_at.isoformat(),
                response_reference=f"/api/v1/inbox/{item.inbox_item_id}",
                metadata={"reminder": reminder, "branch_id": item.branch_id},
            )
            self._audit(
                "delivery_attempted",
                item,
                adapter=adapter.name,
                attempt_number=attempt_number,
            )
            try:
                result = adapter.deliver(notification)
                status = "delivered" if result.delivered else "failed"
                error = result.error
                external_id = result.external_message_id
            except Exception as exc:
                status, error, external_id = "failed", f"{type(exc).__name__}: {exc}", ""
            attempt = DeliveryAttempt(
                inbox_item_id=item.inbox_item_id,
                adapter=adapter.name,
                destination=notification.destination,
                attempt_number=attempt_number,
                status=status,
                error=error,
                timestamp=self.clock(),
                external_message_id=external_id,
            )
            self.repository.save_delivery_attempt(attempt)
            self._audit(f"delivery_{status}", item, adapter=adapter.name, attempt_number=attempt_number, error=error)
            attempts.append(attempt)
            if status == "delivered":
                def delivered(current: InboxItem) -> None:
                    if current.status in UNRESOLVED_STATUSES:
                        current.status = InboxStatus.DELIVERED
                        current.delivered_at = current.delivered_at or self.clock()
                item, _ = self.repository.update(item.inbox_item_id, delivered)
        if reminder and attempts and not reminder_claimed:
            def record_reminder(current: InboxItem) -> None:
                if current.status in UNRESOLVED_STATUSES:
                    current.reminder_count += 1
                    current.last_reminded_at = self.clock()
            self.repository.update(item.inbox_item_id, record_reminder)
        return attempts

    def _audit(self, event_type: str, item: InboxItem, reviewer_id: str = "", **details: Any) -> None:
        self.repository.append_audit(InboxAuditEvent(
            event_type=event_type,
            inbox_item_id=item.inbox_item_id,
            task_id=item.task_id,
            branch_id=item.branch_id,
            policy_decision_id=item.policy_decision_id,
            action_intent_id=item.action_intent_id,
            reviewer_id=reviewer_id,
            created_at=self.clock(),
            details=redact_secrets(details),
        ))
        from mana_agent.utils.durable_diagnostics import append_diagnostic
        append_diagnostic(
            self.repository.root / "logs" / "inbox.jsonl",
            component="human_inbox",
            event=event_type,
            details={"inbox_item_id": item.inbox_item_id, "action_id": item.action_intent_id, "status": item.status.value},
        )

    def _emit(self, event_type: str, item: InboxItem) -> None:
        if self.event_sink is not None:
            self.event_sink({
                "type": event_type,
                "event_type": event_type,
                "kind": "human_inbox",
                "status": item.status.value,
                "title": item.title,
                "message": event_type.replace(".", " "),
                "metadata": {
                    "human_inbox": True,
                    "authoritative_state_required": True,
                    "inbox_item_id": item.inbox_item_id,
                    "task_id": item.task_id,
                    "branch_id": item.branch_id,
                    "request_type": item.request_type.value,
                    "risk_level": item.risk_level.value,
                },
            })
