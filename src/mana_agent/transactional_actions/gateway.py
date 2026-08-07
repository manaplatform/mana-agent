from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mana_agent.human_inbox.models import (
    InboxRequest,
    InboxRequestType,
    InboxStatus,
    ResponseOperation,
    ResponseSubmission,
    ReviewerAssignment,
    ReviewerType,
    RiskLevel,
    canonical_digest,
)
from mana_agent.human_inbox.service import HumanInboxService

from .adapters import ActionAdapter, ActionInvalidatedError, CompensationActionAdapter
from .approvals import ApprovalRegistry
from .compensation import CompensationRegistry, file_compensation_registry
from .events import ActionEventSink, event_payload
from .models import (
    ActionIntent,
    ActionPreview,
    ActionState,
    ApprovalScope,
    CompensationEvidence,
    PolicyOutcome,
    utc_now,
)
from .policy import ActionPolicy
from .store import ActionStore


class ApprovalRequired(PermissionError):
    def __init__(self, action: ActionIntent, *, inbox_item_id: str = "") -> None:
        super().__init__("The action requires approval bound to its exact preview and policy decision.")
        self.action = action
        self.inbox_item_id = inbox_item_id


@dataclass(frozen=True)
class ActionOutcome:
    action: ActionIntent
    result: dict[str, Any]
    duplicate: bool = False


class ActionGateway:
    """Mandatory propose-to-verify gateway for side-effecting adapters."""

    def __init__(
        self,
        *,
        store: ActionStore,
        policy: ActionPolicy,
        approvals: ApprovalRegistry,
        event_sink: ActionEventSink | None = None,
        compensation_registry: CompensationRegistry | None = None,
        inbox_service: HumanInboxService | None = None,
    ) -> None:
        self.store, self.policy, self.approvals, self.event_sink = store, policy, approvals, event_sink
        self.compensation_registry = compensation_registry or file_compensation_registry()
        self.inbox_service = inbox_service

    def propose(self, adapter: ActionAdapter) -> ActionIntent:
        action = adapter.build_intent()
        prior = self.store.action_for_idempotency_key(action.idempotency_key)
        if prior is not None:
            if prior.intent_digest() != action.intent_digest():
                if prior.execution_attempts == 0 and prior.state in {
                    ActionState.AWAITING_APPROVAL, ActionState.APPROVED, ActionState.FAILED
                }:
                    self.approvals.invalidate_for_action(prior.action_id)
                    if self.inbox_service is not None:
                        self.inbox_service.supersede_for_action(prior.action_id)
                    if prior.state in {ActionState.AWAITING_APPROVAL, ActionState.APPROVED}:
                        prior.transition(ActionState.CANCELLED)
                        prior.error = "action material changed after preview; approval invalidated"
                        self.store.save_action(prior)
                    self.store.release_idempotency(prior)
                    self._emit("action.approval.invalidated", prior, reason="action_material_changed")
                else:
                    raise ValueError("idempotency key conflicts with a materially different action")
            else:
                return prior
        self.store.create_action(action)
        protected_context = getattr(adapter, "protected_action_context", lambda: {})()
        if protected_context:
            reference, digest = self.store.save_protected_action_context(
                action.action_id,
                protected_context,
            )
            action.protected_context_ref = reference
            action.protected_context_digest = digest
            self.store.save_action(action)
        self._emit("action.proposed", action)
        action.transition(ActionState.PREVIEWING)
        preview = ActionPreview.model_validate(adapter.preview(action).redacted())
        action.preview_digest = preview.digest()
        action.preview = preview
        self.store.save_action(action)
        self._emit("action.preview.ready", action)
        action.transition(ActionState.AWAITING_POLICY)
        action.policy_decision = self.policy.evaluate(action)
        if action.policy_decision.outcome is PolicyOutcome.DENY:
            action.transition(ActionState.FAILED)
            action.error = action.policy_decision.explanation
            self.store.save_action(action)
            self._emit("action.policy.denied", action)
            return action
        if action.policy_decision.outcome is PolicyOutcome.REQUIRE_APPROVAL:
            action.transition(ActionState.AWAITING_APPROVAL)
            self.store.save_action(action)
            inbox_item_id = self._ensure_inbox(action)
            # The stores are independently atomic. Persist the durable link before
            # publishing the approval event so UIs never receive a dangling prompt.
            if inbox_item_id and action.inbox_item_id != inbox_item_id:
                action.inbox_item_id = inbox_item_id
                self.store.save_action(action)
            self._emit("action.approval.required", action, inbox_item_id=inbox_item_id)
            return action
        action.transition(ActionState.APPROVED)
        self.store.save_action(action)
        self._emit("action.policy.allowed", action)
        return action

    def execute(self, adapter: ActionAdapter, *, approval_id: str = "") -> ActionOutcome:
        proposed = adapter.build_intent()
        action = self.store.action_for_idempotency_key(proposed.idempotency_key)
        if action is None:
            action = self.propose(adapter)
        elif action.intent_digest() != proposed.intent_digest():
            action = self.propose(adapter)
        current_policy_fingerprint = self.policy.config.fingerprint()
        if (
            action.policy_decision is not None
            and action.policy_decision.policy_fingerprint != current_policy_fingerprint
            and action.state in {ActionState.AWAITING_APPROVAL, ActionState.APPROVED, ActionState.FAILED}
            and action.execution_attempts == 0
        ):
            self.approvals.invalidate_for_action(action.action_id)
            if self.inbox_service is not None:
                self.inbox_service.supersede_for_action(action.action_id)
            if action.state in {ActionState.AWAITING_APPROVAL, ActionState.APPROVED}:
                action.transition(ActionState.CANCELLED)
                action.error = "policy changed after preview; approval invalidated"
                self.store.save_action(action)
            self.store.release_idempotency(action)
            self._emit("action.approval.invalidated", action, reason="policy_changed")
            action = self.propose(adapter)
        if action.state is ActionState.COMMITTED:
            if action.verification is None or not action.verification.complete:
                raise RuntimeError("committed action lacks complete verification evidence")
            current = adapter.verify(action, action.execution_result)
            if not current.complete:
                raise RuntimeError(
                    "the previously committed action no longer matches observable state; "
                    "a new model decision and idempotency key are required"
                )
            return ActionOutcome(action=action, result=action.execution_result, duplicate=True)
        if action.state in {ActionState.EXECUTING, ActionState.VERIFYING}:
            raise RuntimeError("duplicate execution is already in progress")
        if action.state is ActionState.FAILED:
            return ActionOutcome(
                action=action,
                result=action.execution_result,
                duplicate=action.execution_attempts > 0,
            )
        if action.state in {ActionState.AWAITING_APPROVAL, ActionState.APPROVED} and (
            action.expires_at <= utc_now()
            or (action.policy_decision and action.policy_decision.expires_at <= utc_now())
        ):
            action.transition(ActionState.EXPIRED)
            self.store.save_action(action)
            self.store.release_idempotency(action)
            self._emit("action.approval.expired", action)
            raise PermissionError("action or policy decision expired before execution")
        if action.state is ActionState.AWAITING_APPROVAL:
            transaction_binding = self._transaction_binding(action)
            inbox_item = None
            if self.inbox_service is not None:
                items = self.inbox_service.repository.find_for_action(action.action_id)
                inbox_item = items[0] if items else None
                if inbox_item is None:
                    inbox_item_id = self._ensure_inbox(action)
                    inbox_item = self.inbox_service.repository.get(inbox_item_id)
                if inbox_item is not None and inbox_item.status is InboxStatus.APPROVED:
                    if inbox_item.action_digest != action.approval_digest():
                        self.inbox_service.supersede_for_action(action.action_id)
                        raise PermissionError("action material changed after the human decision")
                    self.inbox_service.assert_response_actor_is_currently_authorized(inbox_item)
                elif inbox_item is not None and inbox_item.status in {
                    InboxStatus.DENIED,
                    InboxStatus.CANCELLED,
                    InboxStatus.SUPERSEDED,
                    InboxStatus.EXPIRED,
                }:
                    action.transition(
                        ActionState.EXPIRED if inbox_item.status is InboxStatus.EXPIRED else ActionState.CANCELLED
                    )
                    action.error = f"human decision: {inbox_item.status.value}"
                    self.store.save_action(action)
                    self.store.release_idempotency(action)
                    raise PermissionError(action.error)
                elif approval_id:
                    raise PermissionError(
                        "an approval grant cannot bypass the unresolved durable inbox item"
                    )
            if not approval_id:
                pending_grant = self.approvals.find_valid(
                    action, transaction_binding_digest=transaction_binding
                )
                if pending_grant is None and self.inbox_service is not None:
                    if inbox_item is not None and inbox_item.action_digest != action.approval_digest():
                        self.inbox_service.supersede_for_action(action.action_id)
                        raise PermissionError("action material changed after the human decision")
                    if inbox_item is not None and inbox_item.status is InboxStatus.APPROVED:
                        scope = (
                            action.policy_decision.required_approval_scope
                            if action.policy_decision is not None
                            else ApprovalScope.ACTION_ONCE
                        )
                        pending_grant = self.approvals.issue(
                            action,
                            approved_by=inbox_item.response_actor_id,
                            ttl_seconds=max(
                                1,
                                int((inbox_item.expires_at - utc_now()).total_seconds()),
                            ),
                            scope=scope or ApprovalScope.ACTION_ONCE,
                            transaction_binding_digest=transaction_binding,
                        )
                    elif inbox_item is not None and inbox_item.status in {
                        InboxStatus.DENIED,
                        InboxStatus.CANCELLED,
                        InboxStatus.SUPERSEDED,
                        InboxStatus.EXPIRED,
                    }:
                        action.transition(
                            ActionState.EXPIRED if inbox_item.status is InboxStatus.EXPIRED else ActionState.CANCELLED
                        )
                        action.error = f"human decision: {inbox_item.status.value}"
                        self.store.save_action(action)
                        self.store.release_idempotency(action)
                        raise PermissionError(action.error)
                if pending_grant is None:
                    raise ApprovalRequired(action, inbox_item_id=self._ensure_inbox(action))
                approval_id = pending_grant.approval_id
            self.approvals.consume(
                approval_id,
                action,
                transaction_binding_digest=transaction_binding,
            )
            action.transition(ActionState.APPROVED)
            self.store.save_action(action)
            self._emit("action.policy.revalidated", action, approval_id=approval_id)
        if action.state is not ActionState.APPROVED:
            raise PermissionError(f"action is not executable in state {action.state.value}")
        if self.inbox_service is not None:
            approved_items = [
                item
                for item in self.inbox_service.repository.find_for_action(action.action_id)
                if item.status is InboxStatus.APPROVED
            ]
            if approved_items:
                approved_item = approved_items[0]
                if approved_item.action_digest != action.approval_digest():
                    raise PermissionError("approved action digest no longer matches execution intent")
                self.inbox_service.assert_response_actor_is_currently_authorized(approved_item)
                self._assert_branch_runnable(approved_item)
        action = self.store.claim_execution(action.action_id)
        self._emit("action.execution.started", action)
        try:
            result = adapter.execute(action)
            action.execution_result = adapter.persistable_result(result)
            action.transition(ActionState.VERIFYING)
            self.store.save_action(action)
            self._emit("action.verification.started", action)
            evidence = adapter.verify(action, result)
            action.verification = evidence
            self._emit("action.verification.completed", action)
            if not evidence.complete:
                action.transition(ActionState.FAILED)
                action.error = "adapter verification was incomplete or failed"
                self.store.save_action(action)
                self._emit("action.manual_recovery.required", action)
                return ActionOutcome(action=action, result=result)
            action.transition(ActionState.COMMITTED)
            self.store.save_action(action)
            self._emit("action.committed", action)
            return ActionOutcome(action=action, result=result)
        except Exception as exc:
            side_effect_status_unknown = (
                action.state is ActionState.EXECUTING
                and not isinstance(exc, ActionInvalidatedError)
            )
            if action.state in {ActionState.EXECUTING, ActionState.VERIFYING}:
                action.transition(ActionState.FAILED)
            action.error = f"{type(exc).__name__}: {exc}"
            self.store.save_action(action)
            if isinstance(exc, ActionInvalidatedError):
                self.approvals.invalidate_for_action(action.action_id)
                if self.inbox_service is not None:
                    self.inbox_service.supersede_for_action(action.action_id)
                self.store.release_idempotency(action)
                self._emit("action.approval.invalidated", action)
            if side_effect_status_unknown:
                self._emit("action.manual_recovery.required", action)
            raise

    def compensate(self, action_id: str, adapter: ActionAdapter, *, approval_id: str = "") -> ActionIntent:
        original = self.store.get_action(action_id)
        if original is None:
            raise LookupError("unknown action")
        if original.state not in {ActionState.COMMITTED, ActionState.FAILED}:
            raise ValueError("only committed or failed actions may be compensated")
        if original.state is ActionState.FAILED and not original.execution_result:
            raise ValueError(
                "a failed action without durable execution evidence cannot be compensated automatically"
            )
        self.compensation_registry.assert_eligible(original)
        compensation_adapter = CompensationActionAdapter(original, adapter)
        compensation = self.propose(compensation_adapter)
        self._emit("action.compensation.started", compensation, compensates_action_id=action_id)
        outcome = self.execute(compensation_adapter, approval_id=approval_id)
        self._emit("action.compensation.completed", outcome.action, compensates_action_id=action_id)
        if outcome.action.state is ActionState.COMMITTED:
            original.transition(ActionState.COMPENSATING)
            verification = outcome.action.verification
            original.compensation = CompensationEvidence(
                complete=bool(verification and verification.complete),
                summary=verification.summary if verification else "Compensation verification missing.",
                checks=list(verification.checks if verification else []),
                observed_at=verification.observed_at if verification else utc_now(),
            )
            original.transition(ActionState.COMPENSATED)
            self.store.save_action(original)
        return outcome.action

    def approve(
        self,
        action_id: str,
        *,
        approved_by: str,
        reviewer_id: str | None = None,
        ttl_seconds: int = 300,
    ) -> str:
        action = self.store.get_action(action_id)
        if action is None:
            raise LookupError("unknown action")
        if action.state is not ActionState.AWAITING_APPROVAL:
            raise ValueError(f"action is not awaiting approval: {action.state.value}")
        if action.expires_at <= utc_now() or (action.policy_decision and action.policy_decision.expires_at <= utc_now()):
            action.transition(ActionState.EXPIRED)
            self.store.save_action(action)
            self.store.release_idempotency(action)
            self._emit("action.approval.expired", action)
            raise PermissionError("action or policy decision expired before approval")
        if self.inbox_service is not None:
            item_id = self._ensure_inbox(action)
            self.inbox_service.respond(ResponseSubmission(
                inbox_item_id=item_id,
                operation=ResponseOperation.APPROVE,
                actor_id=reviewer_id or approved_by,
                channel=f"transactional_action:{approved_by}",
                idempotency_key=f"legacy-approve:{action.action_id}:{approved_by}",
                comment=f"Submitted through trusted legacy principal {approved_by}.",
                current_action_digest=action.approval_digest(),
            ))
        binding = self._transaction_binding(action)
        scope = (
            action.policy_decision.required_approval_scope
            if action.policy_decision is not None
            else ApprovalScope.ACTION_ONCE
        )
        grant = self.approvals.find_valid(
            action,
            transaction_binding_digest=binding,
        ) or self.approvals.issue(
            action,
            approved_by=approved_by,
            ttl_seconds=ttl_seconds,
            scope=scope or ApprovalScope.ACTION_ONCE,
            transaction_binding_digest=binding,
        )
        self._emit(
            "action.approval.granted",
            action,
            approval_id=grant.approval_id,
            approved_by=grant.approved_by,
            approval_scope=grant.scope.value,
            approved_at=grant.approved_at.isoformat(),
            approval_expires_at=grant.expires_at.isoformat(),
        )
        return grant.approval_id

    def approve_transaction(
        self,
        transaction_id: str,
        *,
        approved_by: str,
        reviewer_id: str | None = None,
        ttl_seconds: int = 300,
    ) -> dict[str, str]:
        transaction = self.store.get_transaction(transaction_id)
        if transaction is None:
            raise LookupError("unknown transaction")
        binding = transaction.binding_digest()
        grants: dict[str, str] = {}
        for action_id in transaction.action_ids:
            action = self.store.get_action(action_id)
            if action is None or action.transaction_id != transaction_id:
                raise ValueError("transaction membership does not match persisted actions")
            if action.state is not ActionState.AWAITING_APPROVAL:
                continue
            if self.inbox_service is not None:
                item_id = self._ensure_inbox(action)
                self.inbox_service.respond(ResponseSubmission(
                    inbox_item_id=item_id,
                    operation=ResponseOperation.APPROVE,
                    actor_id=reviewer_id or approved_by,
                    channel=f"transactional_action:{approved_by}",
                    idempotency_key=f"legacy-transaction-approve:{transaction_id}:{action.action_id}:{approved_by}",
                    comment=f"Submitted through trusted legacy principal {approved_by}.",
                    current_action_digest=action.approval_digest(),
                ))
            grant = self.approvals.find_valid(
                action,
                transaction_binding_digest=binding,
            ) or self.approvals.issue(
                action,
                approved_by=approved_by,
                ttl_seconds=ttl_seconds,
                scope=ApprovalScope.TRANSACTION,
                transaction_binding_digest=binding,
            )
            grants[action_id] = grant.approval_id
            self._emit(
                "action.approval.granted",
                action,
                approval_id=grant.approval_id,
                approved_by=grant.approved_by,
                approval_scope=grant.scope.value,
                approved_at=grant.approved_at.isoformat(),
                approval_expires_at=grant.expires_at.isoformat(),
                transaction_binding_digest=binding,
            )
        if not grants:
            raise ValueError("transaction has no actions awaiting approval")
        return grants

    def _transaction_binding(self, action: ActionIntent) -> str:
        if not action.transaction_id:
            return ""
        transaction = self.store.get_transaction(action.transaction_id)
        return transaction.binding_digest() if transaction is not None else ""

    def resume_verification(self, action_id: str, adapter: ActionAdapter) -> ActionIntent:
        """Recover a persisted post-execution action without repeating its side effect."""
        action = self.store.get_action(action_id)
        if action is None:
            raise LookupError("unknown action")
        if action.state is ActionState.EXECUTING:
            action.transition(ActionState.FAILED)
            action.error = (
                "execution was interrupted before a durable result was recorded; "
                "side-effect status is unknown and manual recovery is required"
            )
            self.store.save_action(action)
            self._emit("action.manual_recovery.required", action)
            return action
        if action.state is not ActionState.VERIFYING:
            raise ValueError(f"action is not recoverable from state {action.state.value}")
        evidence = adapter.verify(action, action.execution_result)
        action.verification = evidence
        action.transition(ActionState.COMMITTED if evidence.complete else ActionState.FAILED)
        if not evidence.complete:
            action.error = "recovered verification was incomplete; manual recovery is required"
        self.store.save_action(action)
        self._emit("action.verification.completed", action, recovered=True)
        self._emit("action.committed" if evidence.complete else "action.manual_recovery.required", action)
        return action

    def deny(
        self,
        action_id: str,
        *,
        denied_by: str,
        reviewer_id: str | None = None,
    ) -> ActionIntent:
        action = self.store.get_action(action_id)
        if action is None:
            raise LookupError("unknown action")
        if action.state is not ActionState.AWAITING_APPROVAL:
            raise ValueError(f"action is not awaiting approval: {action.state.value}")
        if self.inbox_service is not None:
            item_id = self._ensure_inbox(action)
            self.inbox_service.respond(ResponseSubmission(
                inbox_item_id=item_id,
                operation=ResponseOperation.DENY,
                actor_id=reviewer_id or denied_by,
                channel=f"transactional_action:{denied_by}",
                idempotency_key=f"legacy-deny:{action.action_id}:{denied_by}",
                comment=f"Submitted through trusted legacy principal {denied_by}.",
                current_action_digest=action.approval_digest(),
            ))
            refreshed = self.store.get_action(action_id)
            if refreshed is not None and refreshed.state is ActionState.CANCELLED:
                self._emit("action.approval.denied", refreshed, denied_by=denied_by)
                return refreshed
        self.approvals.invalidate_for_action(action_id)
        action.transition(ActionState.CANCELLED)
        action.error = f"denied by {denied_by}"
        self.store.save_action(action)
        self.store.release_idempotency(action)
        self._emit("action.approval.denied", action, denied_by=denied_by)
        return action

    def _ensure_inbox(self, action: ActionIntent) -> str:
        if self.inbox_service is None:
            return ""
        existing = self.inbox_service.repository.find_for_action(action.action_id)
        if existing:
            if action.inbox_item_id != existing[0].inbox_item_id:
                action.inbox_item_id = existing[0].inbox_item_id
                self.store.save_action(action)
            return existing[0].inbox_item_id
        decision = action.policy_decision
        if decision is None or decision.outcome is not PolicyOutcome.REQUIRE_APPROVAL:
            raise ValueError("durable approval inbox requires a validated approval policy decision")
        reviewer = ReviewerAssignment(
            reviewer_type=ReviewerType(decision.assigned_reviewer_type),
            reviewer_id=decision.assigned_reviewer_id,
        )
        risk = (
            RiskLevel.CRITICAL
            if action.reversibility.value == "irreversible" or action.data_disclosure.value == "secret"
            else RiskLevel.HIGH
            if action.blast_radius.value in {"organisation", "physical", "unknown"}
            or action.data_disclosure.value in {"confidential", "external_public", "external_private"}
            else RiskLevel.MEDIUM
        )
        preview = action.preview.redacted() if action.preview else {}
        effect_labels = action.approval_effect_labels()
        checkpoint_id = ""
        execution_attempt_id = ""
        parent_task_id = action.parent_task_id
        project_id = "local"
        branch_controller = self.inbox_service.branch_controller
        if branch_controller is not None:
            task = branch_controller.store.get_task_or_none(action.parent_task_id)
            if task is not None:
                if not task.checkpoint_id:
                    raise ValueError(
                        "a supervised action approval requires the branch's durable checkpoint"
                    )
                checkpoint_id = task.checkpoint_id
                execution_attempt_id = task.attempt_id
                parent_task_id = task.parent_task_id
                project_id = task.repository_id or task.workspace_id or "local"
        item = self.inbox_service.create(InboxRequest(
            request_type=InboxRequestType.APPROVAL,
            project_id=project_id,
            task_id=action.parent_task_id,
            branch_id=action.parent_task_id,
            parent_task_id=parent_task_id,
            checkpoint_id=checkpoint_id,
            execution_attempt_id=execution_attempt_id,
            policy_decision_id=decision.decision_id,
            permission_request_id=action.action_id,
            action_intent_id=action.action_id,
            action_digest=action.approval_digest(),
            requested_by_agent_id=action.originating_agent,
            reviewer=reviewer,
            title=f"Approve {action.tool_name} {action.operation_name}",
            summary=action.preview.summary if action.preview else decision.explanation,
            risk_level=risk,
            allowed_responses=[ResponseOperation.APPROVE, ResponseOperation.DENY],
            minimal_context={
                "action_type": action.tool_name,
                "operation": action.operation_name,
                "action_count": 1,
                "resource_count": len(action.target_resources),
                "side_effect_count": len(action.expected_side_effects),
                "effect_labels": effect_labels,
            },
            protected_context={
                "action_id": action.action_id,
                "binding_digest": action.binding_digest(),
                "approval_digest": action.approval_digest(),
                "preview": preview,
                "target_resources": action.target_resources,
                "normalized_arguments": action.normalized_arguments,
                "action_context_ref": action.protected_context_ref,
                "action_context_digest": action.protected_context_digest,
                "effect_labels": effect_labels,
            },
            disclosed_fields=["action_type", "operation", "action_count", "resource_count", "side_effect_count", "effect_labels"],
            reversibility=action.reversibility.value,
            expires_at=min(action.expires_at, decision.expires_at),
            idempotency_key=f"action-approval:{action.action_id}:{action.approval_digest()}",
            deduplication_key=canonical_digest({
                "action_digest": action.approval_digest(),
                "reviewer": reviewer.model_dump(mode="json"),
            }),
        ))
        if action.inbox_item_id != item.inbox_item_id:
            action.inbox_item_id = item.inbox_item_id
            self.store.save_action(action)
        return item.inbox_item_id

    def _assert_branch_runnable(self, item: Any) -> None:
        if self.inbox_service is None or not item.checkpoint_id:
            return
        controller = self.inbox_service.branch_controller
        if controller is None:
            raise PermissionError("approved action branch controller is unavailable")
        task = controller.store.get_task_or_none(item.task_id)
        if task is None:
            raise PermissionError("approved action branch no longer exists")
        if task.checkpoint_id != item.checkpoint_id:
            raise PermissionError("approved action checkpoint no longer matches the branch")
        if task.state.value != "running" or task.waiting_inbox_item_id:
            raise PermissionError(
                "approved action may execute only from its resumed running branch"
            )

    def _emit(self, event_type: str, action: ActionIntent, **details: Any) -> None:
        self.store.append_audit(action, event_type, details)
        if self.inbox_service is not None:
            self.inbox_service.record_action_event(action.action_id, event_type, details)
        if self.event_sink:
            self.event_sink(event_payload(event_type, action, **details))
