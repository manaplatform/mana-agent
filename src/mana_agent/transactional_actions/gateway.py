from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    def __init__(self, action: ActionIntent) -> None:
        super().__init__("The action requires approval bound to its exact preview and policy decision.")
        self.action = action


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
    ) -> None:
        self.store, self.policy, self.approvals, self.event_sink = store, policy, approvals, event_sink
        self.compensation_registry = compensation_registry or file_compensation_registry()

    def propose(self, adapter: ActionAdapter) -> ActionIntent:
        action = adapter.build_intent()
        prior = self.store.action_for_idempotency_key(action.idempotency_key)
        if prior is not None:
            if prior.intent_digest() != action.intent_digest():
                if prior.execution_attempts == 0 and prior.state in {
                    ActionState.AWAITING_APPROVAL, ActionState.APPROVED, ActionState.FAILED
                }:
                    self.approvals.invalidate_for_action(prior.action_id)
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
            self._emit("action.approval.required", action)
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
            if not approval_id:
                pending_grant = self.approvals.find_valid(
                    action, transaction_binding_digest=transaction_binding
                )
                if pending_grant is None:
                    raise ApprovalRequired(action)
                approval_id = pending_grant.approval_id
            self.approvals.consume(
                approval_id,
                action,
                transaction_binding_digest=transaction_binding,
            )
            action.transition(ActionState.APPROVED)
            self.store.save_action(action)
        if action.state is not ActionState.APPROVED:
            raise PermissionError(f"action is not executable in state {action.state.value}")
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

    def approve(self, action_id: str, *, approved_by: str, ttl_seconds: int = 300) -> str:
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
        grant = self.approvals.issue(action, approved_by=approved_by, ttl_seconds=ttl_seconds)
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
            grant = self.approvals.issue(
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

    def deny(self, action_id: str, *, denied_by: str) -> ActionIntent:
        action = self.store.get_action(action_id)
        if action is None:
            raise LookupError("unknown action")
        if action.state is not ActionState.AWAITING_APPROVAL:
            raise ValueError(f"action is not awaiting approval: {action.state.value}")
        self.approvals.invalidate_for_action(action_id)
        action.transition(ActionState.CANCELLED)
        action.error = f"denied by {denied_by}"
        self.store.save_action(action)
        self.store.release_idempotency(action)
        self._emit("action.approval.denied", action, denied_by=denied_by)
        return action

    def _emit(self, event_type: str, action: ActionIntent, **details: Any) -> None:
        self.store.append_audit(action, event_type, details)
        if self.event_sink:
            self.event_sink(event_payload(event_type, action, **details))
