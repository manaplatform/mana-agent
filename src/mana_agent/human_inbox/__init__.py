"""Durable human-in-the-loop inbox public API."""

from __future__ import annotations

from typing import Any

from mana_agent.config.settings import mana_home

from .identity import FileIdentityDirectory, ReviewerIdentity, StaticIdentityDirectory
from .models import (
    AgentInboxObservation,
    ClarificationField,
    ClarificationValidation,
    DeliveryAttempt,
    EscalationPolicy,
    ExpiryBehavior,
    ExpectedResponseType,
    HumanResponse,
    InboxAuditEvent,
    InboxItem,
    InboxQuery,
    InboxRequest,
    InboxRequestType,
    InboxStatus,
    ReminderPolicy,
    ReconciliationReport,
    ResponseOperation,
    ResponseSubmission,
    ReviewerAssignment,
    ReviewerType,
    RiskLevel,
)
from .notifications import DashboardNotificationAdapter, NotificationAdapter
from .repository import InboxConcurrentUpdateError, LocalInboxRepository
from .service import HumanInboxService
from .tokens import ResponseTokenSigner


def _reconcile_transactional_action(item: InboxItem) -> None:
    if not item.action_intent_id.startswith("act_"):
        return
    from mana_agent.transactional_actions.approvals import ApprovalRegistry
    from mana_agent.transactional_actions.models import ActionState, ApprovalScope, utc_now
    from mana_agent.transactional_actions.store import ActionStore

    root = mana_home() / "transactional_actions"
    store = ActionStore(root)
    action = store.get_action(item.action_intent_id)
    if action is None:
        return
    if item.status is InboxStatus.APPROVED and item.action_digest != action.approval_digest():
        raise PermissionError("persisted human response no longer matches the exact action digest")
    approvals = ApprovalRegistry(root / "approvals")
    if item.status is InboxStatus.APPROVED and action.state is ActionState.AWAITING_APPROVAL:
        transaction_binding = ""
        if action.transaction_id:
            transaction = store.get_transaction(action.transaction_id)
            transaction_binding = transaction.binding_digest() if transaction is not None else ""
        if approvals.find_valid(action, transaction_binding_digest=transaction_binding) is None:
            scope = action.policy_decision.required_approval_scope if action.policy_decision else ApprovalScope.ACTION_ONCE
            approvals.issue(
                action,
                approved_by=item.response_actor_id,
                ttl_seconds=max(1, int((item.expires_at - utc_now()).total_seconds())),
                scope=scope or ApprovalScope.ACTION_ONCE,
                transaction_binding_digest=transaction_binding,
            )
        store.append_audit(action, "action.approval.granted", {
            "inbox_item_id": item.inbox_item_id,
            "response_actor_id": item.response_actor_id,
        })
    elif item.status in {InboxStatus.DENIED, InboxStatus.CANCELLED, InboxStatus.SUPERSEDED, InboxStatus.EXPIRED} and action.state is ActionState.AWAITING_APPROVAL:
        approvals.invalidate_for_action(action.action_id)
        action.transition(ActionState.EXPIRED if item.status is InboxStatus.EXPIRED else ActionState.CANCELLED)
        action.error = f"durable human decision: {item.status.value}"
        store.save_action(action)
        store.release_idempotency(action)
        store.append_audit(action, f"action.approval.{item.status.value}", {
            "inbox_item_id": item.inbox_item_id,
            "response_actor_id": item.response_actor_id,
        })


def _current_action_digest(action_intent_id: str) -> str | None:
    if not action_intent_id:
        return ""
    if action_intent_id.startswith("act_"):
        from mana_agent.transactional_actions.store import ActionStore

        action = ActionStore(mana_home() / "transactional_actions").get_action(action_intent_id)
        return action.approval_digest() if action is not None else ""
    if action_intent_id.startswith(("server:", "remote:")):
        repository = LocalInboxRepository(mana_home() / "inbox")
        items = [
            item
            for item in repository.list()
            if item.action_intent_id == action_intent_id
        ]
        if not items or not items[0].protected_context_ref:
            return ""
        context = repository.read_protected_context(items[0].protected_context_ref)
        if action_intent_id.startswith("server:"):
            server_action = context.get("server_action")
            return (
                str(server_action.get("exact_action_key") or "")
                if isinstance(server_action, dict)
                else ""
            )
        from mana_agent.remote_execution.models import RemoteExecutionRequest

        request = RemoteExecutionRequest.model_validate(context.get("remote_request"))
        return request.exact_action_key()
    return None


def default_human_inbox_service(*, branch_controller: Any | None = None) -> HumanInboxService:
    root = mana_home() / "inbox"
    from mana_agent.services.execution_event_hub import get_execution_event_hub

    if branch_controller is None:
        from mana_agent.config.settings import Settings
        from mana_agent.execution_supervisor import ExecutionSupervisor, ExecutionSupervisorConfig

        supervisor_config = ExecutionSupervisorConfig.from_settings(Settings()).model_copy(
            update={"startup_recovery": False}
        )
        branch_controller = ExecutionSupervisor(supervisor_config)

    hub = get_execution_event_hub()
    publish = lambda payload: hub.publish(payload, persist=False)
    return HumanInboxService(
        repository=LocalInboxRepository(root),
        identities=FileIdentityDirectory(root / "identities.json"),
        token_signer=ResponseTokenSigner(root / "response-signing.key"),
        branch_controller=branch_controller,
        notification_adapters=[DashboardNotificationAdapter(publish)],
        event_sink=publish,
        terminal_response_handler=_reconcile_transactional_action,
        action_digest_resolver=_current_action_digest,
    )


__all__ = [
    "AgentInboxObservation",
    "ClarificationField",
    "ClarificationValidation",
    "DeliveryAttempt",
    "EscalationPolicy",
    "ExpiryBehavior",
    "ExpectedResponseType",
    "FileIdentityDirectory",
    "HumanResponse",
    "HumanInboxService",
    "InboxAuditEvent",
    "InboxConcurrentUpdateError",
    "InboxItem",
    "InboxQuery",
    "InboxRequest",
    "InboxRequestType",
    "InboxStatus",
    "LocalInboxRepository",
    "NotificationAdapter",
    "ReminderPolicy",
    "ReconciliationReport",
    "ResponseOperation",
    "ResponseSubmission",
    "ResponseTokenSigner",
    "ReviewerAssignment",
    "ReviewerIdentity",
    "ReviewerType",
    "RiskLevel",
    "StaticIdentityDirectory",
    "default_human_inbox_service",
]
