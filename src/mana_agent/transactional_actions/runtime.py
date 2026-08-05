from __future__ import annotations

import hashlib
import json
import getpass
from datetime import timedelta
from pathlib import Path
from typing import Any

from mana_agent.config.settings import mana_home
from mana_agent.services.execution_event_hub import get_execution_event_hub
from mana_agent.human_inbox import default_human_inbox_service

from .adapters import FileActionAdapter
from .approvals import ApprovalRegistry
from .gateway import ActionGateway, ApprovalRequired
from .models import TransactionalRequestRecord, TransactionalRequestState, utc_now
from .policy import ActionPolicy, PolicyConfig
from .store import ActionStore


class TransactionalActionRuntime:
    """One durable composition root shared by computer tools and the chat gateway."""

    def __init__(self, *, gateway: ActionGateway, store: ActionStore, inbox_service: Any) -> None:
        self.gateway = gateway
        self.store = store
        self.inbox_service = inbox_service

    def record_request(
        self,
        *,
        state: TransactionalRequestState,
        source_decision_id: str = "",
        session_id: str = "",
        conversation_id: str = "",
        turn_id: str = "",
        task_id: str = "",
        branch_id: str = "",
        actor_id: str = "",
        client_type: str = "",
        tool_name: str = "computer",
        operation_name: str = "",
        resource_digest: str = "",
        outcome_code: str = "",
        action_id: str = "",
        create_notice: bool = False,
    ) -> TransactionalRequestRecord:
        material = json.dumps(
            {
                "source_decision_id": source_decision_id,
                "session_id": session_id,
                "turn_id": turn_id,
                "tool_name": tool_name,
                "operation_name": operation_name,
                "resource_digest": resource_digest,
            },
            sort_keys=True,
        )
        record = self.store.create_request(TransactionalRequestRecord(
            idempotency_key="request:" + hashlib.sha256(material.encode("utf-8")).hexdigest(),
            state=state,
            source_decision_id=source_decision_id,
            session_id=session_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            task_id=task_id,
            branch_id=branch_id,
            actor_id=actor_id,
            client_type=client_type,
            tool_name=tool_name,
            operation_name=operation_name,
            resource_digest=resource_digest,
            outcome_code=outcome_code,
            action_id=action_id,
        ))
        record.update(state, outcome_code=outcome_code, action_id=action_id)
        if create_notice and not record.inbox_item_id:
            record.inbox_item_id = self._record_notice(record)
        self.store.save_request(record)
        from mana_agent.utils.durable_diagnostics import append_diagnostic
        append_diagnostic(
            self.store.root / "logs" / "runtime.jsonl",
            component="transactional_actions",
            event=f"request.{state.value}",
            details={
                "request_id": record.request_id,
                "action_id": record.action_id,
                "inbox_item_id": record.inbox_item_id,
                "outcome_code": record.outcome_code,
                "operation": record.operation_name,
            },
        )
        return record

    def update_request(
        self,
        record: TransactionalRequestRecord,
        state: TransactionalRequestState,
        *,
        outcome_code: str = "",
        action_id: str = "",
        inbox_item_id: str = "",
        create_notice: bool = False,
    ) -> TransactionalRequestRecord:
        record.update(state, outcome_code=outcome_code, action_id=action_id, inbox_item_id=inbox_item_id)
        if create_notice and not record.inbox_item_id:
            record.inbox_item_id = self._record_notice(record)
        self.store.save_request(record)
        from mana_agent.utils.durable_diagnostics import append_diagnostic
        append_diagnostic(
            self.store.root / "logs" / "runtime.jsonl",
            component="transactional_actions",
            event=f"request.{state.value}",
            details={"request_id": record.request_id, "action_id": record.action_id, "inbox_item_id": record.inbox_item_id, "outcome_code": record.outcome_code},
        )
        return record

    def _record_notice(self, record: TransactionalRequestRecord) -> str:
        from mana_agent.human_inbox.models import InboxRequest, InboxRequestType, ReviewerAssignment, ReviewerType, RiskLevel

        reviewer_id = record.actor_id or getpass.getuser()
        item = self.inbox_service.create(InboxRequest(
            request_type=InboxRequestType.NOTICE,
            project_id="local",
            task_id=record.task_id or f"computer-session:{record.session_id or 'unknown'}",
            branch_id=record.branch_id or record.task_id or f"computer-session:{record.session_id or 'unknown'}",
            permission_request_id=record.request_id,
            action_intent_id=record.action_id,
            requested_by_agent_id="model_tool",
            reviewer=ReviewerAssignment(reviewer_type=ReviewerType.PERSON, reviewer_id=reviewer_id),
            title="Computer request recorded without execution",
            summary="The selected computer request did not become an executable approved action.",
            risk_level=RiskLevel.UNKNOWN,
            allowed_responses=[],
            minimal_context={"outcome_code": record.outcome_code, "operation": record.operation_name, "resource_digest": record.resource_digest},
            disclosed_fields=["outcome_code", "operation", "resource_digest"],
            expires_at=utc_now() + timedelta(days=7),
            idempotency_key=f"request-notice:{record.request_id}",
            deduplication_key=f"request-notice:{record.request_id}",
        ))
        if record.action_id:
            action = self.store.get_action(record.action_id)
            if action is not None and action.inbox_item_id != item.inbox_item_id:
                action.inbox_item_id = item.inbox_item_id
                self.store.save_action(action)
            if action is not None:
                self.inbox_service.record_action_event(
                    action.action_id,
                    "action.policy.denied" if action.state.value == "failed" else "action.request.recorded",
                    {"request_id": record.request_id, "outcome_code": record.outcome_code},
                )
        return item.inbox_item_id

    def reconcile_requests(self) -> None:
        """Repair request-to-inbox links without inventing executable actions."""
        terminal_states = {
            TransactionalRequestState.CLARIFICATION_REQUIRED,
            TransactionalRequestState.ROUTE_UNAVAILABLE,
            TransactionalRequestState.CAPABILITY_UNAVAILABLE,
            TransactionalRequestState.PERMISSION_DENIED,
            TransactionalRequestState.POLICY_DENIED,
            TransactionalRequestState.FAILED,
            TransactionalRequestState.MANUAL_RECOVERY_REQUIRED,
        }
        for record in self.store.list_requests():
            if record.action_id and self.store.get_action(record.action_id) is None:
                record.update(TransactionalRequestState.FAILED, outcome_code="action_record_missing")
                self.store.save_request(record)
                continue
            if record.state in terminal_states and not record.inbox_item_id:
                record.inbox_item_id = self._record_notice(record)
                self.store.save_request(record)
                continue
            if record.action_id and not record.inbox_item_id:
                action = self.store.get_action(record.action_id)
                if action is not None and action.inbox_item_id:
                    record.inbox_item_id = action.inbox_item_id
                    self.store.save_request(record)


def create_transactional_runtime(
    workspace_root: Path,
    *,
    inbox_service: Any | None = None,
    allowed_http_hosts: tuple[str, ...] = (),
    surface_approval_events: bool = True,
) -> TransactionalActionRuntime:
    root = mana_home() / "transactional_actions"
    hub = get_execution_event_hub()
    store = ActionStore(root)
    inbox = inbox_service or default_human_inbox_service()
    gateway = ActionGateway(
        store=store,
        policy=ActionPolicy(PolicyConfig(workspace_roots=(workspace_root.resolve(),), allowed_http_hosts=allowed_http_hosts)),
        approvals=ApprovalRegistry(root / "approvals"),
        inbox_service=inbox,
        event_sink=lambda event: (
            hub.publish(event, persist=False)
            if surface_approval_events or event.get("event_type") != "action.approval.required"
            else None
        ),
    )
    runtime = TransactionalActionRuntime(gateway=gateway, store=store, inbox_service=inbox)
    runtime.reconcile_requests()
    return runtime


def default_action_gateway(
    workspace_root: Path,
    *,
    allowed_http_hosts: tuple[str, ...] = (),
    surface_approval_events: bool = True,
    enable_human_inbox: bool = True,
) -> ActionGateway:
    if enable_human_inbox:
        return create_transactional_runtime(
            workspace_root,
            allowed_http_hosts=allowed_http_hosts,
            surface_approval_events=surface_approval_events,
        ).gateway
    root = mana_home() / "transactional_actions"
    hub = get_execution_event_hub()
    return ActionGateway(
        store=ActionStore(root),
        policy=ActionPolicy(PolicyConfig(
            workspace_roots=(workspace_root.resolve(),),
            allowed_http_hosts=allowed_http_hosts,
        )),
        approvals=ApprovalRegistry(root / "approvals"),
        # Gateway startup prepares the transactional store before route
        # selection. Do not materialize the durable human inbox until a route
        # actually needs an approval or structured human response.
        inbox_service=None,
        event_sink=lambda event: (
            hub.publish(event, persist=False)
            if surface_approval_events or event.get("event_type") != "action.approval.required"
            else None
        ),
    )


def execute_file_action(
    *,
    workspace_root: Path,
    operation: str,
    path: str,
    content: str | bytes | None = None,
    destination: str = "",
    actor: str = "model_tool",
    originating_agent: str = "coding_agent",
    parent_task_id: str = "tool_action",
    approval_id: str = "",
    desired_mode: int | None = None,
) -> dict[str, Any]:
    absolute = (workspace_root.resolve() / path).resolve()
    content_bytes = content.encode("utf-8") if isinstance(content, str) else content or b""
    material = json.dumps(
        {
            "operation": operation,
            "path": str(absolute),
            "destination": destination,
            "content_hash": hashlib.sha256(content_bytes).hexdigest(),
            "desired_mode": desired_mode,
        },
        sort_keys=True,
    )
    adapter = FileActionAdapter(
        workspace_root=workspace_root,
        operation=operation,
        path=path,
        content=content,
        destination=destination,
        parent_task_id=parent_task_id,
        actor=actor,
        originating_agent=originating_agent,
        idempotency_key="file:" + hashlib.sha256(material.encode("utf-8")).hexdigest(),
        snapshot_root=mana_home() / "transactional_actions" / "snapshots",
        desired_mode=desired_mode,
    )
    gateway = default_action_gateway(workspace_root)
    try:
        outcome = gateway.execute(adapter, approval_id=approval_id)
    except ApprovalRequired as exc:
        return {
            "ok": False,
            "error_code": "approval_required",
            "error": str(exc),
            "permission_required": True,
            "permission_request_id": exc.action.action_id,
            "inbox_item_id": exc.inbox_item_id,
            "action_id": exc.action.action_id,
            "transaction_id": exc.action.transaction_id,
            "preview": exc.action.preview.redacted() if exc.action.preview else {},
            "preview_digest": exc.action.preview_digest,
            "policy_decision": exc.action.policy_decision.model_dump(mode="json") if exc.action.policy_decision else {},
            "reversibility": exc.action.reversibility.value,
        }
    action = outcome.action
    if action.state.value != "committed":
        return {
            "ok": False,
            "error_code": "action_not_committed",
            "error": action.error or "action did not produce complete verification evidence",
            "action_id": action.action_id,
            "action_state": action.state.value,
            "verification": action.verification.model_dump(mode="json") if action.verification else {},
        }
    return {
        "ok": True,
        **outcome.result,
        "action_id": action.action_id,
        "action_state": action.state.value,
        "preview_digest": action.preview_digest,
        "verification": action.verification.model_dump(mode="json") if action.verification else {},
        "duplicate_suppressed": outcome.duplicate,
    }


def approve_action(
    workspace_root: Path,
    action_id: str,
    *,
    approved_by: str,
    reviewer_id: str | None = None,
    ttl_seconds: int = 300,
) -> str:
    """Issue a narrow approval; execution must separately present the returned ID."""
    gateway = default_action_gateway(workspace_root)
    action = gateway.store.get_action(action_id)
    if action is None:
        raise LookupError("unknown action")
    _assert_workspace_scope(workspace_root, action)
    return gateway.approve(
        action_id,
        approved_by=approved_by,
        reviewer_id=reviewer_id or getpass.getuser(),
        ttl_seconds=ttl_seconds,
    )


def approve_transaction(
    workspace_root: Path,
    transaction_id: str,
    *,
    approved_by: str,
    reviewer_id: str | None = None,
    ttl_seconds: int = 300,
) -> dict[str, str]:
    """Issue exact per-action grants bound to one durable transaction plan."""
    gateway = default_action_gateway(workspace_root)
    transaction = gateway.store.get_transaction(transaction_id)
    if transaction is None:
        raise LookupError("unknown transaction")
    for action_id in transaction.action_ids:
        action = gateway.store.get_action(action_id)
        if action is None:
            raise LookupError(f"unknown transaction action: {action_id}")
        _assert_workspace_scope(workspace_root, action)
    return gateway.approve_transaction(
        transaction_id,
        approved_by=approved_by,
        reviewer_id=reviewer_id or getpass.getuser(),
        ttl_seconds=ttl_seconds,
    )


def deny_action(
    workspace_root: Path,
    action_id: str,
    *,
    denied_by: str,
    reviewer_id: str | None = None,
) -> dict[str, Any]:
    gateway = default_action_gateway(workspace_root)
    pending = gateway.store.get_action(action_id)
    if pending is None:
        raise LookupError("unknown action")
    _assert_workspace_scope(workspace_root, pending)
    action = gateway.deny(
        action_id,
        denied_by=denied_by,
        reviewer_id=reviewer_id or getpass.getuser(),
    )
    return action.model_dump(mode="json")


def execute_approved_computer_action(
    workspace_root: Path, action_id: str, *, approval_id: str,
) -> dict[str, Any]:
    """Resume the durable exact computer action selected before approval."""
    from .computer import adapter_for_stored_action

    gateway = default_action_gateway(workspace_root)
    action = gateway.store.get_action(action_id)
    if action is None or action.tool_name != "computer":
        raise LookupError("unknown computer action")
    protected_context = (
        gateway.store.read_protected_action_context(action.protected_context_ref)
        if action.protected_context_ref
        else None
    )
    outcome = gateway.execute(
        adapter_for_stored_action(action, protected_context=protected_context),
        approval_id=approval_id,
    )
    return outcome.result


def _assert_workspace_scope(workspace_root: Path, action: Any) -> None:
    root = workspace_root.resolve()
    scoped_paths: list[str] = []
    if action.tool_name in {"file", "git"}:
        scoped_paths = action.target_resources
    elif action.tool_name == "shell":
        scoped_paths = [str(action.normalized_arguments.get("cwd") or "")]
    for raw in scoped_paths:
        path = Path(raw).expanduser().resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PermissionError("action belongs to a different workspace") from exc
