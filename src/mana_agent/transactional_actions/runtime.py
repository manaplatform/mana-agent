from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mana_agent.config.settings import mana_home
from mana_agent.services.execution_event_hub import get_execution_event_hub

from .adapters import FileActionAdapter
from .approvals import ApprovalRegistry
from .gateway import ActionGateway, ApprovalRequired
from .policy import ActionPolicy, PolicyConfig
from .store import ActionStore


def default_action_gateway(
    workspace_root: Path,
    *,
    allowed_http_hosts: tuple[str, ...] = (),
    surface_approval_events: bool = True,
) -> ActionGateway:
    root = mana_home() / "transactional_actions"
    hub = get_execution_event_hub()
    return ActionGateway(
        store=ActionStore(root),
        policy=ActionPolicy(PolicyConfig(
            workspace_roots=(workspace_root.resolve(),),
            allowed_http_hosts=allowed_http_hosts,
        )),
        approvals=ApprovalRegistry(root / "approvals"),
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


def approve_action(workspace_root: Path, action_id: str, *, approved_by: str, ttl_seconds: int = 300) -> str:
    """Issue a narrow approval; execution must separately present the returned ID."""
    gateway = default_action_gateway(workspace_root)
    action = gateway.store.get_action(action_id)
    if action is None:
        raise LookupError("unknown action")
    _assert_workspace_scope(workspace_root, action)
    return gateway.approve(action_id, approved_by=approved_by, ttl_seconds=ttl_seconds)


def approve_transaction(
    workspace_root: Path,
    transaction_id: str,
    *,
    approved_by: str,
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
        ttl_seconds=ttl_seconds,
    )


def deny_action(workspace_root: Path, action_id: str, *, denied_by: str) -> dict[str, Any]:
    gateway = default_action_gateway(workspace_root)
    pending = gateway.store.get_action(action_id)
    if pending is None:
        raise LookupError("unknown action")
    _assert_workspace_scope(workspace_root, pending)
    action = gateway.deny(action_id, denied_by=denied_by)
    return action.model_dump(mode="json")


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
