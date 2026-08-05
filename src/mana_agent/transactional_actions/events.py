from __future__ import annotations

from typing import Any, Callable

from .models import ActionIntent

ACTION_EVENT_TYPES = frozenset({
    "action.proposed", "action.preview.ready", "action.policy.allowed", "action.policy.denied",
    "action.approval.required", "action.approval.granted", "action.approval.denied",
    "action.approval.expired", "action.approval.invalidated", "action.execution.started",
    "action.policy.revalidated",
    "action.verification.started", "action.verification.completed", "action.committed",
    "action.compensation.started", "action.compensation.completed", "action.manual_recovery.required",
})


def event_payload(event_type: str, action: ActionIntent, **details: Any) -> dict[str, Any]:
    if event_type not in ACTION_EVENT_TYPES:
        raise ValueError(f"unsupported transactional action event: {event_type}")
    preview = action.preview.redacted() if action.preview else None
    approval_event = event_type.startswith("action.approval.")
    return {
        "type": event_type,
        "event_type": event_type,
        "kind": "transactional_action",
        "status": action.state.value,
        "title": event_type.replace(".", " ").title(),
        "message": action.policy_decision.explanation if action.policy_decision else event_type,
        "metadata": {
            "action_id": action.action_id,
            "inbox_item_id": action.inbox_item_id,
            "permission_request_id": action.inbox_item_id if approval_event else "",
            "permission_scope": "transactional_action.once" if event_type == "action.approval.required" else "",
            "transactional_action_approval": event_type == "action.approval.required",
            "transaction_id": action.transaction_id,
            "parent_task_id": action.parent_task_id,
            "tool_name": action.tool_name,
            "operation_name": action.operation_name,
            "target_resources": action.target_resources,
            "preview": preview,
            "approval_display": {
                "target_resources": action.target_resources,
                "preview": preview,
                "policy_explanation": action.policy_decision.explanation if action.policy_decision else "",
                "policy_reason_codes": action.policy_decision.reason_codes if action.policy_decision else [],
                "reversibility": action.reversibility.value,
                "blast_radius": action.blast_radius.value,
                "data_disclosure": action.data_disclosure.value,
            },
            "preview_digest": action.preview_digest,
            "reversibility": action.reversibility.value,
            "blast_radius": action.blast_radius.value,
            "data_disclosure": action.data_disclosure.value,
            "policy_decision": action.policy_decision.model_dump(mode="json") if action.policy_decision else None,
            "verification": action.verification.model_dump(mode="json") if action.verification else None,
            "compensation": action.compensation.model_dump(mode="json") if action.compensation else None,
            **details,
        },
    }


ActionEventSink = Callable[[dict[str, Any]], None]
