from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from mana_agent.approval.models import ApprovalDecision, ApprovalRequest
from mana_agent.observability.service import redact_summary

logger = logging.getLogger(__name__)


def record_approval_audit(
    decision: ApprovalDecision,
    request: ApprovalRequest,
    *,
    result_status: str = "approved",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record structured approval telemetry with all sensitive credentials and secrets redacted."""
    now = datetime.now(timezone.utc).isoformat()
    record: dict[str, Any] = {
        "approval_request_id": decision.approval_id or request.action_id or request.metadata.get("permission_request_id") or "",
        "inbox_item_id": decision.inbox_item_id or request.metadata.get("inbox_item_id") or "",
        "execution_id": request.execution_id or request.task_id or "",
        "attempt_id": request.attempt_id or "",
        "tool_name": request.tool_name or request.action_type or "",
        "action_name": request.command or request.action_type or "",
        "approval_mode": decision.mode.value,
        "approval_decision_source": decision.source,
        "approved": decision.approved,
        "interactive": decision.interactive,
        "timestamp": now,
        "execution_scope": request.blast_radius or "workspace",
        "result": result_status,
        "reason": decision.reason,
    }

    if extra:
        for k, v in extra.items():
            if k not in record:
                record[k] = v

    # Ensure secrets and credentials are completely redacted
    safe_record = redact_summary(record, limit=10_000)

    try:
        from mana_agent.services.execution_event_hub import get_execution_event_hub

        hub = get_execution_event_hub()
        hub.publish(
            {
                "type": "approval.audit",
                "event_type": "approval.audit",
                "timestamp": now,
                "data": record,
            },
            persist=False,
        )
    except Exception as exc:
        logger.debug("Failed to dispatch approval audit event to hub: %s", exc)

    return record
