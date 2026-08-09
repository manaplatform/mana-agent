"""Create durable human interventions for unrecoverable connector failures."""

from __future__ import annotations

import logging
from typing import Any

from mana_agent.utils.redaction import redact_secrets

from .models import HealthReasonCode

logger = logging.getLogger(__name__)


def _safe_operator_message(message: str) -> str:
    """Return a short operator-facing detail that cannot carry credentials."""
    import re

    text = str(redact_secrets(message) or "")
    # Strip assignment-style secret material (token=..., bearer ..., etc.).
    text = re.sub(
        r"(?i)\b(token|access_token|refresh_token|password|secret|api[_-]?key|bearer|authorization)\b\s*[:=]\s*\S+",
        r"\1=[redacted]",
        text,
    )
    text = re.sub(r"(?i)\b[A-Za-z0-9_-]{32,}\b", "[redacted]", text)
    text = text.strip()
    if not text:
        return "See connector health reason code for details."
    return text[:240]


def build_auth_intervention_request(
    *,
    connector_id: str,
    connector_type: str,
    reason_code: HealthReasonCode,
    message: str,
    tenant_id: str = "local",
    project_id: str = "mana",
    requested_by_agent_id: str = "connector-health",
) -> dict[str, Any]:
    """Build a minimal, secret-free HITL request payload."""
    safe_message = _safe_operator_message(message)
    title = f"Reconnect {connector_type}"
    if reason_code in {
        HealthReasonCode.AUTH_EXPIRED,
        HealthReasonCode.AUTH_REVOKED,
        HealthReasonCode.TOKEN_REFRESH_FAILED,
    }:
        summary = (
            f"Connector: {connector_type}\n"
            f"Id: {connector_id}\n"
            f"State: auth_required\n"
            f"Reason: {reason_code.value.lower().replace('_', ' ')}\n"
            f"Required action: reconnect account / reauthorize credentials\n"
            f"Detail: {safe_message}"
        )
    else:
        summary = (
            f"Connector: {connector_type}\n"
            f"Id: {connector_id}\n"
            f"Reason: {reason_code.value}\n"
            f"Required action: operator intervention\n"
            f"Detail: {safe_message}"
        )
    return {
        "request_type": "approval",
        "tenant_id": tenant_id,
        "project_id": project_id,
        "requested_by_agent_id": requested_by_agent_id,
        "title": title,
        "summary": summary,
        "minimal_context": {
            "connector_id": connector_id,
            "connector_type": connector_type,
            "state": "auth_required",
            "reason_code": reason_code.value,
        },
        "risk_level": "high",
    }


class ConnectorHitlBridge:
    def __init__(self, *, inbox_service: Any | None = None) -> None:
        self.inbox_service = inbox_service
        self._created: dict[str, str] = {}  # connector_id -> inbox_item_id

    def bind(self, inbox_service: Any) -> None:
        self.inbox_service = inbox_service

    def request_auth_intervention(
        self,
        *,
        connector_id: str,
        connector_type: str,
        reason_code: HealthReasonCode,
        message: str,
    ) -> str | None:
        if connector_id in self._created:
            return self._created[connector_id]
        if self.inbox_service is None:
            logger.warning(
                "connector.hitl.skipped connector_id=%s reason=no_inbox_service",
                connector_id,
            )
            return None
        payload = build_auth_intervention_request(
            connector_id=connector_id,
            connector_type=connector_type,
            reason_code=reason_code,
            message=message,
        )
        try:
            from datetime import timedelta
            from uuid import uuid4

            from mana_agent.human_inbox.models import (
                InboxRequest,
                InboxRequestType,
                ResponseOperation,
                ReviewerAssignment,
                ReviewerType,
                RiskLevel,
                utc_now,
            )

            dedupe = f"connector-health:{connector_id}:{reason_code.value}"
            request = InboxRequest(
                request_type=InboxRequestType.APPROVAL,
                tenant_id=payload["tenant_id"],
                project_id=payload["project_id"],
                task_id=f"connector-health:{connector_id}",
                branch_id=f"connector-health:{connector_id}:auth",
                requested_by_agent_id=payload["requested_by_agent_id"],
                reviewer=ReviewerAssignment(
                    reviewer_type=ReviewerType.ROLE,
                    reviewer_id="operator",
                ),
                title=payload["title"],
                summary=payload["summary"],
                minimal_context=payload["minimal_context"],
                risk_level=RiskLevel.HIGH,
                action_intent_id=f"connector:{connector_id}:reauth",
                allowed_responses=[ResponseOperation.APPROVE, ResponseOperation.DENY],
                expires_at=utc_now() + timedelta(days=7),
                idempotency_key=dedupe,
                deduplication_key=dedupe,
            )
            item = self.inbox_service.create(request)
            inbox_id = item.inbox_item_id
            self._created[connector_id] = inbox_id
            logger.info(
                "connector.hitl.created connector_id=%s inbox_item_id=%s reason=%s",
                connector_id,
                inbox_id,
                reason_code.value,
            )
            return inbox_id
        except Exception:
            logger.exception("failed to create connector HITL item for %s", connector_id)
            return None
