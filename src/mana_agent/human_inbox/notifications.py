"""Minimal-disclosure notification adapters."""

from __future__ import annotations

from typing import Any, Callable, Protocol

from pydantic import Field

from .models import StrictModel


class InboxNotification(StrictModel):
    inbox_item_id: str
    destination: str
    title: str
    summary: str
    risk_level: str
    request_type: str
    expires_at: str
    response_reference: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class NotificationResult(StrictModel):
    delivered: bool
    external_message_id: str = ""
    error: str = ""


class NotificationAdapter(Protocol):
    name: str
    def deliver(self, notification: InboxNotification) -> NotificationResult: ...


class DashboardNotificationAdapter:
    name = "dashboard"

    def __init__(self, event_sink: Callable[[dict[str, Any]], None]) -> None:
        self.event_sink = event_sink

    def deliver(self, notification: InboxNotification) -> NotificationResult:
        self.event_sink({
            "type": "inbox.item.created",
            "event_type": "inbox.item.created",
            "kind": "human_inbox",
            "status": "waiting",
            "title": notification.title,
            "message": notification.summary,
            "metadata": {
                "human_inbox": True,
                "authoritative_state_required": True,
                **notification.model_dump(mode="json"),
            },
        })
        return NotificationResult(delivered=True, external_message_id=notification.inbox_item_id)


class ChatHistoryNotificationAdapter:
    """Surface a minimal durable-inbox notice in CLI/TUI chat history."""

    name = "chat_history"

    def deliver(self, notification: InboxNotification) -> NotificationResult:
        from mana_agent.chat.events import CodingActivityEvent
        from mana_agent.chat.history import get_history

        get_history().add(CodingActivityEvent(activity={
            "event_type": "human_inbox.waiting",
            "title": notification.title,
            "metadata": {
                "human_inbox": True,
                "authoritative_state_required": True,
                "inbox_item_id": notification.inbox_item_id,
                "request_type": notification.request_type,
                "risk_level": notification.risk_level,
                "expires_at": notification.expires_at,
            },
        }))
        return NotificationResult(
            delivered=True,
            external_message_id=notification.inbox_item_id,
        )
