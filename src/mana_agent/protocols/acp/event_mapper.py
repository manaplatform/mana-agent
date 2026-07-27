"""Convert Mana runtime events to official ACP session updates."""

from __future__ import annotations

from typing import Any

from mana_agent.chat.events import (
    AssistantMessageEvent,
    StreamTokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from mana_agent.protocols.common.security import redact_protocol_value


class AcpEventMapper:
    def map(self, event: Any) -> list[Any]:
        from acp.helpers import (
            start_tool_call,
            update_agent_message_text,
            update_tool_call,
            update_user_message_text,
        )

        if isinstance(event, UserMessageEvent):
            return [update_user_message_text(event.content)]
        if isinstance(event, (AssistantMessageEvent, StreamTokenEvent)):
            text = event.content if isinstance(event, AssistantMessageEvent) else event.token
            return [update_agent_message_text(text)] if text else []
        if isinstance(event, ToolCallEvent):
            return [
                start_tool_call(
                    event.call_id,
                    event.summary or event.tool_name,
                    kind=self._tool_kind(event.tool_name),
                    status="in_progress",
                    raw_input=redact_protocol_value(event.args),
                )
            ]
        if isinstance(event, ToolResultEvent):
            return [
                update_tool_call(
                    event.call_id,
                    title=event.summary or event.tool_name,
                    status="completed" if event.success else "failed",
                    raw_output=redact_protocol_value(event.result if event.success else event.error),
                )
            ]
        if isinstance(event, dict):
            kind = str(event.get("type") or event.get("event_type") or "")
            if kind in {"assistant", "assistant_message", "text", "token"}:
                text = str(event.get("text") or event.get("content") or event.get("token") or "")
                return [update_agent_message_text(text)] if text else []
            if kind in {"warning", "error", "verification"}:
                text = str(event.get("message") or event.get("summary") or event.get("error") or "")
                return [update_agent_message_text(text)] if text else []
        return []

    @staticmethod
    def _tool_kind(name: str) -> str:
        lowered = str(name).lower()
        for marker, kind in (("read", "read"), ("edit", "edit"), ("write", "edit"), ("delete", "delete"), ("search", "search"), ("terminal", "execute"), ("shell", "execute")):
            if marker in lowered:
                return kind
        return "other"
