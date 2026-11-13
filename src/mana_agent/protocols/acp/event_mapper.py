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
        from acp.schema import (
            AgentMessageChunk,
            TextContentBlock,
            ToolCallProgress,
            ToolCallStart,
            UserMessageChunk,
        )

        if isinstance(event, UserMessageEvent):
            return [UserMessageChunk(content=TextContentBlock(type="text", text=str(event.content)))]
        if isinstance(event, (AssistantMessageEvent, StreamTokenEvent)):
            text = event.content if isinstance(event, AssistantMessageEvent) else event.token
            return [AgentMessageChunk(content=TextContentBlock(type="text", text=str(text)))] if text else []
        if isinstance(event, ToolCallEvent):
            return [
                ToolCallStart(
                    tool_call_id=event.call_id,
                    title=event.summary or event.tool_name,
                    kind=self._tool_kind(event.tool_name),
                    status="in_progress",
                    raw_input=redact_protocol_value(event.args),
                )
            ]
        if isinstance(event, ToolResultEvent):
            return [
                ToolCallProgress(
                    tool_call_id=event.call_id,
                    title=event.summary or event.tool_name,
                    status="completed" if event.success else "failed",
                    raw_output=redact_protocol_value(event.result if event.success else event.error),
                )
            ]
        if isinstance(event, dict):
            kind = str(event.get("type") or event.get("event_type") or "")
            if kind in {"assistant", "assistant_message", "text", "token"}:
                text = str(event.get("text") or event.get("content") or event.get("token") or "")
                return [AgentMessageChunk(content=TextContentBlock(type="text", text=str(text)))] if text else []
            if kind in {"warning", "error", "verification"}:
                text = str(event.get("message") or event.get("summary") or event.get("error") or "")
                return [AgentMessageChunk(content=TextContentBlock(type="text", text=str(text)))] if text else []
        return []

    @staticmethod
    def _tool_kind(name: str) -> str:
        lowered = str(name).lower()
        for marker, kind in (("read", "read"), ("edit", "edit"), ("write", "edit"), ("delete", "delete"), ("search", "search"), ("terminal", "execute"), ("shell", "execute")):
            if marker in lowered:
                return kind
        return "other"
