"""Stable expandable operation rows inside the existing Coding Activity box."""
from __future__ import annotations

import asyncio
from typing import Any
from textual.containers import Horizontal
from textual.widgets import Button, Collapsible, Static
from mana_agent.chat.operations import operation_details
from mana_agent.tui.widgets.selectable_text import SelectableText


class OperationRow(Collapsible):
    DEFAULT_CSS = """
    OperationRow { padding: 0; margin: 0; height: auto; }
    OperationRow SelectableText { max-height: 20; }
    OperationRow Horizontal { height: 3; }
    """

    def __init__(self, event: dict[str, Any]):
        self.event = event
        self.output = SelectableText()
        self._loaded: dict[str, str] = {}
        self._pending_text = ""
        super().__init__(Horizontal(
            Button("Copy command", classes="operation-copy-command"),
            Button("Copy diff", classes="operation-copy-diff"),
            Button("Retained output", classes="operation-retained-output")),
            self.output, title="Operation", collapsed=True)
        self.update_operation(event)

    def update_operation(self, event: dict[str, Any]):
        self.event = event
        meta = event.get("metadata") or {}
        status = str(event.get("status") or "running")
        marker = "✓" if status in {"success", "completed", "done"} else "✗" if status in {"failed", "timed_out"} else "●"
        action = event.get("command") or meta.get("command") or event.get("path") or meta.get("path") or event.get("tool_name") or event.get("title")
        self.title = f"{marker} {action} · {status} · {event.get('duration_ms') or 0}ms"
        body = operation_details(event)
        if self._loaded:
            body += "\n" + "\n".join(f"--- retained {channel} ---\n{text}" for channel, text in self._loaded.items())
        if self.output.selected_text:
            self._pending_text = body
        elif body != self.output.text:
            position = self.output.scroll_offset
            self.output.load_text(body)
            if self.output.is_mounted:
                self.output.scroll_to(x=position.x, y=position.y, animate=False)

    def on_mount(self):
        def flush():
            if self._pending_text and not self.output.selected_text:
                self._pending_text = ""
                self.update_operation(self.event)
        self.set_interval(0.1, flush)

    async def on_button_pressed(self, event: Button.Pressed):
        event.stop()
        meta = self.event.get("metadata") or {}
        if event.button.has_class("operation-copy-command"):
            self.app.copy_to_clipboard(str(self.event.get("command") or meta.get("command") or ""))
        elif event.button.has_class("operation-copy-diff"):
            self.app.copy_to_clipboard(str(self._loaded.get("diff") or (meta.get("streams") or {}).get("diff") or meta.get("diff") or ""))
        else:
            await self.load_retained()

    async def load_retained(self):
        from mana_agent.services.conversation_service import ConversationService
        from mana_agent.services.operation_trace import operation_content
        meta = self.event.get("metadata") or {}
        conversation = str(self.event.get("conversation_id") or self.event.get("session_id") or "")
        execution = str(self.event.get("execution_id") or self.event.get("turn_id") or "")
        if not conversation or not execution:
            return
        service = ConversationService(repository_id=str(self.event.get("repository_id") or ""))
        for channel in ("stdout", "stderr", "combined", "diff", "result"):
            try:
                page = await asyncio.to_thread(operation_content, service, conversation, execution,
                    str(self.event.get("operation_id") or ""), str(self.event.get("attempt_id") or ""),
                    operation_attempt_id=str(self.event.get("operation_attempt_id") or meta.get("operation_attempt_id") or ""),
                    channel=channel, offset=len(self._loaded.get(channel, "")), limit=16000)
                self._loaded[channel] = self._loaded.get(channel, "") + page["content"]
            except (OSError, ValueError, PermissionError) as exc:
                self._loaded[channel] = f"Retained output unavailable: {exc}"
        self.update_operation(self.event)
