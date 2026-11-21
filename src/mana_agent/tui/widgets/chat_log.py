"""The scrollable transcript, including inline execution and steering."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Markdown, Static

from mana_agent.chat.activity import safe_text, sanitize_legacy_envelope
from mana_agent.chat.attachments import format_size
from mana_agent.chat.events import (
    AssistantMessageEvent, ChatEvent, CodingActivityEvent, StreamTokenEvent,
    ToolCallEvent, ToolResultEvent, UserMessageEvent,
)
from mana_agent.chat.history import ChatHistory
from mana_agent.tui.presentation import glyph
from mana_agent.tui.widgets.execution_panel import ExecutionPanel
from mana_agent.tui.widgets.selectable_text import SelectableText


class HistoryReady(Message):
    """Wake the Textual thread once for a burst of history updates."""


class UserMessage(Horizontal):
    DEFAULT_CSS = """
    UserMessage { width: 1fr; height: auto; }
    UserMessage > Static { width: 2; height: 1; }
    """

    def __init__(self, text: str) -> None:
        super().__init__(classes="user-message-row")
        self.source = SelectableText(text, classes="user-message")

    def compose(self):
        yield Static(glyph("❯", ">"), classes="user-prompt", markup=False)
        yield self.source


class AssistantReply(Vertical):
    def __init__(self, text: str, *, streaming: bool) -> None:
        super().__init__(classes="assistant-message")
        self.source = SelectableText(text)
        self.markdown = Markdown(text if not streaming else "", open_links=False)
        self.copy_mode = Button("Copy mode", classes="copy-mode")
        self._streaming = streaming
        self._copying = False
        self._pending: str | None = None

    @property
    def text(self) -> str:
        return self._pending if self._pending is not None else self.source.text

    def compose(self):
        self.markdown.display = not self._streaming
        self.copy_mode.display = not self._streaming
        self.source.display = self._streaming
        yield self.markdown
        yield self.source
        yield self.copy_mode

    def update_text(self, text: str) -> None:
        if self.source.selected_text:
            self._pending = text
        else:
            self._pending = None
            if self.source.text != text:
                self.source.load_text(text)

    def on_text_area_selection_changed(self, event) -> None:
        if self._pending is not None and event.selection.is_empty:
            self.update_text(self._pending)
            if not (self._copying or self._streaming):
                self.source.display = False

    def finalize(self, text: str) -> None:
        self._streaming = False
        self._copying |= bool(self.source.selected_text)
        self.update_text(text)
        if self.markdown.is_mounted:
            self.markdown.update(text)
        else:
            self.markdown = Markdown(text, open_links=False)
        self.markdown.display = not self._copying
        self.source.display = self._copying
        self.copy_mode.display = True
        if self._copying:
            self.copy_mode.label = "Rendered view"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button is not self.copy_mode:
            return
        event.stop()
        self._copying = not self._copying
        self.source.display = self._copying
        self.markdown.display = not self._copying
        self.copy_mode.label = "Rendered view" if self._copying else "Copy mode"
        if self._copying:
            self.source.focus()


class ChatLog(VerticalScroll):
    message_count: reactive[int] = reactive(0)

    def __init__(self, history: ChatHistory | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._history: ChatHistory | None = history
        self._unsubscribe: Callable[[], None] | None = None
        self._assistant_widgets: dict[str, AssistantReply] = {}
        self._streaming: dict[str, tuple[AssistantMessageEvent, AssistantReply]] = {}
        self._execution_panels: dict[str, ExecutionPanel] = {}
        self._user_widgets_by_turn: dict[str, UserMessage] = {}
        self._steered_widgets_by_turn: dict[str, UserMessage] = {}
        self._steering_widgets: dict[str, UserMessage] = {}
        self._current_turn_id: str | None = None
        self._rendered_ids: set[str] = set()
        self._pending_lock = threading.Lock()
        self._pending_events: list[ChatEvent] = []
        self._pending_activity: dict[tuple[str, str], CodingActivityEvent] = {}
        self._pending_tokens: set[tuple[str, str | None]] = set()
        self._pending_wake = False
        self._follow_pending = False
        self.startup = Static("", classes="startup", markup=False)
        self.startup.display = False
        self.live = Static("", id="live-status", markup=False)
        self.live.display = False
        self._active_turn = ""
        self._live_started: float | None = None
        self._status_timer = None

    def compose(self):
        yield self.startup
        yield self.live

    def set_history(self, history: ChatHistory) -> None:
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        self._history = history
        if self.is_mounted:
            self.clear_log()
            self._subscribe()
            self._replay_history()

    def _subscribe(self) -> None:
        if self._history is not None:
            current = self._history
            self._unsubscribe = current.subscribe(lambda event: self._enqueue(event) if self._history is current else None)

    def on_mount(self) -> None:
        self._subscribe()
        self._status_timer = self.set_interval(1, self.refresh_activity_status)
        self.call_after_refresh(self._replay_history)

    def on_unmount(self) -> None:
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        if self._status_timer:
            self._status_timer.stop()
            self._status_timer = None

    def _enqueue(self, event: ChatEvent) -> None:
        from mana_agent.chat.activity import operation_key

        with self._pending_lock:
            if isinstance(event, CodingActivityEvent) and not event.activity["type"].startswith("steering."):
                self._pending_activity[event.turn_id, operation_key(event.activity)] = event
            elif isinstance(event, StreamTokenEvent):
                self._pending_tokens.add((event.turn_id, event.assistant_event_id))
            else:
                self._pending_events.append(event)
            if self._pending_wake:
                return
            self._pending_wake = True
        self.post_message(HistoryReady())

    def on_history_ready(self, _: HistoryReady) -> None:
        with self._pending_lock:
            events, self._pending_events = self._pending_events, []
            activity, self._pending_activity = self._pending_activity, {}
            tokens, self._pending_tokens = self._pending_tokens, set()
            self._pending_wake = False
        follow = self.following
        for event in events:
            self._safe_render(event)
        for event in activity.values():
            self._safe_render(event)
        if tokens:
            for event, reply in self._streaming.values():
                addresses = {(event.turn_id, event.event_id), (event.turn_id, None), ("", event.event_id), ("", None)}
                if addresses & tokens:
                    reply.update_text(event.content)
        self.refresh_activity_status()
        if follow:
            self.follow_after_refresh()

    @property
    def following(self) -> bool:
        return self.max_scroll_y - self.scroll_y <= 2 and not any(
            widget.selected_text for widget in self.query(SelectableText)
        )

    def follow_after_refresh(self) -> None:
        if self._follow_pending:
            return
        self._follow_pending = True

        def follow() -> None:
            self._follow_pending = False
            self.scroll_end(animate=False, force=True)

        self.call_after_refresh(follow)

    def _scroll_to_latest(self) -> None:
        self.scroll_end(animate=False, force=True)
        self.follow_after_refresh()

    def set_active_turn(self, turn_id: str | None) -> None:
        turn = turn_id or ""
        if turn != self._active_turn:
            self._active_turn = turn
            self._live_started = time.monotonic() if turn else None
        self.refresh_activity_status()

    def refresh_activity_status(self) -> None:
        if not self.is_mounted:
            return
        panel = self._execution_panels.get(self._active_turn)
        if panel is None and not self._active_turn and self._current_turn_id:
            panel = self._execution_panels.get(self._current_turn_id)
            if panel is not None and not panel._waiting:
                panel = None
        label = panel.live_label() if panel else ""
        if (self._active_turn or getattr(self.app, "has_pending_execution_approval", False)) and getattr(self.app, "interrupt_requested", False) and not (panel and panel._terminal):
            label = "Interrupt requested"
        elif getattr(self.app, "has_pending_execution_approval", False):
            label = "Waiting for approval"
        elif self._active_turn and not label and not (panel and panel._terminal):
            label = "Thinking"
        elapsed = int(time.monotonic() - self._live_started) if self._live_started is not None else None
        if label:
            suffix = []
            if elapsed is not None:
                suffix.append(f"{elapsed}s")
            if self._active_turn and getattr(self.app, "can_interrupt", False) and not getattr(self.app, "interrupt_requested", False):
                suffix.append("Esc to interrupt")
            separator = glyph(" · ", " | ")
            if suffix:
                label += f" ({separator.join(suffix)})"
            label = f"{glyph('•', '*')} {label}"
        if str(self.live.renderable) != label:
            self.live.update(label)
        self.live.display = bool(label)
        refresh_footer = getattr(self.app, "refresh_footer", None)
        if refresh_footer is not None:
            refresh_footer()

    def get_execution_panel(self, turn_id: str) -> ExecutionPanel | None:
        return self._execution_panels.get(turn_id)

    def _get_or_create_execution_panel(self, turn_id: str) -> ExecutionPanel:
        panel = self._execution_panels.get(turn_id)
        if panel is None:
            panel = self._execution_panels[turn_id] = ExecutionPanel(turn_id=turn_id)
            self.mount(panel, before=self.live)
            self.call_after_refresh(self._position_panel, turn_id)
        return panel

    def _position_panel(self, turn_id: str) -> None:
        panel = self._execution_panels.get(turn_id)
        target = self._steered_widgets_by_turn.get(turn_id) or self._user_widgets_by_turn.get(turn_id)
        children = list(self.children)
        if panel in children and target in children and children.index(panel) != children.index(target) + 1:
            self.move_child(panel, after=target)

    def _add_user_message(self, event: UserMessageEvent) -> UserMessage:
        content = safe_text(event.content, 100000)
        if event.is_steering:
            content += f"\nSteering {event.steering_status or 'pending'}"
        if event.attachments:
            for attachment in event.attachments:
                get = attachment.get if isinstance(attachment, dict) else lambda key, default: getattr(attachment, key, default)
                name = safe_text(get("filename", "file"))
                content += f"\n  {glyph('📎', '+')} {name} ({format_size(get('size_bytes', 0))})"
        widget = UserMessage(content)
        panel = self._execution_panels.get(event.turn_id)
        self.mount(widget, before=panel if panel is not None and panel in self.children else self.live)
        if event.is_steering:
            self._steering_widgets[event.request_id] = widget
            self._steered_widgets_by_turn[event.turn_id] = widget
        else:
            self._user_widgets_by_turn[event.turn_id] = widget
            self._current_turn_id = event.turn_id
        if panel is not None:
            self.call_after_refresh(self._position_panel, event.turn_id)
        return widget

    def _add_or_update_assistant(self, event: AssistantMessageEvent) -> AssistantReply:
        content = event.content or ""
        streaming = None if event.is_streaming else self._streaming.pop(event.turn_id, None)
        if streaming is not None:
            _, reply = streaming
            reply.finalize(content)
        else:
            reply = AssistantReply(content, streaming=event.is_streaming)
            self.mount(reply, before=self.live)
            if event.is_streaming:
                self._streaming[event.turn_id] = event, reply
        self._assistant_widgets[event.event_id] = reply
        return reply

    def _add_tool_call(self, event: ToolCallEvent) -> ExecutionPanel:
        panel = self._get_or_create_execution_panel(event.turn_id)
        panel.update_event({
            "type": "tool.started", "event_id": event.event_id, "tool_name": event.tool_name,
            "tool_call_id": event.call_id, "status": "running", "title": event.tool_name,
            "detail": event.summary, "started_at": event.timestamp,
        }, legacy=True)
        return panel

    def _add_tool_result(self, event: ToolResultEvent) -> ExecutionPanel:
        panel = self._get_or_create_execution_panel(event.turn_id)
        result = event.result
        if isinstance(result, (dict, list)):
            result = sanitize_legacy_envelope({"result": result}).get("result")
            result = json.dumps(result, ensure_ascii=False, indent=2, default=str)
        panel.update_event({
            "type": "tool.finished", "event_id": event.event_id, "tool_name": event.tool_name,
            "tool_call_id": event.call_id, "title": event.tool_name,
            "status": "success" if event.success else "failed", "duration_ms": event.duration_ms,
            "summary": event.summary, "output_preview": safe_text(result),
            "metadata": {"error": safe_text(event.error)},
        }, legacy=True)
        return panel

    def _add_coding_activity(self, event: CodingActivityEvent) -> None:
        data = event.activity
        panel = self._get_or_create_execution_panel(event.turn_id)
        panel.update_event(data, projected=True)
        if data["type"].startswith("steering."):
            meta = data.get("metadata") or {}
            request_id = str(meta.get("request_id") or data.get("operation_id") or "")
            widget = self._steering_widgets.get(request_id)
            if widget is not None and self._history is not None:
                message = next((row for row in self._history.get_events(event.turn_id)
                                if isinstance(row, UserMessageEvent) and row.request_id == request_id), None)
                if message and not widget.source.selected_text:
                    widget.source.load_text(f"{safe_text(message.content, 16000)}\nSteering {message.steering_status}")

    def _safe_render(self, event: ChatEvent) -> None:
        if isinstance(event, CodingActivityEvent):
            self._add_coding_activity(event)
            return
        if event.event_id in self._rendered_ids:
            return
        self._rendered_ids.add(event.event_id)
        if isinstance(event, UserMessageEvent):
            self._add_user_message(event)
            self.message_count += 1
        elif isinstance(event, AssistantMessageEvent):
            self._add_or_update_assistant(event)
            self.message_count += 1
        elif isinstance(event, ToolCallEvent):
            self._add_tool_call(event)
        elif isinstance(event, ToolResultEvent):
            self._add_tool_result(event)

    def clear_log(self) -> None:
        self.remove_children(child for child in self.children if child not in (self.startup, self.live))
        self._assistant_widgets.clear()
        self._streaming.clear()
        self._execution_panels.clear()
        self._user_widgets_by_turn.clear()
        self._steering_widgets.clear()
        self._steered_widgets_by_turn.clear()
        self._current_turn_id = None
        self._rendered_ids.clear()
        self.message_count = 0
        self.set_active_turn(None)
        with self._pending_lock:
            self._pending_events.clear()
            self._pending_activity.clear()
            self._pending_tokens.clear()
            self._pending_wake = False

    def _replay_history(self) -> None:
        if self._history is not None:
            for event in self._history.get_events():
                self._safe_render(event)
        self._scroll_to_latest()
