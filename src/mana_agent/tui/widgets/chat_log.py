"""
ChatLog

Reactive beautiful chat log widget for ManaChatApp.

- Subscribes to a ChatHistory instance.
- Renders distinct visual cards for User / Assistant / ToolCall / ToolResult / tokens.
- Supports live streaming: incoming StreamTokenEvent appends to the current
  assistant message (and also updates the on-screen widget).
- Uses Textual compose + dynamic mounting for message "cards".
- Leverages Rich for syntax / markdown inside Static or Markdown widgets.

This widget + ChatHistory subscription is the core fix for the
"tool events only on first message" bug.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from rich.console import Console, RenderableType
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.text import Text
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Markdown, Static

from mana_agent.chat.events import (
    AssistantMessageEvent,
    ChatEvent,
    StreamTokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from mana_agent.chat.history import ChatHistory
from mana_agent.tui.widgets.tool_card import ToolCard


class ChatHistoryMessage(Message):
    """Thread-safe notification that ChatHistory received a new event.

    Posted via ``Widget.post_message`` (non-blocking, safe from worker threads).
    Prefer this over ``App.call_from_thread`` which blocks the worker until the
    UI callback finishes and can deadlock if the UI thread is waiting on that worker.
    """

    def __init__(self, event: ChatEvent) -> None:
        super().__init__()
        self.event = event


class ChatLog(VerticalScroll):
    """
    Scrollable log that stays in sync with ChatHistory via subscription.

    Usage:
        log = ChatLog()
        log.set_history(history)   # subscribes automatically
        # later:
        history.add(UserMessageEvent("hello"))
    """

    DEFAULT_CSS = """
    ChatLog {
        height: 1fr;
        padding: 0 1;
        background: $surface;
        border: round $primary 40%;
    }

    .user-message {
        background: $primary 15%;
        border: round $primary;
        padding: 0 1;
        margin: 1 0;
    }

    .assistant-message {
        background: $success 8%;
        border: round $success;
        padding: 0 1;
        margin: 1 0;
    }

    .system-note {
        color: $text-muted;
        text-style: italic;
        padding: 0 1;
    }
    """

    # reactive so textual can observe
    message_count: reactive[int] = reactive(0)

    def __init__(self, history: ChatHistory | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._history: ChatHistory | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._assistant_widgets: dict[str, Static | Markdown] = {}  # event_id -> widget
        self._tool_cards: dict[str, ToolCard] = {}  # call_id -> card
        self._current_turn_id: str | None = None
        # Deduplicate live + replay so the same event_id is only painted once
        self._rendered_ids: set[str] = set()
        # App thread id once mounted (for low-latency same-thread rendering)
        self._ui_thread_id: int | None = None

        if history is not None:
            self.set_history(history)

    def set_history(self, history: ChatHistory) -> None:
        """Attach (or re-attach) to a history. Safe to call multiple times.

        NOTE: Replay of past events is deferred to on_mount + call_after_refresh
        to avoid "Can't mount widget(s) before Vertical() is mounted" errors.
        Live events are scheduled onto the UI thread immediately (thread-safe).
        """
        if self._unsubscribe:
            self._unsubscribe()

        self._history = history
        self._unsubscribe = history.subscribe(self._on_event)
        # Do NOT replay here (constructor time). on_mount will handle it safely.

    def _on_event(self, event: ChatEvent) -> None:
        """Callback from ChatHistory.add().

        Must be thread-safe: tool start/end often arrive from worker threads
        (CodingAgent / tool workers via emit_tool_event bridge). We schedule
        rendering onto the UI thread so ToolCards and messages appear immediately
        while the agent is still running, not only after the turn finishes.
        """
        self._schedule_render(event)

    def _schedule_render(self, event: ChatEvent) -> None:
        """Schedule _safe_render on the Textual UI thread with minimal latency.

        - Same UI thread: mount immediately (user pressed Enter → bubble shows now).
        - Other threads: non-blocking ``post_message`` (thread-safe; no deadlock risk
          with ``asyncio.to_thread`` / blocking workers).
        """
        # Same UI thread: mount immediately when possible so user messages appear
        # in the message box without waiting for the next refresh cycle.
        if self._ui_thread_id is not None and threading.get_ident() == self._ui_thread_id:
            if self.is_mounted:
                try:
                    self._safe_render(event)
                    return
                except Exception:
                    pass

        # Worker / other thread (or not yet ready for direct mount):
        # post_message is non-blocking and thread-safe in Textual.
        try:
            self.post_message(ChatHistoryMessage(event))
            return
        except Exception:
            pass

        # Last-resort fallbacks (e.g. very early lifecycle)
        def _render() -> None:
            self._safe_render(event)

        try:
            self.call_later(_render)
        except Exception:
            try:
                self.call_after_refresh(_render)
            except Exception:
                pass

    def on_chat_history_message(self, message: ChatHistoryMessage) -> None:
        """Handle thread-safe history notifications posted from any thread."""
        self._safe_render(message.event)

    def _safe_render(self, event: ChatEvent) -> None:
        """Render only when mounted; skip duplicates (live vs history replay)."""
        if not self.is_mounted:
            return
        event_id = getattr(event, "event_id", None)
        if event_id and event_id in self._rendered_ids:
            return
        if event_id:
            self._rendered_ids.add(event_id)
        self._render_event(event)

    def _render_event(self, event: ChatEvent) -> None:
        """Create appropriate visual representation and mount it."""
        focus_widget = None
        if isinstance(event, UserMessageEvent):
            focus_widget = self._add_user_message(event)
        elif isinstance(event, AssistantMessageEvent):
            focus_widget = self._add_or_update_assistant(event)
        elif isinstance(event, ToolCallEvent):
            focus_widget = self._add_tool_call(event)
        elif isinstance(event, ToolResultEvent):
            focus_widget = self._add_tool_result(event)
        elif isinstance(event, StreamTokenEvent):
            focus_widget = self._handle_stream_token(event)

        self.message_count = len(self._history.get_events()) if self._history else self.message_count + 1
        # Always pin the viewport to the newest content (user requirement).
        self._scroll_to_latest(focus_widget)

    def _scroll_to_latest(self, focus_widget: Static | Markdown | ToolCard | None = None) -> None:
        """Keep chat history pinned to the latest message / tool card.

        Uses Textual ``anchor`` so size growth (streaming tokens, tool results)
        stays visible, and ``scroll_end`` (deferred until after layout) so the
        virtual height is correct when content is newly mounted.
        """
        try:
            target = focus_widget
            if target is None:
                children = list(self.children)
                target = children[-1] if children else None
            if target is not None:
                try:
                    target.anchor(animate=False)
                except Exception:
                    pass
            # force=True ignores overflow restrictions; animate=False for snappy live updates.
            # scroll_end schedules itself after refresh when immediate=False (default),
            # which is required so max_scroll_y includes the newly mounted widget.
            self.scroll_end(animate=False, force=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Individual renderers
    # ------------------------------------------------------------------

    def _add_user_message(self, event: UserMessageEvent) -> Static:
        panel_content = Text(event.content, style="bold blue")
        widget = Static(
            Panel(panel_content, title="You", border_style="blue", padding=(0, 1)),
            classes="user-message",
        )
        self.mount(widget)
        return widget

    def _add_or_update_assistant(self, event: AssistantMessageEvent) -> Static | Markdown:
        """Create or replace the assistant message widget."""
        content = event.content or ""
        md = RichMarkdown(content) if content else Text("(thinking...)", style="dim")

        # Use Textual's Markdown widget for nice rendering when possible
        try:
            widget: Static | Markdown = Markdown(content or "*…*", classes="assistant-message")
        except Exception:
            widget = Static(Panel(md, title="Assistant", border_style="green"), classes="assistant-message")

        # store for streaming updates
        self._assistant_widgets[event.event_id] = widget

        # If we already have a streaming assistant for this turn, replace its content
        # Remove previous placeholder if present for same turn (simple heuristic)
        if event.turn_id:
            for existing_id, w in list(self._assistant_widgets.items()):
                if getattr(w, "_turn_id", None) == event.turn_id and existing_id != event.event_id:
                    try:
                        w.remove()
                    except Exception:
                        pass

        setattr(widget, "_turn_id", event.turn_id)
        self.mount(widget)
        return widget

    def _add_tool_call(self, event: ToolCallEvent) -> ToolCard:
        card = ToolCard(event)
        self._tool_cards[event.call_id] = card
        self.mount(card)
        return card

    def _add_tool_result(self, event: ToolResultEvent) -> Static | ToolCard:
        card = self._tool_cards.get(event.call_id)
        if card is not None:
            card.set_result(event)
            return card
        # Orphan result (shouldn't happen with proper pairing) — render anyway
        fallback = Static(
            f"[tool result for {event.tool_name}] {'ok' if event.success else 'error'}",
            classes="system-note",
        )
        self.mount(fallback)
        return fallback

    def _handle_stream_token(self, event: StreamTokenEvent) -> Static | Markdown | None:
        """Append token to the live assistant message widget if we can find it."""
        # Prefer explicit assistant_event_id
        target_id = event.assistant_event_id
        if target_id and target_id in self._assistant_widgets:
            w = self._assistant_widgets[target_id]
            self._append_to_assistant_widget(w, event.token)
            return w

        # Fallback: update the most recent assistant widget
        if self._assistant_widgets:
            last = list(self._assistant_widgets.values())[-1]
            self._append_to_assistant_widget(last, event.token)
            return last
        return None

    def _append_to_assistant_widget(self, widget: Static | Markdown, token: str) -> None:
        """Best-effort live update of an assistant message."""
        try:
            if isinstance(widget, Markdown):
                # Markdown widget content update
                current = getattr(widget, "update", None)
                if callable(current):
                    # Rebuild markdown with appended content
                    # Note: we rely on history to have already mutated the event.content
                    # so we can re-render from the authoritative source if needed.
                    # For pure token append we do a simple approach:
                    widget.update(getattr(widget, "markdown", "") + token)
                    return
            # Static fallback
            if isinstance(widget, Static):
                # crude but effective for many cases
                renderable = widget.renderable
                if hasattr(renderable, "renderable"):  # inside Panel?
                    inner = renderable.renderable
                    if isinstance(inner, (Text, RichMarkdown)):
                        inner.append(token)
                    widget.refresh()
                    return
                # last resort
                widget.update(str(widget.renderable or "") + token)
        except Exception:
            # If live update fails, the final AssistantMessageEvent will still render correctly
            pass

    def clear_log(self) -> None:
        """Remove all children (used for new session)."""
        for child in list(self.children):
            child.remove()
        self._assistant_widgets.clear()
        self._tool_cards.clear()
        self._rendered_ids.clear()

    def on_mount(self) -> None:
        """Defer initial replay until after this VerticalScroll (and its internal Vertical)
        has been mounted. This prevents MountError when populating from history.
        """
        self._ui_thread_id = threading.get_ident()
        if self._history:
            self.call_after_refresh(self._replay_history)

    def _replay_history(self) -> None:
        """Replay past events safely (called via call_after_refresh)."""
        if not self._history:
            return
        for ev in self._history.get_events():
            self._safe_render(ev)
        # After replaying a session, land on the latest message.
        self._scroll_to_latest()
