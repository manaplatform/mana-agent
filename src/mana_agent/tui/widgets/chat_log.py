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

from rich.console import Console, RenderableType
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.text import Text
from textual.containers import Vertical, VerticalScroll
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
        self._unsubscribe: callable | None = None
        self._assistant_widgets: dict[str, Static | Markdown] = {}  # event_id -> widget
        self._tool_cards: dict[str, ToolCard] = {}  # call_id -> card
        self._current_turn_id: str | None = None

        if history is not None:
            self.set_history(history)

    def set_history(self, history: ChatHistory) -> None:
        """Attach (or re-attach) to a history. Safe to call multiple times.

        NOTE: Replay of past events is deferred to on_mount + call_after_refresh
        to avoid "Can't mount widget(s) before Vertical() is mounted" errors.
        Live events use call_after_refresh in _on_event.
        """
        if self._unsubscribe:
            self._unsubscribe()

        self._history = history
        self._unsubscribe = history.subscribe(self._on_event)
        # Do NOT replay here (constructor time). on_mount will handle it safely.

    def _on_event(self, event: ChatEvent) -> None:
        """Callback from ChatHistory. Always runs on add()."""
        # Schedule render on the Textual thread
        self.call_after_refresh(self._render_event, event)

    def _render_event(self, event: ChatEvent) -> None:
        """Create appropriate visual representation and mount it."""
        if isinstance(event, UserMessageEvent):
            self._add_user_message(event)
        elif isinstance(event, AssistantMessageEvent):
            self._add_or_update_assistant(event)
        elif isinstance(event, ToolCallEvent):
            self._add_tool_call(event)
        elif isinstance(event, ToolResultEvent):
            self._add_tool_result(event)
        elif isinstance(event, StreamTokenEvent):
            self._handle_stream_token(event)

        self.message_count = len(self._history.get_events()) if self._history else self.message_count + 1
        self.scroll_end(animate=False)

    # ------------------------------------------------------------------
    # Individual renderers
    # ------------------------------------------------------------------

    def _add_user_message(self, event: UserMessageEvent) -> None:
        panel_content = Text(event.content, style="bold blue")
        widget = Static(
            Panel(panel_content, title="You", border_style="blue", padding=(0, 1)),
            classes="user-message",
        )
        self.mount(widget)

    def _add_or_update_assistant(self, event: AssistantMessageEvent) -> None:
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

    def _add_tool_call(self, event: ToolCallEvent) -> None:
        card = ToolCard(event)
        self._tool_cards[event.call_id] = card
        self.mount(card)

    def _add_tool_result(self, event: ToolResultEvent) -> None:
        card = self._tool_cards.get(event.call_id)
        if card is not None:
            card.set_result(event)
        else:
            # Orphan result (shouldn't happen with proper pairing) — render anyway
            fallback = Static(
                f"[tool result for {event.tool_name}] {'ok' if event.success else 'error'}",
                classes="system-note",
            )
            self.mount(fallback)

    def _handle_stream_token(self, event: StreamTokenEvent) -> None:
        """Append token to the live assistant message widget if we can find it."""
        # Prefer explicit assistant_event_id
        target_id = event.assistant_event_id
        if target_id and target_id in self._assistant_widgets:
            w = self._assistant_widgets[target_id]
            self._append_to_assistant_widget(w, event.token)
            return

        # Fallback: update the most recent assistant widget
        if self._assistant_widgets:
            last = list(self._assistant_widgets.values())[-1]
            self._append_to_assistant_widget(last, event.token)

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

    def on_mount(self) -> None:
        """Defer initial replay until after this VerticalScroll (and its internal Vertical)
        has been mounted. This prevents MountError when populating from history.
        """
        if self._history:
            self.call_after_refresh(self._replay_history)

    def _replay_history(self) -> None:
        """Replay past events safely (called via call_after_refresh)."""
        if not self._history:
            return
        for ev in self._history.get_events():
            self._render_event(ev)
