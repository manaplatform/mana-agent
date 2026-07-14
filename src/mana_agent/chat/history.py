"""
mana_agent.chat.history

ChatHistory: the single source of truth for all chat events.

Key design for bug fix:
- Tool events (and all events) are appended to one persistent history.
- The TUI subscribes once via .subscribe(listener).
- Every .add(event) immediately notifies all listeners.
- No per-turn "clear" of visible tool state. Subsequent user messages
  continue to receive and display their own ToolCall/ToolResult events.

Usage in agent code (instead of print/rich direct output):

    from mana_agent.chat.history import get_history
    from mana_agent.chat.events import ToolCallEvent, ToolResultEvent, AssistantMessageEvent

    history = get_history()

    # when deciding on a tool
    call = ToolCallEvent(tool_name=..., args=..., call_id=call_id)
    history.add(call)

    # after executing
    history.add(ToolResultEvent(call_id=call_id, tool_name=..., success=True, result=...))

    # final answer
    history.add(AssistantMessageEvent(content=full_text))

The subscription model guarantees the ChatLog widget (or any listener)
sees every event on every turn, solving "tool events disappear after first message".
"""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any

from .events import (
    AssistantMessageEvent,
    ChatEvent,
    StreamTokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)


Listener = Callable[[ChatEvent], None]


class ChatHistory:
    """
    Thread-safe(ish) append-only history with pub/sub.

    Listeners are called synchronously on .add() in the order they subscribed.
    Designed to be used from asyncio event loop (Textual) or sync agent code.
    """

    def __init__(self) -> None:
        self._events: list[ChatEvent] = []
        self._listeners: list[Listener] = []
        self._lock = RLock()
        # For streaming: track the current open assistant message (if any)
        # so StreamTokenEvents can be accumulated without external state.
        self._current_assistant: AssistantMessageEvent | None = None

    def add(self, event: ChatEvent) -> ChatEvent:
        """
        Append event and notify every subscriber.

        Returns the event (for chaining).
        """
        with self._lock:
            self._events.append(event)

            # Streaming assistant accumulation helper
            if isinstance(event, AssistantMessageEvent):
                if event.is_streaming:
                    self._current_assistant = event
                else:
                    self._current_assistant = None
            elif isinstance(event, StreamTokenEvent):
                if self._current_assistant is not None and (
                    event.assistant_event_id is None
                    or event.assistant_event_id == self._current_assistant.event_id
                ):
                    # Mutate the live content so late subscribers / re-renders see full text
                    self._current_assistant.content += event.token

            for listener in list(self._listeners):
                try:
                    listener(event)
                except Exception:
                    # Never let a bad listener kill the chat loop
                    # In production you would log here.
                    pass

        return event

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        """
        Subscribe to every future event.

        Returns an unsubscribe callable.
        The listener will be called for all *subsequent* .add() calls.
        """
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

        def unsubscribe() -> None:
            self.unsubscribe(listener)

        return unsubscribe

    def unsubscribe(self, listener: Listener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def get_events(self, turn_id: str | None = None) -> list[ChatEvent]:
        """Return a snapshot of events. Optionally filtered to a turn."""
        with self._lock:
            if turn_id is None:
                return list(self._events)
            return [e for e in self._events if getattr(e, "turn_id", None) == turn_id]

    def clear(self) -> None:
        """Primarily for tests / fresh sessions. Subscribers stay attached."""
        with self._lock:
            self._events.clear()
            self._current_assistant = None

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def __iter__(self):
        with self._lock:
            return iter(list(self._events))


# ---------------------------------------------------------------------------
# Optional module-level singleton for simple usage
# (agent code can use get_history() without plumbing)
# ---------------------------------------------------------------------------

_global_history: ChatHistory | None = None


def get_history() -> ChatHistory:
    """Return the process-global ChatHistory singleton."""
    global _global_history
    if _global_history is None:
        _global_history = ChatHistory()
    return _global_history


def reset_global_history() -> ChatHistory:
    """Replace the global history (mainly for tests)."""
    global _global_history
    _global_history = ChatHistory()
    return _global_history


__all__ = [
    "ChatHistory",
    "get_history",
    "reset_global_history",
    "Listener",
]
