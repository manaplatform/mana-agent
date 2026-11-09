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
from dataclasses import replace

from .events import (
    AssistantMessageEvent,
    ActivityEvent,
    ChatEvent,
    StreamTokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)


Listener = Callable[[ChatEvent], None]


class ChatHistory:
    """
    Thread-safe conversation history with bounded activity projections and pub/sub.

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
        from .activity import project_activity, safe_text, operation_key, TERMINAL, accepts_activity_update
        from mana_agent.services.activity_context import current_activity_context
        context = current_activity_context()
        protected = context.protected_text if context else ()
        if isinstance(event, ActivityEvent):
            projected = project_activity(event.activity, protected=context.protected_text if context else ())
            if projected is None:
                return event
            event = replace(event, activity=projected)
        elif isinstance(event, ToolCallEvent):
            # Raw invocations belong at the guarded tool boundary, not chat history.
            event = replace(event, args={}, summary=safe_text(event.summary, 240, protected=protected))
        elif isinstance(event, ToolResultEvent):
            event = replace(event, result=None, summary=safe_text(event.summary, 240, protected=protected),
                            error=safe_text(event.error, protected=protected) if event.error else None)
        with self._lock:
            if isinstance(event, UserMessageEvent) and event.is_steering and event.request_id:
                previous = next((row for row in self._events if isinstance(row, UserMessageEvent)
                                 and row.is_steering and row.request_id == event.request_id), None)
                if previous is not None:
                    return previous
            if isinstance(event, ActivityEvent):
                if event.activity["type"].startswith("steering."):
                    metadata = event.activity.get("metadata") or {}
                    request_id = str(metadata.get("request_id") or event.activity.get("request_id") or "")
                    status = event.activity["type"].split(".")[-1]
                    recorded = bool(event.activity.get("sequence"))
                    for message in self._events:
                        if isinstance(message, UserMessageEvent) and message.is_steering and message.request_id == request_id:
                            if not message.steering_recorded or message.steering_status not in {"applied", "rejected", "superseded"}:
                                message.steering_status = status
                                message.steering_recorded = recorded
                key = operation_key(event.activity)
                for index in range(len(self._events) - 1, -1, -1):
                    previous = self._events[index]
                    if isinstance(previous, ActivityEvent) and previous.turn_id == event.turn_id and operation_key(previous.activity) == key:
                        if not accepts_activity_update(previous.activity, event.activity):
                            return previous
                        event = replace(event, activity={**event.activity, "started_at": previous.activity.get("started_at") or event.activity.get("started_at")})
                        self._events[index] = event
                        break
                else:
                    self._events.append(event)
                completed = [row for row in self._events if isinstance(row, ActivityEvent) and row.turn_id == event.turn_id
                             and str(row.activity.get("status")) in TERMINAL]
                discarded = {id(row) for row in completed[:-200]}
                if discarded:
                    self._events[:] = [row for row in self._events if id(row) not in discarded]
            else:
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

            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                # Presentation cannot change execution outcomes.
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
