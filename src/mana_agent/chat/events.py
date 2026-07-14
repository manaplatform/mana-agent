"""
mana_agent.chat.events

Clean, typed event dataclasses for the enhanced Chat TUI.

These events are the contract between the agent runtime and the TUI.

Existing mana-agent code (chat_cli, tool runners, agents) should
emit these instead of (or in addition to) direct prints / rich renders:

    from mana_agent.chat.history import get_history
    from mana_agent.chat.events import ToolCallEvent, ToolResultEvent

    history = get_history()
    history.add(ToolCallEvent(
        tool_name="read_file",
        args={"path": "src/foo.py"},
        call_id="call-123",
    ))

    # after tool execution
    history.add(ToolResultEvent(
        call_id="call-123",
        tool_name="read_file",
        success=True,
        result={"content": "file text here...", "truncated": False},
    ))

This subscription-based approach (see history.py) guarantees that
tool events appear for *every* message/turn, fixing the previous bug
where only the first message showed tool calls/results.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    """ISO-8601 UTC timestamp with microseconds."""
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@dataclass(slots=True)
class UserMessageEvent:
    """A message from the human user."""
    content: str
    timestamp: str = field(default_factory=_utc_now)
    event_id: str = field(default_factory=lambda: _new_id("user"))
    turn_id: str = ""

    def __post_init__(self) -> None:
        if not self.turn_id:
            self.turn_id = _new_id("turn")


@dataclass(slots=True)
class AssistantMessageEvent:
    """Final (or incrementally built) assistant response."""
    content: str
    timestamp: str = field(default_factory=_utc_now)
    event_id: str = field(default_factory=lambda: _new_id("asst"))
    turn_id: str = ""
    # When streaming, this may be appended to over time via StreamTokenEvent
    is_streaming: bool = False

    def __post_init__(self) -> None:
        if not self.turn_id:
            self.turn_id = _new_id("turn")


@dataclass(slots=True)
class ToolCallEvent:
    """Agent decided to invoke a tool. Always paired with a later ToolResultEvent."""
    tool_name: str
    args: dict[str, Any] | str | None = None
    call_id: str = field(default_factory=lambda: _new_id("tool"))
    timestamp: str = field(default_factory=_utc_now)
    event_id: str = field(default_factory=lambda: _new_id("tc"))
    turn_id: str = ""
    # Optional human-friendly summary for UI
    summary: str = ""

    def __post_init__(self) -> None:
        if not self.turn_id:
            self.turn_id = _new_id("turn")
        if self.args is None:
            self.args = {}
        if isinstance(self.args, str):
            # keep raw for display; widgets can try to pretty it
            pass


@dataclass(slots=True)
class ToolResultEvent:
    """Result (success or failure) of a previously emitted ToolCallEvent."""
    call_id: str
    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    timestamp: str = field(default_factory=_utc_now)
    event_id: str = field(default_factory=lambda: _new_id("tr"))
    turn_id: str = ""
    duration_ms: int | None = None
    # Compact summary for header display
    summary: str = ""

    def __post_init__(self) -> None:
        if not self.turn_id:
            self.turn_id = _new_id("turn")


@dataclass(slots=True)
class StreamTokenEvent:
    """Incremental token from the LLM for the current assistant message."""
    token: str
    timestamp: str = field(default_factory=_utc_now)
    event_id: str = field(default_factory=lambda: _new_id("tok"))
    turn_id: str = ""
    # Associates this token with a specific assistant message if desired
    assistant_event_id: str | None = None


# Union type for convenience (runtime checks use isinstance)
ChatEvent = (
    UserMessageEvent
    | AssistantMessageEvent
    | ToolCallEvent
    | ToolResultEvent
    | StreamTokenEvent
)

__all__ = [
    "UserMessageEvent",
    "AssistantMessageEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "StreamTokenEvent",
    "ChatEvent",
    "_utc_now",
    "_new_id",
]
