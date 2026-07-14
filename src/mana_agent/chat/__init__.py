"""
mana_agent.chat

Event-driven chat layer for the enhanced TUI.

This package provides the canonical event types and the subscription-based
ChatHistory that guarantees tool visibility on every turn.
"""

from .events import (
    AssistantMessageEvent,
    StreamTokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from .history import ChatHistory, get_history, reset_global_history

__all__ = [
    "UserMessageEvent",
    "AssistantMessageEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "StreamTokenEvent",
    "ChatHistory",
    "get_history",
    "reset_global_history",
]
