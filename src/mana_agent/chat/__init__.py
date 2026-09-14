"""
mana_agent.chat

Event-driven chat layer for the enhanced TUI.

This package provides the canonical event types and the subscription-based
ChatHistory that guarantees tool visibility on every turn.
"""

from .attachments import (
    AttachmentCategory,
    AttachmentStatus,
    AttachmentStore,
    AttachmentValidator,
    ChatAttachment,
    format_size,
)
from .events import (
    AssistantMessageEvent,
    StreamTokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from .history import ChatHistory, get_history, reset_global_history

__all__ = [
    "AttachmentCategory",
    "AttachmentStatus",
    "AttachmentStore",
    "AttachmentValidator",
    "ChatAttachment",
    "format_size",
    "UserMessageEvent",
    "AssistantMessageEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "StreamTokenEvent",
    "ChatHistory",
    "get_history",
    "reset_global_history",
]
