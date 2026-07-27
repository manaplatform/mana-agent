"""Telegram adapter for the shared chat-command registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import TelegramUpdate


@dataclass(slots=True)
class CommandContext:
    update: TelegramUpdate
    conversation_key: str
    session_id: str
    router: Any
    gateway: Any


class TelegramCommandRegistry:
    """Compatibility facade; command definitions live in chat_commands."""

    def parse(self, text: str) -> str | None:
        raw = str(text or "").strip()
        if not raw.startswith("/"):
            return None
        head, separator, tail = raw.partition(" ")
        command = head.split("@", 1)[0]
        return command + (separator + tail if separator else "")

    async def dispatch(self, command: str, context: CommandContext) -> str:
        core = getattr(context.gateway, "_core", None) or context.gateway
        dispatch = getattr(core, "dispatch_command", None)
        if dispatch is None:
            return "Shared command service is unavailable. No fallback action was executed."
        result = dispatch(
            command, session_id=context.session_id, frontend="telegram",
            frontend_data={
                "user_id": context.update.sender_user_id,
                "chat_id": context.update.chat_id,
                "topic_id": context.update.message_thread_id if context.update.message_thread_id is not None else "none",
            },
        )
        if result is None:
            return "Unsupported command. Use /help to list available commands."
        new_session_id = str(result.data.get("session_id") or "")
        if new_session_id:
            context.router.store.bind_session(context.conversation_key, new_session_id)
        return result.message or ("Started a new Mana-Agent conversation." if new_session_id else "Command completed.")
