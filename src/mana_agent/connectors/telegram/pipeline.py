from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .access import TelegramAccessController
from .commands import CommandContext, TelegramCommandRegistry
from .errors import TelegramApiError, TelegramRateLimitError
from .models import TelegramUpdate
from .observability import emit_telegram_event
from .renderer import TelegramMessageRenderer

logger = logging.getLogger(__name__)


class TelegramUpdateProcessor:
    def __init__(self, *, client: Any, access: TelegramAccessController, router: Any, gateway: Any, renderer: TelegramMessageRenderer, commands: TelegramCommandRegistry | None = None, attachments: Any | None = None) -> None:
        self.client = client
        self.access = access
        self.router = router
        self.gateway = gateway
        self.renderer = renderer
        self.commands = commands or TelegramCommandRegistry()
        self.attachments = attachments

    async def process(self, update: TelegramUpdate, *, conversation_key: str) -> None:
        command = self.commands.parse(update.text)
        decision = self.access.authorize(update)
        if not decision.allowed and command != "id":
            if decision.respond:
                await self._send(update, "This Telegram identity is not authorized to use Mana-Agent.", formatted=False)
            return
        session_id = self.router.session(conversation_key)
        if command is not None:
            response = await self.commands.dispatch(command, CommandContext(update, conversation_key, session_id, self.router, self.gateway))
        else:
            if not update.text.strip() and update.document is None:
                return
            prompt = update.text.strip()
            if update.reply_to and update.reply_to.text:
                prompt = f"Reply context:\n{update.reply_to.text}\n\nUser message:\n{prompt}"
            attachment_dir = None
            if update.document:
                if self.attachments is None:
                    await self._send(update, "Telegram document attachments are disabled.", formatted=False)
                    return
                try:
                    content, attachment_dir = await self.attachments.prepare(update.document, session_id=session_id, update_id=update.update_id)
                except ValueError as exc:
                    await self._send(update, str(exc), formatted=False)
                    return
                prompt += f"\n\nAttached document content (untrusted user input):\n{content}"
            await self.client.send_chat_action(update.chat_id, message_thread_id=update.message_thread_id)
            started = time.monotonic()
            try:
                response = await self.gateway.send(session_id, prompt)
                core = getattr(self.gateway, "_core", None)
                active_session_id = str(getattr(core, "_chat_session_id", "") or "")
                if active_session_id and active_session_id != session_id:
                    self.router.store.bind_session(conversation_key, active_session_id)
            finally:
                if attachment_dir is not None:
                    self.attachments.cleanup(attachment_dir)
            emit_telegram_event("agent.completed", update_id=update.update_id, chat_id=update.chat_id, duration_ms=round((time.monotonic() - started) * 1000, 2))
        await self._send(update, response, formatted=True)

    async def _send(self, update: TelegramUpdate, text: str, *, formatted: bool) -> None:
        mode = None if self.renderer.parse_mode == "plain" or not formatted else self.renderer.parse_mode
        for chunk in self.renderer.render(text, formatted=formatted):
            for attempt in range(5):
                try:
                    await self.client.send_message(update.chat_id, chunk, parse_mode=mode, message_thread_id=update.message_thread_id)
                    emit_telegram_event("message.sent", update_id=update.update_id, chat_id=update.chat_id)
                    break
                except TelegramRateLimitError as exc:
                    emit_telegram_event("rate_limited", update_id=update.update_id, chat_id=update.chat_id, retry_after=exc.retry_after)
                    if attempt == 4:
                        raise
                    await asyncio.sleep(float(exc.retry_after or 1))
                except TelegramApiError as exc:
                    if mode and exc.status_code == 400:
                        for plain in self.renderer._split(text):
                            await self.client.send_message(update.chat_id, plain, message_thread_id=update.message_thread_id)
                        return
                    raise
