from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from .access import TelegramAccessController
from .attachments import TelegramAttachmentManager
from .chat import ManaChatGateway, TelegramConversationRouter
from .client import TelegramBotClient
from .commands import TelegramCommandRegistry
from .config import TelegramConfig
from .errors import TelegramConfigurationError
from .normalizer import TelegramUpdateNormalizer
from .observability import emit_telegram_event
from .pipeline import TelegramUpdateProcessor
from .queue import TelegramTaskQueue
from .renderer import TelegramMessageRenderer
from .store import TelegramUpdateStore
from .transports.polling import TelegramPollingTransport
from .transports.webhook import TelegramWebhookReceiver

logger = logging.getLogger(__name__)


class TelegramConnector:
    def __init__(self, config: TelegramConfig, *, client: TelegramBotClient | None = None, gateway: Any | None = None) -> None:
        self.config = config
        self.client = client
        self.store = TelegramUpdateStore(config.database_path)
        if gateway is None:
            if not config.default_repository:
                raise TelegramConfigurationError("Telegram default_repository is required to execute Mana-Agent chat tasks.")
            gateway = ManaChatGateway(Path(config.default_repository))
        else:
            # If a central gateway (AgentChatGateway) was supplied from outside
            # (api, higher layer, or tests), wrap it so Telegram goes through
            # the gateway to the agents.
            if not isinstance(gateway, ManaChatGateway):
                repo_for_mana = Path(config.default_repository) if config.default_repository else "."
                gateway = ManaChatGateway(repo_for_mana, core_gateway=gateway)
        self.gateway = gateway
        self.identity: Any = None
        self.router: TelegramConversationRouter | None = None
        self.processor: TelegramUpdateProcessor | None = None
        self.task_queue: TelegramTaskQueue | None = None
        self.transport: TelegramPollingTransport | None = None
        self._transport_task: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        self.config.validate_runtime()
        if self.client is None:
            self.client = TelegramBotClient(self.config.bot_token, timeout_seconds=self.config.request_timeout_seconds)
        self.identity = await self.client.get_me()
        self.router = TelegramConversationRouter(self.store, self.gateway)
        access = TelegramAccessController(self.config, bot_id=self.identity.id, bot_username=self.identity.username)
        renderer = TelegramMessageRenderer(parse_mode=self.config.parse_mode, max_length=self.config.max_message_length)
        self.processor = TelegramUpdateProcessor(
            client=self.client, access=access, router=self.router, gateway=self.gateway,
            renderer=renderer, commands=TelegramCommandRegistry(),
            attachments=TelegramAttachmentManager(self.client, self.config.attachments),
        )
        queue = self.config.queue
        self.task_queue = TelegramTaskQueue(
            self.store, self.processor, concurrency=queue.concurrency, lease_seconds=queue.lease_seconds,
            max_attempts=queue.max_attempts, retry_delay_seconds=queue.retry_delay_seconds,
        )

    async def start(self) -> None:
        if self.task_queue is None:
            await self.initialize()
        assert self.task_queue is not None
        await self.task_queue.start()
        if self.config.effective_transport == "polling":
            self.transport = TelegramPollingTransport(
                client=self.client, store=self.store, task_queue=self.task_queue,
                normalizer=TelegramUpdateNormalizer(), bot_id=self.identity.id,
                token=self.config.bot_token, config=self.config.polling,
            )
            self._transport_task = asyncio.create_task(self.transport.run(), name="telegram-polling")
        else:
            await self.register_webhook()
        emit_telegram_event("connector.started", effective_transport=self.config.effective_transport, bot_id=self.identity.id)

    def webhook_receiver(self) -> TelegramWebhookReceiver:
        if self.task_queue is None or self.identity is None:
            raise RuntimeError("Telegram connector must be initialized before creating its webhook receiver.")
        return TelegramWebhookReceiver(
            secret=self.config.webhook_secret, store=self.store, task_queue=self.task_queue,
            normalizer=TelegramUpdateNormalizer(), bot_id=self.identity.id,
            max_request_bytes=self.config.webhook.max_request_bytes,
        )

    async def register_webhook(self) -> bool:
        base = self.config.webhook.public_url.rstrip("/") + "/"
        url = urljoin(base, self.config.webhook.path.lstrip("/"))
        return await self.client.set_webhook(url, self.config.webhook_secret, drop_pending_updates=self.config.webhook.drop_pending_updates)

    async def remove_webhook(self) -> bool:
        return await self.client.delete_webhook(drop_pending_updates=False)

    async def stop(self, *, remove_webhook: bool = False) -> None:
        if self.transport is not None:
            await self.transport.stop()
        if self._transport_task is not None:
            try:
                await asyncio.wait_for(self._transport_task, timeout=self.config.request_timeout_seconds + 2)
            except asyncio.TimeoutError:
                self._transport_task.cancel()
                await asyncio.gather(self._transport_task, return_exceptions=True)
        if remove_webhook and self.config.effective_transport == "webhook":
            await self.remove_webhook()
        if self.task_queue is not None:
            await self.task_queue.stop()
        if self.client is not None:
            await self.client.close()
        emit_telegram_event("connector.stopped", effective_transport=self.config.effective_transport, bot_id=self.identity.id if self.identity else None)

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "configured_transport": self.config.transport,
            "effective_transport": self.config.effective_transport,
            "bot": ({"id": self.identity.id, "username": self.identity.username} if self.identity else None),
            "running": bool(self.task_queue is not None and (self._transport_task is None or not self._transport_task.done())),
            "queue": self.store.stats(),
            "last_completed_update": self.store.latest_completed_update(),
            "last_error": self.store.last_error(),
        }
