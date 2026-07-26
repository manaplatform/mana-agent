from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
from typing import Any

from mana_agent.compat import process_exists

from .client import TelegramBotClient
from .config import TelegramConfig
from .store import TelegramUpdateStore


class TelegramAdminService:
    """Dashboard-safe connector operations that never return secret values."""

    def __init__(self, config: TelegramConfig) -> None:
        self.config = config
        self.pid_path = config.database_path.parent / "connector.pid"

    def status(self) -> dict[str, Any]:
        pid = self._pid()
        running = pid is not None and process_exists(pid)
        store = TelegramUpdateStore(self.config.database_path)
        return {
            "enabled": self.config.enabled,
            "running": running,
            "configured_transport": self.config.transport,
            "effective_transport": self.config.effective_transport,
            "webhook_url": self.config.webhook.public_url,
            "webhook_path": self.config.webhook.path,
            "token_configured": bool(self.config.bot_token),
            "allowed_users": self.config.allowed_users,
            "allowed_chats": self.config.allowed_chats,
            "queue": store.stats(),
            "last_completed_update": store.latest_completed_update(),
            "last_error": store.last_error(),
        }

    def start(self) -> int:
        self.config.validate_runtime()
        if self.status()["running"]:
            raise RuntimeError("Telegram connector is already running.")
        executable = shutil.which("mana-agent")
        if not executable:
            raise RuntimeError("The mana-agent executable is not available to the dashboard process.")
        log_path = self.config.database_path.parent / "connector.log"
        log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with log_path.open("ab") as output:
            process = subprocess.Popen(
                [executable, "connector", "telegram", "start"],
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
            )
        return int(process.pid)

    def stop(self) -> None:
        pid = self._pid()
        if pid is None:
            raise RuntimeError("No running Telegram connector was recorded.")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError as exc:
            self.pid_path.unlink(missing_ok=True)
            raise RuntimeError("The recorded Telegram connector is not running.") from exc

    def test(self) -> dict[str, Any]:
        self.config.validate_runtime()
        async def check() -> dict[str, Any]:
            client = TelegramBotClient(self.config.bot_token, timeout_seconds=self.config.request_timeout_seconds)
            try:
                identity = await client.get_me()
                webhook = await client.get_webhook_info() if self.config.effective_transport == "webhook" else None
                return {
                    "bot": {"id": identity.id, "username": identity.username},
                    "effective_transport": self.config.effective_transport,
                    "webhook": ({
                        "url": str(webhook.get("url") or ""),
                        "pending_update_count": int(webhook.get("pending_update_count") or 0),
                        "last_error_message": str(webhook.get("last_error_message") or "")[:300],
                    } if webhook else None),
                }
            finally:
                await client.close()
        return asyncio.run(check())

    def set_webhook(self) -> None:
        self.config.validate_runtime()
        async def register() -> None:
            from urllib.parse import urljoin
            client = TelegramBotClient(self.config.bot_token, timeout_seconds=self.config.request_timeout_seconds)
            try:
                url = urljoin(self.config.webhook.public_url.rstrip("/") + "/", self.config.webhook.path.lstrip("/"))
                await client.set_webhook(url, self.config.webhook_secret, drop_pending_updates=self.config.webhook.drop_pending_updates)
            finally:
                await client.close()
        asyncio.run(register())

    def delete_webhook(self) -> None:
        async def remove() -> None:
            client = TelegramBotClient(self.config.bot_token, timeout_seconds=self.config.request_timeout_seconds)
            try:
                await client.delete_webhook(drop_pending_updates=False)
            finally:
                await client.close()
        asyncio.run(remove())

    def _pid(self) -> int | None:
        try:
            return int(self.pid_path.read_text(encoding="ascii").strip())
        except (FileNotFoundError, ValueError, OSError):
            return None
