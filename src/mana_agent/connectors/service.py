from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from mana_agent.background.manager import BackgroundProcessManager
from mana_agent.chat_commands.models import CommandContext, CommandResult
from mana_agent.connectors.telegram.client import TelegramBotClient
from mana_agent.connectors.telegram.config import TelegramConfig, load_telegram_config, save_telegram_config


class TelegramConnectRequest(BaseModel):
    transport: str = "auto"
    repository: str
    workspace_id: str = ""
    allowed_users: list[int] = Field(default_factory=list)
    allowed_chats: list[int] = Field(default_factory=list)
    webhook_url: str = ""
    startup_policy: str = "automatic"
    restart_policy: str = "on-failure"
    secret_source: str = "keyring"
    secret_env: str = "TELEGRAM_BOT_TOKEN"


class ConnectorService:
    def __init__(self, processes: BackgroundProcessManager | None = None) -> None:
        self.processes = processes or BackgroundProcessManager()

    def list(self) -> list[dict[str, Any]]:
        config = load_telegram_config()
        running = self._telegram_process()
        return [{
            "name": "telegram", "configured": config.enabled,
            "transport": config.effective_transport, "state": running.state if running else "stopped",
            "process_id": running.process_id if running else None,
        }]

    def connect_telegram(self, request: TelegramConnectRequest, *, token: str) -> CommandResult:
        clean_token = str(token or "").strip()
        if not clean_token:
            raise ValueError("Telegram bot token is required")
        config = TelegramConfig(
            enabled=True, transport=request.transport, default_repository=request.repository,
            allowed_repository_roots=[request.repository], allowed_users=request.allowed_users,
            allowed_chats=request.allowed_chats, bot_token_env=request.secret_env,
            bot_token_secret_ref="telegram.bot-token" if request.secret_source == "keyring" else "",
            webhook={"public_url": request.webhook_url},
        )
        identity = asyncio.run(self._identity(config, clean_token))
        if request.secret_source == "keyring":
            try:
                import keyring

                keyring.set_password("mana-agent.connector", "telegram.bot-token", clean_token)
            except Exception as exc:
                raise RuntimeError("Secure keyring storage is unavailable; select the environment-variable secret source.") from exc
        save_telegram_config(config)
        process = self.start_telegram(restart_policy=request.restart_policy)
        return CommandResult(
            status="success", message=f"Connected Telegram bot @{identity.username or identity.id}.",
            data={"connector": "telegram", "transport": config.effective_transport},
            background_process_id=process.process_id,
        )

    def start_telegram(self, *, restart_policy: str = "on-failure"):
        config = load_telegram_config()
        config.validate_runtime()
        return self.processes.start(
            "connector.telegram", process_type="connector", singleton_key="connector.telegram",
            repository_id="", workspace_id="", ownership="global", restart_policy=restart_policy,
        )

    def stop_telegram(self):
        row = self._telegram_process()
        if row is None:
            raise RuntimeError("Telegram connector is not running")
        return self.processes.stop(row.process_id)

    def command(self, context: CommandContext, args: list[str]) -> CommandResult:
        if not args or args[0].lower() == "list":
            rows = self.list()
            return CommandResult(status="success", message="\n".join(f"{r['name']}: {r['state']}" for r in rows), data={"connectors": rows})
        if args[0].lower() == "disconnect":
            if len(args) != 2 or args[1].lower() != "telegram":
                raise ValueError("Usage: /disconnect telegram")
            row = self._telegram_process()
            if row:
                self.processes.stop(row.process_id)
            config = load_telegram_config()
            config.enabled = False
            save_telegram_config(config)
            if config.bot_token_secret_ref:
                try:
                    import keyring
                    keyring.delete_password("mana-agent.connector", config.bot_token_secret_ref)
                except Exception:
                    pass
            return CommandResult(status="success", message="Disconnected Telegram.")
        if args[0].lower() != "telegram":
            raise ValueError(f"Unsupported connector: {args[0]}")
        action = args[1].lower() if len(args) > 1 else "setup"
        if action == "setup":
            return CommandResult(
                status="input_required", message="Enter Telegram connection settings.",
                next_prompt={"kind": "connector.telegram.setup", "secret_fields": ["token"], "fields": ["transport", "repository", "workspace_id", "allowed_users", "allowed_chats", "webhook_url", "startup_policy", "restart_policy", "secret_source"]},
            )
        if action == "status":
            row = next(item for item in self.list() if item["name"] == "telegram")
            return CommandResult(status="success", message=f"Telegram: {row['state']} ({row['transport']}).", data=row)
        if action == "test":
            config = load_telegram_config()
            config.validate_runtime()
            identity = asyncio.run(self._identity(config, config.bot_token))
            return CommandResult(status="success", message=f"Telegram credentials valid for @{identity.username or identity.id}.")
        if action == "start":
            row = self.start_telegram()
        elif action == "stop":
            row = self.stop_telegram()
        elif action == "restart":
            current = self._telegram_process()
            row = self.processes.restart(current.process_id) if current else self.start_telegram()
        elif action == "logs":
            current = self._telegram_process()
            return CommandResult(status="success", message=self.processes.logs(current.process_id) if current else "No Telegram process logs.")
        else:
            raise ValueError("Usage: /connect telegram [status|test|start|stop|restart|logs]")
        return CommandResult(status="success", message=f"Telegram connector: {row.state}.", data={"process": row.model_dump(mode="json")}, background_process_id=row.process_id)

    async def _identity(self, config: TelegramConfig, token: str):
        client = TelegramBotClient(token, timeout_seconds=config.request_timeout_seconds)
        try:
            return await client.get_me()
        finally:
            await client.close()

    def _telegram_process(self):
        self.processes.recover_stale()
        return next((row for row in self.processes.list() if row.singleton_key == "connector.telegram" and row.state in {"starting", "running"}), None)
