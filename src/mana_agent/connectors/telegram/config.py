from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from mana_agent.config.user_config import load_user_config, save_user_config
from mana_agent.workspaces.paths import mana_home

from .errors import TelegramConfigurationError


class TelegramPollingConfig(BaseModel):
    timeout_seconds: int = Field(default=30, ge=1, le=60)
    drop_pending_updates: bool = False
    reconnect_max_seconds: int = Field(default=60, ge=1, le=600)


class TelegramWebhookConfig(BaseModel):
    public_url: str = ""
    path: str = "/integrations/telegram/webhook"
    secret_env: str = "TELEGRAM_WEBHOOK_SECRET"
    listen_host: str = "127.0.0.1"
    listen_port: int = Field(default=8787, ge=1, le=65535)
    drop_pending_updates: bool = False
    max_request_bytes: int = Field(default=1_048_576, ge=1024, le=10_485_760)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value.startswith("/") or ".." in value:
            raise ValueError("webhook path must be an absolute URL path without traversal")
        return value.rstrip("/") or "/"


class TelegramQueueConfig(BaseModel):
    backend: Literal["local"] = "local"
    max_attempts: int = Field(default=5, ge=1, le=100)
    retry_delay_seconds: int = Field(default=2, ge=1, le=3600)
    concurrency: int = Field(default=4, ge=1, le=64)
    lease_seconds: int = Field(default=300, ge=10, le=86_400)
    database_path: str = ""


class TelegramAttachmentConfig(BaseModel):
    enabled: bool = False
    max_bytes: int = Field(default=10 * 1024 * 1024, ge=1, le=50 * 1024 * 1024)
    allowed_mime_types: list[str] = Field(default_factory=lambda: ["text/plain", "application/pdf", "text/csv"])


class TelegramConfig(BaseModel):
    enabled: bool = False
    transport: Literal["auto", "polling", "webhook"] = "polling"
    bot_token_env: str = "TELEGRAM_BOT_TOKEN"
    bot_token_secret_ref: str = ""
    allowed_users: list[int] = Field(default_factory=list)
    allowed_chats: list[int] = Field(default_factory=list)
    admin_users: list[int] = Field(default_factory=list)
    open_access: bool = False
    private_only: bool = False
    groups_enabled: bool = False
    group_activation: Literal["mention", "reply", "command", "always"] = "mention"
    always_active_chats: list[int] = Field(default_factory=list)
    parse_mode: Literal["MarkdownV2", "HTML", "plain"] = "MarkdownV2"
    request_timeout_seconds: int = Field(default=30, ge=1, le=300)
    max_message_length: int = Field(default=4096, ge=256, le=4096)
    default_repository: str = ""
    allowed_repository_roots: list[str] = Field(default_factory=list)
    polling: TelegramPollingConfig = Field(default_factory=TelegramPollingConfig)
    webhook: TelegramWebhookConfig = Field(default_factory=TelegramWebhookConfig)
    queue: TelegramQueueConfig = Field(default_factory=TelegramQueueConfig)
    attachments: TelegramAttachmentConfig = Field(default_factory=TelegramAttachmentConfig)

    @field_validator("allowed_users", "allowed_chats", "admin_users", "always_active_chats")
    @classmethod
    def numeric_ids(cls, values: list[int]) -> list[int]:
        return list(dict.fromkeys(int(value) for value in values))

    @model_validator(mode="after")
    def validate_repository_policy(self) -> "TelegramConfig":
        roots = [Path(item).expanduser().resolve() for item in self.allowed_repository_roots]
        if self.default_repository:
            default = Path(self.default_repository).expanduser().resolve()
            if not default.is_dir():
                raise ValueError("default_repository must be an existing directory")
            if roots and not any(default == root or root in default.parents for root in roots):
                raise ValueError("default_repository must be within an allowed repository root")
            self.default_repository = str(default)
        self.allowed_repository_roots = [str(root) for root in roots]
        return self

    @property
    def bot_token(self) -> str:
        value = str(os.getenv(self.bot_token_env, "")).strip()
        if value or not self.bot_token_secret_ref:
            return value
        try:
            import keyring

            return str(keyring.get_password("mana-agent.connector", self.bot_token_secret_ref) or "").strip()
        except Exception:
            return ""

    @property
    def webhook_secret(self) -> str:
        return str(os.getenv(self.webhook.secret_env, "")).strip()

    @property
    def effective_transport(self) -> Literal["polling", "webhook"]:
        if self.transport == "auto":
            return "webhook" if self.webhook.public_url.strip() else "polling"
        return self.transport

    @property
    def database_path(self) -> Path:
        configured = self.queue.database_path.strip()
        return Path(configured).expanduser().resolve() if configured else mana_home() / "connectors" / "telegram" / "queue.sqlite3"

    def validate_runtime(self) -> None:
        if not self.enabled:
            raise TelegramConfigurationError("Telegram connector is disabled.")
        if not self.bot_token:
            raise TelegramConfigurationError(f"Telegram bot token is missing from environment variable {self.bot_token_env}.")
        if self.effective_transport == "webhook":
            parsed = urlparse(self.webhook.public_url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise TelegramConfigurationError("Telegram webhook public URL must be a valid HTTPS URL.")
            if len(self.webhook_secret) < 32:
                raise TelegramConfigurationError(f"Telegram webhook secret in {self.webhook.secret_env} must be at least 32 characters.")


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_telegram_config(values: dict[str, Any] | None = None) -> TelegramConfig:
    raw = values if values is not None else load_user_config().get("telegram", {})
    data = dict(raw) if isinstance(raw, dict) else {}
    env: dict[str, Any] = {}
    if "MANA_TELEGRAM_ENABLED" in os.environ:
        env["enabled"] = os.environ["MANA_TELEGRAM_ENABLED"].strip().lower() in {"1", "true", "yes", "on"}
    if os.getenv("MANA_TELEGRAM_TRANSPORT"):
        env["transport"] = os.environ["MANA_TELEGRAM_TRANSPORT"].strip().lower()
    if os.getenv("TELEGRAM_WEBHOOK_URL"):
        env["webhook"] = {"public_url": os.environ["TELEGRAM_WEBHOOK_URL"].strip()}
    try:
        return TelegramConfig.model_validate(_deep_merge(data, env))
    except ValueError as exc:
        raise TelegramConfigurationError(f"Invalid Telegram configuration: {exc}") from exc


def save_telegram_config(config: TelegramConfig) -> None:
    """Persist connector settings, storing only secret environment variable names."""
    save_user_config({"telegram": config.model_dump(mode="json")}, merge=True)
