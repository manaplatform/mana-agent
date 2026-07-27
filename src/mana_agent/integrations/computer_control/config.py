"""Configuration model for the computer-control integration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from mana_agent.config.user_config import get_setting, load_user_config


DEFAULT_PERMISSION_DECISIONS: dict[str, str] = {
    "computer.apps.read": "ask",
    "computer.apps.control": "ask",
    "computer.calendar.read": "ask",
    "computer.calendar.write": "ask",
    "computer.media.read": "ask",
    "computer.media.control": "ask",
    "computer.notes.read": "ask",
    "computer.notes.write": "ask",
    "computer.browser.tabs.read": "ask",
    "computer.browser.page.read": "ask",
    "computer.browser.control": "ask",
    "computer.clipboard.read": "ask",
    "computer.clipboard.write": "ask",
    "computer.files.read": "ask",
    "computer.files.write": "ask",
    "computer.screenshot.capture": "ask",
    "computer.notifications.send": "ask",
    "computer.system.read": "ask",
    "computer.system.control": "ask",
}


class ComputerControlSettings(BaseModel):
    enabled: bool = False
    provider: str = "auto"
    allowed_clients: set[str] = Field(default_factory=lambda: {"local_cli", "tui", "dashboard"})
    allow_remote_control: bool = False
    remote_sensitive_scopes: set[str] = Field(default_factory=set)
    require_local_confirmation_for_high_risk: bool = True
    require_confirmation: bool = True
    audit_enabled: bool = True
    audit_retention_days: int = Field(default=30, ge=1, le=3650)
    timeout_seconds: float = Field(default=30, ge=1, le=300)
    allowed_paths: list[Path] = Field(default_factory=list)
    permissions: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_PERMISSION_DECISIONS))
    defaults: dict[str, str] = Field(default_factory=lambda: {
        "browser": "auto", "calendar": "auto", "music": "auto", "notes": "auto",
    })

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"auto", "macos", "windows", "linux"}:
            raise ValueError("computer-control provider must be auto, macos, windows, or linux")
        return value

    @model_validator(mode="after")
    def validate_security_boundaries(self) -> "ComputerControlSettings":
        unknown_scopes = set(self.permissions) - set(DEFAULT_PERMISSION_DECISIONS)
        if unknown_scopes:
            raise ValueError(f"unknown computer permission scopes: {', '.join(sorted(unknown_scopes))}")
        allowed_decisions = {"denied", "ask", "always"}
        invalid = {
            scope: decision
            for scope, decision in self.permissions.items()
            if decision not in allowed_decisions
        }
        if invalid:
            raise ValueError(f"invalid computer permission decisions: {invalid}")
        invalid_paths = [path for path in self.allowed_paths if not path.expanduser().is_absolute()]
        if invalid_paths:
            raise ValueError("computer-control allowed_paths must contain only absolute paths")
        unknown_remote_scopes = self.remote_sensitive_scopes - set(DEFAULT_PERMISSION_DECISIONS)
        if unknown_remote_scopes:
            raise ValueError(f"unknown remote computer scopes: {', '.join(sorted(unknown_remote_scopes))}")
        return self

    @classmethod
    def load(cls) -> "ComputerControlSettings":
        user_config = load_user_config()
        configured_table = user_config.get("computer_control")
        raw = configured_table if isinstance(configured_table, dict) else get_setting("computer_control", {})
        if not isinstance(raw, dict):
            raw = {}
        legacy_enabled = bool(get_setting("MANA_COMPUTER_CONTROL_ENABLED", False))
        payload = dict(raw)
        if not isinstance(configured_table, dict) or "enabled" not in configured_table:
            payload["enabled"] = legacy_enabled
        merged_permissions = dict(DEFAULT_PERMISSION_DECISIONS)
        if isinstance(payload.get("permissions"), dict):
            merged_permissions.update({str(key): str(value) for key, value in payload["permissions"].items()})
        payload["permissions"] = merged_permissions
        return cls.model_validate(payload)
