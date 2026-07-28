"""Secure Teach Mode configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from mana_agent.config.user_config import load_user_config
from mana_agent.workspaces.paths import mana_home


class TeachSettings(BaseModel):
    enabled: bool = True
    event_sources: set[str] = Field(
        default_factory=lambda: {"browser", "accessibility", "application", "filesystem", "keyboard", "pointer"}
    )
    desktop_capture: bool = False
    storage_path: Path = Field(default_factory=lambda: mana_home() / "teach")
    retention_days: int = Field(default=30, ge=1, le=3650)
    screenshot_policy: str = "never"
    coordinate_fallback: bool = True
    voice_enabled: bool = False
    browser_capture: bool = True
    excluded_applications: set[str] = Field(default_factory=set)
    allowed_applications: set[str] = Field(default_factory=set)
    excluded_domains: set[str] = Field(default_factory=set)
    recording_allowed_paths: list[Path] = Field(default_factory=list)
    sensitive_detection: bool = True
    automatic_verification: bool = True
    replay_retry_limit: int = Field(default=1, ge=0, le=10)
    correction_checkpoints: bool = True
    flow_cards: bool = True
    experimental_sharing: bool = False

    @classmethod
    def load(cls) -> "TeachSettings":
        raw = load_user_config().get("teach", {})
        return cls.model_validate(raw if isinstance(raw, dict) else {})
