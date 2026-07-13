from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class BrowserRisk(str, Enum):
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    SENSITIVE = "sensitive"
    IRREVERSIBLE = "irreversible"


class BrowserActionDecision(BaseModel):
    session_id: str = Field(min_length=1)
    action: Literal["open", "inspect", "click", "type", "select", "scroll", "wait", "screenshot", "upload", "download", "back", "tabs", "switch_tab", "close"]
    tab_id: str | None = None
    target: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    observed_page_version: int | None = Field(default=None, ge=0)
    expected_origin: str | None = None
    risk: BrowserRisk
    confirmation_required: bool = False
    reason: str = Field(min_length=1, max_length=600)

    @model_validator(mode="after")
    def validate_confirmation(self) -> "BrowserActionDecision":
        if self.risk in {BrowserRisk.SENSITIVE, BrowserRisk.IRREVERSIBLE} and not self.confirmation_required:
            raise ValueError("sensitive and irreversible browser decisions must require confirmation")
        return self


class BrowserConfig(BaseModel):
    enabled: bool = True
    headless: bool = True
    action_timeout_ms: int = Field(default=15_000, ge=100, le=300_000)
    navigation_timeout_ms: int = Field(default=30_000, ge=100, le=300_000)
    max_download_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    allowed_upload_roots: list[str] = Field(default_factory=list)
    persistence: Literal["ephemeral", "named"] = "ephemeral"
    profile_name: str | None = None

