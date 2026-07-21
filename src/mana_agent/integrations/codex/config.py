"""Configuration for the managed Codex app-server process."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CodexSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    codex_bin: str = "codex"
    max_workers: int = Field(default=2, ge=1, le=16)
    stream_events: bool = True
    # Codex normally works directly in the repository selected by the user.
    # Isolated worktrees remain available for workflows that require them.
    worktree_isolation: bool = False
    task_timeout_seconds: int = Field(default=1800, ge=1)
    model: str | None = None
    allow_network: bool = False
    approval_policy: str = "never"

    @classmethod
    def from_mana_settings(cls, settings: object) -> "CodexSettings":
        return cls(
            enabled=bool(getattr(settings, "mana_codex_enabled", False)),
            codex_bin=str(getattr(settings, "mana_codex_bin", "codex") or "codex"),
            max_workers=int(getattr(settings, "mana_codex_max_workers", 2) or 2),
            stream_events=bool(getattr(settings, "mana_codex_stream_events", True)),
            worktree_isolation=bool(getattr(settings, "mana_codex_worktree_isolation", False)),
            task_timeout_seconds=int(getattr(settings, "mana_codex_task_timeout_seconds", 1800) or 1800),
            model=str(getattr(settings, "mana_codex_model", "") or "").strip() or None,
            allow_network=bool(getattr(settings, "mana_codex_allow_network", False)),
        )

    @field_validator("approval_policy")
    @classmethod
    def _approval_policy(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if cleaned not in {"never", "untrusted", "on-request"}:
            raise ValueError("approval_policy must be never, untrusted, or on-request")
        return cleaned


__all__ = ["CodexSettings"]
