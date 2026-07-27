"""Configuration for the managed Codex app-server process."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mana_agent.config.inference_provider import resolve_inference_connection


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
    provider: str = ""
    provider_display_name: str = ""
    api_key: str = Field(default="", repr=False, exclude=True)
    base_url: str = ""
    http_headers: dict[str, str] = Field(default_factory=dict)
    env_http_headers: dict[str, str] = Field(default_factory=dict)
    query_params: dict[str, str] = Field(default_factory=dict)
    supports_responses_api: bool = False
    request_max_retries: int = Field(default=4, ge=0)
    stream_max_retries: int = Field(default=5, ge=0)
    stream_idle_timeout_ms: int = Field(default=300_000, ge=1)

    @classmethod
    def from_mana_settings(
        cls,
        settings: object,
        *,
        provider: str | None = None,
    ) -> "CodexSettings":
        connection = resolve_inference_connection(settings, provider=provider, require_api_key=False)
        return cls(
            enabled=bool(getattr(settings, "mana_codex_enabled", False)),
            codex_bin=str(getattr(settings, "mana_codex_bin", "codex") or "codex"),
            max_workers=int(getattr(settings, "mana_codex_max_workers", 2) or 2),
            stream_events=bool(getattr(settings, "mana_codex_stream_events", True)),
            worktree_isolation=bool(getattr(settings, "mana_codex_worktree_isolation", False)),
            task_timeout_seconds=int(getattr(settings, "mana_codex_task_timeout_seconds", 1800) or 1800),
            model=str(getattr(settings, "mana_codex_model", "") or "").strip() or None,
            allow_network=bool(getattr(settings, "mana_codex_allow_network", False)),
            provider=connection.provider,
            provider_display_name=connection.display_name,
            api_key=connection.api_key,
            base_url=connection.base_url,
            http_headers=connection.headers,
            env_http_headers=connection.env_headers,
            query_params=connection.query_params,
            supports_responses_api=connection.supports_responses_api,
        )

    @field_validator("approval_policy")
    @classmethod
    def _approval_policy(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if cleaned not in {"never", "untrusted", "on-request"}:
            raise ValueError("approval_policy must be never, untrusted, or on-request")
        return cleaned


__all__ = ["CodexSettings"]
