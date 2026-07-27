from __future__ import annotations

from pathlib import Path
import stat

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _csv(value: object) -> frozenset[str]:
    if isinstance(value, (set, frozenset, list, tuple)):
        return frozenset(str(item).strip().lower() for item in value if str(item).strip())
    return frozenset(item.strip().lower() for item in str(value or "").split(",") if item.strip())


class GitHubAutopilotSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    app_id: str = ""
    private_key_path: Path | None = None
    webhook_secret: str = Field(default="", repr=False)
    public_webhook_url: str = ""
    invocation_name: str = "@mana-agent"
    fix_label: str = "mana-fix"
    minimum_actor_permission: str = "write"
    allowed_repositories: frozenset[str] = frozenset()
    allowed_organizations: frozenset[str] = frozenset()
    allowed_workflows: frozenset[str] = frozenset()
    allowed_branches: frozenset[str] = frozenset()
    actor_allowlist: frozenset[str] = frozenset()
    security_events_enabled: bool = False
    allow_bots: bool = False
    worker_concurrency: int = Field(default=2, ge=1, le=16)
    maximum_job_iterations: int = Field(default=8, ge=1, le=100)
    maximum_job_runtime: int = Field(default=1800, ge=30)
    maximum_changed_files: int = Field(default=50, ge=1)
    draft_pr_only: bool = True
    workflow_files_write_enabled: bool = False
    api_url: str = "https://api.github.com"

    @field_validator("minimum_actor_permission")
    @classmethod
    def validate_permission(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"read", "triage", "write", "maintain", "admin"}:
            raise ValueError("minimum_actor_permission must be read, triage, write, maintain, or admin")
        return value

    @classmethod
    def from_mana_settings(cls, settings: object) -> "GitHubAutopilotSettings":
        key = str(getattr(settings, "mana_github_app_private_key_path", "") or "").strip()
        return cls(
            enabled=bool(getattr(settings, "mana_github_autopilot_enabled", False)),
            app_id=str(getattr(settings, "mana_github_app_id", "") or "").strip(),
            private_key_path=Path(key).expanduser().resolve() if key else None,
            webhook_secret=str(getattr(settings, "mana_github_webhook_secret", "") or ""),
            public_webhook_url=str(getattr(settings, "mana_github_public_webhook_url", "") or ""),
            invocation_name=str(getattr(settings, "mana_github_invocation_name", "@mana-agent") or "@mana-agent"),
            fix_label=str(getattr(settings, "mana_github_fix_label", "mana-fix") or "mana-fix"),
            minimum_actor_permission=str(getattr(settings, "mana_github_minimum_actor_permission", "write") or "write"),
            allowed_repositories=_csv(getattr(settings, "mana_github_allowed_repositories", "")),
            allowed_organizations=_csv(getattr(settings, "mana_github_allowed_organizations", "")),
            allowed_workflows=_csv(getattr(settings, "mana_github_allowed_workflows", "")),
            allowed_branches=_csv(getattr(settings, "mana_github_allowed_branches", "")),
            actor_allowlist=_csv(getattr(settings, "mana_github_actor_allowlist", "")),
            security_events_enabled=bool(getattr(settings, "mana_github_security_events_enabled", False)),
            allow_bots=bool(getattr(settings, "mana_github_allow_bots", False)),
            worker_concurrency=min(
                int(getattr(settings, "mana_github_worker_concurrency", 2)),
                int(getattr(settings, "mana_codex_max_workers", 2)),
                int(getattr(settings, "mana_lane_global_worker_limit", 8)),
            ),
            maximum_job_iterations=int(getattr(settings, "mana_github_maximum_job_iterations", 8)),
            maximum_job_runtime=int(getattr(settings, "mana_github_maximum_job_runtime", 1800)),
            maximum_changed_files=int(getattr(settings, "mana_github_maximum_changed_files", 50)),
            draft_pr_only=bool(getattr(settings, "mana_github_draft_pr_only", True)),
            workflow_files_write_enabled=bool(getattr(settings, "mana_github_workflow_files_write_enabled", False)),
        )

    def startup_errors(self) -> list[str]:
        if not self.enabled:
            return ["GitHub Autopilot is disabled"]
        errors: list[str] = []
        if not self.app_id.isdigit():
            errors.append("MANA_GITHUB_APP_ID must be configured as a numeric GitHub App ID")
        if self.private_key_path is None or not self.private_key_path.is_file():
            errors.append("MANA_GITHUB_APP_PRIVATE_KEY_PATH must reference a readable private key")
        elif self.private_key_path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            errors.append("GitHub App private key permissions must deny group and other access")
        if not self.webhook_secret:
            errors.append("MANA_GITHUB_WEBHOOK_SECRET must be available from secure configuration")
        if not self.public_webhook_url:
            errors.append("MANA_GITHUB_PUBLIC_WEBHOOK_URL must be configured")
        return errors
