"""Atomic release deployment contracts and health-gated success."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from .models import StrictModel, utc_now


class DeploymentRequest(StrictModel):
    repository: str
    revision: str
    release_root: str
    health_check_url: str
    environment_refs: dict[str, str] = Field(default_factory=dict)
    migration_argv: list[str] = Field(default_factory=list)
    build_argv: list[str] = Field(default_factory=list)

    def validate_secret_refs(self) -> "DeploymentRequest":
        for name, reference in self.environment_refs.items():
            if not name or not reference.startswith("secret://"):
                raise ValueError("Deployment environment values must use named secret references.")
        if not self.release_root.startswith("/"):
            raise ValueError("Deployment release_root must be an absolute remote path.")
        return self


class DeploymentRecord(StrictModel):
    deployment_id: str
    server_id: str
    revision: str
    release_path: str
    previous_release_path: str | None = None
    health_check_passed: bool = False
    rolled_back: bool = False
    created_at: datetime = Field(default_factory=utc_now)


def release_switch_argv(current_link: str, release_path: str) -> list[str]:
    if not current_link.startswith("/") or not release_path.startswith("/"):
        raise ValueError("Release and current paths must be absolute.")
    return ["ln", "-sfn", release_path, current_link]


def health_check_argv(url: str) -> list[str]:
    if not url.startswith(("http://", "https://")):
        raise ValueError("Deployment health check must be an HTTP(S) URL.")
    return ["curl", "--fail", "--silent", "--show-error", "--max-time", "15", url]
