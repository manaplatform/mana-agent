"""Read-only Codex runtime health checks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.runtime_environment import isolated_codex_probe_environment


class CodexHealthReport(BaseModel):
    healthy: bool
    executable: str | None = None
    version: str = ""
    app_server_available: bool = False
    repository_accessible: bool = False
    errors: list[str] = Field(default_factory=list)


def check_codex_health(settings: CodexSettings, repository_path: str | Path) -> CodexHealthReport:
    errors: list[str] = []
    executable = shutil.which(settings.codex_bin)
    repository = Path(repository_path).expanduser().resolve()
    if not settings.enabled:
        errors.append("Codex integration is disabled")
    if executable is None:
        errors.append(f"Codex executable was not found: {settings.codex_bin}")
    if not repository.is_dir():
        errors.append(f"Repository is not accessible: {repository}")
    version = ""
    app_server_available = False
    if executable is not None:
        try:
            with isolated_codex_probe_environment() as environment:
                completed = subprocess.run(
                    [executable, "--version"],
                    cwd=repository if repository.is_dir() else None,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                    env=environment,
                )
                version = (completed.stdout or completed.stderr).strip()[:200]
                if completed.returncode != 0:
                    errors.append("Codex version check failed")
                elif not version.startswith("codex-cli "):
                    errors.append(
                        "Configured executable is not the official OpenAI Codex CLI: "
                        f"{executable} reported {version or '<empty version>'!r}. "
                        "Set MANA_CODEX_BIN to the official Codex executable."
                    )
                else:
                    app_server = subprocess.run(
                        [executable, "app-server", "--help"],
                        cwd=repository if repository.is_dir() else None,
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                        env=environment,
                    )
                    app_server_available = (
                        app_server.returncode == 0
                        and "Usage: codex app-server" in app_server.stdout
                    )
                    if not app_server_available:
                        errors.append(
                            "The configured official Codex CLI does not provide a usable app-server command"
                        )
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"Codex version check failed: {exc}")
    return CodexHealthReport(
        healthy=not errors,
        executable=executable,
        version=version,
        app_server_available=app_server_available,
        repository_accessible=repository.is_dir(),
        errors=errors,
    )


__all__ = ["CodexHealthReport", "check_codex_health"]
