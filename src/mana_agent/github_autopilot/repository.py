from __future__ import annotations

import re
import os
import subprocess
import tempfile
from pathlib import Path

from mana_agent.multi_agent.worktrees import WorkspaceManager
from mana_agent.workspaces.models import RepositoryRecord, RepositoryStatus
from mana_agent.workspaces.paths import repository_dir
from mana_agent.workspaces.store import WorkspaceStore

from .models import GitHubJob


def branch_for(job: GitHubJob) -> str:
    title = str(job.context.get("title") or job.route_decision.trigger or "fix")
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:36] or "fix"
    prefixes = {"issue": "issue", "pull_request": "pr", "workflow_run": "workflow", "dependabot_alert": "dependabot-alert", "code_scanning_alert": "code-alert", "secret_scanning_alert": "secret-alert"}
    return f"mana/{prefixes.get(job.subject_type, job.subject_type)}-{job.subject_number}-{slug}"


def _run(args: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, timeout=120, check=False)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout or "command failed")[:2000])
    return result.stdout.strip()


class RepositoryManager:
    def __init__(self, store: WorkspaceStore | None = None) -> None:
        self.store = store or WorkspaceStore()

    def prepare(self, job: GitHubJob, token: str) -> tuple[Path, str]:
        repository_id = f"github_{job.repository_id}"
        cache = repository_dir(repository_id) / "checkout"
        clone_url = f"https://github.com/{job.repository_full_name}.git"
        with self._credential_environment(token) as env:
            if not (cache / ".git").exists():
                cache.parent.mkdir(parents=True, exist_ok=True)
                _run(["git", "clone", "--filter=blob:none", "--no-checkout", clone_url, str(cache)], env=env)
            else:
                _run(["git", "remote", "set-url", "origin", clone_url], cwd=cache)
            _run(["git", "fetch", "--prune", "origin", job.target_sha or job.base_branch], cwd=cache, env=env)
        record = RepositoryRecord(repository_id=repository_id, name=job.repository_full_name.rsplit("/", 1)[-1], canonical_path=str(cache), git_root=str(cache), remote_url=f"https://github.com/{job.repository_full_name}.git", branch=job.base_branch, head_sha=job.target_sha or None, kind="git", tags=["github-autopilot", f"github-id:{job.repository_id}"], status=RepositoryStatus(available=True))
        self.store.save_repository(record)
        manager = WorkspaceManager(cache, repository_id=repository_id)
        workspace = manager.create_for_task(job.session_id, title=str(job.context.get("title") or job.route_decision.trigger), assigned_agent_id="codex", session_id=job.session_id, base_revision=job.target_sha or f"origin/{job.base_branch}", reuse_existing=True)
        desired = job.branch_name or branch_for(job)
        path = Path(workspace.worktree_path)
        if job.target_sha:
            current = _run(["git", "rev-parse", "HEAD"], cwd=path)
            if current != job.target_sha:
                if _run(["git", "status", "--porcelain"], cwd=path):
                    raise RuntimeError("cannot synchronize a dirty GitHub Autopilot worktree")
                _run(["git", "merge", "--ff-only", job.target_sha], cwd=path)
        if workspace.branch_name != desired:
            _run(["git", "branch", "-m", desired], cwd=path)
            workspace.branch_name = desired
            manager.store.save(workspace)
        return path, desired

    @staticmethod
    def push(worktree: Path, repository: str, branch: str, token: str) -> None:
        url = f"https://github.com/{repository}.git"
        with RepositoryManager._credential_environment(token) as env:
            _run(["git", "push", "-u", url, f"HEAD:refs/heads/{branch}"], cwd=worktree, env=env)

    @staticmethod
    def _credential_environment(token: str):
        """Provide a process-scoped token through askpass, never argv or git config."""
        class CredentialContext:
            def __enter__(self) -> dict[str, str]:
                self.directory = tempfile.TemporaryDirectory(prefix="mana-gh-auth-")
                script = Path(self.directory.name) / "askpass.sh"
                script.write_text('#!/bin/sh\ncase "$1" in *Username*) printf %s "$MANA_GH_USERNAME";; *) printf %s "$MANA_GH_INSTALLATION_TOKEN";; esac\n', encoding="utf-8")
                script.chmod(0o700)
                env = dict(os.environ)
                env.update({"GIT_ASKPASS": str(script), "GIT_TERMINAL_PROMPT": "0", "MANA_GH_USERNAME": "x-access-token", "MANA_GH_INSTALLATION_TOKEN": token})
                return env

            def __exit__(self, *_args: object) -> None:
                self.directory.cleanup()

        return CredentialContext()
