from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .ids import execution_id


class EvalWorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EvalWorkspace:
    source_repository: Path
    path: Path
    commit: str
    retained: bool = False


class WorkspaceBackend(Protocol):
    def create(self, repository: str | Path, commit: str, *, run_id: str) -> EvalWorkspace: ...
    def cleanup(self, workspace: EvalWorkspace) -> None: ...


class LocalWorktreeBackend:
    def __init__(self, root: str | Path, *, retain: bool = False) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.retain = retain

    def create(self, repository: str | Path, commit: str, *, run_id: str) -> EvalWorkspace:
        repository_path = Path(repository).expanduser().resolve()
        if not (repository_path / ".git").exists():
            probe = subprocess.run(["git", "rev-parse", "--git-dir"], cwd=repository_path, capture_output=True, text=True)
            if probe.returncode != 0:
                raise EvalWorkspaceError(f"not a Git repository: {repository_path}")
        resolved = subprocess.run(["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=repository_path, capture_output=True, text=True)
        if resolved.returncode != 0:
            raise EvalWorkspaceError(f"invalid repository commit {commit!r}: {resolved.stderr.strip()}")
        path = self.root / f"{run_id}-{execution_id('ws')[-12:]}"
        result = subprocess.run(["git", "worktree", "add", "--detach", str(path), resolved.stdout.strip()], cwd=repository_path, capture_output=True, text=True)
        if result.returncode != 0:
            raise EvalWorkspaceError(result.stderr.strip() or "git worktree creation failed")
        status = subprocess.run(["git", "status", "--porcelain"], cwd=path, capture_output=True, text=True)
        if status.returncode != 0 or status.stdout.strip():
            self.cleanup(EvalWorkspace(repository_path, path, resolved.stdout.strip()))
            raise EvalWorkspaceError("evaluation worktree did not start clean")
        return EvalWorkspace(repository_path, path, resolved.stdout.strip(), self.retain)

    def cleanup(self, workspace: EvalWorkspace) -> None:
        if workspace.retained or not workspace.path.exists():
            return
        # This checkout is uniquely created for the run and its diff has already
        # been finalized as an immutable artifact before cleanup.
        result = subprocess.run(["git", "worktree", "remove", "--force", str(workspace.path)], cwd=workspace.source_repository, capture_output=True, text=True)
        if result.returncode != 0:
            raise EvalWorkspaceError(result.stderr.strip() or "git worktree cleanup failed")


class UnsupportedWorkspaceBackend:
    def __init__(self, name: str) -> None:
        self.name = name

    def create(self, repository: str | Path, commit: str, *, run_id: str) -> EvalWorkspace:
        raise EvalWorkspaceError(f"workspace backend {self.name!r} is not implemented")

    def cleanup(self, workspace: EvalWorkspace) -> None:
        _ = workspace


def workspace_backend(name: str, root: str | Path, *, retain: bool = False) -> WorkspaceBackend:
    if name == "local-worktree":
        return LocalWorktreeBackend(root, retain=retain)
    if name in {"execution-fabric", "fleet"}:
        return UnsupportedWorkspaceBackend(
            f"{name} (a validated ExecutionManager/FleetService decision is required)"
        )
    return UnsupportedWorkspaceBackend(name)
