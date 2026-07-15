"""Persistence for Mana-managed coding worktrees."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from mana_agent.multi_agent.worktrees.models import ManagedWorkspace
from mana_agent.workspaces.paths import repository_dir


def managed_worktrees_root(repository_id: str) -> Path:
    return repository_dir(repository_id) / "managed_worktrees"


def managed_worktree_checkouts_root(repository_id: str) -> Path:
    return repository_dir(repository_id) / "worktrees"


def managed_workspace_metadata_path(repository_id: str, workspace_id: str) -> Path:
    return managed_worktrees_root(repository_id) / "metadata" / f"{workspace_id}.json"


def managed_workspace_index_path(repository_id: str) -> Path:
    return managed_worktrees_root(repository_id) / "index.json"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, default=str)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class ManagedWorkspaceStore:
    """Load and persist managed workspace metadata for one repository."""

    def __init__(self, repository_id: str) -> None:
        self.repository_id = str(repository_id).strip()
        if not self.repository_id:
            raise ValueError("repository_id is required")
        self.root = managed_worktrees_root(self.repository_id)
        self.metadata_dir = self.root / "metadata"
        self.index_path = managed_workspace_index_path(self.repository_id)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        managed_worktree_checkouts_root(self.repository_id).mkdir(parents=True, exist_ok=True)

    def list(self) -> list[ManagedWorkspace]:
        rows: list[ManagedWorkspace] = []
        for path in sorted(self.metadata_dir.glob("*.json")):
            try:
                rows.append(self.load(path.stem))
            except (OSError, json.JSONDecodeError, ValueError, TypeError):
                continue
        return rows

    def get(self, workspace_id: str) -> ManagedWorkspace:
        path = managed_workspace_metadata_path(self.repository_id, workspace_id)
        if not path.is_file():
            raise FileNotFoundError(f"managed workspace not found: {workspace_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid managed workspace metadata: {workspace_id}")
        return ManagedWorkspace.from_dict(payload)

    def get_by_task_id(self, task_id: str) -> ManagedWorkspace | None:
        task = str(task_id or "").strip()
        if not task:
            return None
        for item in self.list():
            if item.task_id == task:
                return item
        return None

    def save(self, workspace: ManagedWorkspace) -> ManagedWorkspace:
        if workspace.repository_id != self.repository_id:
            raise ValueError("workspace repository_id does not match store")
        path = managed_workspace_metadata_path(self.repository_id, workspace.workspace_id)
        _atomic_write_json(path, workspace.to_dict())
        self._rewrite_index()
        return workspace

    def delete(self, workspace_id: str) -> None:
        path = managed_workspace_metadata_path(self.repository_id, workspace_id)
        if path.is_file():
            path.unlink()
        self._rewrite_index()

    def load(self, workspace_id: str) -> ManagedWorkspace:
        return self.get(workspace_id)

    def _rewrite_index(self) -> None:
        rows = [item.list_row() for item in self.list()]
        _atomic_write_json(
            self.index_path,
            {
                "repository_id": self.repository_id,
                "count": len(rows),
                "workspaces": rows,
            },
        )
