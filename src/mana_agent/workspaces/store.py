from __future__ import annotations

import json
import os
import tempfile
import time
import shutil
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from mana_agent.workspaces.models import RepositoryRecord, SessionRecord, WorkspaceRecord
from mana_agent.workspaces.paths import ensure_home_layout, repository_dir, session_dir, workspace_dir

T = TypeVar("T", bound=BaseModel)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Windows may briefly hold the destination open for indexing or virus
        # scanning. Retry only that sharing violation; other I/O errors fail.
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.01 * (2**attempt))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path: Path, cls: type[T]) -> T:
    return cls.model_validate_json(path.read_text(encoding="utf-8"))


class WorkspaceStore:
    def __init__(self) -> None:
        self.home = ensure_home_layout()

    def save_repository(self, record: RepositoryRecord) -> RepositoryRecord:
        atomic_write_json(repository_dir(record.repository_id) / "repository.json", record.model_dump(mode="json"))
        return record

    def get_repository(self, repository_id: str) -> RepositoryRecord:
        return _read(repository_dir(repository_id) / "repository.json", RepositoryRecord)

    def list_repositories(self) -> list[RepositoryRecord]:
        rows: list[RepositoryRecord] = []
        for path in sorted((self.home / "repositories").glob("*/repository.json")):
            try:
                rows.append(_read(path, RepositoryRecord))
            except Exception:
                continue
        return rows

    def find_repository_by_path(self, path: str | Path) -> RepositoryRecord | None:
        canonical = os.path.normcase(str(Path(path).expanduser().resolve()))
        return next(
            (
                item
                for item in self.list_repositories()
                if canonical
                in {
                    os.path.normcase(str(item.canonical_path or "")),
                    os.path.normcase(str(item.git_root or "")),
                }
            ),
            None,
        )

    def save_workspace(self, record: WorkspaceRecord) -> WorkspaceRecord:
        atomic_write_json(workspace_dir(record.workspace_id) / "workspace.json", record.model_dump(mode="json"))
        return record

    def get_workspace(self, workspace_id: str) -> WorkspaceRecord:
        return _read(workspace_dir(workspace_id) / "workspace.json", WorkspaceRecord)

    def list_workspaces(self) -> list[WorkspaceRecord]:
        rows: list[WorkspaceRecord] = []
        for path in sorted((self.home / "workspaces").glob("*/workspace.json")):
            try:
                rows.append(_read(path, WorkspaceRecord))
            except Exception:
                continue
        return rows

    def delete_workspace(self, workspace_id: str) -> None:
        path = workspace_dir(workspace_id) / "workspace.json"
        if path.exists():
            path.unlink()

    def save_session(self, record: SessionRecord) -> SessionRecord:
        atomic_write_json(session_dir(record.session_id) / "session.json", record.model_dump(mode="json"))
        return record

    def get_session(self, session_id: str) -> SessionRecord:
        return _read(session_dir(session_id) / "session.json", SessionRecord)

    def list_sessions(self) -> list[SessionRecord]:
        rows: list[SessionRecord] = []
        for path in sorted((self.home / "sessions").glob("*/session.json")):
            try:
                rows.append(_read(path, SessionRecord))
            except Exception:
                continue
        return sorted(rows, key=lambda item: item.updated_at, reverse=True)

    def delete_session(self, session_id: str) -> None:
        """Physically delete a session after proving its path is below Mana home.

        A session is a directory, rather than an archive marker.  Refuse symlinked
        or malformed targets so a corrupt record can never widen deletion scope.
        """

        sid = str(session_id or "").strip()
        if not sid or not sid.startswith("session_") or not sid.replace("_", "").isalnum():
            raise ValueError("invalid session id")
        sessions_root = (self.home / "sessions").resolve()
        target = self.home / "sessions" / sid
        if target.is_symlink():
            raise ValueError("refusing to delete a symlinked session directory")
        resolved = target.resolve(strict=False)
        if resolved.parent != sessions_root:
            raise ValueError("session deletion target is outside Mana home")
        if resolved.exists():
            shutil.rmtree(resolved)
