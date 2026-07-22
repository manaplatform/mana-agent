from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def mana_home() -> Path:
    configured = str(os.getenv("MANA_HOME") or "").strip()
    return Path(configured).expanduser().resolve() if configured else (Path.home() / ".mana").resolve()


def repository_id_for_path(path: str | Path) -> str:
    """Return a deterministic compatibility id before registry assignment.

    Registered repositories use UUID ids.  This stable path id lets legacy
    single-repository callers share the global index before registration is
    threaded through every call site.
    """

    resolved = str(Path(path).expanduser().resolve())
    identity = os.path.normcase(resolved)
    repositories = mana_home() / "repositories"
    if repositories.is_dir():
        for metadata in repositories.glob("*/repository.json"):
            try:
                payload = json.loads(metadata.read_text(encoding="utf-8"))
            except Exception:
                continue
            persisted_paths = {
                os.path.normcase(str(payload.get("canonical_path") or "")),
                os.path.normcase(str(payload.get("git_root") or "")),
            }
            if identity in persisted_paths:
                repository_id = str(payload.get("repository_id") or "").strip()
                if repository_id:
                    return repository_id
    return "repo_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def workspace_dir(workspace_id: str) -> Path:
    return mana_home() / "workspaces" / str(workspace_id)


def repository_dir(repository_id: str) -> Path:
    return mana_home() / "repositories" / str(repository_id)


def repository_index_dir(repository_id: str) -> Path:
    return repository_dir(repository_id) / "index"


def repository_analysis_dir(repository_id: str) -> Path:
    return repository_dir(repository_id) / "analysis"


def repository_worktrees_dir(repository_id: str) -> Path:
    """Checkout root for Mana-managed agent worktrees (outside the source tree)."""

    return repository_dir(repository_id) / "worktrees"


def repository_managed_worktrees_metadata_dir(repository_id: str) -> Path:
    return repository_dir(repository_id) / "managed_worktrees"


def session_dir(session_id: str) -> Path:
    return mana_home() / "sessions" / str(session_id)


def ensure_home_layout() -> Path:
    home = mana_home()
    for child in (
        "workspaces",
        "repositories",
        "sessions",
        "cache/files",
        "cache/parsed-chunks",
        "cache/embeddings",
        "logs",
        "llm_logs",
        "observability",
        "automations",
        "skills",
        "skill-proposals",
        "skill-quarantine",
    ):
        (home / child).mkdir(parents=True, exist_ok=True)
    return home
