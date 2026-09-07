"""Durable thread state persistence for Codex coding sessions."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mana_agent.integrations.codex.runtime_environment import get_codex_session_home, get_session_state_hash

logger = logging.getLogger(__name__)


def _thread_state_file(repository_id: str, session_id: str) -> Path:
    home = get_codex_session_home(repository_id, session_id)
    return home / "thread_state.json"


def save_codex_session_thread(repository_id: str, session_id: str, thread_id: str) -> None:
    """Durably persist the Codex thread ID associated with a Mana repository/session."""
    repo = str(repository_id or "").strip()
    sess = str(session_id or "").strip()
    th = str(thread_id or "").strip()
    if not repo or not sess or not th:
        return
    path = _thread_state_file(repo, sess)
    data = {
        "repository_id": repo,
        "session_id": sess,
        "thread_id": th,
        "session_state_hash": get_session_state_hash(repo, sess),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_path.chmod(0o600)
        tmp_path.replace(path)
    except OSError as exc:
        logger.warning(
            "Failed to persist Codex thread state for repo=%s session=%s: %s",
            repo,
            sess,
            exc,
        )


def load_codex_session_thread(repository_id: str, session_id: str) -> str:
    """Load the persisted Codex thread ID associated with a Mana repository/session."""
    repo = str(repository_id or "").strip()
    sess = str(session_id or "").strip()
    if not repo or not sess:
        return ""
    path = _thread_state_file(repo, sess)
    if not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return str(data.get("thread_id") or "").strip()
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to read Codex thread state for repo=%s session=%s: %s",
            repo,
            sess,
            exc,
        )
    return ""


def clear_codex_session_thread(repository_id: str, session_id: str) -> None:
    """Invalidate only the persisted Codex thread ID binding for a session."""
    repo = str(repository_id or "").strip()
    sess = str(session_id or "").strip()
    if not repo or not sess:
        return
    path = _thread_state_file(repo, sess)
    if path.is_file():
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "Failed to clear Codex thread state for repo=%s session=%s: %s",
                repo,
                sess,
                exc,
            )


__all__ = [
    "clear_codex_session_thread",
    "load_codex_session_thread",
    "save_codex_session_thread",
]
