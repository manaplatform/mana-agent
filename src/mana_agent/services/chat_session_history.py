"""Durable, exact message history for an active workspace chat session."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mana_agent.utils.redaction import redact_json_line, redact_secrets
from mana_agent.workspaces.paths import session_dir


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ChatSessionMessage:
    message_id: str
    session_id: str
    conversation_id: str
    turn_id: str
    role: str
    content: str
    created_at: str = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChatSessionHistory:
    """Append-only session message log with backward-compatible reads."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def path(self, session_id: str) -> Path:
        return session_dir(session_id) / "messages.jsonl"

    def append(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        turn_id: str,
        metadata: dict[str, Any] | None = None,
        message_id: str | None = None,
        conversation_id: str | None = None,
        created_at: str | None = None,
    ) -> ChatSessionMessage:
        sid = str(session_id or "").strip()
        if not sid:
            raise ValueError("session_id is required")
        message = ChatSessionMessage(
            message_id=message_id or f"msg_{uuid.uuid4().hex[:20]}",
            session_id=sid,
            conversation_id=str(conversation_id or sid),
            turn_id=str(turn_id or ""),
            role=str(role or "system").strip().lower(),
            content=redact_json_line(str(content or "")),
            created_at=created_at or _utc_now(),
            metadata=dict(redact_secrets(dict(metadata or {}))),
        )
        path = self.path(sid)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(message.to_dict(), ensure_ascii=False, default=str) + "\n")
        return message

    def list(self, session_id: str, *, limit: int = 500) -> list[ChatSessionMessage]:
        path = self.path(session_id)
        if not path.exists():
            return []
        rows: list[ChatSessionMessage] = []
        with self._lock:
            lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines[-max(1, min(int(limit or 500), 5000)) :]:
            try:
                payload = json.loads(line)
                # Older records may not have conversation_id, turn_id, or metadata.
                rows.append(
                    ChatSessionMessage(
                        message_id=str(payload.get("message_id") or f"legacy_{uuid.uuid4().hex[:20]}"),
                        session_id=str(payload.get("session_id") or session_id),
                        conversation_id=str(payload.get("conversation_id") or payload.get("session_id") or session_id),
                        turn_id=str(payload.get("turn_id") or ""),
                        role=str(payload.get("role") or "system"),
                        content=str(payload.get("content") or ""),
                        created_at=str(payload.get("created_at") or _utc_now()),
                        metadata=dict(payload.get("metadata") or {}),
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return rows
