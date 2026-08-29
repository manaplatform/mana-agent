"""Durable, message-scoped state for gateway chat turns.

Conversation lifetime and execution lifetime are deliberately separate.  This
store is the idempotency boundary for an incoming user message; a terminal
execution can never make a later message look like a duplicate.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from mana_agent.workspaces.paths import session_dir
from mana_agent.workspaces.store import atomic_write_json

if os.name == "nt":  # pragma: no cover
    import msvcrt
else:  # pragma: no cover
    import fcntl


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ChatTurnRecord:
    turn_id: str
    conversation_id: str
    user_message_id: str
    message_fingerprint: str
    received_at: str = field(default_factory=_now)
    status: str = "received"
    normalized_intent: str = ""
    routing_decision_id: str = ""
    related_task_ids: list[str] = field(default_factory=list)
    created_task_ids: list[str] = field(default_factory=list)
    response_message_id: str = ""
    response_execution_id: str = ""
    response: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChatTurnRecord":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in payload.items() if key in fields})


class ChatTurnStore:
    """Cross-process, atomic turn ledger stored below the canonical session."""

    def __init__(self, session_id: str) -> None:
        self.session_id = str(session_id)
        self.path = session_dir(self.session_id) / "turns.json"
        self.lock_path = session_dir(self.session_id) / ".turns.lock"
        self._thread_lock = threading.RLock()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(exist_ok=True)
        with self._thread_lock, self.lock_path.open("r+b") as handle:
            if os.name == "nt":  # pragma: no cover
                handle.seek(0)
                if not handle.read(1):
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:  # pragma: no cover
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":  # pragma: no cover
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("chat turn ledger is corrupt; no fallback ledger was used") from exc
        return dict(payload.get("records") or {})

    def _save(self, records: dict[str, dict[str, Any]]) -> None:
        from mana_agent.utils.tool_results import json_safe_tool_payload

        atomic_write_json(self.path, {"schema_version": 1, "records": json_safe_tool_payload(records)})

    def create_or_get(self, *, conversation_id: str, user_message_id: str, turn_id: str, text: str) -> tuple[ChatTurnRecord, bool]:
        from mana_agent.utils.tool_results import json_safe_tool_payload

        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        key = f"{conversation_id}:{user_message_id}"
        with self._locked():
            records = self._load()
            existing = records.get(key)
            if existing is not None:
                record = ChatTurnRecord.from_dict(existing)
                if record.message_fingerprint != fingerprint:
                    raise ValueError("user_message_id already belongs to a different message")
                return record, True
            record = ChatTurnRecord(turn_id=turn_id, conversation_id=conversation_id, user_message_id=user_message_id, message_fingerprint=fingerprint)
            records[key] = json_safe_tool_payload(asdict(record))
            self._save(records)
            return record, False

    def update(self, record: ChatTurnRecord) -> ChatTurnRecord:
        from mana_agent.utils.tool_results import json_safe_tool_payload

        record.response = json_safe_tool_payload(record.response)
        key = f"{record.conversation_id}:{record.user_message_id}"
        with self._locked():
            records = self._load()
            records[key] = json_safe_tool_payload(asdict(record))
            self._save(records)
        return record
