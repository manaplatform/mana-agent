"""Durable, message-scoped state for gateway chat turns backed by SQLite."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mana_agent.utils.tool_results import json_safe_tool_payload
from mana_agent.workspaces.paths import session_dir


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
    """Cross-process, atomic turn ledger stored below the canonical session using SQLite."""

    def __init__(self, session_id: str) -> None:
        self.session_id = str(session_id)
        self.dir = session_dir(self.session_id)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.dir / "turns.db"
        self.legacy_json_path = self.dir / "turns.json"
        self._thread_lock = threading.RLock()
        self._init_db()
        self._migrate_legacy_json_if_needed()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 30000;")
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    user_message_id TEXT NOT NULL,
                    message_fingerprint TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'received',
                    normalized_intent TEXT NOT NULL DEFAULT '',
                    routing_decision_id TEXT NOT NULL DEFAULT '',
                    related_task_ids TEXT NOT NULL DEFAULT '[]',
                    created_task_ids TEXT NOT NULL DEFAULT '[]',
                    response_message_id TEXT NOT NULL DEFAULT '',
                    response_execution_id TEXT NOT NULL DEFAULT '',
                    response TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(conversation_id, user_message_id)
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_turns_conv_user ON turns(conversation_id, user_message_id);"
            )
            conn.commit()

    def _migrate_legacy_json_if_needed(self) -> None:
        if not self.legacy_json_path.exists():
            return
        try:
            payload = json.loads(self.legacy_json_path.read_text(encoding="utf-8"))
            records = payload.get("records") or {}
            with self._connect() as conn:
                for item in records.values():
                    if isinstance(item, dict):
                        rec = ChatTurnRecord.from_dict(item)
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO turns (
                                turn_id, conversation_id, user_message_id, message_fingerprint,
                                received_at, status, normalized_intent, routing_decision_id,
                                related_task_ids, created_task_ids, response_message_id,
                                response_execution_id, response, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                            """,
                            (
                                rec.turn_id,
                                rec.conversation_id,
                                rec.user_message_id,
                                rec.message_fingerprint,
                                rec.received_at,
                                rec.status,
                                rec.normalized_intent,
                                rec.routing_decision_id,
                                json.dumps(rec.related_task_ids, ensure_ascii=False),
                                json.dumps(rec.created_task_ids, ensure_ascii=False),
                                rec.response_message_id,
                                rec.response_execution_id,
                                json.dumps(json_safe_tool_payload(rec.response), ensure_ascii=False),
                                rec.received_at,
                                _now(),
                            ),
                        )
                conn.commit()
        except Exception:
            pass

    def _hydrate_turn(self, row: sqlite3.Row) -> ChatTurnRecord:
        try:
            related_task_ids = json.loads(row["related_task_ids"])
        except Exception:
            related_task_ids = []
        try:
            created_task_ids = json.loads(row["created_task_ids"])
        except Exception:
            created_task_ids = []
        try:
            response = json.loads(row["response"])
        except Exception:
            response = {}

        return ChatTurnRecord(
            turn_id=str(row["turn_id"]),
            conversation_id=str(row["conversation_id"]),
            user_message_id=str(row["user_message_id"]),
            message_fingerprint=str(row["message_fingerprint"]),
            received_at=str(row["received_at"]),
            status=str(row["status"]),
            normalized_intent=str(row["normalized_intent"] or ""),
            routing_decision_id=str(row["routing_decision_id"] or ""),
            related_task_ids=related_task_ids,
            created_task_ids=created_task_ids,
            response_message_id=str(row["response_message_id"] or ""),
            response_execution_id=str(row["response_execution_id"] or ""),
            response=response,
        )

    def _load(self) -> dict[str, dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM turns ORDER BY created_at ASC;").fetchall()
            result = {}
            for r in rows:
                rec = self._hydrate_turn(r)
                key = f"{rec.conversation_id}:{rec.user_message_id}"
                result[key] = json_safe_tool_payload(asdict(rec))
            return result

    def create_or_get(
        self,
        *,
        conversation_id: str,
        user_message_id: str,
        turn_id: str,
        text: str,
    ) -> tuple[ChatTurnRecord, bool]:
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        now_str = _now()

        with self._thread_lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            row = conn.execute(
                "SELECT * FROM turns WHERE conversation_id = ? AND user_message_id = ?;",
                (conversation_id, user_message_id),
            ).fetchone()
            if row is not None:
                record = self._hydrate_turn(row)
                if record.message_fingerprint != fingerprint:
                    conn.rollback()
                    raise ValueError("user_message_id already belongs to a different message")
                conn.commit()
                return record, True

            record = ChatTurnRecord(
                turn_id=turn_id,
                conversation_id=conversation_id,
                user_message_id=user_message_id,
                message_fingerprint=fingerprint,
                received_at=now_str,
            )
            conn.execute(
                """
                INSERT INTO turns (
                    turn_id, conversation_id, user_message_id, message_fingerprint,
                    received_at, status, normalized_intent, routing_decision_id,
                    related_task_ids, created_task_ids, response_message_id,
                    response_execution_id, response, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.turn_id,
                    record.conversation_id,
                    record.user_message_id,
                    record.message_fingerprint,
                    record.received_at,
                    record.status,
                    record.normalized_intent,
                    record.routing_decision_id,
                    json.dumps(record.related_task_ids, ensure_ascii=False),
                    json.dumps(record.created_task_ids, ensure_ascii=False),
                    record.response_message_id,
                    record.response_execution_id,
                    json.dumps(json_safe_tool_payload(record.response), ensure_ascii=False),
                    now_str,
                    now_str,
                ),
            )
            conn.commit()
            return record, False

    def update(self, record: ChatTurnRecord) -> ChatTurnRecord:
        now_str = _now()
        safe_response = json_safe_tool_payload(record.response)
        record.response = safe_response

        with self._thread_lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                INSERT OR REPLACE INTO turns (
                    turn_id, conversation_id, user_message_id, message_fingerprint,
                    received_at, status, normalized_intent, routing_decision_id,
                    related_task_ids, created_task_ids, response_message_id,
                    response_execution_id, response, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.turn_id,
                    record.conversation_id,
                    record.user_message_id,
                    record.message_fingerprint,
                    record.received_at,
                    record.status,
                    record.normalized_intent,
                    record.routing_decision_id,
                    json.dumps(record.related_task_ids, ensure_ascii=False),
                    json.dumps(record.created_task_ids, ensure_ascii=False),
                    record.response_message_id,
                    record.response_execution_id,
                    json.dumps(safe_response, ensure_ascii=False),
                    record.received_at,
                    now_str,
                ),
            )
            conn.commit()
        return record
