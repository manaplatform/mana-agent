from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from .errors import TelegramQueueError
from .models import TelegramJob, TelegramUpdate


class TelegramUpdateStore:
    """Durable idempotent ingress queue with FIFO execution lanes."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._initialize()
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA busy_timeout=30000")
            yield db
        finally:
            db.close()

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS telegram_updates (
                    update_id INTEGER PRIMARY KEY,
                    conversation_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued' CHECK(status IN ('queued','processing','completed','failed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    available_at REAL NOT NULL DEFAULT 0,
                    lease_until REAL,
                    last_error TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS telegram_updates_claim_idx
                    ON telegram_updates(status, available_at, update_id);
                CREATE TABLE IF NOT EXISTS telegram_state (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS telegram_sessions (
                    conversation_key TEXT PRIMARY KEY, session_id TEXT NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS telegram_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS telegram_history_session_idx ON telegram_history(session_id,id);
                """
            )

    def persist(self, update: TelegramUpdate, *, conversation_key: str, commit_offset: int | None = None) -> bool:
        now = time.time()
        payload = json.dumps(update.model_dump(mode="json", exclude={"raw"}), ensure_ascii=False, separators=(",", ":"))
        try:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                cursor = db.execute(
                    "INSERT OR IGNORE INTO telegram_updates(update_id,conversation_key,payload,created_at,updated_at) VALUES(?,?,?,?,?)",
                    (update.update_id, conversation_key, payload, now, now),
                )
                if commit_offset is not None:
                    db.execute(
                        "INSERT INTO telegram_state(key,value,updated_at) VALUES('polling_offset',?,?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
                        (str(int(commit_offset)), now),
                    )
                db.commit()
                return cursor.rowcount == 1
        except sqlite3.Error as exc:
            raise TelegramQueueError("Unable to persist Telegram update.") from exc

    def polling_offset(self) -> int | None:
        with self._connect() as db:
            row = db.execute("SELECT value FROM telegram_state WHERE key='polling_offset'").fetchone()
        return int(row["value"]) if row else None

    def recover_abandoned(self, *, now: float | None = None) -> int:
        timestamp = time.time() if now is None else now
        with self._connect() as db:
            result = db.execute(
                "UPDATE telegram_updates SET status='queued',lease_until=NULL,updated_at=? "
                "WHERE status='processing' AND (lease_until IS NULL OR lease_until<=?)",
                (timestamp, timestamp),
            )
        return result.rowcount

    def claim(self, *, lease_seconds: int, now: float | None = None) -> TelegramJob | None:
        timestamp = time.time() if now is None else now
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """
                SELECT candidate.* FROM telegram_updates candidate
                WHERE candidate.status='queued' AND candidate.available_at<=?
                  AND NOT EXISTS (
                    SELECT 1 FROM telegram_updates prior
                    WHERE prior.conversation_key=candidate.conversation_key
                      AND prior.update_id<candidate.update_id
                      AND prior.status IN ('queued','processing')
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM telegram_updates active
                    WHERE active.conversation_key=candidate.conversation_key
                      AND active.status='processing'
                  )
                ORDER BY candidate.update_id LIMIT 1
                """,
                (timestamp,),
            ).fetchone()
            if row is None:
                db.commit()
                return None
            changed = db.execute(
                "UPDATE telegram_updates SET status='processing',attempts=attempts+1,lease_until=?,updated_at=? "
                "WHERE update_id=? AND status='queued'",
                (timestamp + lease_seconds, timestamp, row["update_id"]),
            )
            db.commit()
            if changed.rowcount != 1:
                return None
        return TelegramJob(
            update_id=row["update_id"], conversation_key=row["conversation_key"],
            payload=json.loads(row["payload"]), status="processing", attempts=row["attempts"] + 1,
            available_at=row["available_at"],
        )

    def complete(self, update_id: int) -> None:
        self._set_terminal(update_id, "completed", "")

    def fail(self, update_id: int, error: str, *, max_attempts: int, retry_delay_seconds: int, transient: bool) -> str:
        now = time.time()
        safe_error = str(error or "Telegram update processing failed.")[:500]
        with self._connect() as db:
            row = db.execute("SELECT attempts FROM telegram_updates WHERE update_id=?", (update_id,)).fetchone()
            if row is None:
                raise TelegramQueueError("Telegram queue job was not found.")
            retry = transient and int(row["attempts"]) < max_attempts
            status = "queued" if retry else "failed"
            available = now + retry_delay_seconds * (2 ** max(0, int(row["attempts"]) - 1)) if retry else now
            db.execute(
                "UPDATE telegram_updates SET status=?,available_at=?,lease_until=NULL,last_error=?,updated_at=? WHERE update_id=?",
                (status, available, safe_error, now, update_id),
            )
        return status

    def requeue(self, update_id: int) -> None:
        with self._connect() as db:
            db.execute("UPDATE telegram_updates SET status='queued',lease_until=NULL,updated_at=? WHERE update_id=? AND status='processing'", (time.time(), update_id))

    def _set_terminal(self, update_id: int, status: str, error: str) -> None:
        with self._connect() as db:
            result = db.execute(
                "UPDATE telegram_updates SET status=?,lease_until=NULL,last_error=?,updated_at=? WHERE update_id=? AND status='processing'",
                (status, error, time.time(), update_id),
            )
        if result.rowcount != 1:
            raise TelegramQueueError("Telegram queue job is not processing.")

    def stats(self) -> dict[str, int]:
        result = {status: 0 for status in ("queued", "processing", "completed", "failed")}
        with self._connect() as db:
            for row in db.execute("SELECT status,COUNT(*) count FROM telegram_updates GROUP BY status"):
                result[str(row["status"])] = int(row["count"])
        return result

    def latest_completed_update(self) -> int | None:
        with self._connect() as db:
            row = db.execute("SELECT MAX(update_id) value FROM telegram_updates WHERE status='completed'").fetchone()
        return int(row["value"]) if row and row["value"] is not None else None

    def last_error(self) -> str:
        with self._connect() as db:
            row = db.execute(
                "SELECT last_error FROM telegram_updates WHERE last_error<>'' ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        return str(row["last_error"])[:500] if row else ""

    def session_id(self, conversation_key: str) -> str | None:
        with self._connect() as db:
            row = db.execute("SELECT session_id FROM telegram_sessions WHERE conversation_key=?", (conversation_key,)).fetchone()
        return str(row["session_id"]) if row else None

    def bind_session(self, conversation_key: str, session_id: str) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO telegram_sessions(conversation_key,session_id,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(conversation_key) DO UPDATE SET session_id=excluded.session_id,updated_at=excluded.updated_at",
                (conversation_key, session_id, time.time()),
            )

    def clear_session(self, conversation_key: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM telegram_sessions WHERE conversation_key=?", (conversation_key,))

    def history(self, session_id: str, *, limit: int | None = None) -> list[tuple[str, str]]:
        with self._connect() as db:
            if limit is None:
                rows = db.execute(
                    "SELECT question,answer FROM telegram_history WHERE session_id=? ORDER BY id DESC",
                    (session_id,),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT question,answer FROM telegram_history WHERE session_id=? ORDER BY id DESC LIMIT ?",
                    (session_id, max(1, int(limit))),
                ).fetchall()
        return [(str(row["question"]), str(row["answer"])) for row in reversed(rows)]

    def append_history(self, session_id: str, question: str, answer: str) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO telegram_history(session_id,question,answer,created_at) VALUES(?,?,?,?)",
                (session_id, str(question)[:50_000], str(answer)[:100_000], time.time()),
            )
