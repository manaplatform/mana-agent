"""Durable transport checkpoints; task authority remains in the Supervisor."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from mana_agent.batch.store import BatchJobStore


class BatchOperationStore:
    def __init__(self, store: BatchJobStore) -> None:
        self.store = store
        self._lock = threading.RLock()
        with store._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS batch_operations (
                    operation_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    input_file_id TEXT NOT NULL DEFAULT '',
                    batch_id TEXT NOT NULL DEFAULT '',
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batch_runs (
                    task_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
            """)

    def prepare(self, operation_id: str, fingerprint: str, task_id: str, payload: dict[str, Any]) -> bool:
        """Claim submission once, before making any remote request."""
        with self._lock, self.store._get_connection() as conn:
            result = conn.execute(
                "INSERT OR IGNORE INTO batch_operations VALUES (?, ?, ?, 'prepared', '', '', ?, ?)",
                (operation_id, fingerprint, task_id, json.dumps(payload), time.time()),
            )
            row = conn.execute("SELECT fingerprint FROM batch_operations WHERE operation_id=?", (operation_id,)).fetchone()
            if row["fingerprint"] != fingerprint:
                raise ValueError("Batch operation ID was reused with different input")
            return result.rowcount == 1

    def get(self, operation_id: str) -> dict[str, Any] | None:
        with self._lock, self.store._get_connection() as conn:
            row = conn.execute("SELECT * FROM batch_operations WHERE operation_id=?", (operation_id,)).fetchone()
            return dict(row) if row else None

    def update(self, operation_id: str, *, state: str, input_file_id: str = "", batch_id: str = "") -> None:
        with self._lock, self.store._get_connection() as conn:
            conn.execute(
                "UPDATE batch_operations SET state=?, input_file_id=CASE WHEN ?='' THEN input_file_id ELSE ? END, "
                "batch_id=CASE WHEN ?='' THEN batch_id ELSE ? END, updated_at=? WHERE operation_id=?",
                (state, input_file_id, input_file_id, batch_id, batch_id, time.time(), operation_id),
            )

    def load_run(self, task_id: str) -> dict[str, Any] | None:
        with self._lock, self.store._get_connection() as conn:
            row = conn.execute("SELECT version, payload FROM batch_runs WHERE task_id=?", (task_id,)).fetchone()
            if not row:
                return None
            return {**json.loads(row["payload"]), "version": row["version"]}

    def save_run(self, task_id: str, payload: dict[str, Any]) -> None:
        """CAS prevents two resumed workers from advancing the same checkpoint."""
        version = int(payload.get("version", 0))
        encoded = json.dumps({k: v for k, v in payload.items() if k != "version"}, ensure_ascii=False)
        with self._lock, self.store._get_connection() as conn:
            if not version:
                changed = conn.execute(
                    "INSERT OR IGNORE INTO batch_runs VALUES (?, 1, ?, ?)", (task_id, encoded, time.time()),
                ).rowcount
            else:
                changed = conn.execute(
                    "UPDATE batch_runs SET version=version+1, payload=?, updated_at=? WHERE task_id=? AND version=?",
                    (encoded, time.time(), task_id, version),
                ).rowcount
            if changed != 1:
                raise RuntimeError("Batch checkpoint changed concurrently; reload before continuing")
        payload["version"] = version + 1

    def pending_runs(self, root: str = "") -> list[dict[str, Any]]:
        with self._lock, self.store._get_connection() as conn:
            rows = conn.execute("SELECT payload, version FROM batch_runs ORDER BY updated_at").fetchall()
        runs = [{**json.loads(row["payload"]), "version": row["version"]} for row in rows]
        return [run for run in runs if (run.get("status") in {"waiting", "running"} or run.get("final_delivery_pending"))
                and (not root or run.get("owner_root", run.get("root")) == root)]
