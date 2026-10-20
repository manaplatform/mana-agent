"""Durable SQLite storage for OpenAI Batch API jobs and request-level results."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mana_agent.batch.models import (
    BatchItemRequest,
    BatchItemResult,
    BatchJobRecord,
    BatchLifecycleStatus,
)

logger = logging.getLogger(__name__)


def default_batch_db_path() -> Path:
    """Return default durable SQLite database path for OpenAI Batch jobs."""
    try:
        from mana_agent.workspaces.paths import mana_home

        return mana_home() / "batch" / "batches.db"
    except Exception:
        return Path.home() / ".mana" / "batch" / "batches.db"


class BatchJobStore:
    """Thread-safe SQLite store maintaining durable batch job states and item results."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._mem_conn: sqlite3.Connection | None = None
        if db_path is None:
            path = default_batch_db_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db_path = str(path.resolve())
        elif db_path != ":memory:":
            path = Path(db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db_path = str(path.resolve())
        else:
            self._db_path = ":memory:"
            self._mem_conn = sqlite3.connect(":memory:", timeout=30.0, check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row

        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        if self._db_path != ":memory:":
            conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_schema(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_file_id TEXT,
                    output_file_id TEXT,
                    error_file_id TEXT,
                    completion_window TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    expires_at REAL,
                    total_requests INTEGER DEFAULT 0,
                    completed_requests INTEGER DEFAULT 0,
                    failed_requests INTEGER DEFAULT 0,
                    task_id TEXT,
                    run_id TEXT,
                    idempotency_key TEXT,
                    git_commit_sha TEXT,
                    metadata_json TEXT,
                    usage_json TEXT,
                    cost_estimate_usd REAL DEFAULT 0.0,
                    error_message TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status);
                CREATE INDEX IF NOT EXISTS idx_batches_idempotency ON batches(idempotency_key);
                CREATE INDEX IF NOT EXISTS idx_batches_task ON batches(task_id);

                CREATE TABLE IF NOT EXISTS batch_items (
                    batch_id TEXT NOT NULL,
                    custom_id TEXT NOT NULL,
                    method TEXT DEFAULT 'POST',
                    url TEXT,
                    origin_task_id TEXT,
                    origin_run_id TEXT,
                    target_revision TEXT,
                    file_path TEXT,
                    expected_sha256 TEXT,
                    request_body TEXT,
                    status_code INTEGER,
                    response_body TEXT,
                    error TEXT,
                    duration_ms REAL DEFAULT 0.0,
                    PRIMARY KEY (batch_id, custom_id),
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_batch_items_custom_id ON batch_items(custom_id);
                """
            )

    def save_job(self, record: BatchJobRecord) -> None:
        """Insert or update a BatchJobRecord atomically."""
        sql = """
        INSERT INTO batches (
            batch_id, status, endpoint, model, input_file_id, output_file_id,
            error_file_id, completion_window, created_at, updated_at, completed_at,
            expires_at, total_requests, completed_requests, failed_requests,
            task_id, run_id, idempotency_key, git_commit_sha, metadata_json,
            usage_json, cost_estimate_usd, error_message
        ) VALUES (
            :batch_id, :status, :endpoint, :model, :input_file_id, :output_file_id,
            :error_file_id, :completion_window, :created_at, :updated_at, :completed_at,
            :expires_at, :total_requests, :completed_requests, :failed_requests,
            :task_id, :run_id, :idempotency_key, :git_commit_sha, :metadata_json,
            :usage_json, :cost_estimate_usd, :error_message
        )
        ON CONFLICT(batch_id) DO UPDATE SET
            status=excluded.status,
            output_file_id=coalesce(excluded.output_file_id, batches.output_file_id),
            error_file_id=coalesce(excluded.error_file_id, batches.error_file_id),
            updated_at=excluded.updated_at,
            completed_at=coalesce(excluded.completed_at, batches.completed_at),
            expires_at=coalesce(excluded.expires_at, batches.expires_at),
            total_requests=excluded.total_requests,
            completed_requests=excluded.completed_requests,
            failed_requests=excluded.failed_requests,
            usage_json=excluded.usage_json,
            cost_estimate_usd=excluded.cost_estimate_usd,
            error_message=excluded.error_message
        """
        params = {
            "batch_id": record.batch_id,
            "status": record.status,
            "endpoint": record.endpoint,
            "model": record.model,
            "input_file_id": record.input_file_id,
            "output_file_id": record.output_file_id,
            "error_file_id": record.error_file_id,
            "completion_window": record.completion_window,
            "created_at": record.created_at,
            "updated_at": time.time(),
            "completed_at": record.completed_at,
            "expires_at": record.expires_at,
            "total_requests": record.total_requests,
            "completed_requests": record.completed_requests,
            "failed_requests": record.failed_requests,
            "task_id": record.task_id,
            "run_id": record.run_id,
            "idempotency_key": record.idempotency_key,
            "git_commit_sha": record.git_commit_sha,
            "metadata_json": json.dumps(record.metadata, sort_keys=True),
            "usage_json": json.dumps(record.usage, sort_keys=True),
            "cost_estimate_usd": record.cost_estimate_usd,
            "error_message": record.error_message,
        }
        with self._get_connection() as conn:
            conn.execute(sql, params)

    def get_job(self, batch_id: str) -> BatchJobRecord | None:
        """Retrieve a BatchJobRecord by its ID."""
        sql = "SELECT * FROM batches WHERE batch_id = ?"
        with self._get_connection() as conn:
            row = conn.execute(sql, (batch_id,)).fetchone()
            if row is None:
                return None
            return self._row_to_job(row)

    def find_by_idempotency_key(self, idempotency_key: str) -> BatchJobRecord | None:
        """Find an existing batch job with the same idempotency key to prevent duplicates."""
        if not idempotency_key:
            return None
        sql = "SELECT * FROM batches WHERE idempotency_key = ? ORDER BY created_at DESC LIMIT 1"
        with self._get_connection() as conn:
            row = conn.execute(sql, (idempotency_key,)).fetchone()
            if row is None:
                return None
            return self._row_to_job(row)

    def list_jobs(self, *, status: str | None = None, task_id: str | None = None, limit: int = 50) -> list[BatchJobRecord]:
        """List recent batch jobs, optionally filtered by status and/or task_id."""
        sql = "SELECT * FROM batches"
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_job(r) for r in rows]

    def list_pending_jobs(self) -> list[BatchJobRecord]:
        """List all non-terminal batch jobs that require active synchronization."""
        terminal_statuses = [
            BatchLifecycleStatus.RESULTS_IMPORTED.value,
            BatchLifecycleStatus.VERIFIED.value,
            BatchLifecycleStatus.FAILED.value,
            BatchLifecycleStatus.EXPIRED.value,
            BatchLifecycleStatus.CANCELLED.value,
        ]
        placeholders = ",".join("?" for _ in terminal_statuses)
        sql = f"SELECT * FROM batches WHERE status NOT IN ({placeholders}) ORDER BY created_at ASC"
        with self._get_connection() as conn:
            rows = conn.execute(sql, terminal_statuses).fetchall()
            return [self._row_to_job(r) for r in rows]

    def save_items(self, batch_id: str, items: Sequence[BatchItemRequest]) -> None:
        """Store item requests for a batch."""
        sql = """
        INSERT OR REPLACE INTO batch_items (
            batch_id, custom_id, method, url, origin_task_id, origin_run_id,
            target_revision, file_path, expected_sha256, request_body
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        records = [
            (
                batch_id,
                item.custom_id,
                item.method,
                item.url,
                item.origin_task_id,
                item.origin_run_id,
                item.target_revision,
                item.file_path,
                item.expected_sha256,
                json.dumps(item.body),
            )
            for item in items
        ]
        with self._get_connection() as conn:
            conn.executemany(sql, records)

    def save_results(self, batch_id: str, results: Sequence[BatchItemResult]) -> None:
        """Store parsed results corresponding to batch items."""
        sql = """
        UPDATE batch_items SET
            status_code = ?,
            response_body = ?,
            error = ?,
            duration_ms = ?
        WHERE batch_id = ? AND custom_id = ?
        """
        records = [
            (
                res.status_code,
                json.dumps(res.response_body),
                json.dumps(res.error) if res.error is not None else None,
                res.duration_ms,
                batch_id,
                res.custom_id,
            )
            for res in results
        ]
        with self._get_connection() as conn:
            conn.executemany(sql, records)

    def get_results(self, batch_id: str) -> list[BatchItemResult]:
        """Retrieve all parsed results for a batch."""
        sql = "SELECT * FROM batch_items WHERE batch_id = ?"
        with self._get_connection() as conn:
            rows = conn.execute(sql, (batch_id,)).fetchall()
            results: list[BatchItemResult] = []
            for r in rows:
                status_code = r["status_code"]
                if status_code is None:
                    continue
                body_raw = r["response_body"]
                err_raw = r["error"]
                body = json.loads(body_raw) if body_raw else {}
                err = json.loads(err_raw) if err_raw else None
                results.append(
                    BatchItemResult(
                        custom_id=r["custom_id"],
                        status_code=int(status_code),
                        response_body=body,
                        error=err,
                        duration_ms=float(r["duration_ms"] or 0.0),
                    )
                )
            return results

    def import_results(self, job: BatchJobRecord, results: Sequence[BatchItemResult]) -> None:
        """Commit correlated item results and import status in one transaction."""
        with self._get_connection() as conn:
            for result in results:
                changed = conn.execute(
                    "UPDATE batch_items SET status_code=?, response_body=?, error=?, duration_ms=? "
                    "WHERE batch_id=? AND custom_id=?",
                    (result.status_code, json.dumps(result.response_body), json.dumps(result.error) if result.error else None,
                     result.duration_ms, job.batch_id, result.custom_id),
                ).rowcount
                if changed != 1:
                    raise ValueError("Cannot import a result without its original Batch request")
            conn.execute(
                "UPDATE batches SET status=?, usage_json=?, metadata_json=?, cost_estimate_usd=?, completed_at=?, updated_at=? WHERE batch_id=?",
                (job.status, json.dumps(job.usage), json.dumps(job.metadata), job.cost_estimate_usd, job.completed_at, time.time(), job.batch_id),
            )

    def get_items(self, batch_id: str) -> list[BatchItemRequest]:
        """Retrieve all initial requests for a batch."""
        sql = "SELECT * FROM batch_items WHERE batch_id = ?"
        with self._get_connection() as conn:
            rows = conn.execute(sql, (batch_id,)).fetchall()
            items: list[BatchItemRequest] = []
            for r in rows:
                body_raw = r["request_body"]
                body = json.loads(body_raw) if body_raw else {}
                items.append(
                    BatchItemRequest(
                        custom_id=r["custom_id"],
                        method=r["method"] or "POST",
                        url=r["url"] or "/v1/responses",
                        origin_task_id=r["origin_task_id"] or "",
                        origin_run_id=r["origin_run_id"] or "",
                        target_revision=r["target_revision"] or "",
                        file_path=r["file_path"] or "",
                        expected_sha256=r["expected_sha256"] or "",
                        body=body,
                    )
                )
            return items

    def _row_to_job(self, r: sqlite3.Row) -> BatchJobRecord:
        meta_raw = r["metadata_json"]
        usage_raw = r["usage_json"]
        return BatchJobRecord(
            batch_id=r["batch_id"],
            status=r["status"],
            endpoint=r["endpoint"],
            model=r["model"],
            input_file_id=r["input_file_id"] or "",
            output_file_id=r["output_file_id"] or "",
            error_file_id=r["error_file_id"] or "",
            completion_window=r["completion_window"] or "24h",
            created_at=float(r["created_at"]),
            updated_at=float(r["updated_at"]),
            completed_at=float(r["completed_at"]) if r["completed_at"] is not None else None,
            expires_at=float(r["expires_at"]) if r["expires_at"] is not None else None,
            total_requests=int(r["total_requests"] or 0),
            completed_requests=int(r["completed_requests"] or 0),
            failed_requests=int(r["failed_requests"] or 0),
            task_id=r["task_id"] or "",
            run_id=r["run_id"] or "",
            idempotency_key=r["idempotency_key"] or "",
            git_commit_sha=r["git_commit_sha"] or "",
            metadata=json.loads(meta_raw) if meta_raw else {},
            usage=json.loads(usage_raw) if usage_raw else {},
            cost_estimate_usd=float(r["cost_estimate_usd"] or 0.0),
            error_message=r["error_message"] or "",
        )
