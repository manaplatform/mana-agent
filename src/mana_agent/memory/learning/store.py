"""Persistent SQLite storage for the Global Self-Learning Memory System."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from mana_agent.memory.learning.models import (
    LearningCategory,
    LearningHistoryEntry,
    LearningMemory,
    LearningScope,
    LearningStatus,
    utc_now,
)
from mana_agent.workspaces.paths import mana_home

logger = logging.getLogger(__name__)


def default_learning_db_path() -> Path:
    return mana_home() / "memory" / "global_learning.sqlite3"


class LearningMemoryStore:
    """ACID SQLite store for persistent self-learning memory records."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or default_learning_db_path()).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        with self._lock:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=10.0,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA foreign_keys=ON;")
                conn.execute("PRAGMA busy_timeout=5000;")
                with conn:
                    yield conn
            finally:
                conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS learning_memories (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    context TEXT NOT NULL,
                    attempted_action TEXT NOT NULL,
                    failure_signature TEXT,
                    error TEXT NOT NULL,
                    root_cause TEXT,
                    correction TEXT,
                    prevention_rule TEXT,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_execution_id TEXT NOT NULL,
                    evidence_references_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_applied_at TEXT,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 1,
                    project_id TEXT,
                    environment_fingerprint TEXT,
                    dependency_versions_json TEXT NOT NULL,
                    ownership TEXT,
                    provenance_json TEXT NOT NULL,
                    expires_at TEXT,
                    superseded_by TEXT,
                    replaces_memory_id TEXT,
                    is_hypothesis INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS learning_memory_history (
                    id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    execution_id TEXT,
                    evidence TEXT,
                    previous_confidence REAL,
                    new_confidence REAL,
                    details_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_learning_category_scope ON learning_memories(category, scope);
                CREATE INDEX IF NOT EXISTS idx_learning_failure_sig ON learning_memories(failure_signature);
                CREATE INDEX IF NOT EXISTS idx_learning_project ON learning_memories(project_id);
                CREATE INDEX IF NOT EXISTS idx_learning_status ON learning_memories(status);
                CREATE INDEX IF NOT EXISTS idx_learning_updated ON learning_memories(updated_at);
                CREATE INDEX IF NOT EXISTS idx_learning_history_mem ON learning_memory_history(memory_id);
                """
            )

    def _row_to_memory(self, row: sqlite3.Row) -> LearningMemory:
        return LearningMemory(
            id=str(row["id"]),
            category=LearningCategory(row["category"]),
            scope=LearningScope(row["scope"]),
            context=str(row["context"]),
            attempted_action=str(row["attempted_action"]),
            failure_signature=str(row["failure_signature"]) if row["failure_signature"] is not None else None,
            error=str(row["error"]),
            root_cause=str(row["root_cause"]) if row["root_cause"] is not None else None,
            correction=str(row["correction"]) if row["correction"] is not None else None,
            prevention_rule=str(row["prevention_rule"]) if row["prevention_rule"] is not None else None,
            status=LearningStatus(row["status"]),
            confidence=float(row["confidence"]),
            source_execution_id=str(row["source_execution_id"] or ""),
            evidence_references=json.loads(row["evidence_references_json"] or "[]"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_applied_at=datetime.fromisoformat(row["last_applied_at"]) if row["last_applied_at"] else None,
            success_count=int(row["success_count"]),
            failure_count=int(row["failure_count"]),
            project_id=str(row["project_id"]) if row["project_id"] is not None else None,
            environment_fingerprint=str(row["environment_fingerprint"]) if row["environment_fingerprint"] is not None else None,
            dependency_versions=json.loads(row["dependency_versions_json"] or "{}"),
            ownership=str(row["ownership"]) if row["ownership"] is not None else None,
            provenance=json.loads(row["provenance_json"] or "{}"),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            superseded_by=str(row["superseded_by"]) if row["superseded_by"] is not None else None,
            replaces_memory_id=str(row["replaces_memory_id"]) if row["replaces_memory_id"] is not None else None,
            is_hypothesis=bool(row["is_hypothesis"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def save(self, memory: LearningMemory) -> LearningMemory:
        now = utc_now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO learning_memories (
                    id, category, scope, context, attempted_action, failure_signature,
                    error, root_cause, correction, prevention_rule, status, confidence,
                    source_execution_id, evidence_references_json, created_at, updated_at,
                    last_applied_at, success_count, failure_count, project_id,
                    environment_fingerprint, dependency_versions_json, ownership,
                    provenance_json, expires_at, superseded_by, replaces_memory_id,
                    is_hypothesis, metadata_json
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(id) DO UPDATE SET
                    category = excluded.category,
                    scope = excluded.scope,
                    context = excluded.context,
                    attempted_action = excluded.attempted_action,
                    failure_signature = excluded.failure_signature,
                    error = excluded.error,
                    root_cause = excluded.root_cause,
                    correction = excluded.correction,
                    prevention_rule = excluded.prevention_rule,
                    status = excluded.status,
                    confidence = excluded.confidence,
                    source_execution_id = excluded.source_execution_id,
                    evidence_references_json = excluded.evidence_references_json,
                    updated_at = excluded.updated_at,
                    last_applied_at = excluded.last_applied_at,
                    success_count = excluded.success_count,
                    failure_count = excluded.failure_count,
                    project_id = excluded.project_id,
                    environment_fingerprint = excluded.environment_fingerprint,
                    dependency_versions_json = excluded.dependency_versions_json,
                    ownership = excluded.ownership,
                    provenance_json = excluded.provenance_json,
                    expires_at = excluded.expires_at,
                    superseded_by = excluded.superseded_by,
                    replaces_memory_id = excluded.replaces_memory_id,
                    is_hypothesis = excluded.is_hypothesis,
                    metadata_json = excluded.metadata_json;
                """,
                (
                    memory.id,
                    memory.category.value,
                    memory.scope.value,
                    memory.context,
                    memory.attempted_action,
                    memory.failure_signature,
                    memory.error,
                    memory.root_cause,
                    memory.correction,
                    memory.prevention_rule,
                    memory.status.value,
                    memory.confidence,
                    memory.source_execution_id,
                    json.dumps(memory.evidence_references),
                    memory.created_at.isoformat(),
                    now,
                    memory.last_applied_at.isoformat() if memory.last_applied_at else None,
                    memory.success_count,
                    memory.failure_count,
                    memory.project_id,
                    memory.environment_fingerprint,
                    json.dumps(memory.dependency_versions),
                    memory.ownership,
                    json.dumps(memory.provenance),
                    memory.expires_at.isoformat() if memory.expires_at else None,
                    memory.superseded_by,
                    memory.replaces_memory_id,
                    1 if memory.is_hypothesis else 0,
                    json.dumps(memory.metadata),
                ),
            )
        return self.get(memory.id) or memory

    def get(self, memory_id: str) -> LearningMemory | None:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM learning_memories WHERE id = ?;",
                (memory_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_memory(row)

    def find_by_signature(
        self,
        signature: str,
        *,
        project_id: str | None = None,
        include_superseded: bool = False,
    ) -> list[LearningMemory]:
        if not signature:
            return []
        query = "SELECT * FROM learning_memories WHERE failure_signature = ?"
        params: list[Any] = [signature]
        if not include_superseded:
            query += " AND status != 'SUPERSEDED'"
        if project_id:
            query += " AND (project_id = ? OR scope = 'GLOBAL')"
            params.append(project_id)

        query += " ORDER BY confidence DESC, updated_at DESC"
        with self._connect() as conn:
            cursor = conn.execute(query, tuple(params))
            return [self._row_to_memory(row) for row in cursor.fetchall()]

    def find_by_action_exact(
        self,
        attempted_action: str,
        *,
        project_id: str | None = None,
        category: LearningCategory | str | None = None,
        include_superseded: bool = False,
    ) -> list[LearningMemory]:
        clean_action = attempted_action.strip()
        if not clean_action:
            return []
        query = "SELECT * FROM learning_memories WHERE attempted_action = ?"
        params: list[Any] = [clean_action]
        if not include_superseded:
            query += " AND status != 'SUPERSEDED'"
        if category:
            cat_val = category.value if isinstance(category, LearningCategory) else str(category)
            query += " AND category = ?"
            params.append(cat_val)
        if project_id:
            query += " AND (project_id = ? OR scope = 'GLOBAL')"
            params.append(project_id)

        query += " ORDER BY confidence DESC, updated_at DESC"
        with self._connect() as conn:
            cursor = conn.execute(query, tuple(params))
            return [self._row_to_memory(row) for row in cursor.fetchall()]

    def search(
        self,
        query: str,
        *,
        category: LearningCategory | str | None = None,
        scope: LearningScope | str | None = None,
        project_id: str | None = None,
        environment_fingerprint: str | None = None,
        status: LearningStatus | str | None = None,
        active_only: bool = True,
        limit: int = 10,
    ) -> list[LearningMemory]:
        clauses: list[str] = []
        params: list[Any] = []

        if active_only:
            clauses.append("status != 'SUPERSEDED' AND status != 'DISPUTED'")
        elif status:
            stat_val = status.value if isinstance(status, LearningStatus) else str(status)
            clauses.append("status = ?")
            params.append(stat_val)

        if category:
            cat_val = category.value if isinstance(category, LearningCategory) else str(category)
            clauses.append("category = ?")
            params.append(cat_val)

        if scope:
            scp_val = scope.value if isinstance(scope, LearningScope) else str(scope)
            clauses.append("scope = ?")
            params.append(scp_val)
        elif project_id:
            clauses.append("(project_id = ? OR scope = 'GLOBAL')")
            params.append(project_id)

        if environment_fingerprint:
            clauses.append("(environment_fingerprint = ? OR environment_fingerprint IS NULL)")
            params.append(environment_fingerprint)

        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM learning_memories{where_sql} ORDER BY confidence DESC, updated_at DESC"

        with self._connect() as conn:
            cursor = conn.execute(sql, tuple(params))
            rows = cursor.fetchall()

        memories = [self._row_to_memory(r) for r in rows]

        clean_query = query.strip().lower()
        if not clean_query:
            return memories[: max(1, limit)]

        tokens = set(clean_query.split())

        def score(item: LearningMemory) -> float:
            target_text = f"{item.attempted_action} {item.error} {item.root_cause or ''} {item.correction or ''} {item.prevention_rule or ''} {item.context}".lower()
            if clean_query in target_text:
                base = 2.0
            else:
                target_tokens = set(target_text.split())
                overlap = len(tokens & target_tokens)
                base = overlap / max(1, len(tokens))
            return base + (item.confidence * 0.5)

        scored = [(score(m), m) for m in memories]
        scored = [pair for pair in scored if pair[0] > 0.1]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [pair[1] for pair in scored[: max(1, limit)]]

    def delete(self, memory_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM learning_memories WHERE id = ?;", (memory_id,))
            conn.execute("DELETE FROM learning_memory_history WHERE memory_id = ?;", (memory_id,))
            return cursor.rowcount > 0

    def record_history(self, entry: LearningHistoryEntry) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO learning_memory_history (
                    id, memory_id, action, timestamp, execution_id, evidence,
                    previous_confidence, new_confidence, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    entry.id,
                    entry.memory_id,
                    entry.action,
                    entry.timestamp.isoformat(),
                    entry.execution_id,
                    entry.evidence,
                    entry.previous_confidence,
                    entry.new_confidence,
                    json.dumps(entry.details),
                ),
            )

    def get_history(self, memory_id: str) -> list[LearningHistoryEntry]:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM learning_memory_history WHERE memory_id = ? ORDER BY timestamp ASC;",
                (memory_id,),
            )
            rows = cursor.fetchall()
            return [
                LearningHistoryEntry(
                    id=str(r["id"]),
                    memory_id=str(r["memory_id"]),
                    action=str(r["action"]),
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    execution_id=str(r["execution_id"] or ""),
                    evidence=str(r["evidence"] or ""),
                    previous_confidence=float(r["previous_confidence"] or 0.0),
                    new_confidence=float(r["new_confidence"] or 0.0),
                    details=json.loads(r["details_json"] or "{}"),
                )
                for r in rows
            ]

    def prune(self, max_records: int = 2000) -> int:
        """Prune expired and superseded memories when storage bounds are reached."""
        with self._connect() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM learning_memories;")
            total = int(cursor.fetchone()["count"])
            if total <= max_records:
                return 0

            # Delete superseded and expired items first
            now_iso = utc_now().isoformat()
            del_cursor = conn.execute(
                """
                DELETE FROM learning_memories
                WHERE status = 'SUPERSEDED'
                   OR (expires_at IS NOT NULL AND expires_at <= ?)
                   OR (status = 'DISPUTED' AND confidence < 0.2);
                """,
                (now_iso,),
            )
            pruned = del_cursor.rowcount
            return pruned
