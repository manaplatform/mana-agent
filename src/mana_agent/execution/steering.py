"""Durable input ledger for future live steering integration.

This is only a storage primitive: it does not authorize a sender, validate an
execution lease, apply instructions, or publish them to an agent. Callers must
not treat persistence as acceptance by a running execution.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Iterator

from pydantic import Field, model_validator

from mana_agent.execution_supervisor.models import StrictModel


class SteeringKind(str, Enum):
    GUIDANCE = "guidance"
    CONSTRAINT = "constraint"
    PRIORITY = "priority"
    CORRECTION = "correction"
    SCOPE_CHANGE = "scope_change"
    REPLAN = "replan"
    INTERRUPT = "interrupt"


class SteeringScope(str, Enum):
    EXECUTION = "execution"
    TASK = "task"
    WORKER = "worker"


class SteeringSubmission(StrictModel):
    execution_id: str = Field(min_length=1, max_length=256)
    session_id: str = Field(min_length=1, max_length=256)
    attempt_id: str = Field(min_length=1, max_length=256)
    generation: int = Field(ge=1)
    request_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=16000)
    kind: SteeringKind
    scope: SteeringScope = SteeringScope.EXECUTION
    target_id: str = Field(default="", max_length=256)

    @model_validator(mode="after")
    def validate_scope(self) -> "SteeringSubmission":
        if not self.text.strip():
            raise ValueError("steering text must not be blank")
        for value in (self.execution_id, self.session_id, self.attempt_id, self.request_id):
            if not value.strip() or value != value.strip():
                raise ValueError("identifiers must be nonblank and have no surrounding whitespace")
        if self.scope == SteeringScope.EXECUTION and self.target_id:
            raise ValueError("execution scope must not specify a target")
        if self.scope != SteeringScope.EXECUTION and (
            not self.target_id.strip() or self.target_id != self.target_id.strip()
        ):
            raise ValueError("task and worker scope require a target")
        return self


class SteeringEntry(SteeringSubmission):
    sequence: int = Field(ge=1)


class SteeringLedger:
    """Ordered, idempotent storage; never an execution or authorization authority.

    The caller must place the database in a private session directory; payloads
    contain user text. SQLite serializes writers even across processes. A request ID is unique
    within an execution and retrying it with a different payload fails closed.
    This ledger deliberately has no apply/ack method: changing status and
    checkpointing must be coordinated with supervisor fencing first.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS steering_inputs (
                    execution_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    request_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (execution_id, sequence),
                    UNIQUE (execution_id, request_id)
                )"""
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout = 30000")
            with conn:
                yield conn
        finally:
            conn.close()

    def append(self, submission: SteeringSubmission) -> tuple[SteeringEntry, bool]:
        """Persist a submission and return (entry, was_duplicate)."""
        payload = submission.model_dump_json()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT sequence, payload FROM steering_inputs WHERE execution_id = ? AND request_id = ?",
                (submission.execution_id, submission.request_id),
            ).fetchone()
            if row is not None:
                original = SteeringSubmission.model_validate_json(row[1])
                if original != submission:
                    raise ValueError("request_id already belongs to a different steering submission")
                return SteeringEntry(**original.model_dump(), sequence=row[0]), True
            sequence = conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM steering_inputs WHERE execution_id = ?",
                (submission.execution_id,),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO steering_inputs (execution_id, sequence, request_id, payload) VALUES (?, ?, ?, ?)",
                (submission.execution_id, sequence, submission.request_id, payload),
            )
            return SteeringEntry(**submission.model_dump(), sequence=sequence), False

    def list_entries(self, execution_id: str) -> list[SteeringEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT sequence, payload FROM steering_inputs WHERE execution_id = ? ORDER BY sequence",
                (execution_id,),
            ).fetchall()
        return [
            SteeringEntry(**SteeringSubmission.model_validate_json(payload).model_dump(), sequence=sequence)
            for sequence, payload in rows
        ]
