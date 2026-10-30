"""Storage and fenced admission groundwork for future live steering.

The standalone ledger does not validate leases. The coordinator validates
lifecycle state but neither authenticates a sender nor delivers instructions.
Callers must not treat persistence as application by a running execution.
"""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from pydantic import Field, model_validator

from mana_agent.execution_supervisor.models import StrictModel

if TYPE_CHECKING:
    from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor


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
    requires_approval: bool = False

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


class SteeringCoordinator:
    """Admission groundwork, not a delivery or authorization endpoint.

    The caller must authenticate the session independently. The supervisor's
    atomic update serializes admission against lease replacement and cancellation.
    No consumer is wired yet; an admitted record is only pending, not applied.
    """

    MAX_PENDING = 1000

    def __init__(self, supervisor: "ExecutionSupervisor") -> None:
        self.supervisor = supervisor

    @staticmethod
    def _validate_execution_id(execution_id: str) -> None:
        if (not execution_id or execution_id != execution_id.strip() or len(execution_id) > 256
                or execution_id in {".", ".."}
                or any(c in execution_id for c in ("/", "\\", ":", "\x00"))):
            raise ValueError("invalid execution ID")

    def pending_for_attempt(
        self,
        execution_id: str,
        *,
        attempt_id: str,
        generation: int,
        lease_token: str,
        scope: SteeringScope | str | None = None,
        target_id: str | None = None,
    ) -> list[dict]:
        """Read pending records under the current worker's lease fence.

        This is a snapshot, not a claim or acknowledgement. The consumer must
        revalidate its fence when committing any decision based on the snapshot.
        Records from an earlier attempt are deliberately not routed to a new one.
        """
        from mana_agent.execution_supervisor.errors import LeaseConflictError, StaleLeaseError
        from mana_agent.execution_supervisor.models import ExecutionState

        self._validate_execution_id(execution_id)
        if not lease_token:
            raise StaleLeaseError("steering read requires the worker's lease token")
        token_hash = "sha256:" + hashlib.sha256(lease_token.encode("utf-8")).hexdigest()
        with self.supervisor.store.locked():
            task = self.supervisor.store.get_task(execution_id)
            if (task.attempt_id != attempt_id or task.attempt_generation != generation
                    or not hmac.compare_digest(task.lease_token, token_hash)):
                raise StaleLeaseError("steering read targets a retired or unauthorized attempt")
            now = self.supervisor.clock()
            if (task.state != ExecutionState.RUNNING or task.cancellation_status is not None
                    or task.lease_expires_at is None or task.lease_expires_at <= now
                    or task.wall_clock_deadline_exceeded(now)):
                raise LeaseConflictError("execution is not accepting steering reads")
            scope_val = scope.value if isinstance(scope, SteeringScope) else (scope or "")
            results = []
            for item in task.steering_inputs:
                if item.get("status") != "pending":
                    continue
                if item.get("attempt_id") != attempt_id or item.get("generation") != generation:
                    continue
                item_scope = item.get("scope", "execution")
                item_target = item.get("target_id", "")
                if scope_val:
                    if item_scope == "execution":
                        results.append(dict(item))
                    elif item_scope == scope_val and (not target_id or item_target == target_id):
                        results.append(dict(item))
                else:
                    results.append(dict(item))
            return results

    def apply_steering(
        self,
        execution_id: str,
        *,
        attempt_id: str,
        generation: int,
        lease_token: str,
        request_ids: list[str],
    ) -> list[dict]:
        """Atomically transition pending instructions to applied under the supervisor fence."""
        from mana_agent.execution_supervisor.errors import LeaseConflictError, StaleLeaseError
        from mana_agent.execution_supervisor.models import ExecutionState

        self._validate_execution_id(execution_id)
        if not lease_token:
            raise StaleLeaseError("steering apply requires the worker's lease token")
        token_hash = "sha256:" + hashlib.sha256(lease_token.encode("utf-8")).hexdigest()
        applied_items: list[dict] = []

        def update(task):
            if (
                task.attempt_id != attempt_id
                or task.attempt_generation != generation
                or not hmac.compare_digest(task.lease_token, token_hash)
            ):
                raise StaleLeaseError("steering apply targets a retired or unauthorized attempt")
            now = self.supervisor.clock()
            if (
                task.state != ExecutionState.RUNNING
                or task.cancellation_status is not None
                or task.lease_expires_at is None
                or task.lease_expires_at <= now
                or task.wall_clock_deadline_exceeded(now)
            ):
                raise LeaseConflictError("execution is not accepting steering apply")
            target_set = set(request_ids)
            for item in task.steering_inputs:
                if item.get("request_id") in target_set and item.get("status") == "pending":
                    item["status"] = "applied"
                    item["applied_at"] = now.isoformat()
                    applied_items.append(dict(item))
            task.updated_at = now

        self.supervisor.store.update_task(execution_id, update)
        return applied_items

    def reject_steering(
        self,
        execution_id: str,
        *,
        attempt_id: str,
        generation: int,
        lease_token: str,
        request_id: str,
        reason: str,
    ) -> dict | None:
        """Atomically mark a pending instruction rejected under the supervisor fence."""
        from mana_agent.execution_supervisor.errors import LeaseConflictError, StaleLeaseError
        from mana_agent.execution_supervisor.models import ExecutionState

        self._validate_execution_id(execution_id)
        if not lease_token:
            raise StaleLeaseError("steering reject requires the worker's lease token")
        token_hash = "sha256:" + hashlib.sha256(lease_token.encode("utf-8")).hexdigest()
        rejected_item: dict | None = None

        def update(task):
            nonlocal rejected_item
            if (
                task.attempt_id != attempt_id
                or task.attempt_generation != generation
                or not hmac.compare_digest(task.lease_token, token_hash)
            ):
                raise StaleLeaseError("steering reject targets a retired or unauthorized attempt")
            now = self.supervisor.clock()
            for item in task.steering_inputs:
                if item.get("request_id") == request_id and item.get("status") == "pending":
                    item["status"] = "rejected"
                    item["rejected_at"] = now.isoformat()
                    item["rejection_reason"] = reason
                    rejected_item = dict(item)
                    break
            task.updated_at = now

        self.supervisor.store.update_task(execution_id, update)
        return rejected_item

    def supersede_steering(
        self,
        execution_id: str,
        *,
        attempt_id: str,
        generation: int,
        lease_token: str,
        request_ids: list[str],
    ) -> list[dict]:
        """Atomically mark pending instructions superseded under the supervisor fence."""
        from mana_agent.execution_supervisor.errors import LeaseConflictError, StaleLeaseError
        from mana_agent.execution_supervisor.models import ExecutionState

        self._validate_execution_id(execution_id)
        if not lease_token:
            raise StaleLeaseError("steering supersede requires the worker's lease token")
        token_hash = "sha256:" + hashlib.sha256(lease_token.encode("utf-8")).hexdigest()
        superseded_items: list[dict] = []

        def update(task):
            if (
                task.attempt_id != attempt_id
                or task.attempt_generation != generation
                or not hmac.compare_digest(task.lease_token, token_hash)
            ):
                raise StaleLeaseError("steering supersede targets a retired or unauthorized attempt")
            now = self.supervisor.clock()
            target_set = set(request_ids)
            for item in task.steering_inputs:
                if item.get("request_id") in target_set and item.get("status") == "pending":
                    item["status"] = "superseded"
                    item["superseded_at"] = now.isoformat()
                    superseded_items.append(dict(item))
            task.updated_at = now

        self.supervisor.store.update_task(execution_id, update)
        return superseded_items

    def submit(
        self,
        submission: SteeringSubmission,
        *,
        authenticated_session_id: str,
        allow_scoped: bool = False,
        allow_interrupt: bool = False,
    ) -> tuple[int, bool]:
        from mana_agent.execution_supervisor.errors import LeaseConflictError, StaleLeaseError
        from mana_agent.execution_supervisor.models import ExecutionState

        if not authenticated_session_id or authenticated_session_id != submission.session_id:
            raise PermissionError("steering requires the authenticated owning session")
        self._validate_execution_id(submission.execution_id)
        if not allow_scoped and submission.scope != SteeringScope.EXECUTION:
            raise ValueError("steering scope is not supported by admission")
        if not allow_interrupt and submission.kind == SteeringKind.INTERRUPT:
            raise ValueError("steering interrupt is not supported by admission")
        digest = hashlib.sha256(submission.model_dump_json().encode("utf-8")).hexdigest()

        def admit(task):
            if not task.session_id or task.session_id != authenticated_session_id:
                raise PermissionError("execution does not belong to this session")
            if task.attempt_id != submission.attempt_id or task.attempt_generation != submission.generation:
                raise StaleLeaseError("steering targets a retired attempt")
            if (task.state != ExecutionState.RUNNING or task.cancellation_status is not None
                    or not task.lease_token or task.lease_expires_at is None
                    or task.lease_expires_at <= self.supervisor.clock()
                    or task.wall_clock_deadline_exceeded(self.supervisor.clock())):
                raise LeaseConflictError("execution is not accepting steering")
            for item in task.steering_inputs:
                if item["request_id"] == submission.request_id:
                    if item["digest"] != digest:
                        raise ValueError("request_id already belongs to a different steering submission")
                    return item["sequence"], True
            if len(task.steering_inputs) >= self.MAX_PENDING:
                raise ValueError("steering admission limit reached")
            sequence = len(task.steering_inputs) + 1
            task.steering_inputs.append({
                "sequence": sequence, "request_id": submission.request_id,
                "digest": digest, "attempt_id": submission.attempt_id,
                "generation": submission.generation, "kind": submission.kind.value,
                "scope": submission.scope.value, "target_id": submission.target_id,
                "text": submission.text,
                "requires_approval": submission.requires_approval,
                "status": "pending",
                "received_at": self.supervisor.clock().isoformat(),
            })
            return sequence, False

        _, result = self.supervisor.store.update_task(submission.execution_id, admit)
        return result
