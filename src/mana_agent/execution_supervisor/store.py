"""Atomic local persistence for supervised executions.

The interface deliberately exposes domain objects rather than filesystem paths so
a transactional database backend can implement the same contract later.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Protocol, TypeVar

from mana_agent.execution_supervisor.errors import ConcurrentUpdateError, TaskNotFoundError
from mana_agent.execution_supervisor.models import (
    AttemptRecord,
    CheckpointRecord,
    EscrowResult,
    ExecutionEvent,
    TaskRecord,
)
from mana_agent.utils.redaction import redact_secrets

if os.name == "nt":  # pragma: no cover - Windows CI
    import msvcrt
else:  # pragma: no cover - platform branch
    import fcntl

T = TypeVar("T")
_SAFE_TOKEN_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
# Bind the primitive at module import so tests or platform adapters that patch
# another module's ``os.replace`` do not accidentally intercept supervisor
# persistence. Supervisor-specific fault injection patches this callable.
_atomic_replace = os.replace


def _redact_for_persistence(payload):
    safe = redact_secrets(payload)

    def restore_hashes(original, redacted) -> None:
        if isinstance(original, dict) and isinstance(redacted, dict):
            for key, value in original.items():
                if (
                    key in {"lease_token", "lease_token_hash", "idempotency_key"}
                    and isinstance(value, str)
                    and _SAFE_TOKEN_HASH.fullmatch(value)
                ):
                    redacted[key] = value
                elif key in redacted:
                    restore_hashes(value, redacted[key])
        elif isinstance(original, list) and isinstance(redacted, list):
            for source, target in zip(original, redacted):
                restore_hashes(source, target)

    restore_hashes(payload, safe)
    return safe


class ExecutionStore(Protocol):
    def create_task(self, task: TaskRecord) -> TaskRecord: ...
    def get_task(self, task_id: str) -> TaskRecord: ...
    def get_task_or_none(self, task_id: str) -> TaskRecord | None: ...
    def list_tasks(self, *, incomplete_only: bool = False) -> list[TaskRecord]: ...
    def list_expired(self, now: datetime) -> list[TaskRecord]: ...
    def compare_and_set(self, task: TaskRecord, expected_version: int) -> TaskRecord: ...
    def update_task(
        self, task_id: str, updater: Callable[[TaskRecord], T]
    ) -> tuple[TaskRecord, T]: ...
    def update_task_and_attempt(
        self, task_id: str, updater: Callable[[TaskRecord], AttemptRecord]
    ) -> tuple[TaskRecord, AttemptRecord]: ...
    def update_task_and_checkpoint(
        self, task_id: str, updater: Callable[[TaskRecord], CheckpointRecord]
    ) -> tuple[TaskRecord, CheckpointRecord]: ...
    def update_task_and_result(
        self, task_id: str, updater: Callable[[TaskRecord], EscrowResult]
    ) -> tuple[TaskRecord, EscrowResult]: ...
    def save_attempt(self, attempt: AttemptRecord) -> None: ...
    def get_attempt(self, attempt_id: str) -> AttemptRecord | None: ...
    def save_checkpoint(self, checkpoint: CheckpointRecord) -> None: ...
    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None: ...
    def checkpoints_for_task(self, task_id: str) -> list[CheckpointRecord]: ...
    def save_result(self, result: EscrowResult) -> None: ...
    def get_result(self, result_id: str) -> EscrowResult | None: ...
    def results_for_task(self, task_id: str) -> list[EscrowResult]: ...
    def unacknowledged_results(self, parent_task_id: str) -> list[EscrowResult]: ...
    def save_artifact_manifest(self, task_id: str, payload: dict) -> None: ...
    def artifact_manifest(self, task_id: str) -> dict | None: ...
    def append_event(self, event: ExecutionEvent) -> None: ...
    def events_for_task(self, task_id: str, *, limit: int = 1000) -> list[dict]: ...


def _acquire_file_lock(handle) -> None:
    if os.name == "nt":  # pragma: no cover - Windows CI
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release_file_lock(handle) -> None:
    if os.name == "nt":  # pragma: no cover - Windows CI
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class LocalExecutionStore:
    """Cross-process, atomic JSON storage rooted below ``~/.mana/execution``."""

    _DIRECTORIES = ("tasks", "attempts", "checkpoints", "results", "artefacts", "events", "logs")

    def __init__(self, root: Path, *, max_log_bytes: int = 10 * 1024 * 1024) -> None:
        self.root = root.expanduser().resolve()
        self.max_log_bytes = max(4096, int(max_log_bytes))
        for name in self._DIRECTORIES:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()
        self._lock_path = self.root / ".store.lock"
        self._lock_path.touch(exist_ok=True)

    @contextmanager
    def locked(self) -> Iterator[None]:
        with self._thread_lock, self._lock_path.open("r+b") as handle:
            _acquire_file_lock(handle)
            try:
                yield
            finally:
                _release_file_lock(handle)

    @staticmethod
    def _atomic_write(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(_redact_for_persistence(payload), stream, sort_keys=True, ensure_ascii=False, default=str)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            for attempt in range(6):
                try:
                    _atomic_replace(temporary, path)
                    break
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(0.01 * (2**attempt))
            if os.name != "nt":  # directory fsync makes the rename crash-durable
                directory_descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)

    def _task_path(self, task_id: str) -> Path:
        return self.root / "tasks" / f"{task_id}.json"

    def create_task(self, task: TaskRecord) -> TaskRecord:
        with self.locked():
            path = self._task_path(task.task_id)
            if path.exists():
                existing = TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))
                if existing.model_dump(mode="json") == task.model_dump(mode="json"):
                    return existing
                raise ConcurrentUpdateError(f"execution task already exists: {task.task_id}")
            self._atomic_write(path, task.model_dump(mode="json"))
        return task

    def get_task(self, task_id: str) -> TaskRecord:
        path = self._task_path(task_id)
        if not path.is_file():
            raise TaskNotFoundError(f"unknown execution task: {task_id}")
        try:
            return TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise TaskNotFoundError(f"execution task is unreadable: {task_id}") from exc

    def get_task_or_none(self, task_id: str) -> TaskRecord | None:
        try:
            return self.get_task(task_id)
        except TaskNotFoundError:
            return None

    def list_tasks(self, *, incomplete_only: bool = False) -> list[TaskRecord]:
        from mana_agent.execution_supervisor.models import TERMINAL_STATES

        rows: list[TaskRecord] = []
        for path in sorted((self.root / "tasks").glob("*.json")):
            try:
                task = TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise ConcurrentUpdateError(
                    f"execution task record is unreadable; recovery stopped: {path.name}"
                ) from exc
            if not incomplete_only or task.state not in TERMINAL_STATES:
                rows.append(task)
        return sorted(rows, key=lambda item: (item.created_at, item.task_id))

    def list_expired(self, now: datetime) -> list[TaskRecord]:
        return [
            task
            for task in self.list_tasks(incomplete_only=True)
            if task.lease_expires_at is not None and task.lease_expires_at <= now
        ]

    def compare_and_set(self, task: TaskRecord, expected_version: int) -> TaskRecord:
        with self.locked():
            current = self.get_task(task.task_id)
            if current.state_version != expected_version:
                raise ConcurrentUpdateError(
                    f"task {task.task_id} changed concurrently: expected version "
                    f"{expected_version}, found {current.state_version}"
                )
            task.state_version = expected_version + 1
            self._atomic_write(self._task_path(task.task_id), task.model_dump(mode="json"))
        return task

    def update_task(self, task_id: str, updater: Callable[[TaskRecord], T]) -> tuple[TaskRecord, T]:
        with self.locked():
            task = self.get_task(task_id)
            version = task.state_version
            result = updater(task)
            task.state_version = version + 1
            self._atomic_write(self._task_path(task_id), task.model_dump(mode="json"))
            return task, result

    def update_task_and_attempt(
        self, task_id: str, updater: Callable[[TaskRecord], AttemptRecord]
    ) -> tuple[TaskRecord, AttemptRecord]:
        with self.locked():
            task = self.get_task(task_id)
            version = task.state_version
            attempt = updater(task)
            task.state_version = version + 1
            self._atomic_write(
                self.root / "attempts" / f"{attempt.attempt_id}.json",
                attempt.model_dump(mode="json"),
            )
            self._atomic_write(self._task_path(task_id), task.model_dump(mode="json"))
            return task, attempt

    def update_task_and_checkpoint(
        self, task_id: str, updater: Callable[[TaskRecord], CheckpointRecord]
    ) -> tuple[TaskRecord, CheckpointRecord]:
        with self.locked():
            task = self.get_task(task_id)
            version = task.state_version
            checkpoint = updater(task)
            task.state_version = version + 1
            self._atomic_write(
                self.root / "checkpoints" / f"{checkpoint.checkpoint_id}.json",
                checkpoint.model_dump(mode="json"),
            )
            self._atomic_write(self._task_path(task_id), task.model_dump(mode="json"))
            return task, checkpoint

    def update_task_and_result(
        self, task_id: str, updater: Callable[[TaskRecord], EscrowResult]
    ) -> tuple[TaskRecord, EscrowResult]:
        with self.locked():
            task = self.get_task(task_id)
            version = task.state_version
            result = updater(task)
            task.state_version = version + 1
            # Result escrow is durable before the task advertises result availability.
            self._atomic_write(
                self.root / "results" / f"{result.result_id}.json",
                result.model_dump(mode="json"),
            )
            self._atomic_write(self._task_path(task_id), task.model_dump(mode="json"))
            return task, result

    def save_attempt(self, attempt: AttemptRecord) -> None:
        with self.locked():
            self._atomic_write(
                self.root / "attempts" / f"{attempt.attempt_id}.json",
                attempt.model_dump(mode="json"),
            )

    def get_attempt(self, attempt_id: str) -> AttemptRecord | None:
        path = self.root / "attempts" / f"{attempt_id}.json"
        if not path.is_file():
            return None
        try:
            return AttemptRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def save_checkpoint(self, checkpoint: CheckpointRecord) -> None:
        with self.locked():
            self._atomic_write(
                self.root / "checkpoints" / f"{checkpoint.checkpoint_id}.json",
                checkpoint.model_dump(mode="json"),
            )

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        path = self.root / "checkpoints" / f"{checkpoint_id}.json"
        if not path.is_file():
            return None
        try:
            return CheckpointRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def checkpoints_for_task(self, task_id: str) -> list[CheckpointRecord]:
        rows: list[CheckpointRecord] = []
        for path in (self.root / "checkpoints").glob("*.json"):
            try:
                item = CheckpointRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if item.task_id == task_id:
                rows.append(item)
        return sorted(rows, key=lambda item: (item.created_at, item.checkpoint_id))

    def save_result(self, result: EscrowResult) -> None:
        with self.locked():
            self._atomic_write(
                self.root / "results" / f"{result.result_id}.json",
                result.model_dump(mode="json"),
            )

    def get_result(self, result_id: str) -> EscrowResult | None:
        path = self.root / "results" / f"{result_id}.json"
        if not path.is_file():
            return None
        try:
            return EscrowResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def results_for_task(self, task_id: str) -> list[EscrowResult]:
        rows: list[EscrowResult] = []
        for path in (self.root / "results").glob("*.json"):
            try:
                item = EscrowResult.model_validate_json(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if item.task_id == task_id:
                rows.append(item)
        return sorted(rows, key=lambda item: (item.created_at, item.result_id))

    def unacknowledged_results(self, parent_task_id: str) -> list[EscrowResult]:
        rows: list[EscrowResult] = []
        for path in (self.root / "results").glob("*.json"):
            try:
                item = EscrowResult.model_validate_json(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if item.parent_task_id == parent_task_id and item.acknowledged_at is None:
                rows.append(item)
        return sorted(rows, key=lambda item: (item.created_at, item.result_id))

    def save_artifact_manifest(self, task_id: str, payload: dict) -> None:
        with self.locked():
            self._atomic_write(self.root / "artefacts" / f"{task_id}.json", payload)

    def artifact_manifest(self, task_id: str) -> dict | None:
        path = self.root / "artefacts" / f"{task_id}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return payload if isinstance(payload, dict) else None

    def append_event(self, event: ExecutionEvent) -> None:
        payload = _redact_for_persistence(event.model_dump(mode="json"))
        line = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str) + "\n"
        task_path = self.root / "events" / f"{event.task_id}.jsonl"
        all_path = self.root / "logs" / "execution.jsonl"
        with self.locked():
            for path in (task_path, all_path):
                if path.exists() and path.stat().st_size + len(line.encode("utf-8")) > self.max_log_bytes:
                    rotated = path.with_suffix(path.suffix + ".1")
                    rotated.unlink(missing_ok=True)
                    os.replace(path, rotated)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(line)
                    stream.flush()
                    os.fsync(stream.fileno())

    def events_for_task(self, task_id: str, *, limit: int = 1000) -> list[dict]:
        path = self.root / "events" / f"{task_id}.jsonl"
        if not path.is_file():
            return []
        rows: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines()[-max(1, limit):]:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows
