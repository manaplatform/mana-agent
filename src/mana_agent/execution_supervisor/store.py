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

from mana_agent.execution_supervisor.errors import (
    ConcurrentUpdateError,
    EscrowConflictError,
    EscrowCorruptError,
    EscrowIncompatibleVersionError,
    EscrowNotFoundError,
    TaskNotFoundError,
)
from mana_agent.execution_supervisor.models import (
    AttemptRecord,
    ActionRecord,
    CheckpointRecord,
    EscrowResult,
    ExecutionEvent,
    RecoveryInterventionRecord,
    ResultAcknowledgement,
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
    from mana_agent.utils.tool_results import json_safe_tool_payload

    normalized = json_safe_tool_payload(payload)
    safe = redact_secrets(normalized)

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

    restore_hashes(normalized, safe)
    return json_safe_tool_payload(safe)


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
        self, task_id: str, updater: Callable[[TaskRecord], CheckpointRecord | None]
    ) -> tuple[TaskRecord, CheckpointRecord | None]: ...
    def update_task_and_result(
        self, task_id: str, updater: Callable[[TaskRecord], EscrowResult]
    ) -> tuple[TaskRecord, EscrowResult]: ...
    def save_attempt(self, attempt: AttemptRecord) -> None: ...
    def save_recovery_intervention(self, intervention: RecoveryInterventionRecord) -> None: ...
    def get_recovery_intervention(self, intervention_id: str) -> RecoveryInterventionRecord | None: ...
    def recovery_interventions_for_task(self, task_id: str) -> list[RecoveryInterventionRecord]: ...
    def save_action(self, action: ActionRecord) -> None: ...
    def get_action(self, action_id: str) -> ActionRecord | None: ...
    def actions_for_task(self, task_id: str) -> list[ActionRecord]: ...
    def get_attempt(self, attempt_id: str) -> AttemptRecord | None: ...
    def save_checkpoint(self, checkpoint: CheckpointRecord) -> None: ...
    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None: ...
    def checkpoints_for_task(self, task_id: str) -> list[CheckpointRecord]: ...
    def save_result(self, result: EscrowResult) -> None: ...
    def get_result(self, result_id: str) -> EscrowResult | None: ...
    def get_result_by_execution_id(self, execution_id: str) -> EscrowResult | None: ...
    def results_for_task(self, task_id: str) -> list[EscrowResult]: ...
    def results_for_turn(self, trigger_turn_id: str) -> list[EscrowResult]: ...
    def results_for_session(self, session_id: str) -> list[EscrowResult]: ...
    def unacknowledged_results(self, parent_task_id: str) -> list[EscrowResult]: ...
    def save_acknowledgement(self, acknowledgement: ResultAcknowledgement) -> None: ...
    def get_acknowledgement(self, result_id: str) -> ResultAcknowledgement | None: ...
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

    _DIRECTORIES = (
        "tasks",
        "attempts",
        "recovery_interventions",
        "actions",
        "checkpoints",
        "results",
        "results_by_execution",
        "acknowledgements",
        "artefacts",
        "events",
        "logs",
    )

    def __init__(self, root: Path, *, max_log_bytes: int = 10 * 1024 * 1024) -> None:
        self.root = root.expanduser().resolve()
        self.max_log_bytes = max(4096, int(max_log_bytes))
        for name in self._DIRECTORIES:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()
        self._lock_path = self.root / ".store.lock"
        self._lock_path.touch(exist_ok=True)
        self._lock_depth = 0
        self._lock_handle = None

    @contextmanager
    def locked(self) -> Iterator[None]:
        with self._thread_lock:
            if self._lock_depth == 0:
                self._lock_handle = self._lock_path.open("r+b")
                _acquire_file_lock(self._lock_handle)
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
                if self._lock_depth == 0 and self._lock_handle is not None:
                    try:
                        _release_file_lock(self._lock_handle)
                    finally:
                        self._lock_handle.close()
                        self._lock_handle = None

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
        self, task_id: str, updater: Callable[[TaskRecord], CheckpointRecord | None]
    ) -> tuple[TaskRecord, CheckpointRecord | None]:
        with self.locked():
            task = self.get_task(task_id)
            version = task.state_version
            checkpoint = updater(task)
            if checkpoint is not None:
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

    def save_recovery_intervention(self, intervention: RecoveryInterventionRecord) -> None:
        with self.locked():
            self._atomic_write(
                self.root / "recovery_interventions" / f"{intervention.intervention_id}.json",
                intervention.model_dump(mode="json"),
            )

    def get_recovery_intervention(
        self, intervention_id: str
    ) -> RecoveryInterventionRecord | None:
        if not intervention_id:
            return None
        path = self.root / "recovery_interventions" / f"{intervention_id}.json"
        if not path.is_file():
            return None
        try:
            return RecoveryInterventionRecord.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (ValueError, OSError):
            return None

    def recovery_interventions_for_task(
        self, task_id: str
    ) -> list[RecoveryInterventionRecord]:
        rows: list[RecoveryInterventionRecord] = []
        for path in (self.root / "recovery_interventions").glob("*.json"):
            try:
                item = RecoveryInterventionRecord.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (ValueError, OSError):
                continue
            if item.task_id == task_id:
                rows.append(item)
        return sorted(rows, key=lambda item: (item.created_at, item.intervention_id))

    def save_action(self, action: ActionRecord) -> None:
        with self.locked():
            self._atomic_write(
                self.root / "actions" / f"{action.action_id}.json",
                action.model_dump(mode="json"),
            )

    def get_action(self, action_id: str) -> ActionRecord | None:
        path = self.root / "actions" / f"{action_id}.json"
        if not path.is_file():
            return None
        try:
            return ActionRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def actions_for_task(self, task_id: str) -> list[ActionRecord]:
        rows: list[ActionRecord] = []
        for path in (self.root / "actions").glob("*.json"):
            try:
                item = ActionRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if item.execution_id == task_id:
                rows.append(item)
        return sorted(rows, key=lambda item: (item.created_at, item.action_id))

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
            result_path = self.root / "results" / f"{result.result_id}.json"
            exec_id = result.execution_id or result.task_id
            exec_path = self.root / "results_by_execution" / f"{exec_id}.json"

            if result_path.is_file():
                try:
                    existing = self.get_result(result.result_id)
                    if existing is not None:
                        if (
                            existing.result_id == result.result_id
                            and (existing.execution_id == result.execution_id or existing.task_id == result.task_id)
                            and existing.attempt_id == result.attempt_id
                        ):
                            if existing.payload != result.payload and existing.result_kind == result.result_kind:
                                raise EscrowConflictError(
                                    f"Conflicting result payload for already persisted result {result.result_id}"
                                )
                        else:
                            raise EscrowConflictError(
                                f"Conflicting result write for result_id {result.result_id} (existing={existing.result_id})"
                            )
                except (EscrowCorruptError, EscrowIncompatibleVersionError):
                    pass

            if exec_path.is_file() and not result_path.is_file():
                try:
                    idx = json.loads(exec_path.read_text(encoding="utf-8"))
                    existing_result_id = idx.get("result_id")
                    if existing_result_id and existing_result_id != result.result_id:
                        try:
                            existing_res = self.get_result(existing_result_id)
                        except (EscrowCorruptError, EscrowIncompatibleVersionError):
                            existing_res = None
                        if (
                            existing_res is not None
                            and existing_res.attempt_id == result.attempt_id
                            and existing_res.payload == result.payload
                            and existing_res.result_kind == result.result_kind
                        ):
                            # Idempotent reconciliation of identical logical result from concurrent writer race
                            return
                        raise EscrowConflictError(
                            f"Conflicting result write for execution {exec_id} (existing result_id={existing_result_id})"
                        )
                except (json.JSONDecodeError, OSError):
                    pass

            payload = _redact_for_persistence(result.model_dump(mode="json"))
            self._atomic_write(result_path, payload)
            if exec_id:
                self._atomic_write(
                    exec_path,
                    {
                        "result_id": result.result_id,
                        "execution_id": exec_id,
                        "attempt_id": result.attempt_id,
                        "task_id": result.task_id,
                        "trigger_turn_id": result.trigger_turn_id,
                        "session_id": result.session_id,
                    },
                )

    def get_result(self, result_id: str) -> EscrowResult | None:
        if not result_id:
            return None
        path = self.root / "results" / f"{result_id}.json"
        if not path.is_file():
            return None
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            raise EscrowCorruptError(
                f"Escrow result {result_id} is corrupted JSON: {exc}"
            ) from exc
        if isinstance(raw, dict):
            version = raw.get("schema_version", 1)
            if isinstance(version, int) and version > 2:
                raise EscrowIncompatibleVersionError(
                    f"Escrow result {result_id} has unsupported schema version {version}"
                )
        try:
            return EscrowResult.model_validate(raw)
        except Exception as exc:
            if isinstance(exc, EscrowIncompatibleVersionError):
                raise
            raise EscrowCorruptError(
                f"Escrow result {result_id} failed validation: {exc}"
            ) from exc

    def get_result_by_execution_id(self, execution_id: str) -> EscrowResult | None:
        if not execution_id:
            return None
        exec_path = self.root / "results_by_execution" / f"{execution_id}.json"
        if exec_path.is_file():
            try:
                idx = json.loads(exec_path.read_text(encoding="utf-8"))
                result_id = idx.get("result_id")
                if result_id:
                    res = self.get_result(result_id)
                    if res is not None:
                        return res
            except (EscrowCorruptError, EscrowIncompatibleVersionError):
                raise
            except (json.JSONDecodeError, OSError):
                pass
        direct = self.get_result(execution_id)
        if direct is not None and (
            direct.execution_id == execution_id or direct.task_id == execution_id
        ):
            return direct
        for path in (self.root / "results").glob("*.json"):
            try:
                item = self.get_result(path.stem)
                if item is not None and (
                    item.execution_id == execution_id or item.task_id == execution_id
                ):
                    return item
            except (EscrowCorruptError, EscrowIncompatibleVersionError):
                raise
            except Exception:
                continue
        return None

    def results_for_task(self, task_id: str) -> list[EscrowResult]:
        rows: list[EscrowResult] = []
        for path in (self.root / "results").glob("*.json"):
            try:
                item = self.get_result(path.stem)
            except (EscrowCorruptError, EscrowIncompatibleVersionError):
                raise
            except (ValueError, OSError, EscrowError):
                continue
            if item is not None and (item.task_id == task_id or item.execution_id == task_id):
                rows.append(item)
        return sorted(rows, key=lambda item: (item.created_at, item.result_id))

    def results_for_turn(self, trigger_turn_id: str) -> list[EscrowResult]:
        rows: list[EscrowResult] = []
        for path in (self.root / "results").glob("*.json"):
            try:
                item = self.get_result(path.stem)
            except (EscrowCorruptError, EscrowIncompatibleVersionError):
                raise
            except (ValueError, OSError, EscrowError):
                continue
            if item is not None and item.trigger_turn_id == trigger_turn_id:
                rows.append(item)
        return sorted(rows, key=lambda item: (item.created_at, item.result_id))

    def results_for_session(self, session_id: str) -> list[EscrowResult]:
        rows: list[EscrowResult] = []
        for path in (self.root / "results").glob("*.json"):
            try:
                item = self.get_result(path.stem)
            except (EscrowCorruptError, EscrowIncompatibleVersionError):
                raise
            except (ValueError, OSError, EscrowError):
                continue
            if item is not None and item.session_id == session_id:
                rows.append(item)
        return sorted(rows, key=lambda item: (item.created_at, item.result_id))

    def save_acknowledgement(self, acknowledgement: ResultAcknowledgement) -> None:
        with self.locked():
            self._atomic_write(
                self.root / "acknowledgements" / f"{acknowledgement.result_id}.json",
                acknowledgement.model_dump(mode="json"),
            )

    def get_acknowledgement(self, result_id: str) -> ResultAcknowledgement | None:
        if not result_id:
            return None
        path = self.root / "acknowledgements" / f"{result_id}.json"
        if not path.is_file():
            return None
        try:
            return ResultAcknowledgement.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (ValueError, OSError):
            return None

    def unacknowledged_results(self, parent_task_id: str) -> list[EscrowResult]:
        rows: list[EscrowResult] = []
        for path in (self.root / "results").glob("*.json"):
            try:
                item = self.get_result(path.stem)
            except (EscrowCorruptError, EscrowIncompatibleVersionError):
                raise
            except (ValueError, OSError, EscrowError):
                continue
            if item is not None and item.parent_task_id == parent_task_id:
                ack = self.get_acknowledgement(item.result_id)
                if ack is None and item.acknowledged_at is None:
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
