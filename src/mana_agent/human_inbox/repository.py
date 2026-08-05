"""Crash-durable repository for human inbox state and immutable evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, Protocol, TypeVar

from mana_agent.utils.redaction import redact_secrets

from .models import (
    DeliveryAttempt,
    InboxAuditEvent,
    InboxItem,
    InboxQuery,
    UNRESOLVED_STATUSES,
)

if os.name == "nt":  # pragma: no cover
    import msvcrt
else:  # pragma: no cover
    import fcntl

T = TypeVar("T")


class InboxConcurrentUpdateError(RuntimeError):
    pass


class InboxItemNotFoundError(LookupError):
    pass


class InboxRepository(Protocol):
    def create(self, item: InboxItem, *, protected_context: dict) -> tuple[InboxItem, bool]: ...
    def get(self, inbox_item_id: str) -> InboxItem: ...
    def get_or_none(self, inbox_item_id: str) -> InboxItem | None: ...
    def find_for_action(self, action_intent_id: str, *, unresolved_only: bool = False) -> list[InboxItem]: ...
    def list(self, query: InboxQuery | None = None) -> list[InboxItem]: ...
    def compare_and_set(self, item: InboxItem, *, expected_version: int) -> InboxItem: ...
    def update(self, inbox_item_id: str, updater: Callable[[InboxItem], T]) -> tuple[InboxItem, T]: ...
    def append_audit(self, event: InboxAuditEvent) -> None: ...
    def audit_for_item(self, inbox_item_id: str) -> list[InboxAuditEvent]: ...
    def save_delivery_attempt(self, attempt: DeliveryAttempt) -> None: ...
    def delivery_attempts(self, inbox_item_id: str) -> list[DeliveryAttempt]: ...
    def read_protected_context(self, reference: str) -> dict: ...
    def save_protected_response(self, inbox_item_id: str, response: dict) -> str: ...
    def read_protected_response(self, reference: str) -> dict: ...
    def due_for_expiration(self, now: datetime) -> list[InboxItem]: ...
    def due_for_reminder(self, now: datetime) -> list[InboxItem]: ...


def _safe_name(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class LocalInboxRepository:
    """Atomic local state below ``~/.mana/inbox``.

    Item files are authoritative mutable projections. Audit records and delivery
    attempts are one-file-per-event append-only evidence, so a partial append
    cannot corrupt the history after a crash.
    """

    _DIRECTORIES = ("items", "protected", "audit", "deliveries", "logs")

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        layout_marker = self.root / ".layout-v2"
        first_initialization = not layout_marker.exists()
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.root.chmod(0o700)
        except OSError:  # pragma: no cover - platform ACLs
            pass
        for directory in self._DIRECTORIES:
            path = self.root / directory
            path.mkdir(parents=True, exist_ok=True)
            try:
                path.chmod(0o700)
            except OSError:  # pragma: no cover - platform ACLs
                pass
        self._thread_lock = threading.RLock()
        self._lock_path = self.root / ".repository.lock"
        self._lock_path.touch(exist_ok=True)
        if first_initialization:
            layout_marker.touch(exist_ok=True)
            from mana_agent.utils.durable_diagnostics import append_diagnostic
            append_diagnostic(
                self.root / "logs" / "inbox.jsonl",
                component="human_inbox",
                event="repository_initialized",
            )

    @contextmanager
    def locked(self) -> Iterator[None]:
        with self._thread_lock, self._lock_path.open("r+b") as handle:
            if os.name == "nt":  # pragma: no cover
                handle.seek(0)
                if not handle.read(1):
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":  # pragma: no cover
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic_write(path: Path, payload: dict, *, redact: bool = True) -> None:
        value = redact_secrets(payload) if redact else payload
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, sort_keys=True, ensure_ascii=False, default=str)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                temporary.chmod(0o600)
            except OSError:  # pragma: no cover
                pass
            os.replace(temporary, path)
            if os.name != "nt":
                directory_descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)

    def _item_path(self, inbox_item_id: str) -> Path:
        return self.root / "items" / f"{_safe_name(inbox_item_id)}.json"

    def create(self, item: InboxItem, *, protected_context: dict) -> tuple[InboxItem, bool]:
        with self.locked():
            for existing in self._list_unlocked():
                if existing.idempotency_key == item.idempotency_key:
                    if existing.deduplication_key != item.deduplication_key:
                        raise InboxConcurrentUpdateError("inbox idempotency key is bound to a different request")
                    return existing, False
                if existing.deduplication_key == item.deduplication_key and existing.status in UNRESOLVED_STATUSES:
                    return existing, False
            path = self._item_path(item.inbox_item_id)
            if path.exists():
                raise InboxConcurrentUpdateError(f"inbox item already exists: {item.inbox_item_id}")
            if protected_context:
                reference = f"protected:{_safe_name(item.inbox_item_id)}"
                self._atomic_write(
                    self.root / "protected" / f"{_safe_name(item.inbox_item_id)}.json",
                    protected_context,
                    redact=False,
                )
                item.protected_context_ref = reference
            self._atomic_write(path, item.model_dump(mode="json"))
            return item, True

    def get(self, inbox_item_id: str) -> InboxItem:
        path = self._item_path(inbox_item_id)
        if not path.is_file():
            raise InboxItemNotFoundError(f"unknown inbox item: {inbox_item_id}")
        try:
            return InboxItem.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise InboxConcurrentUpdateError(f"inbox item is unreadable: {inbox_item_id}") from exc

    def get_or_none(self, inbox_item_id: str) -> InboxItem | None:
        try:
            return self.get(inbox_item_id)
        except InboxItemNotFoundError:
            return None

    def _list_unlocked(self) -> list[InboxItem]:
        rows: list[InboxItem] = []
        for path in sorted((self.root / "items").glob("*.json")):
            try:
                rows.append(InboxItem.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                raise InboxConcurrentUpdateError(f"inbox repository contains an unreadable record: {path.name}") from exc
        return rows

    def list(self, query: InboxQuery | None = None) -> list[InboxItem]:
        query = query or InboxQuery()
        rows = self._list_unlocked()
        if query.statuses:
            rows = [item for item in rows if item.status in query.statuses]
        if query.reviewer_id:
            rows = [item for item in rows if query.reviewer_id in item.eligible_reviewer_ids]
        if query.role:
            rows = [item for item in rows if item.assigned_reviewer_type.value == "role" and item.assigned_reviewer_id == query.role]
        if query.group:
            rows = [item for item in rows if item.assigned_reviewer_type.value == "group" and item.assigned_reviewer_id == query.group]
        if query.task_id:
            rows = [item for item in rows if item.task_id == query.task_id]
        if query.branch_id:
            rows = [item for item in rows if item.branch_id == query.branch_id]
        if query.request_type is not None:
            rows = [item for item in rows if item.request_type is query.request_type]
        if query.tenant_id:
            rows = [item for item in rows if item.tenant_id == query.tenant_id]
        if query.project_id:
            rows = [item for item in rows if item.project_id == query.project_id]
        return sorted(rows, key=lambda item: (item.created_at, item.inbox_item_id), reverse=True)

    def find_for_action(self, action_intent_id: str, *, unresolved_only: bool = False) -> list[InboxItem]:
        rows = [item for item in self._list_unlocked() if item.action_intent_id == action_intent_id]
        if unresolved_only:
            rows = [item for item in rows if item.status in UNRESOLVED_STATUSES]
        return sorted(rows, key=lambda item: (item.created_at, item.inbox_item_id), reverse=True)

    def compare_and_set(self, item: InboxItem, *, expected_version: int) -> InboxItem:
        with self.locked():
            current = self.get(item.inbox_item_id)
            if current.version != expected_version:
                raise InboxConcurrentUpdateError(
                    f"inbox item changed concurrently: expected version {expected_version}, found {current.version}"
                )
            item.version = expected_version + 1
            self._atomic_write(self._item_path(item.inbox_item_id), item.model_dump(mode="json"))
            return item

    def update(self, inbox_item_id: str, updater: Callable[[InboxItem], T]) -> tuple[InboxItem, T]:
        with self.locked():
            item = self.get(inbox_item_id)
            version = item.version
            result = updater(item)
            item.version = version + 1
            self._atomic_write(self._item_path(inbox_item_id), item.model_dump(mode="json"))
            return item, result

    def append_audit(self, event: InboxAuditEvent) -> None:
        with self.locked():
            existing = [
                InboxAuditEvent.model_validate_json(path.read_text(encoding="utf-8"))
                for path in (self.root / "audit").glob("*.json")
            ]
            if any(row.audit_event_id == event.audit_event_id for row in existing):
                return
            event.sequence = max((row.sequence for row in existing), default=0) + 1
            path = self.root / "audit" / (
                f"{event.created_at.strftime('%Y%m%dT%H%M%S.%fZ')}-"
                f"{event.sequence:020d}-{_safe_name(event.audit_event_id)}.json"
            )
            if path.exists():
                return
            self._atomic_write(path, event.model_dump(mode="json"))

    def audit_for_item(self, inbox_item_id: str) -> list[InboxAuditEvent]:
        rows: list[InboxAuditEvent] = []
        for path in sorted((self.root / "audit").glob("*.json")):
            event = InboxAuditEvent.model_validate_json(path.read_text(encoding="utf-8"))
            if event.inbox_item_id == inbox_item_id:
                rows.append(event)
        return sorted(rows, key=lambda event: (event.created_at, event.sequence, event.audit_event_id))

    def save_delivery_attempt(self, attempt: DeliveryAttempt) -> None:
        path = self.root / "deliveries" / f"{attempt.timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}-{_safe_name(attempt.delivery_attempt_id)}.json"
        with self.locked():
            if not path.exists():
                self._atomic_write(path, attempt.model_dump(mode="json"))

    def delivery_attempts(self, inbox_item_id: str) -> list[DeliveryAttempt]:
        rows: list[DeliveryAttempt] = []
        for path in sorted((self.root / "deliveries").glob("*.json")):
            attempt = DeliveryAttempt.model_validate_json(path.read_text(encoding="utf-8"))
            if attempt.inbox_item_id == inbox_item_id:
                rows.append(attempt)
        return rows

    def read_protected_context(self, reference: str) -> dict:
        if not reference.startswith("protected:"):
            raise PermissionError("invalid protected context reference")
        token = reference.removeprefix("protected:")
        if re.fullmatch(r"[0-9a-f]{64}", token) is None:
            raise PermissionError("invalid protected context reference")
        path = self.root / "protected" / f"{token}.json"
        if not path.is_file():
            raise InboxItemNotFoundError("protected context was not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def save_protected_response(self, inbox_item_id: str, response: dict) -> str:
        token = _safe_name(f"response:{inbox_item_id}")
        path = self.root / "protected" / f"{token}.json"
        with self.locked():
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing != response:
                    raise InboxConcurrentUpdateError("protected response changed concurrently")
            else:
                self._atomic_write(path, response, redact=False)
        return f"protected-response:{token}"

    def read_protected_response(self, reference: str) -> dict:
        if not reference.startswith("protected-response:"):
            raise PermissionError("invalid protected response reference")
        token = reference.removeprefix("protected-response:")
        if re.fullmatch(r"[0-9a-f]{64}", token) is None:
            raise PermissionError("invalid protected response reference")
        path = self.root / "protected" / f"{token}.json"
        if not path.is_file():
            raise InboxItemNotFoundError("protected response was not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def due_for_expiration(self, now: datetime) -> list[InboxItem]:
        return [item for item in self._list_unlocked() if item.status in UNRESOLVED_STATUSES and item.expires_at <= now]

    def due_for_reminder(self, now: datetime) -> list[InboxItem]:
        rows: list[InboxItem] = []
        for item in self._list_unlocked():
            policy = item.reminder_policy
            if item.status not in UNRESOLVED_STATUSES or item.reminder_count >= policy.max_reminders:
                continue
            baseline = item.last_reminded_at or item.delivered_at or item.created_at
            if (now - baseline).total_seconds() >= policy.interval_seconds:
                rows.append(item)
        return rows
