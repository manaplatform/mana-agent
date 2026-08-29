"""Durable, idempotent accounting reservation store."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

try:  # pragma: no cover - platform-specific branch
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

try:  # pragma: no cover - platform-specific branch
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None

from mana_agent.workspaces.paths import mana_home


class AccountingStore:
    def __init__(self, root: Path | None = None, *, retention_days: int = 30) -> None:
        self.root = (root or mana_home() / "accounting" / "reservations").resolve()
        self.retention_days = max(1, int(retention_days))
        self._lock = threading.RLock()

    def get(self, reservation_id: str) -> dict[str, Any] | None:
        path = self._path(reservation_id)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def put_if_absent(self, reservation_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            with self._process_lock():
                existing = self.get(reservation_id)
                if existing is not None:
                    return existing
                self._write(reservation_id, record)
                return dict(record)

    def update(self, reservation_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            with self._process_lock():
                self._write(reservation_id, record)
                return dict(record)

    def update_if_status(
        self,
        reservation_id: str,
        *,
        expected_status: str,
        record: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Atomically finalize a reservation once across threads and processes."""
        with self._lock:
            with self._process_lock():
                existing = self.get(reservation_id)
                if existing is None:
                    raise KeyError(reservation_id)
                if existing.get("status") != expected_status:
                    return existing, False
                self._write(reservation_id, record)
                return dict(record), True

    def rows(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        rows: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("reservation_*.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def cleanup(self) -> int:
        if not self.root.exists():
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        removed = 0
        with self._lock:
            with self._process_lock():
                for path in self.root.glob("reservation_*.json"):
                    try:
                        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                        if modified < cutoff:
                            path.unlink()
                            removed += 1
                    except OSError:
                        continue
        return removed

    def _write(self, reservation_id: str, record: Mapping[str, Any]) -> None:
        from mana_agent.utils.tool_results import json_safe_dumps

        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(reservation_id)
        serialized = json_safe_dumps(dict(record), sort_keys=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=self.root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def _process_lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".accounting.lock"
        with lock_path.open("a+", encoding="utf-8") as stream:
            if fcntl is not None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            elif msvcrt is not None:
                stream.seek(0)
                if not stream.read(1):
                    stream.write("\0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

    def _path(self, reservation_id: str) -> Path:
        safe = "".join(character for character in str(reservation_id) if character.isalnum() or character in "-_")
        if not safe or safe != reservation_id:
            raise ValueError("invalid accounting reservation id")
        return self.root / f"{safe}.json"


__all__ = ["AccountingStore"]
