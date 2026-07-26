"""Atomic, bounded Fleet persistence under the user Mana home."""

from __future__ import annotations

import os
import tempfile
from datetime import timedelta
from pathlib import Path

from .errors import FleetPersistenceError
from .events import FleetEvent
from .models import FleetRun, FleetWorker, utc_now


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


class FleetStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.workers_dir = self.root / "workers"
        self.runs_dir = self.root / "runs"
        self.cancellations_dir = self.root / "cancellations"
        self.events_path = self.root / "events.jsonl"
        for directory in (self.root, self.workers_dir, self.runs_dir, self.cancellations_dir):
            directory.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(directory, 0o700)
            except OSError:
                pass

    @staticmethod
    def _safe_id(value: str) -> str:
        if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in value):
            raise FleetPersistenceError("invalid persisted fleet identity")
        return value

    def save_worker(self, worker: FleetWorker) -> None:
        path = self.workers_dir / f"{self._safe_id(worker.worker_id)}.json"
        atomic_write(path, worker.model_dump_json(indent=2))

    def load_workers(self) -> list[FleetWorker]:
        workers: list[FleetWorker] = []
        for path in sorted(self.workers_dir.glob("*.json")):
            try:
                workers.append(FleetWorker.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception as exc:
                raise FleetPersistenceError(f"invalid persisted worker record: {path.name}") from exc
        return workers

    def save_run(self, run: FleetRun) -> None:
        path = self.runs_dir / f"{self._safe_id(run.fleet_run_id)}.json"
        if path.exists():
            previous = FleetRun.model_validate_json(path.read_text(encoding="utf-8"))
            previous_completed = {item.job_id for item in previous.results}
            current_completed = {item.job_id for item in run.results}
            if not previous_completed.issubset(current_completed):
                raise FleetPersistenceError("completed fleet job results are immutable")
        atomic_write(path, run.model_dump_json(indent=2))

    def load_run(self, fleet_run_id: str) -> FleetRun:
        path = self.runs_dir / f"{self._safe_id(fleet_run_id)}.json"
        if not path.exists():
            raise FleetPersistenceError(f"fleet run not found: {fleet_run_id}")
        return FleetRun.model_validate_json(path.read_text(encoding="utf-8"))

    def list_runs(self) -> list[FleetRun]:
        return [
            FleetRun.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.runs_dir.glob("*.json"))
        ]

    def append_event(self, event: FleetEvent) -> None:
        existing = self.events()[-1:] if self.events_path.exists() else []
        if existing and event.sequence <= existing[0].sequence:
            raise FleetPersistenceError("fleet event sequence must increase")
        line = event.model_dump_json() + "\n"
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(self.events_path, 0o600)
        except OSError:
            pass

    def request_cancellation(self, job_id: str) -> None:
        atomic_write(self.cancellations_dir / self._safe_id(job_id), "cancel\n")

    def cancellation_requested(self, job_id: str) -> bool:
        return (self.cancellations_dir / self._safe_id(job_id)).is_file()

    def clear_cancellation(self, job_id: str) -> None:
        (self.cancellations_dir / self._safe_id(job_id)).unlink(missing_ok=True)

    def events(self, *, after_sequence: int = 0, limit: int = 1000) -> list[FleetEvent]:
        if not self.events_path.exists():
            return []
        rows = [
            FleetEvent.model_validate_json(line)
            for line in self.events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [item for item in rows if item.sequence > after_sequence][:limit]

    def prune(self, retain_days: int) -> int:
        cutoff = utc_now() - timedelta(days=retain_days)
        removed = 0
        for path in self.runs_dir.glob("*.json"):
            run = FleetRun.model_validate_json(path.read_text(encoding="utf-8"))
            if run.updated_at < cutoff and run.summary is not None:
                path.unlink()
                removed += 1
        return removed
