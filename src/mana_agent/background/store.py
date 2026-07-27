from __future__ import annotations

import threading
from pathlib import Path

from mana_agent.background.models import ProcessRecord
from mana_agent.workspaces.paths import mana_home
from mana_agent.workspaces.store import atomic_write_json


class ProcessStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or mana_home() / "runtime" / "processes").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def directory(self, process_id: str) -> Path:
        clean = str(process_id or "").strip()
        if not clean.startswith("process_") or not clean.replace("_", "").isalnum():
            raise ValueError("invalid process id")
        path = (self.root / clean).resolve()
        if path.parent != self.root:
            raise ValueError("process path escapes runtime root")
        return path

    def save(self, record: ProcessRecord) -> ProcessRecord:
        with self._lock:
            atomic_write_json(self.directory(record.process_id) / "process.json", record.model_dump(mode="json"))
        return record

    def get(self, process_id: str) -> ProcessRecord:
        path = self.directory(process_id) / "process.json"
        if not path.exists():
            raise FileNotFoundError(process_id)
        return ProcessRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[ProcessRecord]:
        rows: list[ProcessRecord] = []
        for path in self.root.glob("*/process.json"):
            try:
                rows.append(ProcessRecord.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(rows, key=lambda row: row.created_at, reverse=True)

    def delete(self, process_id: str) -> None:
        directory = self.directory(process_id)
        for name in ("process.json", "process.log"):
            (directory / name).unlink(missing_ok=True)
        try:
            directory.rmdir()
        except OSError:
            pass
