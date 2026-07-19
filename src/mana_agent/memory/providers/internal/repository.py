"""Non-destructive storage for canonical records in internal mode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mana_agent.workspaces.store import atomic_write_json


class InternalMemoryRepository:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return list(payload.get("records", [])) if isinstance(payload, dict) else []

    def save(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, {"version": 1, "records": records})
