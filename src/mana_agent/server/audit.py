"""Append-only redacted audit trail for server actions."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from mana_agent.config.settings import mana_home
from mana_agent.utils.redaction import redact_secrets

from .models import utc_now


class ServerAuditLog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (mana_home() / "servers" / "audit.jsonl")
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        payload = {
            "timestamp": utc_now().isoformat(),
            **dict(redact_secrets(event)),
        }
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        os.chmod(self.path, 0o600)

    def read(self, *, server_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if server_id and row.get("server_id") != server_id:
                continue
            rows.append(row)
        return rows[-max(1, limit):]
