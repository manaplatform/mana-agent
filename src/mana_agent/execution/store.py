"""Atomic persisted sandbox-handle store."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from mana_agent.config.settings import mana_home
from mana_agent.execution.models import SandboxHandle


class SandboxStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or mana_home() / "execution" / "sandboxes").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, handle: SandboxHandle) -> None:
        target = self.root / f"{handle.sandbox_id}.json"
        fd, temporary = tempfile.mkstemp(prefix=f".{handle.sandbox_id}.", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(handle.model_dump(mode="json"), stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def get(self, sandbox_id: str) -> SandboxHandle | None:
        path = self.root / f"{sandbox_id}.json"
        if not path.is_file():
            return None
        return SandboxHandle.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[SandboxHandle]:
        rows: list[SandboxHandle] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                rows.append(SandboxHandle.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return rows
