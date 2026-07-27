"""Sanitized JSONL audit logging for desktop actions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from mana_agent.config.settings import mana_home
from mana_agent.integrations.computer_control.models import AuditRecord


class AuditLogger:
    def __init__(self, *, enabled: bool = True, retention_days: int = 30, path: Path | None = None) -> None:
        self.enabled = enabled
        self.retention_days = retention_days
        self.path = path or mana_home() / "audit" / "computer-control.jsonl"

    def record(self, record: AuditRecord) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(record.model_dump_json() + "\n")
        self.path.chmod(0o600)
        self.prune()

    def prune(self) -> None:
        if not self.path.exists():
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        retained: list[str] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                record = AuditRecord.model_validate_json(line)
            except ValueError:
                continue
            if record.finished_at >= cutoff:
                retained.append(record.model_dump_json())
        self.path.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
        self.path.chmod(0o600)
