"""Small redacted owner-only JSONL diagnostic sink for durable runtimes."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mana_agent.utils.redaction import redact_secrets


def resource_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def append_diagnostic(path: Path, *, component: str, event: str, details: dict[str, Any] | None = None) -> None:
    """Append bounded redacted operational evidence without content-bearing data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    if path.exists() and path.stat().st_size >= 5 * 1024 * 1024:
        for index in range(4, 0, -1):
            older = path.with_name(f"{path.stem}.{index}{path.suffix}")
            newer = path.with_name(f"{path.stem}.{index + 1}{path.suffix}")
            if older.exists():
                os.replace(older, newer)
        os.replace(path, path.with_name(f"{path.stem}.1{path.suffix}"))
    record = redact_secrets({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "component": component,
        "event": event,
        "details": details or {},
    })
    with path.open("a", encoding="utf-8") as stream:
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
