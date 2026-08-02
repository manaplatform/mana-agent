"""Redacted structured audit events for capsule access and mutation."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mana_agent.memory.capsules.models import MemoryCapsule, MemoryPrincipal
from mana_agent.workspaces.paths import mana_home


class CapsuleAuditLogger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (mana_home() / "logs" / "memory-capsules.jsonl")
        self._lock = threading.RLock()

    def emit(
        self,
        event: str,
        *,
        principal: MemoryPrincipal,
        capsule: MemoryCapsule | None = None,
        decision_code: str = "",
        correlation_id: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id,
            "principal": principal.audit_ids(),
            "capsule_id": capsule.capsule_id if capsule else None,
            "scope": capsule.scope.value if capsule else None,
            "namespace": capsule.namespace if capsule else None,
            "task_id": capsule.task_id if capsule else principal.task_id,
            "agent_id": capsule.agent_id if capsule else principal.agent_id,
            "decision_code": decision_code,
            "revision": capsule.revision if capsule else None,
            "content_hash": capsule.content_hash if capsule else None,
            **dict(extra or {}),
        }
        # Capsule title, summary, content, evidence, and credentials are never logged.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
