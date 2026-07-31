"""Best-effort redacted daily JSONL context/cost analytics."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from mana_agent.utils.redaction import redact_json_line, redact_secrets
from mana_agent.workspaces.paths import mana_home

logger = logging.getLogger(__name__)
_PRIVATE_CONTENT_KEYS = frozenset({
    "prompt", "raw_prompt", "messages", "content", "full_output", "tool_output",
    "raw_email", "email_body", "document_content", "private_document",
})


def _remove_private_content(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[OMITTED_FROM_CONTEXT_COST_LOG]" if str(key).casefold() in _PRIVATE_CONTENT_KEYS else _remove_private_content(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_remove_private_content(item) for item in value]
    return value


class ContextCostLogger:
    def __init__(self, *, enabled: bool = True, retention_days: int = 30, root: Path | None = None) -> None:
        self.enabled = bool(enabled)
        self.retention_days = max(1, int(retention_days))
        self.root = (root or mana_home() / "logs" / "context-cost").resolve()

    def write(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            complete = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": "", "turn_id": "", "task_id": "", "workspace_id": "", "repository_id": "",
                "agent_id": "main", "subagent_id": None, "step_id": "", "provider": "", "model": "",
                "governor_mode": "observe", "context_window": 0, "used_tokens": 0, "remaining_tokens": None,
                "breakdown": {}, "loaded_capabilities": [], "schema_tokens": 0,
                "original_tool_result_tokens": 0, "compressed_tool_result_tokens": 0,
                "cumulative_tokens": 0, "remaining_task_tokens": None,
                "input_cost": 0.0, "output_cost": 0.0, "cumulative_cost": 0.0, "remaining_cost": None,
                "action": "observe", "reason": "", "threshold": None, "outcome": "recorded",
                "artifact_hash": None, "artifact_ref": None, "estimated": True, "exact": False,
                **record,
            }
            complete["exact"] = not bool(complete.get("estimated", True))
            safe = redact_secrets(_remove_private_content(complete))
            line = redact_json_line(json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str))
            with (self.root / f"{date.today().isoformat()}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(line.rstrip("\n") + "\n")
        except Exception:
            logger.debug("context-cost analytics write failed", exc_info=True)

    def cleanup(self, *, today: date | None = None) -> int:
        if not self.root.exists():
            return 0
        cutoff = (today or date.today()) - timedelta(days=self.retention_days)
        removed = 0
        for path in self.root.glob("????-??-??.jsonl"):
            try:
                if date.fromisoformat(path.stem) < cutoff:
                    path.unlink(missing_ok=True)
                    removed += 1
            except (OSError, ValueError):
                continue
        return removed

    def read(self, *, session_id: str = "", since: datetime | None = None) -> Iterable[dict[str, Any]]:
        if not self.root.exists():
            return ()
        rows: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("????-??-??.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    stamp = datetime.fromisoformat(str(row.get("timestamp", "")).replace("Z", "+00:00"))
                except (json.JSONDecodeError, ValueError):
                    continue
                if session_id and str(row.get("session_id", "")) != session_id:
                    continue
                if since is not None and stamp < since:
                    continue
                rows.append(row)
        return rows


__all__ = ["ContextCostLogger"]
