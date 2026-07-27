from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProcessRecord(BaseModel):
    schema_version: int = 1
    process_id: str = Field(default_factory=lambda: f"process_{uuid.uuid4().hex[:20]}")
    process_type: str
    command_identifier: str
    sanitized_arguments: dict[str, str] = Field(default_factory=dict)
    repository_id: str = ""
    workspace_id: str = ""
    session_id: str = ""
    ownership: Literal["global", "workspace", "session"] = "global"
    transient: bool = False
    singleton_key: str = ""
    state: Literal["pending", "starting", "running", "stopping", "stopped", "failed", "stale"] = "pending"
    os_pid: int | None = None
    process_identity: str = ""
    created_at: str = Field(default_factory=utc_iso)
    started_at: str | None = None
    stopped_at: str | None = None
    heartbeat_at: str | None = None
    restart_policy: Literal["never", "on-failure", "always"] = "never"
    restart_count: int = 0
    last_error_summary: str = ""
    stdout_log: str = ""
    health: Literal["unknown", "healthy", "unhealthy"] = "unknown"
