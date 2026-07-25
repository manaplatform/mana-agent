"""Strict contracts shared by the coordinator and reverse-connected workers."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SSHAuthentication(StrictModel):
    mode: Literal["agent", "key_path"]
    key_path: str | None = None

    @field_validator("key_path")
    @classmethod
    def key_path_only_for_key_mode(cls, value: str | None, info):
        if info.data.get("mode") == "key_path" and not value:
            raise ValueError("key_path authentication requires key_path")
        return value


class SSHTarget(StrictModel):
    host: str = Field(min_length=1)
    port: int = Field(default=22, ge=1, le=65535)
    user: str = Field(min_length=1)


class RemoteCommand(StrictModel):
    argv: list[str] = Field(min_length=1)


class PermissionCategory(str, Enum):
    READ_ONLY = "read_only"
    REMOTE_WRITE = "remote_write"
    PRIVILEGED = "privileged_or_destructive"
    INTERACTIVE = "interactive_shell"
    PORT_FORWARDING = "port_forwarding"
    FILE_UPLOAD = "file_upload"
    FILE_DOWNLOAD = "file_download"


class RemoteExecutionRequest(StrictModel):
    job_id: str
    session_id: str
    worker_id: str
    provider: Literal["local_ssh", "external_worker"] = "external_worker"
    target: SSHTarget
    authentication: SSHAuthentication
    command: RemoteCommand
    working_directory: str | None = None
    timeout_seconds: int = Field(default=60, gt=0, le=3600)
    read_only: bool = True
    pty: bool = False
    permission_request_id: str = ""
    requested_at: datetime = Field(default_factory=utc_now)

    def exact_action_key(self) -> str:
        """Stable, non-secret identity binding approvals to the exact action."""
        import hashlib
        import json
        payload = self.model_dump(mode="json", exclude={"job_id", "requested_at", "permission_request_id"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class RemoteJobState(str, Enum):
    QUEUED = "queued"
    ASSIGNED = "assigned"
    AWAITING_PERMISSION = "awaiting_permission"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    WORKER_DISCONNECTED = "worker_disconnected"


class WorkerCapabilities(StrictModel):
    ssh: bool = True
    local_process: bool = True
    docker: bool = False


class WorkerRegistration(StrictModel):
    worker_id: str
    display_name: str
    capabilities: WorkerCapabilities
    operating_system: str
    ssh_available: bool
    labels: list[str] = Field(default_factory=list)
    max_concurrent_jobs: int = Field(default=1, gt=0, le=64)


class RemoteExecutionEvent(StrictModel):
    job_id: str
    session_id: str
    kind: Literal["worker_selected", "target_resolution", "host_key_verification", "permission_requested", "connection_started", "stdout", "stderr", "exit_code", "timeout", "cancelled", "worker_disconnected"]
    data: dict[str, str | int | bool | None] = Field(default_factory=dict)
    sequence: int = Field(default=0, ge=0)

