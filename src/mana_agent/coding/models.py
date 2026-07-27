"""Typed contracts shared by coding backend implementations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_SECRET_KEY = re.compile(r"(?:api[_-]?key|authorization|password|secret|token|credential)", re.I)
_BEARER = re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/=-]+")
_OUTPUT_LIMIT = 8_000


def redact_event_value(value: Any, *, key: str = "") -> Any:
    """Redact credentials and bound high-volume provider output at the boundary."""

    if key and _SECRET_KEY.search(key) and key not in {
        "token_usage",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "reasoning_tokens",
    }:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact_event_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_event_value(item) for item in value]
    if isinstance(value, str):
        cleaned = _BEARER.sub(r"\1[REDACTED]", value)
        return cleaned if len(cleaned) <= _OUTPUT_LIMIT else f"{cleaned[:_OUTPUT_LIMIT]}… [truncated]"
    return value


class CodingTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    goal: str
    allowed_files: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    relevant_context: str = ""
    requires_repository_write: bool = True

    @field_validator("task_id", "goal")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("value is required")
        return cleaned


class WorkspaceContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    repository_path: Path
    worktree_path: Path
    working_directory: Path | None = None
    branch_name: str = ""
    repository_instructions: str = ""
    sandbox: Literal["readOnly", "workspaceWrite"] = "workspaceWrite"
    approval_policy: Literal["untrusted", "on-request", "never"] = "never"
    allow_in_place_write: bool = False

    @model_validator(mode="after")
    def _validate_paths(self) -> "WorkspaceContext":
        repository = self.repository_path.expanduser().resolve()
        worktree = self.worktree_path.expanduser().resolve()
        working = (
            self.working_directory.expanduser().resolve()
            if self.working_directory is not None
            else worktree
        )
        if not repository.is_dir():
            raise ValueError(f"repository path does not exist: {repository}")
        if not worktree.is_dir():
            raise ValueError(f"worktree path does not exist: {worktree}")
        if not working.is_dir():
            raise ValueError(f"working directory does not exist: {working}")
        try:
            working.relative_to(worktree)
        except ValueError as exc:
            raise ValueError("working directory must be inside the assigned worktree") from exc
        if self.sandbox == "readOnly":
            return self
        if repository == worktree and not self.allow_in_place_write:
            raise ValueError("writing coding tasks require an isolated worktree")
        return self


class AgentEvent(BaseModel):
    """Backend-neutral, persistence-safe live coding event."""

    event_id: str = Field(default_factory=lambda: f"coding-{uuid.uuid4().hex}")
    event_type: str
    task_id: str
    parent_event_id: str | None = None
    backend: Literal["codex", "internal"] = "codex"
    sequence: int = Field(default=0, ge=0)
    status: Literal["queued", "running", "success", "failed", "cancelled"] = "running"
    title: str = ""
    summary: str = ""
    thread_id: str = ""
    turn_id: str = ""
    tool_name: str = ""
    command: str = ""
    path: str = ""
    duration_ms: int | None = Field(default=None, ge=0)
    token_usage: dict[str, Any] | None = None
    cost: float | None = None
    model: str = ""
    error: str = ""
    output_preview: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def _redact_and_bound(self) -> "AgentEvent":
        object.__setattr__(self, "summary", redact_event_value(self.summary))
        object.__setattr__(self, "error", redact_event_value(self.error))
        object.__setattr__(self, "output_preview", redact_event_value(self.output_preview))
        object.__setattr__(self, "payload", redact_event_value(self.payload))
        return self


class CodingTaskResult(BaseModel):
    task_id: str
    worker_id: str
    backend: str
    status: Literal["completed", "failed", "cancelled"]
    summary: str = ""
    changed_files: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    tests_run: list[str] = Field(default_factory=list)
    tests_passed: bool = False
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    branch_name: str = ""
    commit_sha: str | None = None
    token_usage: dict[str, int] | None = None
    thread_id: str = ""
    turn_id: str = ""


class CodingBackendDecision(BaseModel):
    """Model-produced backend selection. There is deliberately no fallback field."""

    decision_id: str
    coding_required: bool
    selected_backend: str | None = None
    parallelizable: bool = False
    estimated_complexity: Literal["low", "medium", "high"]
    requires_repository_write: bool
    required_verification: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(min_length=1)
    safe_to_continue: bool

    @model_validator(mode="after")
    def _validate_selection(self) -> "CodingBackendDecision":
        if self.coding_required and not str(self.selected_backend or "").strip():
            raise ValueError("selected_backend is required for coding tasks")
        if not self.safe_to_continue and self.selected_backend:
            raise ValueError("an unsafe decision must not select a backend")
        return self


__all__ = [
    "AgentEvent",
    "CodingBackendDecision",
    "CodingTask",
    "CodingTaskResult",
    "WorkspaceContext",
    "redact_event_value",
]
