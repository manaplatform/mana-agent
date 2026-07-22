"""Typed contracts shared by coding backend implementations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
    event_type: str
    task_id: str
    status: Literal["queued", "running", "success", "failed", "cancelled"] = "running"
    title: str = ""
    summary: str = ""
    thread_id: str = ""
    turn_id: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)


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
]
