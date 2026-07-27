from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobState(str, Enum):
    RECEIVED = "received"
    IGNORED = "ignored"
    QUEUED = "queued"
    RUNNING = "running"
    INVESTIGATING = "investigating"
    IMPLEMENTING = "implementing"
    VERIFYING = "verifying"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RouteDecision(BaseModel):
    decision_id: str
    supported: bool
    execute: bool
    safe_to_continue: bool
    trigger: str = ""
    subject_type: str = ""
    subject_number: int | None = None
    requires_human_authorization: bool = True
    reason: str

    @model_validator(mode="after")
    def executable_is_complete(self) -> "RouteDecision":
        if self.execute and (not self.supported or not self.safe_to_continue or not self.trigger or not self.subject_type or self.subject_number is None):
            raise ValueError("executable route decision is incomplete or unsafe")
        return self


class GitHubJob(BaseModel):
    job_id: str
    delivery_id: str
    semantic_key: str
    event_name: str
    action: str
    installation_id: int
    repository_id: int
    repository_full_name: str
    sender_login: str = ""
    sender_type: str = ""
    sender_permission: str = ""
    subject_type: str = ""
    subject_number: int | None = None
    trigger_id: int | None = None
    base_branch: str = ""
    target_sha: str = ""
    context: dict[str, Any] = Field(default_factory=dict)
    route_decision: RouteDecision
    session_id: str = ""
    branch_name: str = ""
    worktree_path: str = ""
    pull_request_number: int | None = None
    state: JobState = JobState.RECEIVED
    retry_count: int = 0
    failure_type: str = ""
    failure_detail: str = ""
    feedback: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)


class DeliveryReceipt(BaseModel):
    delivery_id: str
    event_name: str
    accepted: bool
    job_id: str = ""
    result: Literal["queued", "ignored"]
    reason: str = ""
    received_at: str = Field(default_factory=now_iso)
