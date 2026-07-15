"""Managed agent worktree models and lifecycle statuses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class WorkspaceStatus(str, Enum):
    """Lifecycle for Mana-managed coding workspaces."""

    CREATING = "creating"
    READY = "ready"
    RUNNING = "running"
    VERIFYING = "verifying"
    REVIEWING = "reviewing"
    MERGE_CANDIDATE = "merge_candidate"
    MERGED = "merged"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    DIRTY = "dirty"
    CONFLICTED = "conflicted"
    STALE = "stale"
    RETAINED = "retained"

    @classmethod
    def parse(cls, value: Any) -> "WorkspaceStatus":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        try:
            return cls(text)
        except ValueError as exc:
            raise ValueError(f"unknown workspace status: {value!r}") from exc

    @property
    def is_terminal(self) -> bool:
        return self in {
            WorkspaceStatus.MERGED,
            WorkspaceStatus.FAILED,
            WorkspaceStatus.RETAINED,
        }

    @property
    def is_active(self) -> bool:
        return self in {
            WorkspaceStatus.CREATING,
            WorkspaceStatus.READY,
            WorkspaceStatus.RUNNING,
            WorkspaceStatus.VERIFYING,
            WorkspaceStatus.REVIEWING,
        }

    @property
    def is_recoverable(self) -> bool:
        return self in {
            WorkspaceStatus.INTERRUPTED,
            WorkspaceStatus.DIRTY,
            WorkspaceStatus.STALE,
            WorkspaceStatus.CONFLICTED,
            WorkspaceStatus.FAILED,
            WorkspaceStatus.READY,
            WorkspaceStatus.RUNNING,
            WorkspaceStatus.VERIFYING,
            WorkspaceStatus.REVIEWING,
            WorkspaceStatus.MERGE_CANDIDATE,
        }


# Valid forward transitions. Recoverable terminal-ish states may re-enter ready/running.
_ALLOWED_TRANSITIONS: dict[WorkspaceStatus, set[WorkspaceStatus]] = {
    WorkspaceStatus.CREATING: {
        WorkspaceStatus.READY,
        WorkspaceStatus.FAILED,
        WorkspaceStatus.INTERRUPTED,
    },
    WorkspaceStatus.READY: {
        WorkspaceStatus.RUNNING,
        WorkspaceStatus.VERIFYING,
        WorkspaceStatus.REVIEWING,
        WorkspaceStatus.INTERRUPTED,
        WorkspaceStatus.STALE,
        WorkspaceStatus.RETAINED,
        WorkspaceStatus.FAILED,
    },
    WorkspaceStatus.RUNNING: {
        WorkspaceStatus.VERIFYING,
        WorkspaceStatus.REVIEWING,  # allowed when verification is not required
        WorkspaceStatus.DIRTY,
        WorkspaceStatus.FAILED,
        WorkspaceStatus.INTERRUPTED,
        WorkspaceStatus.RETAINED,
        WorkspaceStatus.MERGE_CANDIDATE,  # no-op coding paths with no mutations
    },
    WorkspaceStatus.VERIFYING: {
        WorkspaceStatus.REVIEWING,
        WorkspaceStatus.FAILED,
        WorkspaceStatus.DIRTY,
        WorkspaceStatus.INTERRUPTED,
        WorkspaceStatus.RETAINED,
    },
    WorkspaceStatus.REVIEWING: {
        WorkspaceStatus.MERGE_CANDIDATE,
        WorkspaceStatus.FAILED,
        WorkspaceStatus.RETAINED,
        WorkspaceStatus.DIRTY,
        WorkspaceStatus.CONFLICTED,
        WorkspaceStatus.INTERRUPTED,
    },
    WorkspaceStatus.MERGE_CANDIDATE: {
        WorkspaceStatus.MERGED,
        WorkspaceStatus.CONFLICTED,
        WorkspaceStatus.RETAINED,
        WorkspaceStatus.DIRTY,
        WorkspaceStatus.FAILED,
    },
    WorkspaceStatus.MERGED: {WorkspaceStatus.RETAINED},
    WorkspaceStatus.FAILED: {
        WorkspaceStatus.READY,
        WorkspaceStatus.RUNNING,
        WorkspaceStatus.RETAINED,
        WorkspaceStatus.DIRTY,
        WorkspaceStatus.INTERRUPTED,
    },
    WorkspaceStatus.INTERRUPTED: {
        WorkspaceStatus.READY,
        WorkspaceStatus.RUNNING,
        WorkspaceStatus.RETAINED,
        WorkspaceStatus.DIRTY,
        WorkspaceStatus.STALE,
        WorkspaceStatus.FAILED,
    },
    WorkspaceStatus.DIRTY: {
        WorkspaceStatus.RUNNING,
        WorkspaceStatus.RETAINED,
        WorkspaceStatus.INTERRUPTED,
        WorkspaceStatus.FAILED,
        WorkspaceStatus.MERGE_CANDIDATE,
    },
    WorkspaceStatus.CONFLICTED: {
        WorkspaceStatus.MERGE_CANDIDATE,
        WorkspaceStatus.RETAINED,
        WorkspaceStatus.FAILED,
        WorkspaceStatus.DIRTY,
    },
    WorkspaceStatus.STALE: {
        WorkspaceStatus.READY,
        WorkspaceStatus.RUNNING,
        WorkspaceStatus.RETAINED,
        WorkspaceStatus.FAILED,
        WorkspaceStatus.INTERRUPTED,
    },
    WorkspaceStatus.RETAINED: {
        WorkspaceStatus.READY,
        WorkspaceStatus.RUNNING,
    },
}


def validate_workspace_transition(current: WorkspaceStatus, nxt: WorkspaceStatus) -> None:
    if current == nxt:
        return
    allowed = _ALLOWED_TRANSITIONS.get(current, set())
    if nxt not in allowed:
        raise ValueError(f"invalid workspace transition: {current.value} → {nxt.value}")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return utc_now()


@dataclass
class ManagedWorkspace:
    """Persisted identity for one Mana-managed coding worktree."""

    workspace_id: str
    task_id: str
    repository_id: str
    source_repo_root: str
    worktree_path: str
    branch_name: str
    base_revision: str
    status: WorkspaceStatus = WorkspaceStatus.CREATING
    assigned_agent_id: str = ""
    session_id: str = ""
    multi_agent_workspace_id: str = ""
    current_head: str = ""
    task_slug: str = ""
    dirty: bool = False
    orphaned_metadata: bool = False
    unmanaged_git_worktree: bool = False
    last_error: str = ""
    recovery_notes: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    review_result: dict[str, Any] = field(default_factory=dict)
    merge_result: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value if isinstance(self.status, WorkspaceStatus) else str(self.status)
        payload["created_at"] = self.created_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ManagedWorkspace":
        data = dict(payload or {})
        return cls(
            workspace_id=str(data.get("workspace_id") or ""),
            task_id=str(data.get("task_id") or ""),
            repository_id=str(data.get("repository_id") or ""),
            source_repo_root=str(data.get("source_repo_root") or ""),
            worktree_path=str(data.get("worktree_path") or ""),
            branch_name=str(data.get("branch_name") or ""),
            base_revision=str(data.get("base_revision") or ""),
            status=WorkspaceStatus.parse(data.get("status") or WorkspaceStatus.CREATING.value),
            assigned_agent_id=str(data.get("assigned_agent_id") or ""),
            session_id=str(data.get("session_id") or ""),
            multi_agent_workspace_id=str(data.get("multi_agent_workspace_id") or ""),
            current_head=str(data.get("current_head") or ""),
            task_slug=str(data.get("task_slug") or ""),
            dirty=bool(data.get("dirty")),
            orphaned_metadata=bool(data.get("orphaned_metadata")),
            unmanaged_git_worktree=bool(data.get("unmanaged_git_worktree")),
            last_error=str(data.get("last_error") or ""),
            recovery_notes=[str(item) for item in data.get("recovery_notes") or []],
            events=[dict(item) for item in data.get("events") or [] if isinstance(item, dict)],
            review_result=dict(data.get("review_result") or {}),
            merge_result=dict(data.get("merge_result") or {}),
            created_at=parse_dt(data.get("created_at")),
            updated_at=parse_dt(data.get("updated_at")),
        )

    def list_row(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "workspace_id": self.workspace_id,
            "branch": self.branch_name,
            "status": self.status.value,
            "worktree_path": self.worktree_path,
            "assigned_agent": self.assigned_agent_id,
            "dirty": self.dirty,
            "repository_id": self.repository_id,
            "base_revision": self.base_revision,
            "current_head": self.current_head,
        }

    def status_report(self) -> dict[str, Any]:
        return {
            **self.list_row(),
            "source_repo_root": self.source_repo_root,
            "session_id": self.session_id,
            "multi_agent_workspace_id": self.multi_agent_workspace_id,
            "task_slug": self.task_slug,
            "orphaned_metadata": self.orphaned_metadata,
            "unmanaged_git_worktree": self.unmanaged_git_worktree,
            "last_error": self.last_error,
            "recovery_notes": list(self.recovery_notes),
            "review_result": dict(self.review_result),
            "merge_result": dict(self.merge_result),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "events_count": len(self.events),
            "recent_events": list(self.events[-8:]),
        }
