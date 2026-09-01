from __future__ import annotations

from pathlib import Path
from typing import Any

from mana_agent.multi_agent.core.types import (
    AgentMessage,
    DecisionRecord,
    DiscussionThread,
    HandoffRecord,
    QueueJob,
    TaskBoardItem,
    VerificationResult,
    parse_dt,
)
from mana_agent.multi_agent.taskboard.deserializers import (
    decision_from_dict,
    discussion_from_dict,
    handoff_from_dict,
    message_from_dict,
    queue_job_from_dict,
    serialize,
    task_from_dict,
    verification_from_dict,
)
from mana_agent.persistence.workspace_db import get_workspace_db
from mana_agent.persistence.workspace_repository import WorkspaceRepository
from mana_agent.workspaces.paths import workspace_dir
from mana_agent.workspaces.service import WorkspaceService

__all__ = [
    "JsonStateStore",
    "SqliteStateStore",
    "decision_from_dict",
    "discussion_from_dict",
    "handoff_from_dict",
    "message_from_dict",
    "queue_job_from_dict",
    "serialize",
    "task_from_dict",
    "verification_from_dict",
]


class JsonStateStore:
    """Workspace state store backed authoritatively by the workspace SQLite database."""

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        workspaces = WorkspaceService()
        repo = workspaces.register_repository(self.root)
        workspace = workspaces.workspace_for_repository(repo.repository_id)
        self.repository_id = repo.repository_id
        self.workspace_id = workspace.workspace_id
        self.base_dir = workspace_dir(workspace.workspace_id) / "taskboard"
        self.state_path = self.base_dir / "state.json"
        self.history_path = self.base_dir / "history.jsonl"

        self.db = get_workspace_db(self.workspace_id)
        self.repository = WorkspaceRepository(self.workspace_id, db=self.db)
        # Migrate legacy files if present
        from mana_agent.persistence.migration import TaskboardGatewayMigrator
        migrator = TaskboardGatewayMigrator(self.workspace_id, db=self.db)
        if not migrator.is_taskboard_migrated():
            migrator.migrate_taskboard()

    def load_state(self) -> dict[str, Any]:
        tasks = self.repository.list_tasks()
        return {
            "schema_version": 2,
            "tasks": {t.task_id: serialize(t) for t in tasks},
        }

    def save_state(self, payload: dict[str, Any]) -> None:
        raw_tasks = payload.get("tasks", {}) if isinstance(payload, dict) else {}
        for item in raw_tasks.values():
            if isinstance(item, TaskBoardItem):
                self.repository.save_task(item)
            elif isinstance(item, dict):
                self.repository.save_task(task_from_dict(item))

    def append_history(self, event: dict[str, Any]) -> None:
        task_id = None
        if isinstance(event.get("payload"), dict):
            task_id = event["payload"].get("task_id")
        elif isinstance(event.get("payload"), TaskBoardItem):
            task_id = event["payload"].task_id
        self.repository.append_task_event(
            task_id=task_id,
            event_type=str(event.get("event_type") or "task.event"),
            payload=event.get("payload", {}),
            created_at=parse_dt(event.get("created_at")),
        )


SqliteStateStore = JsonStateStore
