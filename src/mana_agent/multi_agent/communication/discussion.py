from __future__ import annotations

import json
from pathlib import Path

from mana_agent.multi_agent.core.ids import new_discussion_id
from mana_agent.multi_agent.core.types import DiscussionStatus, DiscussionThread, to_jsonable, utc_now
from mana_agent.multi_agent.taskboard.store import discussion_from_dict
from mana_agent.persistence.workspace_db import get_workspace_db
from mana_agent.persistence.workspace_repository import WorkspaceRepository
from mana_agent.workspaces.paths import workspace_dir
from mana_agent.workspaces.service import WorkspaceService


class DiscussionStore:
    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        service = WorkspaceService()
        repo = service.register_repository(self.root)
        workspace = service.workspace_for_repository(repo.repository_id)
        self.workspace_id = workspace.workspace_id
        self.db = get_workspace_db(self.workspace_id)
        self.repository = WorkspaceRepository(self.workspace_id, db=self.db)
        self.path = workspace_dir(workspace.workspace_id) / "taskboard" / "discussions.json"
        self.discussions: dict[str, DiscussionThread] = {}
        self.load()

    def open(self, task_id: str, title: str, participants: list[str], created_by_agent_id: str) -> DiscussionThread:
        discussion = DiscussionThread(
            discussion_id=new_discussion_id(),
            task_id=task_id,
            title=title,
            status=DiscussionStatus.OPEN,
            participant_agent_ids=list(dict.fromkeys(participants)),
            created_by_agent_id=created_by_agent_id,
        )
        self.discussions[discussion.discussion_id] = discussion
        self.save()
        return discussion

    def add_message(self, discussion_id: str, message_id: str) -> None:
        discussion = self.discussions[discussion_id]
        if message_id not in discussion.message_ids:
            discussion.message_ids.append(message_id)
        discussion.updated_at = utc_now()
        self.save()

    def close(self, discussion_id: str, final_decision_id: str | None = None) -> None:
        discussion = self.discussions[discussion_id]
        discussion.status = DiscussionStatus.RESOLVED
        discussion.final_decision_id = final_decision_id
        discussion.updated_at = utc_now()
        self.save()

    def save(self) -> None:
        for discussion in self.discussions.values():
            self.repository.save_discussion(discussion)

    def load(self) -> None:
        self.discussions = {d.discussion_id: d for d in self.repository.list_discussions()}
