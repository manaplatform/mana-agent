from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mana_agent.multi_agent.communication.discussion import DiscussionStore
from mana_agent.multi_agent.communication.message_bus import MessageBus
from mana_agent.multi_agent.core.ids import new_decision_id
from mana_agent.multi_agent.core.types import DecisionRecord, DecisionStatus, MessageType, to_jsonable
from mana_agent.multi_agent.taskboard.store import decision_from_dict
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.persistence.workspace_db import get_workspace_db
from mana_agent.persistence.workspace_repository import WorkspaceRepository
from mana_agent.workspaces.paths import workspace_dir
from mana_agent.workspaces.service import WorkspaceService


class DecisionRoom:
    def __init__(self, root: str | Path, taskboard: TaskBoard, message_bus: MessageBus) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.taskboard = taskboard
        self.message_bus = message_bus
        self.discussions = DiscussionStore(root)
        service = WorkspaceService()
        repo = service.register_repository(self.root)
        workspace = service.workspace_for_repository(repo.repository_id)
        self.workspace_id = workspace.workspace_id
        self.db = get_workspace_db(self.workspace_id)
        self.repository = WorkspaceRepository(self.workspace_id, db=self.db)
        self.path = workspace_dir(workspace.workspace_id) / "taskboard" / "decisions.json"
        self.decisions: dict[str, DecisionRecord] = {}
        self.load()

    def open_discussion(self, task_id: str, title: str, participants: list[str], created_by_agent_id: str | None = None):
        creator = created_by_agent_id or (participants[0] if participants else "agent_head_decision")
        discussion = self.discussions.open(task_id, title, participants, creator)
        self.taskboard.add_discussion(task_id, discussion.discussion_id)
        return discussion

    def post_message(self, discussion_id: str, *, task_id: str, from_agent_id: str, to_agent_id: str | None, message_type: MessageType, content: str, metadata: dict[str, Any] | None = None):
        message = self.message_bus.send(
            task_id=task_id,
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            message_type=message_type,
            content=content,
            discussion_id=discussion_id,
            metadata=metadata,
        )
        self.discussions.add_message(discussion_id, message.message_id)
        return message

    def request_proposal(self, discussion_id: str, task_id: str, from_agent_id: str, to_agent_id: str) -> None:
        self.post_message(discussion_id, task_id=task_id, from_agent_id=from_agent_id, to_agent_id=to_agent_id, message_type=MessageType.QUESTION, content="Provide a concise plan proposal with evidence, risks, and verification needs.")

    def request_objection(self, discussion_id: str, task_id: str, from_agent_id: str, to_agent_id: str) -> None:
        self.post_message(discussion_id, task_id=task_id, from_agent_id=from_agent_id, to_agent_id=to_agent_id, message_type=MessageType.QUESTION, content="Raise any concise quality, safety, or scope objection.")

    def request_verification_plan(self, discussion_id: str, task_id: str, from_agent_id: str, to_agent_id: str) -> None:
        self.post_message(discussion_id, task_id=task_id, from_agent_id=from_agent_id, to_agent_id=to_agent_id, message_type=MessageType.QUESTION, content="Define verification commands and pass/fail criteria.")

    def close_with_decision(
        self,
        *,
        task_id: str,
        discussion_id: str | None,
        made_by_agent_id: str,
        summary: str,
        rationale_summary: str,
        selected_route: str,
        assigned_agent_ids: list[str],
        required_verification: list[str],
        risks: list[str] | None = None,
        assumptions: list[str] | None = None,
        rejected_options: list[str] | None = None,
        decision_status: DecisionStatus = DecisionStatus.APPROVED,
    ) -> DecisionRecord:
        decision = DecisionRecord(
            decision_id=new_decision_id(),
            task_id=task_id,
            discussion_id=discussion_id,
            made_by_agent_id=made_by_agent_id,
            decision_status=decision_status,
            summary=summary,
            rationale_summary=rationale_summary,
            selected_route=selected_route,
            assigned_agent_ids=assigned_agent_ids,
            required_verification=required_verification,
            risks=risks or [],
            assumptions=assumptions or [],
            rejected_options=rejected_options or [],
        )
        self.decisions[decision.decision_id] = decision
        self.repository.save_decision(decision)
        if discussion_id:
            self.discussions.close(discussion_id, decision.decision_id)
        self.taskboard.add_decision(task_id, decision)
        return decision

    def save(self) -> None:
        for decision in self.decisions.values():
            self.repository.save_decision(decision)

    def load(self) -> None:
        self.decisions = {d.decision_id: d for d in self.repository.list_decisions()}
