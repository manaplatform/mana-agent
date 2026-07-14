from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mana_agent.multi_agent.core.ids import new_message_id
from mana_agent.multi_agent.core.types import AgentMessage, MessageType, to_jsonable
from mana_agent.multi_agent.taskboard.store import message_from_dict
from mana_agent.workspaces.paths import workspace_dir
from mana_agent.workspaces.service import WorkspaceService


class MessageBus:
    def __init__(self, root: str | Path = ".", *, max_messages_per_task: int = 24) -> None:
        self.root = Path(root).resolve()
        service = WorkspaceService()
        repo = service.register_repository(self.root)
        workspace = service.workspace_for_repository(repo.repository_id)
        self.path = workspace_dir(workspace.workspace_id) / "taskboard" / "messages.jsonl"
        self.messages: dict[str, AgentMessage] = {}
        self.max_messages_per_task = max(1, int(max_messages_per_task))
        self._load()

    def send(
        self,
        *,
        task_id: str,
        from_agent_id: str,
        to_agent_id: str | None,
        message_type: MessageType,
        content: str,
        discussion_id: str | None = None,
        root_task_id: str = "",
        evidence_references: list[str] | None = None,
        confidence: float = 0.0,
        requested_action: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AgentMessage:
        if not str(task_id).strip() or not str(from_agent_id).strip():
            raise ValueError("task_id and from_agent_id are required")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        task_messages = sum(1 for item in self.messages.values() if item.task_id == task_id)
        if task_messages >= self.max_messages_per_task:
            raise RuntimeError(f"agent communication budget exhausted for task {task_id}")
        message = AgentMessage(
            message_id=new_message_id(),
            discussion_id=discussion_id,
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            task_id=task_id,
            message_type=message_type,
            content=content.strip(),
            root_task_id=str(root_task_id or task_id).strip(),
            evidence_references=list(dict.fromkeys(str(item).strip() for item in (evidence_references or []) if str(item).strip())),
            confidence=float(confidence),
            requested_action=str(requested_action or "").strip(),
            metadata=metadata or {},
        )
        self.messages[message.message_id] = message
        self._append(message)
        return message

    def broadcast(self, task_id: str, from_agent_id: str, message_type: MessageType, content: str) -> AgentMessage:
        return self.send(task_id=task_id, from_agent_id=from_agent_id, to_agent_id=None, message_type=message_type, content=content)

    def inbox(self, agent_id: str) -> list[AgentMessage]:
        return [item for item in self.messages.values() if item.to_agent_id in {agent_id, None}]

    def thread(self, discussion_id: str) -> list[AgentMessage]:
        return [item for item in self.messages.values() if item.discussion_id == discussion_id]

    def mark_read(self, message_id: str, agent_id: str) -> None:
        message = self.messages[message_id]
        read_by = list(message.metadata.get("read_by", []))
        if agent_id not in read_by:
            read_by.append(agent_id)
        message.metadata["read_by"] = read_by

    def _append(self, message: AgentMessage) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(message), sort_keys=True, ensure_ascii=False) + "\n")

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                message = message_from_dict(json.loads(line))
            except Exception:
                continue
            self.messages[message.message_id] = message
