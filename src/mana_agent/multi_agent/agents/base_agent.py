from __future__ import annotations

from typing import Any

from mana_agent.multi_agent.communication.message_bus import MessageBus
from mana_agent.multi_agent.core.types import AgentRole, AgentState, HandoffRecord, MessageType
from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.multi_agent.memory.memory_bundle import AgentMemoryBundle

class BaseAgent:
    def __init__(
        self,
        *,
        agent_id: str,
        role: AgentRole,
        parent_agent_id: str | None,
        capabilities: list[str],
        allowed_tools: list[str] | None = None,
        mailbox: MessageBus,
        taskboard: TaskBoard,
        message_bus: MessageBus,
        registry: AgentRegistry | None = None,
        memory: AgentMemoryBundle | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self.parent_agent_id = parent_agent_id
        self.capabilities = capabilities
        self.allowed_tools = allowed_tools or []
        self.state = AgentState.IDLE
        self.mailbox = mailbox
        self.taskboard = taskboard
        self.message_bus = message_bus
        self.registry = registry
        self.memory = memory

    def run(self, task_id: str, context: dict[str, Any]) -> Any:
        self.state = AgentState.RUNNING
        self.record_evidence(task_id, f"{self.role.value} agent started")
        self._remember_only(f"{self.role.value} agent started task {task_id}", scope="agent")
        self.state = AgentState.DONE
        return context

    def can_handle(self, task) -> bool:
        required = set(getattr(task, "required_capabilities", []) or [])
        return not required or bool(required.intersection(self.capabilities))

    def send_message(self, task_id: str, to_agent_id: str | None, message_type: MessageType, content: str, *, discussion_id: str | None = None):
        message = self.message_bus.send(
            task_id=task_id,
            from_agent_id=self.agent_id,
            to_agent_id=to_agent_id,
            message_type=message_type,
            content=content,
            discussion_id=discussion_id,
        )
        self._remember_only(
            f"Sent {message_type.value} message to {to_agent_id or 'broadcast'}: {str(content)[:300]}",
            scope="agent",
        )
        return message

    def broadcast(self, task_id: str, message_type: MessageType, content: str):
        return self.message_bus.broadcast(task_id, self.agent_id, message_type, content)

    def open_discussion(self, task_id: str, title: str) -> None:
        self.taskboard.add_evidence(task_id, f"{self.role.value} requested discussion: {title}")

    def handoff(self, target_agent_id: str, task_id: str, reason: str) -> HandoffRecord:
        record = HandoffRecord(self.agent_id, target_agent_id, task_id, reason)
        self.taskboard.add_handoff(task_id, record)
        self._remember_only(
        f"Handoff from {self.agent_id} to {target_agent_id} for task {task_id}: {reason[:300]}",
        scope="task",
        )
        self.send_message(task_id, target_agent_id, MessageType.HANDOFF, reason)
        return record

    def create_subagent(self, role: AgentRole, capabilities: list[str]):
        if self.role == AgentRole.MAIN and self.registry is not None:
            return self.registry.create_subagent(role, self.agent_id, capabilities)
        return {
            "type": "capacity_request",
            "requested_by_agent_id": self.agent_id,
            "role": role.value,
            "capabilities": list(capabilities),
            "reason": "Only MainAgent can create agents or subagents.",
        }

    def record_evidence(self, task_id: str, evidence: str) -> None:
        self.taskboard.add_evidence(task_id, evidence)
        self._remember_only(f"Evidence by {self.agent_id}: {evidence}", scope="task")
        self._remember_only(f"Evidence recorded for task {task_id}: {evidence}", scope="agent")
        
    def record_decision(self, task_id: str, decision) -> None:
        self.taskboard.add_decision(task_id, decision)
        summary = getattr(decision, "summary", None) or str(decision)
        self._remember_only(f"Decision by {self.agent_id}: {summary}", scope="task")
        self._remember_only(f"Decision recorded for task {task_id}: {summary}", scope="agent")

    def memory_snapshot(self, max_chars: int = 2000) -> str:
        if self.memory is None:
            return "Multi-Agent Memory Snapshot\n- none available"
        scoped = self.memory.scoped_bundle(
            agent_id=self.agent_id,
            agent_role=self.role.value,
            task_id="agent_snapshot",
        )
        if scoped is not None:
            return scoped.to_prompt_block()[:max_chars]
        return self.memory.snapshot(self.agent_id, max_chars=max_chars)

    def _remember_only(self, summary: str, *, scope: str = "agent") -> None:
        if self.memory is None:
            return
        text = str(summary or "").strip()
        if not text:
            return
        if scope == "task":
            self.memory.remember_task(text)
        elif scope == "repo":
            self.memory.remember_repo_fact(text)
        else:
            self.memory.remember_agent(self.agent_id, text)

    def remember(self, task_id: str, summary: str, *, scope: str = "agent") -> None:
        text = str(summary or "").strip()
        if not text:
            return
        try:
            self._remember_only(text, scope=scope)
            if task_id:
                self.taskboard.add_evidence(
                    task_id,
                    f"Memory updated by {self.agent_id}: {text[:300]}",
                )
        except Exception as exc:
            if task_id:
                self.taskboard.add_blocker(
                    task_id,
                    f"Memory update failed for {self.agent_id}: {exc}",
                )
