from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mana_agent.multi_agent.memory.agent_memory import AgentMemory
from mana_agent.multi_agent.memory.repo_context import RepoContext
from mana_agent.multi_agent.memory.task_memory import TaskMemory
from mana_agent.memory import MultiAgentMemoryService, ScopedMemoryBundle


def _clean(text: object, *, max_chars: int = 1200) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    if "secret" in value.lower():
        return ""
    return value[:max_chars]


@dataclass
class AgentMemoryBundle:
    repo_context: RepoContext
    task_memory: TaskMemory
    agent_memories: dict[str, AgentMemory] = field(default_factory=dict)
    service: MultiAgentMemoryService | None = None

    def for_agent(self, agent_id: str) -> AgentMemory:
        key = str(agent_id or "").strip()
        if not key:
            key = "unknown_agent"
        if key not in self.agent_memories:
            self.agent_memories[key] = AgentMemory()
        return self.agent_memories[key]

    def remember_agent(self, agent_id: str, summary: str) -> None:
        text = _clean(summary)
        if text:
            self.for_agent(agent_id).remember(text)

    def remember_task(self, summary: str) -> None:
        text = _clean(summary)
        if text:
            self.task_memory.remember(text)

    def remember_repo_fact(self, fact: str) -> None:
        text = _clean(fact)
        if text and text not in self.repo_context.facts:
            self.repo_context.facts.append(text)
            if self.service is not None:
                self.service.remember_repository_fact(text)

    def scoped_bundle(
        self,
        *,
        agent_id: str,
        agent_role: str,
        task_id: str,
        parent_task_id: str | None = None,
        target_files: list[str] | None = None,
    ) -> ScopedMemoryBundle | None:
        if self.service is None:
            return None
        return self.service.build_bundle(
            agent_id=agent_id,
            agent_role=agent_role,
            task_id=task_id,
            parent_task_id=parent_task_id,
            target_files=target_files,
        )

    def snapshot(
        self,
        agent_id: str | None = None,
        *,
        max_items: int = 8,
        max_chars: int = 2000,
    ) -> str:
        lines: list[str] = ["Multi-Agent Memory Snapshot"]

        repo_facts = self.repo_context.facts[-max_items:]
        lines.append("Repo facts:")
        lines.extend(f"- {item}" for item in repo_facts)
        if not repo_facts:
            lines.append("- none")

        task_items = self.task_memory.summaries[-max_items:]
        lines.append("Task memory:")
        lines.extend(f"- {item}" for item in task_items)
        if not task_items:
            lines.append("- none")

        lines.append("Agent memory:")
        agent_lines: list[str] = []

        if agent_id:
            agent_items = self.for_agent(agent_id).summaries[-max_items:]
            agent_lines.extend(f"- {item}" for item in agent_items)
        else:
            for aid, memory in sorted(self.agent_memories.items()):
                for item in memory.summaries[-max_items:]:
                    agent_lines.append(f"- {aid}: {item}")

        if not agent_lines:
            agent_lines.append("- none")

        lines.extend(agent_lines)
        
        text = "\n".join(lines).strip()
        return text[:max_chars]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_context": {
                "root": self.repo_context.root,
                "facts": list(self.repo_context.facts),
            },
            "task_memory": list(self.task_memory.summaries),
            "agent_memories": {
                agent_id: list(memory.summaries)
                for agent_id, memory in self.agent_memories.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentMemoryBundle":
        repo_data = data.get("repo_context") or {}
        bundle = cls(
            repo_context=RepoContext(
                root=str(repo_data.get("root") or "."),
                facts=list(repo_data.get("facts") or []),
            ),
            task_memory=TaskMemory(),
        )
        for item in data.get("task_memory") or []:
            bundle.task_memory.remember(str(item))

        for agent_id, summaries in (data.get("agent_memories") or {}).items():
            memory = bundle.for_agent(str(agent_id))
            for item in summaries or []:
                memory.remember(str(item))

        return bundle
