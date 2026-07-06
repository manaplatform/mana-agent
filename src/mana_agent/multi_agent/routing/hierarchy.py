from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mana_agent.multi_agent.core.types import AgentNode, AgentRole
from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard

REQUIRED_HIERARCHY = {
    "MainAgent": {
        "HeadDecisionAgent": [
            "PlannerAgent",
            "CodingAgent",
            "VerifierAgent",
            "ReviewerAgent",
            "SummarizerAgent",
        ],
        "CodingAgent": {
            "QueueManager": {
                "ToolWorkerAgent": ["ToolsManager"],
            },
        },
    }
}


def _role_from_agent_id(agent_id: str, registry: AgentRegistry | None = None) -> AgentRole | None:
    text = str(agent_id or "")
    if registry is not None and text in registry.agents:
        return registry.agents[text].role
    if text.startswith("agent_main_") or text == "main":
        return AgentRole.MAIN
    if text.startswith("agent_head_decision_"):
        return AgentRole.HEAD_DECISION
    if text.startswith("agent_planner_"):
        return AgentRole.PLANNER
    if text.startswith("agent_coding_") or text.startswith("subagent_coding_"):
        return AgentRole.CODING
    if text.startswith("agent_verifier_"):
        return AgentRole.VERIFIER
    if text.startswith("agent_reviewer_"):
        return AgentRole.REVIEWER
    if text.startswith("agent_summarizer_"):
        return AgentRole.SUMMARIZER
    if text.startswith("agent_tool_worker_") or text.startswith("subagent_tool_worker_"):
        return AgentRole.TOOL_WORKER
    if text.startswith("agent_tool_") or text == "agent_tool":
        return AgentRole.TOOL
    if text.startswith("agent_research_"):
        return AgentRole.RESEARCH
    return None


class HierarchyViolation(PermissionError):
    pass


class HierarchyPolicy:
    def __init__(self, registry: AgentRegistry | None = None, taskboard: TaskBoard | None = None) -> None:
        self.registry = registry
        self.taskboard = taskboard

    def assert_can_create_agent(self, actor_agent_id: str, target_role: AgentRole | str, *, task_id: str | None = None) -> None:
        role = self._role(actor_agent_id)
        if role != AgentRole.MAIN:
            self._violate(
                task_id,
                actor_agent_id,
                "create_agent",
                "Only MainAgent can create agents or subagents.",
                f"Request capacity from MainAgent before creating {self._role_value(target_role)}.",
            )

    def assert_can_execute_tool(
        self,
        actor_agent_id: str,
        tool_name: str,
        *,
        task_id: str | None = None,
        queue_job_id: str | None = None,
        assigned_worker_agent_id: str | None = None,
    ) -> None:
        role = self._role(actor_agent_id)
        if role != AgentRole.TOOL_WORKER:
            self._violate(
                task_id,
                actor_agent_id,
                f"execute_tool:{tool_name}",
                "Only ToolWorkerAgent can execute repository tools.",
                "Route tool work through QueueManager and assign a ToolWorkerAgent.",
            )
        if not queue_job_id:
            self._violate(
                task_id,
                actor_agent_id,
                f"execute_tool:{tool_name}",
                "Every tool execution must have a queue_job_id.",
                "Create and reserve a QueueManager job before execution.",
            )
        if assigned_worker_agent_id and actor_agent_id != assigned_worker_agent_id:
            self._violate(
                task_id,
                actor_agent_id,
                f"execute_tool:{tool_name}",
                f"ToolWorkerAgent must match assigned worker {assigned_worker_agent_id}.",
                "Run the job with the assigned worker.",
            )

    def assert_can_create_queue_job(self, actor_agent_id: str, *, task_id: str | None = None) -> None:
        role = self._role(actor_agent_id)
        if role not in {AgentRole.CODING, AgentRole.VERIFIER, AgentRole.TOOL}:
            self._violate(
                task_id,
                actor_agent_id,
                "create_queue_job",
                "Only CodingAgent, VerifierAgent, or delegated ToolAgent can create queue jobs.",
                "MainAgent should delegate queue work to a specialist agent.",
            )

    def assert_can_assign_worker(self, actor_agent_id: str, worker_agent_id: str, *, task_id: str | None = None) -> None:
        actor_role = self._role(actor_agent_id)
        worker_role = self._role(worker_agent_id)
        if actor_role not in {AgentRole.CODING, AgentRole.VERIFIER, AgentRole.TOOL} or worker_role != AgentRole.TOOL_WORKER:
            self._violate(
                task_id,
                actor_agent_id,
                "assign_worker",
                "Queue jobs must be assigned by a specialist to a ToolWorkerAgent.",
                "Ask MainAgent to create workers, then assign through QueueManager.",
            )

    def assert_can_finalize_task(self, actor_agent_id: str, *, task_id: str | None = None) -> None:
        if self._role(actor_agent_id) != AgentRole.SUMMARIZER:
            self._violate(
                task_id,
                actor_agent_id,
                "finalize_task",
                "SummarizerAgent finalizes only after ReviewerAgent approval.",
                "Delegate final summary to SummarizerAgent.",
            )

    def approve_tool_event(self, event: dict[str, Any]) -> dict[str, Any]:
        self.assert_can_execute_tool(
            str(event.get("agent_id") or ""),
            str(event.get("tool_name") or ""),
            task_id=str(event.get("task_id") or "") or None,
            queue_job_id=str(event.get("queue_job_id") or "") or None,
            assigned_worker_agent_id=str(event.get("assigned_worker_agent_id") or "") or None,
        )
        return {**event, "hierarchy_policy_result": "approved"}

    def _role(self, agent_id: str) -> AgentRole | None:
        return _role_from_agent_id(agent_id, self.registry)

    def _role_value(self, role: AgentRole | str) -> str:
        return role.value if isinstance(role, AgentRole) else str(role)

    def _violate(self, task_id: str | None, actor: str, action: str, expected: str, fix_hint: str) -> None:
        payload = {
            "actor_agent_id": actor,
            "attempted_action": action,
            "expected_route": expected,
            "fix_hint": fix_hint,
            "hierarchy_policy_result": "rejected",
        }
        if task_id and self.taskboard is not None:
            self.taskboard.record_hierarchy_violation(task_id, payload)
        raise HierarchyViolation(f"{expected} {fix_hint}")


@dataclass
class AgentFactory:
    registry: AgentRegistry
    policy: HierarchyPolicy
    taskboard: TaskBoard | None = None
    main_agent_id: str = ""

    def __post_init__(self) -> None:
        if not self.main_agent_id:
            self.main_agent_id = self.registry.find_by_role(AgentRole.MAIN).agent_id

    def create_agent(
        self,
        role: AgentRole,
        parent_agent_id: str,
        task_id: str,
        privileges: list[str] | None = None,
        budget: int = 0,
    ) -> AgentNode:
        self.policy.assert_can_create_agent(self.main_agent_id, role, task_id=task_id)
        node = self.registry.register(role, parent_agent_id=parent_agent_id, capabilities=privileges)
        self._record_budget(task_id, node.agent_id, budget, "agent_created")
        return node

    def create_subagent(
        self,
        parent_agent_id: str,
        role: AgentRole,
        task_id: str,
        privileges: list[str] | None = None,
        budget: int = 0,
    ) -> AgentNode:
        self.policy.assert_can_create_agent(self.main_agent_id, role, task_id=task_id)
        node = self.registry.create_subagent(role, parent_agent_id, privileges or [])
        if self.taskboard is not None:
            self.taskboard.assign_subagent(task_id, node.agent_id)
        self._record_budget(task_id, node.agent_id, budget, "subagent_created")
        return node

    def deactivate_agent(self, agent_id: str, reason: str) -> None:
        self.registry.deactivate(agent_id)
        if self.taskboard is not None:
            for task in self.taskboard.tasks.values():
                if agent_id in task.assigned_agent_ids or agent_id in task.assigned_subagent_ids:
                    self.taskboard.add_evidence(task.task_id, f"MainAgent deactivated {agent_id}: {reason}")

    def resize_pool(self, role: AgentRole, target_count: int, reason: str) -> None:
        if self.taskboard is not None:
            for task in self.taskboard.tasks.values():
                self.taskboard.add_evidence(task.task_id, f"MainAgent resized {role.value} pool to {target_count}: {reason}")

    def _record_budget(self, task_id: str, agent_id: str, budget: int, action: str) -> None:
        if self.taskboard is None:
            return
        self.taskboard.record_budget(
            task_id,
            {
                "agent_id": agent_id,
                "approved_by_agent_id": self.main_agent_id,
                "budget_reserved_tokens": int(budget or 0),
                "action": action,
            },
        )
