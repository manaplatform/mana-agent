from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mana_agent.multi_agent.agents.base_agent import BaseAgent
from mana_agent.multi_agent.agents.coding_agent import CodingAgent
from mana_agent.multi_agent.agents.head_decision_agent import HeadDecisionAgent
from mana_agent.multi_agent.agents.planner_agent import PlannerAgent
from mana_agent.multi_agent.agents.research_agent import ResearchAgent
from mana_agent.multi_agent.agents.reviewer_agent import ReviewerAgent
from mana_agent.multi_agent.agents.summarizer_agent import SummarizerAgent
from mana_agent.multi_agent.agents.tool_agent import ToolAgent
from mana_agent.multi_agent.agents.verifier_agent import VerifierAgent
from mana_agent.multi_agent.communication.decision_room import DecisionRoom
from mana_agent.multi_agent.communication.message_bus import MessageBus
from mana_agent.multi_agent.core.types import AgentRole, TaskStatus
from mana_agent.multi_agent.queue.queue_manager import QueueManager
from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
from mana_agent.multi_agent.routing.router import Router
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard


@dataclass
class MainAgentResult:
    task_id: str
    route_name: str
    task_size: str
    answer: str
    required_agents: list[str]
    required_subagents: list[str]


class MainAgent:
    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()
        self.taskboard = TaskBoard(self.root)
        self.message_bus = MessageBus(self.root)
        self.registry = AgentRegistry()
        self.router = Router()
        self.queue_manager = QueueManager(self.root, taskboard=self.taskboard)
        self.decision_room = DecisionRoom(self.root, self.taskboard, self.message_bus)
        self.agents = self._build_agents()

    def run_user_request(self, user_request: str, *, entrypoint: str = "chat") -> MainAgentResult:
        request = str(user_request or "").strip()
        title = request[:80] or entrypoint
        main_node = self.registry.find_by_role(AgentRole.MAIN)
        task = self.taskboard.create_task(
            title=title,
            user_request=request,
            normalized_goal=f"{entrypoint}: {request}",
            owner_agent_id=main_node.agent_id,
        )
        route = self.router.route(task_id=task.task_id, user_request=f"{entrypoint} {request}")
        self.taskboard.add_evidence(task.task_id, f"HeadDecisionAgent classified task size as {route.task_size}.")
        for role_name in route.required_agents:
            node = self._node_by_role_name(role_name)
            if node is not None:
                self.taskboard.assign(task.task_id, node.agent_id)
                if node.model_level:
                    self.taskboard.add_evidence(task.task_id, f"{node.agent_id} uses {node.model_level}.")
        subagent_ids = self._create_required_subagents(task.task_id, route.required_subagents)
        head = self._agent(AgentRole.HEAD_DECISION, HeadDecisionAgent)
        head.decide(task.task_id, route, self.decision_room)
        planner = self._agent(AgentRole.PLANNER, PlannerAgent)
        plan = planner.plan(task.task_id, request, route.route_name)
        self.taskboard.update_status(task.task_id, TaskStatus.IN_PROGRESS, reason="Specialist agents are handling the routed workflow.")
        if route.route_name == "analyze":
            self._agent(AgentRole.RESEARCH, ResearchAgent).collect_evidence(task.task_id, "Analyze flow delegated to existing analyzer after multi-agent route creation.")
        if route.route_name in {"coding", "tool", "high_risk_tool"}:
            self.taskboard.add_evidence(task.task_id, "QueueManager is the only approved tool execution path.")
        if route.risk_level.value in {"medium", "high"} or len(route.required_agents) > 4:
            self._agent(AgentRole.REVIEWER, ReviewerAgent).review(task.task_id, f"Risk level is {route.risk_level.value}; route requires {len(route.required_agents)} agents.")
        if route.requires_verification:
            self.taskboard.update_status(task.task_id, TaskStatus.VERIFYING, reason="VerifierAgent records verification plan.")
            verification = self._agent(AgentRole.VERIFIER, VerifierAgent).verify_no_mutation(task.task_id, plan.verification_commands)
            if not verification.passed:
                self._agent(AgentRole.REVIEWER, ReviewerAgent).reject_weak_evidence(task.task_id, verification.summary)
        self._deactivate_subagents(task.task_id, subagent_ids)
        self.taskboard.update_status(task.task_id, TaskStatus.DONE, reason="Multi-agent route completed; legacy entrypoint continues concrete command behavior.")
        answer = self._agent(AgentRole.SUMMARIZER, SummarizerAgent).summarize(task.task_id)
        return MainAgentResult(task.task_id, route.route_name, route.task_size, answer, route.required_agents, route.required_subagents)

    def _create_required_subagents(self, task_id: str, subagent_names: list[str]) -> list[str]:
        if not subagent_names:
            return []
        parent = self.registry.find_by_role(AgentRole.CODING)
        created: list[str] = []
        for name in subagent_names:
            capabilities = [name, "repo_read"] if name == "repo_inventory" else [name]
            node = self.registry.create_subagent(AgentRole.CODING, parent.agent_id, capabilities)
            created.append(node.agent_id)
            self.taskboard.assign_subagent(task_id, node.agent_id)
            self.taskboard.add_evidence(task_id, f"MainAgent created {node.agent_id} for {name}.")
        return created

    def _deactivate_subagents(self, task_id: str, subagent_ids: list[str]) -> None:
        for subagent_id in subagent_ids:
            self.registry.deactivate(subagent_id)
            self.taskboard.add_evidence(task_id, f"MainAgent deactivated {subagent_id}.")

    def _build_agents(self) -> dict[AgentRole, BaseAgent]:
        agents: dict[AgentRole, BaseAgent] = {}
        class_by_role = {
            AgentRole.MAIN: BaseAgent,
            AgentRole.HEAD_DECISION: HeadDecisionAgent,
            AgentRole.PLANNER: PlannerAgent,
            AgentRole.RESEARCH: ResearchAgent,
            AgentRole.CODING: CodingAgent,
            AgentRole.TOOL: ToolAgent,
            AgentRole.VERIFIER: VerifierAgent,
            AgentRole.REVIEWER: ReviewerAgent,
            AgentRole.SUMMARIZER: SummarizerAgent,
        }
        for node in self.registry.agents.values():
            cls = class_by_role[node.role]
            kwargs = {}
            if node.role in {AgentRole.CODING, AgentRole.TOOL}:
                kwargs["queue_manager"] = self.queue_manager
            agents[node.role] = cls(
                agent_id=node.agent_id,
                role=node.role,
                parent_agent_id=node.parent_agent_id,
                capabilities=node.capabilities,
                mailbox=self.message_bus,
                taskboard=self.taskboard,
                message_bus=self.message_bus,
                registry=self.registry,
                **kwargs,
            )
        return agents

    def _agent(self, role: AgentRole, cls):
        agent = self.agents[role]
        if not isinstance(agent, cls):
            raise TypeError(f"registered agent for {role.value} is not {cls.__name__}")
        return agent

    def _node_by_role_name(self, role_name: str):
        normalized = "head_decision" if role_name == "head_decision" else role_name
        for node in self.registry.agents.values():
            if node.role.value == normalized:
                return node
        return None
