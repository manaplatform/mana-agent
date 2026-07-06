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
from mana_agent.multi_agent.core.types import AgentRole, AgentState, QueueJobType, TaskStatus
from mana_agent.multi_agent.queue.queue_manager import QueueManager
from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
from mana_agent.multi_agent.routing.hierarchy import AgentFactory, HierarchyPolicy
from mana_agent.multi_agent.routing.router import Router
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.multi_agent.memory.memory_bundle import AgentMemoryBundle
from mana_agent.multi_agent.memory.repo_context import RepoContext
from mana_agent.multi_agent.memory.task_memory import TaskMemory
from mana_agent.services.memory_service import MultiAgentMemoryService

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
        self.memory_service = MultiAgentMemoryService(root=self.root)
        self.memory = AgentMemoryBundle(
            repo_context=RepoContext(root=str(self.root)),
            task_memory=TaskMemory(),
            service=self.memory_service,
        )
        self.taskboard = TaskBoard(self.root, memory_service=self.memory_service)
        self.message_bus = MessageBus(self.root)
        self.registry = AgentRegistry()
        self.router = Router()
        self.hierarchy_policy = HierarchyPolicy(self.registry, self.taskboard)
        main_node = self.registry.find_by_role(AgentRole.MAIN)
        self.agent_factory = AgentFactory(
            registry=self.registry,
            policy=self.hierarchy_policy,
            taskboard=self.taskboard,
            main_agent_id=main_node.agent_id,
        )
        self.queue_manager = QueueManager(
            self.root,
            taskboard=self.taskboard,
            memory_service=self.memory_service,
            hierarchy_policy=self.hierarchy_policy,
        )
        self.decision_room = DecisionRoom(self.root, self.taskboard, self.message_bus)
        self.agents = self._build_agents()

    def run_user_request(self, user_request: str, *, entrypoint: str = "chat") -> MainAgentResult:
        request = str(user_request or "").strip()
        self.memory.remember_task(f"User request received via {entrypoint}: {request[:500]}")
        self.memory.remember_repo_fact(f"Repository root: {self.root}")
        title = request[:80] or entrypoint
        main_node = self.registry.find_by_role(AgentRole.MAIN)
        self.memory.remember_agent(
            main_node.agent_id,
            f"Main agent received request via {entrypoint}: {request[:500]}",
        )
        self.memory_service.record_decision(
            agent_id=main_node.agent_id,
            task_id="pending",
            decision_type="main_request_received",
            input_summary=request,
            memory_used=[],
            decision="create_or_reuse_task",
            reason="main agent checks memory before task creation",
        )
        task = self.taskboard.create_task(
            title=title,
            user_request=request,
            normalized_goal=f"{entrypoint}: {request}",
            owner_agent_id=main_node.agent_id,
        )
        self.taskboard.record_budget(
            task.task_id,
            {
                "agent_id": main_node.agent_id,
                "approved_by_agent_id": main_node.agent_id,
                "budget_reserved_tokens": self._budget_for_size("simple"),
                "budget_reserved_ms": 120_000,
                "action": "root_task_budget",
            },
        )
        duplicate_of = str(task.memory_status.get("duplicate_of") or "")
        if duplicate_of:
            self.memory_service.update_task(
                task.task_id,
                status=TaskStatus.SKIPPED.value,
                result_summary=f"duplicate_of:{duplicate_of}",
            )
            answer = f"Skipped duplicate task; reused existing task {duplicate_of}."
            self.memory.remember_task(answer)
            return MainAgentResult(task.task_id, "skipped", "duplicate", answer, [], [])
        self.memory_service.record_decision(
            agent_id=main_node.agent_id,
            task_id=task.task_id,
            decision_type="route_request",
            input_summary=request,
            memory_used=[str(task.memory_status.get("memory_bundle_id") or "")],
            decision="query_router",
            reason="route after task duplicate and bundle checks",
        )
        route = self.router.route(task_id=task.task_id, user_request=f"{entrypoint} {request}")
        self.taskboard.record_budget(
            task.task_id,
            {
                "agent_id": main_node.agent_id,
                "approved_by_agent_id": main_node.agent_id,
                "budget_reserved_tokens": self._budget_for_size(route.task_size),
                "budget_reserved_ms": 120_000,
                "action": "route_budget_estimate",
            },
        )
        self.memory.remember_task(
            "Route selected: "
            f"{route.route_name}; size={route.task_size}; "
            f"agents={', '.join(route.required_agents)}; "
            f"subagents={', '.join(route.required_subagents)}"
        )
        self.taskboard.add_evidence(task.task_id, f"HeadDecisionAgent classified task size as {route.task_size}.")
        for role_name in route.required_agents:
            node = self._node_by_role_name(role_name)
            if node is not None:
                self.taskboard.assign(task.task_id, node.agent_id)
                self.memory.remember_agent(
                    node.agent_id,
                    f"Assigned to task {task.task_id} for route {route.route_name}",
                )
                if node.model_level:
                    self.taskboard.add_evidence(task.task_id, f"{node.agent_id} uses {node.model_level}.")
        subagent_ids = self._create_required_subagents(task.task_id, route.required_subagents)
        worker_ids: list[str] = []
        if route.route_name in {"coding", "tool", "high_risk_tool"} or route.requires_verification:
            worker_ids = self._ensure_tool_workers(task.task_id, target_count=1)
            if worker_ids:
                self.queue_manager.default_worker_agent_id = worker_ids[0]
        head = self._agent(AgentRole.HEAD_DECISION, HeadDecisionAgent)
        head.decide(task.task_id, route, self.decision_room)
        self.memory.remember_agent(
            head.agent_id,
            f"Decided route {route.route_name} for task {task.task_id}: {route.reason_summary}",
        )
        planner = self._agent(AgentRole.PLANNER, PlannerAgent)
        plan = planner.plan(task.task_id, request, route.route_name)
        self.memory.remember_agent(
            planner.agent_id,
            f"Created plan for task {task.task_id}; verification commands: "
            f"{', '.join(getattr(plan, 'verification_commands', []) or [])}",
        )
        self.memory.remember_task(
            f"Plan created for task {task.task_id}; verification commands: "
            f"{', '.join(getattr(plan, 'verification_commands', []) or [])}"
        )
        self.taskboard.update_status(task.task_id, TaskStatus.IN_PROGRESS, reason="Specialist agents are handling the routed workflow.")
        if route.route_name == "analyze":
            self._agent(AgentRole.RESEARCH, ResearchAgent).collect_evidence(task.task_id, "Analyze flow delegated to existing analyzer after multi-agent route creation.")
        if route.route_name in {"coding", "tool", "high_risk_tool"}:
            self.taskboard.add_evidence(task.task_id, "QueueManager is the only approved tool execution path.")
            self._delegate_initial_tool_work(task.task_id, request, route.route_name)
        if route.risk_level.value in {"medium", "high"} or len(route.required_agents) > 4:
            self._agent(AgentRole.REVIEWER, ReviewerAgent).review(task.task_id, f"Risk level is {route.risk_level.value}; route requires {len(route.required_agents)} agents.")
        if route.requires_verification:
            self.taskboard.update_status(task.task_id, TaskStatus.VERIFYING, reason="VerifierAgent executes verification queue jobs.")
            verifier = self._agent(AgentRole.VERIFIER, VerifierAgent)
            verification_commands = self._verification_commands(plan.verification_commands)
            verification = verifier.execute_verification(task.task_id, verification_commands)
            self.memory.remember_agent(
                verifier.agent_id,
                f"Recorded verification for task {task.task_id}: passed={verification.passed}; {verification.summary}",
            )
            self.memory.remember_task(
                f"Verification recorded: passed={verification.passed}; summary={verification.summary}"
            )
            if not verification.passed:
                self._agent(AgentRole.REVIEWER, ReviewerAgent).reject_weak_evidence(task.task_id, verification.summary)
        reviewer = self._agent(AgentRole.REVIEWER, ReviewerAgent)
        approved = reviewer.review_evidence(task.task_id, route_name=route.route_name, requires_verification=route.requires_verification)
        self._deactivate_subagents(task.task_id, subagent_ids + worker_ids)
        if approved:
            self.taskboard.update_status(task.task_id, TaskStatus.DONE, reason="Multi-agent hierarchy completed and reviewer approved evidence.")
            answer = self._agent(AgentRole.SUMMARIZER, SummarizerAgent).summarize(task.task_id)
        else:
            self.taskboard.update_status(task.task_id, TaskStatus.BLOCKED, reason="Reviewer rejected weak or incomplete hierarchy evidence.")
            answer = self._agent(AgentRole.SUMMARIZER, SummarizerAgent).summarize(task.task_id)
        self.memory.remember_task(f"Final summary produced for task {task.task_id}: {answer[:500]}")
        return MainAgentResult(task.task_id, route.route_name, route.task_size, answer, route.required_agents, route.required_subagents)

    def _create_required_subagents(self, task_id: str, subagent_names: list[str]) -> list[str]:
        if not subagent_names:
            return []
        parent = self.registry.find_by_role(AgentRole.CODING)
        created: list[str] = []
        for name in subagent_names:
            capabilities = [name, "repo_read"] if name == "repo_inventory" else [name]
            node = self.agent_factory.create_subagent(parent.agent_id, AgentRole.CODING, task_id, capabilities, budget=1000)
            created.append(node.agent_id)
            self.taskboard.add_evidence(task_id, f"MainAgent created {node.agent_id} for {name}.")
        return created

    def _ensure_tool_workers(self, task_id: str, *, target_count: int) -> list[str]:
        coding = self.registry.find_by_role(AgentRole.CODING)
        existing = [
            node.agent_id
            for node in self.registry.agents.values()
            if node.role == AgentRole.TOOL_WORKER and node.state != AgentState.DONE
        ]
        created: list[str] = []
        while len(existing) + len(created) < target_count:
            node = self.agent_factory.create_subagent(
                coding.agent_id,
                AgentRole.TOOL_WORKER,
                task_id,
                ["tool_execution"],
                budget=2000,
            )
            created.append(node.agent_id)
            self.taskboard.add_evidence(task_id, f"MainAgent created ToolWorkerAgent {node.agent_id}.")
        return existing + created

    def _delegate_initial_tool_work(self, task_id: str, request: str, route_name: str) -> None:
        coding = self._agent(AgentRole.CODING, CodingAgent)
        job = self.queue_manager.enqueue(
            task_id=task_id,
            requested_by_agent_id=coding.agent_id,
            approved_by_agent_id=self.registry.find_by_role(AgentRole.MAIN).agent_id,
            job_type=QueueJobType.REPO_SEARCH,
            payload={"query": request[:120] or route_name, "limit": 5},
            purpose="CodingAgent sniffs repository/task context before deciding further tool jobs.",
            priority=60,
        )
        self.taskboard.add_evidence(task_id, f"CodingAgent created queue job {job.job_id} for repository context sniffing.")
        self.queue_manager.run_next(worker_agent_id=job.assigned_worker_agent_id)

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
            if node.role in {AgentRole.CODING, AgentRole.TOOL, AgentRole.VERIFIER}:
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
                memory=self.memory,
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

    def _budget_for_size(self, task_size: str) -> int:
        return {
            "simple": 4000,
            "small": 8000,
            "medium": 20_000,
            "large": 40_000,
        }.get(str(task_size or "medium"), 20_000)

    def _verification_commands(self, commands: list[str]) -> list[str]:
        if (self.root / "src").exists():
            return ["python -m compileall src"]
        return ["python -m compileall ."]
