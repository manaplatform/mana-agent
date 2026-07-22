from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from pathlib import Path
from typing import Any

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
from mana_agent.multi_agent.core.types import AgentRole, AgentState, GitIntent, QueueJob, QueueJobStatus, QueueJobType, RiskLevel, TaskStatus
from mana_agent.multi_agent.queue.queue_manager import QueueManager
from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
from mana_agent.multi_agent.routing.hierarchy import AgentFactory, HierarchyPolicy
from mana_agent.multi_agent.routing.router import Router
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.multi_agent.memory.memory_bundle import AgentMemoryBundle
from mana_agent.multi_agent.memory.repo_context import RepoContext
from mana_agent.multi_agent.memory.task_memory import TaskMemory
from mana_agent.memory import MultiAgentMemoryService
from mana_agent.workspaces.routing import RepositoryScopeDecisionEngine, ScopeDecisionError
from mana_agent.workspaces.service import WorkspaceService
from mana_agent.multi_agent.worktrees import (
    WorkspaceError,
    WorkspaceManager,
    coding_route_requires_worktree,
    review_task_branch,
)
from mana_agent.config.settings import Settings
from mana_agent.builtin_skills.skill_creator import (
    ExperienceRecord,
    ExperienceWorkshopHook,
    SkillCreator,
    SkillDraft,
    WorkshopConfig,
)
from mana_agent.services.execution_event_hub import get_execution_event_hub

@dataclass
class MainAgentResult:
    task_id: str
    route_name: str
    task_size: str
    answer: str
    required_agents: list[str]
    required_subagents: list[str]
    repository_ids: list[str] | None = None


class _MainAgentSkillDraftGenerator:
    """Adapter around the existing model selected for the task lifecycle."""

    def __init__(self, llm: Any) -> None:
        self.model = llm.with_structured_output(SkillDraft)

    def generate(self, experience: ExperienceRecord, decision) -> SkillDraft:  # noqa: ANN001
        prompt = (
            "As Mana-Agent's trusted built-in skill-creator, make a generalized reusable procedure from the recorded, verified experience below. "
            "Return the SkillDraft schema. Do not include repository-specific paths, source lines, secrets, credentials, private data, or instructions that install the proposal, alter validation, or alter approval records. "
            "Declare only supported permissions and include deterministic verification and bounded failure recovery.\n\n"
            f"Eligibility:\n{decision.model_dump_json(indent=2)}\n\n"
            f"Recorded evidence:\n{experience.model_dump_json(indent=2)}"
        )
        value = self.model.invoke(prompt)
        return value if isinstance(value, SkillDraft) else SkillDraft.model_validate(value)


class MainAgent:
    def __init__(
        self,
        root: str | Path = ".",
        *,
        routing_llm: Any | None = None,
        session_id: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        requested_root = Path(root).resolve()
        self.routing_llm = routing_llm
        self.workspace_service = WorkspaceService()
        try:
            self.workspace_context = self.workspace_service.context_for_session(str(session_id)) if session_id else None
        except (FileNotFoundError, ValueError):
            self.workspace_context = None
        preparation_path = (
            Path(self.workspace_context.session.cwd)
            if self.workspace_context is not None
            else requested_root
        )
        prepared = self.workspace_service.prepare_repository(
            preparation_path,
            allow_create=self.workspace_context is None,
            initialize_if_missing=True,
            expected_workspace_id=(
                self.workspace_context.workspace.workspace_id
                if self.workspace_context is not None
                else workspace_id
            ),
            entry_point="multi-agent-main",
        )
        self.root = prepared.working_directory
        if self.workspace_context is None:
            session = (
                self.workspace_service.create_session(
                    self.root,
                    workspace_id=workspace_id,
                    session_id=session_id,
                )
                if session_id
                else self.workspace_service.restore_or_create_session(
                    self.root,
                    workspace_id=workspace_id,
                )
            )
            self.workspace_context = self.workspace_service.context_for_session(session.session_id)
        self.scope_engine = RepositoryScopeDecisionEngine(routing_llm)
        self.memory_service = MultiAgentMemoryService(
            root=self.root,
            workspace_id=self.workspace_context.workspace.workspace_id,
            repository_id=self.workspace_context.session.primary_repository_id,
            session_id=self.workspace_context.session.session_id,
        )
        self.memory = AgentMemoryBundle(
            repo_context=RepoContext(root=str(self.root)),
            task_memory=TaskMemory(),
            service=self.memory_service,
        )
        self.taskboard = TaskBoard(self.root, memory_service=self.memory_service)
        self.message_bus = MessageBus(self.root)
        self.registry = AgentRegistry()
        self.router = Router(llm=routing_llm)
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
        settings = Settings()
        self.managed_worktrees_enabled = bool(
            getattr(settings, "mana_managed_worktrees_enabled", True)
            and prepared.repository.head_sha
        )
        try:
            self.workspace_manager = WorkspaceManager(
                prepared.repository_root,
                repository_id=self.workspace_context.session.primary_repository_id,
                enabled=self.managed_worktrees_enabled,
            )
            if self.managed_worktrees_enabled:
                self.workspace_manager.reconcile()
        except WorkspaceError:
            self.workspace_manager = None  # type: ignore[assignment]
        self.decision_room = DecisionRoom(self.root, self.taskboard, self.message_bus)
        self.agents = self._build_agents()

    def run_user_request(self, user_request: str, *, entrypoint: str = "chat") -> MainAgentResult:
        request = str(user_request or "").strip()
        try:
            scope = self.scope_engine.decide(request=request, context=self.workspace_context)
        except ScopeDecisionError as exc:
            return MainAgentResult("", "blocked", "scope_error", str(exc), [], [], [])
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
            workspace_id=scope.workspace_id,
            session_id=scope.session_id,
            primary_repository_id=scope.primary_repository_id,
            repository_ids=scope.repository_ids,
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
            return MainAgentResult(task.task_id, "skipped", "duplicate", answer, [], [], scope.repository_ids)
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
        git_intent = self._git_intent_from_request(request)
        if git_intent is not None:
            route = self._route_with_git_contract(route, git_intent)
            task.risk_level = RiskLevel.HIGH
            task.required_capabilities = list(route.required_capabilities)
            self.taskboard.add_evidence(task.task_id, f"GitIntent contract established: {git_intent}")
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
        if len(scope.repository_ids) > 1:
            if not worker_ids:
                worker_ids = self._ensure_tool_workers(task.task_id, target_count=1)
            self._run_multi_repository_context(
                task.task_id,
                request,
                scope.repository_ids,
                worker_ids[0] if worker_ids else self.queue_manager.default_worker_agent_id,
            )
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
        managed_workspace = None
        if (
            self.managed_worktrees_enabled
            and self.workspace_manager is not None
            and coding_route_requires_worktree(route.route_name)
            and git_intent is None
        ):
            # Git intent workflows operate on the user's primary checkout with the existing safety model.
            # Mutation-oriented coding/tool routes get an isolated managed worktree.
            managed_workspace = self._allocate_managed_workspace(
                task.task_id,
                title=title,
                assigned_agent_id=self.registry.find_by_role(AgentRole.CODING).agent_id,
            )
        if route.route_name == "analyze":
            self._agent(AgentRole.RESEARCH, ResearchAgent).collect_evidence(task.task_id, "Analyze flow delegated to existing analyzer after multi-agent route creation.")
        if route.route_name in {"coding", "tool", "high_risk_tool"}:
            self.taskboard.add_evidence(task.task_id, "QueueManager is the only approved tool execution path.")
            if managed_workspace is not None and self.workspace_manager is not None:
                try:
                    self.workspace_manager.mark_running(
                        task.task_id,
                        agent_id=self.registry.find_by_role(AgentRole.CODING).agent_id,
                    )
                    self._sync_task_workspace_fields(task.task_id)
                except WorkspaceError as exc:
                    self.taskboard.add_blocker(task.task_id, f"Managed workspace failed to enter running: {exc}")
            if git_intent is not None:
                self._delegate_git_intent_work(task.task_id, git_intent)
            else:
                self._delegate_initial_tool_work(task.task_id, request, route.route_name)
        if route.risk_level.value in {"medium", "high"} or len(route.required_agents) > 4:
            self._agent(AgentRole.REVIEWER, ReviewerAgent).review(task.task_id, f"Risk level is {route.risk_level.value}; route requires {len(route.required_agents)} agents.")
        verification_passed: bool | None = None
        if route.requires_verification:
            self.taskboard.update_status(task.task_id, TaskStatus.VERIFYING, reason="VerifierAgent executes verification queue jobs.")
            if managed_workspace is not None and self.workspace_manager is not None:
                try:
                    self.workspace_manager.mark_verifying(
                        task.task_id,
                        agent_id=self.registry.find_by_role(AgentRole.VERIFIER).agent_id,
                    )
                    self._sync_task_workspace_fields(task.task_id)
                except WorkspaceError:
                    pass
            verifier = self._agent(AgentRole.VERIFIER, VerifierAgent)
            verification_commands = self._verification_commands(plan.verification_commands)
            if git_intent is not None:
                verification = verifier.execute_git_verification(task.task_id, wants_push=git_intent.wants_push, target_branch=git_intent.target_branch)
            else:
                verification = verifier.execute_verification(task.task_id, verification_commands)
            verification_passed = bool(verification.passed)
            self.memory.remember_agent(
                verifier.agent_id,
                f"Recorded verification for task {task.task_id}: passed={verification.passed}; {verification.summary}",
            )
            self.memory.remember_task(
                f"Verification recorded: passed={verification.passed}; summary={verification.summary}"
            )
            if not verification.passed:
                self._agent(AgentRole.REVIEWER, ReviewerAgent).reject_weak_evidence(task.task_id, verification.summary)
                if managed_workspace is not None and self.workspace_manager is not None:
                    try:
                        self.workspace_manager.mark_failed(task.task_id, error=verification.summary, retain=True)
                        self._sync_task_workspace_fields(task.task_id)
                    except WorkspaceError:
                        pass
        reviewer = self._agent(AgentRole.REVIEWER, ReviewerAgent)
        approved = reviewer.review_evidence(task.task_id, route_name=route.route_name, requires_verification=route.requires_verification)
        if approved and managed_workspace is not None and self.workspace_manager is not None:
            try:
                branch_review = review_task_branch(
                    self.workspace_manager,
                    task.task_id,
                    reviewer_agent_id=reviewer.agent_id,
                    verification_passed=verification_passed if route.requires_verification else True,
                    hierarchy_ok=not bool(self.taskboard.get_task(task.task_id).hierarchy_violations),
                    extra_blockers=list(self.taskboard.get_task(task.task_id).blockers),
                )
                self.taskboard.add_evidence(task.task_id, branch_review.get("summary") or "Managed branch review completed.")
                if not branch_review.get("approved"):
                    approved = False
                    self.taskboard.add_blocker(
                        task.task_id,
                        f"Managed workspace review rejected: {branch_review.get('summary') or 'not approved'}",
                    )
                else:
                    self.taskboard.add_evidence(
                        task.task_id,
                        f"Merge candidate ready on {branch_review.get('branch')} "
                        f"(base {str(branch_review.get('base_revision') or '')[:12]}). "
                        "No automatic merge into the default branch was performed.",
                    )
                self._sync_task_workspace_fields(task.task_id)
            except WorkspaceError as exc:
                approved = False
                self.taskboard.add_blocker(task.task_id, f"Managed workspace review failed: {exc}")
        self._deactivate_subagents(task.task_id, subagent_ids + worker_ids)
        task_after_review = self.taskboard.get_task(task.task_id)
        if git_intent is not None and task_after_review.blockers:
            approved = False
        if approved:
            done_reason = "Multi-agent hierarchy completed and reviewer approved evidence."
            if managed_workspace is not None:
                done_reason += " Managed worktree is a merge candidate; explicit merge intent is still required."
            self.taskboard.update_status(task.task_id, TaskStatus.DONE, reason=done_reason)
            answer = self._agent(AgentRole.SUMMARIZER, SummarizerAgent).summarize(task.task_id)
        else:
            self.taskboard.update_status(task.task_id, TaskStatus.BLOCKED, reason="Reviewer rejected weak or incomplete hierarchy evidence.")
            answer = self._agent(AgentRole.SUMMARIZER, SummarizerAgent).summarize(task.task_id)
            if managed_workspace is not None and self.workspace_manager is not None:
                try:
                    current = self.workspace_manager.get_for_task(task.task_id)
                    if current.status.value not in {"retained", "failed", "merge_candidate", "merged"}:
                        self.workspace_manager.mark_interrupted(task.task_id, error="review rejected or blocked")
                    self._sync_task_workspace_fields(task.task_id)
                except WorkspaceError:
                    pass
        self.memory.remember_task(f"Final summary produced for task {task.task_id}: {answer[:500]}")
        workshop = self._run_experience_workshop(task.task_id, answer, approved=approved)
        if workshop and workshop.proposal_result and workshop.proposal_result.proposal:
            proposal = workshop.proposal_result.proposal
            answer += (
                "\n\nExperience candidate detected\n"
                f"Reusable workflow: {proposal.display_name}\n"
                f"Confidence: {proposal.confidence:.2f}\n"
                f"Proposal: {proposal.proposal_id}\n"
                "Review with `mana-agent skill proposal review " + proposal.proposal_id + "`."
            )
        return MainAgentResult(
            task.task_id,
            route.route_name,
            route.task_size,
            answer,
            route.required_agents,
            route.required_subagents,
            scope.repository_ids,
        )

    def _run_experience_workshop(self, task_id: str, summary: str, *, approved: bool):
        """Evaluate recorded completion facts without affecting task success."""
        try:
            config = WorkshopConfig.load()
            if not config.enabled or not config.auto_propose:
                return None
            task = self.taskboard.get_task(task_id)
            verification_rows = [asdict(item) for item in task.verification_results]
            verification_passed = bool(verification_rows) and all(bool(item.get("passed")) for item in verification_rows)
            changed_files = sorted(
                {
                    *task.files_touched,
                    *(path for job in self.queue_manager.jobs_for_task(task_id) for path in job.changed_files),
                }
            )
            decisions = [asdict(item) for item in self.memory_service.agent_decisions if item.task_id == task_id]
            tools = [asdict(item) for item in self.memory_service.tool_executions.values() if item.task_id == task_id]
            experience = ExperienceRecord(
                session_id=task.session_id or self.workspace_context.session.session_id,
                task_id=task_id,
                summary=task.user_request,
                result=summary,
                workflow_steps=list(task.plan),
                decisions=decisions,
                tool_calls=tools,
                changed_files=changed_files,
                verification_commands=list(task.verification_commands),
                verification_results=verification_rows,
                verification_passed=verification_passed,
                user_accepted=False,
                reusable_trigger_present=len(task.plan) >= 2,
                deterministic_verification=bool(task.verification_commands),
                repository_specificity="medium",
                unresolved_warnings=[*task.blockers, *(risk for row in verification_rows for risk in row.get("risks", []))],
                agent_ids=list(task.assigned_agent_ids),
                subagent_ids=list(task.assigned_subagent_ids),
                source_component="task_completion",
            )

            def event_sink(event_type: str, metadata: dict[str, object]) -> None:
                get_execution_event_hub().emit(
                    event_type,
                    title=event_type.replace("_", " ").title(),
                    conversation_id=task.session_id,
                    execution_id=task_id,
                    repository_id=task.primary_repository_id,
                    status="success" if event_type in {"skill_proposal_created", "skill_candidate_detected"} else "running",
                    metadata=metadata,
                )

            creator = SkillCreator(config=config, event_sink=event_sink)
            generator = _MainAgentSkillDraftGenerator(self.routing_llm) if self.routing_llm is not None else None
            return ExperienceWorkshopHook(creator).run(
                experience,
                generator=generator,
                original_task_succeeded=approved,
            )
        except Exception as exc:
            # The workshop is subordinate to the already completed task.
            get_execution_event_hub().emit(
                "skill_proposal_validation_failed",
                title="Skill proposal generation failed",
                conversation_id=self.workspace_context.session.session_id,
                execution_id=task_id,
                repository_id=self.workspace_context.session.primary_repository_id,
                message=str(exc),
                status="failed",
            )
            return None

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

    def _run_multi_repository_context(
        self,
        parent_task_id: str,
        request: str,
        repository_ids: list[str],
        worker_agent_id: str,
    ) -> None:
        """Fan out one repository-scoped context run per model-selected repo."""

        coding = self._agent(AgentRole.CODING, CodingAgent)
        for repository_id in repository_ids:
            repo = self.workspace_context.repositories[repository_id]
            child = self.taskboard.create_child_task(
                parent_task_id,
                title=f"Repository context: {repo.name}",
                user_request=request,
                owner_agent_id=coding.agent_id,
            )
            child.primary_repository_id = repository_id
            child.repository_ids = [repository_id]
            child.workspace_id = self.workspace_context.workspace.workspace_id
            child.session_id = self.workspace_context.session.session_id
            self.taskboard.save()
            self.taskboard.update_status(child.task_id, TaskStatus.ROUTED, reason="Model-selected repository scope.")
            self.taskboard.update_status(child.task_id, TaskStatus.IN_PROGRESS, reason="Repository worker is collecting context.")
            manager = QueueManager(
                Path(repo.canonical_path),
                taskboard=self.taskboard,
                memory_service=MultiAgentMemoryService(
                    root=repo.canonical_path,
                    workspace_id=self.workspace_context.workspace.workspace_id,
                    repository_id=repository_id,
                    session_id=self.workspace_context.session.session_id,
                ),
                hierarchy_policy=self.hierarchy_policy,
                default_worker_agent_id=worker_agent_id,
            )
            job = manager.enqueue(
                task_id=child.task_id,
                requested_by_agent_id=coding.agent_id,
                approved_by_agent_id=self.registry.find_by_role(AgentRole.MAIN).agent_id,
                job_type=QueueJobType.REPO_SEARCH,
                payload={"query": request[:120], "limit": 5, "repository_id": repository_id},
                purpose=f"Collect model-selected context from repository {repo.name}.",
            )
            manager.run_next(worker_agent_id=worker_agent_id)
            completed = manager.get_job(job.job_id)
            if completed.status == QueueJobStatus.DONE:
                self.taskboard.update_status(child.task_id, TaskStatus.DONE, reason="Repository context run completed.")
            else:
                self.taskboard.update_status(
                    child.task_id,
                    TaskStatus.FAILED,
                    reason=completed.error or "Repository context run failed.",
                )

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

    def _allocate_managed_workspace(self, task_id: str, *, title: str, assigned_agent_id: str):
        """Create or resume an isolated managed worktree for a coding task."""

        if self.workspace_manager is None:
            return None
        task = self.taskboard.get_task(task_id)
        try:
            workspace = self.workspace_manager.create_for_task(
                task_id,
                title=title or task.title,
                assigned_agent_id=assigned_agent_id,
                session_id=task.session_id,
                multi_agent_workspace_id=task.workspace_id,
                reuse_existing=True,
            )
        except WorkspaceError as exc:
            self.taskboard.add_blocker(task_id, f"Managed worktree allocation failed: {exc}")
            self.taskboard.add_evidence(task_id, f"Managed worktree allocation failed safely: {exc}")
            return None
        self.workspace_manager.attach_to_taskboard(task, workspace)
        self.taskboard.save()
        self.taskboard.add_evidence(
            task_id,
            f"Managed worktree ready: branch={workspace.branch_name} path={workspace.worktree_path} "
            f"base={workspace.base_revision[:12]}",
        )
        # Point queue tools at the isolated worktree for this task without changing process cwd.
        if str(getattr(self.queue_manager, "root", "") or "") and Path(workspace.worktree_path).is_dir():
            # Keep QueueManager.source root as the primary checkout for discovery; per-job
            # execution_repo_root from the task drives ToolsManager.
            pass
        return workspace

    def _sync_task_workspace_fields(self, task_id: str) -> None:
        if self.workspace_manager is None:
            return
        try:
            workspace = self.workspace_manager.get_for_task(task_id)
        except WorkspaceError:
            return
        task = self.taskboard.get_task(task_id)
        self.workspace_manager.attach_to_taskboard(task, workspace)
        self.taskboard.save()

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

    def _delegate_git_intent_work(self, task_id: str, intent: GitIntent) -> None:
        coding = self._agent(AgentRole.CODING, CodingAgent)
        inspections = [
            ["status", "--short", "--branch"],
            ["branch", "--show-current"],
            ["remote", "-v"],
            ["diff", "--stat"],
            ["diff"],
            ["diff", "--cached", "--stat"],
            ["log", "-1", "--oneline"],
        ]
        results = [self._run_git_job(task_id, coding.agent_id, args, purpose=f"Inspect Git state: git {' '.join(args)}") for args in inspections]
        if any(job.status == QueueJobStatus.FAILED and _git_args(job)[:1] == ["status"] for job in results):
            self.taskboard.add_blocker(task_id, "Git workflow blocked: repository status inspection failed or target is not a Git repository.")
            return

        status = _stdout_for(results, ["status", "--short", "--branch"])
        current_branch = _stdout_for(results, ["branch", "--show-current"]).strip()
        remotes = _stdout_for(results, ["remote", "-v"]).strip()
        diff_stat = _stdout_for(results, ["diff", "--stat"])
        diff = _stdout_for(results, ["diff"])
        status_paths = _status_paths(status)
        if _has_conflicts(status):
            self.taskboard.add_blocker(task_id, "Git workflow blocked: conflicts are present in repository status.")
            return
        if intent.wants_branch:
            self._handle_branch_intent(task_id, coding.agent_id, intent, status_paths)
            return
        if intent.wants_commit:
            if not status_paths:
                self.taskboard.add_blocker(task_id, "Git commit result: no changes to commit.")
            elif _has_untracked(status):
                self.taskboard.add_blocker(task_id, "Git commit blocked: untracked files are present and were not selected for staging.")
            else:
                paths = sorted(status_paths)
                message = intent.commit_message or self._commit_message_from_diff(diff_stat=diff_stat, diff=diff, paths=paths)
                intent.commit_message = message
                self.taskboard.add_evidence(task_id, f"Git commit message generated from diff: {message}")
                self._run_git_job(task_id, coding.agent_id, ["add", "--", *paths], purpose=f"Stage inspected Git paths: {', '.join(paths)}")
                self._run_git_job(task_id, coding.agent_id, ["diff", "--cached", "--stat"], purpose="Inspect staged Git diff stat before commit.")
                self._run_git_job(task_id, coding.agent_id, ["diff", "--cached"], purpose="Inspect staged Git diff before commit.")
                committed = self._run_git_job(task_id, coding.agent_id, ["commit", "-m", message], purpose="Create Git commit with diff-derived message.")
                if committed.status != QueueJobStatus.DONE:
                    self.taskboard.add_blocker(task_id, f"Git commit blocked: {committed.error or committed.result_summary or 'commit failed'}")
                    return
        if intent.wants_push:
            self._handle_push_intent(task_id, coding.agent_id, intent, current_branch=current_branch, remotes=remotes)

    def _handle_branch_intent(self, task_id: str, agent_id: str, intent: GitIntent, status_paths: set[str]) -> None:
        if status_paths:
            self.taskboard.add_blocker(task_id, "Git branch creation blocked: working tree has local changes.")
            return
        branch = intent.target_branch or ""
        if not branch:
            self.taskboard.add_blocker(task_id, "Git branch creation blocked: target branch was not selected by the model decision.")
            return
        created = self._run_git_job(task_id, agent_id, ["switch", "-c", branch], purpose=f"Create and switch to Git branch {branch}.")
        if created.status != QueueJobStatus.DONE:
            self.taskboard.add_blocker(task_id, f"Git branch creation blocked: {created.error or created.result_summary or 'branch command failed'}")

    def _handle_push_intent(self, task_id: str, agent_id: str, intent: GitIntent, *, current_branch: str, remotes: str) -> None:
        target = intent.target_branch or current_branch
        if not remotes:
            self.taskboard.add_blocker(task_id, "Git push blocked: no remote exists.")
            return
        if target and current_branch and current_branch != target:
            self.taskboard.add_blocker(task_id, f"Git push blocked: current branch is {current_branch}, target branch is {target}.")
            return
        upstream = self._run_git_job(task_id, agent_id, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], purpose="Inspect Git upstream before push.")
        upstream_name = str((upstream.result or {}).get("stdout") or "").strip() if upstream.result else ""
        compare_ref = upstream_name or (f"origin/{target}" if target else "")
        if compare_ref:
            divergence = self._run_git_job(task_id, agent_id, ["rev-list", "--left-right", "--count", f"{compare_ref}...HEAD"], purpose="Inspect Git ahead/behind state before push.")
            counts = str((divergence.result or {}).get("stdout") or "").strip().split() if divergence.result else []
            if len(counts) >= 2:
                behind, ahead = int(counts[0]), int(counts[1])
                if behind and ahead:
                    self.taskboard.add_blocker(task_id, "Git push blocked: branch is diverged from remote.")
                    return
                if behind:
                    self.taskboard.add_blocker(task_id, "Git push blocked: branch is behind remote.")
                    return
        pushed = self._run_git_job(task_id, agent_id, ["push", "origin", target or current_branch], purpose=f"Push Git branch {target or current_branch} to origin.")
        if pushed.status != QueueJobStatus.DONE:
            self.taskboard.add_blocker(task_id, f"Git push blocked: {pushed.error or pushed.result_summary or 'push failed'}")

    def _run_git_job(self, task_id: str, requested_by_agent_id: str, args: list[str], *, purpose: str) -> QueueJob:
        job = self.queue_manager.enqueue(
            task_id=task_id,
            requested_by_agent_id=requested_by_agent_id,
            approved_by_agent_id=self.registry.find_by_role(AgentRole.MAIN).agent_id,
            job_type=QueueJobType.GIT,
            payload={"tool": "git.generic", "args": {"args": args}},
            purpose=purpose,
            priority=70,
            requires_write_lock=args[:1] in (["add"], ["commit"], ["push"], ["switch"], ["checkout"], ["branch"]),
        )
        self.taskboard.add_evidence(task_id, f"CodingAgent created Git queue job {job.job_id}: git {' '.join(args)}")
        self.queue_manager.run_next(worker_agent_id=job.assigned_worker_agent_id)
        return job

    def _git_intent_from_request(self, request: str) -> GitIntent | None:
        text = str(request or "").strip().lower()
        action_re = r"\b(commit|push|branch|checkout|switch|merge|rebase|tag|release)\b"
        if not re.search(action_re, text):
            return None
        wants_commit = bool(re.search(r"\bcommit\b", text))
        wants_push = bool(re.search(r"\bpush\b", text))
        wants_branch = bool(re.search(r"\b(branch|checkout|switch)\b", text))
        target = "main" if re.search(r"\bmain\b", text) else None
        branch_match = re.search(r"\b(?:branch|checkout|switch)\s+(?:to\s+|new\s+|create\s+|create\s+new\s+)?([A-Za-z0-9._/-]+)", text)
        if wants_branch and branch_match:
            target = branch_match.group(1)
        return GitIntent(
            wants_status=True,
            wants_diff=True,
            wants_commit=wants_commit,
            wants_push=wants_push,
            wants_branch=wants_branch,
            target_branch=target,
            commit_message=None,
            requires_remote=wants_push,
            risk_level="high",
        )

    def _route_with_git_contract(self, route, intent: GitIntent):
        capabilities = ["repo_state", "git_status", "git_diff", "verification"]
        if intent.wants_commit:
            capabilities.append("git_commit")
        if intent.wants_push:
            capabilities.append("git_push")
        if intent.wants_branch:
            capabilities.append("git_branch")
        route.route_name = "high_risk_tool"
        route.task_size = "medium"
        route.required_agents = ["main", "head_decision", "tool", "verifier", "reviewer", "summarizer"]
        route.required_capabilities = capabilities
        route.requires_discussion = True
        route.requires_verification = True
        route.risk_level = RiskLevel.HIGH
        route.reason_summary = "Git intent requires repository-state inspection, queued Git execution, verification, and review."
        return route

    def _commit_message_from_diff(self, *, diff_stat: str, diff: str, paths: list[str]) -> str:
        primary = Path(paths[0]).stem.replace("_", "-").replace(" ", "-") if paths else "repository"
        changed_lines = [line for line in str(diff).splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
        scope = primary[:40] or "repository"
        verb = "update" if changed_lines else "record"
        if any(path.lower().endswith((".md", ".rst", ".txt")) for path in paths):
            return f"docs: {verb} {scope}"
        if "test" in " ".join(paths).lower():
            return f"test: {verb} {scope}"
        if diff_stat:
            return f"chore: {verb} {scope}"
        return f"chore: record {scope} changes"

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


def _git_args(job: QueueJob) -> list[str]:
    nested = job.payload.get("args") if isinstance(job.payload.get("args"), dict) else {}
    raw = nested.get("args") if isinstance(nested, dict) else None
    return [str(item) for item in raw] if isinstance(raw, list) else []


def _stdout_for(jobs: list[QueueJob], args: list[str]) -> str:
    for job in jobs:
        if _git_args(job) == args:
            return str((job.result or {}).get("stdout") or "")
    return ""


def _status_paths(status: str) -> set[str]:
    paths: set[str] = set()
    for line in str(status or "").splitlines():
        if not line or line.startswith("##"):
            continue
        path = line[3:] if len(line) > 3 else line
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip()
        if path and not path.startswith(".mana/"):
            paths.add(path)
    return paths


def _has_untracked(status: str) -> bool:
    for line in str(status or "").splitlines():
        if not line.startswith("??"):
            continue
        path = line[3:].strip() if len(line) > 3 else ""
        if not path.startswith(".mana/"):
            return True
    return False


def _has_conflicts(status: str) -> bool:
    conflict_codes = {"UU", "AA", "DD", "AU", "UA", "DU", "UD"}
    for line in str(status or "").splitlines():
        if line[:2] in conflict_codes:
            return True
    return False
