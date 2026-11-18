from __future__ import annotations

from dataclasses import asdict, dataclass
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
from mana_agent.multi_agent.core.types import AgentRole, AgentState, GitIntent, QueueJob, QueueJobStatus, QueueJobType, RiskLevel, RouteDecision, TaskStatus
from mana_agent.multi_agent.queue.queue_manager import QueueManager
from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
from mana_agent.multi_agent.routing.hierarchy import AgentFactory, HierarchyPolicy
from mana_agent.multi_agent.routing.router import Router, RoutingDecisionError
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.multi_agent.memory.memory_bundle import AgentMemoryBundle
from mana_agent.multi_agent.memory.repo_context import RepoContext
from mana_agent.multi_agent.memory.task_memory import TaskMemory
from mana_agent.memory import MultiAgentMemoryService
from mana_agent.workspaces.routing import RepositoryScopeDecisionEngine, ScopeDecisionError
from mana_agent.workspaces.service import WorkspaceService
from mana_agent.workspaces.preparation import RepositoryValidationError
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
from mana_agent.services.activity_context import observed_activity
from mana_agent.execution_supervisor import ExecutionSupervisor, ExecutionSupervisorConfig, SideEffectClassification
from mana_agent.gateway.feature_integration import FeatureIntegrationCoordinator, WiringDecision, connected_wiring_path
from mana_agent.config.autonomous_config import AutonomousExecutionConfig
from mana_agent.execution.autonomous_controller import AutonomousExecutionController
from mana_agent.execution.continuation_evaluator import ContinuationEvaluator

@dataclass
class MainAgentResult:
    task_id: str
    route_name: str
    task_size: str
    answer: str
    required_agents: list[str]
    required_subagents: list[str]
    repository_ids: list[str] | None = None


from mana_agent.utils.text import extract_model_text


class _MainAgentSkillDraftGenerator:
    """Adapter around the existing model selected for the task lifecycle."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def generate(self, experience: ExperienceRecord, decision) -> SkillDraft:  # noqa: ANN001
        prompt = (
            "As Mana-Agent's trusted built-in skill-creator, make a generalized reusable procedure from the recorded, verified experience below. "
            "Return the SkillDraft schema. Do not include repository-specific paths, source lines, secrets, credentials, private data, or instructions that install the proposal, alter validation, or alter approval records. "
            "Declare only supported permissions and include deterministic verification and bounded failure recovery.\n\n"
            f"Eligibility:\n{decision.model_dump_json(indent=2)}\n\n"
            f"Recorded evidence:\n{experience.model_dump_json(indent=2)}"
        )
        return self.llm.decide_sync(prompt, SkillDraft)


class MainAgent:
    def __init__(
        self,
        root: str | Path = ".",
        *,
        routing_llm: Any | None = None,
        session_id: str | None = None,
        workspace_id: str | None = None,
        execution_supervisor: ExecutionSupervisor | None = None,
        initialize_if_missing: bool = True,
        coding_agent: Any | None = None,
        lane_coordinator: Any | None = None,
    ) -> None:
        requested_root = Path(root).resolve()
        self.routing_llm = routing_llm
        self.coding_engine = coding_agent
        self.repository_lanes = lane_coordinator
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
        prepared = None
        try:
            prepared = self.workspace_service.prepare_repository(
                preparation_path,
                allow_create=self.workspace_context is None,
                initialize_if_missing=initialize_if_missing,
                expected_workspace_id=(
                    self.workspace_context.workspace.workspace_id
                    if self.workspace_context is not None
                    else workspace_id
                ),
                entry_point="multi-agent-main",
            )
            self.root = prepared.working_directory
        except RepositoryValidationError:
            if initialize_if_missing:
                raise
            self.root = preparation_path
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
        supervisor_config = ExecutionSupervisorConfig.from_settings(Settings()).model_copy(
            update={"startup_recovery": False}
        )
        self.execution_supervisor = execution_supervisor or ExecutionSupervisor(config=supervisor_config)
        settings = Settings()
        self.managed_worktrees_enabled = bool(
            getattr(settings, "mana_managed_worktrees_enabled", True)
            and prepared is not None
            and prepared.repository.head_sha
        )
        try:
            self.workspace_manager = (
                WorkspaceManager(
                    prepared.repository_root,
                    repository_id=self.workspace_context.session.primary_repository_id,
                    enabled=self.managed_worktrees_enabled,
                )
                if prepared is not None
                else None
            )
            if self.managed_worktrees_enabled and self.workspace_manager is not None:
                self.workspace_manager.reconcile()
        except WorkspaceError:
            self.workspace_manager = None  # type: ignore[assignment]
        self.decision_room = DecisionRoom(self.root, self.taskboard, self.message_bus)
        self.agents = self._build_agents()
        self._agent(AgentRole.CODING, CodingAgent).routing_llm = routing_llm
        self.autonomous_controller = AutonomousExecutionController(
            config=AutonomousExecutionConfig(),
            supervisor=self.execution_supervisor,
            taskboard=self.taskboard,
            router=self.router,
            evaluator=ContinuationEvaluator(llm=routing_llm) if routing_llm is not None else None,
            global_memory=getattr(self.memory_service, "learning", None),
        )

    @observed_activity("MainAgent request")
    def run_user_request(
        self,
        user_request: str,
        *,
        entrypoint: str = "chat",
        git_intent: GitIntent | None = None,
        autonomous: bool = False,
        step_runner: Any | None = None,
        max_turns: int | None = None,
    ) -> MainAgentResult:
        if autonomous:
            return self.run_autonomous(
                user_request,
                entrypoint=entrypoint,
                git_intent=git_intent,
                step_runner=step_runner,
                max_turns=max_turns,
            )
        request = str(user_request or "").strip()
        from mana_agent.prompting.repo_rules import render_repo_rules

        repository_guidance = render_repo_rules(repo_root=self.root)
        try:
            scope = self.scope_engine.decide(
                request=request,
                context=self.workspace_context,
                repository_guidance=repository_guidance,
            )
        except ScopeDecisionError as exc:
            return MainAgentResult("", "blocked", "scope_error", str(exc), [], [], [])
        self.memory.remember_task(f"User request received: {request[:500]}")
        self.memory.remember_repo_fact(f"Repository root: {self.root}")
        title = request[:80] or entrypoint
        main_node = self.registry.find_by_role(AgentRole.MAIN)
        self.memory.remember_agent(
            main_node.agent_id,
            f"Main agent received request: {request[:500]}",
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
            normalized_goal=request,
            owner_agent_id=main_node.agent_id,
            workspace_id=scope.workspace_id,
            session_id=scope.session_id,
            primary_repository_id=scope.primary_repository_id,
            repository_ids=scope.repository_ids,
        )
        self.taskboard.add_evidence(task.task_id, f"Routing command hint: {entrypoint}")
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
        # GitIntent must be an explicit structured decision (caller/model). Keyword
        # inference is forbidden: SWE-bench prompts say "Do not commit, push..." and
        # keyword matchers hijacked those runs into branch/push workflows.
        if git_intent is not None:
            self.memory_service.record_decision(
                agent_id=main_node.agent_id,
                task_id=task.task_id,
                decision_type="route_request",
                input_summary=request,
                memory_used=[str(task.memory_status.get("memory_bundle_id") or "")],
                decision="apply_git_intent_contract",
                reason="an explicit GitIntent contract selects the Git workflow",
            )
            route = self._route_with_git_contract(task.task_id, git_intent)
            task.risk_level = RiskLevel.HIGH
            task.required_capabilities = list(route.required_capabilities)
            self.taskboard.add_evidence(task.task_id, f"GitIntent contract established: {git_intent}")
        else:
            self.memory_service.record_decision(
                agent_id=main_node.agent_id,
                task_id=task.task_id,
                decision_type="route_request",
                input_summary=request,
                memory_used=[str(task.memory_status.get("memory_bundle_id") or "")],
                decision="query_router",
                reason="route after task duplicate and bundle checks",
            )
            try:
                route = self.router.route(
                    task_id=task.task_id,
                    user_request=request,
                    command_hint=entrypoint,
                    repo_root=self.root,
                    repository_guidance=repository_guidance,
                )
            except RoutingDecisionError as exc:
                self.memory_service.record_decision(
                    agent_id=main_node.agent_id,
                    task_id=task.task_id,
                    decision_type="route_request_failed",
                    input_summary=request,
                    memory_used=[str(task.memory_status.get("memory_bundle_id") or "")],
                    decision="blocked",
                    reason=str(exc),
                )
                self.memory_service.update_task(
                    task.task_id,
                    status=TaskStatus.BLOCKED.value,
                    result_summary=str(exc),
                )
                self.taskboard.add_blocker(task.task_id, str(exc))
                self.memory.remember_task(f"Routing blocked for task {task.task_id}: {exc}")
                return MainAgentResult(
                    task.task_id,
                    "blocked",
                    "routing_error",
                    str(exc),
                    [],
                    [],
                    scope.repository_ids,
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
        multi_repository_inspection = len(scope.repository_ids) > 1 and route.route_name in {
            "analyze", "repo_search", "coding",
        }
        if multi_repository_inspection:
            if not worker_ids:
                worker_ids = self._ensure_tool_workers(task.task_id, target_count=1)
            self._run_multi_repository_context(
                task.task_id,
                request,
                scope.repository_ids,
                worker_ids[0] if worker_ids else self.queue_manager.default_worker_agent_id,
            )
        def specialist_runner(step_idx: int, ctx: dict[str, Any]) -> dict[str, Any]:
            head = self._agent(AgentRole.HEAD_DECISION, HeadDecisionAgent)
            head.decide(task.task_id, route, self.decision_room)
            self.memory.remember_agent(
                head.agent_id,
                f"Decided route {route.route_name} for task {task.task_id}: {route.reason_summary}",
            )
            planner = self._agent(AgentRole.PLANNER, PlannerAgent)
            plan = planner.plan(
                task.task_id,
                request,
                route.route_name,
                runtime_capability_change=route.runtime_capability_change,
            )
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
                elif route.route_name == "coding" and (not multi_repository_inspection or managed_workspace is not None):
                    self._delegate_initial_tool_work(task.task_id, request, route.route_name)
            if route.risk_level.value in {"medium", "high"} or len(route.required_agents) > 4:
                self._agent(AgentRole.REVIEWER, ReviewerAgent).review(task.task_id, f"Risk level is {route.risk_level.value}; route requires {len(route.required_agents)} agents.")
            verification_passed: bool | None = None
            if route.requires_verification:
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
                # VERIFYING is an evidence-bearing state. The verifier must first
                # persist its result; planned commands alone are not execution.
                self.taskboard.update_status(
                    task.task_id,
                    TaskStatus.VERIFYING,
                    reason="VerifierAgent recorded executed verification evidence.",
                )
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
            if verification_passed:
                self._record_wiring_reachability(task.task_id, verification)
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
                supervisor_task = self.execution_supervisor.store.get_task_or_none(task.task_id)
                has_verification_evidence = bool(
                    task.verification_results
                    or task.verification_queue_job_ids
                    or (task.supervisor_execution_id and task.supervisor_state and task.supervisor_verification_evidence)
                )
                if supervisor_task is not None and task.status != TaskStatus.VERIFYING and (route.requires_verification or has_verification_evidence):
                    try:
                        self.taskboard.update_status(
                            task.task_id,
                            TaskStatus.VERIFYING,
                            reason=f"{done_reason} Awaiting authoritative supervisor completion projection.",
                        )
                    except Exception as trans_exc:
                        logger.debug("Taskboard status update to VERIFYING skipped: %s", trans_exc)
                step_answer = self._agent(AgentRole.SUMMARIZER, SummarizerAgent).summarize(task.task_id)
            else:
                self.taskboard.update_status(task.task_id, TaskStatus.BLOCKED, reason="Reviewer rejected weak or incomplete hierarchy evidence.")
                step_answer = self._agent(AgentRole.SUMMARIZER, SummarizerAgent).summarize(task.task_id)
                if managed_workspace is not None and self.workspace_manager is not None:
                    try:
                        current = self.workspace_manager.get_for_task(task.task_id)
                        if current.status.value not in {"retained", "failed", "merge_candidate", "merged"}:
                            self.workspace_manager.mark_interrupted(task.task_id, error="review rejected or blocked")
                        self._sync_task_workspace_fields(task.task_id)
                    except WorkspaceError:
                        pass
            self.memory.remember_task(f"Final summary produced for task {task.task_id}: {step_answer[:500]}")
            workshop = self._run_experience_workshop(task.task_id, step_answer, approved=approved)
            if workshop and workshop.proposal_result and workshop.proposal_result.proposal:
                proposal = workshop.proposal_result.proposal
                step_answer += (
                    "\n\nExperience candidate detected\n"
                    f"Reusable workflow: {proposal.display_name}\n"
                    f"Confidence: {proposal.confidence:.2f}\n"
                    f"Proposal: {proposal.proposal_id}\n"
                    "Review with `mana-agent skill proposal review " + proposal.proposal_id + "`."
                )
            return {
                "answer": step_answer,
                "summary": step_answer,
                "verification_passed": verification_passed if verification_passed is not None else True,
                "goal_satisfied": approved,
                "output": step_answer,
            }

        runner = step_runner or specialist_runner
        exec_res = self.autonomous_controller.execute_goal(
            task_id=task.task_id,
            user_goal=request,
            step_runner=runner,
            strategy_hint=route.route_name,
            max_turns=max_turns,
        )
        if self.repository_lanes is not None:
            self.repository_lanes.sync_controller_projection(task.task_id)
        answer = exec_res.final_response or f"Execution completed with status {exec_res.status}."
        return MainAgentResult(
            task.task_id,
            route.route_name,
            route.task_size,
            answer,
            route.required_agents,
            route.required_subagents,
            scope.repository_ids,
        )

    @observed_activity("MainAgent autonomous execution")
    def run_autonomous(
        self,
        user_request: str,
        *,
        entrypoint: str = "chat",
        git_intent: GitIntent | None = None,
        step_runner: Any | None = None,
        max_turns: int | None = None,
    ) -> MainAgentResult:
        if step_runner is None:
            # All domains already use this controller through their actual routed
            # specialist. Repository coordination must not replace other domains.
            return self.run_user_request(user_request, entrypoint=entrypoint,
                git_intent=git_intent, max_turns=max_turns)
        request = str(user_request or "").strip()
        from mana_agent.prompting.repo_rules import render_repo_rules

        repository_guidance = render_repo_rules(repo_root=self.root)
        try:
            scope = self.scope_engine.decide(
                request=request,
                context=self.workspace_context,
                repository_guidance=repository_guidance,
            )
        except ScopeDecisionError as exc:
            return MainAgentResult("", "blocked", "scope_error", str(exc), [], [], [])

        self.memory.remember_task(f"Autonomous request received: {request[:500]}")
        self.memory.remember_repo_fact(f"Repository root: {self.root}")
        title = request[:80] or entrypoint
        main_node = self.registry.find_by_role(AgentRole.MAIN)
        self.memory.remember_agent(
            main_node.agent_id,
            f"Main agent received autonomous request: {request[:500]}",
        )
        task = self.taskboard.create_task(
            title=title,
            user_request=request,
            normalized_goal=request,
            owner_agent_id=main_node.agent_id,
            workspace_id=scope.workspace_id,
            session_id=scope.session_id,
            primary_repository_id=scope.primary_repository_id,
            repository_ids=scope.repository_ids,
        )
        self.taskboard.add_evidence(task.task_id, f"Autonomous routing hint: {entrypoint}")

        runner = step_runner or self._build_autonomous_step_runner(
            task.task_id, request, entrypoint, scope, git_intent
        )
        exec_result = self.autonomous_controller.execute_goal(
            task_id=task.task_id,
            user_goal=request,
            step_runner=runner,
            max_turns=max_turns,
            context={
                "repository_root": str(self.root),
                "repository_guidance": repository_guidance,
            },
        )

        answer = (
            exec_result.final_response
            or f"Autonomous execution ended with status: {exec_result.terminal_reason.value if exec_result.terminal_reason else 'unknown'}."
        )
        return MainAgentResult(
            task.task_id,
            "autonomous",
            "autonomous",
            answer,
            [main_node.agent_id],
            [],
            scope.repository_ids,
        )

    def _build_autonomous_step_runner(self, task_id, request, entrypoint, scope, git_intent=None, *, route=None):
        """Model-directed work on existing tasks, never simulated child completion."""
        route = route or self.router.route(task_id=task_id, user_request=request, command_hint=entrypoint, repo_root=self.root)
        def runner(step_index, ctx):
            from mana_agent.repository_evidence.coordination import coordinate_step
            return coordinate_step(self, task_id, request, route, ctx)
        return runner

    def _execute_evidence_assignment(self, parent_id, child_id, decision_id, route, *, recovery=None, runtime_budget=None):
        from mana_agent.gateway.lane_coordinator import LaneCoordinator
        from mana_agent.gateway.lanes import LaneId
        if self.repository_lanes is None:
            self.repository_lanes = LaneCoordinator(self.root, taskboard=self.taskboard, execution_supervisor=self.execution_supervisor)
        self.repository_lanes.project_active_controller(parent_id,
            lane_id=LaneId.CODING if route.route_name == "coding" else LaneId.RESEARCH)
        with self.repository_lanes.workflow_step_capacity(parent_id, child_id):
            return self._run_evidence_assignment(parent_id, child_id, decision_id, route, recovery=recovery, runtime_budget=runtime_budget)

    def _run_evidence_assignment(self, parent_id, child_id, decision_id, route, *, recovery=None, runtime_budget=None):
        from mana_agent.gateway.lane_coordinator import LaneCoordinator
        from mana_agent.gateway.lanes import LaneId, LaneTaskState
        from mana_agent.coding.facade import UnifiedCodingAgentFacade
        from mana_agent.coding.selection import resolve_coding_backend
        from mana_agent.integrations.codex.config import CodexSettings
        from mana_agent.services.activity_context import ActivityContext, activity_scope, current_activity_context
        from mana_agent.repository_evidence.tasks import assignment_for, investigation_completion_contract
        from mana_agent.repository_evidence.investigation import execute_investigation
        from mana_agent.coding.programmatic import ToolExecutionDispatcher
        if self.repository_lanes is None:
            self.repository_lanes = LaneCoordinator(self.root, taskboard=self.taskboard, execution_supervisor=self.execution_supervisor)
        if self.coding_engine is None:
            settings = Settings()
            selected = resolve_coding_backend(settings)
            self.coding_engine = UnifiedCodingAgentFacade(repo_root=self.root,
                default_mode="internal" if selected.backend == "internal" else "external",
                codex_settings=CodexSettings.from_mana_settings(settings),
                repository_id=self.taskboard.store.repository_id, workspace_id=self.taskboard.store.workspace_id,
                session_id=self.workspace_context.session.session_id)
        child = self.taskboard.get_task(child_id)
        if child.status not in {TaskStatus.NEW, TaskStatus.QUEUED, TaskStatus.ROUTED} and recovery is None:
            raise PermissionError("Only a ready task may be dispatched; recover existing attempts explicitly")
        if any(self.taskboard.get_task(dep).status != TaskStatus.DONE for dep in child.depends_on):
            raise PermissionError("The selected task has unresolved dependencies")
        investigating = child.task_kind in {"investigation", "understanding"}
        if child.task_kind in {"implementation", "integration"} and route.requested_effect != "write":
            raise PermissionError("The original route does not authorize repository mutation")
        if not investigating and self.managed_worktrees_enabled:
            parent = self.taskboard.get_task(parent_id)
            if not parent.execution_repo_root:
                managed = self._allocate_managed_workspace(parent_id, title=parent.title, assigned_agent_id=child.owner_agent_id or "coding_agent")
                parent = self.taskboard.get_task(parent_id)
                def bind_workspace(record):
                    record.workspace_path = parent.execution_repo_root
                self.execution_supervisor.store.update_task(parent_id, bind_workspace)
            child.execution_repo_root = parent.execution_repo_root
            child.managed_worktree_path = parent.managed_worktree_path
            self.taskboard.save()
        elif investigating:
            parent = self.taskboard.get_task(parent_id)
            if parent.execution_repo_root:
                child.execution_repo_root = parent.execution_repo_root
                child.managed_worktree_path = parent.managed_worktree_path
                self.taskboard.save()
        root = Path(child.execution_repo_root or child.managed_worktree_path or self.root)
        lane_id = LaneId.RESEARCH if investigating else LaneId.VERIFY if child.task_kind == "verification" else LaneId.CODING
        if recovery is not None:
            if recovery.task_id != child_id:
                raise PermissionError("Recovery belongs to a different task")
            reservation = self.repository_lanes.retry_task(child_id, decision=recovery, session_id=child.session_id)
        else:
            reservation = self.repository_lanes.reserve(normalized_intent=child.user_request, lane_id=lane_id,
                taskboard_task_id=child_id, parent_task_id=parent_id, session_id=child.session_id,
                workspace_id=child.workspace_id, repository_id=child.primary_repository_id,
                execution_workspace=root, target_files=[str(root / path) for path in child.investigation_scope if path != "."],
                routing_decision_id=decision_id, completion_contract=investigation_completion_contract(child) if investigating else ())
        self.repository_lanes.start(reservation)
        execution = reservation.execution
        parent_activity = current_activity_context()
        activity = ActivityContext(conversation_id=child.session_id,
            execution_id=getattr(parent_activity, "execution_id", "") or child_id,
            task_id=child_id, parent_task_id=parent_id, root_task_id=child.root_task_id,
            repository_id=child.primary_repository_id, agent_id=child.owner_agent_id or "coding_agent",
            attempt_id=execution.supervisor_attempt_id, attempt_generation=self.execution_supervisor.get_task(child_id).attempt_generation,
            lease_token=execution.supervisor_lease_token, supervisor=self.execution_supervisor,
            protected_text=getattr(parent_activity, "protected_text", ()),
            authorize_tool=lambda name: self.repository_lanes.authorize_tool(child_id, name),
            execution_lane=lane_id.value, execution_mode="read_only" if investigating else "act",
            coordinator=self.repository_lanes, execution_registry=self.registry, taskboard=self.taskboard)
        try:
            with activity_scope(activity):
                runtime = {"execution_supervisor": self.execution_supervisor, "taskboard": self.taskboard,
                           "execution_registry": self.registry,
                           "tool_authorize": lambda name: self.repository_lanes.authorize_tool(child_id, name),
                           **(runtime_budget or {})}
                if child.task_kind == "verification":
                    if not child.verification_commands:
                        raise PermissionError("Verification requires model-selected commands; no default check was executed")
                    verification = self._agent(AgentRole.VERIFIER, VerifierAgent).execute_verification(child_id, child.verification_commands)
                    result = {"status": "completed" if verification.passed else "failed", "goal_satisfied": verification.passed,
                              "tests_run": verification.commands_run, "tests_passed": verification.passed,
                              "summary": verification.summary, "verification": {"execution_job_ids": verification.execution_job_ids}}
                elif investigating:
                    dispatcher = ToolExecutionDispatcher.for_coding_task(repo_root=root, task_id=child_id)
                    dispatcher.charge_tool = runtime.get("charge_tool")
                    from mana_agent.repository_evidence.observations import validate_selected_source
                    def validate_source(source):
                        if callable(runtime.get("charge_tool")):
                            runtime["charge_tool"]()
                        return validate_selected_source(dispatcher.runner, source, decision_id=decision_id)
                    result = execute_investigation(taskboard=self.taskboard, task_id=child_id,
                        coding_agent=self.coding_engine, decision_id=decision_id, runtime=runtime,
                        validate_source=validate_source)
                else:
                    assignment = assignment_for(self.taskboard, child_id, backend=self.coding_engine.active_engine_mode, decision_id=decision_id)
                    from mana_agent.repository_evidence.models import EvidenceQuery
                    import json
                    knowledge = self.taskboard.evidence_board(root=root, repository_id=child.primary_repository_id)
                    context = knowledge.query(EvidenceQuery(evidence_ids=tuple(ref.evidence_id for ref in assignment.evidence))) if assignment.evidence else {"records": []}
                    checkpoint = self.execution_supervisor.store.get_checkpoint(child.resume_checkpoint_id) if child.resume_checkpoint_id else None
                    resume = checkpoint.resume_payload.get("continuation", {}) if checkpoint else {}
                    prompt = child.user_request + "\n\nAuthoritative coding assignment and current repository evidence:\n" + json.dumps({
                        "assignment": assignment.model_dump(mode="json"), "evidence": context,
                        "resume_checkpoint": child.resume_checkpoint_id}, ensure_ascii=False)
                    worker = self.coding_engine.for_assignment(root=root)
                    try:
                        result = worker.generate_auto_execute(prompt, gateway_task_id=child_id,
                            execution_mode="read_only" if child.task_kind == "verification" else "auto",
                            relevant_context=assignment.model_dump_json(), execution_runtime=runtime,
                            coding_resume_context=resume,
                            coding_continuation_id=child.resume_checkpoint_id if resume else "")
                        if child.task_kind == "verification" and not (result.get("tests_run") and result.get("tests_passed")):
                            raise PermissionError("Verification task returned no executed passing check")
                    finally:
                        worker.close()
                    if child.task_kind in {"implementation", "integration"} and not result.get("missing_evidence") and not result.get("changed_files"):
                        raise PermissionError("Implementation returned no observed repository changes")
                    result["goal_satisfied"] = result.get("status") in {"success", "completed"}
                if result.get("missing_evidence"):
                    from mana_agent.repository_evidence.models import MissingEvidenceRequest
                    gaps = [MissingEvidenceRequest.model_validate(item) for item in result["missing_evidence"]]
                    if any(gap.task_id != child_id for gap in gaps):
                        raise ValueError("Missing evidence belongs to a different task")
                    self.taskboard.update_task(child_id, missing_evidence=[g.model_dump(mode="json") for g in gaps])
                    self.repository_lanes.checkpoint(child_id, boundary="missing_repository_evidence",
                        resume_payload={"assignment": child.coding_assignment, "missing_evidence": result["missing_evidence"],
                                        "continuation": result.get("continuation_context", {})})
                    result["goal_satisfied"] = False
                if result.get("goal_satisfied") and child.task_kind in {"implementation", "integration"} and route.runtime_capability_change:
                    plan = self._agent(AgentRole.PLANNER, PlannerAgent).plan(child_id, child.user_request,
                        route.route_name, runtime_capability_change=True)
                    self._run_feature_integration_taskboard_lifecycle(child_id, route, plan)
                    updated = self.taskboard.get_task(child_id)
                    if not (updated.integration_verified and updated.runtime_reachability_verified):
                        result["goal_satisfied"] = False
                        result["summary"] = "Production integration verification remains incomplete"
                if not result.get("goal_satisfied"):
                    self.repository_lanes.mark_blocked(child_id, reason="Task completion requirements remain unsatisfied")
                    return {**result, "verification_passed": False}
            finished = self.repository_lanes.finish(child_id, changed_files=result.get("changed_files", []), verification_state=result)
            if finished.state != LaneTaskState.COMPLETED:
                raise PermissionError("Task completion was not verified by the supervisor")
            return {"answer": result.get("summary") or result.get("answer", ""), "goal_satisfied": False,
                    "verification_passed": False, "changed_files": result.get("changed_files", [])}
        except Exception as exc:
            if self.repository_lanes.inspect_task(child_id).state == LaneTaskState.RUNNING:
                self.repository_lanes.mark_blocked(child_id, reason=str(exc))
            raise

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
            child.execution_repo_root = str(Path(repo.canonical_path).resolve())
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
            specialist = CodingAgent(agent_id=coding.agent_id, role=coding.role,
                parent_agent_id=coding.parent_agent_id, capabilities=coding.capabilities,
                mailbox=self.message_bus, message_bus=self.message_bus, taskboard=self.taskboard,
                registry=self.registry, queue_manager=manager)
            specialist.routing_llm = self.routing_llm
            completed = specialist.inspect_task(child.task_id, request)
            if completed.status == QueueJobStatus.DONE:
                self.taskboard.update_status(
                    child.task_id,
                    TaskStatus.VERIFYING,
                    reason="Repository context run completed; supervisor verification is required.",
                )
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
            if node.role == AgentRole.TOOL_WORKER and node.parent_agent_id == coding.agent_id and node.state != AgentState.DONE
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
        for child_id in list(task.required_wiring_task_ids):
            child = self.taskboard.get_task(child_id)
            self.workspace_manager.attach_to_taskboard(child, workspace)
        self.taskboard.save()

    def _delegate_initial_tool_work(self, task_id: str, request: str, route_name: str) -> None:
        self._agent(AgentRole.CODING, CodingAgent).inspect_task(task_id, request)

    def _execute_required_wiring_tasks(self, parent_task_id: str, route: RouteDecision, plan) -> None:  # noqa: ANN001
        """Compatibility adapter for the authoritative coordinator lifecycle."""
        FeatureIntegrationCoordinator().run_taskboard_lifecycle(self, parent_task_id, route, plan)

    def _run_feature_integration_taskboard_lifecycle(self, parent_task_id: str, route: RouteDecision, plan) -> None:  # noqa: ANN001
        """Compatibility shim; FeatureIntegrationCoordinator owns this lifecycle."""
        FeatureIntegrationCoordinator().run_taskboard_lifecycle(self, parent_task_id, route, plan)
        return

    def _wiring_decision(self, child, plan, route, source_refs):
        return self._agent(AgentRole.CODING, CodingAgent).wiring_decision(child, plan, route, source_refs)

    def _project_wiring_completion(self, child_task_id: str, verification) -> None:  # noqa: ANN001
        supervisor = self.execution_supervisor
        child = self.taskboard.get_task(child_task_id)
        supervisor.create_task(
            task_id=child_task_id, assigned_agent=child.owner_agent_id or "coding",
            routing_decision_id=child_task_id,
            workspace_path=Path(child.execution_repo_root or child.managed_worktree_path or self.root).resolve(),
            side_effect_classification=SideEffectClassification.IDEMPOTENT,
            session_id=child.session_id, workspace_id=child.workspace_id,
            repository_id=child.primary_repository_id, normalized_intent=child.user_request,
            requested_operation="wire production runtime", expected_output="verified wiring outcome",
        )
        supervisor.queue(child_task_id)
        leased, token = supervisor.acquire_lease(child_task_id, owner="main", worker=child.owner_agent_id or "coding")
        completed = supervisor.submit_result(
            child_task_id,
            attempt_id=leased.attempt_id,
            lease_token=token,
            payload={"changed_files": child.files_touched, "wiring_outcome": child.wiring_outcome},
        )
        manifest = supervisor.store.artifact_manifest(child_task_id) or {}
        self.taskboard.project_supervisor_completion(
            child_task_id,
            supervisor_task=completed,
            verification_evidence={
                "result_id": completed.result_id,
                "verification": manifest.get("verification"),
                "artefacts": manifest.get("artefacts", []),
            },
        )

    def _record_wiring_reachability(
        self,
        parent_task_id: str,
        verification,
        *,
        child_task_id: str | None = None,
        include_parent: bool = True,
    ) -> None:  # noqa: ANN001
        """Record only connected, model-selected paths with child provenance."""
        parent = self.taskboard.get_task(parent_task_id)
        reviewer = self._agent(AgentRole.REVIEWER, ReviewerAgent)
        child_ids = [child_task_id] if child_task_id else list(parent.required_wiring_task_ids)
        for child_id in child_ids:
            child = self.taskboard.get_task(child_id)
            if child.status not in {TaskStatus.VERIFYING, TaskStatus.DONE} or not child.implementation_verified:
                continue
            edges = [
                edge for edge in child.reachability_edges
                if all(edge.get(key) for key in ("from", "to", "relation", "source_reference"))
            ]
            if len(edges) < 2:
                continue
            path = connected_wiring_path(edges)
            if not path:
                continue
            path.append("observable result: verification passed")
            refs = list(dict.fromkeys([edge["source_reference"] for edge in edges] + list(child.queue_job_ids)))
            verification_source = str(
                child.verification_provenance.get("verification_id")
                or getattr(verification, "verification_id", "")
            )
            if reviewer.verify_runtime_reachability(
                child.task_id,
                path,
                summary="Executed wiring mutation is reachable through discovered repository integration points.",
                source_references=refs,
                observable_result="verification passed",
                verification_source=verification_source,
            ):
                if include_parent:
                    reviewer.verify_runtime_reachability(
                        parent_task_id,
                        path,
                        summary="Wiring child and parent share a provenance-backed production reachability path.",
                        source_references=refs,
                        observable_result="verification passed",
                        verification_source=verification_source,
                    )

    def _validated_execution_files(self, task, paths: list[str]) -> list[str]:  # noqa: ANN001
        """Return changed files that exist under the task's effective repository root."""
        execution_root = Path(task.execution_repo_root or task.managed_worktree_path or self.root).resolve()
        validated: list[str] = []
        for raw_path in paths:
            candidate = Path(str(raw_path)).expanduser()
            resolved = (candidate if candidate.is_absolute() else execution_root / candidate).resolve()
            try:
                relative = resolved.relative_to(execution_root)
            except ValueError:
                continue
            if resolved.is_file():
                validated.append(relative.as_posix())
        return list(dict.fromkeys(validated))

    def _delegate_git_intent_work(self, task_id: str, intent: GitIntent) -> None:
        coding = self._agent(AgentRole.CODING, CodingAgent)
        self.taskboard.assign(task_id, coding.agent_id)
        coding.execute_git_intent(task_id, intent)

    def _run_git_job(self, task_id: str, requested_by_agent_id: str, args: list[str], *, purpose: str) -> QueueJob:
        return self._agent(AgentRole.CODING, CodingAgent).run_git_job(task_id, requested_by_agent_id, args, purpose=purpose)

    def _route_with_git_contract(self, task_id: str, intent: GitIntent) -> RouteDecision:
        capabilities = ["repo_state", "git_status", "git_diff", "verification"]
        if intent.wants_commit:
            capabilities.append("git_commit")
        if intent.wants_push:
            capabilities.append("git_push")
        if intent.wants_branch:
            capabilities.append("git_branch")
        return RouteDecision(
            task_id=task_id,
            route_name="high_risk_tool",
            task_size="medium",
            required_agents=["main", "head_decision", "tool", "verifier", "reviewer", "summarizer"],
            required_subagents=[],
            required_capabilities=capabilities,
            requires_discussion=True,
            requires_verification=True,
            risk_level=RiskLevel.HIGH,
            reason_summary="Git intent requires repository-state inspection, queued Git execution, verification, and review.",
            runtime_capability_change=False,
        )

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
            agent_memory = (
                self.memory
                if node.role is AgentRole.MAIN
                else AgentMemoryBundle(
                    repo_context=RepoContext(root=str(self.root)),
                    task_memory=TaskMemory(),
                    service=None,
                )
            )
            agents[node.role] = cls(
                agent_id=node.agent_id,
                role=node.role,
                parent_agent_id=node.parent_agent_id,
                capabilities=node.capabilities,
                mailbox=self.message_bus,
                taskboard=self.taskboard,
                message_bus=self.message_bus,
                registry=self.registry,
                memory=agent_memory,
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

    def _verification_commands(self, commands: list[str]) -> list[str]:
        valid = []
        for command in commands:
            cmd = str(command).strip()
            if not cmd:
                continue
            if "pytest" in cmd and not ((self.root / "tests").is_dir() or (self.root / "test").is_dir()):
                continue
            if "compileall src" in cmd and not (self.root / "src").is_dir():
                valid.append("python3 -m compileall .")
                continue
            valid.append(cmd)
        if not valid:
            valid = ["python3 -m compileall src"] if (self.root / "src").exists() else ["python3 -m compileall ."]
        return valid


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
