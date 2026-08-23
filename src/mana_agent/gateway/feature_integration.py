"""The single feature-integration gate used by Gateway and MainAgent.

This module deliberately contains orchestration primitives only.  TaskBoard and
ReviewerAgent remain the authority for multi-agent completion; the Gateway
adapter uses the same evidence contract before publishing a turn result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, runtime_checkable
from pydantic import BaseModel, Field
from mana_agent.multi_agent.core.types import (
    AgentRole,
    QueueJobStatus,
    QueueJobType,
    TaskStatus,
    VerificationResult,
)
from mana_agent.multi_agent.agents.coding_agent import CodingAgent
from mana_agent.multi_agent.agents.reviewer_agent import ReviewerAgent
from mana_agent.multi_agent.agents.verifier_agent import VerifierAgent


INCOMPLETE_FEATURE_WIRING = "INCOMPLETE_FEATURE_WIRING"
FEATURE_INTEGRATION_DECISION_INVALID = "FEATURE_INTEGRATION_DECISION_INVALID"
FEATURE_INTEGRATION_VERIFIER_UNAVAILABLE = "FEATURE_INTEGRATION_VERIFIER_UNAVAILABLE"
FEATURE_INTEGRATION_VERIFICATION_PLAN_MISSING = "FEATURE_INTEGRATION_VERIFICATION_PLAN_MISSING"
FEATURE_INTEGRATION_VERIFICATION_FAILED = "FEATURE_INTEGRATION_VERIFICATION_FAILED"
FEATURE_INTEGRATION_REACHABILITY_UNPROVEN = "FEATURE_INTEGRATION_REACHABILITY_UNPROVEN"
FEATURE_INTEGRATION_REVIEW_REJECTED = "FEATURE_INTEGRATION_REVIEW_REJECTED"
FEATURE_INTEGRATION_STATE_INVALID = "FEATURE_INTEGRATION_STATE_INVALID"
INTERNAL_WORK_PENDING = "INTERNAL_WORK_PENDING"
EXTERNAL_DEPENDENCY = "EXTERNAL_DEPENDENCY"
DETERMINISTIC_INTEGRATION_FAILURE = "DETERMINISTIC_INTEGRATION_FAILURE"
HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
_ACCEPTED_OUTCOMES = {"mutation_applied", "already_integrated"}
INTEGRATION_STAGES = (
    "CORE_COMPLETE",
    "INTEGRATION_DISCOVERY",
    "INTEGRATION_MUTATION",
    "INTEGRATION_VERIFY",
    "REACHABILITY_VERIFY",
    "REVIEW",
    "SUPERVISOR_FINALIZE",
)


class WiringDecision(BaseModel):
    outcome: Literal["mutation_required", "mutation_applied", "already_integrated", "incomplete", "failed"]
    patch: str = ""
    wiring_targets: list[str] = Field(default_factory=list)
    runtime_entrypoints: list[str] = Field(default_factory=list)
    configuration_targets: list[str] = Field(default_factory=list)
    edges: list[dict[str, str]] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)
    reason: str = ""


class FeatureIntegrationVerificationPlan(BaseModel):
    commands: list[str] = Field(default_factory=list)
    source: str = ""
    decision_id: str = ""
    expected_evidence: list[str] = Field(default_factory=list)


def connected_wiring_path(edges: list[dict[str, str]]) -> list[str]:
    if len(edges) < 2:
        return []
    current = str(edges[0].get("from") or "")
    nodes = [current]
    for edge in edges:
        if str(edge.get("from") or "") != current:
            return []
        current = str(edge.get("to") or "")
        nodes.append(current)
    if not all(nodes):
        return []
    return [nodes[0], *[f'{edge["relation"]} {node}' for edge, node in zip(edges, nodes[1:])]]


@runtime_checkable
class IntegrationVerificationExecutor(Protocol):
    def execute(
        self,
        *,
        task_id: str,
        commands: list[str],
        workspace_root: Path,
    ) -> VerificationResult:
        ...


class MultiAgentVerificationExecutor:
    """Authoritative multi-agent verification executor using QueueManager and VerifierAgent."""

    def __init__(
        self,
        *,
        taskboard: Any,
        queue_manager: Any | None = None,
        workspace_root: Path | str | None = None,
    ) -> None:
        self.taskboard = taskboard
        self.workspace_root = (
            Path(workspace_root).resolve()
            if workspace_root
            else (
                Path(taskboard.store.root).resolve()
                if hasattr(taskboard, "store") and hasattr(taskboard.store, "root")
                else Path(".").resolve()
            )
        )
        if queue_manager is None and taskboard is not None:
            try:
                from mana_agent.multi_agent.queue.queue_manager import QueueManager

                self.queue_manager = QueueManager(
                    self.workspace_root,
                    taskboard=taskboard,
                )
            except Exception:
                self.queue_manager = None
        else:
            self.queue_manager = queue_manager

    def execute(
        self,
        *,
        task_id: str,
        commands: list[str],
        workspace_root: Path,
    ) -> VerificationResult:
        if self.queue_manager is None:
            from mana_agent.multi_agent.core.ids import new_decision_id

            return VerificationResult(
                verification_id=new_decision_id().replace("decision", "verification", 1),
                task_id=task_id,
                verified_by_agent_id="verifier",
                commands_run=[],
                passed=False,
                summary="Verification blocked: QueueManager unavailable.",
                failures=["queue_manager_unavailable"],
                risks=["queue_manager_unavailable"],
            )
        from mana_agent.multi_agent.agents.verifier_agent import VerifierAgent
        from mana_agent.multi_agent.communication.message_bus import MessageBus
        from mana_agent.multi_agent.registry.agent_registry import AgentRegistry

        root_path = Path(workspace_root or self.workspace_root).resolve()
        bus = MessageBus(root_path)
        registry = AgentRegistry()
        verifier_node = registry.find_by_role(AgentRole.VERIFIER)
        verifier = VerifierAgent(
            agent_id=verifier_node.agent_id,
            role=AgentRole.VERIFIER,
            parent_agent_id=verifier_node.parent_agent_id,
            capabilities=verifier_node.capabilities,
            mailbox=bus,
            taskboard=self.taskboard,
            message_bus=bus,
            registry=registry,
            queue_manager=self.queue_manager,
        )
        return verifier.execute_verification(task_id, commands)


@dataclass(frozen=True, slots=True)
class FeatureIntegrationResult:
    result: dict[str, Any]
    status: str
    error_code: str = ""
    resume_required: bool = False
    pending_classification: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "completed" and not self.error_code


@dataclass(frozen=True, slots=True)
class IntegrationAuthority:
    """Durable execution evidence supplied by the runtime, not the model.

    The model may propose wiring edges, but it cannot certify TaskBoard,
    reviewer, or supervisor state.  Those values are assembled by the
    execution owner after the corresponding state transitions have completed.
    """

    taskboard_state: str
    wiring_child_id: str
    verification_provenance: dict[str, Any]
    reviewer_approval: dict[str, Any]
    runtime_reachability: dict[str, Any]
    supervisor_completion: dict[str, Any]

    @classmethod
    def from_taskboard(cls, taskboard: Any, parent_task_id: str) -> "IntegrationAuthority | None":
        """Build authority only from persisted TaskBoard child state."""
        parent = taskboard.get_task(parent_task_id)
        for child_id in parent.required_wiring_task_ids:
            child = taskboard.get_task(child_id)
            supervisor = dict(child.supervisor_verification_evidence or {})
            candidate = cls(
                taskboard_state=str(child.status.value),
                wiring_child_id=str(child.task_id),
                verification_provenance=dict(child.verification_provenance or {}),
                reviewer_approval={
                    "reviewer_id": str(child.reviewed_by_agent_id or ""),
                    "approved": bool(child.reviewed_by_agent_id),
                },
                runtime_reachability={
                    "verified": bool(child.runtime_reachability_verified),
                    "evidence": list(child.integration_evidence or []),
                },
                supervisor_completion={
                    "state": str(child.supervisor_state or ""),
                    "verification_status": str(child.verification_status or ""),
                    "result_id": str(supervisor.get("result_id") or ""),
                },
            )
            if candidate.is_complete():
                return candidate
        return None

    def is_complete(self) -> bool:
        return (
            self.taskboard_state == "done"
            and bool(self.wiring_child_id)
            and bool(self.verification_provenance)
            and bool(self.reviewer_approval.get("approved"))
            and bool(self.runtime_reachability.get("verified"))
            and self.supervisor_completion.get("state") == "completed"
            and self.supervisor_completion.get("verification_status") == "passed"
        )


class FeatureIntegrationCoordinator:
    """Run and verify the bounded integration continuation for a coding turn."""

    def __init__(self, *, checkpoint: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._checkpoint = checkpoint

    @staticmethod
    def ensure_wiring_child(
        taskboard: Any,
        parent_task_id: str,
        *,
        request: str,
        changed_files: list[str],
        trigger_turn_id: str = "",
    ) -> str | None:
        """Create/reuse the authoritative wiring child and seed core output.

        This is intentionally mechanical: the model may describe wiring, but
        it never chooses the TaskBoard authority or its completion state.
        """
        if not parent_task_id:
            return None
        parent = taskboard.get_task(parent_task_id)
        wiring_child = next(
            (
                taskboard.get_task(child_id)
                for child_id in parent.required_wiring_task_ids
                if taskboard.get_task(child_id).integration_role == "wiring"
            ),
            None,
        )
        if wiring_child is None:
            wiring_child = taskboard.create_child_task(
                parent_task_id,
                title="Feature integration wiring",
                user_request=f"Complete production wiring for: {request}",
                owner_agent_id="gateway:feature_integration",
                trigger_turn_id=trigger_turn_id,
                relation_type="feature_integration",
                integration_role="wiring",
            )
        elif str(getattr(wiring_child.status, "value", wiring_child.status)) == "blocked":
            # A resumed integration turn reopens only the reusable wiring
            # child; the core parent and its durable checkpoint remain intact.
            taskboard.reopen(wiring_child.task_id, reason="resuming feature integration")
        files = [str(path) for path in changed_files if str(path).strip()]
        if files:
            taskboard.add_files_touched(parent_task_id, files)
            taskboard.add_files_touched(wiring_child.task_id, files)
        return wiring_child.task_id

    @staticmethod
    def wiring_child_id(taskboard: Any, parent_task_id: str) -> str | None:
        """Return the persisted wiring child that owns integration completion."""
        if not parent_task_id:
            return None
        parent = taskboard.get_task(parent_task_id)
        for child_id in parent.required_wiring_task_ids:
            child = taskboard.get_task(child_id)
            if child.integration_role == "wiring":
                return child.task_id
        return None

    @classmethod
    def block_wiring_child(
        cls,
        taskboard: Any,
        parent_task_id: str,
        *,
        request: str = "",
        changed_files: list[str] | None = None,
        reason: str = INCOMPLETE_FEATURE_WIRING,
        trigger_turn_id: str = "",
    ) -> str | None:
        """Persist a resumable integration wait on the authoritative child."""
        child_id = cls.ensure_wiring_child(
            taskboard,
            parent_task_id,
            request=request,
            changed_files=list(changed_files or []),
            trigger_turn_id=trigger_turn_id,
        )
        if child_id is None:
            return None
        from mana_agent.multi_agent.core.types import TaskStatus

        child = taskboard.get_task(child_id)
        if not child.owner_agent_id:
            child.owner_agent_id = "gateway:feature_integration"
        if child.status is not TaskStatus.BLOCKED:
            taskboard.update_status(child_id, TaskStatus.BLOCKED, reason=reason)
        return child_id

    def run(
        self,
        *,
        coding_agent: Any = None,
        core_result: dict[str, Any],
        request: str,
        gateway_task_id: str,
        flow_id: str | None,
        runtime_capability_change: bool,
        authority: IntegrationAuthority | None = None,
        authority_provider: Callable[[], IntegrationAuthority | None] | None = None,
        taskboard: Any | None = None,
        taskboard_parent_task_id: str = "",
        trigger_turn_id: str = "",
        execution_supervisor: Any | None = None,
        workspace_root: str | Path | None = None,
        queue_manager: Any | None = None,
        verification_commands: list[str] | None = None,
        verification_plan: FeatureIntegrationVerificationPlan | None = None,
        verification_executor: IntegrationVerificationExecutor | None = None,
        integration_decision_provider: Callable[..., WiringDecision | dict[str, Any] | None] | None = None,
        integration_decision: WiringDecision | dict[str, Any] | None = None,
    ) -> FeatureIntegrationResult:
        result = dict(core_result or {})
        if str(result.get("status") or result.get("run_status") or "").strip().lower() not in {"completed", "success"}:
            return FeatureIntegrationResult(result=result, status="failed")
        changed_files = [str(item) for item in result.get("changed_files") or [] if str(item).strip()]
        if not runtime_capability_change:
            return FeatureIntegrationResult(result=result, status="completed")

        if taskboard is not None and taskboard_parent_task_id:
            self.ensure_wiring_child(
                taskboard,
                taskboard_parent_task_id,
                request=request,
                changed_files=changed_files,
                trigger_turn_id=trigger_turn_id,
            )
            taskboard.get_task(taskboard_parent_task_id).integration_stage = "CORE_COMPLETE"

        checkpoint = {
            "boundary": "after_core_implementation",
            "completed_steps": ["routing", "core_implementation"],
            "pending_steps": ["feature_integration", "verification", "final_response"],
            "gateway_task_id": gateway_task_id,
            "core_changed_files": changed_files,
            "flow_id": flow_id or result.get("flow_id", ""),
            "core_result_status": str(result.get("status") or result.get("run_status") or ""),
            "verification_evidence": result.get("verification_evidence", result.get("tests_run", [])),
            "runtime_capability_change": True,
            "integration_contract_identity": "feature-integration-v1",
            "core_result": {
                key: result.get(key)
                for key in ("answer", "status", "run_status", "warnings", "tests_run", "tests_passed", "flow_id")
                if key in result
            },
        }
        if self._checkpoint is not None:
            self._checkpoint(checkpoint)

        # 1. Resolve typed integration decision (P0.1)
        raw_decision = integration_decision
        if raw_decision is None and integration_decision_provider is not None:
            try:
                raw_decision = integration_decision_provider(
                    request=request,
                    changed_files=changed_files,
                    gateway_task_id=gateway_task_id,
                    flow_id=flow_id,
                    taskboard=taskboard,
                    parent_task_id=taskboard_parent_task_id,
                )
            except Exception:
                raw_decision = None

        if raw_decision is None:
            raw_decision = result.get("integration")
            if not isinstance(raw_decision, (WiringDecision, dict)) and any(
                key in result for key in ("wiring_outcome", "reachability_edges")
            ):
                raw_decision = {
                    "wiring_outcome": result.get("wiring_outcome"),
                    "reachability_edges": result.get("reachability_edges"),
                    "verification_commands": result.get("verification_commands", []),
                }

        validated_decision: WiringDecision | None = None
        if isinstance(raw_decision, WiringDecision):
            validated_decision = raw_decision
        elif isinstance(raw_decision, dict):
            try:
                candidate_dict = dict(raw_decision)
                if "reachability_edges" in candidate_dict and "edges" not in candidate_dict:
                    candidate_dict["edges"] = candidate_dict["reachability_edges"]
                if "wiring_outcome" in candidate_dict and "outcome" not in candidate_dict:
                    candidate_dict["outcome"] = candidate_dict["wiring_outcome"]
                validated_decision = WiringDecision.model_validate(candidate_dict)
            except Exception:
                validated_decision = None

        if validated_decision is None:
            if taskboard is not None and taskboard_parent_task_id:
                self.block_wiring_child(
                    taskboard,
                    taskboard_parent_task_id,
                    request=request,
                    changed_files=[str(item) for item in result.get("changed_files") or []],
                    reason=FEATURE_INTEGRATION_DECISION_INVALID,
                    trigger_turn_id=trigger_turn_id,
                )
            result.update({
                "status": "blocked",
                "error_code": FEATURE_INTEGRATION_DECISION_INVALID,
                "goal_satisfied": False,
                "pending_required_work": True,
                "resume_required": True,
                "core_implementation_preserved": True,
            })
            return FeatureIntegrationResult(
                result=result,
                status="blocked",
                error_code=FEATURE_INTEGRATION_DECISION_INVALID,
                resume_required=True,
                pending_classification=INTERNAL_WORK_PENDING,
            )

        if not self._valid_model_reachability(validated_decision):
            if taskboard is not None and taskboard_parent_task_id:
                self.block_wiring_child(
                    taskboard,
                    taskboard_parent_task_id,
                    request=request,
                    changed_files=[str(item) for item in result.get("changed_files") or []],
                    reason=INCOMPLETE_FEATURE_WIRING,
                    trigger_turn_id=trigger_turn_id,
                )
            result.update({
                "status": "blocked",
                "error_code": INCOMPLETE_FEATURE_WIRING,
                "goal_satisfied": False,
                "pending_required_work": True,
                "resume_required": True,
                "core_implementation_preserved": True,
            })
            return FeatureIntegrationResult(
                result=result,
                status="blocked",
                error_code=INCOMPLETE_FEATURE_WIRING,
                resume_required=True,
                pending_classification=INTERNAL_WORK_PENDING,
            )

        # 2. Resolve verification plan (P0.3)
        plan_commands: list[str] = []
        plan_source = "feature_integration_decision"
        plan_decision_id = getattr(validated_decision, "decision_id", "")
        if verification_plan is not None and verification_plan.commands:
            plan_commands = [str(c).strip() for c in verification_plan.commands if str(c).strip()]
            plan_source = verification_plan.source or plan_source
            plan_decision_id = verification_plan.decision_id or plan_decision_id
        elif verification_commands:
            plan_commands = [str(c).strip() for c in verification_commands if str(c).strip()]
            plan_source = "explicit_verification_commands"
        elif validated_decision.verification_commands:
            plan_commands = [str(c).strip() for c in validated_decision.verification_commands if str(c).strip()]
        elif isinstance(result.get("verification_evidence"), dict):
            ve = result["verification_evidence"]
            plan_commands = [str(c).strip() for c in (ve.get("commands_run") or ve.get("commands") or []) if str(c).strip()]
            plan_source = "core_verification_evidence"
        elif isinstance(raw_decision, dict) and isinstance(raw_decision.get("verification_evidence"), dict):
            ve = raw_decision["verification_evidence"]
            plan_commands = [str(c).strip() for c in (ve.get("commands_run") or ve.get("commands") or []) if str(c).strip()]
            plan_source = "raw_verification_evidence"
        elif result.get("commands_run"):
            plan_commands = [str(c).strip() for c in result["commands_run"] if str(c).strip()]
            plan_source = "core_commands_run"
        elif result.get("tests_run"):
            plan_commands = [str(c).strip() for c in result["tests_run"] if str(c).strip()]
            plan_source = "core_tests_run"

        resolved_plan = FeatureIntegrationVerificationPlan(
            commands=plan_commands,
            source=plan_source,
            decision_id=plan_decision_id,
        )

        result["integration"] = {
            "wiring_outcome": validated_decision.outcome,
            "reachability_edges": list(validated_decision.edges),
            "verification_commands": list(plan_commands),
            "wiring_targets": list(validated_decision.wiring_targets),
            "runtime_entrypoints": list(validated_decision.runtime_entrypoints),
            "configuration_targets": list(validated_decision.configuration_targets),
            "reason": validated_decision.reason,
        }

        # Never poll persisted authority as a precondition. The lifecycle
        # below is what creates that authority.
        resolved_authority = authority
        lifecycle_error = ""
        if (
            resolved_authority is None
            and taskboard is not None
            and taskboard_parent_task_id
            and execution_supervisor is not None
        ):
            resolved_authority, lifecycle_error = self._complete_taskboard_lifecycle(
                taskboard,
                taskboard_parent_task_id,
                decision=validated_decision,
                verification_plan=resolved_plan,
                execution_supervisor=execution_supervisor,
                workspace_root=workspace_root,
                request=request,
                changed_files=[str(item) for item in result.get("changed_files") or []],
                owner_agent_id=str(getattr(coding_agent, "agent_id", "") or ""),
                queue_manager=queue_manager,
                verification_executor=verification_executor,
                trigger_turn_id=trigger_turn_id,
            )

        # A TaskBoard-backed run must obtain authority from the lifecycle it
        # just executed. An injected provider is retained for the standalone
        # adapter contract, where no runtime lifecycle is available.
        if (
            resolved_authority is None
            and authority_provider is not None
            and not (taskboard is not None and taskboard_parent_task_id and execution_supervisor is not None)
        ):
            resolved_authority = authority_provider()

        if resolved_authority is None or not self._proven_reachability(result["integration"], authority=resolved_authority):
            error_code = lifecycle_error or INCOMPLETE_FEATURE_WIRING
            if taskboard is not None and taskboard_parent_task_id:
                self.block_wiring_child(
                    taskboard,
                    taskboard_parent_task_id,
                    request=request,
                    changed_files=[str(item) for item in result.get("changed_files") or []],
                    reason=error_code,
                    trigger_turn_id=trigger_turn_id,
                )
            result.update({
                "status": "blocked",
                "error_code": error_code,
                "goal_satisfied": False,
                "pending_required_work": True,
                "resume_required": True,
                "core_implementation_preserved": True,
            })
            return FeatureIntegrationResult(
                result=result,
                status="blocked",
                error_code=error_code,
                resume_required=True,
                pending_classification=INTERNAL_WORK_PENDING,
            )

        result["integration_authority"] = {
            "taskboard_state": resolved_authority.taskboard_state,
            "wiring_child_id": resolved_authority.wiring_child_id,
            "verification_provenance": dict(resolved_authority.verification_provenance),
            "reviewer_approval": dict(resolved_authority.reviewer_approval),
            "runtime_reachability": dict(resolved_authority.runtime_reachability),
            "supervisor_completion": dict(resolved_authority.supervisor_completion),
        }
        return FeatureIntegrationResult(result=result, status="completed")

    def _complete_taskboard_lifecycle(
        self,
        taskboard: Any,
        parent_task_id: str,
        *,
        decision: WiringDecision,
        verification_plan: FeatureIntegrationVerificationPlan,
        execution_supervisor: Any,
        workspace_root: str | Path | None,
        request: str,
        changed_files: list[str],
        owner_agent_id: str,
        queue_manager: Any | None = None,
        verification_executor: IntegrationVerificationExecutor | None = None,
        trigger_turn_id: str = "",
    ) -> tuple[IntegrationAuthority | None, str]:
        """Materialize model evidence into the runtime-owned integration gate.

        The model supplies only the proposed outcome and edges. Every field
        below is written by this coordinator after the corresponding runtime
        checks and supervisor transition, so a CodingAgent payload cannot
        manufacture completion authority.
        """
        from mana_agent.multi_agent.core.types import AgentRole, TaskStatus
        from mana_agent.multi_agent.agents.reviewer_agent import ReviewerAgent
        from mana_agent.multi_agent.communication.message_bus import MessageBus
        from mana_agent.multi_agent.registry.agent_registry import AgentRegistry

        child_id = self.ensure_wiring_child(
            taskboard,
            parent_task_id,
            request=request,
            changed_files=changed_files,
            trigger_turn_id=trigger_turn_id,
        )
        if not child_id:
            return None, FEATURE_INTEGRATION_STATE_INVALID
        child = taskboard.get_task(child_id)

        # Check if already completed (DONE)
        if child.status is TaskStatus.DONE:
            auth = IntegrationAuthority.from_taskboard(taskboard, parent_task_id)
            if auth is not None and auth.is_complete():
                return auth, ""

        if owner_agent_id and not child.owner_agent_id:
            child.owner_agent_id = owner_agent_id

        edges = list(decision.edges)
        path = connected_wiring_path(edges)
        if decision.outcome not in _ACCEPTED_OUTCOMES or not path:
            return None, INCOMPLETE_FEATURE_WIRING

        required = ("from", "to", "relation", "source_reference")
        if not all(isinstance(edge, dict) and all(str(edge.get(key) or "").strip() for key in required) for edge in edges):
            return None, INCOMPLETE_FEATURE_WIRING

        # Stage-aware state normalization (P0.4)
        if child.status is TaskStatus.NEW:
            taskboard.update_status(child_id, TaskStatus.ROUTED, reason="Gateway selected feature integration lifecycle.")
        if child.status is TaskStatus.ROUTED:
            taskboard.update_status(child_id, TaskStatus.IN_PROGRESS, reason="Gateway is executing feature integration lifecycle.")
        elif child.status is TaskStatus.QUEUED:
            taskboard.update_status(child_id, TaskStatus.IN_PROGRESS, reason="Gateway resumed feature integration lifecycle.")
        elif child.status is TaskStatus.BLOCKED:
            taskboard.reopen(child_id, reason="resuming feature integration")
            taskboard.update_status(child_id, TaskStatus.IN_PROGRESS, reason="Gateway resumed feature integration lifecycle.")

        child.wiring_outcome = str(decision.outcome)
        child.reachability_edges = edges
        child.wiring_targets = list(decision.wiring_targets)
        child.runtime_entrypoints = list(decision.runtime_entrypoints)
        child.configuration_targets = list(decision.configuration_targets)
        child.wiring_reason = decision.reason

        current_stage = str(child.integration_stage or "").strip()
        if not current_stage or current_stage in ("CORE_COMPLETE", "INTEGRATION_DISCOVERY", "INTEGRATION_MUTATION"):
            child.implementation_verified = True
            child.integration_stage = "INTEGRATION_VERIFY"
            taskboard.save()
            current_stage = "INTEGRATION_VERIFY"

        root_path = Path(workspace_root or taskboard.store.root).resolve()
        bus = MessageBus(root_path)
        registry = AgentRegistry()
        reviewer_node = registry.find_by_role(AgentRole.REVIEWER)
        reviewer = ReviewerAgent(
            agent_id=reviewer_node.agent_id,
            role=AgentRole.REVIEWER,
            parent_agent_id=reviewer_node.parent_agent_id,
            capabilities=reviewer_node.capabilities,
            mailbox=bus,
            taskboard=taskboard,
            message_bus=bus,
            registry=registry,
        )

        # Stage: INTEGRATION_VERIFY
        if current_stage == "INTEGRATION_VERIFY":
            commands = [str(item).strip() for item in verification_plan.commands if str(item).strip()]
            if not commands:
                taskboard.update_status(
                    child_id,
                    TaskStatus.BLOCKED,
                    reason=f"{FEATURE_INTEGRATION_VERIFICATION_PLAN_MISSING}: no verification commands provided",
                )
                return None, FEATURE_INTEGRATION_VERIFICATION_PLAN_MISSING

            active_executor = verification_executor
            if active_executor is None:
                if queue_manager is not None:
                    active_executor = MultiAgentVerificationExecutor(
                        taskboard=taskboard,
                        queue_manager=queue_manager,
                        workspace_root=root_path,
                    )
                else:
                    try:
                        active_executor = MultiAgentVerificationExecutor(
                            taskboard=taskboard,
                            workspace_root=root_path,
                        )
                    except Exception:
                        active_executor = None

            if active_executor is None or getattr(active_executor, "queue_manager", True) is None:
                taskboard.update_status(
                    child_id,
                    TaskStatus.BLOCKED,
                    reason=f"{FEATURE_INTEGRATION_VERIFIER_UNAVAILABLE}: verification infrastructure unavailable",
                )
                return None, FEATURE_INTEGRATION_VERIFIER_UNAVAILABLE

            verification = active_executor.execute(
                task_id=child_id,
                commands=commands,
                workspace_root=root_path,
            )
            if queue_manager is not None:
                child.queue_job_ids = list(dict.fromkeys(
                    [*child.queue_job_ids, *(job.job_id for job in queue_manager.jobs_for_task(child_id))]
                ))
            elif getattr(active_executor, "queue_manager", None) is not None:
                qm = active_executor.queue_manager
                child.queue_job_ids = list(dict.fromkeys(
                    [*child.queue_job_ids, *(job.job_id for job in qm.jobs_for_task(child_id))]
                ))
            taskboard.save()
            taskboard.update_status(
                child_id,
                TaskStatus.VERIFYING,
                reason="VerifierAgent recorded executed integration verification evidence.",
            )
            if not verification.passed:
                reviewer.reject_weak_evidence(child_id, verification.summary)
                taskboard.update_status(
                    child_id,
                    TaskStatus.BLOCKED,
                    reason=f"{FEATURE_INTEGRATION_VERIFICATION_FAILED}: verification failed",
                )
                return None, FEATURE_INTEGRATION_VERIFICATION_FAILED

            child.verification_provenance = {
                "verification_id": verification.verification_id,
                "verified_by_agent_id": getattr(verification, "verified_by_agent_id", "verifier"),
                "queue_job_ids": list(child.verification_queue_job_ids or child.queue_job_ids),
                "commands_run": list(verification.commands_run),
                "changed_files": list(child.files_touched),
                "source": verification_plan.source,
                "decision_id": verification_plan.decision_id,
            }
            child.integration_stage = "REACHABILITY_VERIFY"
            taskboard.save()
            current_stage = "REACHABILITY_VERIFY"

        # Stage: REACHABILITY_VERIFY
        if current_stage == "REACHABILITY_VERIFY":
            path = connected_wiring_path(edges)
            path.append("observable result: verification passed")
            refs = list(dict.fromkeys(
                [edge["source_reference"] for edge in edges]
                + list(child.queue_job_ids)
                + list(child.verification_queue_job_ids)
            ))
            verification_id = str((child.verification_provenance or {}).get("verification_id") or "")
            if not reviewer.verify_runtime_reachability(
                child_id,
                path,
                summary="Executed integration is reachable through production integration points.",
                source_references=refs,
                observable_result="verification passed",
                verification_source=verification_id,
            ):
                taskboard.update_status(
                    child_id,
                    TaskStatus.BLOCKED,
                    reason=f"{FEATURE_INTEGRATION_REACHABILITY_UNPROVEN}: reachability verification failed",
                )
                return None, FEATURE_INTEGRATION_REACHABILITY_UNPROVEN
            child.integration_stage = "REVIEW"
            taskboard.save()
            current_stage = "REVIEW"

        # Stage: REVIEW
        if current_stage == "REVIEW":
            if not reviewer.review_evidence(child_id, route_name="coding", requires_verification=True):
                taskboard.update_status(
                    child_id,
                    TaskStatus.BLOCKED,
                    reason=f"{FEATURE_INTEGRATION_REVIEW_REJECTED}: Reviewer rejected integration evidence",
                )
                return None, FEATURE_INTEGRATION_REVIEW_REJECTED
            child.integration_stage = "SUPERVISOR_FINALIZE"
            taskboard.save()
            current_stage = "SUPERVISOR_FINALIZE"

        # Stage: SUPERVISOR_FINALIZE
        if current_stage == "SUPERVISOR_FINALIZE":
            self._project_completion(
                taskboard,
                child,
                verification_evidence=child.verification_provenance,
                execution_supervisor=execution_supervisor,
                workspace_root=root_path,
                trigger_turn_id=trigger_turn_id,
            )
            return IntegrationAuthority.from_taskboard(taskboard, parent_task_id), ""

        return None, FEATURE_INTEGRATION_STATE_INVALID

    def run_taskboard_lifecycle(self, main_agent: Any, parent_task_id: str, route: Any, plan: Any) -> None:
        """Compatibility adapter into the coordinator-owned lifecycle.

        The MainAgent adapter supplies dependencies; it does not implement a
        second integration algorithm.
        """
        parent = main_agent.taskboard.get_task(parent_task_id)
        child_id = self.ensure_wiring_child(
            main_agent.taskboard,
            parent_task_id,
            request=str(getattr(route, "reasoning_summary", "") or parent.user_request),
            changed_files=list(parent.files_touched),
            trigger_turn_id=str(getattr(parent, "trigger_turn_id", "") or ""),
        )
        if not child_id:
            return
        child = main_agent.taskboard.get_task(child_id)
        coding = main_agent._agent(AgentRole.CODING, CodingAgent)
        child.owner_agent_id = coding.agent_id
        main_agent.taskboard.assign(child_id, coding.agent_id)
        if child.status is TaskStatus.NEW:
            main_agent.taskboard.update_status(child_id, TaskStatus.ROUTED, reason="Coordinator delegated wiring to CodingAgent.")
        if child.status is TaskStatus.ROUTED:
            main_agent.taskboard.update_status(child_id, TaskStatus.IN_PROGRESS, reason="Coordinator is executing the wiring lifecycle.")
        elif child.status is TaskStatus.QUEUED:
            main_agent.taskboard.update_status(child_id, TaskStatus.IN_PROGRESS, reason="Coordinator resumed feature integration lifecycle.")
        elif child.status is TaskStatus.BLOCKED:
            main_agent.taskboard.reopen(child_id, reason="resuming feature integration")
            main_agent.taskboard.update_status(child_id, TaskStatus.IN_PROGRESS, reason="Coordinator resumed feature integration lifecycle.")

        decision = WiringDecision(
            outcome="already_integrated" if child.wiring_outcome in _ACCEPTED_OUTCOMES else "incomplete",
            edges=list(child.reachability_edges),
            wiring_targets=list(child.wiring_targets),
            runtime_entrypoints=list(child.runtime_entrypoints),
            configuration_targets=list(child.configuration_targets),
            reason=child.wiring_reason or "",
        )

        if child.wiring_outcome not in _ACCEPTED_OUTCOMES:
            decision = main_agent._wiring_decision(
                child,
                plan,
                route,
                list(child.files_touched) + list(parent.files_touched),
            )
            child.wiring_outcome = decision.outcome
            child.wiring_reason = decision.reason
            child.wiring_targets = list(dict.fromkeys(decision.wiring_targets))
            child.runtime_entrypoints = list(dict.fromkeys(decision.runtime_entrypoints))
            child.configuration_targets = list(dict.fromkeys(decision.configuration_targets))
            child.reachability_edges = list(decision.edges)
            if decision.outcome == "mutation_required" and decision.patch.strip():
                child.integration_stage = "INTEGRATION_MUTATION"
                patch = coding.request_patch(child_id, decision.patch)
                ran = None if patch is None else main_agent.queue_manager.run_next(worker_agent_id=patch.assigned_worker_agent_id)
                if ran is not None and ran.status is QueueJobStatus.DONE:
                    child.wiring_outcome = "mutation_applied"
                    decision = decision.model_copy(update={"outcome": "mutation_applied"})
                    main_agent.taskboard.add_files_touched(child_id, list(ran.changed_files))
                    main_agent.taskboard.add_files_touched(parent_task_id, list(ran.changed_files))
            main_agent.taskboard.save()

        verification_executor = MultiAgentVerificationExecutor(
            taskboard=main_agent.taskboard,
            queue_manager=main_agent.queue_manager,
            workspace_root=main_agent.root,
        )
        plan_commands = list(getattr(plan, "verification_commands", []) or decision.verification_commands)
        self._complete_taskboard_lifecycle(
            main_agent.taskboard,
            parent_task_id,
            decision=decision,
            verification_plan=FeatureIntegrationVerificationPlan(
                commands=plan_commands,
                source="main_agent_plan",
                decision_id=getattr(decision, "decision_id", ""),
            ),
            execution_supervisor=main_agent.execution_supervisor,
            workspace_root=main_agent.root,
            request=str(getattr(route, "reasoning_summary", "") or parent.user_request),
            changed_files=list(parent.files_touched),
            owner_agent_id=coding.agent_id,
            queue_manager=main_agent.queue_manager,
            verification_executor=verification_executor,
            trigger_turn_id=str(getattr(parent, "trigger_turn_id", "") or ""),
        )

    @staticmethod
    def _project_completion(
        taskboard: Any,
        child: Any,
        verification: Any = None,
        *,
        verification_evidence: dict[str, Any] | None = None,
        execution_supervisor: Any,
        workspace_root: str | Path | None,
        trigger_turn_id: str = "",
    ) -> None:
        from mana_agent.execution_supervisor import SideEffectClassification

        supervisor = execution_supervisor
        if supervisor.store.get_task_or_none(child.task_id) is None:
            supervisor.create_task(
                task_id=child.task_id,
                assigned_agent=child.owner_agent_id or "coding",
                routing_decision_id=child.task_id,
                workspace_path=Path(child.execution_repo_root or child.managed_worktree_path or workspace_root or taskboard.store.root).resolve(),
                side_effect_classification=SideEffectClassification.IDEMPOTENT,
                session_id=child.session_id,
                workspace_id=child.workspace_id,
                repository_id=child.primary_repository_id,
                normalized_intent=child.user_request,
                requested_operation="wire production runtime",
                expected_output="verified wiring outcome",
                trigger_turn_id=trigger_turn_id,
            )
            supervisor.queue(child.task_id)
            leased, token = supervisor.acquire_lease(child.task_id, owner="coordinator", worker=child.owner_agent_id or "coding")
            supervisor.start(child.task_id, attempt_id=leased.attempt_id, lease_token=token)
            supervisor.submit_result(
                child.task_id,
                attempt_id=leased.attempt_id,
                lease_token=token,
                payload={"changed_files": child.files_touched, "wiring_outcome": child.wiring_outcome},
            )
        completed = supervisor.verify_completion(child.task_id)
        manifest = supervisor.store.artifact_manifest(child.task_id) or {}
        taskboard.project_supervisor_completion(
            child.task_id,
            supervisor_task=completed,
            verification_evidence={
                "result_id": completed.result_id,
                "verification": manifest.get("verification") or verification_evidence or {},
                "artefacts": manifest.get("artefacts", []),
            },
        )

    @staticmethod
    def _valid_model_reachability(decision_or_dict: WiringDecision | dict[str, Any]) -> bool:
        """Validate only model-owned wiring evidence; authority is separate."""
        if isinstance(decision_or_dict, dict):
            outcome = decision_or_dict.get("wiring_outcome") or decision_or_dict.get("outcome")
            edges = decision_or_dict.get("reachability_edges") or decision_or_dict.get("edges")
        else:
            outcome = decision_or_dict.outcome
            edges = decision_or_dict.edges
        if outcome not in _ACCEPTED_OUTCOMES:
            return False
        if not isinstance(edges, list) or len(edges) < 3:
            return False
        required = ("from", "to", "relation", "source_reference")
        previous_to = ""
        allowed_relations = {"calls", "selects", "constructs", "instantiates", "routes", "registers", "imports"}
        for edge in edges:
            if not isinstance(edge, dict) or not all(str(edge.get(key) or "").strip() for key in required):
                return False
            if str(edge["relation"]).strip().lower() not in allowed_relations:
                return False
            if previous_to and str(edge["from"]).strip() != previous_to:
                return False
            previous_to = str(edge["to"]).strip()
        return True

    @staticmethod
    def _proven_reachability(
        integration: dict[str, Any], *, authority: IntegrationAuthority | None
    ) -> bool:
        if not FeatureIntegrationCoordinator._valid_model_reachability(integration):
            return False
        if authority is None or not authority.is_complete():
            return False
        edges = integration.get("reachability_edges") or integration.get("edges") or []
        return all(str(edge.get("file") or edge.get("source_reference") or "").strip() for edge in edges)


__all__ = [
    "FeatureIntegrationCoordinator",
    "FeatureIntegrationResult",
    "IntegrationAuthority",
    "INCOMPLETE_FEATURE_WIRING",
    "FEATURE_INTEGRATION_DECISION_INVALID",
    "FEATURE_INTEGRATION_VERIFIER_UNAVAILABLE",
    "FEATURE_INTEGRATION_VERIFICATION_PLAN_MISSING",
    "FEATURE_INTEGRATION_VERIFICATION_FAILED",
    "FEATURE_INTEGRATION_REACHABILITY_UNPROVEN",
    "FEATURE_INTEGRATION_REVIEW_REJECTED",
    "FEATURE_INTEGRATION_STATE_INVALID",
    "INTERNAL_WORK_PENDING",
    "EXTERNAL_DEPENDENCY",
    "DETERMINISTIC_INTEGRATION_FAILURE",
    "HUMAN_REVIEW_REQUIRED",
    "INTEGRATION_STAGES",
    "WiringDecision",
    "FeatureIntegrationVerificationPlan",
    "IntegrationVerificationExecutor",
    "MultiAgentVerificationExecutor",
    "connected_wiring_path",
]
