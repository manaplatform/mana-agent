"""The single feature-integration gate used by Gateway and MainAgent.

This module deliberately contains orchestration primitives only.  TaskBoard and
ReviewerAgent remain the authority for multi-agent completion; the Gateway
adapter uses the same evidence contract before publishing a turn result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, runtime_checkable
from langchain_core.messages import HumanMessage, SystemMessage
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
from mana_agent.utils.text import extract_model_text


INCOMPLETE_FEATURE_WIRING = "INCOMPLETE_FEATURE_WIRING"
FEATURE_INTEGRATION_DECISION_INVALID = "FEATURE_INTEGRATION_DECISION_INVALID"
FEATURE_INTEGRATION_VERIFIER_UNAVAILABLE = "FEATURE_INTEGRATION_VERIFIER_UNAVAILABLE"
FEATURE_INTEGRATION_VERIFICATION_PLAN_MISSING = "FEATURE_INTEGRATION_VERIFICATION_PLAN_MISSING"
FEATURE_INTEGRATION_VERIFICATION_FAILED = "FEATURE_INTEGRATION_VERIFICATION_FAILED"
FEATURE_INTEGRATION_REACHABILITY_UNPROVEN = "FEATURE_INTEGRATION_REACHABILITY_UNPROVEN"
FEATURE_INTEGRATION_REVIEW_REJECTED = "FEATURE_INTEGRATION_REVIEW_REJECTED"
FEATURE_INTEGRATION_STATE_INVALID = "FEATURE_INTEGRATION_STATE_INVALID"
CORE_EXECUTION_FAILED = "CORE_EXECUTION_FAILED"
INTERNAL_WORK_PENDING = "INTERNAL_WORK_PENDING"
EXTERNAL_DEPENDENCY = "EXTERNAL_DEPENDENCY"
DETERMINISTIC_INTEGRATION_FAILURE = "DETERMINISTIC_INTEGRATION_FAILURE"
HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
_ACCEPTED_OUTCOMES = {"mutation_applied", "already_integrated", "completed"}
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


FEATURE_INTEGRATION_DECISION_PROMPT = """You are Mana-Agent's feature integration decision layer.
Your role is to analyze a completed code implementation and determine the production wiring and reachability decision.

You must return a structured WiringDecision JSON with the following fields:
- outcome: One of "mutation_required", "mutation_applied", "already_integrated", "incomplete", "failed".
    Use "already_integrated" when the implemented feature is already connected to production runtime entrypoints and reachable without additional code changes.
    Use "mutation_applied" when integration changes have already been applied.
    Use "mutation_required" only when a concrete non-empty patch is needed.
    Use "incomplete" or "failed" if integration cannot be completed.
- patch: The diff/patch string if mutation_required, else empty string.
- wiring_targets: list of files or components that require wiring or were wired.
- runtime_entrypoints: list of production entrypoint files or symbols.
- configuration_targets: list of configuration targets.
- edges: list of reachability edges connecting from entrypoints to capabilities. Each edge must be an object with:
    "from": source symbol/component/file
    "to": target symbol/component/file
    "relation": one of "calls", "selects", "constructs", "instantiates", "routes", "registers", "imports"
    "source_reference": concrete source file or file:line reference (e.g. "path/to/file.py:10")
- verification_commands: list of shell commands to verify the integrated feature (e.g. ["python -m pytest tests/..."]).
- reason: rationale explaining the wiring and reachability decision.

Rules:
1. Never invent files, symbols, line numbers, or reachability edges. Every edge must be source-grounded.
2. The reachability path must form a connected chain from a production entrypoint to the target capability.
3. Do NOT certify runtime authority (reviewer approval, supervisor completion, reachability verified, TaskBoard DONE). Those are owned by the runtime execution layer.
4. Return strict JSON matching the schema.
"""


def _coerce_wiring_decision(response: Any) -> WiringDecision:
    if isinstance(response, WiringDecision):
        return response
    if isinstance(response, dict):
        candidate = dict(response)
        if "reachability_edges" in candidate and "edges" not in candidate:
            candidate["edges"] = candidate["reachability_edges"]
        if "wiring_outcome" in candidate and "outcome" not in candidate:
            candidate["outcome"] = candidate["wiring_outcome"]
        return WiringDecision.model_validate(candidate)
    content = getattr(response, "content", response)
    text = extract_model_text(content)
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start : end + 1]
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        if "reachability_edges" in parsed and "edges" not in parsed:
            parsed["edges"] = parsed["reachability_edges"]
        if "wiring_outcome" in parsed and "outcome" not in parsed:
            parsed["outcome"] = parsed["wiring_outcome"]
        return WiringDecision.model_validate(parsed)
    raise ValueError(f"Expected dict schema for WiringDecision, got {type(parsed)}")


class FeatureIntegrationDecisionProvider:
    """Gateway-owned structured Feature Integration decision provider."""

    def __init__(self, llm: Any, workspace_root: Path | str | None = None) -> None:
        self.llm = llm
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else None

    def decide(
        self,
        *,
        request: str = "",
        user_request: str = "",
        changed_files: list[str] | None = None,
        workspace_root: str | Path | None = None,
        task_id: str = "",
        gateway_task_id: str = "",
        parent_task_id: str = "",
        flow_id: str | None = None,
        current_integration_stage: str = "",
        stage: str = "",
        existing_wiring_child_state: dict[str, Any] | None = None,
        existing_source_evidence: list[str] | None = None,
        taskboard: Any = None,
        **kwargs: Any,
    ) -> WiringDecision | None:
        if self.llm is None or not callable(getattr(self.llm, "invoke", None)):
            return None

        effective_request = request or user_request or kwargs.get("user_prompt", "")
        effective_task_id = gateway_task_id or task_id
        effective_files = list(changed_files or [])
        root_path = Path(workspace_root or self.workspace_root or ".").resolve()

        child_state = dict(existing_wiring_child_state or {})
        source_evidence = list(existing_source_evidence or [])
        if taskboard is not None and parent_task_id:
            try:
                parent = taskboard.get_task(parent_task_id)
                for child_task_id in getattr(parent, "required_wiring_task_ids", []):
                    child = taskboard.get_task(child_task_id)
                    if getattr(child, "integration_role", "") == "wiring":
                        child_state = {
                            "task_id": child.task_id,
                            "status": getattr(child.status, "value", str(child.status)),
                            "integration_stage": getattr(child, "integration_stage", ""),
                            "wiring_outcome": getattr(child, "wiring_outcome", ""),
                            "files_touched": list(getattr(child, "files_touched", [])),
                        }
                        break
            except Exception:
                pass

        payload = {
            "user_request": effective_request,
            "changed_files": effective_files,
            "workspace_root": str(root_path),
            "task_id": effective_task_id,
            "parent_task_id": parent_task_id,
            "flow_id": flow_id or "",
            "current_integration_stage": current_integration_stage or stage or child_state.get("integration_stage", ""),
            "existing_wiring_child_state": child_state,
            "existing_source_evidence": source_evidence,
        }

        messages = [
            SystemMessage(content=FEATURE_INTEGRATION_DECISION_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False, sort_keys=True)),
        ]

        try:
            structured = getattr(self.llm, "with_structured_output", None)
            if callable(structured):
                response = structured(WiringDecision, method="json_schema", strict=True).invoke(messages)
            else:
                response = self.llm.invoke(messages)
            decision = _coerce_wiring_decision(response)
            return decision
        except Exception:
            return None

    def __call__(self, **kwargs: Any) -> WiringDecision | None:
        return self.decide(**kwargs)


def decide_feature_integration(
    llm: Any,
    *,
    workspace_root: Path | str | None = None,
    **kwargs: Any,
) -> WiringDecision | None:
    """Helper function to obtain a structured WiringDecision using FeatureIntegrationDecisionProvider."""
    provider = FeatureIntegrationDecisionProvider(llm, workspace_root=workspace_root)
    return provider.decide(**kwargs)


def validate_or_reconcile_integration_stage(child: Any) -> str:
    """Validate and reconcile the integration stage against durable taskboard evidence.

    Recovery stages are cursors, not proofs. This function determines the first
    incomplete integration stage based on persisted durable evidence on the wiring child.
    Never fabricates missing evidence from the stage label alone.
    """
    raw_stage = str(getattr(child, "integration_stage", "") or "").strip().upper()

    # Stage: CORE_COMPLETE / INTEGRATION_DISCOVERY / INTEGRATION_MUTATION
    # Requires: core implementation checkpoint / changed files
    has_core_evidence = bool(getattr(child, "files_touched", None))
    if not has_core_evidence and raw_stage not in ("CORE_COMPLETE", "INTEGRATION_DISCOVERY", "INTEGRATION_MUTATION", "INTEGRATION_VERIFY"):
        return "CORE_COMPLETE"

    # Stage: INTEGRATION_VERIFY
    # Requires: accepted wiring outcome and reachability edge proposals
    wiring_outcome = str(getattr(child, "wiring_outcome", "") or "")
    edges = list(getattr(child, "reachability_edges", []) or [])
    has_valid_decision = wiring_outcome in _ACCEPTED_OUTCOMES and bool(edges) and bool(connected_wiring_path(edges))

    if not has_valid_decision:
        if raw_stage in ("REACHABILITY_VERIFY", "REVIEW", "SUPERVISOR_FINALIZE", "DONE"):
            return "INTEGRATION_VERIFY"
        return raw_stage or "INTEGRATION_VERIFY"

    # Stage: REACHABILITY_VERIFY
    # Requires: passed VerificationResult, execution identity (verification_queue_job_ids or queue_job_ids),
    # and verification_provenance
    verification_results = list(getattr(child, "verification_results", []) or [])
    has_passed_verification = bool(verification_results) and bool(verification_results[-1].passed)
    execution_ids = list(getattr(child, "verification_queue_job_ids", []) or getattr(child, "queue_job_ids", []) or [])
    verification_provenance = dict(getattr(child, "verification_provenance", {}) or {})
    has_provenance = bool(verification_provenance and verification_provenance.get("verification_id"))

    has_verification_evidence = has_passed_verification and bool(execution_ids) and has_provenance
    if not has_verification_evidence:
        if raw_stage in ("REACHABILITY_VERIFY", "REVIEW", "SUPERVISOR_FINALIZE", "DONE"):
            return "INTEGRATION_VERIFY"
        return raw_stage or "INTEGRATION_VERIFY"

    # Stage: REVIEW
    # Requires: all REACHABILITY_VERIFY prerequisites, runtime_reachability_verified,
    # and integration_evidence_records
    has_reachability_verified = bool(getattr(child, "runtime_reachability_verified", False))
    evidence_records = list(getattr(child, "integration_evidence_records", []) or [])
    has_valid_evidence_records = bool(evidence_records) and all(
        bool(r.get("source_references")) and bool(r.get("observable_result")) for r in evidence_records if isinstance(r, dict)
    )
    has_reachability_evidence = has_reachability_verified and has_valid_evidence_records

    if not has_reachability_evidence:
        if raw_stage in ("REVIEW", "SUPERVISOR_FINALIZE", "DONE"):
            return "REACHABILITY_VERIFY"
        return raw_stage if raw_stage in ("REACHABILITY_VERIFY", "INTEGRATION_VERIFY") else "REACHABILITY_VERIFY"

    # Stage: SUPERVISOR_FINALIZE
    # Requires: all REVIEW prerequisites and reviewer approval (reviewed_by_agent_id)
    has_reviewer_approval = bool(str(getattr(child, "reviewed_by_agent_id", "") or "").strip())
    if not has_reviewer_approval:
        if raw_stage in ("SUPERVISOR_FINALIZE", "DONE"):
            return "REVIEW"
        return raw_stage if raw_stage in ("REVIEW", "REACHABILITY_VERIFY", "INTEGRATION_VERIFY") else "REVIEW"

    # Stage: DONE
    # Requires supervisor completion projection
    supervisor_evidence = dict(getattr(child, "supervisor_verification_evidence", {}) or {})
    has_supervisor_complete = (
        getattr(child, "supervisor_state", "") == "completed"
        and getattr(child, "verification_status", "") == "passed"
        and bool(getattr(child, "supervisor_execution_id", ""))
        and bool(supervisor_evidence.get("verification"))
        and bool(supervisor_evidence.get("result_id"))
    )
    if not has_supervisor_complete:
        if raw_stage == "DONE":
            return "SUPERVISOR_FINALIZE"
        return raw_stage if raw_stage in ("SUPERVISOR_FINALIZE", "REVIEW", "REACHABILITY_VERIFY", "INTEGRATION_VERIFY") else "SUPERVISOR_FINALIZE"

    return "DONE" if raw_stage == "DONE" else "SUPERVISOR_FINALIZE"


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
        self.workspace_root.mkdir(parents=True, exist_ok=True)
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


def integration_pending_classification(error_code: str, metadata: dict[str, Any] | None = None) -> str:
    """Classify an integration outcome without turning local failures into waits.

    External waiting is only valid when the result carries an explicit wake-up
    contract.  Feature-integration failures otherwise describe work that was
    attempted and must be surfaced as a deterministic outcome.
    """
    details = metadata or {}
    if error_code == FEATURE_INTEGRATION_VERIFIER_UNAVAILABLE:
        if str(details.get("wake_up_source") or "").strip() and str(
            details.get("wake_up_reference") or ""
        ).strip():
            return EXTERNAL_DEPENDENCY
    return DETERMINISTIC_INTEGRATION_FAILURE


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
            error_code = str(result.get("error_code") or CORE_EXECUTION_FAILED)
            result.update({
                "status": "failed",
                "error_code": error_code,
                "pending_classification": DETERMINISTIC_INTEGRATION_FAILURE,
                "resume_required": False,
                "pending_required_work": False,
                "goal_satisfied": False,
            })
            if taskboard is not None and taskboard_parent_task_id:
                child_id = self.ensure_wiring_child(
                    taskboard,
                    taskboard_parent_task_id,
                    request=request,
                    changed_files=list(result.get("changed_files") or []),
                    trigger_turn_id=trigger_turn_id,
                )
                if child_id:
                    taskboard.update_status(
                        child_id,
                        TaskStatus.FAILED,
                        reason=error_code,
                    )
            return FeatureIntegrationResult(
                result=result,
                status="failed",
                error_code=error_code,
                resume_required=False,
                pending_classification=DETERMINISTIC_INTEGRATION_FAILURE,
            )
        changed_files = [str(item) for item in result.get("changed_files") or [] if str(item).strip()]
        if not runtime_capability_change:
            if taskboard is not None and taskboard_parent_task_id:
                try:
                    parent = taskboard.get_task(taskboard_parent_task_id)
                    parent.wiring_required = False
                    parent.wiring_outcome = "not_required"
                    taskboard.save()
                except Exception:
                    pass
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
                    user_request=request,
                    changed_files=changed_files,
                    gateway_task_id=gateway_task_id,
                    task_id=gateway_task_id,
                    flow_id=flow_id,
                    taskboard=taskboard,
                    parent_task_id=taskboard_parent_task_id,
                    workspace_root=workspace_root,
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
                "pending_required_work": False,
                "resume_required": False,
                "pending_classification": DETERMINISTIC_INTEGRATION_FAILURE,
                "core_implementation_preserved": True,
            })
            return FeatureIntegrationResult(
                result=result,
                status="blocked",
                error_code=FEATURE_INTEGRATION_DECISION_INVALID,
                resume_required=False,
                pending_classification=DETERMINISTIC_INTEGRATION_FAILURE,
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
                "pending_required_work": False,
                "resume_required": False,
                "pending_classification": DETERMINISTIC_INTEGRATION_FAILURE,
                "core_implementation_preserved": True,
            })
            return FeatureIntegrationResult(
                result=result,
                status="blocked",
                error_code=INCOMPLETE_FEATURE_WIRING,
                resume_required=False,
                pending_classification=DETERMINISTIC_INTEGRATION_FAILURE,
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
                "pending_required_work": False,
                "resume_required": False,
                "pending_classification": integration_pending_classification(error_code),
                "core_implementation_preserved": True,
            })
            return FeatureIntegrationResult(
                result=result,
                status="blocked",
                error_code=error_code,
                resume_required=False,
                pending_classification=integration_pending_classification(error_code),
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
        child.implementation_verified = True
        if getattr(child, "runtime_reachability_verified", False) and getattr(child, "integration_evidence_records", None):
            child.integration_verified = True

        current_stage = validate_or_reconcile_integration_stage(child)
        if not child.integration_stage or child.integration_stage in ("CORE_COMPLETE", "INTEGRATION_DISCOVERY", "INTEGRATION_MUTATION"):
            child.integration_stage = "INTEGRATION_VERIFY"
            current_stage = "INTEGRATION_VERIFY"
        else:
            child.integration_stage = current_stage
        taskboard.save()

        root_path = Path(workspace_root or taskboard.store.root).resolve()
        root_path.mkdir(parents=True, exist_ok=True)
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

            # Idempotently persist VerificationResult
            existing_verif_ids = {
                r.verification_id for r in child.verification_results if hasattr(r, "verification_id")
            }
            if verification.verification_id not in existing_verif_ids:
                taskboard.add_verification_result(child_id, verification)

            # Idempotently persist execution/queue job IDs
            exec_job_ids = list(getattr(verification, "execution_job_ids", []) or [])
            if not exec_job_ids:
                if queue_manager is not None:
                    exec_job_ids = [job.job_id for job in queue_manager.jobs_for_task(child_id)]
                elif getattr(active_executor, "queue_manager", None) is not None:
                    exec_job_ids = [job.job_id for job in active_executor.queue_manager.jobs_for_task(child_id)]
            for job_id in exec_job_ids:
                if job_id not in child.verification_queue_job_ids:
                    taskboard.add_verification_queue_job(child_id, job_id)
                if job_id not in child.queue_job_ids:
                    taskboard.add_queue_job(child_id, job_id)

            if not verification.passed:
                reviewer.reject_weak_evidence(child_id, verification.summary)
                taskboard.update_status(
                    child_id,
                    TaskStatus.BLOCKED,
                    reason=f"{FEATURE_INTEGRATION_VERIFICATION_FAILED}: verification failed",
                )
                return None, FEATURE_INTEGRATION_VERIFICATION_FAILED

            if commands and not (child.verification_queue_job_ids or exec_job_ids):
                taskboard.update_status(
                    child_id,
                    TaskStatus.BLOCKED,
                    reason=f"{FEATURE_INTEGRATION_VERIFICATION_FAILED}: no authoritative execution identity exists",
                )
                return None, FEATURE_INTEGRATION_VERIFICATION_FAILED

            child.verification_provenance = {
                "verification_id": verification.verification_id,
                "verified_by_agent_id": getattr(verification, "verified_by_agent_id", "verifier"),
                "queue_job_ids": list(child.verification_queue_job_ids or exec_job_ids or child.queue_job_ids),
                "commands_run": list(verification.commands_run),
                "changed_files": list(child.files_touched),
                "source": verification_plan.source,
                "decision_id": verification_plan.decision_id,
            }

            taskboard.save()
            taskboard.update_status(
                child_id,
                TaskStatus.VERIFYING,
                reason="VerifierAgent recorded executed integration verification evidence.",
            )

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
            try:
                parent = taskboard.get_task(parent_task_id)
                parent.implementation_verified = True
                parent.integration_verified = True
                parent.runtime_reachability_verified = True
                parent.wiring_outcome = "completed"
                parent.wiring_outcome_reason = "Feature integration lifecycle completed successfully."
                taskboard.save()
            except Exception:
                pass
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
        from mana_agent.execution_supervisor import ExecutionState, SideEffectClassification

        supervisor = execution_supervisor
        existing = supervisor.store.get_task_or_none(child.task_id)
        candidate_parent = getattr(child, "parent_task_id", None) or getattr(child, "taskboard_parent_task_id", None)
        existing_parent = supervisor.store.get_task_or_none(candidate_parent) if candidate_parent else None
        parent_task_id = existing_parent.task_id if existing_parent is not None else None
        if existing is None:
            supervisor.create_task(
                task_id=child.task_id,
                parent_task_id=parent_task_id,
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
            completed = supervisor.submit_result(
                child.task_id,
                attempt_id=leased.attempt_id,
                lease_token=token,
                payload={"changed_files": child.files_touched, "wiring_outcome": child.wiring_outcome},
            )
            if getattr(completed, "state", None) in {"completed_pending_verification", ExecutionState.COMPLETED_PENDING_VERIFICATION}:
                try:
                    completed = supervisor.verify_completion(child.task_id)
                except Exception:
                    pass
            if completed.result_id and parent_task_id:
                try:
                    supervisor.acknowledge_result(completed.result_id, parent_task_id=parent_task_id)
                except Exception:
                    pass
        else:
            state = str(getattr(existing.state, "value", existing.state))
            if state == "completed":
                # submit_result owns normal completion verification. Reuse the
                # durable record during re-entry after a projection crash.
                completed = existing
            elif state == "completed_pending_verification":
                # Recovery may find the escrowed result between publication and
                # verification. This is the sole recovery verification call.
                completed = supervisor.verify_completion(child.task_id)
            elif state == "pending_budget_decision":
                raise RuntimeError(
                    f"supervisor task {child.task_id} is awaiting budget finalization"
                )
            elif state in {"created", "queued", "leased", "running", "checkpointing", "waiting"}:
                raise RuntimeError(
                    f"supervisor task {child.task_id} is still active at SUPERVISOR_FINALIZE ({state})"
                )
            else:
                raise RuntimeError(
                    f"supervisor task {child.task_id} ended in terminal failure state ({state})"
                )
        manifest = supervisor.store.artifact_manifest(child.task_id) or {}
        verification_manifest = manifest.get("verification")
        if not isinstance(verification_manifest, dict) or not verification_manifest:
            escrow_results = supervisor.store.results_for_task(child.task_id)
            ver_status = getattr(completed.verification_status, "value", str(completed.verification_status))
            comp_state = getattr(completed.state, "value", str(completed.state))
            if escrow_results and ver_status in {"succeeded", "completed"}:
                verification_manifest = {
                    "verified": True,
                    "status": ver_status,
                    "result_id": completed.result_id,
                }
            elif comp_state == "completed":
                verification_manifest = {
                    "verified": True,
                    "status": "succeeded",
                    "result_id": completed.result_id,
                }
            else:
                raise RuntimeError(
                    f"supervisor task {child.task_id} has no durable completion verification manifest"
                )
        taskboard.project_supervisor_completion(
            child.task_id,
            supervisor_task=completed,
            verification_evidence={
                "result_id": completed.result_id,
                "verification": verification_manifest,
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
    "CORE_EXECUTION_FAILED",
    "INTERNAL_WORK_PENDING",
    "EXTERNAL_DEPENDENCY",
    "DETERMINISTIC_INTEGRATION_FAILURE",
    "integration_pending_classification",
    "HUMAN_REVIEW_REQUIRED",
    "INTEGRATION_STAGES",
    "WiringDecision",
    "FeatureIntegrationVerificationPlan",
    "FeatureIntegrationDecisionProvider",
    "decide_feature_integration",
    "validate_or_reconcile_integration_stage",
    "IntegrationVerificationExecutor",
    "MultiAgentVerificationExecutor",
    "connected_wiring_path",
]
