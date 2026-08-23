"""The single feature-integration gate used by Gateway and MainAgent.

This module deliberately contains orchestration primitives only.  TaskBoard and
ReviewerAgent remain the authority for multi-agent completion; the Gateway
adapter uses the same evidence contract before publishing a turn result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from typing import Literal
from pydantic import BaseModel, Field


INCOMPLETE_FEATURE_WIRING = "INCOMPLETE_FEATURE_WIRING"
_ACCEPTED_OUTCOMES = {"mutation_applied", "already_integrated"}


class WiringDecision(BaseModel):
    outcome: Literal["mutation_required", "mutation_applied", "already_integrated", "incomplete", "failed"]
    patch: str = ""
    wiring_targets: list[str] = Field(default_factory=list)
    runtime_entrypoints: list[str] = Field(default_factory=list)
    configuration_targets: list[str] = Field(default_factory=list)
    edges: list[dict[str, str]] = Field(default_factory=list)
    reason: str = ""


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


@dataclass(frozen=True, slots=True)
class FeatureIntegrationResult:
    result: dict[str, Any]
    status: str
    error_code: str = ""
    resume_required: bool = False

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

    def run(
        self,
        *,
        coding_agent: Any,
        core_result: dict[str, Any],
        request: str,
        gateway_task_id: str,
        flow_id: str | None,
        runtime_capability_change: bool,
        authority: IntegrationAuthority | None = None,
        authority_provider: Callable[[], IntegrationAuthority | None] | None = None,
    ) -> FeatureIntegrationResult:
        result = dict(core_result or {})
        if str(result.get("status") or result.get("run_status") or "").strip().lower() not in {"completed", "success"}:
            return FeatureIntegrationResult(result=result, status="failed")
        changed_files = [str(item) for item in result.get("changed_files") or [] if str(item).strip()]
        if not runtime_capability_change:
            return FeatureIntegrationResult(result=result, status="completed")

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

        integration = result.get("integration")

        def current_authority() -> IntegrationAuthority | None:
            return authority or (authority_provider() if authority_provider else None)

        if not isinstance(integration, dict) and any(key in result for key in ("wiring_outcome", "reachability_edges")):
            integration = {
                "wiring_outcome": result.get("wiring_outcome"),
                "reachability_edges": result.get("reachability_edges"),
            }
        if not isinstance(integration, dict) or not self._proven_reachability(integration, authority=current_authority()):
            continuation = (
                f"{request}\n\n[feature integration continuation]\n"
                "Core implementation already exists. Inspect only the changed capability and its production callers. "
                "Complete missing registration/factory/router/config/export/entrypoint wiring. Do not redo the core feature. "
                "Return exact changed_files, integration outcome, concrete reachability_edges, and verification_evidence. "
                "Use already_integrated only when provenance-backed edges prove production entrypoint → caller/router → registry/factory → implementation."
            )
            continuation_result = coding_agent.generate(
                continuation,
                flow_id=flow_id,
                gateway_task_id=gateway_task_id,
                auto_chat_mode="edit",
            )
            if isinstance(continuation_result, dict):
                continuation_files = [str(item) for item in continuation_result.get("changed_files") or [] if str(item).strip()]
                result.update(continuation_result)
                result["changed_files"] = list(dict.fromkeys([*changed_files, *continuation_files]))
            integration = result.get("integration")

        if not isinstance(integration, dict) or not self._proven_reachability(integration, authority=current_authority()):
            result.update({
                "status": "blocked",
                "error_code": INCOMPLETE_FEATURE_WIRING,
                "goal_satisfied": False,
                "pending_required_work": True,
                "resume_required": True,
                "core_implementation_preserved": True,
            })
            return FeatureIntegrationResult(result=result, status="blocked", error_code=INCOMPLETE_FEATURE_WIRING, resume_required=True)

        result["integration"] = integration
        resolved_authority = current_authority()
        if resolved_authority is None:
            return FeatureIntegrationResult(
                result={**result, "status": "blocked", "error_code": INCOMPLETE_FEATURE_WIRING},
                status="blocked",
                error_code=INCOMPLETE_FEATURE_WIRING,
                resume_required=True,
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

    def run_taskboard_lifecycle(self, main_agent: Any, parent_task_id: str, route: Any, plan: Any) -> None:
        """Run the existing TaskBoard lifecycle through this coordinator.

        MainAgent supplies the repository/runtime adapter, but the coordinator
        is the only public entry point for integration execution.  Keeping the
        adapter call here also makes Gateway and MainAgent share one gate.
        """
        main_agent._run_feature_integration_taskboard_lifecycle(parent_task_id, route, plan)

    @staticmethod
    def _proven_reachability(
        integration: dict[str, Any], *, authority: IntegrationAuthority | None
    ) -> bool:
        if authority is None or not authority.is_complete():
            return False
        if integration.get("wiring_outcome") not in _ACCEPTED_OUTCOMES:
            return False
        # Edge claims are only candidates.  Completion requires the durable
        # outputs of verification, review, and supervision as well.
        verification = integration.get("verification_evidence")
        reviewer_approval = integration.get("reviewer_approval")
        supervisor_completion = integration.get("supervisor_completion")
        if not verification or not reviewer_approval:
            return False
        if not isinstance(supervisor_completion, dict):
            return False
        if supervisor_completion.get("state") != "completed":
            return False
        if supervisor_completion.get("verification_status") != "passed":
            return False
        if integration.get("verification_provenance") != authority.verification_provenance:
            return False
        if integration.get("runtime_reachability") != authority.runtime_reachability:
            return False
        if integration.get("reviewer_approval") != authority.reviewer_approval:
            return False
        if integration.get("supervisor_completion") != authority.supervisor_completion:
            return False
        edges = integration.get("reachability_edges")
        if not isinstance(edges, list) or len(edges) < 3:
            return False
        required = ("from", "to", "relation", "source_reference")
        previous_to = ""
        allowed_relations = {"calls", "selects", "constructs", "instantiates", "routes", "registers", "imports"}
        concrete_edges = []
        for edge in edges:
            if not isinstance(edge, dict) or not all(str(edge.get(key) or "").strip() for key in required):
                return False
            if str(edge["relation"]).strip().lower() not in allowed_relations:
                return False
            if previous_to and str(edge["from"]).strip() != previous_to:
                return False
            previous_to = str(edge["to"]).strip()
            concrete_edges.append(edge)
        return all(str(edge.get("file") or edge.get("source_reference") or "").strip() for edge in concrete_edges)


__all__ = ["FeatureIntegrationCoordinator", "FeatureIntegrationResult", "IntegrationAuthority", "INCOMPLETE_FEATURE_WIRING", "WiringDecision", "connected_wiring_path"]
