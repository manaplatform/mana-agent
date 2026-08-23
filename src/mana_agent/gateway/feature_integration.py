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
    return [nodes[0], *[f'{edge["relation"]} {node}' for edge, node in zip(edges, nodes[1:])]


@dataclass(frozen=True, slots=True)
class FeatureIntegrationResult:
    result: dict[str, Any]
    status: str
    error_code: str = ""
    resume_required: bool = False

    @property
    def passed(self) -> bool:
        return self.status == "completed" and not self.error_code


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
        if not isinstance(integration, dict) and any(key in result for key in ("wiring_outcome", "reachability_edges")):
            integration = {
                "wiring_outcome": result.get("wiring_outcome"),
                "reachability_edges": result.get("reachability_edges"),
            }
        if not isinstance(integration, dict) or not self._proven_reachability(integration):
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

        if not isinstance(integration, dict) or not self._proven_reachability(integration):
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
        return FeatureIntegrationResult(result=result, status="completed")

    def run_taskboard_lifecycle(self, main_agent: Any, parent_task_id: str, route: Any, plan: Any) -> None:
        """Use the coordinator as the MainAgent adapter for the existing contract."""
        main_agent._execute_required_wiring_tasks(parent_task_id, route, plan)

    @staticmethod
    def _proven_reachability(integration: dict[str, Any]) -> bool:
        if integration.get("wiring_outcome") not in _ACCEPTED_OUTCOMES:
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


__all__ = ["FeatureIntegrationCoordinator", "FeatureIntegrationResult", "INCOMPLETE_FEATURE_WIRING", "WiringDecision", "connected_wiring_path"]
