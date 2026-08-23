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
        if child.status is not TaskStatus.BLOCKED:
            taskboard.update_status(child_id, TaskStatus.BLOCKED, reason=reason)
        return child_id

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
        taskboard: Any | None = None,
        taskboard_parent_task_id: str = "",
        trigger_turn_id: str = "",
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
                if taskboard is not None and taskboard_parent_task_id:
                    self.ensure_wiring_child(
                        taskboard,
                        taskboard_parent_task_id,
                        request=request,
                        changed_files=result["changed_files"],
                        trigger_turn_id=trigger_turn_id,
                    )
            integration = result.get("integration")

        if not isinstance(integration, dict) or not self._proven_reachability(integration, authority=current_authority()):
            if taskboard is not None and taskboard_parent_task_id:
                self.block_wiring_child(
                    taskboard,
                    taskboard_parent_task_id,
                    request=request,
                    changed_files=[str(item) for item in result.get("changed_files") or []],
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
            return FeatureIntegrationResult(result=result, status="blocked", error_code=INCOMPLETE_FEATURE_WIRING, resume_required=True)

        result["integration"] = integration
        resolved_authority = current_authority()
        if resolved_authority is None:
            if taskboard is not None and taskboard_parent_task_id:
                self.block_wiring_child(
                    taskboard,
                    taskboard_parent_task_id,
                    request=request,
                    changed_files=[str(item) for item in result.get("changed_files") or []],
                    trigger_turn_id=trigger_turn_id,
                )
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
        parent = main_agent.taskboard.get_task(parent_task_id)
        self.ensure_wiring_child(
            main_agent.taskboard,
            parent_task_id,
            request=str(getattr(route, "reasoning_summary", "") or parent.user_request),
            changed_files=list(parent.files_touched),
            trigger_turn_id=str(getattr(parent, "trigger_turn_id", "") or ""),
        )
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
        # CodingAgent evidence is limited to the model's wiring report. Review,
        # supervision, TaskBoard status, and verification provenance are runtime
        # authority and are never accepted from the model response.
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
