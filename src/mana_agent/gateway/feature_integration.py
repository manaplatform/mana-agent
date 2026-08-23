"""The single feature-integration gate used by Gateway and MainAgent.

This module deliberately contains orchestration primitives only.  TaskBoard and
ReviewerAgent remain the authority for multi-agent completion; the Gateway
adapter uses the same evidence contract before publishing a turn result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from typing import Literal
from pydantic import BaseModel, Field
from mana_agent.multi_agent.core.types import AgentRole, QueueJobStatus, QueueJobType, TaskStatus
from mana_agent.multi_agent.agents.coding_agent import CodingAgent
from mana_agent.multi_agent.agents.reviewer_agent import ReviewerAgent
from mana_agent.multi_agent.agents.verifier_agent import VerifierAgent


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
        execution_supervisor: Any | None = None,
        workspace_root: str | Path | None = None,
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

        if not isinstance(integration, dict) and any(key in result for key in ("wiring_outcome", "reachability_edges")):
            integration = {
                "wiring_outcome": result.get("wiring_outcome"),
                "reachability_edges": result.get("reachability_edges"),
            }
        if not isinstance(integration, dict) or not self._valid_model_reachability(integration):
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

        if not isinstance(integration, dict) or not self._valid_model_reachability(integration):
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

        result["integration"] = {
            "wiring_outcome": integration.get("wiring_outcome"),
            "reachability_edges": list(integration.get("reachability_edges") or []),
        }
        # Never poll persisted authority as a precondition. The lifecycle
        # below is what creates that authority.
        resolved_authority = authority
        if (
            resolved_authority is None
            and taskboard is not None
            and taskboard_parent_task_id
            and execution_supervisor is not None
        ):
            resolved_authority = self._complete_taskboard_lifecycle(
                taskboard,
                taskboard_parent_task_id,
                integration=result["integration"],
                execution_supervisor=execution_supervisor,
                workspace_root=workspace_root,
                request=request,
                changed_files=[str(item) for item in result.get("changed_files") or []],
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
        if not self._proven_reachability(integration, authority=resolved_authority):
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
        integration: dict[str, Any],
        execution_supervisor: Any,
        workspace_root: str | Path | None,
        request: str,
        changed_files: list[str],
    ) -> IntegrationAuthority | None:
        """Materialize model evidence into the runtime-owned integration gate.

        The model supplies only the proposed outcome and edges. Every field
        below is written by this coordinator after the corresponding runtime
        checks and supervisor transition, so a CodingAgent payload cannot
        manufacture completion authority.
        """
        from mana_agent.multi_agent.core.types import TaskStatus

        child_id = self.ensure_wiring_child(
            taskboard,
            parent_task_id,
            request=request,
            changed_files=changed_files,
        )
        if not child_id:
            return None
        child = taskboard.get_task(child_id)
        edges = list(integration.get("reachability_edges") or [])
        path = connected_wiring_path(edges)
        if integration.get("wiring_outcome") not in _ACCEPTED_OUTCOMES or not path:
            return None
        required = ("from", "to", "relation", "source_reference")
        if not all(isinstance(edge, dict) and all(str(edge.get(key) or "").strip() for key in required) for edge in edges):
            return None

        if child.status is TaskStatus.NEW:
            taskboard.update_status(child_id, TaskStatus.ROUTED, reason="Gateway selected feature integration lifecycle.")
        if child.status is TaskStatus.ROUTED:
            taskboard.update_status(child_id, TaskStatus.IN_PROGRESS, reason="Gateway is executing feature integration lifecycle.")
        child.wiring_outcome = str(integration["wiring_outcome"])
        child.reachability_edges = edges
        child.implementation_verified = True
        taskboard.update_status(child_id, TaskStatus.VERIFYING, reason="Runtime integration evidence is being reviewed.")

        # The Gateway adapter has no MainAgent agent registry. It must not
        # manufacture reviewer or verification evidence from the model payload.
        # Such calls remain incomplete until the coordinator is given the
        # runtime agent bundle through run_taskboard_lifecycle().
        return None

    def run_taskboard_lifecycle(self, main_agent: Any, parent_task_id: str, route: Any, plan: Any) -> None:
        """Own the complete wiring-child lifecycle after core coding."""
        parent = main_agent.taskboard.get_task(parent_task_id)
        child_ids = [
            self.ensure_wiring_child(
                main_agent.taskboard,
                parent_task_id,
                request=str(getattr(route, "reasoning_summary", "") or parent.user_request),
                changed_files=list(parent.files_touched),
                trigger_turn_id=str(getattr(parent, "trigger_turn_id", "") or ""),
            )
        ]
        for child_id in child_ids:
            if not child_id:
                continue
            child = main_agent.taskboard.get_task(child_id)
            if child.status is TaskStatus.DONE:
                continue
            coding = main_agent._agent(AgentRole.CODING, CodingAgent)
            reviewer = main_agent._agent(AgentRole.REVIEWER, ReviewerAgent)
            child.owner_agent_id = coding.agent_id
            main_agent.taskboard.assign(child.task_id, coding.agent_id)
            if child.status is TaskStatus.NEW:
                main_agent.taskboard.update_status(child.task_id, TaskStatus.ROUTED, reason="Coordinator delegated wiring to CodingAgent.")
            if child.status is TaskStatus.ROUTED:
                main_agent.taskboard.update_status(child.task_id, TaskStatus.IN_PROGRESS, reason="Coordinator is executing the wiring lifecycle.")
            # Reuse the model-selected inspection and mutation decisions, but
            # keep their execution and all completion state in this class.
            direct_files = self._validated_execution_files(main_agent, child, list(parent.files_touched))
            read_job = coding.request_batch_read(child.task_id, direct_files) if direct_files else None
            if read_job is not None:
                main_agent.queue_manager.run_next(worker_agent_id=read_job.assigned_worker_agent_id)
            generic = {
                "selected implementation files and their downstream callers.",
                "production construction, registration, routing, and entrypoint wiring.",
                "relevant registry, factory, dependency-injection, or router.",
                "configuration that enables or selects the capability.",
                "a production cli, api, gateway, lifecycle, or supervisor entrypoint.",
            }
            identifiers = list(dict.fromkeys(str(item) for item in (
                list(child.files_touched) + list(parent.implementation_targets)
                + list(getattr(plan, "files_to_inspect", []))
                + list(getattr(plan, "implementation_targets", []))
            ) if str(item).strip() and str(item).strip().lower() not in generic))
            if not identifiers:
                main_agent.taskboard.update_status(child.task_id, TaskStatus.BLOCKED, reason=f"{INCOMPLETE_FEATURE_WIRING}: no model-selected feature identifiers")
                continue
            refs: list[str] = []
            files: list[str] = []
            for identifier in identifiers:
                job = main_agent.queue_manager.enqueue(
                    task_id=child.task_id, requested_by_agent_id=coding.agent_id,
                    approved_by_agent_id=main_agent.registry.find_by_role(AgentRole.MAIN).agent_id,
                    job_type=QueueJobType.REPO_SEARCH,
                    payload={"query": identifier, "regex": False, "limit": 25},
                    purpose="Discover feature-specific callers and integration points.", priority=45,
                )
                ran = main_agent.queue_manager.run_next(worker_agent_id=job.assigned_worker_agent_id)
                if ran is None or ran.status is not QueueJobStatus.DONE:
                    continue
                for match in ((ran.result or {}).get("matches", []) if isinstance(ran.result, dict) else []):
                    if isinstance(match, dict) and match.get("file"):
                        refs.append(f"{match['file']}:{match.get('line', '?')}")
                        files.append(str(match["file"]))
            child.files_to_inspect = list(dict.fromkeys([*direct_files, *files]))
            discovered = [item for item in child.files_to_inspect if item not in direct_files]
            read_job = coding.request_batch_read(child.task_id, discovered) if discovered else None
            if read_job is not None:
                main_agent.queue_manager.run_next(worker_agent_id=read_job.assigned_worker_agent_id)
            decision = main_agent._wiring_decision(child, plan, route, refs)
            child.wiring_outcome = decision.outcome
            child.wiring_outcome_reason = decision.reason
            child.wiring_targets = list(dict.fromkeys(decision.wiring_targets))
            child.runtime_entrypoints = list(dict.fromkeys(decision.runtime_entrypoints))
            child.configuration_targets = list(dict.fromkeys(decision.configuration_targets))
            child.reachability_edges = list(decision.edges)
            main_agent.taskboard.add_files_touched(child.task_id, list(child.files_touched))
            main_agent.taskboard.save()
            if decision.outcome == "mutation_required" and decision.patch.strip():
                patch = coding.request_patch(child.task_id, decision.patch)
                ran_patch = None if patch is None else main_agent.queue_manager.run_next(worker_agent_id=patch.assigned_worker_agent_id)
                if ran_patch is not None and ran_patch.status is QueueJobStatus.DONE:
                    child.wiring_outcome = "mutation_applied"
                    main_agent.taskboard.add_files_touched(
                        child.task_id,
                        list(ran_patch.changed_files),
                    )
                    main_agent.taskboard.add_files_touched(
                        parent_task_id,
                        list(ran_patch.changed_files),
                    )
                    main_agent.taskboard.save()
                else:
                    child.wiring_outcome = "failed"
            if child.wiring_outcome not in _ACCEPTED_OUTCOMES:
                reviewer.reject_weak_evidence(child.task_id, f"{INCOMPLETE_FEATURE_WIRING}: wiring outcome incomplete")
                main_agent.taskboard.update_status(child.task_id, TaskStatus.BLOCKED, reason=f"{INCOMPLETE_FEATURE_WIRING}: wiring outcome incomplete")
                continue
            child.implementation_verified = True
            main_agent.taskboard.update_status(child.task_id, TaskStatus.VERIFYING, reason="Coordinator requires verifier and reviewer evidence.")
            verification = main_agent._agent(AgentRole.VERIFIER, VerifierAgent).execute_verification(
                child.task_id, main_agent._verification_commands(getattr(plan, "verification_commands", []))
            )
            if not verification.passed:
                reviewer.reject_weak_evidence(child.task_id, verification.summary)
                continue
            child.verification_provenance = {
                "verification_id": verification.verification_id,
                "queue_job_ids": list(child.verification_queue_job_ids),
                "changed_files": list(child.files_touched),
            }
            self._record_reachability(main_agent, parent_task_id, child, verification, reviewer)
            if reviewer.review_evidence(child.task_id, route_name="coding", requires_verification=True):
                self._project_completion(main_agent, child, verification)
                self._record_reachability(main_agent, parent_task_id, child, verification, reviewer, include_parent=True)

    @staticmethod
    def _validated_execution_files(main_agent: Any, task: Any, paths: list[str]) -> list[str]:
        root = Path(task.execution_repo_root or task.managed_worktree_path or main_agent.root).resolve()
        valid = []
        for raw in paths:
            candidate = (Path(str(raw)).expanduser() if Path(str(raw)).is_absolute() else root / str(raw)).resolve()
            try:
                relative = candidate.relative_to(root)
            except ValueError:
                continue
            if candidate.is_file():
                valid.append(relative.as_posix())
        return list(dict.fromkeys(valid))

    @staticmethod
    def _record_reachability(main_agent: Any, parent_id: str, child: Any, verification: Any, reviewer: Any, *, include_parent: bool = False) -> None:
        edges = [edge for edge in child.reachability_edges if isinstance(edge, dict) and all(edge.get(key) for key in ("from", "to", "relation", "source_reference"))]
        path = connected_wiring_path(edges)
        if len(edges) < 2 or not path:
            return
        path.append("observable result: verification passed")
        refs = list(dict.fromkeys([edge["source_reference"] for edge in edges] + list(child.queue_job_ids)))
        if reviewer.verify_runtime_reachability(child.task_id, path, summary="Executed wiring mutation is reachable through production integration points.", source_references=refs, observable_result="verification passed", verification_source=child.verification_provenance.get("verification_id", "")) and include_parent:
            reviewer.verify_runtime_reachability(parent_id, path, summary="Wiring child and parent share a provenance-backed production path.", source_references=refs, observable_result="verification passed", verification_source=child.verification_provenance.get("verification_id", ""))

    @staticmethod
    def _project_completion(main_agent: Any, child: Any, verification: Any) -> None:
        from mana_agent.execution_supervisor import SideEffectClassification
        supervisor = main_agent.execution_supervisor
        if supervisor.store.get_task_or_none(child.task_id) is None:
            supervisor.create_task(task_id=child.task_id, assigned_agent=child.owner_agent_id or "coding", routing_decision_id=child.task_id, workspace_path=Path(child.execution_repo_root or child.managed_worktree_path or main_agent.root).resolve(), side_effect_classification=SideEffectClassification.IDEMPOTENT, session_id=child.session_id, workspace_id=child.workspace_id, repository_id=child.primary_repository_id, normalized_intent=child.user_request, requested_operation="wire production runtime", expected_output="verified wiring outcome")
            supervisor.queue(child.task_id)
            leased, token = supervisor.acquire_lease(child.task_id, owner="coordinator", worker=child.owner_agent_id or "coding")
            supervisor.start(child.task_id, attempt_id=leased.attempt_id, lease_token=token)
            supervisor.submit_result(child.task_id, attempt_id=leased.attempt_id, lease_token=token, payload={"changed_files": child.files_touched, "wiring_outcome": child.wiring_outcome})
        completed = supervisor.verify_completion(child.task_id)
        manifest = supervisor.store.artifact_manifest(child.task_id) or {}
        main_agent.taskboard.project_supervisor_completion(child.task_id, supervisor_task=completed, verification_evidence={"result_id": completed.result_id, "verification": manifest.get("verification"), "artefacts": manifest.get("artefacts", [])})

    @staticmethod
    def _valid_model_reachability(integration: dict[str, Any]) -> bool:
        """Validate only model-owned wiring evidence; authority is separate."""
        if integration.get("wiring_outcome") not in _ACCEPTED_OUTCOMES:
            return False
        edges = integration.get("reachability_edges")
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
        # Edge claims are only candidates.  Completion requires the durable
        # outputs of verification, review, and supervision as well.
        # CodingAgent evidence is limited to the model's wiring report. Review,
        # supervision, TaskBoard status, and verification provenance are runtime
        # authority and are never accepted from the model response.
        return all(str(edge.get("file") or edge.get("source_reference") or "").strip() for edge in integration["reachability_edges"])


__all__ = ["FeatureIntegrationCoordinator", "FeatureIntegrationResult", "IntegrationAuthority", "INCOMPLETE_FEATURE_WIRING", "WiringDecision", "connected_wiring_path"]
