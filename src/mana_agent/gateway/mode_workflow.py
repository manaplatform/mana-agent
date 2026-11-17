"""Gateway operations for supervised AUTO, DEBUG, and PLAN tasks."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import getpass
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
from typing import Any

from mana_agent.agent.mode import CodingWorkflowDecision, ModePhase as Phase, PlanRevision, PlanStep, VerificationCommand, repository_relative_path
from mana_agent.agent.plan_service import InspectDecision, InvestigateDecision, MAX_INSPECTION_BATCH, PlanService
from mana_agent.agent.repository_investigation import (
    compact_observation, investigation_scope, observation_fingerprint, read_tool_schemas, run_repository_investigation,
)
from mana_agent.batch.policy import BatchPending, TaskExecutionPolicy
from mana_agent.chat.activity import safe_text
from mana_agent.chat_commands.models import CommandResult
from mana_agent.execution_supervisor.errors import StaleLeaseError
from mana_agent.execution_supervisor.models import ExecutionState
from mana_agent.gateway.lane_coordinator import LaneReservation
from mana_agent.gateway.lanes import LaneId, LaneTaskState
from mana_agent.human_inbox.identity import local_reviewer_actor
from mana_agent.human_inbox.models import (
    ClarificationField, ExpectedResponseType, InboxRequest, InboxRequestType, InboxStatus,
    ReviewerAssignment, ReviewerType, ResponseOperation, ResponseSubmission, canonical_digest,
)
from mana_agent.human_inbox.repository import InboxConcurrentUpdateError
from mana_agent.multi_agent.core.types import TaskStatus
from mana_agent.services.activity_context import (
    ActivityContext, activity_protected_text, activity_scope, current_activity_context, emit_activity,
)
from mana_agent.tools.apply_patch import extract_patch_operations, safe_apply_patch
from mana_agent.utils.redaction import redact_secrets

_OWNER = "autonomous_loop"
logger = logging.getLogger(__name__)


class ScopeRevisionNeeded(RuntimeError):
    pass


class ModeBatchWaiting(RuntimeError):
    pass


class VerificationUnavailable(RuntimeError):
    """A verification prerequisite prevented the approved command from running."""


class VerificationFailed(RuntimeError):
    def __init__(self, command: VerificationCommand, outcome: dict):
        super().__init__(f"Verification failed for {command.argv!r}: {outcome['output']}")
        self.command = command
        self.outcome = outcome


_TERMINAL = {Phase.COMPLETED, Phase.CANCELLED}
_TRANSITIONS = {
    Phase.UNDERSTANDING: {Phase.INVESTIGATING, Phase.CLARIFICATION_REQUIRED, Phase.DRAFTING_PLAN, Phase.COMPLETED},
    Phase.INVESTIGATING: {Phase.UNDERSTANDING, Phase.INVESTIGATING, Phase.CLARIFICATION_REQUIRED,
                          Phase.DRAFTING_PLAN, Phase.COMPLETED},
    Phase.CLARIFICATION_REQUIRED: {Phase.UNDERSTANDING},
    Phase.DRAFTING_PLAN: {Phase.AWAITING_APPROVAL, Phase.APPROVED, Phase.COMPLETED},
    Phase.AWAITING_APPROVAL: {Phase.APPROVED, Phase.UNDERSTANDING},
    Phase.APPROVED: {Phase.EXECUTING, Phase.UNDERSTANDING},
    Phase.EXECUTING: {Phase.EXECUTING, Phase.INTEGRATING, Phase.VERIFYING, Phase.REVIEWING, Phase.UNDERSTANDING, Phase.COMPLETED},
    Phase.INTEGRATING: {Phase.EXECUTING, Phase.REVIEWING},
    Phase.VERIFYING: {Phase.EXECUTING, Phase.INTEGRATING, Phase.REVIEWING, Phase.COMPLETED, Phase.UNDERSTANDING},
    Phase.REVIEWING: {Phase.EXECUTING, Phase.COMPLETED},
    Phase.BLOCKED: {Phase.UNDERSTANDING, Phase.EXECUTING},
}


def _investigation_context(evidence: list[dict]) -> list[dict]:
    return [compact_observation(item) for item in evidence]


def _code_span(value: str) -> str:
    runs = re.findall(r"`+", value)
    if not runs:
        return f"`{value}`"
    fence = "`" * (max(map(len, runs)) + 1)
    return f"{fence} {value} {fence}"


def _changed_files_summary(files: list[str]) -> str:
    if not files:
        return ""
    preview = ", ".join(_code_span(path) for path in files[:4])
    if len(files) > 4:
        preview += f" and {len(files) - 4} more"
    return f"Changed {preview}."


class ModeWorkflow:
    def __init__(self, gateway):
        self.gateway = gateway

    @property
    def lane(self):
        return self.gateway._lane_coordinator

    @property
    def board(self):
        return self.lane.taskboard

    @property
    def inbox(self):
        return self.gateway.human_inbox_service

    def _workspace_root(self, state: dict) -> Path:
        if not state.get("isolated_workspace"):
            if state.get("worktree_required"):
                raise PermissionError("The required task worktree must be reconciled before resuming")
            return self.gateway.root
        from mana_agent.multi_agent.worktrees import WorkspaceManager

        binding = state["isolated_workspace"]
        manager = WorkspaceManager(self.gateway.root, repository_id=binding["repository_id"])
        workspace = manager.get_for_task(binding["task_id"])
        root = Path(workspace.worktree_path)
        if (workspace.workspace_id != binding["workspace_id"]
                or workspace.worktree_path != binding["path"]
                or Path(workspace.source_repo_root).resolve() != self.gateway.root.resolve()
                or not root.is_dir() or root.resolve() != root):
            raise PermissionError("The task's assigned worktree is unavailable or has changed")
        return root

    def _service(self, *, state=None, llm=None):
        from mana_agent.gateway.turn_engine import agent_decision_llm
        from mana_agent.llm.engine import NativeLLMEngine

        state = state or {}
        if llm is None:
            llm = getattr(self.gateway._entry_router, "llm", None) or agent_decision_llm(self.gateway.get_ask_service())
        if isinstance(llm, NativeLLMEngine):
            llm = copy(llm)
            governor = getattr(getattr(self.gateway, "_stack", None), "context_cost_governor", None)
            if governor is not None:
                llm.context_cost_governor = governor
            saved = state.get("investigation") or {}
            model = state.get("model") or saved.get("model")
            provider = state.get("provider") or saved.get("provider")
            if model and provider and (llm.model, llm.provider) != (model, provider):
                llm.update_model_assignment(provider, model, settings=self.gateway.settings)
        return PlanService(self._workspace_root(state), llm=llm)

    def active(self, session_id: str, *, mode: str = "plan"):
        tasks = self.board.store.repository.list_tasks(session_id=session_id)
        for task in sorted(tasks, key=lambda row: row.created_at, reverse=True):
            state = task.mode_workflow
            if state.get("mode") != mode or state.get("phase") in _TERMINAL:
                continue
            supervisor = self.lane.execution_supervisor.store.get_task_or_none(task.task_id)
            if supervisor and supervisor.state not in {ExecutionState.COMPLETED, ExecutionState.CANCELLED}:
                return task
        return None

    def _persist(self, task_id: str, state: dict) -> None:
        self.board.update_task(task_id, mode_workflow=state)

    def _transition(self, task_id: str, state: dict, target: Phase, summary: str = "") -> None:
        current = Phase(state["phase"])
        if target != current and target not in {*_TRANSITIONS.get(current, ()), Phase.BLOCKED, Phase.CANCELLED}:
            raise RuntimeError(f"Invalid mode transition {current.value} → {target.value}")
        if current in _TERMINAL and target != current:
            raise RuntimeError("A terminal mode task cannot change phase")
        state["phase"] = target.value
        self._persist(task_id, state)
        event_type = "agent.planning" if state["mode"] == "plan" and target not in {
            Phase.EXECUTING, Phase.VERIFYING, Phase.COMPLETED
        } else "coding.progress"
        stage = target.value.replace("_", " ").capitalize()
        description = summary or stage
        task = self.board.get_task(task_id)
        emit_activity(event_type, title=f"{state['mode'].upper()} · {stage}", summary=description, task_id=task_id,
                      conversation_id=task.session_id,
                      status="failed" if target == Phase.BLOCKED else "success",
                      operation_id=f"{task_id}:{target.value}",
                      metadata={"phase": target.value, "mode": state["mode"],
                                "mode_reason": state.get("mode_reason", ""),
                                "revision": state["revision"], "step_id": state.get("step_id", "")})

    def start(
        self, session_id: str, goal: str, *, mode: str = "plan", repair: bool = False,
        task_id: str = "", no_code: bool = False, save: bool = True,
        output: str | None = None, question_limit: int | None = None, llm=None,
        skill_names: tuple[str, ...] = (), read_only: bool = False,
        execution_policy: dict | None = None,
        reviewer_id: str = "",
        workflow: CodingWorkflowDecision | None = None,
        resolved_request: str = "",
        conversation_messages: tuple[dict[str, Any], ...] = (),
        runtime_capability_change: bool = False,
        source_message_ids: tuple[str, ...] = (),
        related_task_id: str = "",
        model: str = "",
        provider: str = "",
    ) -> CommandResult:
        activity = current_activity_context()
        protected = (activity.protected_text if activity and activity.conversation_id == session_id
                     else activity_protected_text(getattr(self.gateway, "settings", None)))
        goal = safe_text(goal, limit=None, protected=protected).strip()
        policy = TaskExecutionPolicy.model_validate(execution_policy or {})
        if mode == "auto" and workflow is None:
            return CommandResult(status="error", message="AUTO requires a validated coding workflow decision.")
        if workflow and (mode != workflow.mode or mode == "plan"
                         or (repair and workflow.requested_effect != "write")):
            return CommandResult(status="error", message="Coding workflow authority does not match the requested action.")
        if not goal and workflow:
            return CommandResult(status="error", message="The original repository request is missing from TaskBoard.")
        if not goal:
            return CommandResult(status="input_required", message="What would you like to plan?")
        if output and Path(output).expanduser().resolve().is_relative_to(self.gateway.root):
            return CommandResult(status="error", message="Plan exports must be outside the target checkout.")
        if mode == "plan":
            current = self.active(session_id)
            if current and current.task_id != task_id:
                return self.describe(current)
        if not task_id:
            record = self.gateway._workspaces.store.get_session(session_id)
            reservation = self.lane.reserve(
                normalized_intent=f"{mode}: {goal}",
                lane_id=LaneId.RESEARCH if mode == "plan" else LaneId.CODING,
                workflow_coordinator=mode != "plan",
                session_id=session_id, workspace_id=record.workspace_id,
                repository_id=record.primary_repository_id,
                trigger_turn_id=activity.execution_id if activity and activity.conversation_id == session_id else "",
            )
            task_id = reservation.execution.task_id
            if reservation.duplicate:
                return self.describe(self.board.get_task(task_id))
            if policy.required:
                def bind_policy(task):
                    task.execution_policy = policy
                self.lane.execution_supervisor.store.update_task(task_id, bind_policy)
            self.lane.start(reservation)
        existing = self.board.get_task(task_id).mode_workflow
        if existing:
            if existing.get("mode") not in {"auto", "debug"} or not workflow:
                return self.describe(self.board.get_task(task_id))
            if existing.get("inbox_id") or existing.get("phase") in _TERMINAL:
                return self.describe(self.board.get_task(task_id))
            if existing.get("plan"):
                return self._execute(task_id, existing, self._service(state=existing, llm=llm))
            if existing["phase"] == Phase.BLOCKED:
                existing["investigation_epoch"] = existing.get("investigation_epoch", 0) + 1
                self._transition(task_id, existing, Phase.UNDERSTANDING)
            return self._drive(task_id, existing, llm=llm)
        state = {
            "schema_version": 1, "mode": mode, "phase": Phase.UNDERSTANDING.value,
            "goal": goal, "revision": 1, "constraints": {}, "evidence": [],
            "repair": repair and not read_only, "no_code": no_code or read_only, "save": save,
            "output": output, "question_limit": question_limit, "questions_asked": 0,
            "skills": list(skill_names), "done_steps": {}, "files_changed": [],
            "execution_policy": policy.model_dump(mode="json"),
            "reviewer_id": reviewer_id or getpass.getuser(),
        }
        if workflow:
            resolved_request = safe_text(resolved_request, limit=None, protected=protected).strip()
            state.update({
                "request": goal,
                "goal": resolved_request or goal,
                "resolved_request": resolved_request if resolved_request != goal else "",
                "conversation_context": [
                    {"message_id": message["message_id"], "role": message["role"],
                     "content": safe_text(message.get("content"), limit=None, protected=protected)}
                    for message in conversation_messages
                    if message.get("message_id") and message.get("role") in {"user", "assistant"}
                ],
                "coding_decision": redact_secrets(workflow.model_dump(mode="json")),
                "mode_reason": str(redact_secrets(workflow.reason)),
                "routing_decision_id": self._check_live(task_id).routing_decision_id,
                "source_message_ids": list(source_message_ids),
                "related_task_id": related_task_id,
                "runtime_capability_change": runtime_capability_change,
                "model": model, "provider": provider,
            })
        self._persist(task_id, state)
        if workflow and state["repair"] and not policy.required:
            coding = self.gateway._coding_agent
            settings = getattr(coding, "codex_settings", None)
            if getattr(coding, "active_engine_mode", "external") == "external" and getattr(settings, "worktree_isolation", False):
                try:
                    from mana_agent.multi_agent.worktrees import WorkspaceManager, WorkspaceStatus

                    task = self.board.get_task(task_id)
                    manager = WorkspaceManager(self.gateway.root, repository_id=task.primary_repository_id)
                    state["worktree_required"] = True
                    self._persist(task_id, state)
                    workspace = manager.create_for_task(task_id, title=state["goal"], session_id=session_id,
                                                        assigned_agent_id="coding_agent")
                    manager.transition(task_id, WorkspaceStatus.RUNNING)
                    state["isolated_workspace"] = {
                        "workspace_id": workspace.workspace_id, "repository_id": workspace.repository_id,
                        "task_id": task_id, "path": workspace.worktree_path,
                    }
                    self._persist(task_id, state)
                    execution = self.lane.inspect_task(task_id)
                    supervisor = self.lane.execution_supervisor

                    def bind_workspace(record):
                        supervisor._validate_lease(record, attempt_id=execution.supervisor_attempt_id,
                                                   lease_token=execution.supervisor_lease_token)
                        record.workspace_path = workspace.worktree_path

                    supervisor.store.update_task(task_id, bind_workspace)
                except Exception as exc:
                    return self._block(task_id, state, f"Could not prepare the required isolated worktree: {exc}")
        if workflow:
            with self._activity(task_id, state, read_only=True):
                self._transition(task_id, state, Phase.UNDERSTANDING, state["mode_reason"])
        return self._drive(task_id, state, llm=llm)

    @contextmanager
    def _activity(self, task_id: str, state: dict, *, read_only: bool):
        supervisor = self.lane.execution_supervisor
        execution = self.lane.inspect_task(task_id)
        durable = supervisor.get_task(task_id)
        task = self.board.get_task(task_id)
        activity = current_activity_context()
        if activity and activity.conversation_id != task.session_id:
            activity = None
        turn_id = activity.execution_id if activity else ""
        protected = activity.protected_text if activity else activity_protected_text(getattr(self.gateway, "settings", None))
        governor = getattr(getattr(self.gateway, "_stack", None), "context_cost_governor", None)
        identity = governor.scoped_execution_identity(
            task_id=task_id, root_task_id=task.root_task_id, attempt_id=durable.attempt_id,
            checkpoint_id=durable.checkpoint_id, step_id=state.get("step_id", ""),
            route=state["mode"], lane=execution.owning_lane.value, execution_kind="repository_mode",
        ) if governor else nullcontext()
        with identity, activity_scope(ActivityContext(
            conversation_id=task.session_id,
            execution_id=turn_id or durable.trigger_turn_id or task_id,
            task_id=task_id,
            root_task_id=task.root_task_id, repository_id=task.primary_repository_id,
            attempt_id=execution.supervisor_attempt_id, lease_token=execution.supervisor_lease_token,
            attempt_generation=durable.attempt_generation, supervisor=supervisor,
            protected_text=protected,
            execution_registry=getattr(self.gateway, "_execution_registry", None),
            execution_lane=execution.owning_lane.value,
            parent_task_id=task.parent_task_id or "",
            taskboard=self.board, execution_mode="read_only" if read_only else "act",
        )):
            yield

    def _check_live(self, task_id: str):
        supervisor = self.lane.execution_supervisor
        execution = self.lane.inspect_task(task_id)
        task = supervisor.get_task(task_id)
        supervisor._validate_lease(task, attempt_id=execution.supervisor_attempt_id,
                                   lease_token=execution.supervisor_lease_token)
        if task.state != ExecutionState.RUNNING or task.cancellation_status is not None:
            raise PermissionError("The mode task is no longer active")
        if task.wall_clock_deadline_exceeded(supervisor.clock()):
            raise TimeoutError("The mode task reached its execution deadline")
        return task

    def _drive(self, task_id: str, state: dict, *, service: PlanService | None = None, llm=None) -> CommandResult:
        activity = None
        try:
            with self._activity(task_id, state, read_only=True):
                activity = current_activity_context()
                with activity.publication():
                    if state.get("coding_decision") and "request" not in state:
                        # Older workflows saved the routing objective as their goal.
                        request = safe_text(self.board.get_task(task_id).user_request, limit=None,
                                            protected=activity.protected_text).strip()
                        if not request:
                            raise RuntimeError("The original repository request is missing from TaskBoard.")
                        state["request"] = request
                        if not state.get("plan"):
                            state["goal"] = request
                        state["coding_decision"].pop("clarification", None)
                        self._persist(task_id, state)
                    handoff = state.get("investigation", {})
                    if handoff.get("status") == "running" and handoff["attempt_id"] != activity.attempt_id:
                        # Preserve prior charges and reuse only saved reads.
                        handoff["status"] = "interrupted"
                        self._persist(task_id, state)
                if service is None:
                    service = self._service(state=state, llm=llm)
                return self.gateway._autonomous_controller.drive_repository_workflow(
                    advance=lambda: self._advance_decision(task_id, state, service),
                    decisions_used=lambda: state.get("decision_count", 0),
                    check_attempt=lambda: self._check_live(task_id),
                )
        except Exception as exc:
            durable = self.lane.execution_supervisor.get_task(task_id)
            if (isinstance(exc, StaleLeaseError) or durable.cancellation_status is not None
                    or (activity is not None and durable.attempt_id != activity.attempt_id)):
                return CommandResult(status="error", message="The repository workflow attempt is no longer active.",
                                     data={"task_id": task_id, "phase": self.board.get_task(task_id).mode_workflow["phase"],
                                           "mode": state["mode"], "goal_satisfied": False})
            return self._block(task_id, state, str(redact_secrets(str(exc))))

    def _advance_decision(self, task_id, state, service) -> CommandResult | None:
        with current_activity_context().publication():
            for field in ("model", "provider"):
                selected = getattr(service.llm, field, None)
                if not state.get(field) and isinstance(selected, str) and selected:
                    state[field] = selected
            self._charge_model(task_id, state)
        evidence = state["evidence"]
        scope = investigation_scope(evidence)
        handoff = state.get("investigation", {})
        planning = state["mode"] == "plan"
        remaining_tools = self._remaining_tools(state)
        remaining_decisions = self._remaining_decisions(state)
        boundary = list(self._investigation_boundary(state))
        inspection = state.get("inspection", {})
        stop = inspection.get("stop", "") if inspection.get("boundary") == boundary else ""
        can_inspect = not planning and remaining_tools > 0 and remaining_decisions > 0 and not stop
        can_investigate = (planning and remaining_tools > 0 and remaining_decisions > 1
                           and (handoff.get("revision"), handoff.get("epoch")) != tuple(boundary))
        report = handoff.get("report") or {}
        cites = set(report.get("evidence_ids", []))
        latest = {row["id"] for row in evidence[-8:]}
        observations = [row for row in evidence if row["id"] in cites | latest]
        decision = service.next_decision(state["mode"], {
            "task_id": task_id, "coding_workflow": state.get("coding_decision"),
            "runtime_capability_change": state.get("runtime_capability_change", False),
            "completed_step_ids": list(state["done_steps"]),
            "recheck_required": state.get("required_recheck"),
            "request": state.get("request", state["goal"]), "resolved_request": state.get("resolved_request", ""),
            "conversation_context": state.get("conversation_context", []),
            "constraints": state["constraints"], "skills": state["skills"],
            "evidence": _investigation_context(observations),
            "evidence_levels": scope.evidence_levels,
            "max_next_level": scope.max_next_level,
            "remaining_inspections": min(remaining_tools, MAX_INSPECTION_BATCH) if can_inspect else 0,
            "remaining_workflow_decisions": max(0, remaining_decisions),
            "remaining_workflow_tools": max(0, remaining_tools),
            "inspection_stop": stop or ("Shared workflow budget" if not planning and not can_inspect else ""),
            "investigation_report": report,
            "revision_request": state.get("revision_request", ""),
            "repair_authorized": state["repair"],
        }, investigation_available=can_investigate, inspection_available=can_inspect, scope=scope)
        with current_activity_context().publication():
            self._check_live(task_id)
        allowed = {"investigate" if planning else "inspect", "clarify", "block"}
        if state["mode"] in {"auto", "plan"}:
            allowed.add("draft")
        if state["mode"] == "debug":
            allowed.update({"diagnose", "repair", "research"})
        if state.get("coding_decision"):
            allowed.add("switch_workflow")
        if decision.action not in allowed:
            raise ValueError("The model selected an action outside the active mode")
        if decision.action == "inspect":
            if not can_inspect:
                raise PermissionError("Repository reads are unavailable for this decision.")
            self._inspect(task_id, state, service, decision, remaining_tools)
            return None
        if decision.action == "investigate":
            if not can_investigate:
                raise PermissionError("Repository investigation is unavailable for this PLAN decision.")
            self._investigate(task_id, state, service, decision)
            return None
        if decision.action == "clarify":
            question_total = state["questions_asked"] + len(decision.questions)
            if state["question_limit"] is not None and question_total > state["question_limit"]:
                raise RuntimeError("The clarification limit was reached with material decisions unresolved")
            state["questions_asked"] = question_total
            questions = [redact_secrets(q.model_dump(mode="json")) for q in decision.questions]
            self._transition(task_id, state, Phase.CLARIFICATION_REQUIRED)
            item = self._wait(task_id, state, questions=questions)
            return self._describe_item(task_id, state, item)
        if decision.action == "block":
            return self._block(task_id, state, decision.reason)
        if decision.action == "switch_workflow":
            cited = set(decision.evidence_ids)
            if (decision.task_id != task_id or decision.mode == state["mode"]
                    or not cited.issubset(scope.evidence_levels)
                    or cited.issubset(state.get("last_switch_evidence", []))
                    or (decision.mode == "auto" and not state["repair"])):
                raise PermissionError("Workflow transition requires new observed evidence and the original task authority")
            state["mode"] = decision.mode
            state["mode_reason"] = str(redact_secrets(decision.reason))
            state["last_switch_evidence"] = list(scope.evidence_levels)
            state.setdefault("mode_transitions", []).append(redact_secrets(decision.model_dump(mode="json")))
            self._transition(task_id, state, Phase.INVESTIGATING, state["mode_reason"])
            return None
        if decision.action == "research":
            return self._research(task_id, state, decision.debugging_report)
        if decision.action == "diagnose":
            if not set(decision.cause_evidence).issubset(scope.evidence_levels):
                raise ValueError("Diagnosis referenced unknown evidence")
            text = "\n\n".join(s for s in (decision.root_cause, decision.uncertainty) if s)
            state["root_cause"] = str(redact_secrets(decision.root_cause))
            state["uncertainty"] = str(redact_secrets(decision.uncertainty))
            if state.get("coding_decision") and state["repair"]:
                return self._block(task_id, state, f"{text}\n\nThe requested repair remains incomplete.")
            return self._complete(task_id, state, text)
        if decision.action == "repair":
            if not state["repair"]:
                raise PermissionError("This debugging request only authorizes diagnosis")
            if not set(decision.cause_evidence).issubset(scope.evidence_levels):
                raise ValueError("Repair referenced unknown evidence")
            if not any(step.changes for step in decision.draft.steps) or not any(
                check.expected == "pass" for step in decision.draft.steps for check in step.verification
            ):
                raise ValueError("Repair requires a scoped fix and a regression check")
            state["root_cause"] = str(redact_secrets(decision.root_cause))
            state["uncertainty"] = str(redact_secrets(decision.uncertainty))
        if state["mode"] == "auto" and not scope.evidence_levels:
            raise ValueError("AUTO requires observed repository evidence before drafting")
        return self._draft(task_id, state, decision.draft, service)

    @staticmethod
    def _investigation_boundary(state: dict) -> tuple[int, int]:
        return state["revision"], state.get("investigation_epoch", 0)

    @contextmanager
    def _repository_reader(self, task_id: str, service: PlanService):
        from mana_agent.coding.models import WorkspaceContext
        from mana_agent.gateway.lanes import validate_tool_permission
        from mana_agent.multi_agent.agents.tool_runner_agent import ToolRunnerAgent, bind_tool_authority
        from mana_agent.multi_agent.core.types import AgentRole
        from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
        from mana_agent.multi_agent.tools.operation_policy import builtin_policy

        activity = current_activity_context()
        execution = self.lane.inspect_task(task_id)
        registry = activity.execution_registry or AgentRegistry(default_main_agent_id="main")
        actor = registry.find_by_role(AgentRole.CODING)
        task = self.board.get_task(task_id)
        schemas = {
            name: schema for name, schema in read_tool_schemas().items()
            if name not in task.forbidden_tools
            and (not task.allowed_tools or name in task.allowed_tools)
            and (not actor.allowed_tools or name in actor.allowed_tools)
        }
        if not schemas:
            raise PermissionError("The task is not authorized to use repository investigation tools.")
        policies = {name: builtin_policy(name) for name in schemas}
        if any(policy is None or policy.effect != "read_only" or not policy.repository for policy in policies.values()):
            raise PermissionError("An investigation tool has no registered repository read policy.")

        def authorize(name):
            with activity.publication():
                self._check_live(task_id)
                return validate_tool_permission(self.lane.contracts[execution.owning_lane], name,
                                                task_capabilities=execution.capabilities)

        for name in schemas:
            authorize(name)
        with activity.publication():
            self._check_live(task_id)
            if actor.agent_id not in task.assigned_agent_ids:
                self.board.assign(task_id, actor.agent_id)
        with activity_scope(replace(activity, agent_id=actor.agent_id, parent_agent_id=actor.parent_agent_id or "")):
            authority = bind_tool_authority(
                registry=registry, actor_id=actor.agent_id, task_id=task_id,
                workspace=WorkspaceContext(repository_path=self.gateway.root, worktree_path=service.repo_root,
                                           sandbox="readOnly"),
                policies=policies, schemas=schemas, taskboard=self.board, authorize=authorize,
                record_taskboard_evidence=False,
            )
            yield ToolRunnerAgent(authority)

    def _inspect(self, task_id: str, state: dict, service: PlanService, decision: InspectDecision,
                 available_tools: int) -> None:
        from mana_agent.multi_agent.tools.operation_policy import validate_arguments

        activity = current_activity_context()
        if not decision.reason.strip():
            raise ValueError("Repository inspection must explain what the next coding decision needs.")
        decision_id = f"{task_id}:{activity.attempt_generation}:inspect:{state['decision_count']}"
        evidence = state["evidence"]
        boundary = list(self._investigation_boundary(state))
        previous = state.get("inspection", {})
        inspection = {"id": decision_id, "boundary": boundary, "no_progress": 0, "need": safe_text(
            decision.reason, protected=activity.protected_text,
        )}
        if previous.get("boundary") == boundary:
            inspection["no_progress"] = previous.get("no_progress", 0)

        with self._repository_reader(task_id, service) as runner:
            schemas = runner.authority.schemas
            investigation_scope(evidence).validate_requests(decision.requests)
            selected = []
            for request in decision.requests:
                if not request.reason.strip():
                    raise ValueError("Each repository read must explain the fact it will resolve.")
                if request.tool not in schemas:
                    raise PermissionError("The model selected a repository tool outside the registered read-only set.")
                service.validate_inspection_target(request)
                arguments = request.model_dump(mode="json", include=set(schemas[request.tool]["properties"]))
                validate_arguments(arguments, schemas[request.tool])
                selected.append(runner.request(
                    request.tool, arguments, call_id=f"{decision_id}:{len(selected) + 1}", decision_id=decision_id,
                ))
            with activity.publication():
                self._check_live(task_id)
                state["inspection"] = inspection
                if len(selected) > available_tools:
                    inspection["stop"] = "The requested batch exceeds the remaining shared tool budget."
                    self._persist(task_id, state)
                    return
                self._transition(task_id, state, Phase.INVESTIGATING, "Reading evidence for the next coding decision")

            fingerprints = {observation_fingerprint(row) for row in evidence}
            new_evidence = False
            for tool_call, request in zip(selected, decision.requests, strict=True):
                with activity.publication():
                    self._check_live(task_id)
                    self._charge_tool(task_id, state)
                result = runner.execute(tool_call, lambda _: service.inspect(request)).result
                with activity.publication():
                    self._check_live(task_id)
                    row = {"id": f"evidence-{len(evidence) + 1}", "level": request.level,
                           "fingerprint": request.fingerprint, "result": result}
                    evidence.append(row)
                    fingerprint = observation_fingerprint(row)
                    new_evidence |= fingerprint is not None and fingerprint not in fingerprints
                    self._persist(task_id, state)
            with activity.publication():
                self._check_live(task_id)
                inspection["no_progress"] = 0 if new_evidence else inspection["no_progress"] + 1
                if inspection["no_progress"] >= self.gateway._autonomous_controller.config.max_no_progress_cycles:
                    inspection["stop"] = "Repository reads stopped yielding new evidence."
                self._transition(task_id, state, Phase.UNDERSTANDING, "Repository observations saved")

    def _investigate(self, task_id: str, state: dict, service: PlanService, decision: InvestigateDecision) -> None:
        activity = current_activity_context()
        revision, epoch = self._investigation_boundary(state)
        decision_id = f"{task_id}:{activity.attempt_generation}:investigate:{state['decision_count']}"
        evidence = state["evidence"]
        handoff = {
            "id": decision_id, "attempt_id": activity.attempt_id, "revision": revision, "epoch": epoch,
            "selection": {key: safe_text(value, protected=activity.protected_text)
                          for key, value in decision.model_dump().items()},
            "status": "running", "model": getattr(service.llm, "model", ""),
            "provider": getattr(service.llm, "provider", ""),
        }
        with activity.publication():
            self._check_live(task_id)
            state["investigation"] = handoff
            self._transition(task_id, state, Phase.INVESTIGATING, "Investigating the repository")

        def check_live():
            with activity.publication():
                self._check_live(task_id)

        def charge(kind, charge_operation):
            with activity.publication():
                self._check_live(task_id)
                handoff[kind] = handoff.get(kind, 0) + 1
                charge_operation(task_id, state)

        def checkpoint(rows):
            with activity.publication():
                self._check_live(task_id)
                evidence.extend(rows)
                self.board.update_task(task_id, mode_workflow=state,
                                       progress=f"Investigating the repository · {len(evidence)} observations saved")

        report = None
        try:
            with self._repository_reader(task_id, service) as runner:
                settings = getattr(self.gateway, "settings", None)
                report = run_repository_investigation(
                    service=service, decision=InvestigateDecision.model_validate(handoff["selection"]),
                    decision_id=decision_id, runner=runner, evidence=evidence,
                    remaining_model_turns=lambda: self._remaining_decisions(state) - 1,
                    remaining_tool_calls=lambda: self._remaining_tools(state),
                    charge_model=lambda: charge("model_turns", self._charge_model),
                    charge_tool=lambda: charge("tool_calls", self._charge_tool),
                    checkpoint=checkpoint, check_live=check_live,
                    programmatic_enabled=getattr(settings, "mana_internal_programmatic_tool_calling", True),
                    max_no_progress=self.gateway._autonomous_controller.config.max_no_progress_cycles,
                )
        finally:
            with activity.publication():
                self._check_live(task_id)
                handoff["status"] = "completed" if report is not None else "failed"
                if report is not None:
                    handoff["report"] = {
                        "findings": safe_text(report.findings, protected=activity.protected_text),
                        "evidence_ids": list(report.evidence_ids),
                        "gaps": safe_text(report.gaps, protected=activity.protected_text),
                    }
                progress = "Repository investigation finished" if report is not None else "Repository investigation stopped"
                self.board.update_task(task_id, mode_workflow=state, progress=progress)

    def _remaining_tools(self, state: dict) -> int:
        return self.gateway._autonomous_controller.config.max_tool_calls - state.get("tool_operations", len(state["evidence"]))

    def _remaining_decisions(self, state: dict) -> int:
        return self.gateway._autonomous_controller.config.max_model_turns - state.get("decision_count", 0)

    def _charge_tool(self, task_id: str, state: dict) -> None:
        used = state.get("tool_operations", len(state["evidence"]))
        if used >= self.gateway._autonomous_controller.config.max_tool_calls:
            raise RuntimeError("The configured workflow tool budget was exhausted.")
        state["tool_operations"] = used + 1
        self._persist(task_id, state)

    def _charge_model(self, task_id: str, state: dict) -> None:
        used = state.get("decision_count", 0)
        if used >= self.gateway._autonomous_controller.config.max_model_turns:
            raise RuntimeError("The configured workflow model decision budget was exhausted.")
        state["decision_count"] = used + 1
        self._persist(task_id, state)

    def _research(self, task_id, state, report) -> CommandResult | None:
        from mana_agent.execution.continuation_decision import ContinuationAction
        from mana_agent.gateway.lanes import validate_tool_permission

        if not state["evidence"]:
            raise ValueError("Research requires a local observation about the dependency")
        self._charge_tool(task_id, state)
        self._charge_model(task_id, state)
        settings = self.gateway.settings
        if not settings.mana_coding_research_enabled:
            raise PermissionError("External coding research is disabled")
        execution = self.lane.inspect_task(task_id)
        self._check_live(task_id)

        def authorize(name):
            self._check_live(task_id)
            return validate_tool_permission(self.lane.contracts[execution.owning_lane], name,
                                            task_capabilities=execution.capabilities)

        runtime = self.gateway._autonomous_controller.coding_research_runtime(
            task_id=task_id, root=self._workspace_root(state), attempt_id=execution.supervisor_attempt_id,
            lease_token=execution.supervisor_lease_token, authorize=authorize,
            evaluator=self.gateway._continuation_evaluator, settings=settings,
        )
        status = {"debugging_report": redact_secrets(report.model_dump(mode="json")), "status": "waiting"}
        decision = runtime.decide(goal=state["goal"], result=status)
        if decision.action == ContinuationAction.BLOCKED:
            return self._block(task_id, state, decision.reason)
        observation = runtime.context(status)
        if decision.action == ContinuationAction.WAIT_FOR_USER:
            protected = current_activity_context().protected_text
            observation["clarification_suggestion"] = {
                "question": safe_text(decision.user_message, protected=protected),
                "reason": safe_text(decision.reason, protected=protected),
            }
        evidence = {"id": f"evidence-{len(state['evidence']) + 1}", "fingerprint": canonical_digest(status),
                    "level": 0, "result": observation}
        if any(row["fingerprint"] == evidence["fingerprint"] and row["result"] == evidence["result"]
               for row in state["evidence"]):
            raise RuntimeError("External investigation produced no new evidence")
        state["evidence"].append(evidence)
        state["investigation_epoch"] = state.get("investigation_epoch", 0) + 1
        self._persist(task_id, state)
        return None

    def _draft(self, task_id, state, draft, service):
        limits = state.get("repair_limits")
        if limits is not None:
            if any(operation not in limits.get(path, []) for step in draft.steps
                   for path, operations in step.changes.items() for operation in operations):
                raise PermissionError("The verification repair extends beyond the accepted file scope")
            if not any(check.model_dump(mode="json") == state["required_recheck"]
                       for step in draft.steps for check in step.verification):
                raise ValueError("The revised plan must rerun the failed verification command")
        self._transition(task_id, state, Phase.DRAFTING_PLAN)
        task = self.board.get_task(task_id)
        root = self._workspace_root(state)
        policy = TaskExecutionPolicy.model_validate(state["execution_policy"])
        backend = "internal" if policy.required else getattr(self.gateway._coding_agent, "active_engine_mode", "")
        plan = PlanRevision(
            plan_id=task_id, revision=state["revision"], session_id=task.session_id,
            workspace_id=task.workspace_id, repository_id=task.primary_repository_id,
            repository_root=str(root), goal=state["goal"], constraints=state["constraints"],
            base_file_digests={path: self._file_digest(root, path) for step in draft.steps for path in step.changes},
            base_checkout=self._checkout_identity(root),
            coding_backend=backend, execution_policy=state["execution_policy"],
            draft=draft,
        )
        material = plan.model_dump(mode="json")
        if redact_secrets(material) != material:
            raise ValueError("The model included protected data in the draft")
        state["plan"] = material
        state["approval_id"] = ""
        state["approved_digest"] = ""
        if state["save"] and state["mode"] == "plan":
            state["artifact"] = str(service.save_plan(plan, state["output"]))
        if state["no_code"]:
            return self._complete(task_id, state, plan.to_markdown())
        if state["mode"] in {"auto", "debug"}:
            if not state["repair"]:
                raise PermissionError("The task does not authorize repository changes")
            self._transition(task_id, state, Phase.APPROVED, "Internal plan recorded")
            return self._execute(task_id, state, service)
        self._transition(task_id, state, Phase.AWAITING_APPROVAL)
        item = self._wait(task_id, state, plan=plan)
        return self._describe_item(task_id, state, item)

    def _wait(self, task_id, state, *, questions=(), plan=None):
        checkpoint = self.lane.checkpoint(
            task_id, boundary=f"{state['mode']}.{state['phase']}",
            resume_payload={"mode_task_id": task_id, "revision": state["revision"], "digest": plan.digest if plan else ""},
            completed_steps=list(state["done_steps"]), pending_steps=[s.step_id for s in plan.draft.steps] if plan else (),
        )
        if not checkpoint:
            raise RuntimeError("Could not persist the mode checkpoint")
        context = {"mode_task_id": task_id, "revision": state["revision"], "repository_root": str(self.gateway.root)}
        title = "Review implementation plan" if plan else "Answer clarification questions"
        request = InboxRequest(
            request_type=InboxRequestType.APPROVAL if plan else InboxRequestType.CLARIFICATION,
            task_id=task_id, branch_id=task_id, checkpoint_id=checkpoint,
            requested_by_agent_id=_OWNER, reviewer=ReviewerAssignment(reviewer_type=ReviewerType.PERSON,
                                                                     reviewer_id=state["reviewer_id"]),
            title=title, summary=plan.to_markdown() if plan else "\n".join(q["prompt"] for q in questions),
            action_intent_id=f"plan:{task_id}:r{state['revision']}" if plan else "",
            action_digest=plan.digest if plan else "",
            deduplication_key=f"mode:{task_id}:{checkpoint}",
            minimal_context={"plan_id": task_id, "revision": state["revision"], "digest": plan.digest} if plan else {"questions": list(questions)},
            protected_context={**context, "plan": plan.model_dump(mode="json")} if plan else {**context, "questions": list(questions)},
            requested_fields=[
                ClarificationField(field_id=q["field_id"], prompt=q["prompt"], choices=list(q["choices"]),
                                   allow_free_form=q["allow_free_form"],
                                   expected_type=ExpectedResponseType.CHOICE if q["choices"] else ExpectedResponseType.TEXT)
                for q in questions
            ],
            allowed_responses=[ResponseOperation.APPROVE, ResponseOperation.DENY] if plan else [ResponseOperation.ANSWER],
        )
        item = self.inbox.create(request)
        state["inbox_id"] = item.inbox_item_id
        self._persist(task_id, state)
        return item

    def _describe_item(self, task_id, state, item):
        context = {"task_id": task_id, "phase": state["phase"], "mode": state["mode"],
                   "mode_reason": state.get("mode_reason", ""), "inbox_id": item.inbox_item_id,
                   "revision": state["revision"], "digest": item.action_digest}
        if item.request_type == InboxRequestType.APPROVAL:
            text = f"{item.summary}\nApprove revision {state['revision']} with `/approve {item.inbox_item_id}`, or use `/revise <changes>` or `/reject`."
            return CommandResult(status="input_required", message=text, data=context,
                                 next_prompt={"kind": "mode.approval", "preview": item.summary, **context})
        questions = item.minimal_context["questions"]
        lines = ["Waiting for your answers. This task will continue when you submit them."]
        for question in questions:
            lines.append(f"**{question['field_id']}** · {question['prompt']}")
            for key, explanation in question["choices"].items():
                recommended = " (Recommended)" if key == question["recommended_choice"] else ""
                lines.append(f"- {key}{recommended} · {explanation}")
            if question["allow_free_form"]:
                lines.append("- Custom answer accepted")
        lines.append(f"`/inbox answer {item.inbox_item_id} '<JSON object keyed by question ID>'`")
        return CommandResult(status="input_required", message="\n\n".join(lines), data=context,
                             next_prompt={"kind": "mode.clarification", "questions": questions, **context})

    def pending_prompt(self, session_id: str, task_id: str, inbox_id: str) -> dict | None:
        task = self.board.store.repository.get_task_or_none(task_id)
        if task is None or task.session_id != session_id or not inbox_id:
            return None
        state = task.mode_workflow
        if (state.get("inbox_id") != inbox_id
                or state.get("phase") not in {Phase.CLARIFICATION_REQUIRED, Phase.AWAITING_APPROVAL}):
            return None
        item = self.inbox.repository.get_or_none(inbox_id)
        if item and item.task_id == task_id and item.status in {InboxStatus.PENDING, InboxStatus.DELIVERED}:
            return self._describe_item(task_id, state, item).next_prompt
        return None

    def describe(self, task):
        state = task.mode_workflow
        inbox_id = state.get("inbox_id")
        if inbox_id and state["phase"] in {Phase.CLARIFICATION_REQUIRED, Phase.AWAITING_APPROVAL}:
            item = self.inbox.repository.get(inbox_id)
            if item.status in {InboxStatus.PENDING, InboxStatus.DELIVERED}:
                return self._describe_item(task.task_id, state, item)
        text = state.get("result") or f"{state['mode'].capitalize()} task {task.task_id} is {state['phase'].replace('_', ' ')}."
        if self._can_retry_investigation(state):
            durable = self.lane.execution_supervisor.store.get_task_or_none(task.task_id)
            if state["phase"] == Phase.BLOCKED or (durable and durable.state == ExecutionState.INTERRUPTED):
                text += f"\n\nResume the investigation with `/task resume {task.task_id}`."
        if state["phase"] in {Phase.COMPLETED, Phase.BLOCKED}:
            return self._outcome(task.task_id, state, text, self._repository_report(state),
                                 completed=state["phase"] == Phase.COMPLETED)
        if state.get("plan") and state["phase"] not in _TERMINAL:
            text += "\n\n" + PlanRevision.model_validate(state["plan"]).to_markdown()
        return CommandResult(status="success", message=text,
                             data={"task_id": task.task_id, "phase": state["phase"]})

    def receive(self, session_id, text, *, task=None, read_only=False, question_limit=None, actor_id=None):
        if task is None:
            return self.start(session_id, text, read_only=read_only, question_limit=question_limit, reviewer_id=actor_id or "")
        state = task.mode_workflow
        if state["phase"] == Phase.CLARIFICATION_REQUIRED and state.get("inbox_id"):
            if read_only or actor_id == "":
                return self.describe(task)
            item = self.inbox.repository.get(state["inbox_id"])
            try:
                answer = json.loads(text)
            except ValueError:
                answer = None
            if not isinstance(answer, dict) and len(item.requested_fields) == 1:
                field = item.requested_fields[0]
                answer = {field.field_id: text.strip()}
            if isinstance(answer, dict):
                try:
                    return self.respond(session_id, item.inbox_item_id, ResponseOperation.ANSWER,
                                        frontend="chat", answer=answer, actor_id=actor_id)
                except (InboxConcurrentUpdateError, PermissionError, ValueError) as exc:
                    return CommandResult(status="error", message=str(redact_secrets(str(exc))))
        return self.describe(task)

    def respond(self, session_id, item_id, operation, *, frontend, answer=None, comment="", actor_id=None):
        item = self.inbox.repository.get(item_id)
        task = self.board.get_task(item.task_id)
        state = task.mode_workflow
        if task.session_id != session_id or item_id != state.get("inbox_id"):
            raise PermissionError("This response does not refer to the current mode revision")
        if operation == ResponseOperation.APPROVE and state["phase"] != Phase.AWAITING_APPROVAL:
            raise PermissionError("The current mode task is not awaiting approval")
        if operation == ResponseOperation.ANSWER and state["phase"] != Phase.CLARIFICATION_REQUIRED:
            raise PermissionError("The current mode task is not awaiting clarification")
        if actor_id is None:
            actor_id = local_reviewer_actor(item.assigned_reviewer_id, settings=getattr(self.gateway, "settings", None))
        result = self.inbox.respond(ResponseSubmission(
            inbox_item_id=item_id, operation=operation, actor_id=actor_id, channel=frontend,
            expected_version=item.version, current_action_digest=item.action_digest,
            answer=answer or {}, comment=comment,
        ))
        return CommandResult(status="success", message=f"{result.title}: {result.status.value}.",
                             data={"task_id": task.task_id, "inbox_id": item_id})

    def wait_for_response(self, task_id, inbox_id, *, timeout=600):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            task = self.board.get_task(task_id)
            state = task.mode_workflow
            if state.get("batch_child"):
                return CommandResult(status="success", message="The approved coding step is waiting for its Batch proposal.",
                                     data={"task_id": task_id, "phase": state["phase"]})
            if state.get("inbox_id") != inbox_id and state["phase"] in {
                *_TERMINAL, Phase.BLOCKED, Phase.AWAITING_APPROVAL, Phase.CLARIFICATION_REQUIRED,
            }:
                if state["phase"] not in {Phase.AWAITING_APPROVAL, Phase.CLARIFICATION_REQUIRED} or state.get("inbox_id"):
                    return self.describe(task)
            if state["phase"] in {*_TERMINAL, Phase.BLOCKED}:
                return self.describe(task)
            time.sleep(0.15)
        return CommandResult(status="input_required", message=f"Mode task {task_id} is still pending.", data={"task_id": task_id})

    def recover_queued_responses(self):
        for execution in self.lane.list_tasks():
            if execution.state != LaneTaskState.QUEUED:
                continue
            state = self.board.get_task(execution.task_id).mode_workflow
            if not state.get("mode") or not state.get("inbox_id"):
                continue
            item = self.inbox.repository.get(state["inbox_id"])
            if item.resume_claim_id and item.response:
                self.lane.dispatch_queued_human_resume(execution.task_id, inbox_item_id=item.inbox_item_id,
                    resume_claim_id=item.resume_claim_id, structured_response=item.response.model_dump(mode="json"))

    def retry(self, task_id: str, *, revision_request: str = "") -> CommandResult:
        from mana_agent.execution_supervisor.models import RecoveryAction, RecoveryDecision, RetryCategory

        task = self.board.get_task(task_id)
        state = task.mode_workflow
        retry_investigation = self._can_retry_investigation(state)
        if state.get("inbox_id") or (state["mode"] == "plan" and not state.get("approved_digest")
                                    and not (revision_request or retry_investigation)):
            return self.describe(task)
        if state.get("in_flight_mutation") and not state["in_flight_mutation"].get("action_id"):
            return CommandResult(status="error", message="Reconcile the interrupted patch before resuming or revising this task.")
        lane_task = self.lane.ensure_recoverable_execution(task_id)
        if lane_task.state not in self.lane._RETRYABLE_LANE_STATES and state.get("phase") in {Phase.BLOCKED.value, Phase.BLOCKED}:
            self.lane.mark_blocked(task_id, reason=state.get("result") or "Operator requested mode task recovery")
        reservation = self.lane.retry_task(task_id, session_id=task.session_id, decision=RecoveryDecision(
            decision_id=f"mode-retry:{task_id}:{time.time_ns()}", task_id=task_id,
            action=RecoveryAction.RETRY, retry_category=RetryCategory.MODEL,
            reason="Operator requested mode task recovery", same_task_retry_authorized=True, safe_to_continue=True,
        ))
        self.lane.start(reservation)
        state = self.board.get_task(task_id).mode_workflow
        if revision_request:
            result = self._restart_draft(task_id, state, revision_request)
        elif retry_investigation:
            state["investigation_epoch"] = state.get("investigation_epoch", 0) + 1
            self._transition(task_id, state, Phase.UNDERSTANDING)
            result = self._drive(task_id, state)
        else:
            result = self._execute(task_id, state, self._service(state=state))
        self._publish_result(task_id, result)
        return result

    @staticmethod
    def _can_retry_investigation(state: dict) -> bool:
        return state["phase"] in {Phase.BLOCKED, Phase.UNDERSTANDING, Phase.INVESTIGATING} and not any(state.get(key) for key in (
            "plan", "inbox_id", "approval_id", "approved_digest", "done_steps", "in_flight_mutation",
        ))

    def revise_blocked(self, task, instructions):
        return self.retry(task.task_id, revision_request=instructions)

    def note_cancellation(self, task_id):
        task = self.board.store.repository.get_task_or_none(task_id)
        if task and task.mode_workflow.get("mode") and task.mode_workflow["phase"] not in _TERMINAL:
            state = task.mode_workflow
            approval_id = state.get("approval_id") or state.get("inbox_id")
            if approval_id:
                approval = self.inbox.repository.get(approval_id)
                if approval.action_intent_id:
                    self.inbox.supersede_for_action(approval.action_intent_id)
            self.inbox.cancel_for_task(task_id, reason="Mode task cancelled")
            self._discard_patch_approval(state)
            handoff = state.get("investigation", {})
            if handoff.get("status") == "running":
                handoff["status"] = "cancelled"
            self._transition(task_id, state, Phase.CANCELLED)

    def _restart_draft(self, task_id, state, instructions):
        self._revise(task_id, state, instructions)
        return self._drive(task_id, state)

    def resume(self, task_id, item, protected, claim_id):
        supervisor = self.lane.execution_supervisor
        task = supervisor.get_task(task_id)
        state = dict(self.board.get_task(task_id).mode_workflow)
        if (not state or task.state != ExecutionState.QUEUED or item.task_id != task_id
                or item.resume_claim_id != claim_id or state.get("inbox_id") != item.inbox_item_id
                or state["revision"] != protected.get("revision")):
            return
        self.inbox.assert_response_actor_is_currently_authorized(item)
        plan = None
        if item.status == InboxStatus.APPROVED:
            plan = PlanRevision.model_validate(protected.get("plan"))
            if item.action_digest != plan.digest or plan.digest != PlanRevision.model_validate(state["plan"]).digest:
                raise PermissionError("Approval does not match the current plan revision")
        self.lane.start(LaneReservation(self.lane.inspect_task(task_id)))
        if item.status == InboxStatus.DENIED:
            comment = item.response.comment
            if comment.startswith("Revise plan:\n"):
                self._revise(task_id, state, comment.split("\n", 1)[1])
                result = self._drive(task_id, state)
            else:
                self._transition(task_id, state, Phase.CANCELLED, "User rejected the plan")
                self.lane.cancel_tree(task_id)
                result = CommandResult(status="success", message="Plan rejected.")
        elif item.status == InboxStatus.ANSWERED:
            answer = self.inbox.consume_for_agent(item.inbox_item_id, requesting_agent_id=_OWNER, task_id=task_id).answer
            state["constraints"].update(redact_secrets(answer))
            state["inbox_id"] = ""
            state["investigation_epoch"] = state.get("investigation_epoch", 0) + 1
            self._transition(task_id, state, Phase.UNDERSTANDING)
            result = self._drive(task_id, state)
        elif plan:
            state["approved_digest"] = plan.digest
            state["approval_id"] = item.inbox_item_id
            state["inbox_id"] = ""
            self._transition(task_id, state, Phase.APPROVED, f"Revision {plan.revision} approved")
            result = self._execute(task_id, state, self._service(state=state))
        else:
            return
        self._publish_result(task_id, result)

    def _publish_result(self, task_id, result):
        from mana_agent.chat.events import AssistantMessageEvent
        from mana_agent.chat.history import get_history

        task = self.board.get_task(task_id)
        durable = self.lane.execution_supervisor.get_task(task_id)
        turn_id = durable.trigger_turn_id or f"mode-{task_id}-{task.mode_workflow['revision']}-{task.mode_workflow['phase']}"
        self.gateway._append_session_message(task.session_id, role="assistant", content=result.message, turn_id=turn_id)
        get_history().add(AssistantMessageEvent(content=result.message, turn_id=turn_id))
        inbox_id = result.data.get("inbox_id")
        if (result.next_prompt and inbox_id and durable.state == ExecutionState.WAITING
                and durable.waiting_inbox_item_id == inbox_id):
            # This is a projection of a saved wait. Its worker lease has already ended.
            from mana_agent.services.execution_event_hub import get_execution_event_hub

            try:
                get_execution_event_hub().publish({
                    "type": "mode.input_ready", "title": "Plan review or clarification available",
                    "status": "waiting", "task_id": task_id, "operation_id": inbox_id,
                    "metadata": {"inbox_id": inbox_id},
                }, conversation_id=task.session_id, execution_id=turn_id,
                   repository_id=task.primary_repository_id, authoritative=True)
            except Exception:
                logger.warning("Mode inbox wait activity unavailable", exc_info=True)
        if callable(self.gateway._event_sink):
            self.gateway._event_sink("mode.updated", result.message, status=result.status,
                                     metadata={"task_id": task_id, **result.data})

    def _revise(self, task_id, state, reason):
        self._discard_patch_approval(state)
        approval = state.get("approval_id") or state.get("inbox_id")
        if approval:
            item = self.inbox.repository.get(approval)
            if item.action_intent_id:
                self.inbox.supersede_for_action(item.action_intent_id)
        try:
            answer_changes = json.loads(reason)
        except ValueError:
            answer_changes = None
        if isinstance(answer_changes, dict) and set(answer_changes).issubset(state["constraints"]):
            state["constraints"].update(redact_secrets(answer_changes))
        state["revision"] += 1
        state.update({"revision_request": str(redact_secrets(reason)), "inbox_id": "",
                      "approval_id": "", "approved_digest": "", "plan": None})
        self._transition(task_id, state, Phase.UNDERSTANDING, "Plan scope is being revised")

    def _discard_patch_approval(self, state):
        pending = state.pop("pending_patch_approval", None)
        if pending:
            self.inbox.supersede_for_action(pending["action_id"])

    @staticmethod
    def _checkout_identity(root: Path) -> str:
        outputs = []
        for command in (["symbolic-ref", "--quiet", "HEAD"], ["rev-parse", "--verify", "HEAD"]):
            result = subprocess.run(["git", *command], cwd=root, capture_output=True, text=True,
                                    timeout=10, check=False)
            outputs.append(result.stdout.strip() if result.returncode == 0 else "")
        return canonical_digest(outputs)

    @staticmethod
    def _file_digest(root: Path, relative: str) -> str:
        path = root / repository_relative_path(relative)
        if path.resolve() != path:
            raise ValueError(f"Symbolic links are outside the approved file scope: {relative}")
        if not path.exists():
            return "missing"
        if not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError(f"Expected a regular repository file: {relative}")
        with path.open("rb") as stream:
            digest = sha256()
            for chunk in iter(lambda: stream.read(65_536), b""):
                digest.update(chunk)
            return digest.hexdigest()

    def _validate_approval(self, task_id, state, plan):
        durable = self._check_live(task_id)
        current = self.board.get_task(task_id)
        saved = current.mode_workflow
        if (saved["revision"] != plan.revision or current.session_id != plan.session_id or
                current.workspace_id != plan.workspace_id or current.primary_repository_id != plan.repository_id or
                str(self._workspace_root(saved)) != plan.repository_root or PlanRevision.model_validate(saved["plan"]).digest != plan.digest):
            raise PermissionError("The task, revision, or workspace changed")
        if saved["execution_policy"] != plan.execution_policy:
            raise PermissionError("The approved execution policy changed")
        if self._checkout_identity(Path(plan.repository_root)) != plan.base_checkout:
            raise PermissionError("The repository branch or base commit changed. Revise and approve the plan again.")
        session = self.gateway._workspaces.store.get_session(plan.session_id)
        if session.workspace_id != plan.workspace_id or session.primary_repository_id != plan.repository_id:
            raise PermissionError("The session workspace no longer matches the reviewed checkout")
        if state["mode"] != "plan":
            if not state["repair"]:
                raise PermissionError("Repository changes were not authorized")
            return durable
        if saved["approved_digest"] != plan.digest:
            raise PermissionError("The active revision has no exact approval")
        item = self.inbox.repository.get(saved["approval_id"])
        if item.status != InboxStatus.APPROVED or item.action_digest != plan.digest or item.task_id != task_id:
            raise PermissionError("The approval was invalidated or belongs to another revision")
        self.inbox.assert_response_actor_is_currently_authorized(item)
        return durable

    def _execute(self, task_id, state, service):
        try:
            with self._activity(task_id, state, read_only=False):
                return self._execute_steps(task_id, state, service)
        except ModeBatchWaiting as exc:
            return CommandResult(status="input_required", message=str(exc), data={"task_id": task_id})
        except ScopeRevisionNeeded as exc:
            if state["mode"] == "plan":
                return self._restart_draft(task_id, state, str(exc))
            return self._block(task_id, state, str(exc))
        except VerificationUnavailable as exc:
            return self._block(
                task_id, state, f"**Verification pending.** {exc}\n\n"
                f"Once Docker is ready, run `/task resume {task_id}` to continue.",
            )
        except VerificationFailed as exc:
            if state.get("coding_decision") and state["repair"] and "repair_limits" not in state:
                return self._replan_failed_check(task_id, state, service, exc)
            return self._block(
                task_id, state, "**Verification failed.** See Coding Activity for the command and output.\n\n"
                f"After addressing the failure, run `/task resume {task_id}`.",
            )
        except Exception as exc:
            return self._block(task_id, state, str(redact_secrets(str(exc))))

    def _replan_failed_check(self, task_id, state, service, failure: VerificationFailed):
        try:
            plan = PlanRevision.model_validate(state["plan"])
            self._validate_approval(task_id, state, plan)
            pending = state.get("in_flight_mutation")
            if pending:
                self._reconcile_patch(Path(plan.repository_root), pending)
                state.setdefault("earlier_patches", []).append(pending)
                state.pop("in_flight_mutation")
            limits: dict[str, list[str]] = {}
            for step in plan.draft.steps:
                for path, operations in step.changes.items():
                    limits.setdefault(path, []).extend(operations)
                    if "add" in operations and path in state["files_changed"]:
                        limits[path].append("update")
            state.update(repair_limits=limits, required_recheck=failure.command.model_dump(mode="json"))
            observation = {"command": state["required_recheck"], "result": failure.outcome}
            state["evidence"].append({"id": f"evidence-{len(state['evidence']) + 1}", "level": 0,
                                      "fingerprint": canonical_digest(observation), "result": observation})
            self._revise(task_id, state, "Repair the failed verification within the accepted scope and rerun it.")
            return self._drive(task_id, state, service=service)
        except Exception as exc:
            return self._block(task_id, state, str(exc))

    def _execute_steps(self, task_id, state, service):
        plan = PlanRevision.model_validate(state["plan"])
        self._validate_approval(task_id, state, plan)
        if state.get("in_flight_mutation") and not state["in_flight_mutation"].get("action_id"):
            raise RuntimeError("A patch may already have been applied. Reconcile its action receipt before continuing.")
        self._transition(task_id, state, Phase.EXECUTING)
        expected_files = dict(plan.base_file_digests)
        signatures = {step.step_id: canonical_digest({"step": step.model_dump(mode="json"),
                                                       "constraints": plan.constraints}) for step in plan.draft.steps}
        completed = {step_id: receipt for step_id, receipt in state["done_steps"].items()
                     if receipt.get("signature") == signatures.get(step_id)}
        patches = state.setdefault("patch_steps", {}) if state.get("coding_decision") else None
        root = Path(plan.repository_root)
        latest_digests = {}
        for step in plan.draft.steps:
            latest_digests.update(completed.get(step.step_id, {}).get("files", {}))
            saved_patch = (patches or {}).get(step.step_id, {})
            if saved_patch.get("signature") == signatures[step.step_id]:
                latest_digests.update(saved_patch["files"])
        latest_digests.update(state.get("in_flight_mutation", {}).get("files", {}))
        if any(self._file_digest(root, path) != digest for path, digest in latest_digests.items()):
            raise PermissionError("Files from a verified step changed. Review the workspace before resuming.")
        done: set[str] = set()
        for step in plan.draft.steps:
            if not set(step.depends_on).issubset(done):
                raise PermissionError("Step dependencies have not completed")
            signature = signatures[step.step_id]
            receipt = completed.get(step.step_id)
            if receipt:
                expected_files.update(receipt.get("files", {}))
                done.add(step.step_id)
                continue
            state["step_id"] = step.step_id
            self._transition(task_id, state, Phase.EXECUTING, step.instruction)
            revision_reason = self._steering(task_id, state, plan, step, service)
            if revision_reason:
                self._revise(task_id, state, revision_reason)
                return self._drive(task_id, state, service=service)
            prior_patch = (patches or {}).get(step.step_id, {})
            if prior_patch.get("signature") == signature:
                expected_files.update(prior_patch["files"])
            elif step.changes or patches is None:
                child = self._step_task(task_id, step, plan)
                written = self._run_step_child(
                    task_id, child.task_id, state, plan, step, service, expected_files, verify=patches is None,
                )
                if patches is not None:
                    patches[step.step_id] = {"signature": signature, "files": written}
                state.pop("in_flight_mutation", None)
                self._persist(task_id, state)
            if patches is not None and step.verification:
                self._verify_step(task_id, task_id, state, plan, step)
            state["done_steps"][step.step_id] = {"signature": signature,
                "files": {path: expected_files[path] for path in step.changes}}
            self._persist(task_id, state)
            done.add(step.step_id)
            revision_reason = self._steering(task_id, state, plan, step, service)
            if revision_reason:
                return self._restart_draft(task_id, state, revision_reason)
        if state.get("runtime_capability_change") and state["repair"]:
            self._integrate(task_id, state)
        if state.get("coding_decision"):
            self._transition(task_id, state, Phase.REVIEWING, "Reviewing the verified changes")
            self._validate_approval(task_id, state, plan)
            for relative, digest in expected_files.items():
                if self._file_digest(root, relative) != digest:
                    raise PermissionError("The reviewed checkout changed before finalization")
            if not any(check["passed"] and check["exit_code"] == 0
                       for check in state.get("verification", [])):
                raise RuntimeError("A passing verification command is required before completion")
        summary = _changed_files_summary(state["files_changed"])
        lines = [f"{summary} All required verification checks passed.".strip()]
        if state["mode"] == "debug":
            lines.insert(0, state["root_cause"])
            if state.get("uncertainty"):
                lines.append(state["uncertainty"])
        if state.get("isolated_workspace"):
            lines.append(f"Verified in worktree {_code_span(plan.repository_root)}.")
        return self._complete(task_id, state, "\n\n".join(lines))

    def _run_step_child(self, task_id, child_id, state, plan, step, service, expected_files, *, verify):
        root = Path(plan.repository_root)
        supervisor = self.lane.execution_supervisor
        saved = supervisor.store.get_task_or_none(child_id)
        if saved and saved.state == ExecutionState.COMPLETED:
            escrow = supervisor.get_verified_execution_result(child_id)
            if not escrow.is_verified:
                raise PermissionError("The completed child has no verified receipt")
            receipt = state.get("in_flight_mutation", {})
            if step.changes:
                if receipt.get("step") != step.step_id:
                    raise PermissionError("The completed patch has no receipt for this step")
                self._reconcile_patch(root, receipt)
                expected_files.update(receipt["files"])
            return dict(receipt.get("files", {})) if step.changes else {}

        with self.lane.workflow_step_capacity(task_id, child_id):
            self._start_step(task_id, child_id, plan, step, state)
            try:
                with self._activity(child_id, state, read_only=not step.changes):
                    self._execute_step(task_id, child_id, state, plan, step, service, expected_files, verify=verify)
                self._validate_approval(task_id, state, plan)
                written = state.get("in_flight_mutation", {}).get("files", {}) if step.changes else {}
                finished = self.lane.finish(child_id, changed_files=[str(root / path) for path in written],
                    verification_state={"goal_satisfied": True, "mode_step": step.step_id,
                                        "verification": state.get("verification", [])})
                if finished.state != LaneTaskState.COMPLETED:
                    raise RuntimeError(finished.error or "The supervisor has not verified this step")
                return dict(written)
            except ModeBatchWaiting:
                raise
            except Exception as exc:
                if self.lane.inspect_task(child_id).state == LaneTaskState.RUNNING:
                    self.lane.mark_blocked(child_id, reason=str(redact_secrets(str(exc))))
                raise

    def _integrate(self, task_id: str, state: dict) -> None:
        from mana_agent.gateway.feature_integration import FeatureIntegrationCoordinator, IntegrationAuthority

        self._transition(task_id, state, Phase.INTEGRATING, "Checking production integration")
        authority = IntegrationAuthority.from_taskboard(self.board, task_id)
        if authority and authority.is_complete():
            state["integration"] = redact_secrets(asdict(authority))
            self._persist(task_id, state)
            return
        task = self._check_live(task_id)
        self._charge_tool(task_id, state)
        self._charge_model(task_id, state)
        provider = self.gateway._feature_integration_decision_provider(
            task_id=task_id,
            options={"execution_policy": state["execution_policy"], "_selected_model": state.get("model"),
                     "_selected_provider": state.get("provider")},
        )
        try:
            result = FeatureIntegrationCoordinator().run(
                coding_agent=self.gateway._coding_agent,
                core_result={"status": "completed", "changed_files": state["files_changed"]},
                request=state["goal"], gateway_task_id=task_id, flow_id=None,
                runtime_capability_change=True,
                taskboard=self.board, taskboard_parent_task_id=task_id,
                execution_supervisor=self.lane.execution_supervisor,
                trigger_turn_id=task.trigger_turn_id, workspace_root=self._workspace_root(state),
                integration_decision_provider=provider,
            )
        except BatchPending as pending:
            self._suspend_batch_integration(task_id, state, pending.operation.model_dump(mode="json"))
        if not result.passed:
            raise RuntimeError(f"Production integration is incomplete: {result.error_code}")
        state["integration"] = redact_secrets(result.result.get("integration_authority"))
        self._persist(task_id, state)

    def _start_step(self, task_id, child_id, plan, step, state):
        supervisor = self.lane.execution_supervisor
        saved = supervisor.store.get_task_or_none(child_id)
        if saved:
            lane_task = self.lane.ensure_recoverable_execution(child_id)
            if saved.state == ExecutionState.QUEUED and state.get("batch_child") == child_id:
                lane_task.supervisor_attempt_id = lane_task.supervisor_lease_token = ""
                self.lane.start(LaneReservation(lane_task))
                return
            from mana_agent.execution_supervisor.models import RecoveryAction, RecoveryDecision, RetryCategory
            reservation = self.lane.retry_task(child_id, session_id=plan.session_id, decision=RecoveryDecision(
                decision_id=f"mode-step-retry:{task_id}:{child_id}:{time.time_ns()}", task_id=child_id,
                action=RecoveryAction.RETRY, retry_category=RetryCategory.MODEL,
                reason="Retry a step of the approved revision", same_task_retry_authorized=True, safe_to_continue=True,
            ))
        else:
            reservation = self.lane.reserve(
                normalized_intent=f"{task_id}:r{plan.revision}:{step.step_id}",
                lane_id=LaneId.CODING if step.changes else LaneId.VERIFY,
                session_id=plan.session_id, workspace_id=plan.workspace_id, repository_id=plan.repository_id,
                parent_task_id=task_id, taskboard_task_id=child_id,
                target_files=[str(Path(plan.repository_root) / path) for path in step.changes],
                execution_workspace=Path(plan.repository_root),
            )
        if step.changes:
            policy = TaskExecutionPolicy.model_validate(state["execution_policy"])
            if policy.required:
                def bind_policy(task):
                    task.execution_policy = policy
                supervisor.store.update_task(child_id, bind_policy)
        self.lane.start(reservation)

    def _execute_step(self, task_id, child_id, state, plan, step, service, expected_files, *, verify=True):
        applied = state.get("in_flight_mutation", {})
        if applied.get("step") == step.step_id and applied.get("action_id"):
            self._reconcile_patch(Path(plan.repository_root), applied)
            expected_files.update(applied["files"])
        elif step.changes:
            self._apply_step(task_id, child_id, state, plan, step, service, expected_files)
        if verify and step.verification:
            self._verify_step(task_id, child_id, state, plan, step)

    def _verify_step(self, task_id, owner_id, state, plan, step):
        self._transition(task_id, state, Phase.VERIFYING, f"Verifying {step.step_id}")
        for command in step.verification:
            self._validate_approval(task_id, state, plan)
            self._check_live(owner_id)
            self._verify(task_id, command, state)

    def _reconcile_patch(self, root: Path, receipt: dict) -> None:
        from mana_agent.transactional_actions.runtime import default_action_gateway

        action = default_action_gateway(root).store.get_action(receipt.get("action_id", ""))
        if (action is None or action.state.value != "committed" or any(
            self._file_digest(root, path) != digest for path, digest in receipt["files"].items()
        )):
            raise PermissionError("The applied patch could not be reconciled")

    def _step_task(self, task_id, step, plan):
        existing = self.board.store.repository.list_tasks(session_id=plan.session_id)
        identity = {"step_id": step.step_id, "plan_id": task_id, "revision": plan.revision}
        for task in existing:
            if task.parent_task_id == task_id and task.mode_workflow == identity:
                return task
        dependencies = [
            task.task_id for task in existing
            if task.parent_task_id == task_id and task.mode_workflow.get("step_id") in step.depends_on
            and self.board.get_task(task.task_id).status == TaskStatus.DONE
        ]
        child = self.board.create_child_task(task_id, title=step.instruction, user_request=step.instruction,
                                             owner_agent_id="coding_agent", depends_on=dependencies,
                                             acceptance_criteria=list(step.acceptance_criteria))
        return self.board.update_task(child.task_id, mode_workflow=identity)

    def _apply_step(self, task_id, child_id, state, plan, step, service, expected_files):
        root = self._workspace_root(state)
        for relative in step.changes:
            if self._file_digest(root, relative) != expected_files[relative]:
                raise PermissionError(f"{relative} changed after review. Revise the plan or restore the reviewed workspace.")
        if self.gateway._coding_agent is None:
            raise RuntimeError("No configured coding engine is available")
        selected_backend = ("internal" if TaskExecutionPolicy.model_validate(state["execution_policy"]).required
                            else getattr(self.gateway._coding_agent, "active_engine_mode", ""))
        if plan.coding_backend and selected_backend != plan.coding_backend:
            raise ScopeRevisionNeeded("The configured coding backend changed after the plan was prepared")
        pending = state.get("pending_patch_approval")
        approval_id = ""
        if pending:
            if pending["step"] != step.step_id or pending["revision"] != plan.digest:
                raise PermissionError("Patch approval belongs to a different revision or step")
            patch, approval_id = self._approved_patch(root, pending)
        else:
            patch = self._propose_patch(task_id, child_id, state, plan, step, service)
        self._commit_patch(task_id, child_id, state, plan, step, service, expected_files, patch, approval_id)

    def _propose_patch(self, task_id, child_id, state, plan, step, service):
        self._charge_tool(task_id, state)
        self._charge_model(task_id, state)
        state["used_coding_agent"] = True
        self._persist(task_id, state)
        root = self._workspace_root(state)
        prompt = (
            "Propose the minimal repository patch for the given step. Your turn is read-only. "
            "Return exactly a JSON object with a single patch string using the *** Begin Patch format. "
            "Use separate Add File, Update File, or Delete File operations. "
            "Only the selected step is executable. Do not run project commands. "
            "If scope is insufficient, return an empty patch string and stop.\n\n"
            + json.dumps({"step": step.model_dump(mode="json"), "scope": plan.draft.scope,
                          "exclusions": plan.draft.exclusions, "user_decisions": plan.constraints,
                          "evidence": _investigation_context(state["evidence"][-8:]) if state["mode"] != "plan" else [],
                          "steering": state.get("guidance", []),
                          "root_cause": state.get("root_cause", "")}, ensure_ascii=False)
        )
        proposal = self.gateway._coding_agent.generate(
            prompt, execution_mode="read_only", gateway_task_id=child_id,
            turn_id=f"{task_id}:r{plan.revision}:{step.step_id}", dispatch_source="mode_workflow",
            coding_mode_proposal=True, coding_mode_files=list(step.changes),
            coding_mode_root=str(root), coding_mode_workspace_task_id=task_id,
            coding_mode_repository_id=plan.repository_id,
            execution_policy=state.get("execution_policy", {}),
            **self._batch_arguments(child_id, plan, state),
        )
        self._validate_approval(task_id, state, plan)
        if proposal.get("pending_operation"):
            self._suspend_batch(task_id, child_id, state, plan, proposal["pending_operation"])
        if state.get("batch_child") == child_id:
            state.pop("batch_child", None)
            self._persist(task_id, state)
        if proposal.get("status") not in {"completed", "success"}:
            raise RuntimeError("The configured coding engine did not return a completed patch proposal")
        revision_reason = self._steering(task_id, state, plan, step, service)
        if revision_reason:
            raise ScopeRevisionNeeded(revision_reason)
        return parse_proposal(proposal.get("output") or proposal.get("answer"))

    def _approved_patch(self, root, pending):
        from mana_agent.transactional_actions.runtime import default_action_gateway

        gateway = default_action_gateway(root)
        action = gateway.store.get_action(pending["action_id"])
        if not action or not action.preview or action.state.value != "awaiting_approval":
            raise PermissionError("The patch approval is unavailable or expired. Review the step before retrying.")
        patch = action.preview.diff
        if not isinstance(patch, str) or sha256(patch.encode()).hexdigest() != pending["patch_digest"]:
            raise PermissionError("The stored patch does not match the reviewed action")
        grant = gateway.approvals.find_valid(action)
        if not grant:
            raise PermissionError(f"Approve the exact patch action {action.action_id} in the inbox, then resume the plan.")
        return patch, grant.approval_id

    def _commit_patch(self, task_id, child_id, state, plan, step, service, expected_files, patch, approval_id):
        from mana_agent.gateway.lanes import validate_tool_permission
        from mana_agent.multi_agent.runtime.tool_call_gate import execute_host_operation

        root = self._workspace_root(state)
        child = self.lane.inspect_task(child_id)
        validate_tool_permission(self.lane.contracts[child.owning_lane], "apply_patch",
                                 task_capabilities=tuple(child.capabilities))
        if redact_secrets(patch) != patch:
            raise ValueError("Coding proposal contains protected material")
        operations = validate_step_patch(root, step, patch)
        self._charge_model(task_id, state)
        review = service.review(patch=patch, step=step.model_dump(mode="json"),
                                context={"scope": plan.draft.scope, "exclusions": plan.draft.exclusions,
                                         "constraints": plan.constraints, "goal": plan.goal,
                                         "steering": state.get("guidance", [])})
        if review.disposition == "revision_required":
            raise ScopeRevisionNeeded(str(redact_secrets(review.reason)))
        if review.disposition != "in_scope":
            raise PermissionError(f"Patch review {review.disposition}: {review.reason}")
        preview = safe_apply_patch(repo_root=root, patch=patch, allowed_prefixes=tuple(step.changes),
                                   enable_recovery=False, check_only=True)
        if not preview.get("ok") or preview.get("already_applied"):
            raise RuntimeError(preview.get("error") or "Patch does not produce the reviewed change")

        @contextmanager
        def mutation_fence():
            with self.lane.execution_supervisor.store.locked():
                durable = self._validate_approval(task_id, state, plan)
                self._check_live(child_id)
                if any(
                    item.get("status") == "pending" and item.get("attempt_id") == durable.attempt_id
                    for item in durable.steering_inputs
                ):
                    raise PermissionError("A steering request arrived before the patch. Review it before writing.")
                validate_step_patch(root, step, patch)
                for relative, _ in operations:
                    if self._file_digest(root, relative) != expected_files[relative]:
                        raise PermissionError(f"{relative} changed after the patch was reviewed")
                yield

        self._charge_tool(task_id, state)
        state["in_flight_mutation"] = {"step": step.step_id, "patch_digest": sha256(patch.encode()).hexdigest()}
        self._persist(task_id, state)
        result = execute_host_operation(
            root=root, task_id=child_id, tool_name="apply_patch", arguments={"patch": patch},
            effect="mutation", call_id=f"{task_id}:r{plan.revision}:{step.step_id}:patch",
            implementation=lambda _: safe_apply_patch(repo_root=root, patch=patch,
                allowed_prefixes=tuple(step.changes), enable_recovery=False, mutation_fence=mutation_fence,
                action_approval_id=approval_id),
        )
        if not result.get("ok"):
            if result.get("permission_required"):
                state["pending_patch_approval"] = {"action_id": result["action_id"], "step": step.step_id,
                    "revision": plan.digest, "patch_digest": sha256(patch.encode()).hexdigest()}
                state.pop("in_flight_mutation", None)
                self._persist(task_id, state)
            raise RuntimeError(str(result.get("error") or "The checked patch could not be applied"))
        for relative, _ in operations:
            expected_files[relative] = self._file_digest(root, relative)
        state["files_changed"] = sorted(set(state["files_changed"]).union(relative for relative, _ in operations))
        state["in_flight_mutation"] = {"step": step.step_id, "files": {path: expected_files[path] for path, _ in operations},
                                       "action_id": result.get("action_id", "")}
        state.pop("pending_patch_approval", None)
        self._persist(task_id, state)

    def _batch_arguments(self, child_id, plan, state):
        if not TaskExecutionPolicy.model_validate(state["execution_policy"]).required:
            return {}
        manager = self.gateway._autonomous_controller.batch_manager
        engine = getattr(self.gateway._coding_agent, "_internal_engine", None)
        if manager is None or engine is None:
            raise RuntimeError("The requested Batch coding engine is unavailable")
        lane = self.lane.inspect_task(child_id)
        return {
            "batch_resume": {"mode_task_id": plan.plan_id, "session_id": plan.session_id,
                             "turn_id": f"mode-{plan.plan_id}-r{plan.revision}", "digest": plan.digest},
            "batch_runtime": {
                "execution_supervisor": self.lane.execution_supervisor, "taskboard": self.board,
                "batch_manager": manager, "batch_autonomous_config": self.gateway._autonomous_controller.config,
                "batch_authority": {"attempt_id": lane.supervisor_attempt_id, "lease_token": lane.supervisor_lease_token},
                "batch_authorize": lambda name: self._authorize_batch_read(child_id, name),
                "model": engine.model, "provider": engine.provider,
            },
        }

    def _authorize_batch_read(self, child_id, name):
        from mana_agent.gateway.lanes import validate_tool_permission
        from mana_agent.tools.contracts import coding_tool_contracts

        self._check_live(child_id)
        contract = next((tool for tool in coding_tool_contracts() if tool.name.replace(".", "_") == name), None)
        if contract is None or contract.effect != "read_only" or not contract.repository_operation:
            raise PermissionError("Batch patch proposals can only inspect the repository")
        return validate_tool_permission(self.lane.contracts[LaneId.CODING], name)

    def _suspend_batch(self, task_id, child_id, state, plan, operation):
        from mana_agent.batch.operations import BatchOperationStore
        from mana_agent.batch.watcher import ensure_batch_watcher

        if not TaskExecutionPolicy.model_validate(state["execution_policy"]).required:
            raise PermissionError("A realtime plan step did not authorize Batch execution")
        child = self._check_live(child_id)
        if operation["task_id"] != child_id or not child.checkpoint_id:
            raise PermissionError("Batch proposal has no checkpoint bound to this step")
        manager = self.gateway._autonomous_controller.batch_manager
        operations = BatchOperationStore(manager.store)
        run = operations.load_run(child_id)
        if not run or run.get("resume", {}).get("digest") != plan.digest:
            raise PermissionError("Batch proposal belongs to a different plan revision")
        job = manager.store.get_job(operation["batch_id"])
        expiry = (datetime.fromtimestamp(job.expires_at, timezone.utc) if job and job.expires_at
                  else datetime.now(timezone.utc) + timedelta(hours=24))
        state["batch_child"] = child_id
        self._persist(task_id, state)
        self.lane.checkpoint(task_id, boundary="mode.batch_wait", resume_payload={
            "mode_task_id": task_id, "digest": plan.digest, "batch_child": child_id,
        }, completed_steps=list(state["done_steps"]))
        self.lane.suspend_for_batch(task_id, operation_id=operation["operation_id"], expires_at=expiry)
        self.lane.suspend_for_batch(child_id, operation_id=operation["operation_id"], expires_at=expiry)
        ensure_batch_watcher(self.gateway.root, self.gateway.background_processes)
        raise ModeBatchWaiting(f"Plan revision {plan.revision} is waiting for the coding proposal for step {state['step_id']}.")

    def _suspend_batch_integration(self, task_id, state, operation):
        from mana_agent.batch.operations import BatchOperationStore
        from mana_agent.batch.watcher import ensure_batch_watcher

        if (not TaskExecutionPolicy.model_validate(state["execution_policy"]).required
                or operation["task_id"] != task_id):
            raise PermissionError("The integration decision does not belong to this Batch task")
        plan = PlanRevision.model_validate(state["plan"])
        manager = self.gateway._autonomous_controller.batch_manager
        operations = BatchOperationStore(manager.store)
        run = operations.load_run(task_id) or {
            "task_id": task_id, "root": str(self.gateway.root), "model": state["model"],
            "step": 0, "model_turns": 0, "accounted_operations": [],
            "limits": self.gateway._autonomous_controller.config.model_dump(mode="json"),
        }
        run.update(status="waiting", gateway_pending_operation=operation,
                   resume={"mode_task_id": task_id, "session_id": plan.session_id, "digest": plan.digest})
        operations.save_run(task_id, run)
        job = manager.store.get_job(operation["batch_id"])
        expiry = (datetime.fromtimestamp(job.expires_at, timezone.utc) if job and job.expires_at
                  else datetime.now(timezone.utc) + timedelta(hours=24))
        state["batch_child"] = task_id
        self._persist(task_id, state)
        self.lane.checkpoint(task_id, boundary="mode.integration_batch_wait", resume_payload={
            "mode_task_id": task_id, "digest": plan.digest, "batch_child": task_id,
        }, completed_steps=list(state["done_steps"]))
        self.lane.suspend_for_batch(task_id, operation_id=operation["operation_id"], expires_at=expiry)
        ensure_batch_watcher(self.gateway.root, self.gateway.background_processes)
        raise ModeBatchWaiting("The workflow is waiting for its production integration decision.")

    def resume_batch(self, task_id, child_id, operation_id, run):
        supervisor = self.lane.execution_supervisor
        state = self.board.get_task(task_id).mode_workflow
        plan = PlanRevision.model_validate(state["plan"])
        operation = run.get("gateway_pending_operation") or run.get("pending_operation") or {}
        if (state.get("batch_child") != child_id or operation.get("operation_id") != operation_id
                or run["resume"].get("digest") != plan.digest):
            raise PermissionError("Batch wake does not match the approved plan step")
        manager = self.gateway._autonomous_controller.batch_manager
        job = manager.store.get_job(operation["batch_id"])
        completed_at = datetime.fromtimestamp(job.completed_at, timezone.utc) if job and job.completed_at else None
        self.lane.ensure_recoverable_execution(task_id)
        if child_id != task_id:
            self.lane.ensure_recoverable_execution(child_id)
        with supervisor.store.locked():
            parent = supervisor.get_task(task_id)
            if parent.state != ExecutionState.WAITING or parent.wake_up_reference != operation_id:
                return
            supervisor.resume_from_batch(task_id, operation_id=operation_id, provider_completed_at=completed_at)
            if child_id != task_id:
                supervisor.resume_from_batch(child_id, operation_id=operation_id, provider_completed_at=completed_at)
        self.gateway._bind_runtime_session(plan.session_id)
        lane = self.lane.inspect_task(task_id)
        lane.supervisor_attempt_id = lane.supervisor_lease_token = ""
        self.lane.start(LaneReservation(lane))
        result = self._execute(task_id, state, self._service(state=state))
        if child_id == task_id and result.status != "input_required":
            from mana_agent.batch.operations import BatchOperationStore
            operations = BatchOperationStore(manager.store)
            current = operations.load_run(task_id)
            current.pop("gateway_pending_operation", None)
            current["status"] = "completed" if result.status == "success" else "failed"
            operations.save_run(task_id, current)
        self._publish_result(task_id, result)

    def _verify(self, task_id: str, command: VerificationCommand, state: dict):
        self._charge_tool(task_id, state)
        operation_id = f"{task_id}:verify:{len(state.get('verification', [])) + 1}"
        command_detail = f"{command.argv!r} in {command.cwd}"
        emit_activity("coding.progress", task_id=task_id, title="Running verification command",
                      summary=command_detail, status="running", operation_id=operation_id)
        try:
            outcome = asyncio.run(run_isolated_verification(self._workspace_root(state), command, task_id=task_id))
        except VerificationUnavailable as exc:
            emit_activity("coding.progress", task_id=task_id, title="Verification pending",
                          summary=str(exc), detail=f"{exc}\n{command_detail}",
                          status="unverified", operation_id=operation_id,
                          metadata={"verification_status": "not_run"})
            raise
        state.setdefault("verification", []).append(outcome)
        self._persist(task_id, state)
        emit_activity("coding.progress", task_id=task_id, title="Verification command",
                      summary=f"{command.argv!r} · {outcome['exit_code']}",
                      detail=f"{command_detail}\n{outcome['output']}",
                      status="success" if outcome["passed"] else "failed",
                      operation_id=operation_id)
        if not outcome["passed"]:
            raise VerificationFailed(command, outcome)

    def _steering(self, task_id, state, plan, step, service):
        # The supervisor owns admission and delivery. The mode workflow decides
        # whether guidance can be applied to the existing approved revision.
        from mana_agent.execution.steering import SteeringCoordinator

        execution = self.lane.inspect_task(task_id)
        task = self._check_live(task_id)
        coordinator = SteeringCoordinator(self.lane.execution_supervisor)
        fence = {"attempt_id": task.attempt_id, "generation": task.attempt_generation,
                 "lease_token": execution.supervisor_lease_token}
        items = coordinator.pending_for_attempt(task_id, **fence)
        applied = []
        for item in items:
            text = item["text"]
            explicit_change = item["kind"] in {"scope_change", "replan"} or item.get("requires_approval")
            review = None if explicit_change else service.review(patch="", step=step.model_dump(mode="json"),
                                    context={"steering": text, "scope": plan.draft.scope,
                                             "exclusions": plan.draft.exclusions})
            if explicit_change or review.disposition == "revision_required":
                if state["mode"] != "plan":
                    raise PermissionError("Debug scope changed; submit the revised debugging request")
                coordinator.supersede_steering(task_id, **fence, request_ids=[item["request_id"]])
                return text
            if review.disposition != "in_scope":
                raise PermissionError(review.reason)
            state.setdefault("guidance", []).append(str(redact_secrets(text)))
            applied.append(item["request_id"])
        if applied:
            self._persist(task_id, state)
            coordinator.apply_steering(task_id, **fence, request_ids=applied)
        return ""

    def _complete(self, task_id, state, message):
        message = str(redact_secrets(message))
        report = self._repository_report(state)
        root = self._workspace_root(state)
        finished = self.lane.finish(task_id, changed_files=[str(root / path) for path in state["files_changed"]],
                         verification_state={"goal_satisfied": True, "answer": message,
                                             "verification": state.get("verification", []),
                                             "repository_observations": report,
                                             "integration": state.get("integration", {})})
        if finished.state != LaneTaskState.COMPLETED:
            return self._block(task_id, state, finished.error or "Supervisor verification remains incomplete")
        if state.get("isolated_workspace"):
            from mana_agent.multi_agent.worktrees import WorkspaceManager, WorkspaceStatus

            manager = WorkspaceManager(self.gateway.root, repository_id=state["isolated_workspace"]["repository_id"])
            manager.transition(task_id, WorkspaceStatus.MERGE_CANDIDATE)
        state["result"] = message
        self._transition(task_id, state, Phase.COMPLETED)
        return self._outcome(task_id, state, message, report, completed=True)

    def _block(self, task_id, state, message):
        report = self._repository_report(state)
        sections = [_changed_files_summary(state["files_changed"]), message]
        if state["files_changed"] and state.get("isolated_workspace"):
            sections.insert(1, f"Worktree {_code_span(state['isolated_workspace']['path'])}.")
        for detail in (report["gaps"], report["stop"], report["pending_question"]):
            if detail and not any(detail in section for section in sections):
                sections.append(detail)
        message = str(redact_secrets("\n\n".join(filter(None, sections))))
        state["result"] = message
        durable = self.lane.execution_supervisor.get_task(task_id)
        if durable.cancellation_status is None and durable.state not in {ExecutionState.COMPLETED, ExecutionState.CANCELLED}:
            execution = self.lane.inspect_task(task_id)
            if execution.state in {LaneTaskState.RUNNING, LaneTaskState.QUEUED, LaneTaskState.WAITING}:
                self.lane.mark_blocked(task_id, reason=message)
            self._transition(task_id, state, Phase.BLOCKED)
        return self._outcome(task_id, state, message, report, completed=False)

    @staticmethod
    def _outcome(task_id: str, state: dict, message: str, report: dict, *, completed: bool) -> CommandResult:
        data = {"task_id": task_id, "phase": state["phase"], "mode": state["mode"],
                "mode_reason": state.get("mode_reason", ""), "changed_files": state["files_changed"],
                "goal_satisfied": completed, "repository_observations": report,
                "used_coding_agent": state.get("used_coding_agent", False),
                "verification": state.get("verification", [])}
        if completed:
            data["integration"] = state.get("integration", {})
        return CommandResult(status="success" if completed else "error", message=message, data=data)

    def _repository_report(self, state: dict) -> dict:
        inspection = state.get("inspection", {})
        current = inspection.get("boundary") == list(self._investigation_boundary(state))
        if not current:
            inspection = {}
        stop = inspection.get("stop", "")
        if not stop and inspection and (self._remaining_tools(state) <= 0 or self._remaining_decisions(state) <= 0):
            stop = "The shared workflow budget was exhausted."
        native_report = state.get("investigation", {}).get("report", {})
        plan_findings = state.get("plan", {}).get("draft", {}).get("findings")
        findings = state.get("root_cause") or plan_findings or native_report.get("findings", "")
        return {"evidence_ids": list(investigation_scope(state["evidence"]).evidence_levels)[-12:],
                "findings": findings, "gaps": state.get("uncertainty") or native_report.get("gaps", ""),
                "workflow_model_decisions": state.get("decision_count", 0),
                "workflow_tool_operations": state.get("tool_operations", len(state["evidence"])),
                "stop": stop, "pending_question": inspection.get("need", "") if inspection.get("stop") else ""}


def parse_proposal(raw: Any) -> str:
    if not isinstance(raw, str) or len(raw) > 250_000:
        raise ValueError("Coding proposal must be a bounded JSON object")
    value = json.loads(raw.strip())
    if not isinstance(value, dict) or set(value) != {"patch"} or not isinstance(value["patch"], str) or not value["patch"]:
        raise ValueError("Coding proposal did not include a patch for this step")
    return value["patch"]


def validate_step_patch(root: Path, step: PlanStep, patch: str) -> list[tuple[str, str]]:
    operations = extract_patch_operations(patch)
    seen = set()
    for path, operation in operations:
        if path in seen or operation not in step.changes.get(path, ()):
            raise PermissionError(f"Patch operation {operation} on {path} is outside step {step.step_id}")
        seen.add(path)
        ModeWorkflow._file_digest(root, path)
    return operations


async def run_isolated_verification(root: Path, command: VerificationCommand, *, task_id: str) -> dict:
    from mana_agent.execution.models import ExecutionRequest, NetworkMode, NetworkPolicy, SandboxSpec
    from mana_agent.execution.providers.local_docker import LocalDockerProvider

    if not shutil.which("docker"):
        raise VerificationUnavailable("The Docker CLI is unavailable. Install Docker and start its daemon.")
    cwd = root / command.cwd
    if cwd.resolve() != cwd or not cwd.is_dir():
        raise ValueError("Verification cwd must be a real directory inside the checkout")
    # Copy only versioned source and non-ignored candidate files. Do not mount
    # the task checkout, its .git metadata, host credentials, or runtime state.
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, check=True, stdout=subprocess.PIPE,
    )
    provider = LocalDockerProvider()
    image = command.image or provider.default_image
    image_lookup = subprocess.run([provider.docker_binary, "image", "inspect", "--format", "{{.Id}}", image],
                                  capture_output=True, text=True, timeout=30, check=False)
    if image_lookup.returncode != 0 or not image_lookup.stdout.strip():
        raise VerificationUnavailable(
            f"Docker could not access the local image {_code_span(image)}. Make the image available in Docker."
        )
    with TemporaryDirectory(prefix="mana-mode-verify-") as temporary:
        snapshot = Path(temporary).resolve()
        (snapshot / command.cwd).mkdir(parents=True, exist_ok=True)
        for name in dict.fromkeys(listed.stdout.decode().split("\0")):
            try:
                path = root / repository_relative_path(name)
            except ValueError:
                continue
            if path.is_file() and path.resolve() == path:
                destination = snapshot / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        handle = await provider.provision(SandboxSpec(
            repository_source=snapshot, task_id=task_id, base_image=image_lookup.stdout.strip(),
            network=NetworkPolicy(mode=NetworkMode.DENY_ALL), read_only_root=True,
        ))
        try:
            handle = await provider.start(handle)
            result = await provider.execute(handle, ExecutionRequest(argv=list(command.argv), cwd=command.cwd))
        finally:
            await provider.cleanup(handle)
    output = str(redact_secrets(result.stdout + "\n" + result.stderr))
    passed = (not result.timed_out and not result.cancelled and
              (result.exit_code == 0 if command.expected == "pass" else
               result.exit_code not in {0, 126, 127} and command.output_marker in output))
    return {"argv": list(command.argv), "cwd": command.cwd, "exit_code": result.exit_code,
            "passed": passed, "output": output[-2000:]}
