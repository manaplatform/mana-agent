"""Gateway operations for supervised AUTO, DEBUG, and PLAN tasks."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
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
import sqlite3
import shlex
import subprocess
from tempfile import TemporaryDirectory
import time
from typing import Any, TYPE_CHECKING

from pydantic_core import PydanticCustomError

from mana_agent.agent.mode import CodingWorkflowDecision, ModePhase as Phase, PlanRevision, PlanStep, VerificationCommand, repository_relative_path
from mana_agent.agent.plan_service import (
    InspectDecision, InvestigateDecision, MAX_INSPECTION_BATCH, ModeDecisionContext, PlanService, SavedInvestigation,
    validate_investigation_recovery, validate_observation_references,
)
from mana_agent.agent.repository_investigation import (
    compact_observation, investigation_scope, observation_fingerprint, read_tool_schemas,
)
from mana_agent.batch.policy import BatchPending, TaskExecutionPolicy
from mana_agent.chat.activity import safe_text
from mana_agent.chat_commands.models import CommandResult
from mana_agent.coding.models import ISOLATED_CHECK_TOOL
from mana_agent.execution_supervisor.errors import StaleLeaseError
from mana_agent.execution_supervisor.models import ExecutionState, RecoveryAction, TaskRecord
from mana_agent.gateway.lane_coordinator import LaneReservation
from mana_agent.gateway.lanes import LaneId, LaneTaskState
from mana_agent.human_inbox.identity import local_reviewer_actor
from mana_agent.human_inbox.models import (
    ClarificationField, ExpectedResponseType, InboxRequest, InboxRequestType, InboxStatus,
    ReviewerAssignment, ReviewerType, ResponseOperation, ResponseSubmission, canonical_digest,
)
from mana_agent.human_inbox.repository import InboxConcurrentUpdateError
from mana_agent.multi_agent.core.types import TaskBoardItem, TaskStatus
from mana_agent.services.activity_context import (
    ActivityContext, activity_protected_text, activity_scope, current_activity_context, emit_activity,
)
from mana_agent.tools.apply_patch import PatchFormatError, extract_patch_operations, safe_apply_patch
from mana_agent.utils.redaction import redact_secrets
from mana_agent.repository_evidence.models import EvidenceRef, MissingEvidenceRequest, RuntimeWiringEvidence

if TYPE_CHECKING:
    from mana_agent.repository_evidence.service import RepositoryEvidenceBoard

_OWNER = "autonomous_loop"
_PATCH_HANDOFF_MODEL_TURNS = 3  # Draft, proposal, and review for the first edit.
_PATCH_HANDOFF_TOOL_CALLS = 3  # Proposal, apply, and focused verification.
_DEADLINE_MESSAGE = ("This task reached its execution deadline. Review any existing changes and tool outcomes "
                     "before submitting the request again as a new task.")
logger = logging.getLogger(__name__)


class ScopeRevisionNeeded(RuntimeError):
    pass


class ModeBatchWaiting(RuntimeError):
    pass


class ModeEvidenceWaiting(RuntimeError):
    pass


class VerificationUnavailable(RuntimeError):
    """A verification prerequisite prevented the approved command from running."""

    def __init__(self, message: str, *, command: VerificationCommand | None = None):
        super().__init__(message)
        self.command = command


class VerificationFailed(RuntimeError):
    def __init__(self, command: VerificationCommand, outcome: dict):
        super().__init__(f"Verification failed for {command.argv!r}: {outcome['output']}")
        self.command = command
        self.outcome = outcome


_TERMINAL = {Phase.COMPLETED, Phase.CANCELLED}
_TRANSITIONS = {
    Phase.ACTIVE: {Phase.CLARIFICATION_REQUIRED, Phase.DRAFTING_PLAN, Phase.EXECUTING, Phase.COMPLETED},
    Phase.CLARIFICATION_REQUIRED: {Phase.ACTIVE},
    Phase.DRAFTING_PLAN: {Phase.AWAITING_APPROVAL, Phase.APPROVED, Phase.COMPLETED},
    Phase.AWAITING_APPROVAL: {Phase.APPROVED, Phase.ACTIVE},
    Phase.APPROVED: {Phase.EXECUTING, Phase.ACTIVE},
    Phase.EXECUTING: {Phase.EXECUTING, Phase.VERIFYING, Phase.REVIEWING, Phase.ACTIVE, Phase.COMPLETED,
                      Phase.CLARIFICATION_REQUIRED},
    Phase.VERIFYING: {Phase.EXECUTING, Phase.REVIEWING, Phase.COMPLETED, Phase.ACTIVE},
    Phase.REVIEWING: {Phase.EXECUTING, Phase.COMPLETED},
    Phase.BLOCKED: {Phase.ACTIVE, Phase.EXECUTING},
}


def _investigation_context(evidence: list[dict]) -> list[dict]:
    return [compact_observation(item) for item in evidence]


def _refresh_evidence_rows(rows: Iterable[dict], knowledge: RepositoryEvidenceBoard) -> None:
    from mana_agent.repository_evidence.models import EvidenceRef

    for row in rows:
        if not row.get("board_ref"):
            continue
        try:
            record = knowledge.get(EvidenceRef.model_validate(row["board_ref"]))
        except KeyError:
            record = None
        if record is None or record.freshness != "current" or record.validation != "supported":
            # Keep the ID reserved so a later observation cannot inherit it.
            row.pop("board_ref")
            row["stale"] = True
            row["result"] = {"ok": False, "error": "Repository evidence is no longer current and supported."}


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
        if state.get("schema_version", 1) < 2:
            for row in state.get("evidence", []):
                if not row.get("board_ref") and isinstance(row.get("result"), dict):
                    row["result"] = {**row["result"], "ok": False, "legacy_unvalidated": True}
        if state.get("phase") in {"understanding", "investigating"}:
            state["phase"] = Phase.ACTIVE.value
        elif state.get("phase") == Phase.INTEGRATING.value:
            state["phase"] = Phase.EXECUTING.value
        state["schema_version"] = 2
        self.board.update_task(task_id, mode_workflow=state)

    def _transition(self, task_id: str, state: dict, target: Phase, summary: str = "") -> None:
        if state.get("phase") in {"understanding", "investigating"}:
            state["phase"] = Phase.ACTIVE.value
        elif state.get("phase") == Phase.INTEGRATING.value:
            state["phase"] = Phase.EXECUTING.value
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
                      metadata={"task_status": task.status.value, "mode": state["mode"],
                                "tool_budget_used": state.get("tool_operations", len(state["evidence"])),
                                "tool_budget_remaining": max(0, self._remaining_tools(state)),
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
                workflow_coordinator=True,
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
                self._transition(task_id, existing, Phase.ACTIVE)
            return self._drive(task_id, existing, llm=llm)
        state = {
            "schema_version": 1, "mode": mode, "phase": Phase.ACTIVE.value,
            "goal": goal, "revision": 1, "constraints": {}, "evidence": [],
            "tool_operations": 0, "decision_count": 0,
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
            configured = self.gateway._coding_agent
            if (not policy.required and getattr(configured, "active_engine_mode", "") == "internal"
                    and getattr(configured, "supports_adaptive_repository_loop", False) is True):
                state["adaptive_coding"] = True
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
                self._transition(task_id, state, Phase.ACTIVE, state["mode_reason"])
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
                    original_mode = (state.get("coding_decision") or {}).get("mode")
                    if state.get("mode_transitions") and original_mode in {"auto", "debug"} and state["mode"] != original_mode:
                        state.update(strategy=state["mode"], mode=original_mode)
                        self._persist(task_id, state)
                    handoff = state.get("investigation", {})
                    if handoff.get("status") == "running" and handoff.get("attempt_id") != activity.attempt_id:
                        # Preserve prior charges and reuse only saved reads.
                        handoff["status"] = "interrupted"
                        self._persist(task_id, state)
                if service is None and not state.get("adaptive_coding"):
                    service = self._service(state=state, llm=llm)
                return self.gateway._autonomous_controller.drive_repository_workflow(
                    advance=lambda: self._advance(task_id, state, service),
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

    def _advance(self, task_id, state, service) -> CommandResult | None:
        if state.get("adaptive_coding"):
            return self._execute_adaptive(task_id, state)
        if (state["mode"] != "plan" and state.get("plan")
                and (not state.get("pending_repository_evidence") or state.get("evidence_resume"))):
            return self._execute(task_id, state, service, in_controller=True)
        return self._advance_decision(task_id, state, service)

    def _execute_adaptive(self, task_id, state):
        from mana_agent.coding.adaptive import AdaptiveCodingResult

        configured = self.gateway._coding_agent
        if (getattr(configured, "active_engine_mode", "") != "internal"
                or getattr(configured, "supports_adaptive_repository_loop", False) is not True):
            raise PermissionError("Resume with the originally selected internal coding backend.")
        parent = self.board.get_task(task_id)
        root = self._workspace_root(state).resolve()
        write = state["repair"] and not state["no_code"]
        identity = "adaptive-coding"
        child = next((row for row in self.board.list_tasks(parent_task_id=task_id)
                      if row.decomposition_local_id == identity), None)
        if child is None:
            child = self.board.create_child_task(task_id, title=f"Coding Agent · {state['mode'].upper()}",
                user_request=state["goal"], owner_agent_id="coding_agent", decomposition_local_id=identity)
        state["used_coding_agent"] = True
        self._transition(task_id, state, Phase.EXECUTING, f"Coding Agent · {state['mode'].upper()}")
        supervisor = self.lane.execution_supervisor
        saved = supervisor.store.get_task_or_none(child.task_id)
        if saved and saved.state == ExecutionState.COMPLETED:
            receipt = supervisor.get_verified_execution_result(child.task_id)
            if not receipt.is_verified:
                raise PermissionError("The saved coding result has no verified supervisor receipt.")
            outcome = AdaptiveCodingResult.model_validate(state["adaptive_outcome"])
        else:
            with self.lane.workflow_step_capacity(task_id, child.task_id):
                self._start_adaptive_child(parent, child, saved, state, root, write)
                try:
                    with self._activity(child.task_id, state, read_only=not write):
                        result = self._run_adaptive_model(task_id, child.task_id, state, root, write, configured)
                    if result.get("adaptive_result") is None:
                        detail = "\n".join(result.get("errors") or []) or result.get("answer") or "No coding result was returned."
                        if result.get("human_input", {}).get("inbox_item_id"):
                            detail = f"{detail} Approval item {result['human_input']['inbox_item_id']}."
                        raise RuntimeError(safe_text(detail, limit=1500))
                    outcome = AdaptiveCodingResult.model_validate(result["adaptive_result"])
                    if outcome.disposition == "blocked":
                        raise RuntimeError(outcome.summary)
                    if outcome.disposition == "clarify":
                        self.lane.mark_blocked(child.task_id, reason=outcome.summary)
                        count = state["questions_asked"] + len(outcome.questions)
                        if state["question_limit"] is not None and count > state["question_limit"]:
                            raise RuntimeError("The clarification limit was reached.")
                        state["questions_asked"] = count
                        self._transition(task_id, state, Phase.CLARIFICATION_REQUIRED)
                        item = self._wait(task_id, state, questions=[
                            redact_secrets(question.model_dump(mode="json")) for question in outcome.questions
                        ])
                        return self._describe_item(task_id, state, item)
                    self._adaptive_checks(state, root, write=write)
                    state["adaptive_outcome"] = redact_secrets(outcome.model_dump(mode="json"))
                    self._persist(task_id, state)
                    finished = self.lane.finish(child.task_id,
                        changed_files=[str(root / path) for path in state["files_changed"]],
                        reconciled_file_versions=state.get("adaptive_file_versions", {}),
                        verification_state={"goal_satisfied": True, "verification": state.get("verification", [])})
                    if finished.state != LaneTaskState.COMPLETED:
                        raise RuntimeError(finished.error or "Supervisor verification is incomplete.")
                except Exception as exc:
                    if self.lane.inspect_task(child.task_id).state == LaneTaskState.RUNNING:
                        durable_child = supervisor.get_task(child.task_id)
                        if durable_child.wall_clock_deadline_exceeded(supervisor.clock()) and not durable_child.cancellation_status:
                            self.lane.finish(child.task_id, state=LaneTaskState.TIMED_OUT, error=_DEADLINE_MESSAGE,
                                             changed_files=[str(root / path) for path in state["files_changed"]])
                        elif not durable_child.cancellation_status:
                            self.lane.mark_blocked(child.task_id, reason=safe_text(str(exc), limit=1500))
                    raise
        self._adaptive_checks(state, root, write=write)
        if write and state.get("runtime_capability_change"):
            if outcome.wiring is None:
                raise RuntimeError("The coding result did not identify runtime reachability evidence.")
            state["runtime_wiring"] = outcome.wiring.model_dump(mode="json")
            self._require_runtime_evidence(task_id, state, None)
        return self._complete(task_id, state, outcome.summary)

    def _start_adaptive_child(self, parent, child, saved, state, root, write):
        from mana_agent.context_cost.estimator import estimate_value_tokens
        from mana_agent.execution_supervisor.models import ActionRequestState, RecoveryDecision, RetryCategory

        if saved:
            context = state.get("adaptive_continuation", {})
            completed = {row["call_id"] for row in context.get("executed_operations", [])}
            actions = self.lane.execution_supervisor.store.actions_for_task(child.task_id)
            if any(action.verification_state.get("operation_attempt_id") not in completed or action.request_state in {
                ActionRequestState.PREPARED, ActionRequestState.STARTED, ActionRequestState.OUTCOME_UNKNOWN,
            } for action in actions):
                raise PermissionError("Reconcile the coding action receipts before resuming this checkout.")
            self._adaptive_checks(state, root, write=False)
            recovery = RecoveryDecision(
                decision_id=f"adaptive-retry:{child.task_id}:{time.time_ns()}", task_id=child.task_id,
                action=RecoveryAction.RESUME_CHECKPOINT if saved.checkpoint_id else RecoveryAction.RETRY,
                resume_checkpoint_id=saved.checkpoint_id or "", retry_category=RetryCategory.MODEL,
                reason="Resume coding with reconciled operation receipts", same_task_retry_authorized=True,
                safe_to_continue=True,
            )
            resume = self.lane.resume_checkpoint if saved.checkpoint_id else self.lane.retry_task
            reservation = resume(child.task_id, session_id=parent.session_id, decision=recovery)
        else:
            capabilities = ["repository_read", "git_read", "test_execution"]
            if write:
                capabilities.extend(("repository_write", "shell_read", "shell_write"))
            parent_budget = self.lane.inspect_task(parent.task_id).budget
            reservation = self.lane.reserve(
                normalized_intent=f"{parent.task_id}:{state['mode']}:adaptive", lane_id=LaneId.CODING,
                session_id=parent.session_id, workspace_id=parent.workspace_id, repository_id=parent.primary_repository_id,
                parent_task_id=parent.task_id, taskboard_task_id=child.task_id, execution_workspace=root,
                capabilities=capabilities,
                requested_input_tokens=max(estimate_value_tokens(state["goal"]), parent_budget.reserved_input_tokens
                                           - parent_budget.consumed_input_tokens),
                requested_output_tokens=max(0, parent_budget.reserved_output_tokens - parent_budget.consumed_output_tokens),
            )
        self.lane.start(reservation)

    def _run_adaptive_model(self, task_id, child_id, state, root, write, configured):
        from mana_agent.gateway.lanes import TOOL_CAPABILITIES
        from mana_agent.repository_evidence.tasks import assignment_for
        from mana_agent.tools.contracts import ToolContract

        child = self.board.get_task(child_id)
        self._adaptive_checks(state, root, write=False)
        if not child.coding_assignment:
            assignment_for(self.board, child_id, backend="internal",
                           decision_id=state.get("routing_decision_id") or f"{task_id}:mode")

        def authorize(name):
            parent_attempt = self._check_live(task_id)
            self._check_live(child_id)
            if any(item.get("status") == "pending" and item.get("attempt_id") == parent_attempt.attempt_id
                   for item in parent_attempt.steering_inputs):
                raise PermissionError("Coding received new steering. Reconcile the task request before continuing.")
            return self.lane.authorize_tool(child_id, name)

        def charge(operation):
            with current_activity_context().publication():
                self._check_live(task_id)
                self._check_live(child_id)
                operation(task_id, state)

        def save(context):
            activity = current_activity_context()
            with activity.publication():
                self._check_live(task_id)
                versions = self._adaptive_mutations(child_id)
                if any(self._file_digest(root, path) != digest for path, digest in versions.items()):
                    raise PermissionError("A changed file no longer matches its tool receipt.")
                state["files_changed"] = sorted(versions)
                state["adaptive_file_versions"] = versions
                state["adaptive_evidence_refs"] = self.board.get_task(child_id).evidence_refs[-12:]
                serialized = json.dumps(redact_secrets(context), ensure_ascii=False)
                clean = safe_text(serialized, limit=None, protected=activity.protected_text)
                if clean != json.dumps(context, ensure_ascii=False):
                    # Never replay a provider continuation whose containing history was redacted.
                    context = {key: context[key] for key in (
                        "changed_files", "commands_run", "verification_results", "executed_operations"
                    )}
                    context["executed_operations"] = [
                        {key: row[key] for key in ("tool", "call_id", "fingerprint") if key in row}
                        for row in context["executed_operations"]
                    ]
                    clean = safe_text(json.dumps(redact_secrets(context), ensure_ascii=False),
                                      limit=None, protected=activity.protected_text)
                state["adaptive_continuation"] = json.loads(clean)
                self._persist(task_id, state)
            self.lane.checkpoint(child_id, boundary="adaptive_coding_tools",
                resume_payload={"mode_task_id": task_id, "changed_files": state["files_changed"]})

        resume = state.get("adaptive_continuation", {})
        if resume:
            resume = self._refresh_adaptive_history(root, resume)

        def verify(**arguments):
            return self._verify_adaptive(task_id, child_id, root, state, VerificationCommand.model_validate(arguments))

        verify.input_model = VerificationCommand
        verify.tool_contract = ToolContract(name=ISOLATED_CHECK_TOOL,
            description="Run a focused verification argv in a network-disabled source snapshot. Returns the exact exit code and output.",
            input_schema=VerificationCommand.model_json_schema(), output_schema={}, error_format={},
            required_capability="test_execution", effect="read_only", repository_operation=True)
        caps = set(self.lane.inspect_task(child_id).capabilities)
        allowed = frozenset(name for name, required in TOOL_CAPABILITIES.items()
                            if required <= caps and name != "verify_project")
        worker = configured.for_assignment(root=root, task_id=child_id)
        try:
            return worker.generate(state["goal"], gateway_task_id=child_id, adaptive_mode=state["mode"],
                execution_mode=state["mode"] if write else "read_only",
                runtime_capability_change=state.get("runtime_capability_change", False),
                coding_requirements=state["coding_decision"]["verification_requirements"],
                relevant_context=json.dumps({"routing": state["coding_decision"], "user_answers": state["constraints"],
                    "conversation": state.get("conversation_context", []),
                    "revision_request": state.get("revision_request", "")}, ensure_ascii=False),
                coding_resume_context=resume,
                coding_continuation_id=f"{child_id}:{state['decision_count']}",
                execution_runtime={"taskboard": self.board, "execution_supervisor": self.lane.execution_supervisor,
                    "execution_registry": current_activity_context().execution_registry, "tool_authorize": authorize,
                    "allowed_tools": allowed, "registered_tools": {ISOLATED_CHECK_TOOL: verify},
                    "charge_model": lambda: charge(self._charge_model), "charge_tool": lambda: charge(self._charge_tool),
                    "save_native_continuation": save, "remaining_model_turns": self._remaining_decisions(state),
                    "validate_adaptive_completion": lambda: self._adaptive_checks(state, root, write=write),
                    "remaining_tool_calls": self._remaining_tools(state),
                    "max_no_progress_cycles": self.gateway._autonomous_controller.config.max_no_progress_cycles})
        finally:
            worker.close()

    def _refresh_adaptive_history(self, root, resume):
        messages = []
        stale = False
        versions = {}
        for message in resume.get("native_messages", []):
            if message.get("role") != "tool":
                messages.append(message)
                continue
            result = json.loads(message["content"])
            for row in result.get("files", [result]):
                source = row.get("source_evidence") or {}
                path, revision = source.get("path"), source.get("sha256")
                if path and revision and source.get("source_kind") == "file":
                    if path not in versions:
                        try:
                            versions[path] = self._file_digest(root, path)
                        except ValueError:
                            versions[path] = "unavailable"
                    if versions[path] != revision:
                        row.pop("content", None)
                        row.update(ok=False, stale=True, error="The file changed since this observation. Refresh only if needed.")
                        stale = True
            messages.append({**message, "content": json.dumps(result, ensure_ascii=False)})
        if stale:
            return {**resume, "native_messages": messages, "native_continuation": None, "native_seen_evidence": []}
        return resume

    def _verify_adaptive(self, task_id, child_id, root, state, command):
        from mana_agent.multi_agent.agents.tool_runner_agent import assert_tool_runner, current_operation

        assert_tool_runner(ISOLATED_CHECK_TOOL)
        operation = current_operation()
        versions = self._adaptive_mutations(child_id)
        if any(self._file_digest(root, path) != digest for path, digest in versions.items()):
            raise PermissionError("Verification requires reconciling a change outside its tool receipt.")
        outcome = asyncio.run(run_isolated_verification(root, command, task_id=child_id))
        activity = current_activity_context()
        with activity.publication():
            self._check_live(task_id)
            self._check_live(child_id)
            if any(self._file_digest(root, path) != digest for path, digest in versions.items()):
                raise PermissionError("Verification inputs changed while the check was running.")
            receipt = {**outcome, "argv": list(command.argv), "cwd": command.cwd, "expected": command.expected,
                "operation_id": operation.request.operation_id, "attempt_id": activity.attempt_id,
                "source_versions": versions, "image": command.image,
                "output": safe_text(outcome["output"], protected=activity.protected_text)}
            state.setdefault("verification", []).append(receipt)
            self._persist(task_id, state)
        return {"ok": receipt["passed"], **receipt}

    def _adaptive_mutations(self, child_id) -> dict[str, str]:
        from mana_agent.repository_evidence.tool_queue import ToolQueueStore

        before, after = {}, {}
        for receipt in ToolQueueStore(self.board.store.db).operation_receipts(child_id):
            for mutation in receipt.get("mutations", []):
                if mutation["status"] == "unknown":
                    raise PermissionError("Reconcile an uncertain mutation before completing or verifying the task.")
                path = mutation["path"]
                old_path = mutation.get("old_path")
                if old_path:
                    before.setdefault(old_path, mutation["before_sha256"])
                    after[old_path] = "missing"
                before.setdefault(path, "missing" if old_path else mutation["before_sha256"] or "missing")
                after[path] = mutation["after_sha256"] or "missing"
        return {path: digest for path, digest in after.items() if before[path] != digest}

    def _adaptive_checks(self, state, root, *, write):
        versions = state.get("adaptive_file_versions", {})
        if any(self._file_digest(root, path) != digest for path, digest in versions.items()):
            raise PermissionError("Files from the saved coding assignment changed. Reconcile the checkout before continuing.")
        checks = {}
        for row in state.get("verification", []):
            if row["expected"] == "pass":
                checks[(tuple(row["argv"]), row["cwd"], row["image"])] = (
                    row["source_versions"] == versions and row["passed"] and row["exit_code"] == 0
                )
        if write and (not checks or not all(checks.values())):
            missing = [f"{shlex.join(argv)} in {cwd}" for (argv, cwd, _), passed in checks.items() if not passed]
            raise ValueError("Completion requires passing focused verification against the current changes."
                             + (f" Re-run {', '.join(missing)}." if missing else ""))

    def _saved_investigation(self, task_id: str, child_id: str) -> tuple[TaskBoardItem, TaskRecord, SavedInvestigation] | None:
        child = self.board.store.repository.get_task_or_none(child_id) if child_id else None
        if child is None or child.parent_task_id != task_id or child.task_kind not in {"investigation", "understanding"}:
            return None
        execution = self.lane.execution_supervisor.store.get_task_or_none(child.task_id)
        if execution is None:
            return None
        return child, execution, SavedInvestigation(
            task_id=child.task_id, objective=child.user_request, paths=tuple(child.investigation_scope),
            task_kind=child.task_kind, initial_repository_scope=child.mode_workflow.get("initial_repository_scope", ""),
            scope=child.mode_workflow.get("investigation_context"), checkpoint_id=execution.checkpoint_id,
        )

    def _advance_decision(self, task_id, state, service) -> CommandResult | None:
        with current_activity_context().publication():
            for field in ("model", "provider"):
                selected = getattr(service.llm, field, None)
                if not state.get(field) and isinstance(selected, str) and selected:
                    state[field] = selected
            self._charge_model(task_id, state)
        from mana_agent.repository_evidence.models import EvidenceQuery, EvidenceRef, EvidenceRequirement
        knowledge = self.board.evidence_board(root=self._workspace_root(state))
        root_task = self.board.get_task(task_id)
        knowledge_context = knowledge.query(EvidenceQuery.model_validate(state.get("evidence_query") or {"task_id": task_id}))
        evidence = state["evidence"]
        _refresh_evidence_rows(evidence, knowledge)
        known = {
            (row.get("board_ref") or {}).get("evidence_id", row["id"]): row for row in evidence
        }
        for record in knowledge_context["records"]:
            if record.get("detail_required"):
                continue
            evidence_id = record["evidence_id"]
            previous = known.get(evidence_id)
            if previous is not None and not previous.get("stale"):
                continue
            ref = {"evidence_id": evidence_id, "revision": record["revision"]}
            summary = {"id": evidence_id, "level": 1, "board_ref": ref, "fingerprint": canonical_digest(ref),
                       "result": {"ok": record["validation"] == "supported", "content": record["statement"],
                                  "classification": record["classification"], "sources": record["sources"]}}
            if previous is None:
                evidence.append(summary)
            else:
                previous.clear()
                previous.update(summary)
        scope = investigation_scope(evidence)
        handoff = state.get("investigation", {})
        saved_investigation = None
        recovery_target = None
        saved = self._saved_investigation(task_id, handoff.get("task_id", ""))
        if saved is not None:
            child, child_execution, recovery_target = saved
            saved_investigation = {
                **asdict(recovery_target),
                "status": child.status.value,
                "execution_state": child_execution.state.value,
                "evidence_refs": child.evidence_refs[-12:],
            }
        planning = state["mode"] == "plan"
        remaining_tools = self._remaining_tools(state)
        remaining_decisions = self._remaining_decisions(state)
        boundary = list(self._investigation_boundary(state))
        inspection = state.get("inspection", {})
        current_inspection = inspection.get("boundary") == boundary
        stop = inspection.get("stop", "") if current_inspection else ""
        reused_evidence_ids = inspection.get("reused_evidence_ids", []) if current_inspection else []
        discovery = state.get("evidence_discovery", {})
        discovery_stop = (
            discovery.get("boundary") == boundary
            and discovery.get("no_progress", 0) >= self.gateway._autonomous_controller.config.max_no_progress_cycles
        )
        pending = state.get("pending_repository_evidence")
        unavailable_check = state.get("verification_unavailable")
        defer_pre_patch_verification = bool(unavailable_check and not state.get("plan") and not pending)
        # Leave room for a draft, its first patch proposal and review, and an isolated check.
        # This is a minimum for one edit, while the selected plan still owns actual accounting.
        preparing_patch = not planning and state["repair"] and not state["no_code"] and not pending
        model_reserve = _PATCH_HANDOFF_MODEL_TURNS if preparing_patch else 1
        tool_reserve = _PATCH_HANDOFF_TOOL_CALLS if preparing_patch else 0
        inspection_tools = max(0, remaining_tools - tool_reserve)
        patch_budget_stop = preparing_patch and (remaining_decisions < model_reserve or not inspection_tools)
        can_inspect = inspection_tools > 0 and remaining_decisions > 0 and not stop and not patch_budget_stop
        can_investigate = (inspection_tools > 0 and remaining_decisions > model_reserve
                           and not stop and not discovery_stop)
        if defer_pre_patch_verification:
            can_inspect = can_investigate = False
        report = handoff.get("report") or {}
        cites = set(report.get("evidence_ids", []))
        visible_evidence = cites | set(reused_evidence_ids) | {row["id"] for row in evidence[-8:]}
        if defer_pre_patch_verification:
            visible_evidence.update(state.get("repair_cause_evidence", []))
        if not planning:
            # The shallow layout is useful after it ages out of the recent reads.
            layout = next((row for row in reversed(evidence)
                           if row["level"] == 1 and isinstance(row["result"], dict)
                           and isinstance(row["result"].get("directories"), list)), None)
            if layout is not None:
                visible_evidence.add(layout["id"])
        observations = [row for row in evidence if row["id"] in visible_evidence]
        gap = MissingEvidenceRequest.model_validate(pending) if pending else None

        def satisfies_requirement(refs: tuple[EvidenceRef, ...]) -> bool:
            return gap is not None and knowledge.satisfies(gap.requirement.model_copy(update={"evidence": refs}))

        def resume_context() -> tuple[str, str, bool]:
            if gap is None:
                return "", "", False
            child = self.board.get_task(gap.task_id)
            if child.parent_task_id != task_id:
                return "", "", False
            execution = self.lane.execution_supervisor.get_task(child.task_id)
            ready = bool(child.evidence_requirements) and all(
                knowledge.satisfies(EvidenceRequirement.model_validate(item)) for item in child.evidence_requirements
            )
            return child.task_id, execution.checkpoint_id, ready

        def lookup_observation(evidence_id: str) -> dict | None:
            return next((compact_observation(row) for row in evidence if row["id"] == evidence_id), None)

        strategy = state.get("strategy", state["mode"])
        allowed = {"clarify", "block"}
        if can_inspect:
            allowed.add("inspect")
        if can_investigate:
            allowed.add("investigate")
        if not discovery_stop and not patch_budget_stop and (not stop or pending):
            allowed.add("query_evidence")
        if pending or not discovery_stop:
            allowed.add("use_evidence")
        if pending:
            allowed.add("resume_implementation")
        elif strategy in {"auto", "plan"}:
            allowed.add("draft")
        elif strategy == "debug":
            allowed.add("diagnose")
        if not pending and (strategy == "debug" or (not planning and state.get("verification_failure"))):
            if state["repair"]:
                allowed.add("repair")
            if not preparing_patch or (remaining_decisions > model_reserve and inspection_tools):
                allowed.add("research")
        if state.get("coding_decision") and not pending:
            allowed.add("switch_workflow")
        if defer_pre_patch_verification:
            allowed.intersection_update({"draft", "repair", "clarify", "block"})

        refs = [row["board_ref"] for row in evidence if row.get("board_ref") and row["id"] in scope.evidence_levels]
        refs.extend({"evidence_id": record["evidence_id"], "revision": record["revision"]}
                    for record in knowledge_context["records"]
                    if record.get("freshness") == "current" and record.get("validation") == "supported")
        current_refs = tuple({ref["evidence_id"]: EvidenceRef.model_validate(ref) for ref in refs}.values())
        waiting_task_id, pending_checkpoint, resume_ready = resume_context()
        workflow_context = ModeDecisionContext(
            task_id=task_id, strategy=strategy, allowed_actions=frozenset(allowed), repair_authorized=state["repair"],
            evidence_discovery_stopped=discovery_stop,
            last_switch_evidence=frozenset(state.get("last_switch_evidence", [])), current_evidence_refs=current_refs,
            waiting_task_id=waiting_task_id, checkpoint_id=pending_checkpoint, resume_ready=resume_ready,
            requirement_id=gap.requirement.requirement_id if gap else "", satisfies_requirement=satisfies_requirement,
            repair_limits=state.get("repair_limits"), required_recheck=state.get("required_recheck"),
            required_verification=tuple(state.get("required_verification", [])), saved_investigation=recovery_target,
            runtime_wiring_required=bool(state.get("runtime_capability_change") and state["repair"] and not state["no_code"]),
            defer_pre_patch_verification=defer_pre_patch_verification,
        )
        inspection_stop = stop
        if defer_pre_patch_verification:
            inspection_stop = "Use saved evidence to reassess the unavailable pre-patch check."
        elif not inspection_stop and not planning and not can_inspect:
            inspection_stop = "Shared workflow budget reserved for patching and verification" if patch_budget_stop else "Shared workflow budget"
        decision = service.next_decision(strategy, {
            "task_id": task_id, "coding_workflow": state.get("coding_decision"),
            "allowed_actions": sorted(allowed),
            "user_mode": state["mode"],
            "runtime_capability_change": state.get("runtime_capability_change", False),
            "completed_step_ids": list(state["done_steps"]),
            "recheck_required": state.get("required_recheck"),
            "required_verification": state.get("required_verification", []),
            "verification_failure": state.get("verification_failure"),
            "verification_unavailable": unavailable_check if defer_pre_patch_verification else None,
            "remaining_repair_steps": state.get("remaining_repair_steps", []),
            "repair_limits": state.get("repair_limits"),
            "request": state.get("request", state["goal"]), "resolved_request": state.get("resolved_request", ""),
            "conversation_context": state.get("conversation_context", []),
            "constraints": state["constraints"], "skills": state["skills"],
            "evidence": _investigation_context(observations),
            "reused_evidence_ids": reused_evidence_ids,
            "deduplicated_inspection_requests": inspection.get("deduplicated_requests", 0) if current_inspection else 0,
            "evidence_levels": scope.evidence_levels,
            "max_next_level": scope.max_next_level,
            "remaining_inspections": min(inspection_tools, MAX_INSPECTION_BATCH) if can_inspect else 0,
            "remaining_workflow_decisions": max(0, remaining_decisions),
            "remaining_workflow_tools": max(0, remaining_tools),
            "inspection_stop": inspection_stop,
            "rejected_inspection_batch": inspection.get("budget_rejection", {}) if current_inspection else {},
            "evidence_discovery_stop": (
                "Repeated evidence requests yielded no new current knowledge. Use available evidence, "
                "select a scoped read if needed, or report the unresolved requirement."
            ) if discovery_stop else "",
            "investigation_report": report,
            "saved_investigation": saved_investigation,
            "repository_evidence": knowledge_context,
            "current_evidence_refs": [ref.model_dump(mode="json") for ref in current_refs[-32:]],
            "pending_repository_evidence": pending,
            "pending_coding_checkpoint_id": pending_checkpoint,
            "pending_coding_evidence_satisfied": resume_ready,
            "revision_request": state.get("revision_request", ""),
            "repair_authorized": state["repair"],
        }, investigation_available=can_investigate, inspection_available=can_inspect, scope=scope,
            workflow_context=workflow_context, lookup_observation=lookup_observation)
        with current_activity_context().publication():
            self._check_live(task_id)
        cited_observations = getattr(decision, "cause_evidence", ()) or getattr(decision, "evidence_ids", ())
        if cited_observations:
            cited_ids = set(cited_observations)
            _refresh_evidence_rows((row for row in evidence if row["id"] in cited_ids), knowledge)
            scope = investigation_scope(evidence)
            validate_observation_references(decision, scope)
        if decision.action == "resume_implementation":
            waiting_task_id, pending_checkpoint, resume_ready = resume_context()
            workflow_context = replace(workflow_context, waiting_task_id=waiting_task_id,
                                       checkpoint_id=pending_checkpoint, resume_ready=resume_ready)
        workflow_context.validate(decision)
        if decision.action == "resume_implementation":
            state["evidence_resume"] = decision.recovery.model_dump(mode="json")
            self._persist(task_id, state)
            return self._execute(task_id, state, service) if planning else None
        if decision.action == "query_evidence":
            started = time.monotonic()
            query = decision.query.model_dump(mode="json")
            prior = state.get("evidence_query") or {"task_id": task_id}
            before = {(ref["evidence_id"], ref["revision"]) for row in evidence
                      if (ref := row.get("board_ref")) and not row.get("stale")}
            matches = knowledge_context if EvidenceQuery.model_validate(prior) == decision.query else knowledge.query(decision.query)
            refs = {(record["evidence_id"], record["revision"]) for record in matches["records"]
                    if record.get("freshness") == "current" and record.get("validation") == "supported"}
            state["evidence_query"] = query
            self._record_evidence_progress(task_id, state, new=len(refs - before), reused=len(refs & before),
                                           title="Searched saved repository evidence", reason=decision.reason,
                                           duration_ms=round((time.monotonic() - started) * 1000))
            return None
        if decision.action == "use_evidence":
            for ref in decision.evidence:
                record = knowledge.get(ref)
                if record.freshness != "current" or record.validation != "supported":
                    raise ValueError("Selected evidence is stale, contradictory, or unsupported")
            refs = list(root_task.evidence_refs)
            additions = [ref.model_dump(mode="json") for ref in dict.fromkeys(decision.evidence)
                         if ref.model_dump(mode="json") not in refs]
            if decision.requirement_id:
                if not waiting_task_id:
                    raise PermissionError("The waiting coding task no longer belongs to this workflow.")
                requirement = gap.requirement.model_copy(update={"evidence": decision.evidence})
                child = self.board.get_task(gap.task_id)
                requirements = [requirement.model_dump(mode="json")]
                newly_bound = child.evidence_requirements != requirements
                if discovery_stop and not newly_bound and not additions:
                    raise RuntimeError("The selected evidence has already been applied to this requirement.")
                knowledge.link(child.task_id, decision.evidence)
                self.board.update_task(child.task_id, evidence_requirements=requirements)
            else:
                newly_bound = False
                if discovery_stop and not additions:
                    raise RuntimeError("The selected evidence is already linked to this task.")
            knowledge.link(task_id, decision.evidence)
            self.board.update_task(task_id, evidence_refs=[*refs, *additions])
            state["evidence_query"] = {"task_id": task_id}
            self._record_evidence_progress(task_id, state, new=len(additions) + int(newly_bound),
                                           reused=len(decision.evidence), title="Reusing saved repository evidence",
                                           reason=decision.reason)
            return None
        if decision.action == "inspect":
            if not can_inspect:
                raise PermissionError("Repository reads are unavailable for this decision.")
            self._inspect(task_id, state, service, decision, inspection_tools)
            return None
        if decision.action == "investigate":
            if not can_investigate:
                raise PermissionError("Repository investigation is unavailable for this decision.")
            self._investigate(task_id, state, service, decision, tool_reserve=tool_reserve)
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
            state["strategy"] = decision.mode
            reason = safe_text(decision.reason, protected=current_activity_context().protected_text)
            state["last_switch_evidence"] = list(scope.evidence_levels)
            state.setdefault("mode_transitions", []).append({**decision.model_dump(mode="json"), "reason": reason})
            self._transition(task_id, state, Phase.ACTIVE, f"Coding strategy: {decision.mode} · {reason}")
            return None
        if decision.action == "research":
            return self._research(task_id, state, decision.debugging_report)
        if decision.action == "diagnose":
            text = "\n\n".join(s for s in (decision.root_cause, decision.uncertainty) if s)
            state["root_cause"] = str(redact_secrets(decision.root_cause))
            state["uncertainty"] = str(redact_secrets(decision.uncertainty))
            if state.get("coding_decision") and state["repair"]:
                return self._block(task_id, state, f"{text}\n\nThe requested repair remains incomplete.")
            return self._complete(task_id, state, text)
        if decision.action == "repair":
            state["root_cause"] = str(redact_secrets(decision.root_cause))
            state["uncertainty"] = str(redact_secrets(decision.uncertainty))
            state["repair_cause_evidence"] = list(decision.cause_evidence)
        if decision.runtime_wiring is not None:
            state["runtime_wiring"] = redact_secrets(decision.runtime_wiring.model_dump(mode="json"))
        else:
            state.pop("runtime_wiring", None)
        return self._draft(task_id, state, decision.draft, service)

    def _record_evidence_progress(self, task_id, state, *, new, reused, title, reason, duration_ms=None, status="success",
                                  deduplicated=0):
        discovery = state.setdefault("evidence_discovery", {})
        boundary = list(self._investigation_boundary(state))
        misses = discovery.get("no_progress", 0) if discovery.get("boundary") == boundary else 0
        discovery.update(boundary=boundary, no_progress=0 if new else misses + 1)
        accounting = state.setdefault("evidence_accounting", {})
        accounting["new_evidence"] = accounting.get("new_evidence", 0) + new
        accounting["reused_evidence"] = accounting.get("reused_evidence", 0) + reused
        if duration_ms is not None:
            accounting["evidence_elapsed_ms"] = accounting.get("evidence_elapsed_ms", 0) + duration_ms
        self._persist(task_id, state)
        activity = current_activity_context()
        emit_activity("coding.progress", task_id=task_id, title=title, status=status,
                      summary=safe_text(reason, protected=activity.protected_text),
                      operation_id=f"{task_id}:evidence:{state['decision_count']}",
                      duration_ms=duration_ms, metadata={"new_evidence": new, "reused_evidence": reused,
                          "deduplicated_requests": deduplicated,
                          "tool_budget_used": state.get("tool_operations", len(state["evidence"])),
                          "tool_budget_remaining": max(0, self._remaining_tools(state))})

    @staticmethod
    def _investigation_boundary(state: dict) -> tuple[int, int]:
        return state["revision"], state.get("investigation_epoch", 0)

    @contextmanager
    def _repository_reader(self, task_id: str, service: PlanService, *, record_inspections: bool = False):
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
                record_taskboard_evidence=record_inspections or task.task_kind in {"investigation", "understanding"},
            )
            yield ToolRunnerAgent(authority)

    def execute_investigation_task(self, task_id):
        """Execute a TaskBoard-selected child using the existing lane and engine."""
        from mana_agent.repository_evidence.investigation import execute_investigation
        from mana_agent.repository_evidence.tasks import investigation_completion_contract
        task = self.board.get_task(task_id)
        root = Path(task.execution_repo_root or self.gateway.root)
        state = {"mode": "auto", "phase": Phase.ACTIVE.value, "revision": 1, "execution_policy": task.execution_policy}
        service = self._service(state=state)
        service.repo_root = root
        reservation = self.lane.reserve(normalized_intent=task.user_request, lane_id=LaneId.RESEARCH,
            taskboard_task_id=task_id, parent_task_id=task.parent_task_id, session_id=task.session_id,
            workspace_id=task.workspace_id, repository_id=task.primary_repository_id,
            execution_workspace=root, target_files=[str(root / path) for path in task.investigation_scope if path != "."],
            completion_contract=investigation_completion_contract(task))
        self.lane.start(reservation)
        try:
            with self._activity(task_id, state, read_only=True):
                with self._repository_reader(task_id, service) as runner:
                    def validate_source(source):
                        from mana_agent.repository_evidence.observations import validate_selected_source
                        return validate_selected_source(runner, source, decision_id=f"{task_id}:assigned")
                    result = execute_investigation(taskboard=self.board, task_id=task_id,
                        coding_agent=self.gateway._coding_agent, decision_id=f"{task_id}:assigned", validate_source=validate_source,
                        runtime={"taskboard": self.board, "execution_registry": current_activity_context().execution_registry,
                                 "execution_supervisor": self.lane.execution_supervisor})
            if result["goal_satisfied"]:
                self.lane.finish(task_id, verification_state=result)
            else:
                self.lane.mark_blocked(task_id, reason="Investigation evidence requirements remain unsatisfied")
            return self.board.get_task(task_id)
        except Exception as exc:
            if self.lane.inspect_task(task_id).state == LaneTaskState.RUNNING:
                self.lane.mark_blocked(task_id, reason=str(exc))
            raise

    def _inspect(self, task_id, state, service, decision, available_tools):
        from mana_agent.multi_agent.tools.operation_policy import validate_arguments

        activity = current_activity_context()
        if not decision.reason.strip():
            raise ValueError("Repository inspection must explain what the next coding decision needs.")
        started = time.monotonic()
        decision_id = f"{task_id}:{activity.attempt_generation}:inspect:{state['decision_count']}"
        evidence = state["evidence"]
        boundary = list(self._investigation_boundary(state))
        previous = state.get("inspection", {})
        inspection = {"id": decision_id, "boundary": boundary, "no_progress": 0, "need": safe_text(
            decision.reason, protected=activity.protected_text,
        )}
        if previous.get("boundary") == boundary:
            inspection["no_progress"] = previous.get("no_progress", 0)

        requested_reads = {request.fingerprint for request in decision.requests}
        prior_evidence = {ref for request in decision.requests for ref in request.prior_evidence}
        knowledge = self.board.evidence_board(root=service.repo_root)
        _refresh_evidence_rows(
            (row for row in evidence if row["fingerprint"] in requested_reads or row["id"] in prior_evidence),
            knowledge,
        )
        with self._repository_reader(task_id, service, record_inspections=True) as runner:
            schemas = runner.authority.schemas
            saved_reads = investigation_scope(evidence).validate_requests(decision.requests, reuse_saved_reads=True)
            selected = []
            reused_evidence_ids = []
            seen = set()
            for request, evidence_id in zip(decision.requests, saved_reads, strict=True):
                if not request.reason.strip():
                    raise ValueError("Each repository read must explain the fact it will resolve.")
                if request.tool not in schemas:
                    raise PermissionError("The model selected a repository tool outside the registered read-only set.")
                service.validate_inspection_target(request)
                arguments = request.model_dump(mode="json", include=set(schemas[request.tool]["properties"]))
                validate_arguments(arguments, schemas[request.tool])
                fingerprint = request.fingerprint
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                if evidence_id is not None:
                    reused_evidence_ids.append(evidence_id)
                    continue
                tool_call = runner.request(
                    request.tool, arguments, call_id=f"{decision_id}:{len(selected) + 1}", decision_id=decision_id,
                )
                selected.append((tool_call, request))
            with activity.publication():
                self._check_live(task_id)
                inspection["reused_evidence_ids"] = reused_evidence_ids
                inspection["deduplicated_requests"] = len(decision.requests) - len(seen)
                state["inspection"] = inspection
                if len(selected) > available_tools:
                    inspection["budget_rejection"] = {"required": len(selected), "remaining": available_tools}
                    inspection["no_progress"] += 1
                    if inspection["no_progress"] >= self.gateway._autonomous_controller.config.max_no_progress_cycles:
                        inspection["stop"] = "Repeated read batches exceeded the remaining shared tool budget."
                    self._persist(task_id, state)
                    return
                summary = "Reading evidence for the next coding decision" if selected else "Reusing saved repository observations"
                self._transition(task_id, state, Phase.ACTIVE, summary)

            fingerprints = {observation_fingerprint(row) for row in evidence} if selected else set()
            new_evidence = 0
            for tool_call, request in selected:
                with activity.publication():
                    self._check_live(task_id)
                    self._charge_tool(task_id, state)
                result = runner.execute(tool_call, lambda _: service.inspect(request)).result
                with activity.publication():
                    self._check_live(task_id)
                    row = {"id": f"evidence-{len(evidence) + 1}", "level": request.level,
                           "fingerprint": request.fingerprint, "result": result}
                    refs = result.get("repository_evidence_refs")
                    if result.get("source_evidence") and refs:
                        row["board_ref"] = refs[0]
                    evidence.append(row)
                    fingerprint = observation_fingerprint(row)
                    if fingerprint is not None and fingerprint not in fingerprints:
                        fingerprints.add(fingerprint)
                        new_evidence += 1
                    self._persist(task_id, state)
            with activity.publication():
                self._check_live(task_id)
                inspection["no_progress"] = 0 if new_evidence else inspection["no_progress"] + 1
                if inspection["no_progress"] >= self.gateway._autonomous_controller.config.max_no_progress_cycles:
                    inspection["stop"] = "Repository reads stopped yielding new evidence."
                summary = "Repository observations saved" if selected else "Saved repository observations reused"
                self._transition(task_id, state, Phase.ACTIVE, summary)
                self._record_evidence_progress(task_id, state, new=new_evidence, reused=len(reused_evidence_ids),
                                               title=summary, reason=decision.reason,
                                               deduplicated=inspection["deduplicated_requests"],
                                               duration_ms=round((time.monotonic() - started) * 1000))

    def _investigate(self, task_id, state, service, decision, *, tool_reserve=0):
        from mana_agent.coding.evidence_tools import scoped_paths

        activity = current_activity_context()
        root = self._workspace_root(state).resolve()
        if decision.recovery is not None:
            saved = self._saved_investigation(task_id, decision.recovery.task_id)
            if saved is None:
                raise PermissionError("No saved child investigation belongs to this Coding task.")
            child, execution, recovery_target = saved
            try:
                validate_investigation_recovery(decision, recovery_target)
            except PydanticCustomError as exc:
                raise PermissionError(str(exc)) from None
            knowledge = self.board.evidence_board(root=root)
            refs = []
            for raw in child.evidence_refs:
                ref = EvidenceRef.model_validate(raw)
                record = knowledge.get(ref)
                if record.freshness == "current" and record.validation == "supported":
                    refs.append(ref)
            with activity.publication():
                self._check_live(task_id)
                knowledge.link(task_id, tuple(refs))
                current = self.board.get_task(task_id).evidence_refs
                self.board.update_task(task_id, evidence_refs=[*current, *(
                    ref.model_dump(mode="json") for ref in refs if ref.model_dump(mode="json") not in current
                )])
                if execution.state not in {ExecutionState.COMPLETED, ExecutionState.CANCELLED}:
                    self.lane.ensure_recoverable_execution(child.task_id)
                    self.lane.cancel_task(child.task_id, reason="Scoped repository evidence moved to its Coding task",
                                          source="coding_evidence_migration")
                state["investigation"] = {"status": "reused", "report": {
                    "evidence_ids": [ref.evidence_id for ref in refs], "gaps": "Select fresh reads for any unresolved requirement.",
                }}
                state["evidence_query"] = {"task_id": task_id}
                self._record_evidence_progress(task_id, state, new=0, reused=len(refs),
                    title="Saved investigation evidence imported", reason=decision.objective)
            return

        paths = scoped_paths(root, decision.paths)
        for request in decision.requests:
            if request.level == 4:
                raise PermissionError("A scoped investigation cannot run a repository-wide search")
            scoped_paths(root, request.paths, authority=paths)
        state["investigation"] = {"status": "selected", "objective": safe_text(
            decision.objective, protected=activity.protected_text), "paths": list(paths)}
        self._persist(task_id, state)
        self._inspect(task_id, state, service, decision, max(0, self._remaining_tools(state) - tool_reserve))
        state["investigation"]["status"] = "budget_limited" if state["inspection"].get("budget_rejection") else "returned"
        self._persist(task_id, state)

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
        if state.get("repair_limits") is not None:
            for relative, digest in state.get("preserved_file_digests", {}).items():
                if self._file_digest(self._workspace_root(state), relative) != digest:
                    raise PermissionError("Changes from the previous revision no longer match the saved receipts")
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
            evidence_refs=tuple(self.board.get_task(task_id).evidence_refs),
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
            return None
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
            supervisor = self.lane.execution_supervisor
            durable = supervisor.store.get_task_or_none(task.task_id)
            if durable and durable.wall_clock_deadline_exceeded(supervisor.clock()):
                if _DEADLINE_MESSAGE not in text:
                    text += f"\n\n{_DEADLINE_MESSAGE}"
            elif state["phase"] == Phase.BLOCKED or (durable and durable.state == ExecutionState.INTERRUPTED):
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
        supervisor = self.lane.execution_supervisor
        durable = supervisor.store.get_task_or_none(task_id)
        if (durable and durable.state not in {ExecutionState.COMPLETED, ExecutionState.CANCELLED}
                and durable.wall_clock_deadline_exceeded(supervisor.clock())):
            return self._outcome(task_id, {**state, "deadline_expired": True}, _DEADLINE_MESSAGE,
                                 self._repository_report(state), completed=False)
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
        if not state.get("plan"):
            # An operator retry can make a previously unavailable verifier usable again.
            state.pop("verification_unavailable", None)
        if revision_request:
            result = self._restart_draft(task_id, state, revision_request)
        elif retry_investigation or state.get("pending_repository_evidence") or not state.get("plan"):
            state["investigation_epoch"] = state.get("investigation_epoch", 0) + 1
            self._transition(task_id, state, Phase.ACTIVE)
            result = self._drive(task_id, state)
        else:
            result = self._execute(task_id, state, self._service(state=state))
        self._publish_result(task_id, result)
        return result

    @staticmethod
    def _can_retry_investigation(state: dict) -> bool:
        return state["phase"] in {Phase.BLOCKED, Phase.ACTIVE} and not any(state.get(key) for key in (
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
            self._transition(task_id, state, Phase.ACTIVE)
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
        self._transition(task_id, state, Phase.ACTIVE, "Plan scope is being revised")

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

    def _execute(self, task_id, state, service, *, in_controller=False):
        if state["mode"] != "plan" and not in_controller:
            return self._drive(task_id, state, service=service)
        try:
            with self._activity(task_id, state, read_only=False):
                return self._execute_steps(task_id, state, service)
        except ModeEvidenceWaiting:
            gap = state["pending_repository_evidence"]
            self._transition(task_id, state, Phase.ACTIVE, f"Coding needs evidence · {gap['requirement']['objective']}")
            return None if in_controller else self._drive(task_id, state, service=service)
        except ModeBatchWaiting as exc:
            return CommandResult(status="input_required", message=str(exc), data={"task_id": task_id})
        except ScopeRevisionNeeded as exc:
            if state["mode"] == "plan":
                return self._restart_draft(task_id, state, str(exc))
            return self._block(task_id, state, str(exc))
        except VerificationUnavailable as exc:
            if in_controller and self._reconsider_unavailable_pre_patch_check(task_id, state, exc):
                return None
            if exc.command and exc.command.uses_git_checkout:
                return self._block(task_id, state, f"**Verification pending.** {exc}")
            return self._block(
                task_id, state, f"**Verification pending.** {exc}\n\n"
                f"Once Docker is ready, run `/task resume {task_id}` to continue.",
            )
        except VerificationFailed as exc:
            if in_controller and state.get("coding_decision") and state["repair"]:
                return self._record_failed_check(task_id, state, exc)
            return self._block(
                task_id, state, "**Verification failed.** See Coding Activity for the command and output.\n\n"
                f"After addressing the failure, run `/task resume {task_id}`.",
            )
        except Exception as exc:
            if in_controller:
                raise
            return self._block(task_id, state, str(redact_secrets(str(exc))))

    def _reconsider_unavailable_pre_patch_check(self, task_id, state, failure: VerificationUnavailable) -> bool:
        if (failure.command is None or not state.get("coding_decision") or not state["repair"]
                or state["mode"] == "plan" or state.get("verification_unavailable")
                or state["files_changed"] or state.get("in_flight_mutation")
                or self._remaining_decisions(state) < _PATCH_HANDOFF_MODEL_TURNS
                or self._remaining_tools(state) < _PATCH_HANDOFF_TOOL_CALLS):
            return False
        plan = PlanRevision.model_validate(state["plan"])
        if state.get("required_recheck") == failure.command.model_dump(mode="json"):
            return False
        first_change = next((i for i, step in enumerate(plan.draft.steps) if step.changes), 0)
        if not any(step.step_id == state["step_id"] and failure.command in step.verification
                   for step in plan.draft.steps[:first_change]):
            return False
        self._validate_approval(task_id, state, plan)
        limits: dict[str, list[str]] = {}
        for step in plan.draft.steps:
            for path, operations in step.changes.items():
                limits[path] = sorted(set(limits.get(path, ())).union(operations))
        checks = [check.model_dump(mode="json") for step in plan.draft.steps for check in step.verification
                  if check.expected == "pass" and not check.uses_git_checkout]
        required = [*state.get("required_verification", []), *checks]
        state.update(
            verification_unavailable={"command": failure.command.model_dump(mode="json"),
                                      "reason": str(redact_secrets(str(failure))),
                                      "prior_finding": state.get("root_cause", "")},
            repair_limits=state.get("repair_limits") or limits,
            required_verification=list({canonical_digest(check): check for check in required}.values()),
            preserved_file_digests={**plan.base_file_digests, **state.get("preserved_file_digests", {})},
            remaining_repair_steps=[step.model_dump(mode="json") for step in plan.draft.steps
                                    if step.step_id not in state["done_steps"]],
        )
        self._revise(task_id, state, "Reassess the unavailable pre-patch verification using saved evidence.")
        return True

    def _record_failed_check(self, task_id, state, failure: VerificationFailed):
        plan = PlanRevision.model_validate(state["plan"])
        self._validate_approval(task_id, state, plan)
        command = failure.command.model_dump(mode="json")
        signature = canonical_digest({
            "command": command, "exit_code": failure.outcome["exit_code"],
            "output": failure.outcome["output"].strip(),
        })
        signatures = state.setdefault("verification_failure_counts", {})
        signatures[signature] = signatures.get(signature, 0) + 1
        state["verification_failure"] = {"command": command, "result": failure.outcome}
        self._persist(task_id, state)
        if signatures[signature] >= self.gateway._autonomous_controller.config.max_no_progress_cycles:
            return self._block(task_id, state, "The same verification failure recurred without a new diagnostic. "
                               "See Coding Activity for the exact command and output.")
        limits = state.get("repair_limits")
        if limits is None:
            limits = {}
            for step in plan.draft.steps:
                for path, operations in step.changes.items():
                    limits[path] = sorted(set(limits.get(path, ())).union(operations))
        for path in state["files_changed"]:
            if "add" in limits.get(path, []) and "update" not in limits[path]:
                limits[path].append("update")

        preserved = state.setdefault("preserved_file_digests", {})
        for step in plan.draft.steps:
            for receipts in (state["done_steps"], state.get("patch_steps", {})):
                preserved.update(receipts.get(step.step_id, {}).get("files", {}))
        root = Path(plan.repository_root)
        if any(self._file_digest(root, path) != digest for path, digest in preserved.items()):
            raise PermissionError("Changes from the failed revision no longer match the saved receipts")
        checks = state.get("required_verification", [])
        current_checks = [check.model_dump(mode="json") for step in plan.draft.steps
                          for check in step.verification if check.expected == "pass"]
        checks = list({canonical_digest(check): check for check in [*checks, *current_checks]}.values())
        state.update(repair_limits=limits, required_recheck=command, required_verification=checks)
        state["remaining_repair_steps"] = [
            step.model_dump(mode="json") for step in plan.draft.steps if step.step_id not in state["done_steps"]
        ]
        state["evidence"].append({"id": f"evidence-{len(state['evidence']) + 1}", "level": 0,
                                  "fingerprint": signature, "result": state["verification_failure"]})
        self._revise(task_id, state, "Diagnose the failed verification within the accepted scope and rerun it.")
        emit_activity("coding.progress", task_id=task_id, title="Verification failure needs diagnosis",
                      summary=f"{failure.command.argv!r} · exit {failure.outcome['exit_code']}",
                      status="failed", operation_id=f"{task_id}:diagnose:{state['revision']}")
        return None

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
        latest_digests = dict(state.get("preserved_file_digests", {}))
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
                return None if state["mode"] != "plan" else self._drive(task_id, state, service=service)
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
                self._revise(task_id, state, revision_reason)
                return None if state["mode"] != "plan" else self._drive(task_id, state, service=service)
        if state.get("runtime_capability_change") and state["repair"]:
            self._require_runtime_evidence(task_id, state, plan)
        if state.get("coding_decision"):
            self._transition(task_id, state, Phase.REVIEWING, "Reviewing the verified changes")
            self._validate_approval(task_id, state, plan)
            for relative, digest in {**state.get("preserved_file_digests", {}), **expected_files}.items():
                if self._file_digest(root, relative) != digest:
                    raise PermissionError("The reviewed checkout changed before finalization")
            if not any(check["passed"] and check["exit_code"] == 0
                       for check in state.get("verification", [])):
                raise RuntimeError("A passing verification command is required before completion")
        summary = _changed_files_summary(state["files_changed"])
        lines = [f"{summary} All required verification checks passed.".strip()]
        if state.get("root_cause"):
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
            except ModeEvidenceWaiting:
                self.lane.checkpoint(child_id, boundary="missing_repository_evidence",
                    resume_payload={"plan_digest": plan.digest, "step_id": step.step_id,
                                    "pending_repository_evidence": state["pending_repository_evidence"],
                                    "evidence_refs": self.board.get_task(child_id).evidence_refs,
                                    "continuation": state["coding_continuations"][child_id]})
                self.lane.transition(child_id, LaneTaskState.WAITING, reason="Waiting for repository evidence")
                self.lane.lock_manager.release_task(child_id)
                raise
            except ModeBatchWaiting:
                raise
            except Exception as exc:
                if self.lane.inspect_task(child_id).state == LaneTaskState.RUNNING:
                    self.lane.mark_blocked(child_id, reason=str(redact_secrets(str(exc))))
                raise

    def _require_runtime_evidence(self, task_id: str, state: dict, plan: PlanRevision | None) -> None:
        from mana_agent.gateway.feature_integration import FeatureIntegrationCoordinator, IntegrationAuthority, WiringDecision

        authority = IntegrationAuthority.from_taskboard(self.board, task_id)
        task = self._check_live(task_id)
        root = self._workspace_root(state).resolve()
        saved = state.get("runtime_wiring")
        decision = provider = None
        receipts = None
        if saved is not None:
            wiring = RuntimeWiringEvidence.model_validate(saved)
            edges = []
            for edge in wiring.edges:
                target = root / edge.path
                if target.resolve() != target or not target.is_file():
                    raise VerificationUnavailable(f"Runtime reachability source is unavailable in the Coding checkout: {edge.path}")
                reference = f"{edge.path}:{edge.line}" if edge.line else edge.path
                edges.append({"from": edge.source, "to": edge.target, "relation": edge.relation, "source_reference": reference})
            checks = ({(tuple(command.argv), command.cwd) for step in plan.draft.steps
                       for command in step.verification if command.expected == "pass"} if plan else
                      {(tuple(row["argv"]), row["cwd"]) for row in state["verification"] if row["expected"] == "pass"})
            targets = set(plan.base_file_digests if plan else ()) | set(state["files_changed"])
            receipts = [row for row in state.get("verification", []) if row["passed"] and row["exit_code"] == 0
                        and row.get("operation_id") and row.get("attempt_id")
                        and (tuple(row["argv"]), row["cwd"]) in checks
                        and targets.issubset(row.get("source_versions", {}))
                        and all(self._file_digest(root, path) == version for path, version in row["source_versions"].items())]
            if not receipts:
                raise RuntimeError("Runtime reachability requires a passing approved check against the current patched checkout")
            if authority and not set(authority.verification_provenance.get("isolated_operation_ids", ())).intersection(
                row["operation_id"] for row in receipts
            ):
                raise RuntimeError("The previous runtime completion receipt belongs to a different verification result")
            decision = WiringDecision(outcome="already_integrated", edges=edges, reason=wiring.reason,
                verification_commands=list(dict.fromkeys(shlex.join(row["argv"]) for row in receipts)))
        if authority and authority.is_complete():
            state["integration"] = redact_secrets(asdict(authority))
            self._persist(task_id, state)
            return
        if saved is None:
            # Older approved revisions do not carry a Coding-owned wiring decision.
            self._charge_tool(task_id, state)
            self._charge_model(task_id, state)
            provider = self.gateway._feature_integration_decision_provider(
                task_id=task_id, options={"execution_policy": state["execution_policy"],
                    "_selected_model": state.get("model"), "_selected_provider": state.get("provider")},
            )
        try:
            result = FeatureIntegrationCoordinator().run(
                coding_agent=self.gateway._coding_agent,
                core_result={"status": "completed", "changed_files": state["files_changed"]},
                request=state["goal"], gateway_task_id=task_id, flow_id=None,
                runtime_capability_change=True,
                taskboard=self.board, taskboard_parent_task_id=task_id,
                execution_supervisor=self.lane.execution_supervisor,
                trigger_turn_id=task.trigger_turn_id, workspace_root=root,
                integration_decision=decision, integration_decision_provider=provider,
                isolated_verification=receipts,
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
            if state.get("evidence_resume", {}).get("task_id") == child_id:
                decision = RecoveryDecision.model_validate(state.pop("evidence_resume"))
                reservation = self.lane.resume_checkpoint(child_id, session_id=plan.session_id, decision=decision)
                self.lane.start(reservation)
                state.pop("pending_repository_evidence", None)
                self._persist(task_id, state)
                return
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
        configured = self.gateway._coding_agent
        if configured is None:
            raise RuntimeError("No configured coding engine is available")
        policy = TaskExecutionPolicy.model_validate(state["execution_policy"])
        selected_backend = "internal" if policy.required else getattr(configured, "active_engine_mode", "")
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
        try:
            operations = validate_step_patch(root, step, patch)
        except PatchFormatError as exc:
            diagnostic = f"Invalid patch format. {exc}."
            if pending or policy.required:
                raise RuntimeError(diagnostic) from exc
            if self._remaining_decisions(state) < 2 or self._remaining_tools(state) < _PATCH_HANDOFF_TOOL_CALLS:
                raise RuntimeError(
                    f"{diagnostic} The workflow budget cannot cover a correction, review, apply, and focused check."
                ) from exc
            if (self.gateway._coding_agent is not configured
                    or getattr(configured, "active_engine_mode", "") != selected_backend):
                raise RuntimeError(f"{diagnostic} The configured coding backend changed.") from exc
            self._validate_approval(task_id, state, plan)
            self._check_live(child_id)
            protected = current_activity_context().protected_text
            if any(secret and secret in patch for secret in protected):
                raise ValueError("Coding proposal contains protected material") from exc
            emit_activity("coding.progress", task_id=child_id, title="Patch proposal rejected",
                          summary=diagnostic, status="error",
                          operation_id=f"{child_id}:patch-format:{state['decision_count']}",
                          metadata={"error_code": "invalid_patch_format"})
            try:
                patch = self._propose_patch(task_id, child_id, state, plan, step, service,
                                            correction={"reason": str(exc), "previous_patch": patch})
                operations = validate_step_patch(root, step, patch)
            except ScopeRevisionNeeded:
                raise
            except (ValueError, RuntimeError) as correction_error:
                detail = safe_text(correction_error, limit=500, protected=protected)
                raise RuntimeError(f"{diagnostic} Patch correction failed. {detail}") from correction_error
        self._commit_patch(task_id, child_id, state, plan, step, service, expected_files, patch, operations, approval_id)

    def _propose_patch(self, task_id, child_id, state, plan, step, service, *,
                       correction: dict[str, str] | None = None):
        self._charge_tool(task_id, state)
        self._charge_model(task_id, state)
        state["used_coding_agent"] = True
        self._persist(task_id, state)
        root = self._workspace_root(state)
        from mana_agent.repository_evidence.models import EvidenceQuery, EvidenceRef, EvidenceRequirement
        from mana_agent.repository_evidence.tasks import assignment_for
        knowledge = self.board.evidence_board(root=root)
        current_refs = []
        for ref in self.board.get_task(task_id).evidence_refs:
            record = knowledge.get(EvidenceRef.model_validate(ref))
            if record.freshness == "current" and record.validation == "supported":
                current_refs.append(ref)
        child = self.board.get_task(child_id)
        child.execution_repo_root = str(root)
        self.board.update_task(child_id, task_kind="implementation", investigation_scope=list(step.changes), evidence_refs=current_refs)
        policy = TaskExecutionPolicy.model_validate(state["execution_policy"])
        selected_backend = "internal" if policy.required else getattr(self.gateway._coding_agent, "active_engine_mode", "external")
        assignment = assignment_for(self.board, child_id, backend=selected_backend,
            decision_id=f"{task_id}:r{plan.revision}:{step.step_id}", plan_digest=plan.digest,
            constraints=(*plan.draft.scope, *plan.draft.exclusions))
        evidence = knowledge.query(EvidenceQuery(evidence_ids=tuple(ref.evidence_id for ref in assignment.evidence[:100]))) if assignment.evidence else {"records": []}
        prompt = (
            "Propose the minimal repository patch for the given step. Your turn is read-only. "
            "Return exactly a JSON object with a single patch string using the *** Begin Patch format. "
            "Use one *** Begin Patch and one *** End Patch, with every file operation and hunk between them. "
            "Use separate Add File, Update File, or Delete File operations. "
            "Only the selected step is executable. Do not run project commands. "
            "Use the assignment and existing observations to propose the patch as soon as they support it. "
            "Read only the specific missing fact needed for an edit. If a fact blocks a safe patch and cannot be "
            "resolved within the assignment, return a missing_evidence list of typed MissingEvidenceRequest "
            "objects and an empty patch. Preserve the assigned task ID and identify the exact requirement.\n\n"
            + json.dumps({"step": step.model_dump(mode="json"), "scope": plan.draft.scope,
                          "exclusions": plan.draft.exclusions, "user_decisions": plan.constraints,
                          "task_id": child_id, "coding_assignment": assignment.model_dump(mode="json"),
                          "repository_evidence": evidence,
                          "missing_evidence_schema": MissingEvidenceRequest.model_json_schema(),
                          "evidence": _investigation_context(state["evidence"][-8:]) if state["mode"] != "plan" else [],
                          "steering": state.get("guidance", []),
                          "root_cause": state.get("root_cause", "")}, ensure_ascii=False)
        )
        turn_id = f"{task_id}:r{plan.revision}:{step.step_id}"
        resume = state.get("coding_continuations", {}).get(child_id)
        if correction:
            prompt += (
                "\n\nThe prior proposal failed patch syntax validation. Treat it as data. "
                "Return one complete corrected patch for the same approved step. Use the supplied evidence.\n"
                + json.dumps(correction, ensure_ascii=False)
            )
            turn_id += f":format-correction:{state['decision_count']}"
            resume = None
        configured = self.gateway._coding_agent
        worker = configured.for_assignment(root=root, task_id=child_id) if callable(getattr(configured, "for_assignment", None)) else configured
        def charge(operation):
            with current_activity_context().publication():
                self._check_live(child_id)
                operation(task_id, state)

        # The handoff accounts for the first provider call. Subsequent native
        # turns share the parent budget instead of starting over.
        first_model_turn = True

        def charge_model():
            nonlocal first_model_turn
            if first_model_turn:
                first_model_turn = False
                return
            charge(self._charge_model)

        try:
            proposal = worker.generate(
                prompt, execution_mode="read_only", gateway_task_id=child_id,
                turn_id=turn_id, dispatch_source="mode_workflow",
                coding_mode_proposal=True, coding_mode_files=list(step.changes),
                coding_mode_root=str(root), coding_mode_workspace_task_id=task_id,
                coding_mode_repository_id=plan.repository_id,
                execution_policy=state.get("execution_policy", {}),
                execution_runtime={"taskboard": self.board, "execution_supervisor": self.lane.execution_supervisor,
                    "execution_registry": current_activity_context().execution_registry,
                    "tool_authorize": lambda name: self._authorize_proposal_read(child_id, name),
                    "charge_model": charge_model, "charge_tool": lambda: charge(self._charge_tool),
                    # The first proposal request is precharged. Keep one model turn for review and tools for apply and check.
                    "remaining_model_turns": 1 if correction else self._remaining_decisions(state),
                    "remaining_tool_calls": 0 if correction else max(
                        0, self._remaining_tools(state) - (_PATCH_HANDOFF_TOOL_CALLS - 1)),
                    "max_no_progress_cycles": self.gateway._autonomous_controller.config.max_no_progress_cycles},
                coding_resume_context=resume or {},
                coding_continuation_id=f"{child_id}:{state.get('decision_count', 0)}" if resume is not None else "",
                **self._batch_arguments(child_id, plan, state),
            )
        finally:
            if worker is not configured:
                worker.close()
        self._validate_approval(task_id, state, plan)
        if proposal.get("missing_evidence"):
            if correction:
                raise ValueError("The coding engine could not correct the patch from the supplied evidence")
            gaps = [MissingEvidenceRequest.model_validate(item) for item in proposal["missing_evidence"]]
            if len(gaps) != 1 or gaps[0].task_id != child_id:
                raise ValueError("Report one exact assigned-task evidence requirement at this boundary")
            gap = gaps[0]
            for requirement in assignment.requirements:
                if (requirement.model_dump(exclude={"evidence"}) == gap.requirement.model_dump(exclude={"evidence"})
                        and knowledge.satisfies(requirement)):
                    raise RuntimeError("The coding engine repeated an evidence requirement already satisfied on this assignment.")
            continuation = proposal.get("continuation_context") or {}
            protected = current_activity_context().protected_text
            serialized = json.dumps(continuation, ensure_ascii=False)
            if (not isinstance(continuation, dict) or redact_secrets(continuation) != continuation
                    or safe_text(serialized, limit=None, protected=protected) != serialized):
                raise ValueError("The coding continuation contains protected material or an invalid checkpoint")
            self.board.update_task(child_id, missing_evidence=[gap.model_dump(mode="json")],
                evidence_requirements=[gap.requirement.model_dump(mode="json")])
            state["pending_repository_evidence"] = gap.model_dump(mode="json")
            state.setdefault("coding_continuations", {})[child_id] = continuation
            self._persist(task_id, state)
            raise ModeEvidenceWaiting(gap.reason)
        if proposal.get("pending_operation"):
            self._suspend_batch(task_id, child_id, state, plan, proposal["pending_operation"])
        if state.get("batch_child") == child_id:
            state.pop("batch_child", None)
            self._persist(task_id, state)
        if proposal.get("status") not in {"completed", "success"}:
            detail = safe_text(proposal.get("errors") or proposal.get("error_code") or proposal.get("status"),
                               limit=500, protected=current_activity_context().protected_text)
            raise RuntimeError(f"The configured coding engine did not complete the patch proposal. {detail}".strip())
        revision_reason = self._steering(task_id, state, plan, step, service)
        if revision_reason:
            raise ScopeRevisionNeeded(revision_reason)
        return parse_proposal(proposal.get("output") or proposal.get("answer"))

    def _authorize_proposal_read(self, task_id, name):
        if name in {"repository_evidence_query", "repository_evidence_get",
                    "investigate_repository", "integrate_repository_evidence"}:
            self._check_live(task_id)
            return self.lane.authorize_tool(task_id, name)
        return self._authorize_batch_read(task_id, name)

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

    def _commit_patch(self, task_id, child_id, state, plan, step, service, expected_files, patch, operations, approval_id):
        from mana_agent.gateway.lanes import validate_tool_permission
        from mana_agent.multi_agent.runtime.tool_call_gate import execute_host_operation

        root = self._workspace_root(state)
        child = self.lane.inspect_task(child_id)
        validate_tool_permission(self.lane.contracts[child.owning_lane], "apply_patch",
                                 task_capabilities=tuple(child.capabilities))
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
        operation_id = f"{task_id}:verify:{state['tool_operations']}"
        command_detail = f"{command.argv!r} in {command.cwd}"
        emit_activity("coding.progress", task_id=task_id, title="Running verification command",
                      summary=command_detail, status="running", operation_id=operation_id)
        started = time.monotonic()
        root = self._workspace_root(state)
        plan = PlanRevision.model_validate(state["plan"])
        versions = {path: self._file_digest(root, path) for path in
                    dict.fromkeys([*plan.base_file_digests, *state["files_changed"]])}
        try:
            outcome = asyncio.run(run_isolated_verification(root, command, task_id=task_id))
        except VerificationUnavailable as exc:
            emit_activity("coding.progress", task_id=task_id, title="Verification pending",
                          summary=str(exc), detail=f"{exc}\n{command_detail}",
                          status="unverified", operation_id=operation_id,
                          metadata={"verification_status": "not_run"})
            raise VerificationUnavailable(str(exc), command=command) from exc
        activity = current_activity_context()
        with activity.publication():
            durable = self._check_live(task_id)
            if any(self._file_digest(root, path) != digest for path, digest in versions.items()):
                raise PermissionError("Verification input files changed during the check")
            outcome = {**outcome, "argv": list(command.argv), "cwd": command.cwd,
                       "output": safe_text(outcome["output"], limit=2000, protected=activity.protected_text),
                       "operation_id": operation_id, "attempt_id": durable.attempt_id,
                       "duration_ms": round((time.monotonic() - started) * 1000), "source_versions": versions}
            state.setdefault("verification", []).append(outcome)
            self._persist(task_id, state)
        self._remember_verification(plan, outcome)
        emit_activity("coding.progress", task_id=task_id, title="Verification command",
                      summary=f"{command.argv!r} · {outcome['exit_code']}",
                      detail=f"{command_detail}\n{outcome['output']}",
                      status="success" if outcome["passed"] else "failed",
                      operation_id=operation_id, duration_ms=outcome["duration_ms"])
        if not outcome["passed"]:
            raise VerificationFailed(command, outcome)

    def _remember_verification(self, plan, outcome):
        if not getattr(getattr(self.gateway, "config", None), "coding_memory", False):
            return
        from mana_agent.services.coding_memory_service import CodingMemoryService

        try:
            root = Path(plan.repository_root)
            memory = CodingMemoryService(project_root=root, session_id=plan.session_id)
            identity = canonical_digest([plan.workspace_id, plan.repository_id, str(root), plan.plan_id])
            flow = memory.ensure_flow(flow_id=identity, request=plan.goal, activate=False)
            memory.record_verification_result(
                flow_id=flow, results={"checkout": plan.base_checkout, **outcome},
                status="passed" if outcome["passed"] else "failed",
            )
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Coding verification history unavailable: %s", type(exc).__name__)

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
                         reconciled_file_versions=state.get("adaptive_file_versions") if state.get("adaptive_coding") else None,
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
        supervisor = self.lane.execution_supervisor
        durable = supervisor.get_task(task_id)
        expired = (not durable.waiting_inbox_item_id and durable.wall_clock_deadline_exceeded(supervisor.clock()))
        if expired:
            state["deadline_expired"] = True
            message = _DEADLINE_MESSAGE
        report = self._repository_report(state)
        sections = [_changed_files_summary(state["files_changed"]), message]
        if state["files_changed"] and state.get("isolated_workspace"):
            sections.insert(1, f"Worktree {_code_span(state['isolated_workspace']['path'])}.")
        for detail in (report["gaps"], report["stop"], report["pending_question"]):
            if detail and not any(detail in section for section in sections):
                sections.append(detail)
        message = str(redact_secrets("\n\n".join(filter(None, sections))))
        state["result"] = message
        if durable.cancellation_status is None and durable.state not in {ExecutionState.COMPLETED, ExecutionState.CANCELLED}:
            execution = self.lane.inspect_task(task_id)
            if execution.state in {LaneTaskState.RUNNING, LaneTaskState.QUEUED, LaneTaskState.WAITING}:
                if expired:
                    root = self._workspace_root(state)
                    self.lane.finish(task_id, state=LaneTaskState.TIMED_OUT, error=message,
                                     changed_files=[str(root / path) for path in state["files_changed"]])
                else:
                    self.lane.mark_blocked(task_id, reason=message)
            self._transition(task_id, state, Phase.BLOCKED)
        return self._outcome(task_id, state, message, report, completed=False)

    @staticmethod
    def _outcome(task_id: str, state: dict, message: str, report: dict, *, completed: bool) -> CommandResult:
        data = {"task_id": task_id, "phase": state["phase"], "mode": state["mode"],
                "mode_reason": state.get("mode_reason", ""), "changed_files": state["files_changed"],
                "goal_satisfied": completed, "repository_observations": report,
                "used_coding_agent": state.get("used_coding_agent", False),
                "verification": state.get("verification", []), "strategy": state.get("strategy", state["mode"])}
        if completed:
            data["integration"] = state.get("integration", {})
        elif state.get("deadline_expired"):
            data["error_code"] = "DEADLINE_EXPIRED"
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
        plan_findings = (state.get("plan") or {}).get("draft", {}).get("findings")
        findings = state.get("root_cause") or plan_findings or native_report.get("findings", "")
        evidence_ids = ([ref["evidence_id"] for ref in state.get("adaptive_evidence_refs", [])]
                        if state.get("adaptive_coding") else list(investigation_scope(state["evidence"]).evidence_levels)[-12:])
        return {"evidence_ids": evidence_ids,
                "findings": findings, "gaps": state.get("uncertainty") or native_report.get("gaps", ""),
                "workflow_model_decisions": state.get("decision_count", 0),
                "workflow_tool_operations": state.get("tool_operations", len(state["evidence"])),
                "evidence_accounting": state.get("evidence_accounting", {}),
                "stop": stop, "pending_question": inspection.get("need", "") if inspection.get("stop") else ""}


def parse_proposal(raw: Any) -> str:
    from mana_agent.repository_evidence.handoff import json_response_text

    if not isinstance(raw, str) or len(raw) > 250_000:
        raise ValueError("Coding proposal must be a bounded JSON object")
    value = json.loads(json_response_text(raw))
    if not isinstance(value, dict) or not set(value).issubset({"patch", "missing_evidence"}) or "patch" not in value or value.get("missing_evidence") or not isinstance(value["patch"], str) or not value["patch"]:
        raise ValueError("Coding proposal did not include a patch for this step")
    return value["patch"]


def validate_step_patch(root: Path, step: PlanStep, patch: str) -> list[tuple[str, str]]:
    if redact_secrets(patch) != patch:
        raise ValueError("Coding proposal contains protected material")
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

    if command.uses_git_checkout:
        raise VerificationUnavailable("The isolated source snapshot has no .git metadata. Revise the plan to use a project check.")
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
            "passed": passed, "output": output[-2000:],
            "execution_receipt": result.model_dump(mode="json", exclude={"stdout", "stderr", "resource_usage"})}
