"""Gateway integration for supervised Plan and Debug tasks."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
import getpass
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
from typing import Any

from mana_agent.agent.mode import ModePhase as Phase, PlanRevision, PlanStep, VerificationCommand, repository_relative_path
from mana_agent.agent.plan_service import PlanService
from mana_agent.batch.policy import TaskExecutionPolicy
from mana_agent.chat_commands.models import CommandResult
from mana_agent.execution_supervisor.models import ExecutionState
from mana_agent.gateway.lane_coordinator import LaneReservation
from mana_agent.gateway.lanes import LaneId, LaneTaskState
from mana_agent.human_inbox.models import (
    ClarificationField, ExpectedResponseType, InboxRequest, InboxRequestType, InboxStatus,
    ReviewerAssignment, ReviewerType, ResponseOperation, ResponseSubmission, canonical_digest,
)
from mana_agent.multi_agent.core.types import TaskStatus
from mana_agent.services.activity_context import ActivityContext, activity_scope, current_activity_context, emit_activity
from mana_agent.tools.apply_patch import extract_patch_operations, safe_apply_patch
from mana_agent.utils.redaction import redact_secrets

_OWNER = "autonomous_loop"


class ScopeRevisionNeeded(RuntimeError):
    pass


class ModeBatchWaiting(RuntimeError):
    pass


_TERMINAL = {Phase.COMPLETED, Phase.CANCELLED}
_TRANSITIONS = {
    Phase.UNDERSTANDING: {Phase.INVESTIGATING, Phase.CLARIFICATION_REQUIRED, Phase.DRAFTING_PLAN, Phase.COMPLETED},
    Phase.INVESTIGATING: {Phase.INVESTIGATING, Phase.CLARIFICATION_REQUIRED, Phase.DRAFTING_PLAN, Phase.COMPLETED},
    Phase.CLARIFICATION_REQUIRED: {Phase.UNDERSTANDING},
    Phase.DRAFTING_PLAN: {Phase.AWAITING_APPROVAL, Phase.APPROVED, Phase.COMPLETED},
    Phase.AWAITING_APPROVAL: {Phase.APPROVED, Phase.UNDERSTANDING},
    Phase.APPROVED: {Phase.EXECUTING, Phase.UNDERSTANDING},
    Phase.EXECUTING: {Phase.EXECUTING, Phase.VERIFYING, Phase.UNDERSTANDING, Phase.COMPLETED},
    Phase.VERIFYING: {Phase.EXECUTING, Phase.COMPLETED, Phase.UNDERSTANDING},
    Phase.BLOCKED: {Phase.UNDERSTANDING, Phase.EXECUTING},
}


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

    def _service(self, *, llm=None):
        from mana_agent.gateway.turn_engine import agent_decision_llm

        if llm is None:
            llm = getattr(self.gateway._entry_router, "llm", None) or agent_decision_llm(self.gateway.get_ask_service())
        return PlanService(self.gateway.root, llm=llm)

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
        description = summary or target.value.replace("_", " ").capitalize()
        task = self.board.get_task(task_id)
        emit_activity(event_type, title=description, task_id=task_id,
                      conversation_id=task.session_id,
                      status="failed" if target == Phase.BLOCKED else "success",
                      operation_id=f"{task_id}:{target.value}",
                      metadata={"phase": target.value, "mode": state["mode"],
                                "revision": state["revision"], "step_id": state.get("step_id", "")})

    def start(
        self, session_id: str, goal: str, *, mode: str = "plan", repair: bool = False,
        task_id: str = "", no_code: bool = False, save: bool = True,
        output: str | None = None, question_limit: int | None = None, llm=None,
        skill_names: tuple[str, ...] = (), read_only: bool = False,
        execution_policy: dict | None = None,
        reviewer_id: str = "",
    ) -> CommandResult:
        goal = str(redact_secrets(goal)).strip()
        policy = TaskExecutionPolicy.model_validate(execution_policy or {})
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
            activity = current_activity_context()
            reservation = self.lane.reserve(
                normalized_intent=f"{mode}: {goal}",
                lane_id=LaneId.RESEARCH,
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
        state = {
            "schema_version": 1, "mode": mode, "phase": Phase.UNDERSTANDING.value,
            "goal": goal, "revision": 1, "constraints": {}, "evidence": [],
            "repair": repair and not read_only, "no_code": no_code or read_only, "save": save,
            "output": output, "question_limit": question_limit, "questions_asked": 0,
            "skills": list(skill_names), "done_steps": {}, "files_changed": [],
            "execution_policy": policy.model_dump(mode="json"),
            "reviewer_id": reviewer_id or getpass.getuser(),
        }
        self._persist(task_id, state)
        return self._drive(task_id, state, service=self._service(llm=llm))

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
            protected_text=activity.protected_text if activity else (),
            execution_lane=execution.owning_lane.value,
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

    def _drive(self, task_id: str, state: dict, *, service: PlanService) -> CommandResult:
        try:
            with self._activity(task_id, state, read_only=True):
                return self._plan_or_investigate(task_id, state, service)
        except Exception as exc:
            return self._block(task_id, state, str(redact_secrets(str(exc))))

    def _plan_or_investigate(self, task_id, state, service):
        from mana_agent.multi_agent.runtime.tool_call_gate import execute_host_operation

        config = self.gateway._autonomous_controller.config
        for _ in range(config.max_model_turns):
            self._check_live(task_id)
            if state.get("decision_count", 0) >= config.max_model_turns:
                break
            state["decision_count"] = state.get("decision_count", 0) + 1
            self._persist(task_id, state)
            decision = service.next_decision(state["mode"], {
                "request": state["goal"], "constraints": state["constraints"], "skills": state["skills"],
                "evidence": state["evidence"][-12:], "revision_request": state.get("revision_request", ""),
                "repair_authorized": state["repair"],
            })
            self._check_live(task_id)
            allowed = ({"inspect", "clarify", "draft", "block"} if state["mode"] == "plan"
                       else {"inspect", "clarify", "diagnose", "repair", "research", "block"})
            if decision.action not in allowed:
                raise ValueError("The model selected an action outside the active mode")
            if decision.action == "inspect":
                request = decision.evidence
                evidence = state["evidence"]
                prior = {item["id"] for item in evidence}
                highest = max((item["level"] for item in evidence), default=0)
                initial_broad = not evidence and bool(request.initial_repository_scope)
                if not initial_broad and (request.level > highest + 1 or (
                    request.level > 1 and (
                        not request.prior_evidence or not set(request.prior_evidence).issubset(prior)
                        or (request.level > highest and not any(
                            item["level"] == highest and item["id"] in request.prior_evidence for item in evidence
                        ))
                    )
                )):
                    raise ValueError("Investigation expansion requires observed evidence from the narrower scope")
                if any(item["fingerprint"] == request.fingerprint for item in evidence):
                    raise ValueError("The model selected an unchanged investigation operation")
                if len(evidence) >= config.max_tool_calls:
                    raise RuntimeError("Repository investigation reached the configured tool budget")
                self._transition(task_id, state, Phase.INVESTIGATING, str(redact_secrets(request.reason))[:240])
                args = ({"path": request.paths[0], "start_line": request.start_line,
                         "end_line": request.end_line} if request.tool == "read_file"
                        else {"query": request.query, "paths": list(request.paths), "glob": request.glob})
                result = execute_host_operation(
                    root=self.gateway.root, task_id=task_id, tool_name=request.tool,
                    arguments=args, effect="read_only", implementation=lambda _: service.inspect(request),
                )
                evidence.append({"id": f"evidence-{len(evidence) + 1}", "fingerprint": request.fingerprint,
                                 "level": request.level, "result": redact_secrets(result)})
                self._persist(task_id, state)
                continue
            if decision.action == "clarify":
                if set(q.field_id for q in decision.questions).intersection(state["constraints"]):
                    raise ValueError("The model asked for an already resolved decision")
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
            if decision.action == "research":
                waiting = self._research(task_id, state, decision.debugging_report)
                if waiting:
                    return waiting
                continue
            if decision.action == "diagnose":
                if decision.cause_evidence and not set(decision.cause_evidence).issubset({ob["id"] for ob in state["evidence"]}):
                    raise ValueError("Diagnosis referenced unknown evidence")
                text = "\n\n".join(s for s in (decision.root_cause, decision.uncertainty) if s)
                return self._complete(task_id, state, text)
            if decision.action == "repair":
                if not state["repair"]:
                    raise PermissionError("This debugging request only authorizes diagnosis")
                if not set(decision.cause_evidence).issubset({ob["id"] for ob in state["evidence"]}):
                    raise ValueError("Repair referenced unknown evidence")
                if not any(step.changes for step in decision.draft.steps) or not any(
                    check.expected == "pass" for step in decision.draft.steps for check in step.verification
                ):
                    raise ValueError("Repair requires a scoped fix and a regression check")
                state["root_cause"] = str(redact_secrets(decision.root_cause))
                state["uncertainty"] = str(redact_secrets(decision.uncertainty))
            return self._draft(task_id, state, decision.draft, service)
        return self._block(task_id, state, "The configured model decision budget was exhausted.")

    def _research(self, task_id, state, report):
        from mana_agent.execution.continuation_decision import ContinuationAction
        from mana_agent.gateway.lanes import validate_tool_permission

        if not state["evidence"]:
            raise ValueError("Research requires a local observation about the dependency")
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
            task_id=task_id, root=self.gateway.root, attempt_id=execution.supervisor_attempt_id,
            lease_token=execution.supervisor_lease_token, authorize=authorize,
            evaluator=self.gateway._continuation_evaluator, settings=settings,
        )
        status = {"debugging_report": redact_secrets(report.model_dump(mode="json")), "status": "waiting"}
        decision = runtime.decide(goal=state["goal"], result=status)
        if decision.action == ContinuationAction.WAIT_FOR_USER:
            self._transition(task_id, state, Phase.CLARIFICATION_REQUIRED)
            questions = [{"field_id": decision.decision_id, "prompt": decision.user_message or decision.reason,
                          "choices": {}, "recommended_choice": "", "allow_free_form": True}]
            return self._describe_item(task_id, state, self._wait(task_id, state, questions=questions))
        if decision.action == ContinuationAction.BLOCKED:
            return self._block(task_id, state, decision.reason)
        evidence = {"id": f"evidence-{len(state['evidence']) + 1}", "fingerprint": canonical_digest(status),
                    "level": 0, "result": runtime.context(status)}
        if any(row["fingerprint"] == evidence["fingerprint"] and row["result"] == evidence["result"]
               for row in state["evidence"]):
            raise RuntimeError("External investigation produced no new evidence")
        state["evidence"].append(evidence)
        self._persist(task_id, state)
        return None

    def _draft(self, task_id, state, draft, service):
        self._transition(task_id, state, Phase.DRAFTING_PLAN)
        task = self.board.get_task(task_id)
        root = self.gateway.root
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
        if state["mode"] == "debug":
            self._transition(task_id, state, Phase.APPROVED, "Evidence-supported repair is ready")
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
        context = {"task_id": task_id, "phase": state["phase"], "inbox_id": item.inbox_item_id,
                   "revision": state["revision"], "digest": item.action_digest}
        if item.request_type == InboxRequestType.APPROVAL:
            text = f"{item.summary}\nApprove revision {state['revision']} with `/approve {item.inbox_item_id}`, or use `/revise <changes>` or `/reject`."
            return CommandResult(status="input_required", message=text, data=context,
                                 next_prompt={"kind": "mode.approval", "preview": item.summary, **context})
        questions = item.minimal_context["questions"]
        lines = []
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

    def describe(self, task):
        state = task.mode_workflow
        inbox_id = state.get("inbox_id")
        if inbox_id and state["phase"] in {Phase.CLARIFICATION_REQUIRED, Phase.AWAITING_APPROVAL}:
            item = self.inbox.repository.get(inbox_id)
            if item.status in {InboxStatus.PENDING, InboxStatus.DELIVERED}:
                return self._describe_item(task.task_id, state, item)
        text = state.get("result") or f"{state['mode'].capitalize()} task {task.task_id} is {state['phase'].replace('_', ' ')}."
        if self._can_retry_investigation(state):
            text += f"\n\nResume the investigation with `/task resume {task.task_id}`."
        if state.get("plan"):
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
                except (PermissionError, ValueError) as exc:
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
        result = self.inbox.respond(ResponseSubmission(
            inbox_item_id=item_id, operation=operation, actor_id=getpass.getuser() if actor_id is None else actor_id, channel=frontend,
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
        return CommandResult(status="success", message=f"Mode task {task_id} is still pending.", data={"task_id": task_id})

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
            self._transition(task_id, state, Phase.UNDERSTANDING)
            result = self._drive(task_id, state, service=self._service())
        else:
            result = self._execute(task_id, state, self._service())
        self._publish_result(task_id, result)
        return result

    @staticmethod
    def _can_retry_investigation(state: dict) -> bool:
        return state["phase"] == Phase.BLOCKED and not any(state.get(key) for key in (
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
            self._transition(task_id, state, Phase.CANCELLED)

    def _restart_draft(self, task_id, state, instructions):
        self._revise(task_id, state, instructions)
        return self._drive(task_id, state, service=self._service())

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
                result = self._drive(task_id, state, service=self._service())
            else:
                self._transition(task_id, state, Phase.CANCELLED, "User rejected the plan")
                self.lane.cancel_tree(task_id)
                result = CommandResult(status="success", message="Plan rejected.")
        elif item.status == InboxStatus.ANSWERED:
            answer = self.inbox.consume_for_agent(item.inbox_item_id, requesting_agent_id=_OWNER, task_id=task_id).answer
            state["constraints"].update(redact_secrets(answer))
            state["inbox_id"] = ""
            self._transition(task_id, state, Phase.UNDERSTANDING)
            result = self._drive(task_id, state, service=self._service())
        elif plan:
            state["approved_digest"] = plan.digest
            state["approval_id"] = item.inbox_item_id
            state["inbox_id"] = ""
            self._transition(task_id, state, Phase.APPROVED, f"Revision {plan.revision} approved")
            result = self._execute(task_id, state, self._service())
        else:
            return
        self._publish_result(task_id, result)

    def _publish_result(self, task_id, result):
        from mana_agent.chat.events import AssistantMessageEvent
        from mana_agent.chat.history import get_history

        task = self.board.get_task(task_id)
        turn_id = f"mode-{task_id}-{task.mode_workflow['revision']}-{task.mode_workflow['phase']}"
        self.gateway._append_session_message(task.session_id, role="assistant", content=result.message, turn_id=turn_id)
        get_history().add(AssistantMessageEvent(content=result.message, turn_id=turn_id))
        if result.next_prompt:
            emit_activity("mode.input_ready", title="Plan review or clarification available", task_id=task_id,
                          conversation_id=task.session_id, operation_id=result.data["inbox_id"],
                          metadata={"next_prompt": result.next_prompt})
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
                str(self.gateway.root) != plan.repository_root or PlanRevision.model_validate(saved["plan"]).digest != plan.digest):
            raise PermissionError("The task, revision, or workspace changed")
        if saved["execution_policy"] != plan.execution_policy:
            raise PermissionError("The approved execution policy changed")
        if self._checkout_identity(self.gateway.root) != plan.base_checkout:
            raise PermissionError("The repository branch or base commit changed. Revise and approve the plan again.")
        session = self.gateway._workspaces.store.get_session(plan.session_id)
        if session.workspace_id != plan.workspace_id or session.primary_repository_id != plan.repository_id:
            raise PermissionError("The session workspace no longer matches the reviewed checkout")
        if state["mode"] != "plan":
            if not state["repair"]:
                raise PermissionError("Debug repair was not authorized")
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
        except Exception as exc:
            return self._block(task_id, state, str(redact_secrets(str(exc))))

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
        latest_digests = {}
        for step in plan.draft.steps:
            latest_digests.update(completed.get(step.step_id, {}).get("files", {}))
        latest_digests.update(state.get("in_flight_mutation", {}).get("files", {}))
        if any(self._file_digest(self.gateway.root, path) != digest for path, digest in latest_digests.items()):
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
            child = self._step_task(task_id, step, plan)
            self._start_step(task_id, child.task_id, plan, step, state)
            try:
                with self._activity(child.task_id, state, read_only=not step.changes):
                    self._execute_step(task_id, child.task_id, state, plan, step, service, expected_files)
                self._validate_approval(task_id, state, plan)
                written = list(state.get("in_flight_mutation", {}).get("files", {})) if step.changes else []
                finished = self.lane.finish(child.task_id, changed_files=written,
                    verification_state={"goal_satisfied": True, "mode_step": step.step_id,
                                        "verification": state.get("verification", [])})
                if finished.state != LaneTaskState.COMPLETED:
                    raise RuntimeError(finished.error or "The supervisor has not verified this step")
                state["done_steps"][step.step_id] = {"signature": signature,
                    "files": {path: expected_files[path] for path in step.changes}}
                state.pop("in_flight_mutation", None)
                self._persist(task_id, state)
            except ModeBatchWaiting:
                raise
            except Exception as exc:
                lane_child = self.lane.inspect_task(child.task_id)
                if lane_child.state in {LaneTaskState.RUNNING, LaneTaskState.VERIFYING}:
                    self.lane.mark_blocked(child.task_id, reason=str(redact_secrets(str(exc))))
                raise
            done.add(step.step_id)
            revision_reason = self._steering(task_id, state, plan, step, service)
            if revision_reason:
                return self._restart_draft(task_id, state, revision_reason)
        passed = sum(bool(check["passed"]) for check in state.get("verification", []))
        message = f"Revision {plan.revision} completed. Verified {passed} command(s)."
        if state["mode"] == "debug":
            message = f"Root cause: {state['root_cause']}\n\n{message}"
            if state.get("uncertainty"):
                message += "\n\nRemaining uncertainty: " + state["uncertainty"]
        if state["files_changed"]:
            message += "\n\nChanged files: " + ", ".join(state["files_changed"])
        return self._complete(task_id, state, message)

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
                parent_task_id=task_id, taskboard_task_id=child_id, target_files=list(step.changes),
            )
        if step.changes:
            policy = TaskExecutionPolicy.model_validate(state["execution_policy"])
            if policy.required:
                def bind_policy(task):
                    task.execution_policy = policy
                supervisor.store.update_task(child_id, bind_policy)
        self.lane.start(reservation)

    def _execute_step(self, task_id, child_id, state, plan, step, service, expected_files):
        applied = state.get("in_flight_mutation", {})
        if applied.get("step") == step.step_id and applied.get("action_id"):
            from mana_agent.transactional_actions.runtime import default_action_gateway
            action = default_action_gateway(self.gateway.root).store.get_action(applied["action_id"])
            if (action is None or action.state.value != "committed" or any(
                self._file_digest(self.gateway.root, path) != digest for path, digest in applied["files"].items()
            )):
                raise PermissionError("The applied patch could not be reconciled")
            expected_files.update(applied["files"])
        elif step.changes:
            self._apply_step(task_id, child_id, state, plan, step, service, expected_files)
        if step.verification:
            self._transition(task_id, state, Phase.VERIFYING, f"Verifying {step.step_id}")
            for command in step.verification:
                self._validate_approval(task_id, state, plan)
                self._check_live(child_id)
                self._verify(task_id, command, state)

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
        root = self.gateway.root
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
            patch, approval_id = self._approved_patch(pending)
        else:
            patch = self._propose_patch(task_id, child_id, state, plan, step, service)
        self._commit_patch(task_id, child_id, state, plan, step, service, expected_files, patch, approval_id)

    def _propose_patch(self, task_id, child_id, state, plan, step, service):
        root = self.gateway.root
        prompt = (
            "Propose the minimal repository patch for the given step. Your turn is read-only. "
            "Return exactly a JSON object with a single patch string using the *** Begin Patch format. "
            "Use separate Add File, Update File, or Delete File operations. "
            "Only the selected step is executable. Do not run project commands. "
            "If scope is insufficient, return an empty patch string and stop.\n\n"
            + json.dumps({"step": step.model_dump(mode="json"), "scope": plan.draft.scope,
                          "exclusions": plan.draft.exclusions, "user_decisions": plan.constraints,
                          "evidence": state["evidence"][-8:] if state["mode"] == "debug" else [],
                          "steering": state.get("guidance", []),
                          "root_cause": state.get("root_cause", "")}, ensure_ascii=False)
        )
        proposal = self.gateway._coding_agent.generate(
            prompt, execution_mode="read_only", gateway_task_id=child_id,
            turn_id=f"{task_id}:r{plan.revision}:{step.step_id}", dispatch_source="mode_workflow",
            coding_mode_proposal=True, coding_mode_files=list(step.changes),
            coding_mode_root=str(root),
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

    def _approved_patch(self, pending):
        from mana_agent.transactional_actions.runtime import default_action_gateway

        gateway = default_action_gateway(self.gateway.root)
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
        from mana_agent.multi_agent.runtime.tool_call_gate import execute_host_operation

        root = self.gateway.root
        if redact_secrets(patch) != patch:
            raise ValueError("Coding proposal contains protected material")
        operations = validate_step_patch(root, step, patch)
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

    def resume_batch(self, task_id, child_id, operation_id, run):
        supervisor = self.lane.execution_supervisor
        state = self.board.get_task(task_id).mode_workflow
        plan = PlanRevision.model_validate(state["plan"])
        operation = run.get("pending_operation") or {}
        if (state.get("batch_child") != child_id or operation.get("operation_id") != operation_id
                or run["resume"].get("digest") != plan.digest):
            raise PermissionError("Batch wake does not match the approved plan step")
        manager = self.gateway._autonomous_controller.batch_manager
        job = manager.store.get_job(operation["batch_id"])
        completed_at = datetime.fromtimestamp(job.completed_at, timezone.utc) if job and job.completed_at else None
        self.lane.ensure_recoverable_execution(task_id)
        self.lane.ensure_recoverable_execution(child_id)
        with supervisor.store.locked():
            parent = supervisor.get_task(task_id)
            if parent.state != ExecutionState.WAITING or parent.wake_up_reference != operation_id:
                return
            supervisor.resume_from_batch(task_id, operation_id=operation_id, provider_completed_at=completed_at)
            supervisor.resume_from_batch(child_id, operation_id=operation_id, provider_completed_at=completed_at)
        self.gateway._bind_runtime_session(plan.session_id)
        lane = self.lane.inspect_task(task_id)
        lane.supervisor_attempt_id = lane.supervisor_lease_token = ""
        self.lane.start(LaneReservation(lane))
        self._publish_result(task_id, self._execute(task_id, state, self._service()))

    def _verify(self, task_id: str, command: VerificationCommand, state: dict):
        operation_id = f"{task_id}:verify:{len(state.get('verification', [])) + 1}"
        emit_activity("coding.progress", task_id=task_id, title="Running verification command",
                      summary=f"{command.argv!r} in {command.cwd}",
                      status="running", operation_id=operation_id)
        outcome = asyncio.run(run_isolated_verification(self.gateway.root, command, task_id=task_id))
        state.setdefault("verification", []).append(outcome)
        self._persist(task_id, state)
        emit_activity("coding.progress", task_id=task_id, title="Verification command",
                      summary=f"{command.argv!r} · {outcome['exit_code']}",
                      status="success" if outcome["passed"] else "failed",
                      operation_id=operation_id)
        if not outcome["passed"]:
            raise RuntimeError(f"Verification failed for {command.argv!r}: {outcome['output']}")

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
        finished = self.lane.finish(task_id, changed_files=state["files_changed"],
                         verification_state={"goal_satisfied": True, "answer": message,
                                             "verification": state.get("verification", [])})
        if finished.state != LaneTaskState.COMPLETED:
            return self._block(task_id, state, finished.error or "Supervisor verification remains incomplete")
        state["result"] = message
        self._transition(task_id, state, Phase.COMPLETED)
        return CommandResult(status="success", message=message,
                             data={"task_id": task_id, "phase": state["phase"], "changed_files": state["files_changed"]})

    def _block(self, task_id, state, message):
        message = str(redact_secrets(message))
        state["result"] = message
        durable = self.lane.execution_supervisor.get_task(task_id)
        if durable.cancellation_status is None and durable.state not in {ExecutionState.COMPLETED, ExecutionState.CANCELLED}:
            self._transition(task_id, state, Phase.BLOCKED)
            if durable.state == ExecutionState.RUNNING:
                self.lane.mark_blocked(task_id, reason=message)
        return CommandResult(status="error", message=message,
                             data={"task_id": task_id, "phase": state["phase"]})


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
        raise RuntimeError(f"Verification pending. An isolated Docker environment is required for {command.argv!r}.")
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
        raise RuntimeError(f"Verification pending. Docker image {image} is unavailable locally for {command.argv!r}.")
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
