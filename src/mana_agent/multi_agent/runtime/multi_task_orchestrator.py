"""Structured decomposition and bounded DAG execution for compound gateway turns."""

from __future__ import annotations

import contextvars
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from mana_agent.llm.messages import UserMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mana_agent.evals.ids import stable_hash
from mana_agent.evals.recorder import record_current
from mana_agent.multi_agent.core.types import TaskStatus
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.utils.text import extract_model_text


MAX_MULTI_TASK_CHILDREN = 12


def _verified_completion(task) -> bool:
    evidence = task.supervisor_verification_evidence
    return (task.status == TaskStatus.DONE and task.supervisor_state == "completed"
            and task.verification_status == "passed" and bool(evidence.get("result_id"))
            and bool(evidence.get("verification")))


@contextmanager
def taskboard_wakeup(root_task_id):
    """Wake the existing executor on relevant task/evidence events, coalescing updates."""
    from mana_agent.services.execution_event_hub import get_execution_event_hub
    wakeup = Event()
    def changed(event):
        if event.get("root_task_id") == root_task_id or event.get("task_id") == root_task_id:
            kind = event.get("type") or event.get("event_type")
            if kind in {"task.updated", "task.created", "task.cancelled"} or (event.get("metadata") or {}).get("evidence_id"):
                wakeup.set()
    unsubscribe = get_execution_event_hub().subscribe_all(changed)
    try:
        yield wakeup
    finally:
        unsubscribe()


class MultiTaskError(RuntimeError):
    """A compound-task model decision was missing, invalid, or unsafe."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MultiTaskItem(_StrictModel):
    local_id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    request: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(min_length=1)
    preferred_parallelism: Literal["parallel", "sequential", "automatic"] = "automatic"
    reason: str = Field(min_length=1)
    task_kind: Literal["general", "investigation", "understanding", "implementation", "integration", "verification"] = "general"
    investigation_scope: list[str] = Field(default_factory=list)
    evidence_requirements: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_item(self) -> "MultiTaskItem":
        self.local_id = self.local_id.strip()
        self.title = self.title.strip()
        self.request = self.request.strip()
        self.reason = self.reason.strip()
        self.depends_on = [item.strip() for item in self.depends_on]
        self.acceptance_criteria = [item.strip() for item in self.acceptance_criteria if item.strip()]
        if not all((self.local_id, self.title, self.request, self.reason)) or not self.acceptance_criteria:
            raise ValueError("child fields and acceptance criteria must be meaningful")
        from mana_agent.repository_evidence.models import EvidenceRequirement, relative_source
        for path in self.investigation_scope:
            relative_source(path)
        for requirement in self.evidence_requirements:
            EvidenceRequirement.model_validate(requirement)
        if self.task_kind in {"investigation", "understanding"} and (not self.investigation_scope or not self.evidence_requirements):
            raise ValueError("Investigation and understanding tasks need bounded scope and evidence requirements")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError(f"task {self.local_id!r} contains duplicate dependencies")
        if self.local_id in self.depends_on:
            raise ValueError(f"task {self.local_id!r} cannot depend on itself")
        return self


class MultiTaskPlan(_StrictModel):
    goal: str = Field(min_length=1)
    tasks: list[MultiTaskItem] = Field(min_length=2, max_length=MAX_MULTI_TASK_CHILDREN)
    final_acceptance_criteria: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_graph(self) -> "MultiTaskPlan":
        identifiers = [task.local_id for task in self.tasks]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("multi-task local IDs must be unique")
        known = set(identifiers)
        for task in self.tasks:
            unknown = set(task.depends_on) - known
            if unknown:
                raise ValueError(
                    f"task {task.local_id!r} references unknown dependencies: {', '.join(sorted(unknown))}"
                )
        visiting: set[str] = set()
        visited: set[str] = set()
        graph = {task.local_id: task.depends_on for task in self.tasks}

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError("multi-task dependency graph contains a cycle")
            if node in visited:
                return
            visiting.add(node)
            for dependency in graph[node]:
                visit(dependency)
            visiting.remove(node)
            visited.add(node)

        for identifier in identifiers:
            visit(identifier)
        self.goal = self.goal.strip()
        self.reason = self.reason.strip()
        self.final_acceptance_criteria = [
            item.strip() for item in self.final_acceptance_criteria if item.strip()
        ]
        if not self.goal or not self.reason or not self.final_acceptance_criteria:
            raise ValueError("goal, reason, and final acceptance criteria are required")
        return self


MULTI_TASK_DECOMPOSITION_PROMPT = """You are Mana-Agent's compound-task planner.
Return only a strict MultiTaskPlan for the complete user request. Create two to twelve meaningful
children, each with its own execution lifecycle and acceptance criteria. Preserve every user
constraint. Dependencies must reference local_id values and only express genuine ordering. Keep
independent work independent. Do not combine unrelated operations. Do not invent credentials,
recipients, URLs, repositories, approvals, destructive actions, or missing values. Each child will
be independently entry-routed and capability-checked after this decision; do not choose routes or
tools here. Documentation inspection, integration persistence, and execution against that same API
form one atomic child rather than independently completable children. A child is complete only when
its acceptance criteria are met; a prose explanation that an intended action was not executed is a
blocker, not completion. Investigation and understanding are optional independent task_kind values with investigation_scope and
explicit evidence_requirements. They are never mandatory stages or prerequisites for unrelated work.
This is decomposition only, never execution or a prose answer.
"""


@dataclass(slots=True)
class MultiTaskChildResult:
    local_id: str
    task_id: str
    title: str
    route: str
    status: str
    result: str = ""
    blocker: str = ""
    verification_status: str = ""
    changed_files: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    approval_request_ids: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


class MultiTaskOrchestrator:
    """Uses the existing TaskBoard and a bounded executor to run a validated DAG."""

    def __init__(self, *, llm: Any, taskboard: TaskBoard, maximum_concurrency: int = 4) -> None:
        self.llm = llm
        self.taskboard = taskboard
        self.maximum_concurrency = max(1, int(maximum_concurrency))

    def decompose(self, *, user_prompt: str, context: dict[str, Any]) -> MultiTaskPlan:
        if self.llm is None or not callable(getattr(self.llm, "generate_sync", None)):
            raise MultiTaskError(
                "Model decision failed: multi_task_decomposition. No fallback action was executed. "
                "Reason: decomposition model is unavailable."
            )
        request = {"user_prompt": user_prompt, "context": context, "maximum_children": MAX_MULTI_TASK_CHILDREN}
        messages = [
            SystemMessage(content=MULTI_TASK_DECOMPOSITION_PROMPT),
            UserMessage(content=json.dumps(request, ensure_ascii=False, sort_keys=True)),
        ]
        started = time.perf_counter()
        try:
            plan = self.llm.decide_sync(messages, MultiTaskPlan)
        except Exception as exc:
            record_current(
                "model.call.failed",
                {"boundary": "multi_task_decomposition", "error_type": type(exc).__name__, "error": str(exc)},
            )
            raise MultiTaskError(
                "Model decision failed: multi_task_decomposition. No fallback action was executed. "
                f"Reason: {exc}"
            ) from exc
        record_current(
            "model.decision",
            {
                "boundary": "multi_task_decomposition",
                "prompt_template": "MULTI_TASK_DECOMPOSITION_PROMPT",
                "prompt_hash": stable_hash(MULTI_TASK_DECOMPOSITION_PROMPT),
                "request_hash": stable_hash(request),
                "response": plan.model_dump(mode="json"),
                "usage": None,
                "latency_seconds": time.perf_counter() - started,
            },
        )
        return plan

    def create_children(self, *, root_task_id: str, plan: MultiTaskPlan) -> dict[str, str]:
        root = self.taskboard.get_task(root_task_id)
        existing = dict(root.decomposition_id_map)
        mapping: dict[str, str] = {}
        for item in plan.tasks:
            existing_id = existing.get(item.local_id)
            if existing_id and existing_id in self.taskboard.tasks:
                mapping[item.local_id] = existing_id
                continue
            child = self.taskboard.create_child_task(
                root_task_id,
                title=item.title,
                user_request=item.request,
                owner_agent_id=root.owner_agent_id,
                acceptance_criteria=item.acceptance_criteria,
                plan=[item.reason],
                decomposition_local_id=item.local_id,
                preferred_parallelism=item.preferred_parallelism,
            )
            self.taskboard.update_task(child.task_id, task_kind=item.task_kind,
                investigation_scope=item.investigation_scope, evidence_requirements=item.evidence_requirements)
            mapping[item.local_id] = child.task_id
        for item in plan.tasks:
            child = self.taskboard.get_task(mapping[item.local_id])
            child.depends_on = [mapping[dependency] for dependency in item.depends_on]
        root.acceptance_criteria = list(plan.final_acceptance_criteria)
        root.plan = [plan.reason]
        root.decomposition_id_map.update(mapping)
        root.child_task_ids = list(dict.fromkeys([*root.child_task_ids, *mapping.values()]))
        self.taskboard.save()
        record_current(
            "multi_task.children_persisted",
            {"root_task_id": root_task_id, "local_id_map": mapping},
        )
        return mapping

    def execute(
        self,
        *,
        root_task_id: str,
        plan: MultiTaskPlan,
        execute_child: Callable[[MultiTaskItem, str], MultiTaskChildResult],
        is_cancelled: Callable[[], bool] | None = None,
    ) -> list[MultiTaskChildResult]:
        mapping = self.create_children(root_task_id=root_task_id, plan=plan)
        by_id = {item.local_id: item for item in plan.tasks}
        results: dict[str, MultiTaskChildResult] = {}
        running: dict[Future[MultiTaskChildResult], str] = {}
        pending = set(by_id)
        persisted_statuses = {
            TaskStatus.DONE: "completed",
            TaskStatus.SKIPPED: "skipped",
            TaskStatus.CANCELLED: "cancelled",
            TaskStatus.FAILED: "failed",
            TaskStatus.BLOCKED: "blocked",
            TaskStatus.WAITING_FOR_TOOLS: "awaiting_approval",
        }
        for local_id in list(pending):
            task = self.taskboard.get_task(mapping[local_id])
            status = persisted_statuses.get(task.status)
            if status is None:
                continue
            if status == "completed" and not _verified_completion(task):
                status = "blocked"
            results[local_id] = MultiTaskChildResult(
                local_id=local_id,
                task_id=task.task_id,
                title=task.title,
                route=task.entry_route,
                status=status,
                result=task.result_summary,
                blocker="; ".join(task.blockers),
                verification_status=task.verification_status,
                changed_files=list(task.files_touched),
                artifacts=list(task.output_artifacts),
                approval_request_ids=list(task.approval_request_ids),
            )
            pending.remove(local_id)

        with taskboard_wakeup(root_task_id) as wakeup, ThreadPoolExecutor(max_workers=self.maximum_concurrency, thread_name_prefix="mana-multi-task") as pool:
            while pending or running:
                wakeup.clear()
                self.taskboard.refresh()
                current_tasks = self.taskboard.list_tasks(root_task_id=root_task_id)
                reverse = {task_id: local_id for local_id, task_id in mapping.items()}
                for task in current_tasks:
                    if task.task_id == root_task_id or task.parent_task_id != root_task_id:
                        continue
                    local_id = reverse.get(task.task_id) or task.decomposition_local_id or task.task_id
                    if local_id not in by_id:
                        mapping[local_id] = task.task_id
                        reverse[task.task_id] = local_id
                        by_id[local_id] = MultiTaskItem(local_id=local_id, title=task.title, request=task.user_request,
                            acceptance_criteria=task.acceptance_criteria, reason="Model-directed TaskBoard update")
                        pending.add(local_id)
                for local_id in list(pending):
                    task = self.taskboard.get_task(mapping[local_id])
                    unknown = [dep for dep in task.depends_on if dep not in reverse]
                    if unknown:
                        raise MultiTaskError("A dynamic task dependency is outside this admitted task graph")
                    by_id[local_id].depends_on = [reverse[dep] for dep in task.depends_on]
                    if task.status == TaskStatus.DONE:
                        verified = _verified_completion(task)
                        results[local_id] = MultiTaskChildResult(local_id, task.task_id, task.title, task.entry_route,
                            "completed" if verified else "blocked", result=task.result_summary,
                            blocker="" if verified else "Saved task lacks durable supervisor verification")
                        pending.remove(local_id)
                if is_cancelled and is_cancelled():
                    for local_id in sorted(pending):
                        task_id = mapping[local_id]
                        task = self.taskboard.get_task(task_id)
                        if task.status not in {TaskStatus.CANCELLED, TaskStatus.DONE}:
                            self.taskboard.update_status(task_id, TaskStatus.CANCELLED)
                        results[local_id] = MultiTaskChildResult(
                            local_id, task_id, by_id[local_id].title, task.entry_route, "cancelled",
                            blocker="root task was cancelled",
                        )
                    pending.clear()
                made_progress = False
                for local_id in list(pending):
                    item = by_id[local_id]
                    # If any dependency is still in flight or not yet evaluated, wait
                    if any(dependency not in results for dependency in item.depends_on):
                        continue
                    # If an explicit cancellation occurred, mark cancelled
                    cancelled_deps = [
                        dep for dep in item.depends_on
                        if dep in results and results[dep].status == "cancelled"
                    ]
                    if cancelled_deps:
                        reason = "cancelled by prerequisite(s): " + ", ".join(cancelled_deps)
                        task_id = mapping[local_id]
                        self.taskboard.update_status(task_id, TaskStatus.CANCELLED, reason=reason)
                        results[local_id] = MultiTaskChildResult(
                            local_id, task_id, item.title, "", "cancelled", blocker=reason
                        )
                        pending.remove(local_id)
                        made_progress = True
                        continue
                    failed_deps = [
                        dep for dep in item.depends_on
                        if dep in results and results[dep].status in {"failed", "blocked"}
                    ]
                    if failed_deps:
                        reason = "blocked by prerequisite failure(s): " + ", ".join(failed_deps)
                        task_id = mapping[local_id]
                        self.taskboard.update_status(task_id, TaskStatus.BLOCKED, reason=reason)
                        results[local_id] = MultiTaskChildResult(
                            local_id, task_id, item.title, "", "blocked", blocker=reason
                        )
                        pending.remove(local_id)
                        made_progress = True
                        continue
                    if any(results[dep].status not in {"completed", "skipped"} for dep in item.depends_on):
                        continue
                    if len(running) >= self.maximum_concurrency:
                        break
                    # ThreadPoolExecutor workers do not inherit ContextVars.
                    # Propagate the parent turn context so authenticated
                    # computer-client identity, evals, event sinks, and other
                    # process-local scopes remain available to child routes.
                    parent_context = contextvars.copy_context()
                    future = pool.submit(
                        parent_context.run,
                        execute_child,
                        item,
                        mapping[local_id],
                    )
                    future.add_done_callback(lambda _: wakeup.set())
                    running[future] = local_id
                    pending.remove(local_id)
                    made_progress = True
                if running:
                    if not any(future.done() for future in running):
                        wakeup.wait(timeout=30)
                    done = {future for future in running if future.done()}
                    for future in done:
                        local_id = running.pop(future)
                        try:
                            result = future.result()
                            task = self.taskboard.get_task(mapping[local_id])
                            if result.status == "completed" and not _verified_completion(task):
                                result.status = "blocked"
                                result.blocker = "Worker completion lacks TaskBoard and supervisor verification"
                            results[local_id] = result
                        except Exception as exc:
                            task_id = mapping[local_id]
                            task = self.taskboard.get_task(task_id)
                            if task.status not in {TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.CANCELLED}:
                                self.taskboard.update_status(task_id, TaskStatus.FAILED, reason=str(exc))
                            results[local_id] = MultiTaskChildResult(
                                local_id, task_id, by_id[local_id].title, task.entry_route, "failed",
                                blocker=str(exc),
                            )
                elif pending and not made_progress:
                    # The only valid no-progress state is a prerequisite that is
                    # intentionally waiting for a child-scoped approval. Persist
                    # dependents as queued so a resume can execute them without
                    # recreating or prematurely blocking the child.
                    for local_id in sorted(pending):
                        task_id = mapping[local_id]
                        task = self.taskboard.get_task(task_id)
                        if task.status == TaskStatus.NEW:
                            self.taskboard.update_status(task_id, TaskStatus.ROUTED)
                            self.taskboard.update_status(task_id, TaskStatus.QUEUED)
                        results[local_id] = MultiTaskChildResult(
                            local_id=local_id,
                            task_id=task_id,
                            title=by_id[local_id].title,
                            route=task.entry_route,
                            status="queued",
                            blocker="waiting for prerequisite approval or completion",
                        )
                    pending.clear()
                complete = sum(result.status in {"completed", "skipped"} for result in results.values())
                self.taskboard.update_orchestration(
                    root_task_id,
                    aggregate_progress=f"{complete}/{len(by_id)} completed",
                )
        return [results[local_id] for local_id in by_id if local_id in results]


__all__ = [
    "MAX_MULTI_TASK_CHILDREN",
    "MULTI_TASK_DECOMPOSITION_PROMPT",
    "MultiTaskChildResult",
    "MultiTaskError",
    "MultiTaskItem",
    "MultiTaskOrchestrator",
    "MultiTaskPlan",
]
