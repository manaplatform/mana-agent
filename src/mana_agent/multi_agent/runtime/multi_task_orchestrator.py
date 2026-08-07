"""Structured decomposition and bounded DAG execution for compound gateway turns."""

from __future__ import annotations

import contextvars
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mana_agent.evals.ids import stable_hash
from mana_agent.evals.recorder import record_current
from mana_agent.multi_agent.core.types import TaskStatus
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard


MAX_MULTI_TASK_CHILDREN = 12


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
blocker, not completion. This is decomposition only, never execution or a prose answer.
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
        if self.llm is None or not callable(getattr(self.llm, "invoke", None)):
            raise MultiTaskError(
                "Model decision failed: multi_task_decomposition. No fallback action was executed. "
                "Reason: decomposition model is unavailable."
            )
        request = {"user_prompt": user_prompt, "context": context, "maximum_children": MAX_MULTI_TASK_CHILDREN}
        messages = [
            SystemMessage(content=MULTI_TASK_DECOMPOSITION_PROMPT),
            HumanMessage(content=json.dumps(request, ensure_ascii=False, sort_keys=True)),
        ]
        started = time.perf_counter()
        try:
            structured = getattr(self.llm, "with_structured_output", None)
            if callable(structured):
                response = structured(MultiTaskPlan, method="json_schema", strict=True).invoke(messages)
                plan = MultiTaskPlan.model_validate(response)
            else:
                response = self.llm.invoke(messages)
                raw = getattr(response, "content", response)
                if isinstance(raw, list):
                    raw = " ".join(
                        str(item.get("text", item)) if isinstance(item, dict) else str(item)
                        for item in raw
                    )
                text = str(raw).strip()
                if text.startswith("```"):
                    text = text.removeprefix("```json").removeprefix("```").strip().removesuffix("```").strip()
                start, end = text.find("{"), text.rfind("}")
                plan = MultiTaskPlan.model_validate_json(text[start : end + 1])
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
                "usage": getattr(response, "usage_metadata", None),
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
            mapping[item.local_id] = child.task_id
        for item in plan.tasks:
            child = self.taskboard.get_task(mapping[item.local_id])
            child.depends_on = [mapping[dependency] for dependency in item.depends_on]
        root.acceptance_criteria = list(plan.final_acceptance_criteria)
        root.plan = [plan.reason]
        root.decomposition_id_map = mapping
        root.child_task_ids = [mapping[item.local_id] for item in plan.tasks]
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

        with ThreadPoolExecutor(max_workers=self.maximum_concurrency, thread_name_prefix="mana-multi-task") as pool:
            while pending or running:
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
                    failed_dependencies = [
                        dependency for dependency in item.depends_on
                        if dependency in results
                        and results[dependency].status in {"failed", "blocked", "cancelled"}
                    ]
                    if failed_dependencies:
                        reason = "blocked by prerequisite(s): " + ", ".join(failed_dependencies)
                        task_id = mapping[local_id]
                        self.taskboard.update_status(task_id, TaskStatus.BLOCKED, reason=reason)
                        results[local_id] = MultiTaskChildResult(
                            local_id, task_id, item.title, "", "blocked", blocker=reason
                        )
                        pending.remove(local_id)
                        made_progress = True
                        continue
                    if any(
                        dependency not in results
                        or results[dependency].status not in {"completed", "skipped"}
                        for dependency in item.depends_on
                    ):
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
                    running[future] = local_id
                    pending.remove(local_id)
                    made_progress = True
                if running:
                    done, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
                    for future in done:
                        local_id = running.pop(future)
                        try:
                            results[local_id] = future.result()
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
                    aggregate_progress=f"{complete}/{len(plan.tasks)} completed",
                )
        return [results[item.local_id] for item in plan.tasks]


__all__ = [
    "MAX_MULTI_TASK_CHILDREN",
    "MULTI_TASK_DECOMPOSITION_PROMPT",
    "MultiTaskChildResult",
    "MultiTaskError",
    "MultiTaskItem",
    "MultiTaskOrchestrator",
    "MultiTaskPlan",
]
