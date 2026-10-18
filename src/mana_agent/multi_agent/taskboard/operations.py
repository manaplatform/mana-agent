"""Model-directed TaskBoard operations handler.

Enables the Autonomous Loop and registered agent tools to create, edit,
split, verify, reopen, and delete tasks dynamically, with TaskBoard remaining
the authoritative state workspace and single source of truth for task progress.
"""

from __future__ import annotations

import logging
from typing import Any
from pydantic import BaseModel, Field

from mana_agent.multi_agent.core.types import RiskLevel, TaskBoardItem, TaskStatus, VerificationResult
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard

logger = logging.getLogger(__name__)


class CreateTaskParams(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="")
    priority: int = Field(default=100, ge=1, le=1000)
    parent_task_id: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list)
    owner_agent_id: str | None = None


class UpdateTaskParams(BaseModel):
    task_id: str = Field(min_length=1)
    title: str | None = None
    description: str | None = None
    priority: int | None = None
    progress: str | None = None
    owner_agent_id: str | None = None
    plan: list[str] | None = None
    acceptance_criteria: list[str] | None = None
    blockers: list[str] | None = None
    crossed_off: bool | None = None


class DeleteTaskParams(BaseModel):
    task_id: str = Field(min_length=1)
    cascade: bool = Field(default=True)


class MarkCompletedParams(BaseModel):
    task_id: str = Field(min_length=1)
    evidence: str = Field(default="")
    cross_off: bool = Field(default=True)


class VerifyTaskParams(BaseModel):
    task_id: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    passed: bool = Field(default=True)
    verifier: str = Field(default="verifier")
    commands_run: list[str] = Field(default_factory=list)


class ReopenTaskParams(BaseModel):
    task_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class TaskBoardOperationsHandler:
    """Executes model-directed operations against the authoritative TaskBoard."""

    def __init__(self, taskboard: TaskBoard) -> None:
        self.taskboard = taskboard

    def create_task(self, params: CreateTaskParams | dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        data = kwargs if params is None else (params if isinstance(params, (CreateTaskParams, dict)) else kwargs)
        p = data if isinstance(data, CreateTaskParams) else CreateTaskParams.model_validate(data)
        task = self.taskboard.create_task(
            title=p.title,
            description=p.description,
            user_request=p.description,
            priority=p.priority,
            parent_task_id=p.parent_task_id,
            depends_on=p.depends_on,
            acceptance_criteria=p.acceptance_criteria,
            plan=p.plan,
            owner_agent_id=p.owner_agent_id,
        )
        return {
            "status": "created",
            "task_id": task.task_id,
            "title": task.title,
            "parent_task_id": task.parent_task_id,
            "root_task_id": task.root_task_id,
        }

    def add_subtask(
        self,
        parent_task_id: str,
        title: str,
        description: str = "",
        acceptance_criteria: list[str] | None = None,
        depends_on: list[str] | None = None,
    ) -> dict[str, Any]:
        subtask = self.taskboard.add_subtask(
            parent_task_id=parent_task_id,
            title=title,
            description=description,
            acceptance_criteria=acceptance_criteria,
            depends_on=depends_on,
        )
        return {
            "status": "subtask_created",
            "task_id": subtask.task_id,
            "parent_task_id": subtask.parent_task_id,
            "title": subtask.title,
        }

    def update_task(self, params: UpdateTaskParams | dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        data = kwargs if params is None else (params if isinstance(params, (UpdateTaskParams, dict)) else kwargs)
        p = data if isinstance(data, UpdateTaskParams) else UpdateTaskParams.model_validate(data)
        task = self.taskboard.update_task(
            p.task_id,
            title=p.title,
            description=p.description,
            priority=p.priority,
            progress=p.progress,
            owner_agent_id=p.owner_agent_id,
            plan=p.plan,
            acceptance_criteria=p.acceptance_criteria,
            blockers=p.blockers,
            crossed_off=p.crossed_off,
        )
        return {
            "status": "updated",
            "task_id": task.task_id,
            "title": task.title,
            "crossed_off": task.crossed_off,
            "status_value": task.status.value,
        }

    def delete_task(self, params: DeleteTaskParams | dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        data = kwargs if params is None else (params if isinstance(params, (DeleteTaskParams, dict)) else kwargs)
        p = data if isinstance(data, DeleteTaskParams) else DeleteTaskParams.model_validate(data)
        self.taskboard.delete_task(p.task_id, cascade=p.cascade)
        return {"status": "deleted", "task_id": p.task_id}

    def mark_completed(self, params: MarkCompletedParams | dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        data = kwargs if params is None else (params if isinstance(params, (MarkCompletedParams, dict)) else kwargs)
        p = data if isinstance(data, MarkCompletedParams) else MarkCompletedParams.model_validate(data)
        self.taskboard.mark_completed(p.task_id, evidence=p.evidence, cross_off=p.cross_off)
        task = self.taskboard.get_task(p.task_id)
        return {
            "status": "completed",
            "task_id": task.task_id,
            "crossed_off": task.crossed_off,
            "task_status": task.status.value,
        }

    def verify_task(self, params: VerifyTaskParams | dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        data = kwargs if params is None else (params if isinstance(params, (VerifyTaskParams, dict)) else kwargs)
        p = data if isinstance(data, VerifyTaskParams) else VerifyTaskParams.model_validate(data)
        v_result = self.taskboard.verify_task(
            p.task_id,
            evidence=p.evidence,
            verifier=p.verifier,
            commands_run=p.commands_run,
            passed=p.passed,
        )
        task = self.taskboard.get_task(p.task_id)
        return {
            "status": "verified" if p.passed else "verification_failed",
            "task_id": task.task_id,
            "verification_passed": v_result.passed,
            "verification_status": task.verification_status,
            "task_status": task.status.value,
            "summary": v_result.summary,
        }

    def reopen_task(self, params: ReopenTaskParams | dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        data = kwargs if params is None else (params if isinstance(params, (ReopenTaskParams, dict)) else kwargs)
        p = data if isinstance(data, ReopenTaskParams) else ReopenTaskParams.model_validate(data)
        self.taskboard.reopen(p.task_id, reason=p.reason)
        task = self.taskboard.get_task(p.task_id)
        return {
            "status": "reopened",
            "task_id": task.task_id,
            "task_status": task.status.value,
            "crossed_off": task.crossed_off,
        }

    def list_tasks(
        self,
        status: str | None = None,
        parent_task_id: str | None = None,
        root_task_id: str | None = None,
    ) -> list[dict[str, Any]]:
        target_status = TaskStatus(status) if status else None
        tasks = self.taskboard.list_tasks(
            status=target_status,
            parent_task_id=parent_task_id,
            root_task_id=root_task_id,
        )
        return [
            {
                "task_id": t.task_id,
                "parent_task_id": t.parent_task_id,
                "root_task_id": t.root_task_id,
                "title": t.title,
                "status": t.status.value,
                "crossed_off": getattr(t, "crossed_off", False),
                "verification_status": t.verification_status,
                "implementation_verified": t.implementation_verified,
                "priority": t.priority,
                "depends_on": list(t.depends_on),
                "child_task_ids": list(t.child_task_ids),
                "blockers": list(t.blockers),
                "acceptance_criteria": list(t.acceptance_criteria),
            }
            for t in tasks
        ]

    def get_progress(self, root_task_id: str | None = None) -> dict[str, Any]:
        return self.taskboard.get_progress(root_task_id=root_task_id)
