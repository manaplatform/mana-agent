"""Model-directed TaskBoard tools for Autonomous Loop agents.

Exposes controlled task management operations to the model during execution turns,
keeping TaskBoard as the authoritative state workspace for task progress.
"""

from __future__ import annotations

from typing import Any, Callable
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from mana_agent.multi_agent.taskboard.operations import (
    CreateTaskParams,
    DeleteTaskParams,
    MarkCompletedParams,
    ReopenTaskParams,
    TaskBoardOperationsHandler,
    UpdateTaskParams,
    VerifyTaskParams,
)
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard


def build_taskboard_tools(taskboard: TaskBoard) -> list[StructuredTool]:
    """Construct structured LangChain tools bound to an authoritative TaskBoard instance."""
    handler = TaskBoardOperationsHandler(taskboard)

    def create_task(
        title: str,
        description: str = "",
        priority: int = 100,
        parent_task_id: str | None = None,
        depends_on: list[str] | None = None,
        acceptance_criteria: list[str] | None = None,
        plan: list[str] | None = None,
    ) -> dict[str, Any]:
        return handler.create_task(
            CreateTaskParams(
                title=title,
                description=description,
                priority=priority,
                parent_task_id=parent_task_id,
                depends_on=depends_on or [],
                acceptance_criteria=acceptance_criteria or [],
                plan=plan or [],
            )
        )

    def update_task(
        task_id: str,
        title: str | None = None,
        description: str | None = None,
        priority: int | None = None,
        progress: str | None = None,
        plan: list[str] | None = None,
        acceptance_criteria: list[str] | None = None,
        blockers: list[str] | None = None,
        crossed_off: bool | None = None,
    ) -> dict[str, Any]:
        return handler.update_task(
            UpdateTaskParams(
                task_id=task_id,
                title=title,
                description=description,
                priority=priority,
                progress=progress,
                plan=plan,
                acceptance_criteria=acceptance_criteria,
                blockers=blockers,
                crossed_off=crossed_off,
            )
        )

    def delete_task(task_id: str, cascade: bool = True) -> dict[str, Any]:
        return handler.delete_task(DeleteTaskParams(task_id=task_id, cascade=cascade))

    def add_subtask(
        parent_task_id: str,
        title: str,
        description: str = "",
        acceptance_criteria: list[str] | None = None,
        depends_on: list[str] | None = None,
    ) -> dict[str, Any]:
        return handler.add_subtask(
            parent_task_id=parent_task_id,
            title=title,
            description=description,
            acceptance_criteria=acceptance_criteria,
            depends_on=depends_on,
        )

    def mark_completed(task_id: str, evidence: str = "", cross_off: bool = True) -> dict[str, Any]:
        return handler.mark_completed(MarkCompletedParams(task_id=task_id, evidence=evidence, cross_off=cross_off))

    def verify_task(
        task_id: str,
        evidence: str,
        passed: bool = True,
        commands_run: list[str] | None = None,
    ) -> dict[str, Any]:
        return handler.verify_task(
            VerifyTaskParams(
                task_id=task_id,
                evidence=evidence,
                passed=passed,
                commands_run=commands_run or [],
            )
        )

    def reopen_task(task_id: str, reason: str) -> dict[str, Any]:
        return handler.reopen_task(ReopenTaskParams(task_id=task_id, reason=reason))

    def list_tasks(status: str | None = None, parent_task_id: str | None = None) -> list[dict[str, Any]]:
        return handler.list_tasks(status=status, parent_task_id=parent_task_id)

    def get_progress(root_task_id: str | None = None) -> dict[str, Any]:
        return handler.get_progress(root_task_id=root_task_id)

    return [
        StructuredTool.from_function(
            func=create_task,
            name="taskboard_create_task",
            description="Create a new task or subtask on the authoritative TaskBoard workspace.",
        ),
        StructuredTool.from_function(
            func=update_task,
            name="taskboard_update_task",
            description="Update an existing task's description, priority, progress, plan, acceptance criteria, or checklist state.",
        ),
        StructuredTool.from_function(
            func=delete_task,
            name="taskboard_delete_task",
            description="Safely remove an obsolete or superseded task from the TaskBoard.",
        ),
        StructuredTool.from_function(
            func=add_subtask,
            name="taskboard_add_subtask",
            description="Decompose a task by adding a subtask with specific acceptance criteria and dependencies.",
        ),
        StructuredTool.from_function(
            func=mark_completed,
            name="taskboard_mark_completed",
            description="Mark a task as completed or crossed off with completion evidence.",
        ),
        StructuredTool.from_function(
            func=verify_task,
            name="taskboard_verify_task",
            description="Verify task completion with independent evidence against acceptance criteria. Reopens or marks failed if verification fails.",
        ),
        StructuredTool.from_function(
            func=reopen_task,
            name="taskboard_reopen_task",
            description="Reopen a failed, blocked, or incomplete task for autonomous replanning and continued work.",
        ),
        StructuredTool.from_function(
            func=list_tasks,
            name="taskboard_list_tasks",
            description="Query the authoritative TaskBoard to inspect active, pending, or completed tasks.",
        ),
        StructuredTool.from_function(
            func=get_progress,
            name="taskboard_get_progress",
            description="Inspect overall live TaskBoard progress metrics and item statuses.",
        ),
    ]
