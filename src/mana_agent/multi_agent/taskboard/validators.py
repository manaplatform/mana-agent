from __future__ import annotations

from mana_agent.multi_agent.core.errors import InvalidTaskTransition
from mana_agent.multi_agent.core.types import TaskBoardItem, TaskStatus

_TERMINAL = {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED}
_ALLOWED: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.NEW: {TaskStatus.PLANNING, TaskStatus.DISCUSSING, TaskStatus.ROUTED, TaskStatus.BLOCKED, TaskStatus.FAILED},
    TaskStatus.PLANNING: {TaskStatus.DISCUSSING, TaskStatus.ROUTED, TaskStatus.QUEUED, TaskStatus.BLOCKED, TaskStatus.FAILED},
    TaskStatus.DISCUSSING: {TaskStatus.ROUTED, TaskStatus.BLOCKED, TaskStatus.FAILED},
    TaskStatus.ROUTED: {TaskStatus.WAITING_FOR_TOOLS, TaskStatus.QUEUED, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.FAILED},
    TaskStatus.WAITING_FOR_TOOLS: {TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED, TaskStatus.BLOCKED, TaskStatus.FAILED},
    TaskStatus.QUEUED: {TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED, TaskStatus.BLOCKED, TaskStatus.FAILED},
    TaskStatus.IN_PROGRESS: {TaskStatus.NEEDS_REVIEW, TaskStatus.VERIFYING, TaskStatus.DONE, TaskStatus.BLOCKED, TaskStatus.FAILED},
    TaskStatus.NEEDS_REVIEW: {TaskStatus.VERIFYING, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.FAILED},
    TaskStatus.VERIFYING: {TaskStatus.DONE, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.FAILED},
}


def validate_transition(task: TaskBoardItem, next_status: TaskStatus, *, reason: str | None = None) -> None:
    if task.status == next_status:
        return
    if task.status in _TERMINAL:
        raise InvalidTaskTransition(f"{task.status.value} cannot transition to {next_status.value} without reopen")
    allowed = _ALLOWED.get(task.status, set())
    if next_status not in allowed:
        raise InvalidTaskTransition(f"{task.status.value} cannot transition to {next_status.value}")
    if next_status == TaskStatus.FAILED and not str(reason or "").strip():
        raise InvalidTaskTransition("failed status requires a reason")
    if next_status == TaskStatus.BLOCKED and not str(reason or "").strip() and not task.blockers:
        raise InvalidTaskTransition("blocked status requires a blocker")
    if next_status == TaskStatus.VERIFYING and not task.verification_commands and not str(reason or "").strip():
        raise InvalidTaskTransition("verifying status requires verification commands or an explicit reason")
