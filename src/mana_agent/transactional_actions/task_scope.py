"""Task-wide approval scope helpers for durable multi-step computer work."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .approvals import ApprovalGrant
    from .models import ActionIntent

# Safe computer filesystem mutations that may share one human approval for the
# remainder of a durable task lineage. High-risk ops (trash, recording, system
# power) stay action-once and never enter this family.
TASK_WIDE_COMPUTER_OPERATIONS: frozenset[str] = frozenset(
    {
        "filesystem.mkdir",
        "filesystem.copy",
        "filesystem.move",
        "filesystem.rename",
    }
)


def task_scope_id_for_action(action: ActionIntent) -> str:
    """Return the durable task lineage ID used for task-wide grants.

    Prefer the supervisor root so multi-task children share one approval. Fall
    back to the owning parent task, never to synthetic session placeholders.
    """
    context = action.normalized_arguments.get("execution_context")
    context = context if isinstance(context, dict) else {}
    root = str(context.get("root_task_id") or "").strip()
    if root:
        return root
    parent = str(action.parent_task_id or "").strip()
    if parent and not parent.startswith("computer-session:"):
        return parent
    task = str(context.get("task_id") or "").strip()
    if task and not task.startswith("computer-session:"):
        return task
    return ""


def computer_task_wide_eligible(action: ActionIntent) -> bool:
    """Whether this action may participate in a multi-use task grant."""
    if action.tool_name != "computer":
        return False
    if action.operation_name not in TASK_WIDE_COMPUTER_OPERATIONS:
        return False
    if not task_scope_id_for_action(action):
        return False
    return True


def task_wide_operations_for(action: ActionIntent) -> list[str]:
    """Operations covered when issuing a task-wide grant for this action."""
    if computer_task_wide_eligible(action):
        return sorted(TASK_WIDE_COMPUTER_OPERATIONS)
    return [str(action.operation_name)]


def action_matches_task_grant(grant: ApprovalGrant, action: ActionIntent) -> bool:
    """Return whether a multi-use task grant covers this awaiting action."""
    from .models import ApprovalScope

    if grant.scope is not ApprovalScope.TASK:
        return False
    scope_id = task_scope_id_for_action(action)
    if not scope_id or grant.task_scope_id != scope_id:
        return False
    if grant.allowed_tool_name and grant.allowed_tool_name != action.tool_name:
        return False
    allowed_ops = set(grant.allowed_operations or ())
    if allowed_ops and action.operation_name not in allowed_ops:
        return False
    allowed_caps = set(grant.allowed_permission_scopes or ())
    if allowed_caps and not set(action.requested_capabilities).issubset(allowed_caps):
        return False
    return True
