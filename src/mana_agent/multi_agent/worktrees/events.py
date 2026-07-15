"""Structured runtime events for managed agent worktrees."""

from __future__ import annotations

from typing import Any

from mana_agent.cli.events import ChatEvent, make_event
from mana_agent.services.execution_event_hub import get_execution_event_hub


WORKSPACE_EVENT_KINDS = {
    "workspace.created",
    "workspace.reused",
    "workspace.resumed",
    "workspace.status_changed",
    "workspace.branch_created",
    "workspace.dirty_detected",
    "workspace.verifying",
    "workspace.reviewing",
    "workspace.reviewed",
    "workspace.conflict",
    "workspace.merge_candidate",
    "workspace.merged",
    "workspace.retained",
    "workspace.cleanup",
    "workspace.reconciled",
    "workspace.failed",
}


def emit_workspace_event(
    kind: str,
    *,
    task_id: str = "",
    workspace_id: str = "",
    repository_id: str = "",
    session_id: str = "",
    status: str = "running",
    title: str = "",
    summary: str | None = None,
    details: dict[str, Any] | None = None,
    agent_id: str | None = "workspace_manager",
) -> ChatEvent:
    """Emit a concise workspace event into the shared CLI/chat/dashboard stream."""

    normalized = str(kind or "").strip()
    if normalized not in WORKSPACE_EVENT_KINDS and not normalized.startswith("workspace."):
        normalized = f"workspace.{normalized}" if normalized else "workspace.status_changed"
    meta = {
        "kind": normalized,
        "task_id": task_id,
        "workspace_id": workspace_id,
        "repository_id": repository_id,
        **dict(details or {}),
    }
    event = make_event(
        normalized,
        title=title or normalized.replace("workspace.", "workspace ").replace("_", " "),
        status=status,
        message=summary or "",
        session_id=session_id,
        agent_id=agent_id,
        metadata=meta,
    )
    try:
        get_execution_event_hub().publish(
            event,
            conversation_id=session_id or "",
            execution_id=task_id or workspace_id or "",
            repository_id=repository_id or "",
            persist=bool(session_id and repository_id),
        )
    except Exception:
        # Event fan-out must never break workspace lifecycle.
        pass
    return event
