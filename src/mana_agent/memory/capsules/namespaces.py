"""Trusted namespace construction and validation."""

from __future__ import annotations

import re

from mana_agent.memory.capsules.models import CapsuleScope, CapsuleTaskContext

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def validate_identifier(value: str | None, label: str) -> str:
    text = str(value or "").strip()
    if not text or not _SAFE_ID.fullmatch(text) or ".." in text:
        raise ValueError(f"{label} is missing or contains unsafe namespace characters")
    return text


def namespace_for(
    scope: CapsuleScope,
    context: CapsuleTaskContext,
    *,
    team_id: str | None = None,
) -> str:
    """Build a namespace only from authenticated task context."""
    if scope is CapsuleScope.PRIVATE:
        return f"tasks/{validate_identifier(context.task_id, 'task_id')}"
    if scope is CapsuleScope.PARENT_CHILD:
        parent_id = validate_identifier(context.parent_task_id, "parent_task_id")
        task_id = validate_identifier(context.task_id, "task_id")
        return f"tasks/{parent_id}/children/{task_id}"
    if scope is CapsuleScope.TEAM:
        selected = validate_identifier(team_id, "team_id")
        if selected not in context.team_ids:
            raise ValueError("team_id is not present in the trusted task context")
        return f"teams/{selected}"
    if scope is CapsuleScope.PROJECT:
        return f"projects/{validate_identifier(context.project_id, 'project_id')}"
    if scope is CapsuleScope.ORGANISATION:
        return f"organisations/{validate_identifier(context.organisation_id, 'organisation_id')}"
    if scope is CapsuleScope.USER:
        return f"users/{validate_identifier(context.user_id, 'user_id')}"
    raise ValueError(f"unsupported capsule scope: {scope}")


def validate_namespace(namespace: str) -> str:
    value = str(namespace or "").strip()
    if not value or value.startswith("/") or value.endswith("/"):
        raise ValueError("capsule namespace is invalid")
    if ".." in value or "*" in value or "?" in value or "\\" in value:
        raise ValueError("capsule namespace contains traversal or wildcard characters")
    parts = value.split("/")
    if any(not _SAFE_ID.fullmatch(part) for part in parts):
        raise ValueError("capsule namespace contains unsafe characters")
    return value
