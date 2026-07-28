"""Sanitized Teach Mode progress events for CLI, TUI, dashboard and gateway."""

from __future__ import annotations

from typing import Any

from mana_agent.services.execution_event_hub import get_execution_event_hub


def publish_teach_event(
    event_type: str,
    *,
    session_id: str = "",
    flow_id: str = "",
    title: str,
    status: str = "running",
    metadata: dict[str, Any] | None = None,
) -> None:
    safe_metadata = {
        key: value
        for key, value in (metadata or {}).items()
        if key not in {"value", "text", "password", "secret", "token", "cookie"}
    }
    get_execution_event_hub().emit(
        f"teach.{event_type}",
        title=title,
        conversation_id="",
        execution_id=flow_id or session_id,
        session_id=session_id,
        status=status,
        metadata=safe_metadata,
        persist=False,
    )
