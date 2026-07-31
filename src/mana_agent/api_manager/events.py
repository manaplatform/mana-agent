"""API Manager event adapter for the shared CLI/API/dashboard stream."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator

from mana_agent.services.execution_event_hub import get_execution_event_hub
from mana_agent.workspaces.paths import repository_id_for_path


_CONTEXT: ContextVar[tuple[str, str, str]] = ContextVar(
    "mana_api_event_context",
    default=("", "", ""),
)


@contextmanager
def api_event_scope(
    *,
    session_id: str,
    execution_id: str,
    root: str | Path,
) -> Iterator[None]:
    token = _CONTEXT.set(
        (
            str(session_id),
            str(execution_id),
            repository_id_for_path(Path(root).resolve()),
        )
    )
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def publish_api_event(event_type: str, payload: dict[str, Any]) -> None:
    conversation_id, execution_id, repository_id = _CONTEXT.get()
    get_execution_event_hub().emit(
        event_type,
        title=event_type.replace(".", " ").title(),
        conversation_id=conversation_id,
        execution_id=execution_id,
        repository_id=repository_id,
        status=(
            "failed"
            if event_type.endswith(".failed")
            else "success"
            if event_type.endswith((".completed", ".saved", ".updated", ".deleted", ".refreshed"))
            else "running"
        ),
        metadata=payload,
        persist=bool(conversation_id and repository_id),
    )

