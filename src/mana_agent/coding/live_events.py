"""Thread-safe process-local delivery for normalized live coding events."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
import uuid
from collections.abc import Callable

from mana_agent.coding.models import AgentEvent

logger = logging.getLogger(__name__)
CodingEventSubscriber = Callable[[AgentEvent], None]
_lock = threading.RLock()
_subscribers: list[CodingEventSubscriber] = []
_scoped_subscriber: ContextVar[CodingEventSubscriber | None] = ContextVar(
    "mana_coding_event_subscriber", default=None
)
_execution: ContextVar[dict[str, object] | None] = ContextVar("mana_coding_execution", default=None)


def publish_coding_event(event: AgentEvent) -> bool:
    from mana_agent.chat.activity import project_activity
    from mana_agent.services.activity_context import current_activity_context
    from mana_agent.services.execution_event_hub import get_execution_event_hub
    context = current_activity_context()
    persisted = True
    if context is not None:
        try:
            public = project_activity(event, protected=context.protected_text)
            if public is not None:
                published = get_execution_event_hub().publish(public)
                if published.get("visibility") == "internal":
                    return False
        except Exception:
            logger.warning("Coding activity projection unavailable")
            persisted = False
            event = event.model_copy(update={"payload": {**event.payload, "trace_complete": False,
                "observability_degraded": True}})
    scoped = _scoped_subscriber.get()
    if scoped is not None:
        try:
            scoped(event)
        except Exception:
            logger.debug("scoped coding event subscriber raised", exc_info=True)
    with _lock:
        subscribers = list(_subscribers)
    for callback in subscribers:
        try:
            callback(event)
        except Exception:
            logger.debug("coding event subscriber raised", exc_info=True)
    return persisted


def subscribe_coding_events(callback: CodingEventSubscriber) -> Callable[[], None]:
    with _lock:
        _subscribers.append(callback)

    def unsubscribe() -> None:
        with _lock:
            if callback in _subscribers:
                _subscribers.remove(callback)

    return unsubscribe


@contextmanager
def coding_event_scope(callback: CodingEventSubscriber):
    """Route events from one execution context to its owning frontend turn."""

    token = _scoped_subscriber.set(callback)
    try:
        yield
    finally:
        _scoped_subscriber.reset(token)


@contextmanager
def coding_execution_context(*, task_id: str, backend: str, model: str = ""):
    state: dict[str, object] = {
        "task_id": task_id,
        "backend": backend,
        "model": model,
        "sequence": 0,
    }
    token = _execution.set(state)
    try:
        yield state
    finally:
        _execution.reset(token)


def current_coding_execution() -> dict[str, object] | None:
    return _execution.get()


def publish_internal_tool_event(
    kind: str,
    tool: str,
    *,
    args: str = "",
    duration: float | None = None,
    error: str = "",
    event_id: str | None = None,
) -> None:
    state = _execution.get()
    if not state or state.get("backend") != "internal":
        return
    sequence = int(state.get("sequence") or 0) + 1
    state["sequence"] = sequence
    kind_l = str(kind or "").lower()
    failed = "fail" in kind_l or "error" in kind_l
    completed = kind_l in {"end", "finished", "done", "success"} or kind_l.endswith("_end")
    phase = "failed" if failed else "completed" if completed else "started"
    cid = str(event_id or f"internal-{uuid.uuid4().hex}").strip()
    publish_coding_event(AgentEvent(
        event_id=f"{cid}-{phase}",
        event_type=f"tool.call.{phase}",
        task_id=str(state["task_id"]),
        backend="internal",
        sequence=sequence,
        status="failed" if failed else "success" if completed else "running",
        title=str(tool),
        summary=f"{tool} {phase}",
        operation_id=cid,
        producer_sequence=sequence,
        tool_name=str(tool),
        duration_ms=int(float(duration) * 1000) if duration is not None else None,
        error=str(error or ""),
        model=str(state.get("model") or ""),
        payload={"call_id": cid},
    ))


__all__ = [
    "coding_event_scope",
    "coding_execution_context",
    "publish_coding_event",
    "publish_internal_tool_event",
    "subscribe_coding_events",
]
