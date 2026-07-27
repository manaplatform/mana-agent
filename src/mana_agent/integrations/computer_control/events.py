"""Process-local live event stream shared by CLI, TUI, dashboard, and gateway."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar

from mana_agent.chat.events import CodingActivityEvent
from mana_agent.chat.history import get_history
from mana_agent.integrations.computer_control.models import ComputerControlEvent

logger = logging.getLogger(__name__)
EventSubscriber = Callable[[ComputerControlEvent], None]
_lock = threading.RLock()
_subscribers: list[EventSubscriber] = []
_frontend_sink: ContextVar[Callable[..., None] | None] = ContextVar(
    "mana_computer_frontend_sink",
    default=None,
)


def publish_computer_event(event: ComputerControlEvent) -> None:
    """Publish sanitized progress without sensitive result values."""
    event_id = f"computer-{event.execution_id}-{event.event_type}"
    get_history().add(CodingActivityEvent(activity={
        "event_id": event_id,
        "event_type": f"computer.{event.event_type}",
        "task_id": event.execution_id,
        "backend": "computer-control",
        "status": event.state.value,
        "title": event.message,
        "summary": event.message,
        "metadata": event.metadata,
    }))
    frontend_sink = _frontend_sink.get()
    if frontend_sink is not None:
        try:
            frontend_sink(
                f"computer.{event.event_type}",
                event.message,
                execution_id=event.execution_id,
                status=event.state.value,
                metadata=event.metadata,
                event_id=event_id,
            )
        except Exception:
            logger.debug("computer-control frontend event sink raised", exc_info=True)
    with _lock:
        subscribers = list(_subscribers)
    for callback in subscribers:
        try:
            callback(event)
        except Exception:
            logger.debug("computer-control event subscriber raised", exc_info=True)


def subscribe_computer_events(callback: EventSubscriber) -> Callable[[], None]:
    with _lock:
        _subscribers.append(callback)

    def unsubscribe() -> None:
        with _lock:
            if callback in _subscribers:
                _subscribers.remove(callback)

    return unsubscribe


@contextmanager
def computer_event_scope(callback: Callable[..., None] | None):
    token = _frontend_sink.set(callback)
    try:
        yield
    finally:
        _frontend_sink.reset(token)
