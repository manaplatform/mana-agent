"""Scoped publication context using existing supervisor ownership, never UI state."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from dataclasses import replace
from functools import wraps
import uuid
import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActivityContext:
    conversation_id: str
    execution_id: str
    repository_id: str = ""
    task_id: str = ""
    root_task_id: str = ""
    agent_id: str = "main"
    subagent_id: str = ""
    parent_agent_id: str = ""
    attempt_id: str = ""
    attempt_generation: int = 0
    execution_mode: str = ""
    supervisor: Any = field(default=None, repr=False, compare=False)
    lease_token: str = field(default="", repr=False, compare=False)
    protected_text: tuple[str, ...] = field(default=(), repr=False, compare=False)
    execution_registry: Any = field(default=None, repr=False, compare=False)
    authorize_tool: Any = field(default=None, repr=False, compare=False)
    execution_lane: str = ""
    taskboard: Any = field(default=None, repr=False, compare=False)
    parent_task_id: str = ""
    coordinator: Any = field(default=None, repr=False, compare=False)

    def identity(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in (
            "conversation_id", "execution_id", "repository_id", "task_id", "root_task_id", "parent_task_id",
            "agent_id", "subagent_id", "parent_agent_id", "attempt_id", "attempt_generation", "execution_mode",
        ) if getattr(self, key) not in (None, "", 0)}

    @contextmanager
    def publication(self) -> Iterator[None]:
        """Keep the ownership check and append inside the existing store fence."""
        if self.supervisor is None:
            if self.attempt_id:
                raise ValueError("attempt activity requires its supervising execution")
            yield
            return
        from mana_agent.execution_supervisor.errors import StaleLeaseError
        from mana_agent.execution_supervisor.models import ExecutionState
        with self.supervisor.store.locked():
            task = self.supervisor.get_task(self.task_id)
            attempt_id = self.attempt_id
            lease_token = self.lease_token
            coordinator = getattr(self, "coordinator", None)
            if coordinator is not None and hasattr(coordinator, "inspect_task"):
                try:
                    lane_exec = coordinator.inspect_task(self.task_id)
                    if lane_exec and lane_exec.supervisor_attempt_id and lane_exec.supervisor_lease_token:
                        attempt_id = lane_exec.supervisor_attempt_id
                        lease_token = lane_exec.supervisor_lease_token
                except Exception:
                    pass
            if task.state == ExecutionState.QUEUED and not task.cancellation_status and not task.waiting_inbox_item_id:
                try:
                    task = self.supervisor.resume_running(self.task_id, attempt_id=attempt_id, lease_token=lease_token)
                except Exception:
                    pass
            self.supervisor._validate_lease(task, attempt_id=attempt_id, lease_token=lease_token)
            if task.state != ExecutionState.RUNNING or task.cancellation_status is not None:
                raise StaleLeaseError("activity belongs to a retired or cancelled execution attempt")
            yield


_CONTEXT: ContextVar[ActivityContext | None] = ContextVar("mana_activity_context", default=None)


def current_activity_context() -> ActivityContext | None:
    return _CONTEXT.get()


@contextmanager
def activity_scope(context: ActivityContext) -> Iterator[ActivityContext]:
    token = _CONTEXT.set(context)
    try:
        yield context
    finally:
        _CONTEXT.reset(token)


def emit_activity(event_type: str, *, title: str, summary: str = "", **fields: Any) -> dict[str, Any]:
    from mana_agent.services.execution_event_hub import get_execution_event_hub
    try:
        return get_execution_event_hub().publish({"type": event_type, "title": title, "summary": summary, **fields})
    except Exception:
        # Observability cannot retry, interrupt, or finalize the operation it describes.
        logger.warning("Activity publication unavailable", exc_info=True)
        return {"type": "activity.degraded", "visibility": "internal", "sequence": 0}


def observed_activity(title: str):
    """Observe a synchronous runtime boundary without interpreting its prose."""
    def decorate(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            if current_activity_context() is None:
                return method(*args, **kwargs)
            operation_id = f"operation-{uuid.uuid4().hex}"
            emit_activity("agent.started", title=title, operation_id=operation_id, status="running")
            try:
                result = method(*args, **kwargs)
            except BaseException:
                emit_activity("agent.failed", title=f"{title} failed", operation_id=operation_id, status="failed")
                raise
            emit_activity("agent.completed", title=f"{title} returned", operation_id=operation_id,
                          status="failed" if getattr(result, "route_name", "") == "blocked" else "success")
            return result
        return wrapped
    return decorate


def activity_turn(method):
    """Bind the admitted gateway turn, including pre-task routing observations."""
    @wraps(method)
    def wrapped(gateway, session_id: str, text: str, **options):
        from mana_agent.workspaces.paths import repository_id_for_path
        turn_id = str(options.get("turn_id") or f"turn_{uuid.uuid4().hex}")
        options["turn_id"] = turn_id
        state = gateway._session(session_id)
        state["activity_execution_id"] = turn_id
        settings = getattr(gateway, "settings", None)
        protected = tuple(str(getattr(settings, name, "") or "") for name in (
            "openai_api_key", "openrouter_api_key", "nvidia_api_key", "anthropic_api_key", "xai_api_key", "mana_github_token", "mana_api_token"
        ) if len(str(getattr(settings, name, "") or "")) >= 8)
        context = ActivityContext(conversation_id=session_id, execution_id=turn_id, protected_text=protected,
                                  repository_id=repository_id_for_path(gateway.root),
                                  execution_mode=str(state.get("execution_mode") or ""))
        with activity_scope(context):
            emit_activity("agent.routing", title="Understanding the request", status="running",
                          operation_id=f"{turn_id}:routing")
            try:
                result = method(gateway, session_id, text, **options)
            except BaseException:
                emit_activity("turn.failed", title="Gateway execution failed", status="failed",
                              operation_id=f"{turn_id}:response")
                raise
            emit_activity("turn.response_ready", title="Gateway response returned", status="failed" if getattr(result, "error", None) else "success",
                          operation_id=f"{turn_id}:response", metadata={"evidence_status": "observed"})
            return result
    return wrapped


def run_with_activity_attempt(callback, *, _activity_supervisor, _activity_task_id: str,
                              _activity_attempt_id: str, _activity_lease_token: str,
                              _activity_registry=None, _activity_authorize=None, _activity_lane: str = "", _activity_taskboard=None,
                              _activity_coordinator=None, **kwargs):
    previous = current_activity_context()
    if previous is None or not _activity_task_id or not _activity_attempt_id:
        return callback(**kwargs)
    attempt = _activity_supervisor.store.get_attempt(_activity_attempt_id)
    if attempt is None:
        raise ValueError("activity attempt is unavailable")
    task = _activity_supervisor.get_task(_activity_task_id)
    context = replace(previous, task_id=_activity_task_id, root_task_id=task.root_task_id, parent_task_id=task.parent_task_id or "", attempt_id=_activity_attempt_id,
                      attempt_generation=attempt.generation, supervisor=_activity_supervisor,
                      lease_token=_activity_lease_token, execution_registry=_activity_registry or previous.execution_registry,
                      authorize_tool=_activity_authorize or previous.authorize_tool,
                      execution_lane=_activity_lane or previous.execution_lane, taskboard=_activity_taskboard or previous.taskboard,
                      coordinator=_activity_coordinator or getattr(previous, "coordinator", None))
    with activity_scope(context):
        return callback(**kwargs)


def controller_activity(method):
    @wraps(method)
    def wrapped(controller, **options):
        previous = current_activity_context()
        context = previous or ActivityContext(conversation_id=str(options.get("session_id") or ""),
                                               execution_id=str(options["task_id"]))
        with activity_scope(context):
            return method(controller, **options)
    return wrapped


def bind_activity_attempt(supervisor, task_id: str, attempt_id: str, lease_token: str) -> None:
    """Called only inside a managed activity scope after lease admission."""
    previous = current_activity_context()
    if previous is None:
        return
    task = supervisor.get_task(task_id)
    attempt = supervisor.store.get_attempt(attempt_id) if attempt_id else None
    generation = attempt.generation if attempt is not None else getattr(task, "attempt_generation", 0)
    _CONTEXT.set(replace(previous, task_id=task_id, root_task_id=task.root_task_id, parent_task_id=task.parent_task_id or "",
                         conversation_id=previous.conversation_id or task.session_id,
                         repository_id=previous.repository_id or task.repository_id,
                         attempt_id=attempt_id or getattr(task, "attempt_id", "") or "",
                         attempt_generation=generation,
                         supervisor=supervisor, lease_token=lease_token))
