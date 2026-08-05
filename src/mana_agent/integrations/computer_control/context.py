"""Authenticated frontend identity propagated into computer tools."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable
from mana_agent.runtime_context import DurableExecutionContext


@dataclass(frozen=True, slots=True)
class ComputerClientContext:
    session_id: str
    client_type: str
    locally_confirmed: bool = False
    allowed_decision_ids: frozenset[str] = frozenset()
    workspace_root: str = ""
    execution_context: DurableExecutionContext | None = None
    transactional_runtime: Any | None = None


_context: ContextVar[ComputerClientContext | None] = ContextVar("mana_computer_client", default=None)


def current_computer_client() -> ComputerClientContext | None:
    return _context.get()


@contextmanager
def computer_client_scope(
    session_id: str,
    client_type: str,
    *,
    locally_confirmed: bool = False,
    allowed_decision_ids: frozenset[str] = frozenset(),
    workspace_root: str = "",
    execution_context: DurableExecutionContext | None = None,
    transactional_runtime: Any | None = None,
):
    token = _context.set(ComputerClientContext(
        session_id,
        client_type,
        locally_confirmed,
        allowed_decision_ids,
        workspace_root,
        execution_context,
        transactional_runtime,
    ))
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def computer_decision_scope(source_decision_id: str):
    current = _context.get()
    if current is None:
        raise RuntimeError("Computer decision scope requires an authenticated client context.")
    token = _context.set(ComputerClientContext(
        current.session_id,
        current.client_type,
        current.locally_confirmed,
        frozenset({source_decision_id}),
        current.workspace_root,
        current.execution_context,
        current.transactional_runtime,
    ))
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def computer_execution_context_scope(context: DurableExecutionContext):
    current = _context.get()
    if current is None:
        raise RuntimeError("Computer execution context requires an authenticated client context.")
    token = _context.set(ComputerClientContext(
        current.session_id, current.client_type, current.locally_confirmed,
        current.allowed_decision_ids, current.workspace_root, context, current.transactional_runtime,
    ))
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def computer_transactional_runtime_scope(runtime: Any):
    """Bind the gateway-owned durable runtime to model tool execution."""
    current = _context.get()
    if current is None:
        raise RuntimeError("Computer transactional runtime requires an authenticated client context.")
    token = _context.set(ComputerClientContext(
        current.session_id,
        current.client_type,
        current.locally_confirmed,
        current.allowed_decision_ids,
        current.workspace_root,
        current.execution_context,
        runtime,
    ))
    try:
        yield
    finally:
        _context.reset(token)


def _normalized_frontend(frontend: str) -> str:
    return {
        "cli": "local_cli", "console": "local_cli", "tui": "tui",
        "dashboard": "dashboard", "telegram": "telegram", "api": "remote_api",
        "a2a": "a2a", "acp": "remote_api",
    }.get(frontend, frontend or "untrusted")


def authenticated_computer_client(function: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate gateway turn methods so the model cannot forge client identity."""

    @wraps(function)
    def wrapper(self, session_id: str, *args: Any, **kwargs: Any) -> Any:
        state = self._session(session_id)
        frontend = _normalized_frontend(str(state.get("frontend") or "untrusted"))
        with computer_client_scope(session_id, frontend, workspace_root=str(self.root)):
            return function(self, session_id, *args, **kwargs)

    return wrapper
