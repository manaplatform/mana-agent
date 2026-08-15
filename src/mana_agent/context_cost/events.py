"""Governor event adapter using the existing ChatEvent-compatible sink."""

from __future__ import annotations

from typing import Any, Callable

CONTEXT_COST_EVENT_TYPES = frozenset({
    "context.budget", "context.compacted", "context.capabilities_loaded",
    "context.capabilities_unloaded", "cost.updated", "budget.warning", "budget.blocked",
    "budget.exhausted", "context.forecast", "accounting.forecast",
    "accounting.reservation", "accounting.revision", "accounting.reconciliation",
    "accounting.rejection",
})


def emit_context_event(
    sink: Callable[..., Any] | None,
    event_type: str,
    *,
    title: str,
    metadata: dict[str, Any],
    session_id: str,
    turn_id: str = "",
    agent_id: str = "main",
    subagent_id: str | None = None,
    step_id: str | None = None,
) -> None:
    if sink is None or event_type not in CONTEXT_COST_EVENT_TYPES:
        return
    # Import only when emitting. Importing mana_agent.cli.events while the
    # context_cost package initializes executes mana_agent.cli.__init__, whose
    # ChatUIState annotation refers back to the governor.
    from mana_agent.cli.events import make_event

    event = make_event(
        event_type, title=title, session_id=session_id, turn_id=turn_id,
        agent_id=agent_id, subagent_id=subagent_id, step_id=step_id, metadata=metadata,
        status="failed" if event_type == "budget.blocked" else "running",
    )
    try:
        sink(event)
    except TypeError:
        try:
            sink(event_type, title, metadata=metadata)
        except TypeError:
            sink(event_type, metadata)


__all__ = ["CONTEXT_COST_EVENT_TYPES", "emit_context_event"]
