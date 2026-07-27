"""Strict sandbox lifecycle transition validation."""

from __future__ import annotations

from mana_agent.execution.errors import LifecycleTransitionError
from mana_agent.execution.models import SandboxState


_TRANSITIONS: dict[SandboxState, frozenset[SandboxState]] = {
    SandboxState.REQUESTED: frozenset({SandboxState.PROVISIONING, SandboxState.FAILED}),
    SandboxState.PROVISIONING: frozenset({SandboxState.READY, SandboxState.FAILED, SandboxState.TERMINATING}),
    SandboxState.READY: frozenset({SandboxState.RUNNING, SandboxState.SUSPENDING, SandboxState.TERMINATING, SandboxState.FAILED}),
    SandboxState.RUNNING: frozenset({SandboxState.READY, SandboxState.SUSPENDING, SandboxState.TERMINATING, SandboxState.FAILED}),
    SandboxState.SUSPENDING: frozenset({SandboxState.SUSPENDED, SandboxState.FAILED}),
    SandboxState.SUSPENDED: frozenset({SandboxState.RESUMING, SandboxState.TERMINATING, SandboxState.FAILED}),
    SandboxState.RESUMING: frozenset({SandboxState.READY, SandboxState.FAILED}),
    SandboxState.TERMINATING: frozenset({SandboxState.TERMINATED, SandboxState.FAILED}),
    SandboxState.TERMINATED: frozenset({SandboxState.CLEANING, SandboxState.CLEANED}),
    SandboxState.CLEANING: frozenset({SandboxState.CLEANED, SandboxState.FAILED}),
    SandboxState.CLEANED: frozenset(),
    SandboxState.FAILED: frozenset({SandboxState.TERMINATING, SandboxState.CLEANING, SandboxState.CLEANED}),
}


def validate_transition(current: SandboxState, target: SandboxState) -> None:
    if current == target:
        return
    if target not in _TRANSITIONS[current]:
        raise LifecycleTransitionError(f"invalid sandbox transition: {current.value} -> {target.value}")
