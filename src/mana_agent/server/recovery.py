"""Recovery-point policy for consequential server actions."""

from __future__ import annotations

from .models import ServerActionDecision


def require_recovery_point(decision: ServerActionDecision, recovery_point_id: str | None) -> str:
    if decision.destructive and not recovery_point_id:
        raise ValueError("A verified recovery point is required before this destructive action.")
    return recovery_point_id or "not-required"
