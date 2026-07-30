"""Desired-state plan execution with explicit drift and rollback evidence."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .models import ServerPlan, ServerPlanStep


@dataclass(slots=True)
class PlanEvidence:
    plan_id: str
    drifted_steps: list[str] = field(default_factory=list)
    applied_steps: list[str] = field(default_factory=list)
    verified_steps: list[str] = field(default_factory=list)
    rolled_back_steps: list[str] = field(default_factory=list)
    failed_step: str | None = None


class ServerPlanExecutor:
    """Mechanical plan runner; the model supplies every step and verifier."""

    def __init__(
        self,
        *,
        inspect_drift: Callable[[ServerPlanStep], Awaitable[bool]],
        apply_step: Callable[[ServerPlanStep], Awaitable[Any]],
        verify_step: Callable[[ServerPlanStep], Awaitable[bool]],
        rollback_step: Callable[[ServerPlanStep], Awaitable[Any]],
    ) -> None:
        self.inspect_drift = inspect_drift
        self.apply_step = apply_step
        self.verify_step = verify_step
        self.rollback_step = rollback_step

    async def apply(self, plan: ServerPlan) -> PlanEvidence:
        evidence = PlanEvidence(plan_id=plan.plan_id)
        changed: list[ServerPlanStep] = []
        for step in plan.steps:
            if not await self.inspect_drift(step):
                continue
            evidence.drifted_steps.append(step.step_id)
            if step.rollback is None:
                raise ValueError(f"Drifted plan step {step.step_id!r} has no rollback action.")
            try:
                await self.apply_step(step)
                changed.append(step)
                evidence.applied_steps.append(step.step_id)
                if not await self.verify_step(step):
                    raise RuntimeError(f"Verification failed for plan step {step.step_id!r}.")
                evidence.verified_steps.append(step.step_id)
            except Exception:
                evidence.failed_step = step.step_id
                for applied in reversed(changed):
                    await self.rollback_step(applied)
                    evidence.rolled_back_steps.append(applied.step_id)
                raise
        return evidence
