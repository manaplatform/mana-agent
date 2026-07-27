"""Targeted selector repair with revision history."""

from __future__ import annotations

from copy import deepcopy

from .models import ManaFlow, SelectorCandidate, TeachError, utc_now
from .normalizer import rank_selectors


class TargetedSelectorRepair:
    def repair(self, flow: ManaFlow, step_id: str, candidate: SelectorCandidate) -> ManaFlow:
        updated = deepcopy(flow)
        for step in updated.steps:
            if step.id != step_id:
                continue
            for existing in step.selectors:
                existing.failures += 1
            candidate.successes += 1
            candidate.last_verified_at = utc_now()
            step.selectors = rank_selectors([candidate, *step.selectors])
            step.confidence = step.selectors[0].confidence
            step.requires_review = step.confidence < 0.65
            updated.version += 1
            updated.updated_at = utc_now()
            updated.statistics.repair_count += 1
            return updated
        raise TeachError(f"Flow step not found: {step_id}")
