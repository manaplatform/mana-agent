from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from .models import EvalRun

Classification = Literal["improved", "unchanged", "regressed", "new", "missing", "inconclusive"]


@dataclass(slots=True)
class TaskComparison:
    task_id: str
    classification: Classification
    baseline_score: float | None
    candidate_score: float | None
    score_change: float | None
    baseline_success: bool | None
    candidate_success: bool | None
    routing_changed: bool = False
    tool_selection_changed: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "classification": self.classification,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "score_change": self.score_change,
            "baseline_success": self.baseline_success,
            "candidate_success": self.candidate_success,
            "routing_changed": self.routing_changed,
            "tool_selection_changed": self.tool_selection_changed,
            "reasons": self.reasons,
        }


def _best_by_task(runs: Iterable[EvalRun]) -> dict[str, EvalRun]:
    grouped: dict[str, list[EvalRun]] = {}
    for run in runs:
        grouped.setdefault(run.task_id, []).append(run)
    return {
        task: sorted(items, key=lambda item: (item.trial_number, item.started_at))[0]
        for task, items in grouped.items()
    }


def compare_runs(baseline: Iterable[EvalRun], candidate: Iterable[EvalRun]) -> dict[str, Any]:
    left = _best_by_task(baseline)
    right = _best_by_task(candidate)
    comparisons: list[TaskComparison] = []
    for task_id in sorted(set(left) | set(right)):
        base = left.get(task_id)
        cand = right.get(task_id)
        if base is None:
            comparisons.append(TaskComparison(task_id, "new", None, cand.outcome.normalized_score if cand and cand.outcome else None, None, None, cand.outcome.task_success if cand and cand.outcome else None))
            continue
        if cand is None:
            comparisons.append(TaskComparison(task_id, "missing", base.outcome.normalized_score if base.outcome else None, None, None, base.outcome.task_success if base.outcome else None, None, reasons=["candidate result is missing"]))
            continue
        if base.outcome is None or cand.outcome is None:
            comparisons.append(TaskComparison(task_id, "inconclusive", base.outcome.normalized_score if base.outcome else None, cand.outcome.normalized_score if cand.outcome else None, None, base.outcome.task_success if base.outcome else None, cand.outcome.task_success if cand.outcome else None, reasons=["one or both outcomes are incomplete"]))
            continue
        delta = cand.outcome.normalized_score - base.outcome.normalized_score
        if base.outcome.task_success and not cand.outcome.task_success:
            classification: Classification = "regressed"
        elif not base.outcome.task_success and cand.outcome.task_success:
            classification = "improved"
        elif delta < -1e-9:
            classification = "regressed"
        elif delta > 1e-9:
            classification = "improved"
        else:
            classification = "unchanged"
        base_route = base.routes[-1] if base.routes else None
        cand_route = cand.routes[-1] if cand.routes else None
        comparisons.append(TaskComparison(
            task_id, classification, base.outcome.normalized_score, cand.outcome.normalized_score,
            delta, base.outcome.task_success, cand.outcome.task_success,
            routing_changed=bool(
                base_route and cand_route
                and (base_route.execution_path or base_route.intent)
                != (cand_route.execution_path or cand_route.intent)
            ),
            tool_selection_changed=bool(base_route and cand_route and base_route.selected_tools != cand_route.selected_tools),
        ))
    base_outcomes = [item.outcome for item in left.values() if item.outcome]
    cand_outcomes = [item.outcome for item in right.values() if item.outcome]
    base_costs = [item.calculated_cost for item in left.values() if item.calculated_cost is not None]
    cand_costs = [item.calculated_cost for item in right.values() if item.calculated_cost is not None]
    return {
        "schema_version": 1,
        "baseline_task_count": len(left),
        "candidate_task_count": len(right),
        "baseline_pass_rate": sum(item.task_success for item in base_outcomes) / len(base_outcomes) if base_outcomes else 0.0,
        "candidate_pass_rate": sum(item.task_success for item in cand_outcomes) / len(cand_outcomes) if cand_outcomes else 0.0,
        "baseline_mean_score": statistics.fmean(item.normalized_score for item in base_outcomes) if base_outcomes else 0.0,
        "candidate_mean_score": statistics.fmean(item.normalized_score for item in cand_outcomes) if cand_outcomes else 0.0,
        "baseline_routing_accuracy": statistics.fmean(item.score_dimensions.get("routing_correctness", 0.0) for item in base_outcomes) if base_outcomes else 0.0,
        "candidate_routing_accuracy": statistics.fmean(item.score_dimensions.get("routing_correctness", 0.0) for item in cand_outcomes) if cand_outcomes else 0.0,
        "baseline_mean_latency": statistics.fmean(item.latency_seconds for item in left.values()) if left else 0.0,
        "candidate_mean_latency": statistics.fmean(item.latency_seconds for item in right.values()) if right else 0.0,
        "baseline_mean_cost": statistics.fmean(base_costs) if base_costs else None,
        "candidate_mean_cost": statistics.fmean(cand_costs) if cand_costs else None,
        "baseline_policy_violations": sum(len(item.policy_violations) for item in base_outcomes),
        "candidate_policy_violations": sum(len(item.policy_violations) for item in cand_outcomes),
        "classifications": dict(Counter(item.classification for item in comparisons)),
        "tasks": [item.to_dict() for item in comparisons],
    }
