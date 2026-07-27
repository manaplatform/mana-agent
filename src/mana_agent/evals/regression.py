from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import GateThresholds


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "reasons": list(self.reasons)}


def evaluate_gate(comparison: dict[str, Any], thresholds: GateThresholds) -> GateResult:
    reasons: list[str] = []
    baseline_count = int(comparison.get("baseline_task_count", 0))
    candidate_count = int(comparison.get("candidate_task_count", 0))
    if baseline_count < thresholds.minimum_task_count:
        reasons.append(f"baseline has {baseline_count} tasks; minimum is {thresholds.minimum_task_count}")
    if thresholds.fail_on_missing_candidate and candidate_count < baseline_count:
        reasons.append("candidate suite is missing required tasks")
    pass_drop = float(comparison.get("baseline_pass_rate", 0)) - float(comparison.get("candidate_pass_rate", 0))
    if pass_drop > thresholds.maximum_pass_rate_drop:
        reasons.append(f"pass-rate drop {pass_drop:.4f} exceeds {thresholds.maximum_pass_rate_drop:.4f}")
    score_drop = float(comparison.get("baseline_mean_score", 0)) - float(comparison.get("candidate_mean_score", 0))
    if score_drop > thresholds.maximum_mean_score_drop:
        reasons.append(f"mean-score drop {score_drop:.4f} exceeds {thresholds.maximum_mean_score_drop:.4f}")
    routing_drop = float(comparison.get("baseline_routing_accuracy", 0)) - float(comparison.get("candidate_routing_accuracy", 0))
    if routing_drop > thresholds.maximum_routing_accuracy_drop:
        reasons.append(f"routing-accuracy drop {routing_drop:.4f} exceeds {thresholds.maximum_routing_accuracy_drop:.4f}")
    if not thresholds.allow_new_policy_violations and int(comparison.get("candidate_policy_violations", 0)) > int(comparison.get("baseline_policy_violations", 0)):
        reasons.append("candidate introduced new policy violations")
    _ratio_gate(comparison, "mean_latency", thresholds.maximum_latency_increase_ratio, reasons)
    base_cost = comparison.get("baseline_mean_cost")
    candidate_cost = comparison.get("candidate_mean_cost")
    if thresholds.require_known_cost and (base_cost is None or candidate_cost is None):
        reasons.append("cost is unknown but the gate requires known pricing")
    elif base_cost not in (None, 0) and candidate_cost is not None:
        ratio = (float(candidate_cost) - float(base_cost)) / float(base_cost)
        if ratio > thresholds.maximum_cost_increase_ratio:
            reasons.append(f"cost increase {ratio:.2%} exceeds {thresholds.maximum_cost_increase_ratio:.2%}")
    for task in comparison.get("tasks", []):
        if task.get("classification") in {"missing", "inconclusive"}:
            reasons.append(f"task {task.get('task_id')} is {task.get('classification')}")
        elif task.get("classification") == "regressed" and (
            task.get("routing_changed")
            or (task.get("baseline_success") is True and task.get("candidate_success") is False)
        ):
            reasons.append(f"protected task {task.get('task_id')} regressed")
    return GateResult(not reasons, tuple(dict.fromkeys(reasons)))


def _ratio_gate(comparison: dict[str, Any], suffix: str, maximum: float, reasons: list[str]) -> None:
    baseline = float(comparison.get(f"baseline_{suffix}", 0) or 0)
    candidate = float(comparison.get(f"candidate_{suffix}", 0) or 0)
    if baseline > 0:
        ratio = (candidate - baseline) / baseline
        if ratio > maximum:
            reasons.append(f"{suffix.replace('_', ' ')} increase {ratio:.2%} exceeds {maximum:.2%}")
