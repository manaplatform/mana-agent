from __future__ import annotations

import csv
import io
import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

from .models import EvalRun, RunStatus


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def build_leaderboard(runs: Iterable[EvalRun], *, suite_name: str, suite_version: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[EvalRun]] = defaultdict(list)
    for run in runs:
        groups[(run.variant_id, suite_version)].append(run)
    rows: list[dict[str, Any]] = []
    for (variant_id, version), items in sorted(groups.items()):
        completed = [item for item in items if item.status in {RunStatus.COMPLETED, RunStatus.FAILED}]
        outcomes = [item.outcome for item in completed if item.outcome is not None]
        scores = [item.normalized_score for item in outcomes]
        successes = sum(item.task_success for item in outcomes)
        first_attempts = sum(item.first_attempt_success for item in outcomes)
        routing = [item.score_dimensions.get("routing_correctness", 0.0) for item in outcomes]
        reviewer = [item.reviewer_scores.get("normalized", 0.0) for item in outcomes]
        verifier = [bool(item.verifier_result.get("passed")) for item in outcomes]
        latencies = [item.latency_seconds for item in completed]
        token_usage = [sum(usage.total_tokens for usage in item.usage) for item in completed]
        costs = [item.calculated_cost for item in completed if item.calculated_cost is not None]
        violations = sum(len(item.policy_violations) for item in outcomes)
        failure_categories = Counter(item.failure_category or "none" for item in outcomes)
        mean_cost = statistics.fmean(costs) if costs else None
        rows.append({
            "suite_name": suite_name,
            "suite_version": version,
            "variant_id": variant_id,
            "completed_runs": len(completed),
            "successful_runs": successes,
            "pass_rate": successes / len(outcomes) if outcomes else 0.0,
            "first_attempt_pass_rate": first_attempts / len(outcomes) if outcomes else 0.0,
            "mean_score": statistics.fmean(scores) if scores else 0.0,
            "median_score": statistics.median(scores) if scores else 0.0,
            "test_pass_rate": sum(all(test.passed for test in item.tests) for item in outcomes) / len(outcomes) if outcomes else 0.0,
            "routing_accuracy": statistics.fmean(routing) if routing else 0.0,
            "reviewer_score": statistics.fmean(reviewer) if reviewer else 0.0,
            "verifier_pass_rate": sum(verifier) / len(verifier) if verifier else 0.0,
            "policy_violation_count": violations,
            "mean_latency_seconds": statistics.fmean(latencies) if latencies else 0.0,
            "p95_latency_seconds": percentile(latencies, 0.95),
            "mean_token_usage": statistics.fmean(token_usage) if token_usage else 0.0,
            "mean_cost": mean_cost,
            "success_per_dollar": (successes / sum(costs)) if costs and sum(costs) > 0 else None,
            "reproducibility_rate": sum(bool(item.environment and item.environment.reproducible) for item in completed) / len(completed) if completed else 0.0,
            "failure_categories": dict(sorted(failure_categories.items())),
        })
    return sorted(rows, key=lambda row: (-row["pass_rate"], -row["mean_score"], row["variant_id"]))


def leaderboard_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    columns = [key for key in rows[0] if key != "failure_categories"]
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in columns})
    return output.getvalue()


def leaderboard_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Mana Eval Lab Leaderboard", "", "| Variant | Suite | Runs | Pass rate | Mean score | Routing | p95 latency | Mean cost |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        cost = "unknown" if row["mean_cost"] is None else f"${row['mean_cost']:.4f}"
        lines.append(
            f"| {row['variant_id']} | {row['suite_name']} v{row['suite_version']} | {row['completed_runs']} | "
            f"{row['pass_rate']:.1%} | {row['mean_score']:.2f} | {row['routing_accuracy']:.1%} | "
            f"{row['p95_latency_seconds']:.2f}s | {cost} |"
        )
    return "\n".join(lines) + "\n"
