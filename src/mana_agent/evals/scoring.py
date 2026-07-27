from __future__ import annotations

from .config import ScoreWeights
from .models import EvalRun, EvalTask, EvaluationResult, TestResult


def score_run(
    *,
    task: EvalTask,
    run: EvalRun,
    tests: list[TestResult],
    weights: ScoreWeights,
    setup_success: bool,
    first_attempt_success: bool,
) -> EvaluationResult:
    route = run.routes[-1] if run.routes else None
    actual_tools = {tool.tool_name for tool in run.tool_calls}
    expected = task.expected
    route_correct = True
    if expected.intent:
        route_correct = route_correct and bool(route and route.intent == expected.intent)
    if expected.route:
        route_correct = route_correct and bool(
            route and (route.execution_path or route.intent) == expected.route
        )
    if expected.expected_lane:
        route_correct = route_correct and bool(route and route.lane_selection == expected.expected_lane)
    if expected.forbidden_execution_paths:
        route_correct = route_correct and not bool(
            route and (route.execution_path or route.intent) in expected.forbidden_execution_paths
        )
    route_correct = route_correct and set(expected.required_tools).issubset(actual_tools)
    route_correct = route_correct and not bool(set(expected.forbidden_tools) & actual_tools)
    tests_pass = all(item.passed for item in tests) if tests else task.expected.no_repository_mutation
    changed = {name for patch in run.patches for name in patch.changed_files}
    policy_violations: list[str] = []
    if set(expected.forbidden_changed_files) & changed:
        policy_violations.append("forbidden file changed")
    if set(task.forbidden_files) & changed:
        policy_violations.append("task-level forbidden file changed")
    if task.allowed_files and not changed.issubset(set(task.allowed_files)):
        policy_violations.append("changed file is outside the task allowlist")
    if expected.no_repository_mutation and changed:
        policy_violations.append("repository mutation was forbidden")
    if not set(expected.required_changed_files).issubset(changed):
        policy_violations.append("required changed file is missing")
    total_tokens = sum(item.total_tokens for item in run.usage)
    known_cost = run.calculated_cost
    if expected.maximum_tokens is not None and total_tokens > expected.maximum_tokens:
        policy_violations.append("token budget exceeded")
    if expected.maximum_cost is not None and (known_cost is None or known_cost > expected.maximum_cost):
        policy_violations.append("cost budget exceeded or unknown")
    if expected.maximum_latency_seconds is not None and run.latency_seconds > expected.maximum_latency_seconds:
        policy_violations.append("latency budget exceeded")
    reviewer = sum(
        float(item.get("score", 0.0)) for item in run.reviewer_results if isinstance(item, dict)
    ) / max(1, len(run.reviewer_results))
    reviewer = max(0.0, min(1.0, reviewer))
    verifier = bool(run.verifier_results) and all(bool(item.get("passed")) for item in run.verifier_results)
    completed = bool(run.final_answer.strip()) and not run.errors
    reproducible = bool(run.environment and run.environment.reproducible)
    dimensions = {
        "task_completion": float(completed),
        "tests": float(tests_pass),
        "reviewer_quality": reviewer,
        "routing_correctness": float(route_correct),
        "policy_compliance": float(not policy_violations),
        "efficiency": float(not any("budget exceeded" in item for item in policy_violations)),
        "reproducibility": float(reproducible),
        "verifier": float(verifier),
        "patch_quality": reviewer,
        "safety": float(not policy_violations),
    }
    weight_values = weights.model_dump()
    score = sum(dimensions[name] * weight for name, weight in weight_values.items()) / sum(weight_values.values()) * 100.0
    task_success = setup_success and tests_pass and route_correct and not policy_violations and completed
    if not tests_pass:
        task_success = False
    failures = []
    if not setup_success:
        failures.append("setup")
    if not tests_pass:
        failures.append("tests")
    if not route_correct:
        failures.append("routing")
    if policy_violations:
        failures.append("policy")
    return EvaluationResult(
        setup_success=setup_success,
        tests=tests,
        reviewer_scores={"normalized": reviewer},
        verifier_result={"passed": verifier},
        policy_violations=policy_violations,
        task_success=task_success,
        normalized_score=round(score, 4),
        score_dimensions=dimensions,
        score_explanation="; ".join(failures) if failures else "all required objective checks passed",
        failure_category=failures[0] if failures else None,
        first_attempt_success=first_attempt_success,
        final_success_after_retries=task_success,
    )
