from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ids import stable_hash
from .models import EvalRun
from .redaction import redact
from .storage import EvalStorageError, atomic_write

BASELINE_SCHEMA_VERSION = 1


class BaselineTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    score: float
    success: bool
    route: str = ""
    selected_tools: list[str] = Field(default_factory=list)
    latency_seconds: float = 0.0
    cost: float | None = None
    policy_violations: list[str] = Field(default_factory=list)
    artifact_hash: str


class EvalBaseline(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = BASELINE_SCHEMA_VERSION
    name: str
    suite_name: str
    suite_version: str
    variant_fingerprint: str
    mana_agent_commit: str
    environment_fingerprint: str
    aggregate_metrics: dict[str, Any]
    tasks: list[BaselineTask]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def complete(self) -> "EvalBaseline":
        if not self.tasks:
            raise ValueError("baseline requires at least one task")
        return self


def create_baseline(*, name: str, suite_name: str, suite_version: str, variant_fingerprint: str, runs: Iterable[EvalRun], allow_non_reproducible: bool = False) -> EvalBaseline:
    selected = list(runs)
    if not selected:
        raise EvalStorageError("cannot create a baseline from an empty experiment")
    invalid = [item.run_id for item in selected if item.outcome is None or not item.environment or not item.environment.reproducible]
    if invalid and not allow_non_reproducible:
        raise EvalStorageError(f"baseline rejected because runs are incomplete or non-reproducible: {', '.join(invalid)}")
    tasks = []
    for run in selected:
        if run.outcome is None:
            continue
        route = run.routes[-1] if run.routes else None
        tasks.append(BaselineTask(
            task_id=run.task_id,
            score=run.outcome.normalized_score,
            success=run.outcome.task_success,
            route=route.intent if route else "",
            selected_tools=route.selected_tools if route else [],
            latency_seconds=run.latency_seconds,
            cost=run.calculated_cost,
            policy_violations=run.outcome.policy_violations,
            artifact_hash=stable_hash(run.model_dump(mode="json")),
        ))
    first = selected[0]
    pass_rate = sum(item.success for item in tasks) / len(tasks) if tasks else 0.0
    mean_score = sum(item.score for item in tasks) / len(tasks) if tasks else 0.0
    return EvalBaseline(
        name=name,
        suite_name=suite_name,
        suite_version=suite_version,
        variant_fingerprint=variant_fingerprint,
        mana_agent_commit=first.repository_commit,
        environment_fingerprint=stable_hash(first.environment.model_dump(mode="json") if first.environment else {}),
        aggregate_metrics={"task_count": len(tasks), "pass_rate": pass_rate, "mean_score": mean_score},
        tasks=tasks,
    )


def write_baseline(path: str | Path, baseline: EvalBaseline, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise EvalStorageError(f"baseline already exists: {path}")
    atomic_write(path, json.dumps(redact(baseline.model_dump(mode="json")), indent=2, sort_keys=True, default=str) + "\n")
    return path


def load_baseline(path: str | Path) -> EvalBaseline:
    path = Path(path)
    if not path.exists():
        raise EvalStorageError(f"baseline not found: {path}")
    return EvalBaseline.model_validate_json(path.read_text(encoding="utf-8"))


def baseline_as_runs(baseline: EvalBaseline) -> list[EvalRun]:
    from .models import EvaluationResult, RouteRecord, RunStatus

    now = baseline.created_at
    runs = []
    for item in baseline.tasks:
        outcome = EvaluationResult(
            setup_success=True,
            task_success=item.success,
            normalized_score=item.score,
            score_dimensions={"routing_correctness": 1.0 if item.route else 0.0},
            policy_violations=item.policy_violations,
            first_attempt_success=item.success,
            final_success_after_retries=item.success,
        )
        routes = [RouteRecord(route_input_hash="baseline", router_model="baseline", intent=item.route, selected_tools=item.selected_tools)] if item.route else []
        runs.append(EvalRun(
            run_id=f"baseline_{stable_hash(item.task_id)[:16]}", run_fingerprint=item.artifact_hash,
            experiment_id=baseline.name, task_id=item.task_id, variant_id="baseline",
            started_at=now, completed_at=now, status=RunStatus.COMPLETED,
            repository_commit=baseline.mana_agent_commit, routes=routes,
            latency_seconds=item.latency_seconds, calculated_cost=item.cost, outcome=outcome,
        ))
    return runs
