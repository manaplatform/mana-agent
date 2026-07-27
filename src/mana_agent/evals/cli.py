from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .baseline import baseline_as_runs, create_baseline, load_baseline, write_baseline
from .comparison import compare_runs
from .config import EvalConfigurationError, EvalSuite, GateThresholds, load_suite
from .ids import execution_id
from .inspector import inspect_run, inspect_text
from .leaderboard import build_leaderboard
from .matrix import ExperimentRunner
from .regression import evaluate_gate
from .redaction import redact, redact_text
from .replay import task_replay, trajectory_replay
from .reports import write_leaderboard, write_regression
from .runner import EvalRunner
from .storage import EvalStorage, EvalStorageError

CONFIGURATION_FAILURE = 2
EXECUTION_FAILURE = 3
REGRESSION_FAILURE = 4
INCOMPLETE_COMPARISON = 5
SUCCESS = 0

eval_app = typer.Typer(help="Run reproducible evaluations and regression gates.", no_args_is_help=True)
baseline_app = typer.Typer(help="Create and inspect checked-in evaluation baselines.", no_args_is_help=True)
eval_app.add_typer(baseline_app, name="baseline")


def _storage(root: Path) -> EvalStorage:
    return EvalStorage(root)


def _echo(value: Any, *, json_output: bool) -> None:
    value = redact(value)
    if json_output:
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))
    elif isinstance(value, str):
        typer.echo(redact_text(value))
    else:
        typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


def _load_experiment(storage: EvalStorage, experiment_id: str):
    rows = storage.list_runs(experiment_id=experiment_id)
    return [storage.load_run(str(row["run_id"]), allow_incomplete=True) for row in rows]


def _suite_from_run(storage: EvalStorage, run_id: str) -> EvalSuite:
    path = storage.run_dir(run_id) / "config.yaml"
    try:
        import yaml
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return EvalSuite.model_validate(payload["suite"])
    except Exception as exc:
        raise EvalConfigurationError(f"cannot reconstruct suite from run {run_id}: {exc}") from exc


@eval_app.command("run")
def run_command(
    suite_path: Path = typer.Argument(..., exists=True, readable=True),
    variant: list[str] = typer.Option([], "--variant", help="Run only this variant ID (repeatable)."),
    task: list[str] = typer.Option([], "--task", help="Run only this task ID (repeatable)."),
    eval_root: Path = typer.Option(Path(".mana/evals"), "--eval-root"),
    concurrency: int | None = typer.Option(None, "--concurrency", min=1),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Execute tasks × variants × trials in isolated Git worktrees."""
    try:
        suite = load_suite(suite_path)
        storage = _storage(eval_root)
        progress_events: list[dict[str, Any]] = []
        console = Console(stderr=True, no_color=bool(os.environ.get("NO_COLOR")))
        progress = None if json_output or os.environ.get("NO_COLOR") else Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console)

        def sink(event: dict[str, Any]) -> None:
            progress_events.append(event)
            if progress is not None:
                progress.update(progress_task, description=f"{event['event']}: {event['task_id']} / {event['variant_id']}")

        runner = EvalRunner(storage=storage, progress_sink=sink, retain_workspaces=suite.defaults.retain_workspaces)
        experiment = ExperimentRunner(
            runner,
            concurrency=concurrency or suite.defaults.concurrency,
            provider_limits=suite.defaults.provider_concurrency,
        )
        if progress is not None:
            with progress:
                progress_task = progress.add_task("Starting Mana Eval Lab", total=None)
                result = experiment.execute(suite, task_ids=set(task) or None, variant_ids=set(variant) or None)
        else:
            progress_task = 0
            result = experiment.execute(suite, task_ids=set(task) or None, variant_ids=set(variant) or None)
        report = {
            "experiment_id": result.experiment_id,
            "runs": [{"run_id": item.run_id, "task_id": item.task_id, "variant_id": item.variant_id, "status": item.status.value, "success": bool(item.outcome and item.outcome.task_success), "artifact_path": str(storage.run_dir(item.run_id))} for item in result.runs],
            "cancelled": result.cancelled,
            "artifact_root": str(storage.root),
        }
        _echo(report, json_output=json_output)
        if result.cancelled or not result.runs or any(not item.outcome or not item.outcome.task_success for item in result.runs):
            raise typer.Exit(EXECUTION_FAILURE)
    except (EvalConfigurationError, ValueError) as exc:
        _echo({"error": str(exc), "exit_code": CONFIGURATION_FAILURE}, json_output=json_output)
        raise typer.Exit(CONFIGURATION_FAILURE) from exc
    except EvalStorageError as exc:
        _echo({"error": str(exc), "exit_code": EXECUTION_FAILURE}, json_output=json_output)
        raise typer.Exit(EXECUTION_FAILURE) from exc


@eval_app.command("list")
def list_command(
    suite: str | None = typer.Option(None, "--suite"),
    eval_root: Path = typer.Option(Path(".mana/evals"), "--eval-root"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    rows = _storage(eval_root).list_runs(suite=suite)
    if json_output:
        _echo(rows, json_output=True)
        return
    if not rows:
        typer.echo("No evaluation runs.")
        return
    for row in rows:
        typer.echo(redact_text(f"{row['run_id']}  {row['status']:<10}  {row['task_id']}  {row['variant_id']}  {row['artifact_path']}"))


@eval_app.command("inspect")
def inspect_command(
    run_id: str,
    event_type: str | None = typer.Option(None, "--event-type"),
    eval_root: Path = typer.Option(Path(".mana/evals"), "--eval-root"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        payload = inspect_run(_storage(eval_root), run_id, event_type=event_type)
        _echo(payload if json_output else inspect_text(payload), json_output=json_output)
    except EvalStorageError as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(EXECUTION_FAILURE) from exc


@eval_app.command("replay")
def replay_command(
    run_id: str,
    variant: str | None = typer.Option(None, "--variant"),
    eval_root: Path = typer.Option(Path(".mana/evals"), "--eval-root"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        storage = _storage(eval_root)
        suite = _suite_from_run(storage, run_id)
        selected = next((item for item in suite.variants if item.variant_id == (variant or storage.load_run(run_id).variant_id)), None)
        if selected is None:
            raise EvalConfigurationError(f"variant not found: {variant}")
        replayed = task_replay(storage=storage, runner=EvalRunner(storage=storage), suite=suite, run_id=run_id, variant=selected, experiment_id=execution_id("replay"))
        _echo({"run_id": replayed.run_id, "replayed_from_run_id": run_id, "artifact_path": str(storage.run_dir(replayed.run_id)), "status": replayed.status.value}, json_output=json_output)
        if not replayed.outcome or not replayed.outcome.task_success:
            raise typer.Exit(EXECUTION_FAILURE)
    except (EvalConfigurationError, EvalStorageError) as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(CONFIGURATION_FAILURE) from exc


@eval_app.command("replay-trajectory")
def replay_trajectory_command(run_id: str, eval_root: Path = typer.Option(Path(".mana/evals"), "--eval-root"), json_output: bool = typer.Option(False, "--json")) -> None:
    try:
        result = trajectory_replay(storage=_storage(eval_root), run_id=run_id)
        _echo(result, json_output=json_output)
        if result["diverged"]:
            raise typer.Exit(EXECUTION_FAILURE)
    except EvalStorageError as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(EXECUTION_FAILURE) from exc


@eval_app.command("leaderboard")
def leaderboard_command(
    suite: str = typer.Option(..., "--suite"),
    suite_version: str = typer.Option("1", "--suite-version"),
    eval_root: Path = typer.Option(Path(".mana/evals"), "--eval-root"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    storage = _storage(eval_root)
    rows = storage.list_runs(suite=suite)
    runs = [storage.load_run(str(row["run_id"]), allow_incomplete=True) for row in rows]
    leaderboard = build_leaderboard(runs, suite_name=suite, suite_version=suite_version)
    paths = write_leaderboard(storage.reports_dir / suite, leaderboard)
    _echo({"leaderboard": leaderboard, "artifacts": {key: str(value) for key, value in paths.items()}}, json_output=json_output)


def _baseline_runs(storage: EvalStorage, reference: str):
    path = Path(reference)
    if path.exists():
        return baseline_as_runs(load_baseline(path))
    return _load_experiment(storage, reference)


@eval_app.command("compare")
def compare_command(
    baseline: str = typer.Option(..., "--baseline"),
    candidate: str = typer.Option(..., "--candidate"),
    eval_root: Path = typer.Option(Path(".mana/evals"), "--eval-root"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    storage = _storage(eval_root)
    comparison = compare_runs(_baseline_runs(storage, baseline), _load_experiment(storage, candidate))
    gate = evaluate_gate(
        comparison,
        load_suite(Path("evals/suites/routing-smoke.yaml"), validate_runtime=False).gate
        if Path("evals/suites/routing-smoke.yaml").exists()
        else GateThresholds(),
    )
    paths = write_regression(storage.reports_dir / f"{baseline}-vs-{candidate}", comparison, gate)
    _echo({**comparison, "gate": gate.to_dict(), "artifacts": {key: str(value) for key, value in paths.items()}}, json_output=json_output)


@eval_app.command("gate")
def gate_command(
    baseline: str = typer.Option(..., "--baseline"),
    candidate: str = typer.Option(..., "--candidate"),
    suite_path: Path = typer.Option(Path("evals/suites/routing-smoke.yaml"), "--suite"),
    eval_root: Path = typer.Option(Path(".mana/evals"), "--eval-root"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        storage = _storage(eval_root)
        suite = load_suite(suite_path, validate_runtime=False)
        baseline_path = Path(baseline)
        if baseline_path.exists():
            baseline_metadata = load_baseline(baseline_path)
            if (baseline_metadata.suite_name, baseline_metadata.suite_version) != (suite.name, suite.version):
                raise EvalConfigurationError(
                    "baseline is incompatible with the selected suite version: "
                    f"{baseline_metadata.suite_name} v{baseline_metadata.suite_version} != {suite.name} v{suite.version}"
                )
        comparison = compare_runs(_baseline_runs(storage, baseline), _load_experiment(storage, candidate))
        gate = evaluate_gate(comparison, suite.gate)
        paths = write_regression(storage.reports_dir / f"{Path(baseline).stem}-vs-{candidate}", comparison, gate)
        _echo({"gate": gate.to_dict(), "artifacts": {key: str(value) for key, value in paths.items()}}, json_output=json_output)
        if not gate.passed:
            incomplete = any("missing" in reason or "inconclusive" in reason for reason in gate.reasons)
            raise typer.Exit(INCOMPLETE_COMPARISON if incomplete else REGRESSION_FAILURE)
    except (EvalConfigurationError, EvalStorageError) as exc:
        typer.echo(redact_text(str(exc)), err=True)
        raise typer.Exit(CONFIGURATION_FAILURE) from exc


@baseline_app.command("create")
def baseline_create_command(
    experiment_id: str,
    name: str = typer.Option(..., "--name"),
    suite: str = typer.Option(..., "--suite"),
    suite_version: str = typer.Option("1", "--suite-version"),
    output_dir: Path = typer.Option(Path("evals/baselines"), "--output-dir"),
    eval_root: Path = typer.Option(Path(".mana/evals"), "--eval-root"),
    yes: bool = typer.Option(False, "--yes"),
    allow_non_reproducible: bool = typer.Option(False, "--allow-non-reproducible", hidden=True),
) -> None:
    storage = _storage(eval_root)
    runs = _load_experiment(storage, experiment_id)
    if not runs:
        typer.echo(f"experiment not found: {experiment_id}", err=True)
        raise typer.Exit(EXECUTION_FAILURE)
    path = output_dir / f"{name}.json"
    overwrite = path.exists()
    approved_overwrite = yes
    if overwrite and not yes:
        if not os.isatty(0) or not typer.confirm(f"Replace baseline {path}?"):
            typer.echo("Baseline replacement requires explicit confirmation (--yes in non-interactive runs).", err=True)
            raise typer.Exit(CONFIGURATION_FAILURE)
        approved_overwrite = True
    baseline = create_baseline(name=name, suite_name=suite, suite_version=suite_version, variant_fingerprint=runs[0].run_fingerprint, runs=runs, allow_non_reproducible=allow_non_reproducible)
    write_baseline(path, baseline, overwrite=overwrite and approved_overwrite)
    typer.echo(str(path.resolve()))


@baseline_app.command("show")
def baseline_show_command(name: str, baseline_dir: Path = typer.Option(Path("evals/baselines"), "--baseline-dir"), json_output: bool = typer.Option(False, "--json")) -> None:
    baseline = load_baseline(baseline_dir / f"{name}.json")
    _echo(baseline.model_dump(mode="json"), json_output=json_output)


@eval_app.command("doctor")
def doctor_command(eval_root: Path = typer.Option(Path(".mana/evals"), "--eval-root"), json_output: bool = typer.Option(False, "--json")) -> None:
    checks = {
        "git": subprocess.run(["git", "--version"], capture_output=True, text=True).returncode == 0,
        "repository": subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True).returncode == 0,
        "eval_root_writable": os.access(eval_root.parent.resolve(), os.W_OK),
        "incomplete_runs": _storage(eval_root).recover_incomplete(),
    }
    checks["ok"] = checks["git"] and checks["repository"] and checks["eval_root_writable"] and not checks["incomplete_runs"]
    _echo(checks, json_output=json_output)
    if not checks["ok"]:
        raise typer.Exit(EXECUTION_FAILURE)
