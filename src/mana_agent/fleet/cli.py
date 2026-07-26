"""Professional, bounded Fleet CLI."""

from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
from pathlib import Path

import typer

from mana_agent.config.settings import Settings
from mana_agent.execution.config import ExecutionConfig, build_provider_registry
from mana_agent.execution.manager import ExecutionManager
from mana_agent.execution.store import SandboxStore

from .config import FleetConfig
from .errors import FleetError
from .health import effective_status
from .models import (
    FleetSelectionRequest, RuntimeRequirements, WorkerStatus, fleet_id,
)
from .registry import FleetRegistry
from .service import FleetService
from .store import FleetStore

fleet_app = typer.Typer(
    help="Operate trusted workers and run cross-platform repository verification.",
    no_args_is_help=True,
)


def _runtime() -> tuple[FleetConfig, FleetStore, FleetRegistry, FleetService]:
    settings = Settings()
    config = FleetConfig.from_settings(settings)
    store = FleetStore(config.root)
    registry = FleetRegistry(store, config)
    execution_config = ExecutionConfig.from_settings(settings)
    manager = ExecutionManager(
        build_provider_registry(execution_config), execution_config,
        store=SandboxStore(config.root / "sandboxes"),
    )
    return config, store, registry, FleetService(
        config=config, registry=registry, execution_manager=manager, store=store,
    )


def _json(value) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _fail(exc: Exception) -> None:
    typer.echo(f"Fleet operation failed: {exc}", err=True)
    raise typer.Exit(2) from exc


@fleet_app.command("list")
def list_workers(json_output: bool = typer.Option(False, "--json")) -> None:
    """List persisted trusted workers and effective health."""
    config, _store, registry, _service = _runtime()
    rows = []
    for worker in registry.list():
        row = worker.model_dump(mode="json")
        row["effective_status"] = effective_status(
            worker,
            heartbeat_timeout_seconds=config.heartbeat_timeout_seconds,
            capability_ttl_seconds=config.capability_ttl_seconds,
        ).value
        rows.append(row)
    typer.echo(_json(rows) if json_output else "\n".join(
        f"{row['identity']['worker_id']}  {row['effective_status']}  "
        f"{row['capabilities']['platform']}/{row['capabilities']['architecture']}  "
        f"{','.join(row['capabilities']['labels']['values'])}"
        for row in rows
    ) or "No Fleet workers are registered.")


@fleet_app.command("show")
def show_worker(worker_id: str) -> None:
    try:
        _config, _store, registry, _service = _runtime()
        typer.echo(_json(registry.require(worker_id)))
    except FleetError as exc:
        _fail(exc)


@fleet_app.command("capabilities")
def worker_capabilities(worker_id: str) -> None:
    try:
        _config, _store, registry, _service = _runtime()
        typer.echo(_json(registry.require(worker_id).capabilities))
    except FleetError as exc:
        _fail(exc)


@fleet_app.command("doctor")
def fleet_doctor(worker_id: str = typer.Argument("")) -> None:
    config, store, registry, _service = _runtime()
    workers = [registry.require(worker_id)] if worker_id else registry.list()
    findings = {
        "enabled": config.enabled,
        "storage_writable": store.root.exists() and store.root.is_dir(),
        "workers": [
            {
                "worker_id": item.worker_id,
                "status": effective_status(
                    item,
                    heartbeat_timeout_seconds=config.heartbeat_timeout_seconds,
                    capability_ttl_seconds=config.capability_ttl_seconds,
                ).value,
                "identity": item.health.identity_status,
                "capability_fingerprint": item.capability_fingerprint,
            }
            for item in workers
        ],
    }
    typer.echo(_json(findings))
    if config.enabled and any(row["status"] != "connected" for row in findings["workers"]):
        raise typer.Exit(1)


def _status_change(worker_id: str, status: WorkerStatus) -> None:
    try:
        _config, _store, registry, _service = _runtime()
        typer.echo(_json(registry.set_status(worker_id, status)))
    except FleetError as exc:
        _fail(exc)


@fleet_app.command("drain")
def drain(worker_id: str) -> None:
    _status_change(worker_id, WorkerStatus.DRAINING)


@fleet_app.command("enable")
def enable(worker_id: str) -> None:
    _status_change(worker_id, WorkerStatus.CONNECTED)


@fleet_app.command("revoke")
def revoke(worker_id: str, yes: bool = typer.Option(False, "--yes")) -> None:
    if not yes and not typer.confirm(f"Revoke worker {worker_id}? It cannot be automatically re-enabled.", default=False):
        raise typer.Abort()
    _status_change(worker_id, WorkerStatus.REVOKED)


@fleet_app.command("jobs")
def jobs() -> None:
    _config, store, _registry, _service = _runtime()
    rows = [job.model_dump(mode="json") for run in store.list_runs() for job in run.jobs]
    typer.echo(_json(rows))


def _find_job(job_id: str):
    _config, store, _registry, _service = _runtime()
    for run in store.list_runs():
        for job in run.jobs:
            if job.job_id == job_id:
                result = next((item for item in run.results if item.job_id == job_id), None)
                return run, job, result
    raise FleetError(f"fleet job not found: {job_id}")


@fleet_app.command("job")
def job(job_id: str) -> None:
    try:
        run, selected, result = _find_job(job_id)
        typer.echo(_json({"run_id": run.fleet_run_id, "job": selected, "result": result}))
    except FleetError as exc:
        _fail(exc)


@fleet_app.command("cancel")
def cancel(job_id: str) -> None:
    try:
        _config, _store, _registry, service = _runtime()
        service.cancel(job_id)
        typer.echo(f"Cancellation requested for {job_id}.")
    except FleetError as exc:
        _fail(exc)


@fleet_app.command("logs")
def logs(job_id: str) -> None:
    try:
        _run, _job, result = _find_job(job_id)
        if result is None:
            raise FleetError("fleet job has no completed log result")
        typer.echo(result.stdout)
        if result.stderr:
            typer.echo(result.stderr, err=True)
    except FleetError as exc:
        _fail(exc)


@fleet_app.command("artifacts")
def artifacts(job_id: str) -> None:
    try:
        _run, _job, result = _find_job(job_id)
        typer.echo(_json(list(result.artifacts) if result else []))
    except FleetError as exc:
        _fail(exc)


@fleet_app.command("compare")
def compare(fleet_run_id: str) -> None:
    try:
        _config, store, _registry, _service = _runtime()
        run = store.load_run(fleet_run_id)
        if run.summary is None:
            raise FleetError("fleet run has not completed comparison")
        typer.echo(_json(run.summary))
    except FleetError as exc:
        _fail(exc)


@fleet_app.command("verify")
def verify(
    root_dir: Path = typer.Option(Path("."), "--root-dir", exists=True, file_okay=False),
    platform: list[str] = typer.Option([], "--platform"),
    architecture: list[str] = typer.Option([], "--architecture"),
    python: list[str] = typer.Option([], "--python"),
    node: list[str] = typer.Option([], "--node"),
    tool: list[str] = typer.Option([], "--tool"),
    label: list[str] = typer.Option([], "--label"),
    exclude_label: list[str] = typer.Option([], "--exclude-label"),
    command: list[str] = typer.Option([], "--command", help="Validated argv command; repeatable."),
    suite: Path | None = typer.Option(None, "--suite", exists=True, dir_okay=False),
    maximum_workers: int = typer.Option(4, "--maximum-workers", min=1, max=64),
    timeout: int = typer.Option(1800, "--timeout", min=1, max=86400),
    retain_workspaces: bool = typer.Option(False, "--retain-workspaces"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run an explicit verification matrix; this is not an unrestricted remote shell."""
    if suite is not None:
        raise typer.BadParameter(
            "--suite Fleet execution must be started through `mana-agent eval run` "
            "after the suite declares workspace_backend: fleet."
        )
    if not platform:
        raise typer.BadParameter("at least one --platform is required; coverage is never inferred")
    if not command:
        raise typer.BadParameter("at least one --command is required")
    allowed_platforms = {"linux", "windows", "macos"}
    if set(platform) - allowed_platforms:
        raise typer.BadParameter("--platform must be linux, windows, or macos")
    argv_commands = [shlex.split(item, posix=True) for item in command]
    if any(not item for item in argv_commands):
        raise typer.BadParameter("--command cannot be empty")
    repository = root_dir.expanduser().resolve()
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        capture_output=True, text=True, check=False,
    )
    if commit_result.returncode != 0:
        raise typer.BadParameter("--root-dir must be a Git repository")
    try:
        _config, _store, _registry, service = _runtime()
        request = FleetSelectionRequest(
            decision_id=fleet_id("fleet_decision"),
            repository_id=str(repository),
            required_platforms=frozenset(platform),
            allowed_platforms=frozenset(platform),
            required_architectures=frozenset(architecture),
            runtime=RuntimeRequirements(python=frozenset(python), node=frozenset(node)),
            required_tools=frozenset(tool),
            maximum_workers=maximum_workers,
            timeout_seconds=timeout,
            preferred_labels=frozenset(label),
            forbidden_labels=frozenset(exclude_label),
        )
        plan = service.create_plan(
            request=request, repository_path=repository,
            repository_commit=commit_result.stdout.strip(),
            commands=argv_commands, retain_workspaces=retain_workspaces,
        )
        completed = asyncio.run(service.execute(service.create_run(plan)))
        typer.echo(_json(completed.summary) if json_output else (
            f"Fleet run {completed.fleet_run_id}: {completed.summary.outcome.value}"
        ))
        if completed.summary.outcome.value != "fully_verified":
            raise typer.Exit(1)
    except FleetError as exc:
        _fail(exc)
