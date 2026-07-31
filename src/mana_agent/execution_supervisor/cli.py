"""Operator CLI for durable supervised tasks."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from mana_agent.config.settings import Settings
from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.errors import ExecutionSupervisorError, RetrySafetyError
from mana_agent.execution_supervisor.models import (
    ExecutionState,
    RecoveryAction,
    RecoveryDecision,
    RetryCategory,
)
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor


tasks_app = typer.Typer(
    help="Inspect and control durable supervised executions.",
    no_args_is_help=True,
)


def _supervisor() -> ExecutionSupervisor:
    return ExecutionSupervisor(ExecutionSupervisorConfig.from_settings(Settings()))


def _render(value) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def _fail(exc: Exception) -> None:
    typer.echo(f"Task operation failed: {exc}", err=True)
    raise typer.Exit(2) from exc


def _decision(value: str) -> RecoveryDecision:
    source = value.strip()
    if not source:
        raise typer.BadParameter(
            "A validated recovery decision is required; no fallback retry was selected."
        )
    path = Path(source).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else json.loads(source)
    if isinstance(payload, dict) and "decision_id" not in payload and any(
        isinstance(item, dict) and "decision_id" in item for item in payload.values()
    ):
        raise ValueError(
            "the selected file is a taskboard routing-decision registry, not a "
            "RecoveryDecision; omit --decision-json to create the operator retry "
            "decision automatically"
        )
    return RecoveryDecision.model_validate(payload)


def _operator_retry_decision(
    supervisor: ExecutionSupervisor,
    task_id: str,
    *,
    category: RetryCategory,
) -> RecoveryDecision:
    task = supervisor.store.get_task(task_id)
    return RecoveryDecision(
        decision_id=f"operator-cli:{task.task_id}:{task.state_version}:retry",
        task_id=task.task_id,
        action=RecoveryAction.RETRY,
        retry_category=category,
        reason="operator requested retry from the task CLI",
        safe_to_continue=True,
    )


@tasks_app.command("list")
def list_tasks(
    incomplete: bool = typer.Option(False, "--incomplete", help="Show only non-terminal tasks."),
) -> None:
    try:
        rows = _supervisor().store.list_tasks(incomplete_only=incomplete)
        typer.echo(_render([item.model_dump(mode="json") for item in rows]))
    except ExecutionSupervisorError as exc:
        _fail(exc)


@tasks_app.command("status")
def status(task_id: str) -> None:
    try:
        supervisor = _supervisor()
        task = supervisor.store.get_task(task_id)
        payload = task.model_dump(mode="json")
        payload["parent_progress"] = supervisor.parent_progress(task_id).model_dump(mode="json")
        checkpoint = supervisor.store.get_checkpoint(task.checkpoint_id) if task.checkpoint_id else None
        payload["checkpoint"] = checkpoint.model_dump(mode="json") if checkpoint else None
        typer.echo(_render(payload))
    except ExecutionSupervisorError as exc:
        _fail(exc)


@tasks_app.command("tree")
def tree(task_id: str) -> None:
    try:
        supervisor = _supervisor()
        root = supervisor.store.get_task(task_id)
        rows: list[dict] = []
        pending = [(root, 0)]
        while pending:
            task, depth = pending.pop()
            rows.append({
                "depth": depth,
                "task_id": task.task_id,
                "state": task.state.value,
                "attempt_id": task.attempt_id,
                "children": list(task.child_task_ids),
            })
            pending.extend(
                (supervisor.store.get_task(child_id), depth + 1)
                for child_id in reversed(task.child_task_ids)
            )
        typer.echo(_render(rows))
    except ExecutionSupervisorError as exc:
        _fail(exc)


@tasks_app.command("logs")
def logs(task_id: str, limit: int = typer.Option(200, "--limit", min=1, max=5000)) -> None:
    try:
        supervisor = _supervisor()
        supervisor.store.get_task(task_id)
        typer.echo(_render(supervisor.store.events_for_task(task_id, limit=limit)))
    except ExecutionSupervisorError as exc:
        _fail(exc)


@tasks_app.command("artefacts")
def artefacts(task_id: str) -> None:
    try:
        supervisor = _supervisor()
        supervisor.store.get_task(task_id)
        payload = supervisor.store.artifact_manifest(task_id)
        if payload is None:
            _fail(ExecutionSupervisorError(f"no artifact manifest exists for task {task_id}"))
        typer.echo(_render(payload))
    except ExecutionSupervisorError as exc:
        _fail(exc)


@tasks_app.command("cancel")
def cancel(
    task_id: str,
    reason: str = typer.Option(..., "--reason", help="Durable operator cancellation reason."),
    attempt_id: str = typer.Option("", "--attempt-id", help="Cancel only this active attempt."),
) -> None:
    try:
        supervisor = _supervisor()
        changed = (
            supervisor.cancel_attempt(task_id, attempt_id=attempt_id, reason=reason)
            if attempt_id
            else supervisor.cancel(task_id, reason=reason)
        )
        typer.echo(_render({"cancelled": changed}))
    except ExecutionSupervisorError as exc:
        _fail(exc)


@tasks_app.command("retry")
def retry(
    task_id: str,
    decision_json: str = typer.Option(
        "",
        "--decision-json",
        help="Optional standalone RecoveryDecision JSON or file.",
    ),
    category: RetryCategory = typer.Option(
        RetryCategory.MODEL,
        "--category",
        help="Retry budget category used by the automatic operator decision.",
    ),
) -> None:
    try:
        supervisor = _supervisor()
        decision = (
            _decision(decision_json)
            if decision_json.strip()
            else _operator_retry_decision(supervisor, task_id, category=category)
        )
        task = supervisor.retry(task_id, decision)
        typer.echo(_render(task))
    except (ExecutionSupervisorError, ValueError, json.JSONDecodeError, OSError) as exc:
        _fail(exc)


@tasks_app.command("resume")
def resume(
    task_id: str,
    decision_json: str = typer.Option("", "--decision-json", help="Required unless retry is already scheduled."),
) -> None:
    try:
        supervisor = _supervisor()
        task = supervisor.store.get_task(task_id)
        if task.state not in {ExecutionState.RETRY_SCHEDULED, ExecutionState.REPLANNING}:
            task = supervisor.retry(task_id, _decision(decision_json))
        resumed = supervisor.release_retry(task.task_id)
        payload = resumed.model_dump(mode="json")
        payload["checkpoint"] = (
            supervisor.resume_checkpoint(task.task_id).model_dump(mode="json")
            if task.checkpoint_id
            else None
        )
        typer.echo(_render(payload))
    except (ExecutionSupervisorError, ValueError, json.JSONDecodeError, OSError) as exc:
        _fail(exc)


@tasks_app.command("recover")
def recover() -> None:
    try:
        supervisor = _supervisor()
        repaired = supervisor.reconnect_tree()
        typer.echo(_render({"tree_links_repaired": repaired, **supervisor.recover().model_dump(mode="json")}))
    except (ExecutionSupervisorError, RetrySafetyError) as exc:
        _fail(exc)
