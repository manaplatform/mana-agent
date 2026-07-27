"""Teach Mode command group."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from .models import RecordedEvent, SelectorCandidate, TeachError
from .service import TeachService


teach_app = typer.Typer(help="Record one demonstration and turn it into a safe reusable Mana Flow.")


def _service() -> TeachService:
    return TeachService()


def _json(value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True)
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


def _run(call):
    try:
        return call()
    except TeachError as exc:
        raise typer.BadParameter(str(exc)) from exc


@teach_app.command("start")
def start_command(
    task_name: str = typer.Argument(..., help="Name of the task being demonstrated."),
    permission: list[str] = typer.Option([], "--permission", help="Explicit recording permission scope; repeatable."),
) -> None:
    """Start visible, local semantic recording."""
    session = _run(lambda: _service().start(task_name, permissions=permission))
    typer.echo(f"● REC  {session.task_name}\nSession: {session.id}")


@teach_app.command("pause")
def pause_command(session_id: str | None = typer.Option(None, "--session")) -> None:
    session = _run(lambda: _service().pause(session_id))
    typer.echo(f"Ⅱ PAUSED  {session.id}")


@teach_app.command("resume")
def resume_command(session_id: str | None = typer.Option(None, "--session")) -> None:
    session = _run(lambda: _service().resume(session_id))
    typer.echo(f"● REC  {session.task_name}\nSession: {session.id}")


@teach_app.command("status")
def status_command(
    session_id: str | None = typer.Option(None, "--session"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    service = _service()
    session = _run(lambda: service.storage.load_session(session_id) if session_id else service.active_session())
    if json_output:
        _json(session)
    else:
        typer.echo(
            f"{session.state.value}: {session.task_name}\n"
            f"Session: {session.id}\n"
            f"Events: {session.raw_event_count} raw / {session.normalized_event_count} normalized\n"
            f"Sources: {', '.join(session.recorder_capabilities) or 'none'}"
        )


@teach_app.command("explain")
def explain_command(
    explanation: str = typer.Argument(...),
    session_id: str | None = typer.Option(None, "--session"),
) -> None:
    session = _run(lambda: _service().explain(explanation, session_id))
    typer.echo(f"Explanation added to {session.id}.")


@teach_app.command("stop")
def stop_command(session_id: str | None = typer.Option(None, "--session")) -> None:
    session, flow = _run(lambda: _service().stop(session_id))
    _json(
        {
            "session_id": session.id,
            "state": session.state,
            "flow_id": flow.id,
            "flow_version": flow.version,
            "status": flow.status,
            "steps": [step.model_dump(mode="json", by_alias=True) for step in flow.steps],
            "inputs": {name: value.model_dump(mode="json") for name, value in flow.inputs.items()},
            "verification": [rule.model_dump(mode="json") for rule in flow.verify],
            "review_required": True,
        }
    )


@teach_app.command("cancel")
def cancel_command(session_id: str | None = typer.Option(None, "--session")) -> None:
    session = _run(lambda: _service().cancel(session_id))
    typer.echo(f"Cancelled {session.id}. Recording remains local for audit/retention cleanup.")


@teach_app.command("review")
def review_command(
    flow_id: str = typer.Argument(...),
    accept: bool = typer.Option(False, "--accept", help="Activate after successful verification."),
    accept_unverified: bool = typer.Option(False, "--accept-unverified", help="Explicitly activate without verification."),
) -> None:
    service = _service()
    if accept or accept_unverified:
        flow = _run(lambda: service.accept(flow_id, explicit_unverified_acceptance=accept_unverified))
    else:
        flow = _run(lambda: service.storage.load_flow(flow_id))
    _json(flow)


@teach_app.command("replay")
def replay_command(
    flow_id: str = typer.Argument(...),
    version: int | None = typer.Option(None, "--version", min=1),
    mode: str = typer.Option("dry_run", "--mode", help="dry_run, guided, or normal."),
    input_value: list[str] = typer.Option([], "--input", help="Input as name=value; repeatable."),
) -> None:
    inputs = _parse_inputs(input_value)
    result = _run(lambda: _service().replay(flow_id, version=version, mode=mode, inputs=inputs))
    _json(result)
    if result.verification_status == "failed":
        raise typer.Exit(code=1)


@teach_app.command("edit")
def edit_command(flow_id: str = typer.Argument(...)) -> None:
    service = _service()
    flow = _run(lambda: service.storage.load_flow(flow_id))
    path = service.storage.root / "flows" / flow.id / f"v{flow.version}.yaml"
    typer.echo(f"Edit {path}, then run `mana-agent teach review {flow.id}` to validate it.")


@teach_app.command("repair")
def repair_command(
    flow_id: str = typer.Argument(...),
    step_id: str = typer.Argument(...),
    selector_type: str = typer.Option(..., "--type"),
    selector_value: str = typer.Option(..., "--value"),
    confidence: float = typer.Option(0.9, "--confidence", min=0, max=1),
) -> None:
    candidate = SelectorCandidate(type=selector_type, value=selector_value, confidence=confidence)
    _json(_run(lambda: _service().repair(flow_id, step_id, candidate)))


@teach_app.command("export")
def export_command(
    flow_id: str = typer.Argument(...),
    output: Path = typer.Option(..., "--output"),
) -> None:
    path = _run(lambda: _service().export(flow_id, output))
    typer.echo(str(path))


@teach_app.command("import")
def import_command(package: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    flow = _run(lambda: _service().import_package(package))
    _json({"flow": flow, "activation": "Run a dry replay, review permissions and inputs, then explicitly accept."})


@teach_app.command("card")
def card_command(
    flow_id: str = typer.Argument(...),
    minutes_saved: int = typer.Option(1, "--minutes-saved", min=0),
) -> None:
    _json(_run(lambda: _service().flow_card(flow_id, estimated_minutes_saved=minutes_saved)))


@teach_app.command("doctor")
def doctor_command(json_output: bool = typer.Option(False, "--json")) -> None:
    report = _service().doctor()
    if json_output:
        _json(report)
        return
    typer.echo(f"Platform: {report['platform']}  Headless: {report['headless']}")
    for name, item in report["recorders"].items():
        marker = "✓" if item["available"] else "✗"
        typer.echo(f"{marker} {name}: {item['reason'] or 'available'}")


@teach_app.command("schedule")
def schedule_command(
    flow_id: str = typer.Argument(...),
    cron: str = typer.Option(..., "--cron", help="Five-field POSIX cron expression (UTC)."),
    target: list[str] = typer.Option(["local"], "--target"),
    version_policy: str = typer.Option("pinned", "--version-policy", help="pinned or latest."),
) -> None:
    from mana_agent.automations.service import ScheduleDefinition, deploy_schedule

    service = _service()
    flow = _run(lambda: service.storage.load_flow(flow_id))
    if flow.status not in {"verified", "active"}:
        raise typer.BadParameter("Only a verified or explicitly active flow can be scheduled.")
    missing_defaults = [name for name, item in flow.inputs.items() if item.required and item.default is None]
    if missing_defaults:
        raise typer.BadParameter("Mandatory scheduled inputs have no defaults: " + ", ".join(missing_defaults))
    if version_policy not in {"pinned", "latest"}:
        raise typer.BadParameter("--version-policy must be pinned or latest.")
    schedule = ScheduleDefinition.create(
        name=f"Teach flow: {flow.name}",
        action="teach-flow",
        cron=cron,
        targets=target,
        action_config={
            "flow_id": flow.id,
            "flow_version": flow.version if version_policy == "pinned" else None,
            "version_policy": version_policy,
            "inputs": {name: item.default for name, item in flow.inputs.items()},
        },
    )
    _json(_run(lambda: deploy_schedule(schedule, Path.cwd())))


@teach_app.command("record-event", hidden=True)
def record_event_command(
    event_json: str = typer.Argument(..., help="Versioned RecordedEvent JSON from an integrated adapter."),
) -> None:
    event = RecordedEvent.model_validate_json(event_json)
    _run(lambda: _service().record_event(event))
    typer.echo(event.event_id)


def _parse_inputs(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise typer.BadParameter("--input values must use name=value.")
        key, item = value.split("=", 1)
        if not key:
            raise typer.BadParameter("Flow input name cannot be empty.")
        result[key] = item
    return result
