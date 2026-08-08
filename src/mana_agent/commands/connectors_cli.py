"""CLI for connector health status, incidents, and recovery."""

from __future__ import annotations

import asyncio
import json

import typer

from mana_agent.commands.email_cli import connector_app
from mana_agent.connectors.health import (
    bootstrap_health_manager,
    format_status_report,
    reset_health_manager,
)

health_app = typer.Typer(help="Inspect and recover connector health.")
connectors_app = typer.Typer(help="Connector inventory and path health (not process-alive alone).")
connector_app.add_typer(health_app, name="health")


def _manager():
    reset_health_manager()
    return bootstrap_health_manager()


def _match_reports(manager, name: str):
    needle = name.strip().lower()
    if not needle:
        return manager.status()
    return [
        r
        for r in manager.status()
        if r.connector_id.lower() == needle
        or r.connector_type.lower() == needle
        or r.connector_id.lower().endswith(f":{needle}")
        or r.connector_id.lower().startswith(f"{needle}:")
    ]


def _status_impl(name: str, json_output: bool, *, probe: bool = True) -> None:
    """Show connector health.

    By default runs a live safe probe. Without a probe, every connector stays
    ``UNKNOWN`` / ``STARTUP_PENDING`` after registration — that is intentional
    until the path is verified, not a process-alive shortcut.
    """
    manager = _manager()
    reports = _match_reports(manager, name)
    if name and not reports:
        typer.echo(f"No health report for connector {name!r}. Is it configured and registered?")
        raise typer.Exit(code=1)
    if not reports:
        typer.echo("No connectors registered for health monitoring.")
        return
    if probe:
        probed = []
        for report in reports:
            try:
                probed.append(asyncio.run(manager.probe(report.connector_id, force=True)))
            except Exception as exc:
                # Keep the registered snapshot and surface the failure in message.
                failed = report.model_copy(
                    update={
                        "message": f"Probe failed: {type(exc).__name__}: {exc}"[:300],
                    }
                )
                probed.append(failed)
        reports = probed
    if json_output:
        typer.echo(json.dumps([r.model_dump(mode="json") for r in reports], indent=2, default=str))
        return
    typer.echo("\n\n".join(format_status_report(r) for r in reports))


def _health_impl(name: str, probe: bool, json_output: bool) -> None:
    manager = _manager()
    matches = _match_reports(manager, name)
    if not matches:
        typer.echo(f"Unknown connector: {name}")
        raise typer.Exit(code=1)
    reports = []
    for report in matches:
        if probe:
            reports.append(asyncio.run(manager.probe(report.connector_id, force=True)))
        else:
            current = manager.get_report(report.connector_id)
            if current:
                reports.append(current)
    if json_output:
        typer.echo(json.dumps([r.model_dump(mode="json") for r in reports], indent=2, default=str))
        return
    typer.echo("\n\n".join(format_status_report(r) for r in reports))


def _incidents_impl(name: str, limit: int, json_output: bool) -> None:
    manager = _manager()
    connector_id = name.strip() or None
    if connector_id:
        matched = _match_reports(manager, connector_id)
        connector_id = matched[0].connector_id if matched else connector_id
    incidents = manager.list_incidents(connector_id=connector_id, limit=limit)
    if json_output:
        typer.echo(json.dumps([i.model_dump(mode="json") for i in incidents], indent=2, default=str))
        return
    if not incidents:
        typer.echo("No connector incidents recorded.")
        return
    for incident in incidents:
        status = "OPEN" if incident.open else ("RECOVERED" if incident.recovered else "CLOSED")
        typer.echo(
            f"{incident.incident_id}\t{incident.connector_id}\t{status}\t"
            f"{incident.opening_state.value}\t{incident.opening_reason.value}\t"
            f"events={len(incident.events)}"
        )
        for event in incident.events[-5:]:
            typer.echo(
                f"  {event.occurred_at.isoformat()} {event.event_type} "
                f"{event.reason_code.value} {event.message}"
            )


def _recover_impl(name: str, json_output: bool) -> None:
    manager = _manager()
    matches = _match_reports(manager, name)
    if not matches:
        typer.echo(f"Unknown connector: {name}")
        raise typer.Exit(code=1)
    reports = [asyncio.run(manager.recover(r.connector_id, force=True)) for r in matches]
    if json_output:
        typer.echo(json.dumps([r.model_dump(mode="json") for r in reports], indent=2, default=str))
        return
    typer.echo("\n\n".join(format_status_report(r) for r in reports))


@connectors_app.command("status")
def connectors_status(
    name: str = typer.Argument("", help="Optional connector id (gmail, telegram, gmail:account-id)."),
    probe: bool = typer.Option(
        True,
        "--probe/--no-probe",
        help="Run a live safe probe (default). Use --no-probe for cached registration state only.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show connector health — never equates process-alive with online."""
    _status_impl(name, json_output, probe=probe)


@connector_app.command("status")
def connector_status(
    name: str = typer.Argument("", help="Optional connector id (gmail, telegram, gmail:account-id)."),
    probe: bool = typer.Option(
        True,
        "--probe/--no-probe",
        help="Run a live safe probe (default). Use --no-probe for cached registration state only.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show connector health — never equates process-alive with online."""
    _status_impl(name, json_output, probe=probe)


@connectors_app.command("health")
def connectors_health(
    name: str = typer.Argument(..., help="Connector id or type."),
    probe: bool = typer.Option(True, "--probe/--no-probe"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Probe and display detailed connector health."""
    _health_impl(name, probe, json_output)


@health_app.command("show")
def health_show(
    name: str = typer.Argument(..., help="Connector id or type."),
    probe: bool = typer.Option(True, "--probe/--no-probe"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Probe and display detailed connector health."""
    _health_impl(name, probe, json_output)


@connectors_app.command("incidents")
def connectors_incidents(
    name: str = typer.Argument("", help="Optional connector id filter."),
    limit: int = typer.Option(20, "--limit", min=1, max=200),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List durable connector incidents (including recovered ones)."""
    _incidents_impl(name, limit, json_output)


@health_app.command("incidents")
def health_incidents(
    name: str = typer.Argument("", help="Optional connector id filter."),
    limit: int = typer.Option(20, "--limit", min=1, max=200),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List durable connector incidents (including recovered ones)."""
    _incidents_impl(name, limit, json_output)


@connectors_app.command("recover")
def connectors_recover(
    name: str = typer.Argument(..., help="Connector id or type to recover."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Trigger safe automatic recovery for a connector."""
    _recover_impl(name, json_output)


@health_app.command("recover")
def health_recover(
    name: str = typer.Argument(..., help="Connector id or type to recover."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Trigger safe automatic recovery for a connector."""
    _recover_impl(name, json_output)
