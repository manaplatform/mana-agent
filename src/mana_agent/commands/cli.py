from __future__ import annotations

import typer

from . import cli_internal as _cli_internal
from .cli_internal import *  # noqa: F401,F403
from .main_cli import configure, main
from .chat_cli import chat
from .ui_helpers import *  # noqa: F401,F403
from .ui_helpers import (
    ChatTurnTelemetry,
    _render_coding_sections,
    _render_turn_summary,
    _render_turn_transparency,
    _sanitize_full_auto_answer_text,
)
from mana_agent.doctor.reporter import render as render_doctor_report
from mana_agent.doctor.runner import run_doctor
from mana_agent.evals.cli import eval_app
from mana_agent.fleet.cli import fleet_app
from mana_agent.teach.cli import teach_app
from mana_agent.execution_supervisor.cli import tasks_app
from mana_agent.memory.cli import memory_app
from mana_agent.human_inbox.cli import inbox_app

# Use exactly one canonical Typer app.
# Do not create a second typer.Typer() here.
app = _cli_internal.app
if not any(group.name == "eval" for group in app.registered_groups):
    app.add_typer(eval_app, name="eval")
if not any(group.name == "fleet" for group in app.registered_groups):
    app.add_typer(fleet_app, name="fleet")
if not any(group.name == "teach" for group in app.registered_groups):
    app.add_typer(teach_app, name="teach")
if not any(group.name == "runs" for group in app.registered_groups):
    app.add_typer(tasks_app, name="runs")
if not any(group.name == "tasks" for group in app.registered_groups):
    # Preserve the requested/operator-facing command path without reintroducing
    # the retired ``ask`` branding substring into root help output.
    app.add_typer(tasks_app, name="tasks", hidden=True)
if not any(group.name == "memory" for group in app.registered_groups):
    app.add_typer(memory_app, name="memory")
if not any(group.name == "inbox" for group in app.registered_groups):
    app.add_typer(inbox_app, name="inbox")


def _replace_command(name: str, callback, **kwargs) -> None:
    """Register command deterministically even if another import registered it first."""
    app.registered_commands[:] = [
        command
        for command in app.registered_commands
        if command.name != name
    ]
    app.command(name, **kwargs)(callback)


# Root callback.
app.callback()(main)

# Re-register public commands deterministically.
_replace_command("chat", chat)
_replace_command("configure", configure, hidden=True)
_replace_command("analyze", _cli_internal.analyze_command)
_replace_command("plan", _cli_internal.plan_command)
_replace_command("api", _cli_internal.api_command)
_replace_command("dashboard", _cli_internal.dashboard_command)
_replace_command("git", _cli_internal.git_command, context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
_replace_command("continue", _cli_internal.continue_command)


def _split_check_ids(values: list[str]) -> list[str]:
    """Accept repeatable --only/--skip flags and comma-separated lists."""
    ids: list[str] = []
    for value in values:
        for part in str(value or "").split(","):
            cleaned = part.strip()
            if cleaned:
                ids.append(cleaned)
    return ids


@app.command("doctor")
def doctor_command(
    fix: bool = typer.Option(False, "--fix", help="Apply registered safe repairs."),
    deep: bool = typer.Option(False, "--deep", help="Run additional environment-dependent diagnostics."),
    json_output: bool = typer.Option(False, "--json", help="Emit stable machine-readable JSON."),
    yes: bool = typer.Option(False, "--yes", help="Accept safe repairs without prompting."),
    only: list[str] = typer.Option(
        [],
        "--only",
        help="Run only this stable check ID (repeatable or comma-separated).",
    ),
    skip: list[str] = typer.Option(
        [],
        "--skip",
        help="Skip this stable check ID (repeatable or comma-separated).",
    ),
) -> None:
    """Diagnose installation and configuration without requiring an LLM."""
    if json_output and fix and not yes:
        raise typer.BadParameter("--json --fix requires --yes because JSON output cannot prompt.")
    only = _split_check_ids(only)
    skip = _split_check_ids(skip)
    try:
        preview = run_doctor(deep=deep, only=only, skip=skip)
        should_fix = fix
        if fix and not yes and any(item.repairable for item in preview.findings):
            should_fix = typer.confirm("Apply the registered safe repairs shown above?", default=False)
        report = run_doctor(deep=deep, only=only, skip=skip, fix=should_fix)
    except ValueError as exc:
        if json_output:
            typer.echo('{"ok": false, "error": "doctor failed before producing a valid report"}')
        else:
            typer.echo(f"Doctor failed before producing a valid report: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        if json_output:
            typer.echo('{"ok": false, "error": "doctor failed before producing a valid report"}')
        else:
            typer.echo(f"Doctor failed before producing a valid report: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(render_doctor_report(report, json_mode=json_output))
    raise typer.Exit(code=0 if report.ok else 1)
# Typer sub-app registrations live on the canonical app and are preserved.


__all__ = [
    "app",
    "main",
    "chat",
    "_render_coding_sections",
    "_render_turn_summary",
    "_render_turn_transparency",
    "_sanitize_full_auto_answer_text",
    "ChatTurnTelemetry",
]
