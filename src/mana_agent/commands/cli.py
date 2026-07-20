from __future__ import annotations

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

# Use exactly one canonical Typer app.
# Do not create a second typer.Typer() here.
app = _cli_internal.app


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


@app.command("doctor")
def doctor_command(
    fix: bool = typer.Option(False, "--fix", help="Apply registered safe repairs."),
    deep: bool = typer.Option(False, "--deep", help="Run additional environment-dependent diagnostics."),
    json_output: bool = typer.Option(False, "--json", help="Emit stable machine-readable JSON."),
    yes: bool = typer.Option(False, "--yes", help="Accept safe repairs without prompting."),
    only: list[str] = typer.Option([], "--only", help="Run only this stable check ID (repeatable)."),
    skip: list[str] = typer.Option([], "--skip", help="Skip this stable check ID (repeatable)."),
) -> None:
    """Diagnose installation and configuration without requiring an LLM."""
    if json_output and fix and not yes:
        raise typer.BadParameter("--json --fix requires --yes because JSON output cannot prompt.")
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
