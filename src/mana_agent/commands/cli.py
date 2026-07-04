from __future__ import annotations

from . import cli_internal as _cli_internal
from .cli_internal import *  # noqa: F401,F403
from .main_cli import main
from .chat_cli import chat
from .ui_helpers import *  # noqa: F401,F403
from .ui_helpers import (
    ChatTurnTelemetry,
    _render_coding_sections,
    _render_turn_summary,
    _render_turn_transparency,
    _sanitize_full_auto_answer_text,
)

# Use exactly one canonical Typer app.
# Do not create a second typer.Typer() here.
app = _cli_internal.app


def _replace_command(name: str, callback) -> None:
    """Register command deterministically even if another import registered it first."""
    app.registered_commands[:] = [
        command
        for command in app.registered_commands
        if command.name != name
    ]
    app.command(name)(callback)


# Root callback.
app.callback()(main)

# Re-register public commands deterministically.
_replace_command("chat", chat)
_replace_command("analyze", _cli_internal.analyze_command)
_replace_command("plan", _cli_internal.plan_command)
_replace_command("continue", _cli_internal.continue_command)


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