"""Authentication entrypoints for provider credential references."""
from __future__ import annotations

import typer

from mana_agent.commands.codex_cli import codex_auth

auth_app = typer.Typer(help="Register secure provider credential references.", no_args_is_help=True)
auth_app.command("codex")(codex_auth)

__all__ = ["auth_app"]
