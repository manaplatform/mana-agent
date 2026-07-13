from __future__ import annotations

import json
import typer
from mana_agent.connectors.browser.session import BrowserSessionManager

browser_app = typer.Typer(help="Inspect the optional Playwright browser runtime.")


@browser_app.command("status")
def browser_status(json_output: bool = typer.Option(False, "--json")) -> None:
    """Check Playwright and Chromium without installing or downloading anything."""
    status = BrowserSessionManager.status()
    if json_output:
        typer.echo(json.dumps(status, ensure_ascii=False, sort_keys=True))
    else:
        typer.echo("Browser ready." if status.get("ok") else f"Browser unavailable: {status.get('error') or 'Chromium is not installed.'}")
    if not status.get("ok"):
        raise typer.Exit(code=1)
