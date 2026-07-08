from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def success(message: str, *, console: Console | None = None) -> None:
    (console or Console()).print(Panel(message, title="Success", border_style="green"))


def error(message: str, *, console: Console | None = None) -> None:
    (console or Console()).print(Panel(message, title="Error", border_style="red"))


def info(message: str, *, console: Console | None = None) -> None:
    (console or Console()).print(Panel(message, title="Mana Agent", border_style="cyan"))


def config_table(values: dict[str, object], *, console: Console | None = None) -> None:
    table = Table(title="Current Mana Config", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for key, value in values.items():
        table.add_row(key, str(value))
    (console or Console()).print(table)
