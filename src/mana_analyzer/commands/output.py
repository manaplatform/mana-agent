from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass(frozen=True)
class OutputStyle:
    """Visual theme for command output."""

    info_border: str = "bright_blue"
    success_border: str = "green"
    warning_border: str = "yellow"
    error_border: str = "red"
    header_style: str = "bold magenta"
    key_style: str = "bold cyan"


DEFAULT_STYLE = OutputStyle()
_SHARED_CONSOLE = Console()


def get_shared_console() -> Console:
    return _SHARED_CONSOLE


class OutputSink:
    """Unified sink for command output in text and JSON modes."""

    def __init__(
        self,
        *,
        command_name: str,
        json_mode: bool,
        output_file: Path | None = None,
        console: Console | None = None,
        style: OutputStyle = DEFAULT_STYLE,
    ) -> None:
        self.command_name = command_name
        self.json_mode = bool(json_mode)
        self.output_file = output_file
        self.console = console or get_shared_console()
        self.style = style

    def _write_mirror(self, text: str) -> None:
        if not self.output_file:
            return
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(text, encoding="utf-8")

    def _panel_border(self, level: Literal["info", "success", "warning", "error"]) -> str:
        if level == "success":
            return self.style.success_border
        if level == "warning":
            return self.style.warning_border
        if level == "error":
            return self.style.error_border
        return self.style.info_border

    def emit_json(self, payload: Any) -> None:
        text = json.dumps(payload, indent=2)
        # JSON mode must stay script-friendly: no ANSI decoration.
        self.console.out(text, highlight=False)
        self._write_mirror(text)

    def emit_text(self, text: str) -> None:
        text = str(text)
        self.console.print(text)
        self._write_mirror(text)

    def emit_panel(
        self,
        *,
        title: str,
        body: str,
        level: Literal["info", "success", "warning", "error"] = "info",
    ) -> None:
        if self.json_mode:
            self.emit_text(body)
            return
        panel = Panel(body, title=title, border_style=self._panel_border(level))
        self.console.print(panel)
        self._write_mirror(body)

    def emit_kv(self, *, title: str, items: list[tuple[str, str]]) -> None:
        if self.json_mode:
            payload = {key: value for key, value in items}
            self.emit_json(payload)
            return

        table = Table(title=title, show_header=False, border_style=self.style.info_border)
        table.add_column("key", style=self.style.key_style, no_wrap=True)
        table.add_column("value")
        for key, value in items:
            table.add_row(str(key), str(value))
        self.console.print(table)
        self._write_mirror("\n".join(f"{k}: {v}" for k, v in items))

    def emit_table(self, *, title: str, columns: list[str], rows: list[list[str]]) -> None:
        if self.json_mode:
            payload = [dict(zip(columns, row, strict=False)) for row in rows]
            self.emit_json(payload)
            return

        table = Table(title=title, border_style=self.style.info_border)
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*[str(cell) for cell in row])
        self.console.print(table)

        text_lines = [" | ".join(columns)]
        for row in rows:
            text_lines.append(" | ".join(str(cell) for cell in row))
        self._write_mirror("\n".join(text_lines))

    def emit_warning(self, text: str) -> None:
        self.emit_panel(title="Warning", body=text, level="warning")

    def emit_error(self, text: str) -> None:
        self.emit_panel(title="Error", body=text, level="error")

    def emit_success(self, text: str) -> None:
        if self.json_mode:
            self.emit_text(text)
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        body = Text(f"[{stamp}] {text}", style="bold green")
        self.console.print(Panel(body, title="Success", border_style=self.style.success_border))
        self._write_mirror(text)


def build_output_sink(
    *,
    command_name: str,
    json_mode: bool = False,
    output_file: Path | None = None,
    console: Console | None = None,
) -> OutputSink:
    return OutputSink(
        command_name=command_name,
        json_mode=json_mode,
        output_file=output_file,
        console=console,
    )
