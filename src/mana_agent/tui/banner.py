from __future__ import annotations

from rich.console import Console

from mana_agent.ui.banner import render_banner as _render_banner


def render_banner(console: Console | None = None) -> None:
    _render_banner(console)
