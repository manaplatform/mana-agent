from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class CliCapabilityEntry(BaseModel):
    cli_command: str
    chat_status: Literal["direct", "adapter", "unavailable"]
    chat_command: str | None = None
    reason: str


def build_cli_capability_matrix(app=None) -> list[CliCapabilityEntry]:  # noqa: ANN001
    """Inventory every public Typer leaf and state its chat disposition."""
    if app is None:
        from mana_agent.commands.cli import app as canonical_app

        app = canonical_app
    import typer

    from mana_agent.chat_commands.registry import build_default_registry

    click_root = typer.main.get_command(app)
    registry = build_default_registry()
    adapters = {
        "connector telegram setup": "/connect telegram",
        "connector telegram status": "/connect telegram status",
        "connector telegram test": "/connect telegram test",
        "connector telegram start": "/connect telegram start",
        "connector telegram stop": "/connect telegram stop",
        "session new": "/new",
        "session list": "/sessions list",
        "session show": "/sessions show",
        "session switch": "/sessions switch",
    }
    rows: list[CliCapabilityEntry] = []

    def visit(command, prefix: tuple[str, ...] = ()) -> None:  # noqa: ANN001
        children = getattr(command, "commands", None)
        if children:
            for name, child in sorted(children.items()):
                visit(child, (*prefix, name))
            return
        path = " ".join(prefix)
        root_name = prefix[0] if prefix else ""
        definition = registry.resolve(root_name)
        if definition is not None:
            rows.append(CliCapabilityEntry(
                cli_command=path, chat_status="direct", chat_command=f"/{definition.canonical_name}",
                reason="Uses the shared typed command handler.",
            ))
        elif path in adapters:
            rows.append(CliCapabilityEntry(
                cli_command=path, chat_status="adapter", chat_command=adapters[path],
                reason="Uses a safe chat-specific application-service adapter.",
            ))
        else:
            rows.append(CliCapabilityEntry(
                cli_command=path, chat_status="unavailable",
                reason="This CLI surface requires flags, files, or an interactive terminal contract that has no validated chat argument schema.",
            ))

    visit(click_root)
    return rows
