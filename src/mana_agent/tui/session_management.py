from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ListItem, ListView, Select, Static


@dataclass(frozen=True, slots=True)
class SessionAction:
    action: str
    session_id: str
    title: str = ""


class SessionPickerScreen(ModalScreen[SessionAction | None]):
    CSS = """
    SessionPickerScreen { align: center middle; }
    #session-dialog { width: 88; height: 82%; padding: 1 2; border: round #6366f1; background: #111827; }
    #session-list { height: 1fr; border: round #334155; }
    .actions { height: 3; align-horizontal: right; }
    .actions Button { margin-left: 1; }
    """

    def __init__(self, sessions: list[dict[str, Any]]) -> None:
        super().__init__()
        self.sessions = sessions
        self._delete_armed = False

    def compose(self) -> ComposeResult:
        with Vertical(id="session-dialog"):
            yield Label("Chats")
            yield Input(placeholder="Search sessions…", id="session-search")
            yield ListView(id="session-list")
            yield Input(placeholder="New title for selected chat", id="session-title")
            yield Static("Select a chat to switch, rename, or delete.", id="session-note")
            with Horizontal(classes="actions"):
                yield Button("Switch", id="session-switch", variant="primary")
                yield Button("Rename", id="session-rename")
                yield Button("Delete", id="session-delete", variant="error")
                yield Button("Close", id="session-close")

    def on_mount(self) -> None:
        self._render(self.sessions)
        self.query_one("#session-search", Input).focus()

    def _render(self, rows: list[dict[str, Any]]) -> None:
        view = self.query_one("#session-list", ListView)
        view.clear()
        self._visible = rows
        for row in rows:
            marker = "●" if row.get("current") else " "
            view.append(ListItem(Label(f"{marker} {row.get('title')} · {row.get('short_id')} · {row.get('status')} · {row.get('message_count')} messages")))

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "session-search":
            return
        needle = event.value.strip().lower()
        self._render([row for row in self.sessions if needle in f"{row.get('title')} {row.get('session_id')}".lower()])

    def _selected(self) -> dict[str, Any] | None:
        view = self.query_one("#session-list", ListView)
        if view.index is None or view.index >= len(self._visible):
            return None
        return self._visible[view.index]

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "session-close":
            self.dismiss(None)
            return
        selected = self._selected()
        if selected is None:
            self.query_one("#session-note", Static).update("Select a chat first.")
            return
        sid = str(selected["session_id"])
        if event.button.id == "session-switch":
            self.dismiss(SessionAction("switch", sid))
        elif event.button.id == "session-rename":
            self.dismiss(SessionAction("rename", sid, self.query_one("#session-title", Input).value))
        elif event.button.id == "session-delete":
            if not self._delete_armed:
                self._delete_armed = True
                self.query_one("#session-note", Static).update("Permanent deletion cannot be undone. Press Delete again to confirm.")
            else:
                self.dismiss(SessionAction("delete", sid))


@dataclass(frozen=True, slots=True)
class TelegramSetup:
    token: str
    transport: str
    repository: str
    allowed_users: list[int]
    allowed_chats: list[int]
    webhook_url: str
    secret_source: str


class TelegramSetupScreen(ModalScreen[TelegramSetup | None]):
    CSS = """
    TelegramSetupScreen { align: center middle; }
    #telegram-dialog { width: 84; height: auto; padding: 1 2; border: round #22c55e; background: #111827; }
    .actions { height: 3; align-horizontal: right; }
    """

    def __init__(self, repository: Path) -> None:
        super().__init__()
        self.repository = repository

    def compose(self) -> ComposeResult:
        with Vertical(id="telegram-dialog"):
            yield Label("Connect Telegram")
            yield Input(placeholder="Bot token", password=True, id="telegram-token")
            yield Select([("Auto", "auto"), ("Polling", "polling"), ("Webhook", "webhook")], value="auto", id="telegram-transport")
            yield Input(value=str(self.repository), placeholder="Repository", id="telegram-repository")
            yield Input(placeholder="Allowed user IDs, comma-separated", id="telegram-users")
            yield Input(placeholder="Allowed chat IDs, comma-separated", id="telegram-chats")
            yield Input(placeholder="Webhook public URL", id="telegram-webhook")
            yield Select([("OS keyring", "keyring"), ("Environment variable", "environment")], value="keyring", id="telegram-secret-source")
            yield Static("The token is validated but never added to chat history or events.", id="telegram-note")
            with Horizontal(classes="actions"):
                yield Button("Connect", id="telegram-connect", variant="success")
                yield Button("Cancel", id="telegram-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "telegram-cancel":
            self.dismiss(None)
            return
        if event.button.id != "telegram-connect":
            return
        def ids(widget_id: str) -> list[int]:
            return [int(item.strip()) for item in self.query_one(widget_id, Input).value.split(",") if item.strip()]
        try:
            setup = TelegramSetup(
                token=self.query_one("#telegram-token", Input).value,
                transport=str(self.query_one("#telegram-transport", Select).value),
                repository=self.query_one("#telegram-repository", Input).value,
                allowed_users=ids("#telegram-users"), allowed_chats=ids("#telegram-chats"),
                webhook_url=self.query_one("#telegram-webhook", Input).value,
                secret_source=str(self.query_one("#telegram-secret-source", Select).value),
            )
        except ValueError:
            self.query_one("#telegram-note", Static).update("User and chat IDs must be integers.")
            return
        if not setup.token.strip():
            self.query_one("#telegram-note", Static).update("Bot token is required.")
            return
        self.query_one("#telegram-token", Input).value = ""
        self.dismiss(setup)
