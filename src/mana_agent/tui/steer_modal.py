"""Modal dialog for Live Steer Mode in Mana Chat TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


class SteerModal(ModalScreen[str | None]):
    """Modal screen allowing the user to provide live steering instructions to active execution."""

    CSS = """
    SteerModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.65);
    }
    #steer-dialog {
        width: 72;
        height: auto;
        padding: 1 2;
        border: round #6366f1;
        background: #111827;
    }
    #steer-title {
        text-style: bold;
        color: #e2e8f0;
        margin-bottom: 1;
    }
    #steer-input {
        margin: 1 0;
        border: tall #4f46e5;
    }
    #steer-hints {
        color: #94a3b8;
    }
    .steer-actions {
        margin-top: 1;
        height: 3;
        align-horizontal: right;
    }
    .steer-actions Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="steer-dialog"):
            yield Static("⚡ Steer Active Execution", id="steer-title")
            yield Input(
                placeholder="> Prioritize security tests first",
                id="steer-input",
            )
            with Horizontal(classes="steer-actions"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button("Submit", variant="primary", id="btn-submit")
            yield Static("Enter: Submit   •   Esc: Cancel", id="steer-hints")

    def on_mount(self) -> None:
        try:
            self.query_one("#steer-input", Input).focus()
        except Exception:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        val = event.value.strip()
        self.dismiss(val if val else None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-submit":
            try:
                inp = self.query_one("#steer-input", Input)
                val = inp.value.strip()
                self.dismiss(val if val else None)
            except Exception:
                self.dismiss(None)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
