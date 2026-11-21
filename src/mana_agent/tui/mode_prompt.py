"""Plan approval and model-selected clarification in the terminal UI."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import HorizontalGroup, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Markdown, Select, Static


class ModePrompt(ModalScreen[tuple[str, Any] | None]):
    AUTO_FOCUS = "#mode-content"

    DEFAULT_CSS = """
    ModePrompt { align: center middle; }
    ModePrompt > #mode-dialog {
        width: 100%;
        max-width: 88;
        height: 90%;
        max-height: 32;
        background: $surface;
        border: round $primary;
        padding: 0 1;
    }
    ModePrompt #mode-content { height: 1fr; }
    ModePrompt Button { width: 1fr; min-width: 0; height: 3; }
    """

    def __init__(self, prompt: dict[str, Any]) -> None:
        super().__init__()
        self.prompt = prompt

    def compose(self) -> ComposeResult:
        approval = self.prompt["kind"] == "mode.approval"
        with Vertical(id="mode-dialog"):
            with VerticalScroll(id="mode-content"):
                if approval:
                    yield Label(f"Plan revision {self.prompt['revision']}")
                    yield Markdown(self.prompt["preview"])
                    yield Input(placeholder="Describe changes if you want a revision", id="revision-request")
                else:
                    yield Label("Clarification")
                    for index, question in enumerate(self.prompt["questions"]):
                        yield Static(question["prompt"], markup=False)
                        choices = [
                            (f"{label}{' (Recommended)' if label == question['recommended_choice'] else ''} · {description}", label)
                            for label, description in question["choices"].items()
                        ]
                        if choices:
                            yield Select(choices, prompt="Select an answer", id=f"answer-{index}")
                        if question["allow_free_form"]:
                            yield Input(placeholder="Custom answer" if choices else "Answer", id=f"custom-{index}")
                    yield Static("", id="answer-error", markup=False)
            if approval:
                with HorizontalGroup():
                    yield Button("Approve", variant="primary", id="approve")
                    yield Button("Revise", id="revise")
                with HorizontalGroup():
                    yield Button("Reject", variant="error", id="reject")
                    yield Button("Later", id="later")
            else:
                with HorizontalGroup():
                    yield Button("Submit", variant="primary", id="submit")
                    yield Button("Later", id="later")

    @on(Button.Pressed)
    def respond(self, event: Button.Pressed) -> None:
        action = event.button.id
        if action == "later":
            self.dismiss(None)
            return
        if action in {"approve", "reject"}:
            self.dismiss((action, None))
            return
        if action == "revise":
            revision = self.query_one("#revision-request", Input).value.strip()
            if revision:
                self.dismiss(("revise", revision))
            else:
                self.query_one("#revision-request", Input).focus()
            return
        if action != "submit":
            return
        answers = {}
        for index, question in enumerate(self.prompt["questions"]):
            custom = self.query_one(f"#custom-{index}", Input).value.strip() if question["allow_free_form"] else ""
            selected = self.query_one(f"#answer-{index}", Select).value if question["choices"] else Select.BLANK
            if not custom and selected is Select.BLANK:
                error = self.query_one("#answer-error", Static)
                error.update("Answer every question before submitting.")
                error.scroll_visible()
                return
            answers[question["field_id"]] = custom or selected
        self.dismiss(("answer", answers))
