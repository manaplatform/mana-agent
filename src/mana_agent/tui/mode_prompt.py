"""Plan approval and model-selected clarification in the terminal UI."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Markdown, Select, Static


class ModePrompt(ModalScreen[tuple[str, Any] | None]):
    DEFAULT_CSS = """
    ModePrompt { align: center middle; }
    ModePrompt > VerticalScroll {
        width: 82%;
        max-height: 85%;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }
    ModePrompt Button { margin-right: 1; }
    """

    def __init__(self, prompt: dict[str, Any]) -> None:
        super().__init__()
        self.prompt = prompt

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            if self.prompt["kind"] == "mode.approval":
                yield Label(f"Plan revision {self.prompt['revision']}")
                yield Markdown(self.prompt["preview"])
                yield Input(placeholder="Describe changes if you want a revision", id="revision-request")
                with Horizontal():
                    yield Button("Approve", variant="primary", id="approve")
                    yield Button("Revise", id="revise")
                    yield Button("Reject", variant="error", id="reject")
                    yield Button("Later", id="later")
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
                with Horizontal():
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
                self.query_one("#answer-error", Static).update("Answer every question before submitting.")
                return
            answers[question["field_id"]] = custom or selected
        self.dismiss(("answer", answers))
