"""Trusted local permission prompt for a pending computer action."""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from mana_agent.integrations.computer_control.models import PermissionDecision


@dataclass(frozen=True, slots=True)
class ComputerPermissionChoice:
    request_id: str
    decision: PermissionDecision | None
    remote: bool = False
    server: bool = False
    api: bool = False
    transactional: bool = False
    approve_all: bool = False
    expected_version: int | None = None


class ComputerPermissionRequested(Message):
    """Non-blocking cross-thread request delivered through Textual's message pump."""

    def __init__(
        self,
        *,
        request_id: str,
        scope: str,
        preview: str,
        remote: bool = False,
        server: bool = False,
        api: bool = False,
        transactional: bool = False,
        pending_count: int = 1,
    ) -> None:
        super().__init__()
        self.request_id = request_id
        self.scope = scope
        self.preview = preview
        self.remote = remote
        self.server = server
        self.api = api
        self.transactional = transactional
        self.pending_count = pending_count


class InlineApproval(Vertical):
    """Action controls embedded in an existing activity box, not a dialog."""

    DEFAULT_CSS = """
    InlineApproval { height: auto; }
    InlineApproval .inline-approval-actions { height: auto; }
    InlineApproval Button { margin-right: 1; }
    """

    def __init__(self, request: ComputerPermissionRequested, callback) -> None:
        super().__init__()
        self.request = request
        self.callback = callback

    def compose(self) -> ComposeResult:
        yield Static(self.request.preview, markup=False)
        yield Static(f"Scope: {self.request.scope}", markup=False)
        with Horizontal(classes="inline-approval-actions"):
            yield Button("Deny", classes="inline-deny", variant="error")
            yield Button("Allow once", classes="inline-once", variant="primary")
            if self.request.transactional:
                yield Button("Allow all matching queued", classes="inline-all", variant="success")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not any(event.button.has_class(name) for name in ("inline-deny", "inline-once", "inline-all")):
            return
        event.stop()
        for button in self.query(Button):
            button.disabled = True
        self.callback(ComputerPermissionChoice(
            self.request.request_id,
            None if event.button.has_class("inline-deny") else PermissionDecision.ALLOW_ONCE,
            server=self.request.server,
            api=self.request.api,
            transactional=self.request.transactional,
            approve_all=event.button.has_class("inline-all"),
        ))

    def enable_choices(self) -> None:
        for button in self.query(Button):
            button.disabled = False


class ComputerPermissionScreen(ModalScreen[ComputerPermissionChoice]):
    CSS = """
    ComputerPermissionScreen { align: center middle; }
    #computer-permission-dialog {
        width: 76;
        height: auto;
        padding: 1 2;
        border: round #f59e0b;
        background: #111827;
    }
    #computer-permission-preview { margin: 1 0; color: #e5e7eb; }
    #computer-permission-scope { color: #93c5fd; }
    .computer-permission-actions { height: 3; align-horizontal: right; }
    .computer-permission-actions Button { margin-left: 1; }
    """

    def __init__(
        self,
        *,
        request_id: str,
        scope: str,
        preview: str,
        remote: bool = False,
        server: bool = False,
        api: bool = False,
        transactional: bool = False,
        pending_count: int = 1,
    ) -> None:
        super().__init__()
        self.request_id = request_id
        self.scope = scope
        self.preview = preview
        self.remote = remote
        self.server = server
        self.api = api
        self.transactional = transactional
        self.pending_count = pending_count

    def compose(self) -> ComposeResult:
        with Vertical(id="computer-permission-dialog"):
            title = (
                "Mana-Agent needs server action approval"
                if self.server
                else "Mana-Agent needs transactional action approval"
                if self.transactional
                else "Mana-Agent needs API request approval"
                if self.api
                else "Mana-Agent needs remote SSH permission"
                if self.remote
                else "Mana-Agent needs computer permission"
            )
            yield Label(title)
            yield Static(self.preview, id="computer-permission-preview")
            yield Static(f"Scope: {self.scope}", id="computer-permission-scope")
            with Horizontal(classes="computer-permission-actions"):
                yield Button("Deny", id="permission-deny", variant="error")
                yield Button("Allow once", id="permission-once", variant="primary")
                if self.transactional:
                    yield Button("Allow all matching queued", id="permission-all", variant="success")
                if not self.server and not self.api and not self.transactional:
                    yield Button("This session", id="permission-session", variant="success")
                    yield Button("Always", id="permission-always", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "permission-all":
            self.dismiss(
                ComputerPermissionChoice(
                    self.request_id,
                    PermissionDecision.ALLOW_ONCE,
                    self.remote,
                    self.server,
                    self.api,
                    self.transactional,
                    approve_all=True,
                )
            )
            return
        decisions = {
            "permission-deny": None,
            "permission-once": PermissionDecision.ALLOW_ONCE,
            "permission-session": PermissionDecision.ALLOW_SESSION,
            "permission-always": PermissionDecision.ALWAYS_ALLOW,
        }
        if event.button.id in decisions:
            self.dismiss(
                ComputerPermissionChoice(
                    self.request_id,
                    decisions[event.button.id],
                    self.remote,
                    self.server,
                    self.api,
                    self.transactional,
                )
            )
