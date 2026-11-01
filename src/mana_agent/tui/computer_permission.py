"""Trusted local permission prompt for a pending computer action."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, HorizontalScroll, Vertical, VerticalScroll
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
        expires_at: datetime | None = None,
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
        self.expires_at = expires_at


class InlineApprovalDecided(Message):
    """Deliver an externally recorded decision to the UI thread."""

    def __init__(self, request_ids: set[str], approved: bool) -> None:
        super().__init__()
        self.request_ids = request_ids
        self.approved = approved


ApprovalDecided = InlineApprovalDecided


class InlineApproval(Vertical):
    """One request's controls in the shared approval drawer."""

    DEFAULT_CSS = """
    InlineApproval {
        height: auto;
        padding: 1 0;
        border-bottom: solid #334155;
    }
    InlineApproval .inline-approval-title {
        text-style: bold;
        color: #fcd34d;
        width: 100%;
    }
    InlineApproval .inline-approval-preview {
        color: #e5e7eb;
        width: 100%;
    }
    InlineApproval .inline-approval-scope {
        color: #93c5fd;
        width: 100%;
    }
    InlineApproval .inline-approval-actions {
        height: auto;
        margin-top: 1;
    }
    InlineApproval Button {
        margin-right: 1;
    }
    """

    def __init__(self, request: ComputerPermissionRequested, callback, on_popup: Any = None) -> None:
        super().__init__()
        self.request = request
        self.callback = callback
        self.on_popup = on_popup
        self.add_class("inline-approval")

    def on_mount(self) -> None:
        if getattr(self.request, "expires_at", None):
            now = datetime.now(timezone.utc)
            remaining = (self.request.expires_at - now).total_seconds()
            if remaining <= 0:
                self.post_message(InlineApprovalDecided({self.request.request_id}, approved=False))
            elif remaining <= 86400:
                self.set_timer(remaining, self._on_expired)

    def _on_expired(self) -> None:
        self.post_message(InlineApprovalDecided({self.request.request_id}, approved=False))

    def _title(self) -> str:
        request = self.request
        if request.transactional:
            return "Transactional action approval required"
        if request.server:
            return "Server action approval required"
        if request.api:
            return "API request approval required"
        if request.remote:
            return "Mana-Agent needs remote SSH permission"
        return "Mana-Agent needs computer permission"

    def compose(self) -> ComposeResult:
        yield Static(self._title(), classes="inline-approval-title")
        yield Static(self.request.preview, markup=False, classes="inline-approval-preview")
        yield Static(f"Scope: {self.request.scope}", markup=False, classes="inline-approval-scope")
        with HorizontalScroll(classes="inline-approval-actions"):
            yield Button("Deny", classes="inline-deny", variant="error")
            yield Button("Allow once", classes="inline-once", variant="primary")
            if self.request.transactional:
                yield Button("Allow all matching queued", classes="inline-all", variant="success")
            if not self.request.server and not self.request.api and not self.request.transactional:
                yield Button("This session", classes="inline-session", variant="success")
                yield Button("Always", classes="inline-always", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not any(event.button.has_class(name) for name in ("inline-deny", "inline-once", "inline-all", "inline-session", "inline-always")):
            return
        event.stop()
        for button in self.query(Button):
            button.disabled = True
        decision = None if event.button.has_class("inline-deny") else (
            PermissionDecision.ALLOW_SESSION if event.button.has_class("inline-session") else
            PermissionDecision.ALWAYS_ALLOW if event.button.has_class("inline-always") else
            PermissionDecision.ALLOW_ONCE
        )
        self.callback(ComputerPermissionChoice(
            self.request.request_id,
            decision,
            remote=self.request.remote,
            server=self.request.server,
            api=self.request.api,
            transactional=self.request.transactional,
            approve_all=event.button.has_class("inline-all"),
        ))

    def enable_choices(self) -> None:
        for button in self.query(Button):
            button.disabled = False


class ComputerPermissionScreen(ModalScreen[ComputerPermissionChoice]):
    """Modal screen for computer and transactional approval decisions."""

    BINDINGS = [
        Binding("escape", "dismiss_deny", "Deny", show=False),
    ]

    CSS = """
    ComputerPermissionScreen { align: center middle; }
    #computer-permission-dialog {
        width: 76;
        height: 18;
        padding: 1 2;
        border: round #f59e0b;
        background: #111827;
    }
    #computer-permission-title {
        color: #fcd34d;
        text-style: bold;
        height: 1;
    }
    #computer-permission-scroll {
        height: 1fr;
        margin: 1 0;
    }
    #computer-permission-preview {
        color: #e5e7eb;
    }
    #computer-permission-scope {
        color: #93c5fd;
        height: 1;
        margin-bottom: 1;
    }
    .computer-permission-actions {
        height: 3;
        align-horizontal: right;
    }
    .computer-permission-actions Button {
        margin-left: 1;
    }
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
        expires_at: datetime | None = None,
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
        self.expires_at = expires_at

    @property
    def request(self) -> ComputerPermissionRequested:
        return ComputerPermissionRequested(
            request_id=self.request_id,
            scope=self.scope,
            preview=self.preview,
            remote=self.remote,
            server=self.server,
            api=self.api,
            transactional=self.transactional,
            pending_count=self.pending_count,
            expires_at=self.expires_at,
        )

    def on_mount(self) -> None:
        if self.expires_at is not None:
            now = datetime.now(timezone.utc)
            remaining = (self.expires_at - now).total_seconds()
            if remaining <= 0:
                self.dismiss(None)
            elif remaining <= 86400:
                self.set_timer(remaining, self._on_expired)

    def _on_expired(self) -> None:
        self.dismiss(None)

    def action_dismiss_deny(self) -> None:
        self.dismiss(
            ComputerPermissionChoice(
                self.request_id,
                None,
                self.remote,
                self.server,
                self.api,
                self.transactional,
            )
        )

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
            yield Label(title, id="computer-permission-title")
            with VerticalScroll(id="computer-permission-scroll"):
                yield Static(self.preview, id="computer-permission-preview")
            yield Static(f"Scope: {self.scope}", id="computer-permission-scope")
            with Horizontal(classes="computer-permission-actions modal-actions"):
                yield Button("Deny", id="permission-deny", classes="inline-deny", variant="error")
                yield Button("Allow once", id="permission-once", classes="inline-once", variant="primary")
                if self.transactional:
                    yield Button("Allow all matching queued", id="permission-all", classes="inline-all", variant="success")
                if not self.server and not self.api and not self.transactional:
                    yield Button("This session", id="permission-session", classes="inline-session", variant="success")
                    yield Button("Always", id="permission-always", classes="inline-always", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button
        if btn.id == "permission-all" or btn.has_class("inline-all"):
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
        if btn.id == "permission-deny" or btn.has_class("inline-deny"):
            self.dismiss(
                ComputerPermissionChoice(
                    self.request_id,
                    None,
                    self.remote,
                    self.server,
                    self.api,
                    self.transactional,
                )
            )
            return
        if btn.id == "permission-once" or btn.has_class("inline-once"):
            self.dismiss(
                ComputerPermissionChoice(
                    self.request_id,
                    PermissionDecision.ALLOW_ONCE,
                    self.remote,
                    self.server,
                    self.api,
                    self.transactional,
                )
            )
            return
        if btn.id == "permission-session" or btn.has_class("inline-session"):
            self.dismiss(
                ComputerPermissionChoice(
                    self.request_id,
                    PermissionDecision.ALLOW_SESSION,
                    self.remote,
                    self.server,
                    self.api,
                    self.transactional,
                )
            )
            return
        if btn.id == "permission-always" or btn.has_class("inline-always"):
            self.dismiss(
                ComputerPermissionChoice(
                    self.request_id,
                    PermissionDecision.ALWAYS_ALLOW,
                    self.remote,
                    self.server,
                    self.api,
                    self.transactional,
                )
            )
            return

    def disable_choices(self) -> None:
        for button in self.query(Button):
            button.disabled = True

    def enable_choices(self) -> None:
        for button in self.query(Button):
            button.disabled = False


ApprovalToast = ComputerPermissionScreen
InlineApproval = ComputerPermissionScreen
