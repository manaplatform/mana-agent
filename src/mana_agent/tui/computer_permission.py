"""Trusted local permission prompt for a pending computer action."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from textual.app import ComposeResult
from textual.containers import HorizontalScroll, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Static

from mana_agent.chat.activity import safe_text
from mana_agent.tui.presentation import glyph
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

    def __init__(self, request_ids: set[str], approved: bool, *, expired: bool = False) -> None:
        super().__init__()
        self.request_ids = request_ids
        self.approved = approved
        self.expired = expired


ApprovalDecided = InlineApprovalDecided


class InlineApproval(Vertical):
    """One request's controls in the shared approval drawer."""


    DEFAULT_CSS = """
    InlineApproval { height: auto; padding: 0 1; background: $surface; }
    InlineApproval .inline-approval-title { text-style: bold; color: $warning; height: auto; }
    InlineApproval .inline-approval-preview { width: 1fr; height: auto; }
    InlineApproval VerticalScroll { height: auto; max-height: 5; }
    InlineApproval .inline-approval-scope { color: $text-muted; height: auto; }
    InlineApproval .inline-approval-actions { height: 2; }
    InlineApproval Button { height: 1; min-width: 1; padding: 0 1; border: none; }
    """

    def __init__(self, request: ComputerPermissionRequested, callback) -> None:
        super().__init__()
        self.request = request
        self.callback = callback
        self._pending_count = request.pending_count
        self._heading = Static(self._title(), classes="inline-approval-title", markup=False)
        self._choices = HorizontalScroll(classes="inline-approval-actions")

    def set_pending_count(self, count: int) -> None:
        self._pending_count = count
        self._heading.update(self._title())

    def focus_choices(self) -> None:
        button = next((control for control in self.query(Button) if not control.disabled), None)
        if button is not None:
            button.focus()

    def on_mount(self) -> None:
        if getattr(self.request, "expires_at", None):
            now = datetime.now(timezone.utc)
            remaining = (self.request.expires_at - now).total_seconds()
            if remaining <= 0:
                self.post_message(InlineApprovalDecided({self.request.request_id}, approved=False, expired=True))
            elif remaining <= 86400:
                self.set_timer(remaining, self._on_expired)

    def _on_expired(self) -> None:
        self.post_message(InlineApprovalDecided({self.request.request_id}, approved=False, expired=True))

    def _title(self) -> str:
        heading = self._request_title()
        pending = f" {glyph('·', '|')} {self._pending_count} pending" if self._pending_count > 1 else ""
        return f"{heading}{pending} {glyph('·', '|')} Ctrl+G for choices"

    def _request_title(self) -> str:
        request = self.request
        if request.transactional:
            return "Approve this action?"
        if request.server:
            return "Approve server action?"
        if request.api:
            return "Approve API request?"
        if request.remote:
            return "Approve remote SSH action?"
        return "Approve computer action?"

    def compose(self) -> ComposeResult:
        yield self._heading
        with VerticalScroll():
            yield Static(safe_text(self.request.preview, None), markup=False, classes="inline-approval-preview")
        yield Static(f"Scope: {safe_text(self.request.scope)}", markup=False, classes="inline-approval-scope")
        with self._choices:
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
