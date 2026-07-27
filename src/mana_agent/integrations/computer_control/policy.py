"""Action validation, risk policy, exact-action confirmation, and client trust."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from secrets import token_urlsafe
from urllib.parse import urlparse

from mana_agent.integrations.computer_control.config import ComputerControlSettings
from mana_agent.integrations.computer_control.errors import (
    ConfirmationRequired,
    InvalidActionDecision,
    InvalidConfirmation,
    RemoteControlDenied,
)
from mana_agent.integrations.computer_control.models import ComputerAction, ExecutionRisk


@dataclass(frozen=True, slots=True)
class ActionSpec:
    permission_scope: str
    risk: ExecutionRisk
    allowed_arguments: frozenset[str] = frozenset()
    sensitive_result: bool = False


ACTION_SPECS: dict[str, ActionSpec] = {
    "capabilities.discover": ActionSpec("computer.apps.read", ExecutionRisk.LOW),
    "applications.list": ActionSpec("computer.apps.read", ExecutionRisk.LOW),
    "applications.active": ActionSpec("computer.apps.read", ExecutionRisk.LOW),
    "applications.open": ActionSpec("computer.apps.control", ExecutionRisk.LOW),
    "applications.focus": ActionSpec("computer.apps.control", ExecutionRisk.LOW),
    "applications.close": ActionSpec("computer.apps.control", ExecutionRisk.MEDIUM, frozenset({"force"})),
    "calendar.list": ActionSpec("computer.calendar.read", ExecutionRisk.MEDIUM, frozenset({"starts_after", "ends_before", "calendar_id", "limit"}), True),
    "calendar.create": ActionSpec("computer.calendar.write", ExecutionRisk.MEDIUM, frozenset({"event"})),
    "calendar.update": ActionSpec("computer.calendar.write", ExecutionRisk.MEDIUM, frozenset({"event"})),
    "calendar.delete": ActionSpec("computer.calendar.write", ExecutionRisk.HIGH),
    "media.status": ActionSpec("computer.media.read", ExecutionRisk.LOW),
    "media.play": ActionSpec("computer.media.control", ExecutionRisk.LOW, frozenset({"query", "kind"})),
    "media.pause": ActionSpec("computer.media.control", ExecutionRisk.LOW),
    "media.resume": ActionSpec("computer.media.control", ExecutionRisk.LOW),
    "media.stop": ActionSpec("computer.media.control", ExecutionRisk.LOW),
    "media.next": ActionSpec("computer.media.control", ExecutionRisk.LOW),
    "media.previous": ActionSpec("computer.media.control", ExecutionRisk.LOW),
    "media.volume": ActionSpec("computer.media.control", ExecutionRisk.LOW, frozenset({"volume"})),
    "notes.search": ActionSpec("computer.notes.read", ExecutionRisk.MEDIUM, frozenset({"query", "limit"}), True),
    "notes.read": ActionSpec("computer.notes.read", ExecutionRisk.MEDIUM, frozenset(), True),
    "notes.create": ActionSpec("computer.notes.write", ExecutionRisk.MEDIUM, frozenset({"note"})),
    "notes.update": ActionSpec("computer.notes.write", ExecutionRisk.MEDIUM, frozenset({"note"})),
    "notes.delete": ActionSpec("computer.notes.write", ExecutionRisk.HIGH),
    "browser.active_page": ActionSpec("computer.browser.tabs.read", ExecutionRisk.MEDIUM, frozenset(), True),
    "browser.read_page": ActionSpec("computer.browser.page.read", ExecutionRisk.MEDIUM, frozenset(), True),
    "browser.list_tabs": ActionSpec("computer.browser.tabs.read", ExecutionRisk.MEDIUM, frozenset(), True),
    "browser.open_url": ActionSpec("computer.browser.control", ExecutionRisk.LOW),
    "browser.activate_tab": ActionSpec("computer.browser.control", ExecutionRisk.LOW),
    "browser.close_tab": ActionSpec("computer.browser.control", ExecutionRisk.MEDIUM),
    "clipboard.read": ActionSpec("computer.clipboard.read", ExecutionRisk.MEDIUM, frozenset(), True),
    "clipboard.write": ActionSpec("computer.clipboard.write", ExecutionRisk.MEDIUM, frozenset({"text"})),
    "clipboard.clear": ActionSpec("computer.clipboard.write", ExecutionRisk.MEDIUM),
    "filesystem.open": ActionSpec("computer.files.read", ExecutionRisk.LOW),
    "filesystem.reveal": ActionSpec("computer.files.read", ExecutionRisk.LOW),
    "filesystem.metadata": ActionSpec("computer.files.read", ExecutionRisk.LOW),
    "filesystem.copy": ActionSpec("computer.files.write", ExecutionRisk.MEDIUM, frozenset({"destination"})),
    "filesystem.move": ActionSpec("computer.files.write", ExecutionRisk.MEDIUM, frozenset({"destination"})),
    "filesystem.rename": ActionSpec("computer.files.write", ExecutionRisk.MEDIUM, frozenset({"destination"})),
    "filesystem.mkdir": ActionSpec("computer.files.write", ExecutionRisk.MEDIUM),
    "filesystem.trash": ActionSpec("computer.files.write", ExecutionRisk.HIGH),
    "screenshots.capture": ActionSpec("computer.screenshot.capture", ExecutionRisk.MEDIUM, frozenset({"mode"}), True),
    "notifications.send": ActionSpec("computer.notifications.send", ExecutionRisk.MEDIUM, frozenset({"title", "body"})),
    "system.status": ActionSpec("computer.system.read", ExecutionRisk.LOW),
    "system.volume": ActionSpec("computer.system.control", ExecutionRisk.MEDIUM, frozenset({"volume", "muted"})),
    "system.settings": ActionSpec("computer.system.control", ExecutionRisk.LOW, frozenset({"pane"})),
    "system.lock": ActionSpec("computer.system.control", ExecutionRisk.HIGH),
    "system.sleep": ActionSpec("computer.system.control", ExecutionRisk.CRITICAL),
    "system.restart": ActionSpec("computer.system.control", ExecutionRisk.CRITICAL),
    "system.shutdown": ActionSpec("computer.system.control", ExecutionRisk.CRITICAL),
}

REMOTE_CLIENTS = {"telegram", "slack", "teams", "remote_api", "a2a"}
REMOTE_SENSITIVE_SCOPES = {
    "computer.calendar.read", "computer.notes.read", "computer.browser.tabs.read",
    "computer.browser.page.read", "computer.clipboard.read", "computer.screenshot.capture",
}


class ConfirmationService:
    def __init__(self, *, ttl_seconds: int = 120) -> None:
        self.ttl_seconds = ttl_seconds
        self._tokens: dict[str, tuple[str, datetime, bool, str]] = {}

    @staticmethod
    def binding(action: ComputerAction) -> str:
        payload = action.model_dump(exclude={"confirmation_token"}, mode="json")
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def issue(self, action: ComputerAction, *, preview: str) -> str:
        token = token_urlsafe(24)
        self._tokens[token] = (
            self.binding(action),
            datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds),
            False,
            preview,
        )
        return token

    def approve(self, request_id: str) -> None:
        stored = self._tokens.get(request_id)
        if stored is None:
            raise InvalidConfirmation("Confirmation request is missing, unknown, or already used.")
        binding, expires_at, _approved, preview = stored
        if expires_at <= datetime.now(timezone.utc):
            self._tokens.pop(request_id, None)
            raise InvalidConfirmation("Confirmation request expired; review the action again.")
        self._tokens[request_id] = (binding, expires_at, True, preview)

    def pending(self) -> list[dict[str, str]]:
        now = datetime.now(timezone.utc)
        return [
            {
                "confirmation_request_id": request_id,
                "expires_at": expires_at.isoformat(),
                "preview": preview,
            }
            for request_id, (_binding, expires_at, approved, preview) in self._tokens.items()
            if not approved and expires_at > now
        ]

    def consume(self, action: ComputerAction) -> None:
        token = action.confirmation_token or ""
        stored = self._tokens.get(token)
        if stored is None:
            raise InvalidConfirmation("Confirmation token is missing, unknown, or already used.")
        binding, expires_at, approved, _preview = stored
        if expires_at <= datetime.now(timezone.utc):
            self._tokens.pop(token, None)
            raise InvalidConfirmation("Confirmation token expired; review the action again.")
        if not approved:
            raise InvalidConfirmation("Confirmation is still waiting for approval in a trusted local UI.")
        if binding != self.binding(action):
            raise InvalidConfirmation("Confirmation token does not match this exact action.")
        self._tokens.pop(token, None)


class ExecutionPolicy:
    def __init__(self, settings: ComputerControlSettings, confirmations: ConfirmationService | None = None) -> None:
        self.settings = settings
        self.confirmations = confirmations or ConfirmationService()

    def validate_action(self, action: ComputerAction) -> ActionSpec:
        spec = ACTION_SPECS.get(action.operation)
        if spec is None:
            raise InvalidActionDecision(f"Operation {action.operation!r} is not registered; no fallback action was executed.")
        if action.permission_scope != spec.permission_scope or action.risk is not spec.risk:
            raise InvalidActionDecision(
                f"Action decision does not match the registered permission/risk contract for {action.operation!r}."
            )
        unknown = set(action.arguments) - spec.allowed_arguments
        if unknown:
            raise InvalidActionDecision(f"Unsupported action arguments: {', '.join(sorted(unknown))}.")
        if action.operation == "browser.open_url":
            parsed = urlparse(action.target.url or "")
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise InvalidActionDecision("Browser URLs must be absolute HTTP(S) URLs.")
        return spec

    def validate_client(self, action: ComputerAction, *, client_type: str, locally_confirmed: bool) -> None:
        if client_type not in self.settings.allowed_clients:
            raise RemoteControlDenied(f"Client {client_type!r} is not allowed to control this computer.")
        remote = client_type in REMOTE_CLIENTS
        if remote and not self.settings.allow_remote_control:
            raise RemoteControlDenied("Remote computer control is disabled.")
        if remote and action.permission_scope in REMOTE_SENSITIVE_SCOPES:
            if action.permission_scope not in self.settings.remote_sensitive_scopes:
                raise RemoteControlDenied(f"Remote access to {action.permission_scope!r} is disabled.")
        # High/critical remote requests continue only as pending confirmations.
        # ConfirmationService can be approved solely through a trusted local UI;
        # no remote client can set ``locally_confirmed`` through model arguments.

    def require_confirmation(self, action: ComputerAction) -> None:
        # High and critical confirmation is a security invariant, not a
        # user-disableable convenience setting.
        required = action.risk in {ExecutionRisk.HIGH, ExecutionRisk.CRITICAL}
        if not required:
            return
        if action.confirmation_token:
            self.confirmations.consume(action)
            return
        preview = self.preview(action)
        request_id = self.confirmations.issue(action, preview=preview)
        raise ConfirmationRequired(
            "This action requires exact-action confirmation immediately before execution.",
            preview=preview,
            confirmation_request_id=request_id,
        )

    @staticmethod
    def preview(action: ComputerAction) -> str:
        target = (
            action.target.application_id or action.target.resource_id or action.target.path
            or action.target.url or "the selected computer"
        )
        return f"Mana-Agent wants to {action.operation.replace('.', ' ')} on {target}. Risk: {action.risk.value}."
