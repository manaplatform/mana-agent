"""Single execution service used by every Mana-Agent client."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from secrets import token_urlsafe
from typing import Callable

from mana_agent.integrations.computer_control.audit import AuditLogger
from mana_agent.integrations.computer_control.config import ComputerControlSettings
from mana_agent.integrations.computer_control.contracts import ComputerControlProvider
from mana_agent.integrations.computer_control.discovery import select_provider
from mana_agent.integrations.computer_control.errors import (
    ActionCancelled,
    ActionTimedOut,
    ComputerControlDisabled,
    ComputerControlError,
    CapabilityUnavailable,
    ConfirmationRequired,
    InvalidConfirmation,
    ManaPermissionDenied,
    OperatingSystemPermissionDenied,
    PermissionApprovalRequired,
)
from mana_agent.integrations.computer_control.events import publish_computer_event
from mana_agent.integrations.computer_control.models import (
    AuditRecord,
    CapabilityReport,
    ComputerAction,
    ComputerActionResult,
    ComputerControlEvent,
    ExecutionState,
    PermissionDecision,
    PermissionStatus,
)
from mana_agent.integrations.computer_control.permissions import PermissionService
from mana_agent.integrations.computer_control.policy import ExecutionPolicy
from mana_agent.integrations.computer_control.registry import ApplicationAdapterRegistry


@dataclass(frozen=True, slots=True)
class PendingPermissionAction:
    action: ComputerAction
    session_id: str
    client_type: str
    expires_at: datetime
    preview: str


class ComputerControlService:
    def __init__(
        self,
        *,
        settings: ComputerControlSettings | None = None,
        provider: ComputerControlProvider | None = None,
        permissions: PermissionService | None = None,
        policy: ExecutionPolicy | None = None,
        audit: AuditLogger | None = None,
        event_sink: Callable[[ComputerControlEvent], None] = publish_computer_event,
        adapters: ApplicationAdapterRegistry | None = None,
    ) -> None:
        self.settings = settings or ComputerControlSettings.load()
        self.provider = provider or select_provider(self.settings)
        self.permissions = permissions or PermissionService(self.settings)
        self.policy = policy or ExecutionPolicy(self.settings)
        self.audit = audit or AuditLogger(
            enabled=self.settings.audit_enabled,
            retention_days=self.settings.audit_retention_days,
        )
        self.event_sink = event_sink
        self.adapters = adapters or ApplicationAdapterRegistry()
        self._tasks: dict[str, asyncio.Task[ComputerActionResult]] = {}
        self._execution_sessions: dict[str, str] = {}
        self._pending_confirmation_actions: dict[str, tuple[ComputerAction, str, str]] = {}
        self._pending_permission_actions: dict[str, PendingPermissionAction] = {}

    def _ensure_enabled(self) -> None:
        if not self.settings.enabled:
            raise ComputerControlDisabled(
                "Computer control is disabled by default.",
                corrective_action="Set [computer_control].enabled = true in ~/.mana/config.toml.",
            )

    def _emit(self, action: ComputerAction, event_type: str, state: ExecutionState, message: str, **metadata: object) -> None:
        # Metadata is deliberately limited to non-content identifiers/state.
        safe = {key: value for key, value in metadata.items() if key in {
            "capability", "operation", "application_id", "permission_scope", "risk", "provider",
            "permission_request_id", "preview",
        }}
        self.event_sink(ComputerControlEvent(
            event_type=event_type,
            execution_id=action.execution_id,
            state=state,
            message=message,
            metadata=safe,
        ))

    async def capabilities(self) -> CapabilityReport:
        self._ensure_enabled()
        placeholder = ComputerAction(
            capability="applications",
            operation="capabilities.discover",
            permission_scope="computer.apps.read",
            risk="low",
            source_decision_id="system:capability-discovery",
        )
        self._emit(placeholder, "capability_discovery", ExecutionState.RUNNING, "Discovering computer capabilities")
        report = await self.provider.discover_capabilities()
        if not self.permissions.status("computer.apps.read").granted:
            report = report.model_copy(update={"applications": []})
        self._emit(placeholder, "capability_discovery", ExecutionState.COMPLETED, "Computer capability discovery completed", provider=report.provider)
        return report

    async def permission_status(self, scope: str) -> PermissionStatus:
        self._ensure_enabled()
        return self.permissions.status(scope)

    def pending_confirmations(self) -> list[dict[str, str]]:
        return self.policy.confirmations.pending()

    def pending_permissions(self) -> list[dict[str, str]]:
        now = datetime.now(timezone.utc)
        expired = [
            request_id
            for request_id, pending in self._pending_permission_actions.items()
            if pending.expires_at <= now
        ]
        for request_id in expired:
            self._pending_permission_actions.pop(request_id, None)
        return [
            {
                "permission_request_id": request_id,
                "permission_scope": pending.action.permission_scope,
                "preview": pending.preview,
                "expires_at": pending.expires_at.isoformat(),
                "requesting_client": pending.client_type,
            }
            for request_id, pending in self._pending_permission_actions.items()
        ]

    def _request_permission(
        self,
        action: ComputerAction,
        *,
        session_id: str,
        client_type: str,
    ) -> PermissionApprovalRequired:
        request_id = token_urlsafe(24)
        preview = self.policy.preview(action)
        self._pending_permission_actions[request_id] = PendingPermissionAction(
            action=action.model_copy(deep=True),
            session_id=session_id,
            client_type=client_type,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=120),
            preview=preview,
        )
        self._emit(
            action,
            "waiting_permission",
            ExecutionState.WAITING_PERMISSION,
            preview,
            permission_scope=action.permission_scope,
            permission_request_id=request_id,
            preview=preview,
        )
        return PermissionApprovalRequired(
            "This permission is waiting for a decision in the active local TUI or dashboard.",
            permission_request_id=request_id,
            permission_scope=action.permission_scope,
            preview=preview,
            execution_id=action.execution_id,
        )

    async def approve_permission_and_execute(
        self,
        request_id: str,
        *,
        decision: PermissionDecision,
        client_type: str,
    ) -> ComputerActionResult:
        if client_type not in {"local_cli", "tui", "dashboard"}:
            raise OperatingSystemPermissionDenied(
                "Computer permission decisions must come from a trusted local client."
            )
        if decision not in {
            PermissionDecision.ALLOW_ONCE,
            PermissionDecision.ALLOW_SESSION,
            PermissionDecision.ALWAYS_ALLOW,
        }:
            raise OperatingSystemPermissionDenied(
                "Choose allow once, allow for this session, or always allow."
            )
        pending = self._pending_permission_actions.get(request_id)
        if pending is None:
            raise ManaPermissionDenied("No computer action is waiting for that permission request.")
        if pending.expires_at <= datetime.now(timezone.utc):
            self._pending_permission_actions.pop(request_id, None)
            raise ManaPermissionDenied("The permission request expired; ask Mana-Agent to try the action again.")
        approved = False
        try:
            self.permissions.grant(pending.action.permission_scope, decision)
            approved = True
            return await self.execute(
                pending.action,
                session_id=pending.session_id,
                client_type=pending.client_type,
            )
        finally:
            if approved:
                self._pending_permission_actions.pop(request_id, None)

    def deny_permission_request(self, request_id: str, *, client_type: str) -> None:
        if client_type not in {"local_cli", "tui", "dashboard"}:
            raise OperatingSystemPermissionDenied(
                "Computer permission decisions must come from a trusted local client."
            )
        if self._pending_permission_actions.pop(request_id, None) is None:
            raise ManaPermissionDenied("No computer action is waiting for that permission request.")

    def approve_confirmation(self, request_id: str, *, client_type: str, explicitly_confirmed: bool) -> None:
        if client_type not in {"local_cli", "tui", "dashboard"}:
            raise OperatingSystemPermissionDenied("Computer-action confirmation must come from a trusted local client.")
        if not explicitly_confirmed:
            raise OperatingSystemPermissionDenied("Explicit confirmation was not supplied.")
        self.policy.confirmations.approve(request_id)

    async def approve_and_execute(
        self,
        request_id: str,
        *,
        client_type: str,
        explicitly_confirmed: bool,
    ) -> ComputerActionResult:
        pending = self._pending_confirmation_actions.get(request_id)
        if pending is None:
            raise InvalidConfirmation("No exact computer action is pending for this confirmation request.")
        action, session_id, original_client_type = pending
        approved = False
        try:
            self.approve_confirmation(
                request_id,
                client_type=client_type,
                explicitly_confirmed=explicitly_confirmed,
            )
            approved = True
            confirmed = action.model_copy(update={"confirmation_token": request_id})
            return await self.execute(
                confirmed,
                session_id=session_id,
                client_type=original_client_type,
                locally_confirmed=True,
            )
        finally:
            if approved:
                self._pending_confirmation_actions.pop(request_id, None)

    async def execute(
        self,
        action: ComputerAction,
        *,
        session_id: str,
        client_type: str,
        locally_confirmed: bool = False,
        transactional_authorized: bool = False,
    ) -> ComputerActionResult:
        self._ensure_enabled()
        spec = self.policy.validate_action(action)
        self.policy.validate_client(action, client_type=client_type, locally_confirmed=locally_confirmed)
        started_at = datetime.now(timezone.utc)
        permission_decision = self.permissions.decision_for(action.permission_scope)
        confirmation_result = "not_required"
        final_state = ExecutionState.FAILED
        error_code: str | None = None
        sensitive = False
        try:
            capability_report = await self.provider.discover_capabilities()
            if not capability_report.supports(action.capability, action.operation):
                raise CapabilityUnavailable(
                    f"{action.operation!r} is not reported as implemented by {capability_report.provider}.",
                    corrective_action="Run computer_capabilities and select one of its reported operations.",
                )
            self._emit(action, "permission_check", ExecutionState.WAITING_PERMISSION, f"Checking {action.permission_scope} permission", permission_scope=action.permission_scope)
            permission_status = self.permissions.status(action.permission_scope)
            if permission_status.decision is PermissionDecision.ASK_EVERY_TIME and not transactional_authorized:
                raise self._request_permission(
                    action,
                    session_id=session_id,
                    client_type=client_type,
                )
            mana_permission = (
                PermissionDecision.ALLOW_ONCE
                if transactional_authorized else self.permissions.require(action.permission_scope)
            )
            os_permission = await self.provider.check_permission(action.capability)
            if not os_permission.granted:
                raise OperatingSystemPermissionDenied(
                    os_permission.reason or f"Operating-system permission for {action.capability!r} is unavailable.",
                    corrective_action="Enable the required permission in operating-system privacy settings.",
                )
            self._emit(action, "permission_check", ExecutionState.COMPLETED, "Computer permission granted", permission_scope=action.permission_scope)
            try:
                if not transactional_authorized:
                    self.policy.require_confirmation(action)
            except ConfirmationRequired as exc:
                confirmation_result = "requested"
                self._pending_confirmation_actions[exc.confirmation_request_id] = (
                    action,
                    session_id,
                    client_type,
                )
                self._emit(action, "waiting_confirmation", ExecutionState.WAITING_CONFIRMATION, self.policy.preview(action), risk=action.risk.value)
                raise
            if action.risk.value in {"high", "critical"}:
                confirmation_result = "confirmed"
            if not transactional_authorized:
                self.permissions.consume_once(action.permission_scope, mana_permission)
            self._emit(action, "action_started", ExecutionState.RUNNING, self.policy.preview(action), capability=action.capability, operation=action.operation, application_id=action.target.application_id or "")
            adapter = await self.adapters.select(
                action,
                platform=self.provider.platform,
                preference=action.target.application_id,
                configured_default=self.settings.defaults.get(action.capability),
            )
            executor = self.provider.execute
            if adapter is not None:
                executor = adapter.execute
                self._emit(
                    action,
                    "adapter_selected",
                    ExecutionState.RUNNING,
                    f"Selected {adapter.application_id} application adapter",
                    application_id=adapter.application_id,
                )
            task = asyncio.create_task(executor(action), name=action.execution_id)
            self._tasks[action.execution_id] = task
            self._execution_sessions[action.execution_id] = session_id
            try:
                result = await asyncio.wait_for(task, timeout=action.timeout_seconds)
            except asyncio.TimeoutError as exc:
                await self.provider.cancel(action.execution_id)
                raise ActionTimedOut(
                    f"Computer action exceeded its {action.timeout_seconds:g}-second timeout."
                ) from exc
            sensitive = spec.sensitive_result or result.sensitive_content_accessed
            result.sensitive_content_accessed = sensitive
            final_state = result.state
            self._emit(action, "action_completed", result.state, result.message, capability=action.capability, operation=action.operation)
            return result
        except ActionCancelled:
            final_state = ExecutionState.CANCELLED
            error_code = "action_cancelled"
            self._emit(action, "action_cancelled", final_state, "Computer action cancelled")
            raise
        except asyncio.CancelledError as exc:
            final_state = ExecutionState.CANCELLED
            error_code = "action_cancelled"
            self._emit(action, "action_cancelled", final_state, "Computer action cancelled")
            raise ActionCancelled("Computer action was cancelled; no later action was executed.") from exc
        except ActionTimedOut:
            final_state = ExecutionState.TIMED_OUT
            error_code = "action_timed_out"
            self._emit(action, "action_failed", final_state, "Computer action timed out")
            raise
        except ComputerControlError as exc:
            error_code = exc.code
            if exc.code in {"mana_permission_denied", "os_permission_denied", "remote_control_denied"}:
                final_state = ExecutionState.DENIED
                event_type = "action_denied"
            elif exc.code == "permission_required":
                final_state = ExecutionState.WAITING_PERMISSION
                event_type = "waiting_permission"
            elif exc.code in {"confirmation_required", "invalid_confirmation"}:
                final_state = ExecutionState.WAITING_CONFIRMATION
                event_type = "waiting_confirmation"
            else:
                event_type = "action_failed"
            if not isinstance(exc, (ConfirmationRequired, PermissionApprovalRequired)):
                self._emit(action, event_type, final_state, str(exc))
            raise
        finally:
            self._tasks.pop(action.execution_id, None)
            self._execution_sessions.pop(action.execution_id, None)
            if action.confirmation_token:
                self._pending_confirmation_actions.pop(action.confirmation_token, None)
            self.audit.record(AuditRecord(
                execution_id=action.execution_id,
                session_id=session_id,
                client_type=client_type,
                capability=action.capability,
                operation=action.operation,
                target_application=action.target.application_id,
                risk=action.risk,
                permission_decision=permission_decision,
                confirmation_result=confirmation_result,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                final_state=final_state,
                sensitive_content_accessed=sensitive,
                error_code=error_code,
            ))

    async def cancel(self, execution_id: str) -> bool:
        task = self._tasks.get(execution_id)
        provider_cancelled = await self.provider.cancel(execution_id)
        if task is not None and not task.done():
            task.cancel()
            return True
        return provider_cancelled

    async def cancel_session(self, session_id: str) -> bool:
        execution_ids = [
            execution_id
            for execution_id, owner_session in self._execution_sessions.items()
            if owner_session == session_id
        ]
        results = [await self.cancel(execution_id) for execution_id in execution_ids]
        return any(results)


_default_service: ComputerControlService | None = None


def default_computer_control_service() -> ComputerControlService:
    global _default_service
    if _default_service is None:
        _default_service = ComputerControlService()
    return _default_service


def reset_default_computer_control_service() -> None:
    global _default_service
    _default_service = None
