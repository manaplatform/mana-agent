"""Deterministic provider for unit/integration tests; never touches the desktop."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from mana_agent.integrations.computer_control.models import (
    ApplicationDescriptor,
    CapabilityAvailability,
    CapabilityReport,
    ComputerAction,
    ComputerActionResult,
    ExecutionState,
    PermissionDecision,
    PermissionResult,
    PermissionStatus,
    SupportedPlatform,
)
from mana_agent.integrations.computer_control.policy import ACTION_SPECS


class FakeComputerControlProvider:
    provider_id = "fake"

    def __init__(
        self,
        *,
        platform: SupportedPlatform = SupportedPlatform.LINUX,
        capabilities: set[str] | None = None,
        delay_seconds: float = 0,
        os_permissions: dict[str, bool] | None = None,
        operations: set[str] | None = None,
    ) -> None:
        self.platform = platform
        self.capabilities = capabilities or {
            "applications", "calendar", "media", "notes", "browser", "clipboard",
            "filesystem", "screenshots", "notifications", "system",
        }
        self.delay_seconds = delay_seconds
        self.os_permissions = dict(os_permissions or {})
        self.operations = operations
        self.executed: list[ComputerAction] = []
        self.cancelled: set[str] = set()

    async def discover_capabilities(self) -> CapabilityReport:
        return CapabilityReport(
            platform=self.platform,
            provider=self.provider_id,
            capabilities=[
                CapabilityAvailability(
                    name=name,
                    available=True,
                    provider=self.provider_id,
                    operations={
                        operation
                        for operation in ACTION_SPECS
                        if operation.split(".", 1)[0] == name
                        and (self.operations is None or operation in self.operations)
                    },
                )
                for name in sorted(self.capabilities)
            ],
            applications=[
                ApplicationDescriptor(
                    application_id="fake.editor",
                    name="Fake Editor",
                    capabilities={"applications"},
                )
            ],
        )

    async def check_permission(self, capability: str) -> PermissionStatus:
        available = capability in self.capabilities and self.os_permissions.get(capability, True)
        return PermissionStatus(
            scope=capability,
            decision=PermissionDecision.MANAGED_BY_OS if available else PermissionDecision.UNAVAILABLE,
            granted=available,
            os_managed=True,
            reason="" if available else f"Fake OS denied {capability}.",
        )

    async def request_permission(self, capability: str) -> PermissionResult:
        status = await self.check_permission(capability)
        return PermissionResult(**status.model_dump(), requested=True)

    async def execute(self, action: ComputerAction) -> ComputerActionResult:
        self.executed.append(action)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if action.execution_id in self.cancelled:
            return ComputerActionResult(
                execution_id=action.execution_id,
                state=ExecutionState.CANCELLED,
                capability=action.capability,
                operation=action.operation,
                message="Fake action cancelled.",
                finished_at=datetime.now(timezone.utc),
            )
        data = {
            "applications.list": {"applications": [{"application_id": "fake.editor", "name": "Fake Editor"}]},
            "calendar.list": {"events": []},
            "calendar.create": {"event": action.arguments.get("event")},
            "calendar.update": {"event": action.arguments.get("event")},
            "calendar.delete": {"deleted": action.target.resource_id},
            "media.status": {"state": "paused"},
            "notes.search": {"notes": [{"note_id": "note-1", "title": "Roadmap"}]},
            "notes.read": {"note": {"note_id": action.target.resource_id, "title": "Roadmap", "content": "private fake content"}},
            "browser.active_page": {"title": "Example", "url": "https://example.test"},
            "browser.read_page": {"title": "Example", "url": "https://example.test", "text": "private fake page"},
            "clipboard.read": {"type": "text", "content": "private fake clipboard"},
            "system.status": {"volume": 0.5, "muted": False},
        }.get(action.operation, {"performed": action.operation})
        sensitive = action.operation in {
            "calendar.list", "notes.search", "notes.read", "browser.active_page",
            "browser.read_page", "browser.list_tabs", "clipboard.read", "screenshots.capture",
        }
        return ComputerActionResult(
            execution_id=action.execution_id,
            state=ExecutionState.COMPLETED,
            capability=action.capability,
            operation=action.operation,
            message="Fake action completed.",
            data=data,
            finished_at=datetime.now(timezone.utc),
            sensitive_content_accessed=sensitive,
        )

    async def cancel(self, execution_id: str) -> bool:
        self.cancelled.add(execution_id)
        return True
