"""Provider and application-adapter protocols."""

from __future__ import annotations

from typing import Protocol

from mana_agent.integrations.computer_control.models import (
    ApplicationStatus,
    CapabilityReport,
    ComputerAction,
    ComputerActionResult,
    PermissionResult,
    PermissionStatus,
    SupportedPlatform,
)


class ComputerControlProvider(Protocol):
    platform: SupportedPlatform
    provider_id: str

    async def discover_capabilities(self) -> CapabilityReport: ...
    async def check_permission(self, capability: str) -> PermissionStatus: ...
    async def request_permission(self, capability: str) -> PermissionResult: ...
    async def execute(self, action: ComputerAction) -> ComputerActionResult: ...
    async def cancel(self, execution_id: str) -> bool: ...


class ApplicationAdapter(Protocol):
    application_id: str
    supported_platforms: set[SupportedPlatform]
    capabilities: set[str]

    async def is_available(self) -> bool: ...
    async def get_status(self) -> ApplicationStatus: ...
    async def execute(self, action: ComputerAction) -> ComputerActionResult: ...

