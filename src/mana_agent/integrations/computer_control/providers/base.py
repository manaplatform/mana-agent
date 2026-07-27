"""Shared provider mechanics; platform decisions remain in platform modules."""

from __future__ import annotations

import asyncio
import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from mana_agent.integrations.computer_control.config import ComputerControlSettings
from mana_agent.integrations.computer_control.errors import (
    ActionCancelled,
    ApplicationNotResponding,
    CapabilityUnavailable,
    InvalidActionDecision,
)
from mana_agent.integrations.computer_control.models import (
    CapabilityReport,
    ComputerAction,
    ComputerActionResult,
    ExecutionState,
    PermissionDecision,
    PermissionResult,
    PermissionStatus,
    SupportedPlatform,
)


class BaseProvider(ABC):
    platform: SupportedPlatform
    provider_id: str

    def __init__(self, settings: ComputerControlSettings) -> None:
        self.settings = settings
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._cancelled: set[str] = set()
        self._report: CapabilityReport | None = None

    @abstractmethod
    async def discover_capabilities(self) -> CapabilityReport: ...

    async def check_permission(self, capability: str) -> PermissionStatus:
        report = self._report or await self.discover_capabilities()
        available = report.supports(capability)
        return PermissionStatus(
            scope=capability,
            decision=PermissionDecision.MANAGED_BY_OS if available else PermissionDecision.UNAVAILABLE,
            granted=available,
            os_managed=True,
            reason="" if available else f"{capability} is unavailable on this desktop session.",
        )

    async def request_permission(self, capability: str) -> PermissionResult:
        current = await self.check_permission(capability)
        return PermissionResult(**current.model_dump(), requested=True)

    async def cancel(self, execution_id: str) -> bool:
        self._cancelled.add(execution_id)
        process = self._processes.get(execution_id)
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        return process is not None

    def _ensure_not_cancelled(self, execution_id: str) -> None:
        if execution_id in self._cancelled:
            raise ActionCancelled("Computer action was cancelled; no later action was executed.")

    async def _run(
        self,
        action: ComputerAction,
        argv: Sequence[str],
        *,
        stdin: bytes | None = None,
    ) -> tuple[str, str]:
        """Run only provider-constructed argv, never model-provided command text."""
        self._ensure_not_cancelled(action.execution_id)
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._processes[action.execution_id] = process
        try:
            stdout, stderr = await process.communicate(input=stdin)
        finally:
            self._processes.pop(action.execution_id, None)
        self._ensure_not_cancelled(action.execution_id)
        if process.returncode:
            diagnostic = stderr.decode(errors="replace").strip()[-500:]
            raise ApplicationNotResponding(
                f"Desktop integration returned exit code {process.returncode}.",
                corrective_action=diagnostic,
            )
        return stdout.decode(errors="replace"), stderr.decode(errors="replace")

    def _allowed_path(self, raw_path: str | None, *, must_exist: bool = True) -> Path:
        if not raw_path:
            raise InvalidActionDecision("This operation requires a target path.")
        path = Path(raw_path).expanduser().resolve()
        roots = [root.expanduser().resolve() for root in self.settings.allowed_paths]
        if not roots:
            raise InvalidActionDecision("No computer-control filesystem paths are configured.")
        if not any(path == root or root in path.parents for root in roots):
            raise InvalidActionDecision(f"Path {path} is outside configured computer-control roots.")
        if must_exist and not path.exists():
            raise InvalidActionDecision(f"Path {path} does not exist.")
        return path

    async def _filesystem(self, action: ComputerAction) -> ComputerActionResult:
        source = self._allowed_path(action.target.path, must_exist=action.operation != "filesystem.mkdir")
        data: dict[str, object] = {}
        if action.operation == "filesystem.metadata":
            stat = source.stat()
            data = {
                "name": source.name, "size": stat.st_size, "is_directory": source.is_dir(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        elif action.operation == "filesystem.copy":
            destination = self._allowed_path(str(action.arguments.get("destination") or ""), must_exist=False)
            if destination.exists():
                raise InvalidActionDecision("Copy destination already exists.")
            shutil.copytree(source, destination) if source.is_dir() else shutil.copy2(source, destination)
            data = {"path": str(destination)}
        elif action.operation in {"filesystem.move", "filesystem.rename"}:
            destination = self._allowed_path(str(action.arguments.get("destination") or ""), must_exist=False)
            if destination.exists():
                raise InvalidActionDecision("Move destination already exists.")
            shutil.move(str(source), str(destination))
            data = {"path": str(destination)}
        elif action.operation == "filesystem.mkdir":
            source.mkdir(parents=False, exist_ok=False)
            data = {"path": str(source)}
        elif action.operation in {"filesystem.open", "filesystem.reveal", "filesystem.trash"}:
            return await self.execute_platform_action(action)
        else:
            raise CapabilityUnavailable(f"{action.operation!r} is not implemented by the shared provider.")
        return self._result(action, data=data)

    @staticmethod
    def _result(
        action: ComputerAction,
        *,
        message: str = "Computer action completed.",
        data: dict[str, object] | None = None,
        sensitive: bool = False,
    ) -> ComputerActionResult:
        return ComputerActionResult(
            execution_id=action.execution_id,
            state=ExecutionState.COMPLETED,
            capability=action.capability,
            operation=action.operation,
            message=message,
            data=dict(data or {}),
            finished_at=datetime.now(timezone.utc),
            sensitive_content_accessed=sensitive,
        )

    async def execute(self, action: ComputerAction) -> ComputerActionResult:
        self._ensure_not_cancelled(action.execution_id)
        report = self._report or await self.discover_capabilities()
        if not report.supports(action.capability):
            raise CapabilityUnavailable(
                f"{action.capability!r} is unavailable from {self.provider_id}.",
                corrective_action="Run computer_capabilities and choose a reported capability.",
            )
        if action.operation.startswith("filesystem."):
            return await self._filesystem(action)
        return await self.execute_platform_action(action)

    @abstractmethod
    async def execute_platform_action(self, action: ComputerAction) -> ComputerActionResult: ...


def command_available(command: str) -> bool:
    return shutil.which(command) is not None


def desktop_session_available(platform: SupportedPlatform) -> bool:
    if platform is SupportedPlatform.WINDOWS:
        return bool(os.environ.get("SESSIONNAME"))
    if platform is SupportedPlatform.MACOS:
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
