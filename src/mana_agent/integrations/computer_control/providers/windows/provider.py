"""Windows provider with fixed PowerShell and Shell command templates."""

from __future__ import annotations

import re

from mana_agent.integrations.computer_control.errors import CapabilityUnavailable, InvalidActionDecision
from mana_agent.integrations.computer_control.models import (
    CapabilityAvailability,
    CapabilityReport,
    ComputerAction,
    SupportedPlatform,
)
from mana_agent.integrations.computer_control.providers.base import BaseProvider, command_available, desktop_session_available

_APP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.!+-]{0,300}$")


class WindowsProvider(BaseProvider):
    platform = SupportedPlatform.WINDOWS
    provider_id = "windows-native"

    async def discover_capabilities(self) -> CapabilityReport:
        headless = not desktop_session_available(self.platform)
        shell = command_available("powershell") or command_available("pwsh")
        capabilities = [
            CapabilityAvailability(name="applications", available=not headless, provider=self.provider_id, permission_scopes={"computer.apps.read", "computer.apps.control"}, operations={"applications.list", "applications.open"}),
            CapabilityAvailability(name="browser", available=not headless, provider=self.provider_id, permission_scopes={"computer.browser.control"}, operations={"browser.open_url"}),
            CapabilityAvailability(name="clipboard", available=not headless and shell, provider=self.provider_id, permission_scopes={"computer.clipboard.read", "computer.clipboard.write"}, operations={"clipboard.read", "clipboard.write", "clipboard.clear"}),
            CapabilityAvailability(name="filesystem", available=True, provider=self.provider_id, permission_scopes={"computer.files.read", "computer.files.write"}, operations={"filesystem.open", "filesystem.reveal", "filesystem.metadata", "filesystem.copy", "filesystem.move", "filesystem.rename", "filesystem.mkdir", "filesystem.trash"}),
            CapabilityAvailability(name="calendar", available=False, provider=self.provider_id, reason="Outlook/WinRT calendar adapter is not configured."),
            CapabilityAvailability(name="notes", available=False, provider=self.provider_id, reason="OneNote adapter is not configured."),
            CapabilityAvailability(name="media", available=False, provider=self.provider_id, reason="Windows media-session adapter is not configured."),
            CapabilityAvailability(name="system", available=False, provider=self.provider_id, reason="Windows system-control adapter is not configured."),
            CapabilityAvailability(name="notifications", available=False, provider=self.provider_id, reason="Windows notification adapter is not configured."),
            CapabilityAvailability(name="screenshots", available=False, provider=self.provider_id, reason="Windows Graphics Capture adapter is not configured."),
        ]
        self._report = CapabilityReport(platform=self.platform, provider=self.provider_id, capabilities=capabilities, headless=headless)
        return self._report

    async def execute_platform_action(self, action: ComputerAction):
        op = action.operation
        if op == "applications.list":
            return self._result(action, data={"applications": []}, message="Application discovery requires the Windows app registry adapter.")
        if op == "applications.open":
            app_id = action.target.application_id or ""
            if not _APP_ID.fullmatch(app_id):
                raise InvalidActionDecision("Invalid Windows application identifier.")
            await self._run(action, ["explorer.exe", f"shell:AppsFolder\\{app_id}"])
            return self._result(action, message=f"Opened {app_id}.")
        if op == "browser.open_url":
            await self._run(action, ["rundll32.exe", "url.dll,FileProtocolHandler", action.target.url or ""])
            return self._result(action, message="Opened URL in the default browser.")
        if op in {"filesystem.open", "filesystem.reveal"}:
            path = self._allowed_path(action.target.path)
            argv = ["explorer.exe", f"/select,{path}"] if op == "filesystem.reveal" else ["explorer.exe", str(path)]
            await self._run(action, argv)
            return self._result(action, message="Opened filesystem target.")
        if op == "filesystem.trash":
            path = self._allowed_path(action.target.path)
            shell = "powershell" if command_available("powershell") else "pwsh"
            if path.is_dir():
                method = "[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($args[0],'OnlyErrorDialogs','SendToRecycleBin')"
            else:
                method = "[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($args[0],'OnlyErrorDialogs','SendToRecycleBin')"
            script = f"Add-Type -AssemblyName Microsoft.VisualBasic; {method}"
            await self._run(action, [shell, "-NoProfile", "-NonInteractive", "-Command", script, str(path)])
            return self._result(action, message="Moved target to Recycle Bin.")
        if op == "clipboard.read":
            shell = "powershell" if command_available("powershell") else "pwsh"
            stdout, _ = await self._run(action, [shell, "-NoProfile", "-NonInteractive", "-Command", "Get-Clipboard -Raw"])
            return self._result(action, data={"type": "text", "content": stdout}, sensitive=True)
        if op in {"clipboard.write", "clipboard.clear"}:
            shell = "powershell" if command_available("powershell") else "pwsh"
            # Clipboard data travels over stdin and never enters PowerShell source.
            await self._run(action, [shell, "-NoProfile", "-NonInteractive", "-Command", "$input | Set-Clipboard"], stdin=str(action.arguments.get("text") or "").encode())
            return self._result(action, message="Clipboard updated.")
        raise CapabilityUnavailable(f"Windows operation {op!r} is unavailable.")
