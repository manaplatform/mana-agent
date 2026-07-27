"""Linux provider using freedesktop, MPRIS, and desktop-native utilities."""

from __future__ import annotations

import os
import re
from pathlib import Path

from mana_agent.integrations.computer_control.errors import ApplicationNotInstalled, CapabilityUnavailable, HeadlessEnvironment, InvalidActionDecision
from mana_agent.integrations.computer_control.models import (
    ApplicationDescriptor,
    CapabilityAvailability,
    CapabilityReport,
    ComputerAction,
    SupportedPlatform,
)
from mana_agent.integrations.computer_control.providers.base import BaseProvider, command_available, desktop_session_available

_DESKTOP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,254}$")


class LinuxProvider(BaseProvider):
    platform = SupportedPlatform.LINUX
    provider_id = "linux-freedesktop"

    async def discover_capabilities(self) -> CapabilityReport:
        headless = not desktop_session_available(self.platform)
        applications: list[ApplicationDescriptor] = []
        for root in (Path.home() / ".local/share/applications", Path("/usr/share/applications")):
            if root.is_dir():
                for item in sorted(root.glob("*.desktop"))[:300]:
                    applications.append(ApplicationDescriptor(
                        application_id=item.stem, name=item.stem.replace("-", " ").title(),
                        executable=None, capabilities={"applications"},
                    ))
        opener = command_available("gio") or command_available("xdg-open")
        clipboard = command_available("wl-copy") or command_available("xclip")
        capabilities = [
            CapabilityAvailability(name="applications", available=not headless and command_available("gtk-launch"), provider=self.provider_id, reason="No graphical desktop session or gtk-launch." if headless else "", permission_scopes={"computer.apps.read", "computer.apps.control"}, operations={"applications.list", "applications.open"}),
            CapabilityAvailability(name="browser", available=not headless and opener, provider=self.provider_id, reason="No graphical desktop opener." if headless else "", permission_scopes={"computer.browser.control"}, operations={"browser.open_url"}),
            CapabilityAvailability(name="media", available=not headless and command_available("playerctl"), provider=self.provider_id, reason="playerctl/MPRIS is unavailable.", permission_scopes={"computer.media.read", "computer.media.control"}, operations={"media.play", "media.resume", "media.pause", "media.stop", "media.next", "media.previous"}),
            CapabilityAvailability(name="clipboard", available=not headless and clipboard, provider=self.provider_id, reason="No Wayland/X11 clipboard utility.", permission_scopes={"computer.clipboard.read", "computer.clipboard.write"}, operations={"clipboard.read", "clipboard.write", "clipboard.clear"}),
            CapabilityAvailability(name="filesystem", available=True, provider=self.provider_id, permission_scopes={"computer.files.read", "computer.files.write"}, operations={"filesystem.open", "filesystem.reveal", "filesystem.metadata", "filesystem.copy", "filesystem.move", "filesystem.rename", "filesystem.mkdir", "filesystem.trash"}),
            CapabilityAvailability(name="notifications", available=not headless and command_available("notify-send"), provider=self.provider_id, permission_scopes={"computer.notifications.send"}, operations={"notifications.send"}),
            CapabilityAvailability(name="system", available=not headless and command_available("pactl"), provider=self.provider_id, permission_scopes={"computer.system.read", "computer.system.control"}, operations={"system.volume"}),
            CapabilityAvailability(name="calendar", available=False, provider=self.provider_id, reason="No calendar D-Bus adapter is configured."),
            CapabilityAvailability(name="notes", available=False, provider=self.provider_id, reason="No notes D-Bus adapter is configured."),
            CapabilityAvailability(name="screenshots", available=False, provider=self.provider_id, reason="No trusted desktop-portal screenshot adapter is configured."),
        ]
        self._report = CapabilityReport(platform=self.platform, provider=self.provider_id, capabilities=capabilities, applications=applications, headless=headless)
        return self._report

    def _require_desktop(self) -> None:
        if not desktop_session_available(self.platform):
            raise HeadlessEnvironment("Computer control requires a graphical Linux desktop session.")

    async def execute_platform_action(self, action: ComputerAction):
        self._require_desktop()
        op = action.operation
        if op == "applications.list":
            report = self._report or await self.discover_capabilities()
            return self._result(action, data={"applications": [item.model_dump() for item in report.applications]})
        if op == "applications.open":
            app = action.target.application_id or ""
            if not _DESKTOP_ID.fullmatch(app):
                raise InvalidActionDecision("Invalid freedesktop application identifier.")
            report = self._report or await self.discover_capabilities()
            if app not in {item.application_id for item in report.applications}:
                raise ApplicationNotInstalled(f"Application {app!r} was not found in freedesktop application directories.")
            await self._run(action, ["gtk-launch", app])
            return self._result(action, message=f"Opened {app}.")
        if op == "browser.open_url":
            argv = ["gio", "open", action.target.url or ""] if command_available("gio") else ["xdg-open", action.target.url or ""]
            await self._run(action, argv)
            return self._result(action, message="Opened URL in the default browser.")
        if op in {"filesystem.open", "filesystem.reveal"}:
            path = self._allowed_path(action.target.path)
            target = path.parent if op == "filesystem.reveal" else path
            argv = ["gio", "open", str(target)] if command_available("gio") else ["xdg-open", str(target)]
            await self._run(action, argv)
            return self._result(action, message="Opened filesystem target.")
        if op == "filesystem.trash":
            path = self._allowed_path(action.target.path)
            if not command_available("gio"):
                raise CapabilityUnavailable("Safe Trash requires the freedesktop gio utility; permanent deletion was not attempted.")
            await self._run(action, ["gio", "trash", str(path)])
            return self._result(action, message="Moved target to Trash.")
        if op.startswith("media."):
            command = {"media.play": "play", "media.resume": "play", "media.pause": "pause", "media.stop": "stop", "media.next": "next", "media.previous": "previous"}.get(op)
            if command is None:
                raise CapabilityUnavailable(f"Linux media operation {op!r} is unavailable.")
            application_id = action.target.application_id
            if application_id and not _DESKTOP_ID.fullmatch(application_id):
                raise InvalidActionDecision("Invalid MPRIS player identifier.")
            argv = ["playerctl"]
            if application_id:
                argv.extend(["--player", application_id])
            argv.append(command)
            await self._run(action, argv)
            return self._result(action, message=f"Media action {command} completed.")
        if op == "clipboard.read":
            argv = ["wl-paste", "--no-newline"] if os.environ.get("WAYLAND_DISPLAY") and command_available("wl-paste") else ["xclip", "-selection", "clipboard", "-o"]
            stdout, _ = await self._run(action, argv)
            return self._result(action, data={"type": "text", "content": stdout}, sensitive=True)
        if op in {"clipboard.write", "clipboard.clear"}:
            text = str(action.arguments.get("text") or "")
            argv = ["wl-copy"] if os.environ.get("WAYLAND_DISPLAY") and command_available("wl-copy") else ["xclip", "-selection", "clipboard"]
            await self._run(action, argv, stdin=text.encode())
            return self._result(action, message="Clipboard updated.")
        if op == "notifications.send":
            await self._run(action, ["notify-send", str(action.arguments.get("title") or "")[:200], str(action.arguments.get("body") or "")[:1000]])
            return self._result(action, message="Notification sent.")
        if op == "system.volume":
            volume = action.arguments.get("volume")
            if not isinstance(volume, (int, float)) or not 0 <= float(volume) <= 1:
                raise InvalidActionDecision("Volume must be between 0 and 1.")
            await self._run(action, ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{round(float(volume) * 100)}%"])
            return self._result(action, message="System volume updated.")
        raise CapabilityUnavailable(f"Linux operation {op!r} is unavailable.")
