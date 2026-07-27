"""macOS provider using fixed native command templates."""

from __future__ import annotations

import re
import plistlib
from pathlib import Path
from mana_agent.config.settings import mana_home

from mana_agent.integrations.computer_control.errors import (
    ApplicationNotInstalled,
    ApplicationNotResponding,
    CapabilityUnavailable,
    InvalidActionDecision,
)
from mana_agent.integrations.computer_control.models import (
    ApplicationDescriptor,
    CapabilityAvailability,
    CapabilityReport,
    ComputerAction,
    SupportedPlatform,
)
from mana_agent.integrations.computer_control.providers.base import BaseProvider, command_available

_BUNDLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{1,254}$")


class MacOSProvider(BaseProvider):
    platform = SupportedPlatform.MACOS
    provider_id = "macos-native"

    async def discover_capabilities(self) -> CapabilityReport:
        applications: list[ApplicationDescriptor] = []
        roots = (Path("/Applications"), Path("/System/Applications"), Path.home() / "Applications")
        for root in roots:
            if root.is_dir():
                for item in sorted(root.glob("*.app"))[:200]:
                    try:
                        with (item / "Contents" / "Info.plist").open("rb") as stream:
                            metadata = plistlib.load(stream)
                    except (OSError, plistlib.InvalidFileException):
                        continue
                    bundle_id = str(metadata.get("CFBundleIdentifier") or "")
                    if not _BUNDLE_ID.fullmatch(bundle_id):
                        continue
                    applications.append(ApplicationDescriptor(
                        application_id=bundle_id,
                        name=str(metadata.get("CFBundleDisplayName") or metadata.get("CFBundleName") or item.stem),
                        version=str(metadata.get("CFBundleShortVersionString") or "") or None,
                        executable=str(item),
                        capabilities={"applications"},
                    ))
        capabilities = [
            CapabilityAvailability(name="applications", available=command_available("open"), provider=self.provider_id, permission_scopes={"computer.apps.read", "computer.apps.control"}, operations={"applications.list", "applications.open"}),
            CapabilityAvailability(name="browser", available=command_available("open"), provider=self.provider_id, permission_scopes={"computer.browser.control"}, operations={"browser.open_url"}),
            CapabilityAvailability(name="clipboard", available=command_available("pbcopy") and command_available("pbpaste"), provider=self.provider_id, permission_scopes={"computer.clipboard.read", "computer.clipboard.write"}, operations={"clipboard.read", "clipboard.write", "clipboard.clear"}),
            CapabilityAvailability(name="media", available=command_available("osascript"), provider=self.provider_id, permission_scopes={"computer.media.read", "computer.media.control"}, operations={"media.play", "media.resume", "media.pause", "media.next", "media.previous", "media.stop"}),
            CapabilityAvailability(name="system", available=command_available("osascript"), provider=self.provider_id, permission_scopes={"computer.system.read", "computer.system.control"}, operations={"system.volume"}),
            CapabilityAvailability(name="notifications", available=command_available("osascript"), provider=self.provider_id, permission_scopes={"computer.notifications.send"}, operations={"notifications.send"}),
            CapabilityAvailability(name="filesystem", available=True, provider=self.provider_id, permission_scopes={"computer.files.read", "computer.files.write"}, operations={"filesystem.open", "filesystem.reveal", "filesystem.metadata", "filesystem.copy", "filesystem.move", "filesystem.rename", "filesystem.mkdir", "filesystem.trash"}),
            CapabilityAvailability(name="calendar", available=False, provider=self.provider_id, reason="EventKit bridge is not installed."),
            CapabilityAvailability(name="notes", available=False, provider=self.provider_id, reason="Notes automation adapter is not installed."),
            CapabilityAvailability(name="screenshots", available=command_available("screencapture"), provider=self.provider_id, permission_scopes={"computer.screenshot.capture"}, operations={"screenshots.capture"}),
        ]
        self._report = CapabilityReport(platform=self.platform, provider=self.provider_id, capabilities=capabilities, applications=applications)
        return self._report

    async def execute_platform_action(self, action: ComputerAction):
        op = action.operation
        if op == "applications.list":
            report = self._report or await self.discover_capabilities()
            return self._result(action, data={"applications": [item.model_dump() for item in report.applications]})
        if op == "applications.open":
            app = action.target.application_id or ""
            if not _BUNDLE_ID.fullmatch(app):
                raise InvalidActionDecision("Invalid macOS application identifier.")
            report = self._report or await self.discover_capabilities()
            if app not in {item.application_id for item in report.applications}:
                raise ApplicationNotInstalled(f"Application {app!r} was not found during macOS discovery.")
            await self._run(action, ["open", "-b", app])
            return self._result(action, message=f"Opened {app}.")
        if op == "browser.open_url":
            await self._run(action, ["open", action.target.url or ""])
            return self._result(action, message="Opened URL in the default browser.")
        if op in {"filesystem.open", "filesystem.reveal"}:
            path = self._allowed_path(action.target.path)
            argv = ["open", "-R", str(path)] if op == "filesystem.reveal" else ["open", str(path)]
            await self._run(action, argv)
            return self._result(action, message="Opened filesystem target.")
        if op == "filesystem.trash":
            path = self._allowed_path(action.target.path)
            script = 'on run argv\n tell application "Finder" to delete POSIX file (item 1 of argv)\nend run'
            await self._run(action, ["osascript", "-e", script, str(path)])
            return self._result(action, message="Moved target to Trash.")
        if op == "clipboard.read":
            stdout, _ = await self._run(action, ["pbpaste"])
            return self._result(action, data={"type": "text", "content": stdout}, sensitive=True)
        if op in {"clipboard.write", "clipboard.clear"}:
            text = str(action.arguments.get("text") or "")
            await self._run(action, ["pbcopy"], stdin=text.encode())
            return self._result(action, message="Clipboard updated.")
        if op in {"media.play", "media.resume", "media.pause", "media.next", "media.previous", "media.stop"}:
            application_id = action.target.application_id or "com.apple.Music"
            if application_id != "com.apple.Music":
                raise CapabilityUnavailable(
                    f"The macOS media adapter for {application_id!r} is not implemented; Apple Music was not controlled instead."
                )
            if op == "media.play":
                query = str(action.arguments.get("query") or "").strip()
                script = (
                    "on run argv\n"
                    ' tell application "Music"\n'
                    "  activate\n"
                    "  set requestedQuery to item 1 of argv\n"
                    "  if requestedQuery is \"\" then\n"
                    "   set libraryTrackCount to count of tracks of library playlist 1\n"
                    "   if libraryTrackCount is 0 then error \"The Music library has no playable tracks.\"\n"
                    "   set selectedIndex to random number from 1 to libraryTrackCount\n"
                    "   play track selectedIndex of library playlist 1\n"
                    "  else\n"
                    "   set matchingTracks to search library playlist 1 for requestedQuery\n"
                    "   if (count of matchingTracks) is 0 then error \"No matching track was found in the Music library.\"\n"
                    "   play item 1 of matchingTracks\n"
                    "  end if\n"
                    "  delay 1\n"
                    "  return player state as text\n"
                    " end tell\n"
                    "end run"
                )
                stdout, _ = await self._run(action, ["osascript", "-e", script, query])
                state = stdout.strip().lower()
                if state != "playing":
                    raise ApplicationNotResponding(
                        f"Music accepted the play command but reported {state or 'no playback state'}.",
                        corrective_action="Add a playable track to the Music library or choose a specific library song/playlist.",
                    )
                return self._result(
                    action,
                    message="Music playback started and was verified.",
                    data={"playback_state": state},
                )
            command = {
                "media.resume": "play",
                "media.pause": "pause",
                "media.next": "next track",
                "media.previous": "previous track",
                "media.stop": "stop",
            }[op]
            script = (
                f'tell application "Music"\n {command}\n delay 0.3\n'
                " return player state as text\nend tell"
            )
            stdout, _ = await self._run(action, ["osascript", "-e", script])
            state = stdout.strip().lower()
            expected = {
                "media.resume": "playing",
                "media.pause": "paused",
                "media.stop": "stopped",
            }.get(op)
            if expected is not None and state != expected:
                raise ApplicationNotResponding(
                    f"Music accepted {command!r} but reported {state or 'no playback state'}."
                )
            return self._result(
                action,
                message=f"Music {command} completed; playback state is {state or 'unknown'}.",
                data={"playback_state": state or "unknown"},
            )
        if op == "system.volume":
            volume = action.arguments.get("volume")
            if not isinstance(volume, (int, float)) or not 0 <= float(volume) <= 1:
                raise InvalidActionDecision("Volume must be between 0 and 1.")
            await self._run(action, ["osascript", "-e", f"set volume output volume {round(float(volume) * 100)}"])
            return self._result(action, message="System volume updated.")
        if op == "notifications.send":
            title = str(action.arguments.get("title") or "")[:200]
            body = str(action.arguments.get("body") or "")[:1000]
            # Values are argv parameters to the fixed script, not script source.
            script = 'on run argv\n display notification (item 2 of argv) with title (item 1 of argv)\nend run'
            await self._run(action, ["osascript", "-e", script, title, body])
            return self._result(action, message="Notification sent.")
        if op == "screenshots.capture":
            artifact_dir = mana_home() / "artifacts" / "computer-control"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            path = artifact_dir / f"{action.execution_id}.png"
            mode = str(action.arguments.get("mode") or "full_screen")
            argv = ["screencapture", "-x"]
            if mode != "full_screen":
                raise CapabilityUnavailable("Active-window and selected-display capture require a ScreenCaptureKit adapter.")
            argv.append(str(path))
            await self._run(action, argv)
            return self._result(action, message="Screenshot captured.", data={"artifact_path": str(path)}, sensitive=True)
        raise CapabilityUnavailable(f"macOS operation {op!r} is unavailable.")
