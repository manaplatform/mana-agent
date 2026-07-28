"""Optional global desktop recorder with privacy-aware keyboard aggregation."""

from __future__ import annotations

import threading
import sys
import subprocess
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .models import (
    EventApplication,
    EventSource,
    EventTarget,
    RecordedEvent,
    RelativePosition,
    TeachError,
    TeachSession,
)


class NativeDesktopRecorder:
    """Capture shortcuts, navigation, typed text and pointer clicks.

    Characters are buffered only in memory and emitted as one semantic input
    event. Secure accessibility controls never expose their value, and the
    service redacts known secrets before an event reaches local storage.
    """

    recorder_id = "native-desktop"

    def __init__(self):
        self._session: TeachSession | None = None
        self._emit: Callable[[RecordedEvent], None] | None = None
        self._keyboard: Any = None
        self._mouse: Any = None
        self._paused = False
        self._lock = threading.RLock()
        self._modifiers: set[str] = set()
        self._typed_count = 0
        self._typed_characters: list[str] = []
        self._typing_target: EventTarget | None = None
        self._typing_application: EventApplication | None = None
        self._last_desktop_context: tuple[str, str] | None = None

    def capabilities(self) -> set[str]:
        return {"keyboard_shortcuts", "redacted_typing", "pointer_clicks", "active_application"}

    def start(self, session: TeachSession, emit: Callable[[RecordedEvent], None]) -> None:
        try:
            from pynput import keyboard, mouse  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TeachError(
                "Native desktop recorder is unavailable. Install `mana-agent[teach-desktop]`."
            ) from exc
        self._session = session
        self._emit = emit
        self._paused = False
        try:
            self._keyboard = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self._mouse = mouse.Listener(on_click=self._on_click)
            self._keyboard.start()
            self._mouse.start()
        except Exception as exc:
            self.stop()
            raise TeachError(
                "Native desktop recorder could not attach. Confirm OS Accessibility/Input Monitoring permission."
            ) from exc

    def pause(self) -> None:
        with self._lock:
            self._flush_typing()
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def stop(self) -> None:
        with self._lock:
            self._flush_typing()
            for listener in (self._keyboard, self._mouse):
                if listener is not None:
                    try:
                        listener.stop()
                    except Exception:
                        pass
            self._keyboard = None
            self._mouse = None
            self._session = None
            self._emit = None
            self._modifiers.clear()
            self._typed_characters.clear()
            self._typing_target = None
            self._typing_application = None
            self._last_desktop_context = None
            self._paused = False

    def poll_desktop_context(self) -> None:
        """Record application/window changes even when there is no input event."""
        with self._lock:
            if self._paused or self._session is None:
                return
            application = _active_application()
            window_title = _focused_window_title()
            context = (application.id, window_title)
            if context == self._last_desktop_context:
                return
            self._last_desktop_context = context
            self._publish(
                EventSource.APPLICATION,
                "activate",
                data={},
                context={"window_title": window_title} if window_title else {},
            )

    def _on_press(self, key: Any) -> None:
        with self._lock:
            if self._paused:
                return
            name = _key_name(key)
            if name in {"ctrl", "ctrl_l", "ctrl_r", "alt", "alt_l", "alt_r", "cmd", "cmd_l", "cmd_r", "shift", "shift_l", "shift_r"}:
                self._modifiers.add(name.split("_")[0])
                return
            character = getattr(key, "char", None)
            if (
                isinstance(character, str)
                and character.isprintable()
                and self._modifiers.issubset({"shift"})
            ):
                self._append_typed_character(character)
                return
            if name == "space" and self._modifiers.issubset({"shift"}):
                self._append_typed_character(" ")
                return
            if name == "backspace" and not self._modifiers:
                if self._typed_count:
                    self._typed_count -= 1
                    self._typed_characters.pop()
                return
            self._flush_typing()
            action = "shortcut" if self._modifiers else "navigate"
            self._publish(
                EventSource.KEYBOARD,
                action,
                data={"keys": sorted(self._modifiers) + [name]},
            )

    def _on_release(self, key: Any) -> None:
        with self._lock:
            name = _key_name(key).split("_")[0]
            self._modifiers.discard(name)

    def _on_click(self, x: float, y: float, button: Any, pressed: bool) -> None:
        if not pressed:
            return
        with self._lock:
            if self._paused:
                return
            self._flush_typing()
            width, height = _screen_size()
            self._publish(
                EventSource.POINTER,
                "click",
                data={"button": str(button).split(".")[-1]},
                fallback=RelativePosition(
                    x=max(0, min(1, float(x) / max(1, width))),
                    y=max(0, min(1, float(y) / max(1, height))),
                ),
            )

    def _flush_typing(self) -> None:
        if not self._typed_count:
            return
        count = self._typed_count
        value = "".join(self._typed_characters)
        target = self._typing_target
        application = self._typing_application
        self._typed_count = 0
        self._typed_characters.clear()
        self._typing_target = None
        self._typing_application = None
        if _is_secure_target(target):
            data: dict[str, Any] = {"character_count": count, "content_captured": False}
            sensitive = True
        else:
            data = {"value": value, "character_count": count, "content_captured": True}
            sensitive = False
        self._publish(
            EventSource.KEYBOARD,
            "type",
            data=data,
            sensitive=sensitive,
            application=application,
            target=target,
        )

    def _append_typed_character(self, character: str) -> None:
        if not self._typed_count:
            self._typing_application = _active_application()
            self._typing_target = _focused_accessibility_target()
        self._typed_count += 1
        self._typed_characters.append(character)

    def _publish(
        self,
        source: EventSource,
        action: str,
        *,
        data: dict[str, Any],
        context: dict[str, Any] | None = None,
        fallback: RelativePosition | None = None,
        sensitive: bool = False,
        application: EventApplication | None = None,
        target: EventTarget | None = None,
    ) -> None:
        if self._session is None or self._emit is None:
            return
        application = application or _active_application()
        target = target or _focused_accessibility_target()
        try:
            self._emit(
                RecordedEvent(
                    session_id=self._session.id,
                    timestamp=datetime.now(timezone.utc),
                    source=source,
                    action=action,
                    application=application,
                    target=target,
                    context=context or {},
                    data=data,
                    fallback_position=fallback,
                    sensitive=sensitive,
                )
            )
        except TeachError:
            # A paused/stopped session or an excluded application is expected
            # to drop the event without terminating the native listener.
            return


def _key_name(key: Any) -> str:
    return str(key).replace("Key.", "").strip("'").lower()


def _is_secure_target(target: EventTarget | None) -> bool:
    if target is None:
        return False
    details = " ".join(
        value for value in (target.role, target.name, target.label, target.automation_id) if value
    ).lower()
    return "secure" in details or any(
        marker in details for marker in ("password", "passcode", "secret", "token", "pin", "credit card")
    )


def _active_application() -> EventApplication:
    if sys.platform == "win32":
        try:
            import ctypes

            user32 = ctypes.windll.user32
            window = user32.GetForegroundWindow()
            title = ctypes.create_unicode_buffer(1024)
            class_name = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(window, title, len(title))
            user32.GetClassNameW(window, class_name, len(class_name))
            return EventApplication(id=class_name.value, name=title.value)
        except Exception:
            return EventApplication()
    if sys.platform.startswith("linux") and shutil.which("xdotool"):
        try:
            window_id = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True,
                text=True,
                timeout=1,
                check=True,
            ).stdout.strip()
            title = subprocess.run(
                ["xdotool", "getwindowname", window_id],
                capture_output=True,
                text=True,
                timeout=1,
                check=True,
            ).stdout.strip()
            class_name = subprocess.run(
                ["xdotool", "getwindowclassname", window_id],
                capture_output=True,
                text=True,
                timeout=1,
                check=True,
            ).stdout.strip()
            return EventApplication(id=class_name, name=title)
        except (OSError, subprocess.SubprocessError):
            return EventApplication()
    try:
        from AppKit import NSWorkspace  # type: ignore[import-not-found]

        application = NSWorkspace.sharedWorkspace().frontmostApplication()
        return EventApplication(
            id=str(application.bundleIdentifier() or ""),
            name=str(application.localizedName() or ""),
        )
    except Exception:
        return EventApplication()


def _screen_size() -> tuple[float, float]:
    if sys.platform == "win32":
        try:
            import ctypes

            return float(ctypes.windll.user32.GetSystemMetrics(0)), float(
                ctypes.windll.user32.GetSystemMetrics(1)
            )
        except Exception:
            return 1, 1
    try:
        from AppKit import NSScreen  # type: ignore[import-not-found]

        frame = NSScreen.mainScreen().frame()
        return float(frame.size.width), float(frame.size.height)
    except Exception:
        try:
            import tkinter

            root = tkinter.Tk()
            root.withdraw()
            size = float(root.winfo_screenwidth()), float(root.winfo_screenheight())
            root.destroy()
            return size
        except Exception:
            return 1, 1


def _focused_accessibility_target() -> EventTarget:
    """Best-effort semantic target; absence never falls back to captured text."""
    try:
        import ApplicationServices  # type: ignore[import-not-found]

        system = ApplicationServices.AXUIElementCreateSystemWide()
        error, element = ApplicationServices.AXUIElementCopyAttributeValue(
            system, ApplicationServices.kAXFocusedUIElementAttribute, None
        )
        if error != 0 or element is None:
            return EventTarget()

        def attribute(name: str) -> str | None:
            result, value = ApplicationServices.AXUIElementCopyAttributeValue(element, name, None)
            return str(value) if result == 0 and value is not None else None

        role = attribute(ApplicationServices.kAXRoleAttribute)
        name = attribute(ApplicationServices.kAXTitleAttribute) or attribute(
            ApplicationServices.kAXDescriptionAttribute
        )
        identifier_name = getattr(ApplicationServices, "kAXIdentifierAttribute", "AXIdentifier")
        automation_id = attribute(identifier_name)
        return EventTarget(
            role=role,
            name=name,
            automation_id=automation_id,
            hierarchy=[],
        )
    except Exception:
        return EventTarget()


def _focused_window_title() -> str:
    if sys.platform != "darwin":
        return _active_application().name
    try:
        import ApplicationServices  # type: ignore[import-not-found]

        system = ApplicationServices.AXUIElementCreateSystemWide()
        error, application = ApplicationServices.AXUIElementCopyAttributeValue(
            system, ApplicationServices.kAXFocusedApplicationAttribute, None
        )
        if error != 0 or application is None:
            return ""
        result, window = ApplicationServices.AXUIElementCopyAttributeValue(
            application, ApplicationServices.kAXFocusedWindowAttribute, None
        )
        if result != 0 or window is None:
            return ""
        result, title = ApplicationServices.AXUIElementCopyAttributeValue(
            window, ApplicationServices.kAXTitleAttribute, None
        )
        return str(title) if result == 0 and title is not None else ""
    except Exception:
        return ""
