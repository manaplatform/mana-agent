"""Import-safe platform capability detection and semantic capture hooks."""

from __future__ import annotations

import os
import platform as runtime_platform
import sys
from typing import Any


def platform_name() -> str:
    return {"darwin": "macos", "win32": "windows"}.get(sys.platform, "linux" if sys.platform.startswith("linux") else "unsupported")


def doctor_report(*, browser_enabled: bool = True, voice_enabled: bool = False) -> dict[str, Any]:
    current = platform_name()
    headless = not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) if current == "linux" else False
    accessibility_available = False
    reason = ""
    if current == "macos":
        try:
            import ApplicationServices  # type: ignore[import-not-found]
            accessibility_available = bool(ApplicationServices.AXIsProcessTrusted())
            if not accessibility_available:
                reason = "Grant Mana-Agent Accessibility permission in macOS System Settings."
        except (ImportError, AttributeError):
            reason = "Install the optional macOS accessibility dependency and grant Accessibility permission."
    elif current == "windows":
        try:
            import uiautomation  # type: ignore[import-not-found]  # noqa: F401
            accessibility_available = True
        except ImportError:
            reason = "Install the optional Windows UI Automation dependency."
    elif current == "linux" and not headless:
        try:
            import pyatspi  # type: ignore[import-not-found]  # noqa: F401
            accessibility_available = True
        except ImportError:
            reason = "Install AT-SPI Python bindings and enable the accessibility bus."
    else:
        reason = "Desktop accessibility capture is unavailable in this environment."
    try:
        import playwright  # type: ignore[import-not-found]  # noqa: F401
        browser_available = browser_enabled
    except ImportError:
        browser_available = False
    recorders = {
        "manual_semantic": {"available": True, "reason": "Events can be supplied by integrated Mana tools."},
        "accessibility": {"available": accessibility_available, "reason": reason},
        "browser": {
            "available": browser_available,
            "reason": "" if browser_available else "Install mana-agent[browser] and enable browser capture.",
        },
        "voice": {
            "available": False,
            "reason": "Voice capture is optional and disabled." if not voice_enabled else "No voice adapter installed.",
        },
    }
    return {
        "platform": current,
        "platform_release": runtime_platform.release(),
        "headless": headless,
        "recorders": recorders,
        "limitations": [item["reason"] for item in recorders.values() if not item["available"] and item["reason"]],
    }
