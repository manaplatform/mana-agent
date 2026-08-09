"""Import-safe platform capability detection and semantic capture hooks."""

from __future__ import annotations

import importlib.util
import os
import platform as runtime_platform
import sys
from typing import Any


def platform_name() -> str:
    return {"darwin": "macos", "win32": "windows"}.get(sys.platform, "linux" if sys.platform.startswith("linux") else "unsupported")


def _optional_module_available(name: str) -> bool:
    """Detect an optional adapter without importing platform-specific code."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def doctor_report(*, browser_enabled: bool = True, voice_enabled: bool = False) -> dict[str, Any]:
    current = platform_name()
    headless = not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) if current == "linux" else False
    accessibility_available = False
    reason = ""
    if current == "macos":
        if not _optional_module_available("ApplicationServices"):
            reason = "Install the optional macOS accessibility dependency and grant Accessibility permission."
        else:
            try:
                import ApplicationServices  # type: ignore[import-not-found]
                accessibility_available = bool(ApplicationServices.AXIsProcessTrusted())
                if not accessibility_available:
                    reason = "Grant Mana-Agent Accessibility permission in macOS System Settings."
            except (ImportError, AttributeError):
                reason = "Install the optional macOS accessibility dependency and grant Accessibility permission."
    elif current == "windows":
        if _optional_module_available("uiautomation"):
            accessibility_available = True
        else:
            reason = "Install the optional Windows UI Automation dependency."
    elif current == "linux" and not headless:
        if _optional_module_available("pyatspi"):
            accessibility_available = True
        else:
            reason = "Install AT-SPI Python bindings and enable the accessibility bus."
    else:
        reason = "Desktop accessibility capture is unavailable in this environment."
    browser_available = browser_enabled and _optional_module_available("playwright")
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
