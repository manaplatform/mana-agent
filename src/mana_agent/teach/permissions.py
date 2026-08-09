"""Explicit local and operating-system grants for desktop teaching."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .models import TeachError


TeachGrantScope = Literal[
    "teach.record.accessibility",
    "teach.record.keyboard",
    "teach.record.pointer",
    "teach.record.applications",
]
DESKTOP_GRANTS: tuple[TeachGrantScope, ...] = (
    "teach.record.accessibility",
    "teach.record.keyboard",
    "teach.record.pointer",
    "teach.record.applications",
)


def _restrict_descriptor_to_owner(fd: int) -> None:
    """Apply POSIX permissions when the operating system supports them."""
    fchmod = getattr(os, "fchmod", None)
    if fchmod is not None:
        fchmod(fd, 0o600)


def _optional_module_available(name: str) -> bool:
    """Check optional desktop dependencies without initializing their adapters."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


class GrantStatus(BaseModel):
    scope: TeachGrantScope
    mana_granted: bool
    os_granted: bool | None
    available: bool
    reason: str = ""


class TeachGrantStore:
    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()

    def grant(self, scopes: list[TeachGrantScope]) -> None:
        payload = self._load()
        for scope in scopes:
            payload[scope] = True
        self._save(payload)

    def revoke(self, scopes: list[TeachGrantScope]) -> None:
        payload = self._load()
        for scope in scopes:
            payload.pop(scope, None)
        self._save(payload)

    def is_granted(self, scope: TeachGrantScope) -> bool:
        return self._load().get(scope) is True

    def _load(self) -> dict[str, bool]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {str(key): bool(item) for key, item in value.items()} if isinstance(value, dict) else {}

    def _save(self, payload: dict[str, bool]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary = tempfile.mkstemp(prefix=".teach-grants.", dir=self.path.parent)
        try:
            _restrict_descriptor_to_owner(fd)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def grant_status(store: TeachGrantStore) -> list[GrantStatus]:
    availability = _native_availability()
    return [
        GrantStatus(
            scope=scope,
            mana_granted=store.is_granted(scope),
            os_granted=availability[scope]["os_granted"],
            available=availability[scope]["available"],
            reason=availability[scope]["reason"],
        )
        for scope in DESKTOP_GRANTS
    ]


def require_desktop_grants(store: TeachGrantStore) -> None:
    statuses = grant_status(store)
    missing_mana = [item.scope for item in statuses if not item.mana_granted]
    unavailable = [item.scope for item in statuses if not item.available or item.os_granted is False]
    if missing_mana:
        raise TeachError(
            "Desktop recording requires explicit Mana grants: "
            + ", ".join(missing_mana)
            + ". Run `mana-agent teach grant --scope full --allow`."
        )
    if unavailable:
        raise TeachError(
            "Desktop recording OS permissions or dependencies are unavailable: "
            + ", ".join(unavailable)
            + ". Run `mana-agent teach grant --scope full --open-settings`, approve them, then retry."
        )


def open_permission_settings(scopes: list[TeachGrantScope]) -> list[str]:
    """Open OS-owned settings panes; Mana never edits OS privacy databases."""
    opened: list[str] = []
    if sys.platform == "darwin":
        panes: set[str] = set()
        if "teach.record.accessibility" in scopes:
            panes.add("x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")
        if {"teach.record.keyboard", "teach.record.pointer"} & set(scopes):
            panes.add("x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent")
        for pane in sorted(panes):
            subprocess.run(["open", pane], check=False, capture_output=True)
            opened.append(pane)
    elif sys.platform == "win32":
        subprocess.run(["cmd", "/c", "start", "ms-settings:privacy"], check=False, capture_output=True)
        opened.append("ms-settings:privacy")
    return opened


def _native_availability() -> dict[TeachGrantScope, dict[str, object]]:
    input_available = _optional_module_available("pynput")
    input_reason = (
        ""
        if input_available
        else "Install the Teach Mode desktop extra: pip install 'mana-agent[teach-desktop]'."
    )
    accessibility_available = False
    accessibility_os: bool | None = None
    accessibility_reason = ""
    if sys.platform == "darwin":
        if not _optional_module_available("ApplicationServices"):
            accessibility_reason = "Install the macOS Teach dependency (PyObjC Quartz)."
        else:
            try:
                import ApplicationServices  # type: ignore[import-not-found]

                accessibility_os = bool(ApplicationServices.AXIsProcessTrusted())
                accessibility_available = accessibility_os
                if not accessibility_os:
                    accessibility_reason = "macOS Accessibility permission is not granted."
            except (ImportError, AttributeError):
                accessibility_reason = "Install the macOS Teach dependency (PyObjC Quartz)."
    elif sys.platform == "win32":
        if _optional_module_available("uiautomation"):
            accessibility_available = True
            accessibility_os = True
        else:
            accessibility_reason = "Install the Windows UI Automation adapter."
    elif sys.platform.startswith("linux"):
        if _optional_module_available("pyatspi"):
            accessibility_available = True
            accessibility_os = True
        else:
            accessibility_reason = "Install AT-SPI Python bindings and enable accessibility."
    return {
        "teach.record.accessibility": {
            "available": accessibility_available,
            "os_granted": accessibility_os,
            "reason": accessibility_reason,
        },
        "teach.record.keyboard": {
            "available": input_available,
            # Global input permission cannot be probed safely without attaching a listener.
            "os_granted": None if input_available else False,
            "reason": input_reason or "OS input permission is confirmed when the recorder attaches.",
        },
        "teach.record.pointer": {
            "available": input_available,
            "os_granted": None if input_available else False,
            "reason": input_reason or "OS input permission is confirmed when the recorder attaches.",
        },
        "teach.record.applications": {
            "available": True,
            "os_granted": True,
            "reason": "",
        },
    }
