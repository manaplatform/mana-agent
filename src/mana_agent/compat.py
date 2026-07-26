"""Small standard-library compatibility imports for supported Python versions."""

from __future__ import annotations

import os

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - exercised on Python 3.10 CI
    from enum import Enum

    class StrEnum(str, Enum):
        """Python 3.10-compatible subset used by Mana's explicit-value enums."""

        def __str__(self) -> str:
            return str(self.value)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib


def _windows_process_exists(pid: int) -> bool:
    """Check a Windows PID without sending it a signal.

    Unlike POSIX, Windows implements ``os.kill(pid, 0)`` with
    ``TerminateProcess``. A read-only process handle is therefore required for
    a non-destructive liveness probe.
    """
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    error_invalid_parameter = 87

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        # Access-denied and other indeterminate failures must be treated as
        # alive so recovery cannot steal another process's state.
        return ctypes.get_last_error() != error_invalid_parameter

    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def process_exists(pid: int) -> bool:
    """Return whether *pid* identifies a live process without modifying it."""
    if pid <= 0:
        return False
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        return _windows_process_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


__all__ = ["StrEnum", "process_exists", "tomllib"]
