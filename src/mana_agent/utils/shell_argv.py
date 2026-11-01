"""Mechanical local shell argv construction after a validated decision."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path


def local_shell_argv(command: str) -> list[str]:
    """Build argv for a local shell one-liner without login-shell PATH reset.

    Prefer non-login ``sh -c`` / ``cmd /c`` so the process inherits the caller's
    ``PATH`` (including SWE-bench ``agent_bin`` Python 3 shims). Login shells
    (``sh -lc``) re-source profile files and commonly put host Python 2.7 ahead
    of those shims, which breaks ``python -m compileall`` and empty-patch runs.
    """
    text = str(command or "")
    if os.name == "nt":
        return ["cmd.exe", "/d", "/s", "/c", text]
    return ["/bin/sh", "-c", text]


def windows_safe_subprocess_argv(argv: Sequence[str] | str) -> Sequence[str] | str:
    """Format argv for subprocess execution on Windows without list2cmdline quote mangling.

    When running cmd.exe /d /s /c on Windows, passing argv as a list causes Python's
    subprocess module to invoke list2cmdline(), which escapes internal double quotes
    with backslashes (\\"). Because cmd.exe does not unescape backslashes, downstream
    commands (such as python -c "print(123)") receive literal quotes, turning executable
    code into no-op string literals or syntax errors.

    Passing a single command string wrapped in outer quotes allows cmd.exe /s to strip
    the outer quotes and execute the command with arguments intact.
    """
    if (
        os.name == "nt"
        and isinstance(argv, (list, tuple))
        and len(argv) >= 3
        and Path(str(argv[0])).name.lower() in {"cmd.exe", "cmd"}
    ):
        c_idx = next((i for i, arg in enumerate(argv[1:], 1) if str(arg).lower() == "/c"), None)
        if c_idx is not None and c_idx + 1 < len(argv):
            flags = " ".join(argv[1:c_idx + 1])
            if not any(str(arg).lower() == "/s" for arg in argv[1:c_idx + 1]):
                flags = f"/s {flags}"
            cmd = argv[c_idx + 1] if len(argv) == c_idx + 2 else " ".join(argv[c_idx + 1:])
            return f'"{argv[0]}" {flags} "{cmd}"'
    return argv


__all__ = ["local_shell_argv", "windows_safe_subprocess_argv"]
