"""Mechanical local shell argv construction after a validated decision."""

from __future__ import annotations

import os


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


__all__ = ["local_shell_argv"]
