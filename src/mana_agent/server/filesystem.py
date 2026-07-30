"""Remote filesystem command builders with atomic-write and backup semantics."""

from __future__ import annotations

import base64
from pathlib import PurePosixPath


def validate_remote_path(path: str) -> str:
    value = PurePosixPath(path)
    if not value.is_absolute() or ".." in value.parts:
        raise ValueError("Remote paths must be absolute and must not contain parent traversal.")
    return str(value)


def read_file_argv(path: str) -> list[str]:
    return ["cat", "--", validate_remote_path(path)]


def list_directory_argv(path: str) -> list[str]:
    return ["ls", "-la", "--", validate_remote_path(path)]


def atomic_write_script(path: str, content: str, *, mode: str = "0644", backup: bool = True) -> list[str]:
    target = validate_remote_path(path)
    if not mode.isdigit() or len(mode) not in {3, 4}:
        raise ValueError("mode must be an octal permission string")
    encoded = base64.b64encode(content.encode()).decode("ascii")
    # Values are positional parameters; secret-bearing file contents remain out
    # of the script text and are redacted before audit persistence.
    script = (
        "set -eu; target=$1; mode=$2; payload=$3; dir=$(dirname -- \"$target\"); "
        "tmp=$(mktemp \"$dir/.mana-write.XXXXXX\"); "
        "trap 'rm -f -- \"$tmp\"' EXIT; printf %s \"$payload\" | base64 -d >\"$tmp\"; "
        "chmod \"$mode\" \"$tmp\"; "
        + ("if [ -e \"$target\" ]; then cp -a -- \"$target\" \"$target.mana-backup\"; fi; " if backup else "")
        + "mv -f -- \"$tmp\" \"$target\"; trap - EXIT"
    )
    return ["sh", "-c", script, "mana-atomic-write", target, mode, encoded]
