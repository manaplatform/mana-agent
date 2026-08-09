"""Deterministic path confinement helpers for API and routing surfaces.

These helpers do not choose workflows or tools. They only validate that a
user-supplied path stays inside an explicit allowlist after normalization.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Sequence


_NULL_BYTE = "\x00"
_UNSAFE_SEGMENT = re.compile(r"^(?:\.|\.\.)$")


def env_allowed_roots(env_var: str = "MANA_WORKSPACE_ALLOWED_ROOTS") -> list[Path]:
    """Parse an allowlist of roots from a comma- or pathsep-separated env var."""
    raw = str(os.getenv(env_var) or "")
    parts = [
        item.strip()
        for item in re.split(r"[," + re.escape(os.pathsep) + r"]", raw)
        if item.strip()
    ]
    roots: list[Path] = []
    for item in parts:
        try:
            roots.append(Path(item).expanduser().resolve(strict=False))
        except OSError:
            continue
    return roots


def is_within_root(path: Path, root: Path) -> bool:
    """Return True when path equals root or is a descendant of root."""
    try:
        path_s = str(path)
        root_s = str(root)
    except Exception:
        return False
    if path_s == root_s:
        return True
    prefix = root_s if root_s.endswith(os.sep) else root_s + os.sep
    return path_s.startswith(prefix)


def is_within_any_root(path: Path, roots: Sequence[Path]) -> bool:
    return any(is_within_root(path, root) for root in roots)


def reject_unsafe_path_text(raw: str) -> str:
    """Reject empty, null-byte, or traversal-like path text before resolve."""
    text = str(raw or "").strip()
    if not text or _NULL_BYTE in text:
        raise ValueError("Invalid path.")
    # Normalize separators for segment inspection without resolving.
    normalized = text.replace("\\", "/")
    if "//" in normalized:
        raise ValueError("Invalid path.")
    for segment in normalized.split("/"):
        if not segment:
            continue
        if _UNSAFE_SEGMENT.match(segment):
            raise ValueError("Invalid path.")
    return text


def resolve_user_path(raw: str) -> Path:
    """Expand and resolve a user path after rejecting null bytes."""
    text = reject_unsafe_path_text(raw)
    try:
        return safe_resolve(text)
    except OSError as exc:
        raise ValueError("Invalid path.") from exc


def safe_resolve(path: str | Path, *, strict: bool = False) -> Path:
    """Resolve *path* without crashing when the process CWD is unusable.

    On Windows, ``ntpath.realpath`` (used by ``Path.resolve``) always calls
    ``os.getcwd()`` even for absolute paths. When the CWD was deleted—or tests
    simulate that by making ``getcwd`` raise—every ``resolve()`` fails.

    Prefer a real resolve when possible. If it fails and the path is already
    absolute, return a normalized absolute path. Relative paths that cannot be
    made absolute without a working CWD are returned normalized as-is.
    """
    candidate = Path(path).expanduser()
    try:
        return candidate.resolve(strict=strict)
    except (FileNotFoundError, OSError, RuntimeError):
        if candidate.is_absolute():
            return Path(os.path.normpath(candidate))
        try:
            cwd = os.getcwd()
        except (FileNotFoundError, OSError):
            return Path(os.path.normpath(candidate))
        return Path(os.path.normpath(os.path.join(cwd, str(candidate))))


def safe_cwd(*, fallback: str | Path | None = None) -> Path:
    """Return the process CWD, or a stable fallback when it was deleted.

    Unix processes can keep running after their working directory is unlinked
    (common when a SWE-bench worktree is recreated under a live agent). In that
    state ``os.getcwd()`` / ``Path.cwd()`` raise ``FileNotFoundError``. Callers
    that only need a stable path for log naming or child ``cwd`` must not crash.

    On Windows the process normally locks its CWD, but ``Path.resolve`` still
    calls ``os.getcwd`` via ``ntpath.realpath``; fallbacks must not re-raise.
    """
    try:
        return safe_resolve(Path.cwd())
    except (FileNotFoundError, OSError):
        pass
    if fallback is not None:
        candidate = Path(fallback).expanduser()
        try:
            if candidate.is_dir():
                return safe_resolve(candidate)
        except OSError:
            pass
    # Prefer Mana-managed state, then the user's home, then a stable absolute root.
    try:
        from mana_agent.config.settings import mana_home

        home = mana_home()
        if home.is_dir():
            return safe_resolve(home)
    except Exception:
        pass
    try:
        return safe_resolve(Path.home())
    except Exception:
        pass
    try:
        import tempfile

        return safe_resolve(Path(tempfile.gettempdir()))
    except Exception:
        # Never raise: last-ditch absolute roots differ by platform.
        if os.name == "nt":
            return Path(os.path.normpath("C:\\"))
        return Path("/")


def resolve_within_allowed_roots(
    raw: str,
    allowed_roots: Sequence[Path],
    *,
    require_allowlist: bool = True,
) -> Path:
    """Resolve raw and require it to sit under one of the allowed roots.

    When ``require_allowlist`` is True and allowed_roots is empty, raise.
    When False and allowed_roots is empty, return the resolved path only after
    basic validation (used for single-user local dashboard paths that are
    further constrained by callers).
    """
    path = resolve_user_path(raw)
    roots = [root.resolve(strict=False) for root in allowed_roots]
    if not roots:
        if require_allowlist:
            raise PermissionError("Path API is disabled until an allowlist is configured.")
        return path
    if not is_within_any_root(path, roots):
        raise PermissionError("Path is outside the configured allowlist.")
    return path


def resolve_under_base(raw: str, base: Path) -> tuple[Path, bool]:
    """Resolve a path for membership checks under a known base directory.

    Relative paths are joined under base. Absolute paths are resolved and
    checked with a startswith confinement test. Returns (resolved, is_member).
    """
    base_r = base.expanduser().resolve(strict=False)
    text = reject_unsafe_path_text(raw)
    candidate = Path(text).expanduser()
    try:
        if candidate.is_absolute():
            resolved = candidate.resolve(strict=False)
        else:
            # Strip leading ./ before joining so confinement stays under base.
            cleaned = text.replace("\\", "/").lstrip("./")
            resolved = (base_r / cleaned).resolve(strict=False)
    except OSError:
        return base_r, False
    return resolved, is_within_root(resolved, base_r)


def parse_absolute_allowed_paths(lines: Iterable[str]) -> list[Path]:
    """Parse one absolute path per line for computer-control allowlists."""
    paths: list[Path] = []
    for line in lines:
        text = str(line or "").strip()
        if not text:
            continue
        if _NULL_BYTE in text:
            raise ValueError("allowed_paths must not contain null bytes")
        path = Path(text).expanduser()
        if not path.is_absolute():
            raise ValueError("computer-control allowed_paths must contain only absolute paths")
        try:
            paths.append(path.resolve(strict=False))
        except OSError as exc:
            raise ValueError(f"invalid allowed path: {text}") from exc
    return paths


__all__ = [
    "env_allowed_roots",
    "is_within_any_root",
    "is_within_root",
    "parse_absolute_allowed_paths",
    "reject_unsafe_path_text",
    "resolve_under_base",
    "resolve_user_path",
    "resolve_within_allowed_roots",
    "safe_cwd",
    "safe_resolve",
]
