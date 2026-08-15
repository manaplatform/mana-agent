"""Single-source package version resolution from pyproject.toml."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_DIST_NAME = "mana-agent"


def _read_pyproject_version(start: Path) -> str | None:
    """Walk parents from *start* and return ``[project].version`` if present."""
    for directory in (start, *start.parents):
        path = directory / "pyproject.toml"
        if not path.is_file():
            continue
        try:
            try:
                import tomllib
            except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
                import tomli as tomllib

            data = tomllib.loads(path.read_text(encoding="utf-8"))
            version = data.get("project", {}).get("version")
            if isinstance(version, str) and version.strip():
                return version.strip()
        except Exception:
            continue
    return None


@lru_cache(maxsize=1)
def get_version() -> str:
    """Return the package version.

    Preference order:
    1. ``[project].version`` from the nearest ``pyproject.toml`` (source tree).
    2. Installed distribution metadata for ``mana-agent``.
    3. ``\"dev\"`` when neither is available.
    """
    from_pyproject = _read_pyproject_version(Path(__file__).resolve().parent)
    if from_pyproject:
        return from_pyproject
    try:
        from importlib.metadata import version

        return version(_DIST_NAME)
    except Exception:
        return "dev"


@lru_cache(maxsize=1)
def get_runtime_git_sha() -> str:
    """Return build-provided source revision metadata without invoking git."""
    return str(
        os.environ.get("MANA_RUNTIME_GIT_SHA")
        or os.environ.get("BUILD_GIT_SHA")
        or os.environ.get("GIT_COMMIT_SHA")
        or ""
    ).strip()
