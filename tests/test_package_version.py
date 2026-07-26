"""Package version is single-sourced from pyproject.toml."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
    import tomli as tomllib

from mana_agent import __version__
from mana_agent._version import get_version
from mana_agent.api.app import create_app

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SEMVER_LIKE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def _pyproject_version() -> str:
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = data["project"]["version"]
    assert isinstance(version, str) and version.strip()
    return version.strip()


def test_get_version_matches_pyproject() -> None:
    expected = _pyproject_version()
    assert get_version() == expected
    assert __version__ == expected
    assert _SEMVER_LIKE.fullmatch(expected)


def test_fastapi_app_version_matches_package() -> None:
    app = create_app(telegram_config=type("Cfg", (), {"enabled": False, "effective_transport": "polling"})())
    assert app.version == __version__
