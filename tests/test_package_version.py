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


def test_langchain_constraints_are_resolvable_and_consistent() -> None:
    expected = {
        "langchain>=0.3.27,<1.0.0",
        "langchain-community>=0.3.27,<1.0.0",
        "langchain-openai>=0.3.27,<1.0.0",
    }
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_constraints = {
        item
        for item in pyproject["project"]["dependencies"]
        if item.startswith(("langchain>", "langchain-community>", "langchain-openai>"))
    }
    requirements_constraints = {
        line.strip()
        for line in (_REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.startswith(("langchain>", "langchain-community>", "langchain-openai>"))
    }

    assert package_constraints == expected
    assert requirements_constraints == expected


def test_full_extra_does_not_build_linux_desktop_input_dependencies() -> None:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]

    assert "pynput>=1.7,<2.0; sys_platform != 'linux'" in optional["full"]
    assert "pynput>=1.7,<2.0" in optional["teach-desktop"]
