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
    get_version.cache_clear()
    expected = _pyproject_version()
    assert get_version() == expected
    assert __version__ == expected
    assert _SEMVER_LIKE.fullmatch(expected)


def test_fastapi_app_version_matches_package() -> None:
    app = create_app(telegram_config=type("Cfg", (), {"enabled": False, "effective_transport": "polling"})())
    assert app.version == __version__


def test_native_dependencies_are_direct_and_forbidden_families_absent() -> None:
    from packaging.requirements import Requirement
    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirements = [line.strip() for line in (_REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.startswith("#")]
    names = {Requirement(item).name.lower().replace("_", "-") for item in project["dependencies"]}
    for group in [project["dependencies"], *project["optional-dependencies"].values(), requirements]:
        for item in group:
            name = Requirement(item).name.lower().replace("_", "-")
            assert not name.startswith(("langchain", "langgraph", "langsmith", "faiss"))
    assert {"openai", "anthropic", "httpx"} <= names
    assert {str(Requirement(item)) for item in requirements} == {str(Requirement(item)) for item in project["dependencies"]}


def test_audit_sources_handles_utf8_source_files_with_multibyte_characters(tmp_path: Path) -> None:
    import importlib.util

    script_path = _REPO_ROOT / "scripts" / "audit_native_dependencies.py"
    spec = importlib.util.spec_from_file_location("audit_native_dependencies", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Multi-byte UTF-8 sequences containing byte 0x9d (e.g. right double quote \u201d -> \xe2\x80\x9d)
    source_file = tmp_path / "sample_unicode.py"
    source_file.write_bytes(
        '"""Docstring with curly quotes “test” and symbols — ┝."""\n'
        'MSG = "Hello from Windows UTF-8 test"\n'.encode("utf-8")
    )
    assert b"\x9d" in source_file.read_bytes()

    failures = module.audit_sources(tmp_path)
    assert failures == []

    # Verify forbidden import detection continues to work in UTF-8 source files
    bad_file = tmp_path / "forbidden.py"
    forbidden_import = "import " + "lang" + "chain\n"
    bad_file.write_bytes(
        '"""Docstring with ”."""\n'.encode("utf-8")
        + forbidden_import.encode("utf-8")
    )
    forbidden_failures = module.audit_sources(tmp_path)
    assert any("forbidden import lang" in failure for failure in forbidden_failures)


def test_full_extra_does_not_build_linux_desktop_input_dependencies() -> None:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]

    assert "pynput>=1.7,<2.0; sys_platform != 'linux'" in optional["full"]
    assert "pynput>=1.7,<2.0" in optional["teach-desktop"]
