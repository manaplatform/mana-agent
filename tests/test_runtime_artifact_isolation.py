from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _metadata(path: Path) -> list[tuple[str, int, int]]:
    if not path.exists():
        return []
    return sorted(
        (str(item.relative_to(path)), item.stat().st_mtime_ns, item.stat().st_size)
        for item in path.rglob("*")
    )


def _run_nested_pytest(tmp_path: Path, *, failing: bool) -> tuple[subprocess.CompletedProcess[str], Path]:
    conftest = Path(__file__).with_name("conftest.py").read_text(encoding="utf-8")
    (tmp_path / "conftest.py").write_text(conftest, encoding="utf-8")
    record = tmp_path / "record.json"
    failure = "pytest.fail('intentional failure after creating runtime state')" if failing else ""
    (tmp_path / "test_child.py").write_text(
        "\n".join(
            [
                "import json",
                "import os",
                "from pathlib import Path",
                "import pytest",
                "",
                "def test_runtime_artifacts(real_mana_home):",
                "    home = Path(os.environ['MANA_HOME'])",
                "    assert Path.home() / '.mana' == home",
                "    for directory in ('repositories/repo-1', 'sessions/session-1', 'workspaces/workspace-1'):",
                "        target = home / directory",
                "        target.mkdir(parents=True)",
                "        (target / 'state.json').write_text('{}', encoding='utf-8')",
                "    with pytest.raises(PermissionError):",
                "        (real_mana_home / 'pytest-must-not-create-this').write_text('blocked', encoding='utf-8')",
                "    Path(os.environ['MANA_ISOLATION_RECORD']).write_text(json.dumps({'home': str(home)}), encoding='utf-8')",
                f"    {failure}" if failure else "",
            ]
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["MANA_ISOLATION_RECORD"] = str(record)
    return (
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        ),
        record,
    )


@pytest.mark.parametrize("failing", [False, True])
def test_nested_pytest_removes_all_mana_runtime_artifacts(tmp_path: Path, failing: bool) -> None:
    result, record = _run_nested_pytest(tmp_path, failing=failing)

    assert result.returncode == (1 if failing else 0), result.stdout + result.stderr
    isolated_home = Path(json.loads(record.read_text(encoding="utf-8"))["home"])
    assert not isolated_home.exists()


def test_real_mana_home_write_guard_preserves_existing_user_data(real_mana_home: Path) -> None:
    before = _metadata(real_mana_home)

    with pytest.raises(PermissionError, match="real Mana home"):
        (real_mana_home / "pytest-must-not-create-this").write_text("blocked", encoding="utf-8")

    assert _metadata(real_mana_home) == before
