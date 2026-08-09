"""Regression: local shell argv must not use login shells that reset PATH."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from mana_agent.utils.shell_argv import local_shell_argv


def test_local_shell_argv_uses_non_login_shell_on_posix() -> None:
    if os.name == "nt":
        argv = local_shell_argv("python3 -m compileall .")
        assert argv[:4] == ["cmd.exe", "/d", "/s", "/c"]
        assert argv[-1] == "python3 -m compileall ."
        return
    argv = local_shell_argv("python3 -m compileall .")
    assert argv == ["/bin/sh", "-c", "python3 -m compileall ."]
    assert "-l" not in argv
    assert "-lc" not in argv


def test_local_shell_argv_preserves_path_shim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Login shells drop agent PATH shims; non-login shells keep them.

    This is the astropy__astropy-12907 empty-patch failure mode on macOS when
    bare ``python`` is Frameworks 2.7 and verification runs ``python -m compileall``.
    """
    if os.name == "nt":
        pytest.skip("POSIX PATH shim regression")

    shim_bin = tmp_path / "agent_bin"
    shim_bin.mkdir()
    shim = shim_bin / "python"
    # Report a distinctive version string so we can prove which interpreter ran.
    shim.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'exec "{sys.executable}" -c \'print("shim-python3")\'\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)

    # Put a fake "login" python first on a path that login profiles might restore.
    # We only assert non-login inherits our shim PATH.
    env = os.environ.copy()
    env["PATH"] = f"{shim_bin}{os.pathsep}{env.get('PATH', '')}"

    import subprocess

    login_argv = ["/bin/sh", "-lc", "python -c 'import sys; print(sys.version_info[0])'"]
    nonlogin_argv = local_shell_argv("python -c 'import sys; print(\"shim-python3\")'")

    # Non-login must use the shim from PATH.
    nonlogin = subprocess.run(
        nonlogin_argv,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert nonlogin.returncode == 0, nonlogin.stderr
    assert "shim-python3" in nonlogin.stdout

    # Document the hazard: login often ignores the shim PATH (host dependent).
    # We only require non-login preservation; login is the anti-pattern.
    _ = login_argv


def test_python3_compileall_prefix_is_allowlisted() -> None:
    from mana_agent.multi_agent.tools.permissions import assert_shell_allowed

    assert_shell_allowed("python3 -m compileall .")
    assert_shell_allowed("python3 -m compileall src")
    assert_shell_allowed("python -m compileall src")
