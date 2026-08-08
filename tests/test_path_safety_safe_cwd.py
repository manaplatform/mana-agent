"""safe_cwd must not crash when the process working directory is deleted."""

from __future__ import annotations

import errno
import os
import shutil
import sys
from pathlib import Path

from mana_agent.utils.path_safety import safe_cwd


def _make_cwd_unusable(work: Path, monkeypatch) -> None:
    """Leave the process as if its CWD was unlinked (SWE-bench worktree thrash).

    POSIX: chdir into *work* then delete it so ``os.getcwd()`` raises.
    Windows: the process locks its CWD (``rmtree`` → WinError 32), so simulate
    the same ``FileNotFoundError`` branch that ``safe_cwd`` handles on Unix.
    """
    if sys.platform == "win32":
        def _getcwd_gone() -> str:
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", str(work))

        monkeypatch.setattr(os, "getcwd", _getcwd_gone)
        return
    monkeypatch.chdir(work)
    shutil.rmtree(work)


def test_safe_cwd_returns_existing_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = safe_cwd()
    assert resolved == tmp_path.resolve()
    assert resolved.is_dir()


def test_safe_cwd_falls_back_when_directory_deleted(
    tmp_path: Path, monkeypatch
) -> None:
    work = tmp_path / "live"
    work.mkdir()
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    _make_cwd_unusable(work, monkeypatch)
    resolved = safe_cwd(fallback=fallback)
    assert resolved == fallback.resolve()
    assert resolved.is_dir()


def test_safe_cwd_prefers_mana_home_when_no_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    work = tmp_path / "gone"
    work.mkdir()
    mana = tmp_path / "mana-home"
    mana.mkdir()
    _make_cwd_unusable(work, monkeypatch)
    monkeypatch.setenv("MANA_HOME", str(mana))
    resolved = safe_cwd()
    assert resolved == mana.resolve()
