"""safe_cwd must not crash when the process working directory is deleted."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from mana_agent.utils.path_safety import safe_cwd


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
    monkeypatch.chdir(work)
    shutil.rmtree(work)
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
    monkeypatch.chdir(work)
    shutil.rmtree(work)
    monkeypatch.setenv("MANA_HOME", str(mana))
    resolved = safe_cwd()
    assert resolved == mana.resolve()
