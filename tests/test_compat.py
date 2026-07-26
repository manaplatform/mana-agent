from __future__ import annotations

import os

from mana_agent import compat


def test_process_exists_rejects_non_positive_pid_without_probe(monkeypatch) -> None:
    def unexpected_probe(_pid: int, _signal: int) -> None:
        raise AssertionError("non-positive PIDs must not be probed")

    monkeypatch.setattr(compat.os, "kill", unexpected_probe)

    assert compat.process_exists(0) is False
    assert compat.process_exists(-1) is False


def test_process_exists_uses_read_only_windows_probe(monkeypatch) -> None:
    probed: list[int] = []

    monkeypatch.setattr(compat.os, "name", "nt")
    monkeypatch.setattr(compat, "_windows_process_exists", lambda pid: probed.append(pid) or True)
    monkeypatch.setattr(
        compat.os,
        "kill",
        lambda _pid, _signal: (_ for _ in ()).throw(
            AssertionError("os.kill(pid, 0) is destructive on Windows")
        ),
    )

    assert compat.process_exists(1234) is True
    assert probed == [1234]


def test_process_exists_uses_signal_zero_on_posix(monkeypatch) -> None:
    probed: list[tuple[int, int]] = []

    monkeypatch.setattr(compat.os, "name", "posix")
    monkeypatch.setattr(compat.os, "kill", lambda pid, signal: probed.append((pid, signal)))

    assert compat.process_exists(os.getpid()) is True
    assert probed == [(os.getpid(), 0)]
