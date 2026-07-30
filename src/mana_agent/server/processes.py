"""Process administration command builders."""

from __future__ import annotations


def process_list_argv() -> list[str]:
    return ["ps", "axo", "pid,ppid,user,%cpu,%mem,stat,lstart,command"]


def process_signal_argv(pid: int, signal: str = "TERM") -> list[str]:
    if pid < 2 or signal not in {"TERM", "HUP", "INT", "KILL", "USR1", "USR2"}:
        raise ValueError("An exact non-system PID and supported signal are required.")
    return ["sudo", "kill", f"-{signal}", str(pid)]
