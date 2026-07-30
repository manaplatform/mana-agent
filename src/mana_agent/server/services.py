"""Service-manager detection and safe command construction."""

from __future__ import annotations

from typing import Literal

ServiceManager = Literal["systemd", "openrc", "sysv"]


def detect_service_manager(available_commands: set[str], pid_one_name: str) -> ServiceManager:
    if pid_one_name == "systemd" and "systemctl" in available_commands:
        return "systemd"
    matches = []
    if "rc-service" in available_commands:
        matches.append("openrc")
    if "service" in available_commands:
        matches.append("sysv")
    if len(matches) != 1:
        raise ValueError("Service manager evidence is missing or ambiguous; no service action was selected.")
    return matches[0]  # type: ignore[return-value]


def service_argv(manager: ServiceManager, action: str, service: str) -> list[str]:
    allowed = {"status", "start", "stop", "restart", "reload", "enable", "disable"}
    if action not in allowed or not service or service.startswith("-"):
        raise ValueError("Service action and exact service name are required.")
    if manager == "systemd":
        return ["sudo", "systemctl", action, "--", service]
    if action in {"enable", "disable"}:
        if manager != "openrc":
            raise ValueError("Enable/disable is unsupported for the selected SysV manager.")
        return ["sudo", "rc-update", "add" if action == "enable" else "del", service, "default"]
    return ["sudo", "rc-service" if manager == "openrc" else "service", service, action]


def service_logs_argv(manager: ServiceManager, service: str, lines: int = 100) -> list[str]:
    if manager != "systemd":
        raise ValueError("Structured service log reading currently requires systemd/journalctl evidence.")
    return ["journalctl", "-u", service, "-n", str(max(1, min(lines, 10_000))), "--no-pager"]
