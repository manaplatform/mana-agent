"""Package-manager detection and idempotent command construction."""

from __future__ import annotations

import shlex
from typing import Literal

PackageManager = Literal["apt", "dnf", "yum", "pacman", "apk", "zypper", "brew"]
PackageManagerSelection = Literal["auto", "apt", "dnf", "yum", "pacman", "apk", "zypper", "brew"]


def validate_package_arguments(
    arguments: dict[str, object],
) -> tuple[PackageManagerSelection, list[str]]:
    """Validate the exact model-selected package manager and package names."""
    extra = sorted(set(arguments) - {"manager", "packages"})
    if extra:
        raise ValueError(f"package arguments contain unknown fields: {', '.join(extra)}")
    manager = str(arguments.get("manager") or "")
    supported = {"auto", "apt", "dnf", "yum", "pacman", "apk", "zypper", "brew"}
    if manager not in supported:
        raise ValueError(
            "package manager must be one of auto, apt, dnf, yum, pacman, apk, zypper, or brew"
        )
    packages = arguments.get("packages")
    if (
        not isinstance(packages, list)
        or not packages
        or any(
            not isinstance(item, str) or not item or item.startswith("-")
            for item in packages
        )
    ):
        raise ValueError("packages must be a non-empty list of explicit non-option names")
    return manager, packages  # type: ignore[return-value]


def detect_package_manager(os_release: str, available_commands: set[str]) -> PackageManager:
    """Select only from observed remote evidence, never from a default fallback."""
    ordered = ("apt", "dnf", "yum", "pacman", "apk", "zypper", "brew")
    matches = [item for item in ordered if item in available_commands]
    if len(matches) == 1:
        return matches[0]  # type: ignore[return-value]
    if not matches:
        raise LookupError(
            f"No supported package manager was observed for {os_release or 'the remote OS'}. "
            "No package action was selected."
        )
    # Alternatives such as dnf/yum can coexist. Require the model decision to
    # name one rather than silently choosing based on a heuristic.
    raise ValueError(f"Multiple package managers are available ({', '.join(matches)}); select one explicitly.")


def package_installed_argv(manager: PackageManager, package: str) -> list[str]:
    commands = {
        "apt": ["dpkg-query", "-W", "-f=${Status}", package],
        "dnf": ["rpm", "-q", package],
        "yum": ["rpm", "-q", package],
        "pacman": ["pacman", "-Q", package],
        "apk": ["apk", "info", "-e", package],
        "zypper": ["rpm", "-q", package],
        "brew": ["brew", "list", "--versions", package],
    }
    return commands[manager]


def package_install_argv(manager: PackageManager, packages: list[str]) -> list[str]:
    if not packages or any(not item or item.startswith("-") for item in packages):
        raise ValueError("Package names must be explicit non-option values.")
    commands = {
        "apt": ["sudo", "apt-get", "install", "-y", "--", *packages],
        "dnf": ["sudo", "dnf", "install", "-y", "--", *packages],
        "yum": ["sudo", "yum", "install", "-y", "--", *packages],
        "pacman": ["sudo", "pacman", "-S", "--needed", "--noconfirm", "--", *packages],
        "apk": ["sudo", "apk", "add", "--", *packages],
        "zypper": ["sudo", "zypper", "--non-interactive", "install", "--", *packages],
        "brew": ["brew", "install", *packages],
    }
    return commands[manager]


def package_install_auto_argv(packages: list[str]) -> list[str]:
    """Discover exactly one supported remote manager before executing the selected install."""
    _, validated_packages = validate_package_arguments(
        {"manager": "auto", "packages": packages}
    )
    commands = {
        "apt": ["sudo", "apt-get", "install", "-y", "--", *validated_packages],
        "dnf": ["sudo", "dnf", "install", "-y", "--", *validated_packages],
        "yum": ["sudo", "yum", "install", "-y", "--", *validated_packages],
        "pacman": [
            "sudo", "pacman", "-S", "--needed", "--noconfirm", "--", *validated_packages,
        ],
        "apk": ["sudo", "apk", "add", "--", *validated_packages],
        "zypper": [
            "sudo", "zypper", "--non-interactive", "install", "--", *validated_packages,
        ],
        "brew": ["brew", "install", *validated_packages],
    }
    executables = {
        "apt": "apt-get",
        "dnf": "dnf",
        "yum": "yum",
        "pacman": "pacman",
        "apk": "apk",
        "zypper": "zypper",
        "brew": "brew",
    }
    lines = ["set -eu", "manager=''", "manager_count=0"]
    for manager, executable in executables.items():
        lines.extend(
            [
                f"if command -v {executable} >/dev/null 2>&1; then",
                f"  manager={manager}",
                "  manager_count=$((manager_count + 1))",
                "fi",
            ]
        )
    lines.extend(
        [
            'if [ "$manager_count" -ne 1 ]; then',
            '  echo "Expected exactly one supported package manager; observed $manager_count." >&2',
            "  exit 78",
            "fi",
            'case "$manager" in',
        ]
    )
    for manager, argv in commands.items():
        lines.append(f"  {manager}) exec {shlex.join(argv)} ;;")
    lines.extend(["  *) exit 78 ;;", "esac"])
    return ["sh", "-c", "\n".join(lines)]


def package_remove_argv(manager: PackageManager, packages: list[str], *, purge: bool = False) -> list[str]:
    if not packages or any(not item or item.startswith("-") for item in packages):
        raise ValueError("Package names must be explicit non-option values.")
    if manager == "apt":
        return ["sudo", "apt-get", "purge" if purge else "remove", "-y", "--", *packages]
    commands = {
        "dnf": ["sudo", "dnf", "remove", "-y", "--", *packages],
        "yum": ["sudo", "yum", "remove", "-y", "--", *packages],
        "pacman": ["sudo", "pacman", "-R", "--noconfirm", "--", *packages],
        "apk": ["sudo", "apk", "del", "--", *packages],
        "zypper": ["sudo", "zypper", "--non-interactive", "remove", "--", *packages],
        "brew": ["brew", "uninstall", *packages],
    }
    return commands[manager]
