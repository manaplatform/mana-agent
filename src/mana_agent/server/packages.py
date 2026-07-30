"""Package-manager detection and idempotent command construction."""

from __future__ import annotations

from typing import Literal

PackageManager = Literal["apt", "dnf", "yum", "pacman", "apk", "zypper", "brew"]


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
