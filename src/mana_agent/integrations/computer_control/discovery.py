"""Strict platform auto-detection and provider construction."""

from __future__ import annotations

import sys

from mana_agent.integrations.computer_control.config import ComputerControlSettings
from mana_agent.integrations.computer_control.errors import UnsupportedPlatform
from mana_agent.integrations.computer_control.models import SupportedPlatform
from mana_agent.integrations.computer_control.providers.linux import LinuxProvider
from mana_agent.integrations.computer_control.providers.macos import MacOSProvider
from mana_agent.integrations.computer_control.providers.windows import WindowsProvider
from mana_agent.integrations.computer_control.registry import ProviderRegistry


def detect_platform(platform_name: str | None = None) -> SupportedPlatform:
    value = platform_name or sys.platform
    if value == "darwin" or value == "macos":
        return SupportedPlatform.MACOS
    if value.startswith("win") or value == "windows":
        return SupportedPlatform.WINDOWS
    if value.startswith("linux"):
        return SupportedPlatform.LINUX
    raise UnsupportedPlatform(f"Operating system {value!r} is not supported.")


def default_registry(settings: ComputerControlSettings) -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(MacOSProvider(settings))
    registry.register(WindowsProvider(settings))
    registry.register(LinuxProvider(settings))
    return registry


def select_provider(settings: ComputerControlSettings, registry: ProviderRegistry | None = None):
    selected = detect_platform() if settings.provider == "auto" else detect_platform(settings.provider)
    return (registry or default_registry(settings)).get(selected)

