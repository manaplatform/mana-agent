"""Validated sandbox provider registry."""

from __future__ import annotations

from mana_agent.execution.errors import ProviderConfigurationError, ProviderUnavailableError
from mana_agent.execution.provider import SandboxProvider


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, SandboxProvider] = {}

    def register(self, provider: SandboxProvider) -> None:
        name = str(getattr(provider, "name", "") or "").strip()
        if not name:
            raise ProviderConfigurationError("provider name is required")
        if name in self._providers:
            raise ProviderConfigurationError(f"provider already registered: {name}")
        self._providers[name] = provider

    def get(self, name: str) -> SandboxProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise ProviderUnavailableError(f"sandbox provider is not registered: {name}", provider=name) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def providers(self) -> tuple[SandboxProvider, ...]:
        return tuple(self._providers[name] for name in sorted(self._providers))
