"""Provider and application-adapter registries."""

from __future__ import annotations

from mana_agent.integrations.computer_control.contracts import ApplicationAdapter, ComputerControlProvider
from mana_agent.integrations.computer_control.errors import UnsupportedPlatform
from mana_agent.integrations.computer_control.models import ComputerAction, SupportedPlatform


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: dict[SupportedPlatform, ComputerControlProvider] = {}

    def register(self, provider: ComputerControlProvider) -> None:
        self._providers[provider.platform] = provider

    def get(self, platform: SupportedPlatform) -> ComputerControlProvider:
        try:
            return self._providers[platform]
        except KeyError as exc:
            raise UnsupportedPlatform(f"No provider is registered for {platform.value}.") from exc


class ApplicationAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ApplicationAdapter] = {}

    def register(self, adapter: ApplicationAdapter) -> None:
        if adapter.application_id in self._adapters:
            raise ValueError(f"Adapter {adapter.application_id!r} is already registered")
        self._adapters[adapter.application_id] = adapter

    async def select(
        self,
        action: ComputerAction,
        *,
        platform: SupportedPlatform,
        preference: str | None = None,
        active_application: str | None = None,
        configured_default: str | None = None,
    ) -> ApplicationAdapter | None:
        ordered_ids = [
            preference, active_application, configured_default, action.target.application_id,
            *sorted(self._adapters),
        ]
        seen: set[str] = set()
        for application_id in ordered_ids:
            if not application_id or application_id == "auto" or application_id in seen:
                continue
            seen.add(application_id)
            adapter = self._adapters.get(application_id)
            if (
                adapter is not None
                and platform in adapter.supported_platforms
                and action.capability in adapter.capabilities
                and await adapter.is_available()
            ):
                return adapter
        return None
