"""Modal provider using the optional official SDK behind an adapter."""

from __future__ import annotations

import importlib
from typing import Any

from mana_agent.execution.errors import ProviderConfigurationError
from mana_agent.execution.models import EnforcementStrength, SandboxCapabilities
from mana_agent.execution.providers.kubernetes import KubernetesProvider


class ModalProvider(KubernetesProvider):
    name = "modal"

    def __init__(self, *, client: Any = None) -> None:
        super().__init__(namespace="", client=client)

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            importlib.import_module("modal")
        except ImportError as exc:
            raise ProviderConfigurationError("Modal provider requires the optional 'modal' package", provider=self.name) from exc
        raise ProviderConfigurationError("Modal SDK is installed but no configured runtime adapter was supplied", provider=self.name)

    async def capabilities(self) -> SandboxCapabilities:
        if self._client is not None and hasattr(self._client, "capabilities"):
            return SandboxCapabilities.model_validate(await self._call("capabilities"))
        return SandboxCapabilities(snapshots=True, emulated_suspend_resume=True, resource_isolation=EnforcementStrength.ENFORCED, gpu_execution=True, network_isolation=EnforcementStrength.BEST_EFFORT, secret_files=True, secret_environment_variables=True, persistent_volumes=True, artifact_streaming=True, parallel_execution=True)
