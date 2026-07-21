"""Kubernetes provider using the optional official client behind an adapter."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from mana_agent.execution.errors import ProviderConfigurationError
from mana_agent.execution.models import (
    ArtifactRequest,
    ArtifactResult,
    EnforcementStrength,
    ExecutionRequest,
    ExecutionResult,
    ProviderHealth,
    SandboxCapabilities,
    SandboxHandle,
    SandboxSpec,
    SnapshotRef,
    SnapshotRequest,
)
from mana_agent.execution.providers.base import ProviderBase


class KubernetesProvider(ProviderBase):
    name = "kubernetes"

    def __init__(self, *, namespace: str = "mana-runtimes", client: Any = None) -> None:
        self.namespace = namespace
        self._client = client

    def _client_or_raise(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            importlib.import_module("kubernetes")
        except ImportError as exc:
            raise ProviderConfigurationError("Kubernetes provider requires the optional 'kubernetes' package", provider=self.name) from exc
        raise ProviderConfigurationError("Kubernetes client is installed but no configured runtime adapter was supplied", provider=self.name)

    async def _call(self, name: str, *args: Any) -> Any:
        client = self._client_or_raise()
        method = getattr(client, name, None)
        if method is None:
            raise ProviderConfigurationError(f"Kubernetes runtime adapter is missing {name}()", provider=self.name)
        result = method(*args)
        return await result if hasattr(result, "__await__") else result

    async def capabilities(self) -> SandboxCapabilities:
        if self._client is not None and hasattr(self._client, "capabilities"):
            return SandboxCapabilities.model_validate(await self._call("capabilities"))
        return SandboxCapabilities(snapshots=True, emulated_suspend_resume=True, resource_isolation=EnforcementStrength.ENFORCED, gpu_execution=True, network_isolation=EnforcementStrength.ENFORCED, secret_files=True, secret_environment_variables=True, persistent_volumes=True, artifact_streaming=True, parallel_execution=True)

    async def provision(self, spec: SandboxSpec) -> SandboxHandle:
        payload = await self._call("provision", spec)
        return SandboxHandle.model_validate(payload)
    async def start(self, handle: SandboxHandle) -> SandboxHandle:
        return SandboxHandle.model_validate(await self._call("start", handle))
    async def execute(self, handle: SandboxHandle, request: ExecutionRequest) -> ExecutionResult:
        return ExecutionResult.model_validate(await self._call("execute", handle, request))
    async def upload(self, handle: SandboxHandle, source: Path, destination: str) -> None:
        await self._call("upload", handle, source, destination)
    async def download_artifacts(self, handle: SandboxHandle, request: ArtifactRequest) -> list[ArtifactResult]:
        return [ArtifactResult.model_validate(item) for item in await self._call("download_artifacts", handle, request)]
    async def snapshot(self, handle: SandboxHandle, request: SnapshotRequest) -> SnapshotRef:
        return SnapshotRef.model_validate(await self._call("snapshot", handle, request))
    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec) -> SandboxHandle:
        return SandboxHandle.model_validate(await self._call("restore", snapshot, spec))
    async def suspend(self, handle: SandboxHandle) -> SandboxHandle:
        return SandboxHandle.model_validate(await self._call("suspend", handle))
    async def resume(self, handle: SandboxHandle) -> SandboxHandle:
        return SandboxHandle.model_validate(await self._call("resume", handle))
    async def terminate(self, handle: SandboxHandle) -> None:
        await self._call("terminate", handle)
    async def cleanup(self, handle: SandboxHandle) -> None:
        await self._call("cleanup", handle)
    async def healthcheck(self) -> ProviderHealth:
        try:
            return ProviderHealth.model_validate(await self._call("healthcheck"))
        except Exception as exc:
            return ProviderHealth(provider=self.name, available=False, message=str(exc))
