"""Reusable provider support with explicit unsupported-operation failures."""

from __future__ import annotations

from pathlib import Path

from mana_agent.execution.errors import CapabilityMismatchError
from mana_agent.execution.models import (
    ArtifactRequest,
    ArtifactResult,
    SandboxHandle,
    SandboxSpec,
    SnapshotRef,
    SnapshotRequest,
)


class ProviderBase:
    name = "base"

    async def upload(self, handle: SandboxHandle, source: Path, destination: str) -> None:
        raise CapabilityMismatchError(f"{self.name} does not support upload", provider=self.name)

    async def download_artifacts(self, handle: SandboxHandle, request: ArtifactRequest) -> list[ArtifactResult]:
        raise CapabilityMismatchError(f"{self.name} does not support artifacts", provider=self.name)

    async def snapshot(self, handle: SandboxHandle, request: SnapshotRequest) -> SnapshotRef:
        raise CapabilityMismatchError(f"{self.name} does not support snapshots", provider=self.name)

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec) -> SandboxHandle:
        raise CapabilityMismatchError(f"{self.name} does not support restore", provider=self.name)

    async def suspend(self, handle: SandboxHandle) -> SandboxHandle:
        raise CapabilityMismatchError(f"{self.name} does not support suspend", provider=self.name)

    async def resume(self, handle: SandboxHandle) -> SandboxHandle:
        raise CapabilityMismatchError(f"{self.name} does not support resume", provider=self.name)
