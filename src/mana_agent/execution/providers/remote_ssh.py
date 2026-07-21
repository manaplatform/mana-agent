"""SSH host-pool sandbox provider with host-key verification enabled."""

from __future__ import annotations

import asyncio
import shlex
import uuid
from pathlib import Path
from typing import Any

from mana_agent.execution.errors import ExecutionFailedError, ProviderConfigurationError
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
    SandboxState,
    SnapshotRef,
    SnapshotRequest,
    utc_now,
)
from mana_agent.execution.providers.base import ProviderBase


class RemoteSSHProvider(ProviderBase):
    name = "remote-ssh"

    def __init__(self, *, hosts: list[dict[str, Any]] | None = None, ssh_binary: str = "ssh", scp_binary: str = "scp") -> None:
        self.hosts = list(hosts or [])
        self.ssh_binary = ssh_binary
        self.scp_binary = scp_binary
        self._next_host = 0

    def _host(self) -> dict[str, Any]:
        if not self.hosts:
            raise ProviderConfigurationError("remote-ssh requires a configured host pool", provider=self.name)
        host = self.hosts[self._next_host % len(self.hosts)]
        self._next_host += 1
        if not host.get("hostname"):
            raise ProviderConfigurationError("SSH host is missing hostname", provider=self.name)
        return host

    def _target(self, host: dict[str, Any]) -> str:
        user = str(host.get("user") or "").strip()
        return f"{user}@{host['hostname']}" if user else str(host["hostname"])

    def _base_args(self, host: dict[str, Any]) -> list[str]:
        args = ["-o", "StrictHostKeyChecking=yes", "-o", f"ConnectTimeout={int(host.get('connect_timeout', 10))}", "-o", "ServerAliveInterval=15"]
        if host.get("known_hosts_file"):
            args += ["-o", f"UserKnownHostsFile={host['known_hosts_file']}"]
        if host.get("identity_file"):
            args += ["-i", str(host["identity_file"])]
        if host.get("port"):
            args += ["-p", str(host["port"])]
        return args

    async def _ssh(self, host: dict[str, Any], argv: list[str], timeout: int = 120) -> tuple[int, bytes, bytes]:
        remote_command = shlex.join(argv)
        proc = await asyncio.create_subprocess_exec(
            self.ssh_binary, *self._base_args(host), self._target(host), remote_command,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return int(proc.returncode or 0), out, err

    async def capabilities(self) -> SandboxCapabilities:
        enforced = all(bool(host.get("resource_wrapper")) for host in self.hosts) if self.hosts else False
        return SandboxCapabilities(
            snapshots=True, native_suspend_resume=True,
            resource_isolation=EnforcementStrength.ENFORCED if enforced else EnforcementStrength.BEST_EFFORT,
            network_isolation=EnforcementStrength.ENFORCED if all(h.get("network_wrapper") for h in self.hosts) and self.hosts else EnforcementStrength.NONE,
            secret_files=True, secret_environment_variables=True, persistent_volumes=True,
            artifact_streaming=True, parallel_execution=True,
        )

    async def provision(self, spec: SandboxSpec) -> SandboxHandle:
        host = self._host()
        remote_root = str(host.get("workspace_root") or "/tmp/mana-sandboxes")
        external = f"mana-{uuid.uuid4().hex}"
        remote_path = f"{remote_root.rstrip('/')}/{external}"
        code, _, err = await self._ssh(host, ["mkdir", "-p", "--", remote_path])
        if code:
            raise ExecutionFailedError("SSH workspace creation failed", provider=self.name, diagnostics=err.decode(errors="replace")[-2000:])
        handle = SandboxHandle(
            sandbox_id=f"sbx_{uuid.uuid4().hex}", provider=self.name, external_id=external,
            state=SandboxState.PROVISIONING, task_id=spec.task_id, session_id=spec.session_id,
            workspace_id=spec.workspace_id, workspace_path=remote_path,
            metadata={"host": {key: value for key, value in host.items() if key not in {"identity_data", "password"}}},
        )
        await self.upload(handle, spec.repository_source, remote_path)
        return handle

    async def start(self, handle: SandboxHandle) -> SandboxHandle:
        handle.state = SandboxState.READY
        return handle

    async def execute(self, handle: SandboxHandle, request: ExecutionRequest) -> ExecutionResult:
        host = handle.metadata["host"]
        cwd = str(Path(handle.workspace_path) / request.cwd)
        command = ["env", *[f"{k}={v}" for k, v in request.environment.items()], *request.argv]
        wrapper = list(host.get("resource_wrapper") or [])
        started = utc_now()
        code, out, err = await self._ssh(host, ["sh", "-c", f"cd {shlex.quote(cwd)} && exec {shlex.join([*wrapper, *command])}"], request.timeout_seconds)
        return ExecutionResult(exit_code=code, stdout=out[:request.capture_limit_bytes].decode(errors="replace"), stderr=err[:request.capture_limit_bytes].decode(errors="replace"), started_at=started, completed_at=utc_now(), provider=self.name, sandbox_id=handle.sandbox_id)

    async def upload(self, handle: SandboxHandle, source: Path, destination: str) -> None:
        host = handle.metadata["host"]
        args = self._base_args(host)
        # scp uses -P instead of ssh's -p.
        args = ["-P" if item == "-p" else item for item in args]
        proc = await asyncio.create_subprocess_exec(self.scp_binary, *args, "-r", str(source) + "/.", f"{self._target(host)}:{destination}", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, err = await proc.communicate()
        if proc.returncode:
            raise ExecutionFailedError("SSH upload failed", provider=self.name, diagnostics=err.decode(errors="replace")[-2000:])

    async def download_artifacts(self, handle: SandboxHandle, request: ArtifactRequest) -> list[ArtifactResult]:
        if request.destination is None:
            raise ValueError("SSH artifact downloads require a destination")
        from mana_agent.execution.artifacts import collect_local_artifacts
        host = handle.metadata["host"]
        staging = request.destination / handle.sandbox_id
        staging.mkdir(parents=True, exist_ok=True)
        args = ["-P" if item == "-p" else item for item in self._base_args(host)]
        for path in request.paths:
            proc = await asyncio.create_subprocess_exec(self.scp_binary, *args, "-r", f"{self._target(host)}:{handle.workspace_path}/{path}", str(staging / path), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            _, err = await proc.communicate()
            if proc.returncode and not request.missing_ok:
                raise ExecutionFailedError("SSH artifact download failed", provider=self.name, diagnostics=err.decode(errors="replace")[-2000:])
        return collect_local_artifacts(staging, request.model_copy(update={"destination": request.destination}))

    async def snapshot(self, handle: SandboxHandle, request: SnapshotRequest) -> SnapshotRef:
        host = handle.metadata["host"]
        snapshot_id = f"snap_{uuid.uuid4().hex}"
        location = f"{handle.workspace_path}/../{snapshot_id}.tar.gz"
        code, out, err = await self._ssh(host, ["sh", "-c", f"tar -C {shlex.quote(handle.workspace_path)} -czf {shlex.quote(location)} . && sha256sum {shlex.quote(location)}"])
        if code:
            raise ExecutionFailedError("SSH snapshot failed", provider=self.name, diagnostics=err.decode(errors="replace")[-2000:])
        return SnapshotRef(snapshot_id=snapshot_id, provider=self.name, location=location, checksum=out.decode().split()[0], metadata={"host": host})

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec) -> SandboxHandle:
        handle = await self.provision(spec)
        host = handle.metadata["host"]
        code, _, err = await self._ssh(host, ["tar", "-C", handle.workspace_path, "-xzf", snapshot.location])
        if code:
            raise ExecutionFailedError("SSH snapshot restore failed", provider=self.name, diagnostics=err.decode(errors="replace")[-2000:])
        return await self.start(handle)

    async def suspend(self, handle: SandboxHandle) -> SandboxHandle:
        host = handle.metadata["host"]
        await self._ssh(host, ["sh", "-c", f"test ! -f {shlex.quote(handle.workspace_path)}/.mana.pid || kill -STOP -$(cat {shlex.quote(handle.workspace_path)}/.mana.pid)"])
        handle.state = SandboxState.SUSPENDED
        return handle

    async def resume(self, handle: SandboxHandle) -> SandboxHandle:
        host = handle.metadata["host"]
        await self._ssh(host, ["sh", "-c", f"test ! -f {shlex.quote(handle.workspace_path)}/.mana.pid || kill -CONT -$(cat {shlex.quote(handle.workspace_path)}/.mana.pid)"])
        handle.state = SandboxState.READY
        return handle

    async def terminate(self, handle: SandboxHandle) -> None:
        host = handle.metadata["host"]
        await self._ssh(host, ["sh", "-c", f"test ! -f {shlex.quote(handle.workspace_path)}/.mana.pid || kill -TERM -$(cat {shlex.quote(handle.workspace_path)}/.mana.pid)"], timeout=15)

    async def cleanup(self, handle: SandboxHandle) -> None:
        host = handle.metadata["host"]
        await self._ssh(host, ["rm", "-rf", "--", handle.workspace_path], timeout=30)

    async def healthcheck(self) -> ProviderHealth:
        if not self.hosts:
            return ProviderHealth(provider=self.name, available=False, message="no SSH hosts configured")
        try:
            code, out, err = await self._ssh(self.hosts[0], ["true"], timeout=10)
            return ProviderHealth(provider=self.name, available=code == 0, message=(out or err).decode(errors="replace")[-500:])
        except Exception as exc:
            return ProviderHealth(provider=self.name, available=False, message=str(exc))
