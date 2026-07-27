"""Docker sandbox provider using safely constructed argv commands."""

from __future__ import annotations

import asyncio
import shutil
import os
import tempfile
import uuid
from pathlib import Path

from mana_agent.execution.errors import ExecutionFailedError, ProviderConfigurationError
from mana_agent.execution.models import (
    ArtifactRequest,
    ArtifactResult,
    EnforcementStrength,
    ExecutionRequest,
    ExecutionResult,
    NetworkMode,
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
from mana_agent.execution.secrets import EnvironmentSecretResolver, SecretResolver


class LocalDockerProvider(ProviderBase):
    name = "local-docker"

    def __init__(self, *, default_image: str = "python:3.12", docker_binary: str = "docker", secret_resolver: SecretResolver | None = None) -> None:
        self.default_image = default_image
        self.docker_binary = docker_binary
        self.secret_resolver = secret_resolver or EnvironmentSecretResolver()

    async def _run(self, *args: str, timeout: int = 120, input_data: bytes | None = None) -> tuple[int, bytes, bytes]:
        try:
            proc = await asyncio.create_subprocess_exec(
                self.docker_binary, *args,
                stdin=asyncio.subprocess.PIPE if input_data is not None else None,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise ProviderConfigurationError("Docker CLI is not installed", provider=self.name) from exc
        try:
            out, err = await asyncio.wait_for(proc.communicate(input_data), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise
        return int(proc.returncode or 0), out, err

    async def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            snapshots=True, native_suspend_resume=True,
            resource_isolation=EnforcementStrength.ENFORCED,
            gpu_execution=True, network_isolation=EnforcementStrength.ENFORCED,
            secret_files=True, secret_environment_variables=True,
            persistent_volumes=True, artifact_streaming=True, parallel_execution=True,
        )

    async def provision(self, spec: SandboxSpec) -> SandboxHandle:
        if spec.network.mode in {NetworkMode.RESTRICTED_EGRESS, NetworkMode.ALLOWLIST}:
            raise ProviderConfigurationError(
                "Docker restricted-egress/allowlist requires a configured managed network; refusing to weaken policy",
                provider=self.name,
            )
        if spec.resources.disk_bytes:
            raise ProviderConfigurationError(
                "Docker disk limits require a configured storage driver and are not portable",
                provider=self.name,
            )
        name = f"mana-{uuid.uuid4().hex[:20]}"
        image = spec.base_image or self.default_image
        args = ["create", "--name", name, "--label", "mana.managed=true"]
        for key, value in {"task": spec.task_id, "session": spec.session_id, "workspace": spec.workspace_id, **spec.labels}.items():
            if value:
                args += ["--label", f"mana.{key}={value}"]
        args += ["--mount", f"type=bind,src={spec.repository_source},dst={spec.workspace_mount_path}"]
        if spec.read_only_root:
            args.append("--read-only")
        resources = spec.resources
        if resources.cpu_cores:
            args += ["--cpus", str(resources.cpu_cores)]
        if resources.memory_bytes:
            args += ["--memory", str(resources.memory_bytes)]
        if resources.pid_limit:
            args += ["--pids-limit", str(resources.pid_limit)]
        if resources.gpu_count:
            args += ["--gpus", str(resources.gpu_count)]
        network_modes = {
            NetworkMode.DENY_ALL: "none", NetworkMode.UNRESTRICTED: "bridge",
            NetworkMode.RESTRICTED_EGRESS: "none", NetworkMode.ALLOWLIST: "none",
        }
        args += ["--network", network_modes[spec.network.mode]]
        env_file = ""
        secret_environment: dict[str, str] = {}
        for secret in spec.secrets:
            if secret.mode != "environment":
                raise ProviderConfigurationError("Docker currently supports secret environment injection only", provider=self.name)
            secret_environment[secret.target] = self.secret_resolver.resolve(secret.reference)
        if spec.environment or secret_environment:
            fd, env_file = tempfile.mkstemp(prefix="mana-docker-env-")
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                for key, value in {**spec.environment, **secret_environment}.items():
                    stream.write(f"{key}={value}\n")
            args += ["--env-file", env_file]
        args += [image, "sleep", "infinity"]
        try:
            code, out, err = await self._run(*args, timeout=spec.execution_timeout_seconds)
        finally:
            if env_file:
                Path(env_file).unlink(missing_ok=True)
        if code:
            raise ExecutionFailedError("Docker sandbox provisioning failed", provider=self.name, diagnostics=err.decode(errors="replace")[-2000:])
        return SandboxHandle(
            sandbox_id=f"sbx_{uuid.uuid4().hex}", provider=self.name,
            external_id=out.decode().strip() or name, state=SandboxState.PROVISIONING,
            task_id=spec.task_id, session_id=spec.session_id, workspace_id=spec.workspace_id,
            workspace_path=spec.workspace_mount_path,
            metadata={"container_name": name, "image": image},
        )

    async def start(self, handle: SandboxHandle) -> SandboxHandle:
        code, _, err = await self._run("start", handle.external_id)
        if code:
            raise ExecutionFailedError("Docker sandbox start failed", provider=self.name, diagnostics=err.decode(errors="replace")[-2000:])
        handle.state = SandboxState.READY
        return handle

    async def execute(self, handle: SandboxHandle, request: ExecutionRequest) -> ExecutionResult:
        started = utc_now()
        args = ["exec", "--workdir", str(Path(handle.workspace_path) / request.cwd)]
        for key, value in request.environment.items():
            args += ["--env", f"{key}={value}"]
        args += [handle.external_id, *request.argv]
        code, out, err = await self._run(*args, timeout=request.timeout_seconds, input_data=request.stdin.encode() if request.stdin is not None else None)
        return ExecutionResult(
            exit_code=code, stdout=out[:request.capture_limit_bytes].decode(errors="replace"),
            stderr=err[:request.capture_limit_bytes].decode(errors="replace"),
            started_at=started, completed_at=utc_now(), provider=self.name, sandbox_id=handle.sandbox_id,
        )

    async def upload(self, handle: SandboxHandle, source: Path, destination: str) -> None:
        code, _, err = await self._run("cp", str(source), f"{handle.external_id}:{destination}")
        if code:
            raise ExecutionFailedError("Docker upload failed", provider=self.name, diagnostics=err.decode(errors="replace")[-2000:])

    async def download_artifacts(self, handle: SandboxHandle, request: ArtifactRequest) -> list[ArtifactResult]:
        from mana_agent.execution.artifacts import collect_local_artifacts
        if request.destination is None:
            raise ValueError("Docker artifact downloads require a destination")
        staging = request.destination / handle.sandbox_id
        staging.mkdir(parents=True, exist_ok=True)
        for relative in request.paths:
            code, _, err = await self._run("cp", f"{handle.external_id}:{Path(handle.workspace_path) / relative}", str(staging / relative))
            if code and not request.missing_ok:
                raise ExecutionFailedError("Docker artifact download failed", provider=self.name, diagnostics=err.decode(errors="replace")[-2000:])
        local_request = request.model_copy(update={"destination": request.destination})
        return collect_local_artifacts(staging, local_request)

    async def snapshot(self, handle: SandboxHandle, request: SnapshotRequest) -> SnapshotRef:
        _ = request
        tag = f"mana-snapshot:{uuid.uuid4().hex}"
        code, out, err = await self._run("commit", handle.external_id, tag, timeout=300)
        if code:
            raise ExecutionFailedError("Docker snapshot failed", provider=self.name, diagnostics=err.decode(errors="replace")[-2000:])
        identity = out.decode().strip()
        checksum = identity.removeprefix("sha256:")
        return SnapshotRef(snapshot_id=f"snap_{uuid.uuid4().hex}", provider=self.name, location=tag, checksum=checksum, image_identity=identity)

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec) -> SandboxHandle:
        restored = spec.model_copy(update={"base_image": snapshot.location})
        return await self.start(await self.provision(restored))

    async def suspend(self, handle: SandboxHandle) -> SandboxHandle:
        code, _, err = await self._run("pause", handle.external_id)
        if code:
            raise ExecutionFailedError("Docker pause failed", provider=self.name, diagnostics=err.decode(errors="replace")[-2000:])
        handle.state = SandboxState.SUSPENDED
        return handle

    async def resume(self, handle: SandboxHandle) -> SandboxHandle:
        code, _, err = await self._run("unpause", handle.external_id)
        if code:
            raise ExecutionFailedError("Docker unpause failed", provider=self.name, diagnostics=err.decode(errors="replace")[-2000:])
        handle.state = SandboxState.READY
        return handle

    async def terminate(self, handle: SandboxHandle) -> None:
        await self._run("stop", "--time", "5", handle.external_id, timeout=15)

    async def cleanup(self, handle: SandboxHandle) -> None:
        await self._run("rm", "--force", "--volumes", handle.external_id, timeout=30)

    async def healthcheck(self) -> ProviderHealth:
        if shutil.which(self.docker_binary) is None:
            return ProviderHealth(provider=self.name, available=False, message="Docker CLI not installed")
        code, out, err = await self._run("version", "--format", "{{json .Server}}", timeout=10)
        message = out.decode(errors="replace")[:500] if code == 0 else err.decode(errors="replace")[-500:]
        return ProviderHealth(provider=self.name, available=code == 0, message=message)
