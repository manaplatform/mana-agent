"""Local process provider preserving Mana's existing repository behavior."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import uuid
try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]
from datetime import timedelta
from pathlib import Path

from mana_agent.config.settings import mana_home
from mana_agent.execution.artifacts import collect_local_artifacts, confined_path
from mana_agent.execution.errors import ExecutionTimeoutError, PolicyEnforcementError, SnapshotError
from mana_agent.execution.models import (
    ArtifactRequest,
    ArtifactResult,
    EnforcementStrength,
    ExecutionRequest,
    ExecutionResult,
    NetworkMode,
    ProviderHealth,
    ResourceLimits,
    SandboxCapabilities,
    SandboxHandle,
    SandboxSpec,
    SandboxState,
    SnapshotRef,
    SnapshotRequest,
    utc_now,
)
from mana_agent.execution.providers.base import ProviderBase
from mana_agent.execution.secrets import EnvironmentSecretResolver, SecretResolver, redact_values
from mana_agent.execution.snapshots import create_archive_snapshot, restore_archive_snapshot


def _decode_captured_output(value: bytes) -> str:
    """Decode subprocess output using the provider's platform-independent text contract."""
    return value.decode(errors="replace").replace("\r\n", "\n")


class LocalProcessProvider(ProviderBase):
    name = "local-process"

    def __init__(self, *, secret_resolver: SecretResolver | None = None, snapshot_root: Path | None = None) -> None:
        self.secret_resolver = secret_resolver or EnvironmentSecretResolver()
        self.snapshot_root = (snapshot_root or mana_home() / "execution" / "snapshots").resolve()
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self._processes: dict[str, asyncio.subprocess.Process] = {}

    async def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            snapshots=True, native_suspend_resume=os.name == "posix",
            resource_isolation=EnforcementStrength.BEST_EFFORT,
            network_isolation=EnforcementStrength.NONE,
            secret_files=True, secret_environment_variables=True,
            persistent_volumes=True, artifact_streaming=True,
        )

    async def provision(self, spec: SandboxSpec) -> SandboxHandle:
        if spec.network.required_enforcement != EnforcementStrength.NONE or spec.network.mode != NetworkMode.UNRESTRICTED:
            raise PolicyEnforcementError("local-process cannot enforce the requested network policy", provider=self.name)
        if spec.resources.gpu_count:
            raise PolicyEnforcementError("local-process cannot guarantee GPU allocation", provider=self.name)
        now = utc_now()
        return SandboxHandle(
            sandbox_id=f"sbx_{uuid.uuid4().hex}", provider=self.name,
            external_id=f"local:{os.getpid()}", state=SandboxState.PROVISIONING,
            task_id=spec.task_id, session_id=spec.session_id, workspace_id=spec.workspace_id,
            workspace_path=str(spec.repository_source), lease_expires_at=now + timedelta(seconds=spec.max_lifetime_seconds),
            metadata={"resources": spec.resources.model_dump(mode="json")},
        )

    async def start(self, handle: SandboxHandle) -> SandboxHandle:
        handle.state = SandboxState.READY
        return handle

    async def execute(self, handle: SandboxHandle, request: ExecutionRequest) -> ExecutionResult:
        root = Path(handle.workspace_path).resolve()
        cwd = confined_path(root, request.cwd, require_exists=True)
        if not cwd.is_dir():
            raise ValueError("execution cwd must be a directory")
        env = os.environ.copy()
        env.update(request.environment)
        secret_values: list[str] = []
        for item in handle.metadata.get("secrets", []):
            reference, target, mode = item["reference"], item["target"], item["mode"]
            value = self.secret_resolver.resolve(reference)
            secret_values.append(value)
            if mode == "environment":
                env[target] = value
        started = utc_now()
        kwargs: dict[str, object] = {}
        if os.name == "posix" and resource is not None:
            kwargs["start_new_session"] = True
            limits = ResourceLimits.model_validate(handle.metadata.get("resources") or {})
            def apply_limits() -> None:
                if limits.memory_bytes and hasattr(resource, "RLIMIT_AS"):
                    resource.setrlimit(resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes))
                if limits.disk_bytes and hasattr(resource, "RLIMIT_FSIZE"):
                    resource.setrlimit(resource.RLIMIT_FSIZE, (limits.disk_bytes, limits.disk_bytes))
                if limits.pid_limit and hasattr(resource, "RLIMIT_NPROC"):
                    resource.setrlimit(resource.RLIMIT_NPROC, (limits.pid_limit, limits.pid_limit))
            kwargs["preexec_fn"] = apply_limits
        process = await asyncio.create_subprocess_exec(
            *request.argv, cwd=str(cwd), env=env,
            stdin=asyncio.subprocess.PIPE if request.stdin is not None else None,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs,
        )
        self._processes[handle.sandbox_id] = process
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                process.communicate(request.stdin.encode() if request.stdin is not None else None),
                timeout=request.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            await self._kill_group(process)
            raise ExecutionTimeoutError(f"command exceeded {request.timeout_seconds}s", provider=self.name) from exc
        except asyncio.CancelledError:
            await self._kill_group(process)
            raise
        finally:
            self._processes.pop(handle.sandbox_id, None)
        limit = request.capture_limit_bytes
        stdout = redact_values(_decode_captured_output(stdout_b[:limit]), secret_values)
        stderr = redact_values(_decode_captured_output(stderr_b[:limit]), secret_values)
        return ExecutionResult(
            exit_code=int(process.returncode or 0), stdout=stdout, stderr=stderr,
            started_at=started, completed_at=utc_now(), provider=self.name, sandbox_id=handle.sandbox_id,
        )

    async def _kill_group(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        await process.wait()

    async def upload(self, handle: SandboxHandle, source: Path, destination: str) -> None:
        target = confined_path(Path(handle.workspace_path), destination, require_exists=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target, follow_symlinks=False)

    async def download_artifacts(self, handle: SandboxHandle, request: ArtifactRequest) -> list[ArtifactResult]:
        return collect_local_artifacts(Path(handle.workspace_path), request)

    async def snapshot(self, handle: SandboxHandle, request: SnapshotRequest) -> SnapshotRef:
        return create_archive_snapshot(Path(handle.workspace_path), request, self.snapshot_root, self.name)

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec) -> SandboxHandle:
        if snapshot.provider != self.name:
            raise SnapshotError("snapshot provider is incompatible with local-process")
        restore_archive_snapshot(snapshot, spec.repository_source)
        handle = await self.provision(spec)
        handle.snapshot_refs.append(snapshot.snapshot_id)
        return await self.start(handle)

    async def suspend(self, handle: SandboxHandle) -> SandboxHandle:
        process = self._processes.get(handle.sandbox_id)
        if process is not None and os.name == "posix":
            os.killpg(process.pid, signal.SIGSTOP)
        handle.state = SandboxState.SUSPENDED
        return handle

    async def resume(self, handle: SandboxHandle) -> SandboxHandle:
        process = self._processes.get(handle.sandbox_id)
        if process is not None and os.name == "posix":
            os.killpg(process.pid, signal.SIGCONT)
        handle.state = SandboxState.READY
        return handle

    async def terminate(self, handle: SandboxHandle) -> None:
        process = self._processes.get(handle.sandbox_id)
        if process is not None:
            await self._kill_group(process)

    async def cleanup(self, handle: SandboxHandle) -> None:
        await self.terminate(handle)

    async def healthcheck(self) -> ProviderHealth:
        return ProviderHealth(provider=self.name, available=True, message="host process execution available; network isolation unavailable")
