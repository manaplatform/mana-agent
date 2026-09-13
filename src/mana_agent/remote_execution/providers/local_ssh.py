"""Local SSH provider. Keys are referenced by path and never read by Mana."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import os
import shlex
import stat
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from mana_agent.remote_execution.models import RemoteExecutionEvent, RemoteExecutionRequest


@contextmanager
def _askpass_environment(password: str):
    temp_dir = tempfile.mkdtemp(prefix="mana_ssh_")
    dir_path = Path(temp_dir)
    try:
        dir_path.chmod(stat.S_IRWXU)
        secret_file = dir_path / "secret.tok"
        secret_file.write_text(password, encoding="utf-8")
        secret_file.chmod(stat.S_IRUSR | stat.S_IWUSR)

        if sys.platform == "win32":
            helper = dir_path / "askpass.bat"
            helper.write_text(f"@type \"{secret_file}\"\r\n", encoding="utf-8")
        else:
            helper = dir_path / "askpass.sh"
            helper.write_text(f"#!/bin/sh\nexec cat \"{secret_file}\"\n", encoding="utf-8")
            helper.chmod(stat.S_IRWXU)

        env = os.environ.copy()
        env["SSH_ASKPASS"] = str(helper)
        env["SSH_ASKPASS_REQUIRE"] = "force"
        env["DISPLAY"] = env.get("DISPLAY", ":0")
        yield env
    finally:
        try:
            for p in dir_path.iterdir():
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            dir_path.rmdir()
        except OSError:
            pass


def build_ssh_argv(
    request: RemoteExecutionRequest,
    *,
    ssh_binary: str = "ssh",
    connect_timeout_seconds: int | None = None,
    known_hosts_file: str | None = None,
) -> list[str]:
    """Return an argv-only OpenSSH invocation with strict host-key checking."""
    target = request.target
    effective_timeout = connect_timeout_seconds or request.connect_timeout_seconds
    effective_known_hosts = known_hosts_file or request.known_hosts_file
    args = [ssh_binary, "-p", str(target.port)]
    if request.authentication.mode == "password":
        args.extend([
            "-o", "BatchMode=no",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"ConnectTimeout={effective_timeout}",
            "-o", "PreferredAuthentications=password,keyboard-interactive",
            "-o", "PubkeyAuthentication=no",
            "-o", "NumberOfPasswordPrompts=1",
        ])
    else:
        args.extend([
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", f"ConnectTimeout={effective_timeout}",
        ])
    if effective_known_hosts:
        args.extend(["-o", f"UserKnownHostsFile={str(Path(effective_known_hosts).expanduser())}"])
    if request.keepalive_seconds:
        args.extend(["-o", f"ServerAliveInterval={request.keepalive_seconds}", "-o", "ServerAliveCountMax=3"])
    if request.control_path:
        args.extend(["-o", "ControlMaster=auto", "-o", "ControlPersist=60", "-o", f"ControlPath={request.control_path}"])
    if request.jump_host:
        args.extend(["-J", request.jump_host])
    if request.agent_forwarding:
        args.append("-A")
    if request.authentication.mode == "key_path":
        # Expansion happens only in the worker process; this does not read the key.
        identity_path = Path(request.authentication.key_path or "").expanduser()
        if request.provider == "remote-ssh" and not identity_path.is_file():
            raise ValueError("Direct SSH identity path does not exist or is not a regular file.")
        args.extend(["-i", str(identity_path)])
    if request.pty:
        args.append("-tt")
    command = shlex.join(request.command.argv)
    if request.environment:
        exports = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in request.environment.items()
        )
        command = f"env {exports} {command}"
    if request.working_directory:
        command = f"cd -- {shlex.quote(request.working_directory)} && exec {command}"
    return [*args, f"{target.user}@{target.host}", "--", command]


class LocalSSHProvider:
    name = "local_ssh"

    async def execute(self, request: RemoteExecutionRequest, emit: Callable[[RemoteExecutionEvent], None], cancel: asyncio.Event) -> tuple[int, str, str]:
        argv = build_ssh_argv(
            request,
            connect_timeout_seconds=request.connect_timeout_seconds,
            known_hosts_file=request.known_hosts_file,
        )
        emit(RemoteExecutionEvent(job_id=request.job_id, session_id=request.session_id, kind="resolving_host"))
        emit(RemoteExecutionEvent(job_id=request.job_id, session_id=request.session_id, kind="connection_started"))
        emit(RemoteExecutionEvent(job_id=request.job_id, session_id=request.session_id, kind="authenticating"))
        emit(RemoteExecutionEvent(job_id=request.job_id, session_id=request.session_id, kind="command_started"))

        if request.authentication.mode == "password":
            from mana_agent.config.user_config import get_ssh_password

            password = get_ssh_password(
                request.authentication.password_ref,
                profile_name=request.target.host,
            )
            if not password:
                raise RuntimeError(
                    f"Missing password credential for SSH target {request.target.user}@{request.target.host}. "
                    "Please configure a password in ~/.mana/secrets.toml or provide a valid --password-ref."
                )
            with _askpass_environment(password) as env:
                return await self._run_process(argv, request, emit, cancel, env=env)
        else:
            return await self._run_process(argv, request, emit, cancel, env=None)

    async def _run_process(
        self,
        argv: list[str],
        request: RemoteExecutionRequest,
        emit: Callable[[RemoteExecutionEvent], None],
        cancel: asyncio.Event,
        *,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            env=env,
        )
        async def read(stream, kind: str) -> str:
            chunks: list[bytes] = []
            while line := await stream.readline():
                chunks.append(line)
                emit(RemoteExecutionEvent(job_id=request.job_id, session_id=request.session_id, kind=kind, data={"chunk": line.decode(errors="replace")}))
            return b"".join(chunks).decode(errors="replace")
        out_task, err_task = asyncio.create_task(read(process.stdout, "stdout")), asyncio.create_task(read(process.stderr, "stderr"))
        try:
            done, _ = await asyncio.wait({asyncio.create_task(process.wait()), asyncio.create_task(cancel.wait())}, timeout=request.timeout_seconds, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                process.terminate()
                await process.wait()
                raise TimeoutError("SSH command timed out")
            if cancel.is_set():
                process.terminate()
                await process.wait()
                raise asyncio.CancelledError
            await process.wait()
            return process.returncode or 0, await out_task, await err_task
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            await asyncio.gather(out_task, err_task, return_exceptions=True)
            emit(RemoteExecutionEvent(job_id=request.job_id, session_id=request.session_id, kind="connection_closed"))
