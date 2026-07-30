"""Interactive zsh-compatible sessions through the system OpenSSH client."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from mana_agent.remote_execution.providers.local_ssh import build_ssh_argv

from .connection import ServerConnectionFactory
from .executor import ServerExecutor
from .models import ServerActionDecision, ServerApproval


@dataclass
class ShellSession:
    session_id: str
    server_id: str
    process: asyncio.subprocess.Process


class ServerShellSessions:
    def __init__(self, executor: ServerExecutor) -> None:
        self.executor = executor
        self.connections = ServerConnectionFactory()
        self.sessions: dict[str, ShellSession] = {}

    async def start(self, decision: ServerActionDecision, approval: ServerApproval, *, shell: str | None = None) -> ShellSession:
        server, _key = self.executor.validate_decision(decision, approval)
        requested_shell = shell or server.shell
        if requested_shell not in {"zsh", "bash", "sh"}:
            raise ValueError("Interactive shell must be explicitly selected as zsh, bash, or sh.")
        session_id = f"shell_{uuid.uuid4().hex}"
        request = self.connections.request(
            server,
            command_id=session_id,
            session_id=session_id,
            argv=[requested_shell, "-l"],
            pty=True,
            read_only=False,
            timeout_seconds=3600,
        )
        argv = build_ssh_argv(request, connect_timeout_seconds=server.connect_timeout_seconds, known_hosts_file=server.known_hosts_file)
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        session = ShellSession(session_id=session_id, server_id=server.server_id, process=process)
        self.sessions[session_id] = session
        return session

    async def write(self, session_id: str, data: str) -> None:
        session = self._session(session_id)
        if session.process.stdin is None:
            raise RuntimeError("Shell session stdin is unavailable.")
        session.process.stdin.write(data.encode())
        await session.process.stdin.drain()

    def resize(self, session_id: str, columns: int, rows: int) -> None:
        self._session(session_id)
        if columns < 1 or rows < 1:
            raise ValueError("Terminal dimensions must be positive.")
        raise NotImplementedError("PTY resizing requires a native terminal provider; no resize fallback was attempted.")

    async def stop(self, session_id: str) -> None:
        session = self._session(session_id)
        if session.process.returncode is None:
            session.process.terminate()
            await session.process.wait()
        del self.sessions[session_id]

    def _session(self, session_id: str) -> ShellSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise LookupError(f"Shell session {session_id!r} is not active.") from exc
