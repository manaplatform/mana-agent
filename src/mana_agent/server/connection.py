"""Conversion from enrolled servers to the existing strict OpenSSH transport."""

from __future__ import annotations

from pathlib import Path
import hashlib

from mana_agent.config.settings import mana_home

from mana_agent.remote_execution.models import (
    RemoteCommand,
    RemoteExecutionRequest,
    SSHAuthentication,
    SSHTarget,
)

from .credentials import ServerCredentialResolver
from .models import ServerDefinition


class ServerConnectionFactory:
    def __init__(self, credentials: ServerCredentialResolver | None = None) -> None:
        self.credentials = credentials or ServerCredentialResolver()

    def request(
        self,
        server: ServerDefinition,
        *,
        command_id: str,
        session_id: str,
        argv: list[str],
        cwd: str | None = None,
        timeout_seconds: int = 60,
        pty: bool = False,
        read_only: bool = True,
        environment: dict[str, str] | None = None,
    ) -> RemoteExecutionRequest:
        known_hosts = Path(server.known_hosts_file).expanduser()
        if not server.host_key_fingerprint:
            raise ValueError("The enrolled server has no pinned host-key fingerprint.")
        if not known_hosts.is_file():
            raise ValueError("The enrolled server's pinned known-hosts file does not exist.")
        authentication: SSHAuthentication
        if server.auth_method == "ssh_agent":
            authentication = SSHAuthentication(mode="agent")
        elif server.auth_method == "ssh_key":
            authentication = SSHAuthentication(
                mode="key_path",
                key_path=self.credentials.resolve_key_path(server.credential_ref or ""),
            )
        else:
            self.credentials.require_external_secret(server.credential_ref or "")
            raise AssertionError("credential resolver must stop unsupported authentication")
        sockets = mana_home() / "servers" / "ssh-control"
        sockets.mkdir(mode=0o700, parents=True, exist_ok=True)
        socket_name = hashlib.sha256(f"{server.username}@{server.host}:{server.port}".encode()).hexdigest()[:20]
        return RemoteExecutionRequest(
            job_id=command_id,
            session_id=session_id,
            provider="remote-ssh",
            target=SSHTarget(host=server.host, port=server.port, user=server.username),
            authentication=authentication,
            command=RemoteCommand(argv=argv),
            working_directory=cwd,
            connect_timeout_seconds=server.connect_timeout_seconds,
            known_hosts_file=str(known_hosts),
            jump_host=server.jump_host,
            agent_forwarding=server.agent_forwarding,
            keepalive_seconds=server.keepalive_seconds,
            control_path=str(sockets / socket_name),
            environment=environment or {},
            timeout_seconds=timeout_seconds,
            read_only=read_only,
            pty=pty,
        )
