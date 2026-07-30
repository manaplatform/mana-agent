"""Facade reused by CLI, gateway, automations, and protocol surfaces."""

from __future__ import annotations

from .audit import ServerAuditLog
from .executor import ServerExecutor
from .models import ServerActionDecision, ServerApproval
from .monitoring import ServerMonitor
from .registry import ServerRegistry


class ServerManagementService:
    def __init__(self, *, registry: ServerRegistry | None = None, executor: ServerExecutor | None = None) -> None:
        self.registry = registry or ServerRegistry()
        self.executor = executor or ServerExecutor(registry=self.registry)
        self.monitor = ServerMonitor(self.executor)
        self.audit = self.executor.audit

    def list_servers(self):
        return self.registry.list()

    def server(self, identity: str):
        return self.registry.get(identity)

    def remove_server(self, identity: str):
        return self.registry.remove(identity)

    def logs(self, identity: str, *, limit: int = 100):
        server = self.registry.get(identity)
        return self.audit.read(server_id=server.server_id, limit=limit)

    async def inspect(self, decision: ServerActionDecision, *, session_id: str = "server"):
        return await self.monitor.inspect(decision, session_id=session_id)

    async def execute(self, decision: ServerActionDecision, argv: list[str], *, approval: ServerApproval | None = None, session_id: str = "server", cwd: str | None = None, timeout_seconds: int = 60, pty: bool = False, environment: dict[str, str] | None = None):
        return await self.executor.execute_argv(
            decision,
            argv,
            approval=approval,
            session_id=session_id,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            pty=pty,
            environment=environment,
        )
