"""Evidence-based health collection for enrolled servers."""

from __future__ import annotations

from .executor import ServerExecutor
from .models import ServerActionDecision, ServerHealthReport
from .tools import validate_tool_decision


class ServerMonitor:
    def __init__(self, executor: ServerExecutor) -> None:
        self.executor = executor

    async def inspect(self, decision: ServerActionDecision, *, session_id: str = "server") -> ServerHealthReport:
        validate_tool_decision(decision)
        commands = [
            ["uname", "-srm"],
            ["uptime"],
            ["free", "-h"],
            ["df", "-hP"],
            ["systemctl", "--failed", "--no-pager", "--no-legend"],
            ["ss", "-lntup"],
        ]
        evidence = []
        for argv in commands:
            evidence.append(await self.executor.execute_argv(decision, argv, session_id=session_id))
        uname, uptime, memory, disks, failed, ports = evidence
        return ServerHealthReport(
            server_id=decision.server_id,
            operating_system=uname.stdout.strip(),
            architecture=uname.stdout.strip(),
            load_average=uptime.stdout.strip(),
            memory=memory.stdout,
            disks=disks.stdout,
            failed_services=failed.stdout,
            listening_ports=ports.stdout,
            evidence=evidence,
        )
