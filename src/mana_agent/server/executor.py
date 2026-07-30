"""Fail-closed execution of validated, model-selected server actions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
import uuid
from collections.abc import Callable

from mana_agent.remote_execution.models import RemoteExecutionEvent
from mana_agent.remote_execution.providers.local_ssh import LocalSSHProvider
from mana_agent.utils.redaction import redact_secrets

from .audit import ServerAuditLog
from .connection import ServerConnectionFactory
from .locks import ServerLockManager
from .models import (
    DESTRUCTIVE_ACTIONS,
    RemoteCommandResult,
    ServerActionDecision,
    ServerApproval,
    ServerDefinition,
    utc_now,
)
from .registry import ServerRegistry


class ServerDecisionError(RuntimeError):
    pass


class ServerApprovalRequired(RuntimeError):
    def __init__(self, message: str, *, exact_action_key: str) -> None:
        super().__init__(message)
        self.exact_action_key = exact_action_key


def action_key(decision: ServerActionDecision) -> str:
    payload = decision.model_dump(mode="json", exclude={"reason"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ServerExecutor:
    def __init__(
        self,
        *,
        registry: ServerRegistry | None = None,
        connection_factory: ServerConnectionFactory | None = None,
        audit: ServerAuditLog | None = None,
        locks: ServerLockManager | None = None,
        event_sink: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.registry = registry or ServerRegistry()
        self.connection_factory = connection_factory or ServerConnectionFactory()
        self.audit = audit or ServerAuditLog()
        self.locks = locks or ServerLockManager()
        self.event_sink = event_sink
        self._cancel_events: dict[str, asyncio.Event] = {}

    def validate_decision(
        self,
        decision: ServerActionDecision,
        approval: ServerApproval | None = None,
    ) -> tuple[ServerDefinition, str]:
        from .tools import validate_tool_decision

        validate_tool_decision(decision)
        server = self.registry.get(decision.server_id)
        if decision.required_capability not in server.allowed_capabilities:
            raise ServerDecisionError(
                f"Server {server.server_id!r} does not authorize capability {decision.required_capability!r}. "
                "No server action was executed. Grant it explicitly with "
                f"`mana-agent server authorize {server.server_id} "
                f"--capability {decision.required_capability}`."
            )
        if not decision.safe_to_continue:
            raise ServerDecisionError(
                f"Model decision {decision.decision_id!r} is not safe to continue. No server action was executed."
            )
        if server.mode == "inspect_only" and not decision.read_only:
            raise ServerDecisionError("Server is inspect_only. No mutating server action was executed.")
        if decision.action.value == "shell" and server.mode != "trusted_admin":
            raise ServerDecisionError("Advanced shell execution requires trusted_admin mode.")
        if decision.action in DESTRUCTIVE_ACTIONS and not decision.destructive:
            raise ServerDecisionError("Destructive action classification is missing.")
        exact_key = action_key(decision)
        if decision.consequential:
            if approval is None or approval.decision_id != decision.decision_id or approval.server_id != server.server_id or approval.exact_action_key != exact_key:
                raise ServerApprovalRequired(
                    f"Approval required for {decision.action.value} on {server.name} ({server.server_id}). "
                    f"Affected resources: {', '.join(decision.affected_resources)}.",
                    exact_action_key=exact_key,
                )
        return server, exact_key

    async def execute_argv(
        self,
        decision: ServerActionDecision,
        argv: list[str],
        *,
        approval: ServerApproval | None = None,
        session_id: str = "server",
        cwd: str | None = None,
        timeout_seconds: int = 60,
        pty: bool = False,
        environment: dict[str, str] | None = None,
    ) -> RemoteCommandResult:
        server, exact_key = self.validate_decision(decision, approval)
        command_id = f"server_{uuid.uuid4().hex}"
        request = self.connection_factory.request(
            server,
            command_id=command_id,
            session_id=session_id,
            argv=argv,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            pty=pty,
            read_only=decision.read_only,
            environment=environment,
        )
        started = utc_now()
        audit_argv = list(argv)
        if decision.tool_name in {"server_file_write", "server_file_patch"} and audit_argv:
            audit_argv[-1] = "<redacted-file-content>"
        visible_command = shlex.join(audit_argv)
        stdout: list[str] = []
        stderr: list[str] = []
        cancel = asyncio.Event()
        self._cancel_events[command_id] = cancel

        def emit(event: RemoteExecutionEvent) -> None:
            chunk = str(event.data.get("chunk") or "")
            if event.kind == "stdout":
                stdout.append(chunk)
            elif event.kind == "stderr":
                stderr.append(chunk)
            if self.event_sink:
                self.event_sink({
                    "type": "server.command",
                    "status": event.kind,
                    "server_id": server.server_id,
                    "command_id": command_id,
                    "data": redact_secrets(event.data),
                })

        timed_out = False
        cancelled = False
        exit_code: int | None = None
        self.audit.append({
            "event": "executing",
            "server_id": server.server_id,
            "resolved_target": f"{server.username}@{server.host}:{server.port}",
            "decision_id": decision.decision_id,
            "selected_route": decision.action.value,
            "tool_name": decision.tool_name,
            "command": visible_command,
            "approval_id": approval.approval_id if approval else None,
            "exact_action_key": exact_key,
        })
        try:
            async with self.locks.acquire(
                server.server_id,
                mutation=not decision.read_only,
                concurrency_limit=server.max_concurrent_operations,
            ):
                exit_code, _out, _err = await LocalSSHProvider().execute(request, emit, cancel)
        except TimeoutError:
            timed_out = True
        except asyncio.CancelledError:
            cancelled = True
        finally:
            self._cancel_events.pop(command_id, None)
        result = RemoteCommandResult(
            server_id=server.server_id,
            command_id=command_id,
            command=visible_command,
            cwd=cwd,
            exit_code=exit_code,
            stdout=str(redact_secrets("".join(stdout))),
            stderr=str(redact_secrets("".join(stderr))),
            started_at=started,
            completed_at=utc_now(),
            timed_out=timed_out,
            cancelled=cancelled,
            changed_system=not decision.read_only and exit_code == 0,
        )
        self.audit.append({
            "event": "completed" if exit_code == 0 else "failed",
            "server_id": server.server_id,
            "decision_id": decision.decision_id,
            "command_id": command_id,
            "exit_code": exit_code,
            "changed_resources": decision.affected_resources if result.changed_system else [],
            "verification_evidence": [shlex.join(item) for item in decision.verification_commands],
            "timed_out": timed_out,
            "cancelled": cancelled,
        })
        return result

    def cancel(self, command_id: str) -> None:
        try:
            self._cancel_events[command_id].set()
        except KeyError as exc:
            raise LookupError(f"Server command {command_id!r} is not running.") from exc
