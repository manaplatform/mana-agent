"""Deterministic execution boundary. No model calls, planning, or hidden retries.

Bindings are host-owned capabilities, never fields supplied by a tool caller.
TaskBoard remains the task record; supervisor action receipts remain recovery
authority; the event hub and context artifacts remain the public trace stores.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import difflib
import copy
import hashlib
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any
from types import MappingProxyType
import uuid

from mana_agent.chat.activity import safe_text
from mana_agent.coding.models import AgentEvent, WorkspaceContext
from mana_agent.multi_agent.core.types import AgentRole, AgentState
from mana_agent.multi_agent.queue.locks import LockTable
from mana_agent.multi_agent.tools.operation_policy import OperationPolicy, validate_arguments
from mana_agent.multi_agent.tools.tool_request import ToolOperationRequest
from mana_agent.multi_agent.tools.tool_result import FileMutationEvidence, ToolOperationResult
from mana_agent.services.activity_context import activity_scope, current_activity_context

logger = logging.getLogger(__name__)
_LOCKS = LockTable()
_ACTIVE: ContextVar["RunningOperation | None"] = ContextVar("mana_running_operation", default=None)
_SEAL = object()


class ExecutionAuthorityError(PermissionError):
    pass


@dataclass(frozen=True)
class ToolExecutionAuthority:
    """Nonserializable host binding; deserializing a request cannot create one."""
    task_id: str
    actor_id: str
    actor_role: AgentRole
    worker_id: str
    workspace: WorkspaceContext
    policies: Mapping[str, OperationPolicy]
    schemas: Mapping[str, dict[str, Any]]
    activity: Any = field(default=None, repr=False)
    cancel: Callable[[], bool] = field(default=lambda: False, repr=False)
    authorize: Callable[[str], Any] | None = field(default=None, repr=False)
    taskboard: Any = field(default=None, repr=False, compare=False)
    registry: Any = field(default=None, repr=False, compare=False)
    externally_locked: bool = field(default=False, repr=False)
    scope_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    seal: object = field(default=None, repr=False)


def bind_tool_authority(*, registry, actor_id: str, task_id: str, workspace: WorkspaceContext,
                        policies: Mapping[str, OperationPolicy], schemas: Mapping[str, dict[str, Any]],
                        cancel: Callable[[], bool] = lambda: False,
                        authorize: Callable[[str], Any] | None = None, taskboard=None,
                        worker_id: str = "", externally_locked: bool = False) -> ToolExecutionAuthority:
    """Host-only binding after lane admission, with exact registered identities."""
    node = registry.agents.get(actor_id)
    if node is None or node.state == AgentState.DONE or node.role in {AgentRole.MAIN, AgentRole.HEAD_DECISION, AgentRole.TOOL_WORKER}:
        raise ExecutionAuthorityError("Tool execution requires an assigned specialist, not MainAgent or a worker decision")
    if not task_id:
        raise ExecutionAuthorityError("Tool execution requires an authoritative task identity")
    if any(not policy.capability or policy.effect not in {"read_only", "mutation", "command", "external"}
           or name not in schemas for name, policy in policies.items()):
        raise ExecutionAuthorityError("Registered tools require explicit effect, capability, and argument schema")
    if node.role != AgentRole.CODING and any(p.repository for p in policies.values()):
        raise ExecutionAuthorityError("Repository operations must be delegated through CodingAgent")
    if any(p.repository and "coding" not in node.capabilities and p.capability not in node.capabilities for p in policies.values()):
        raise ExecutionAuthorityError("Assigned CodingAgent lacks repository execution capability")
    parent = node.parent_agent_id
    visited = {actor_id}
    while parent:
        if parent in visited or parent not in registry.agents:
            raise ExecutionAuthorityError("Invalid or circular delegation lineage")
        visited.add(parent)
        parent_node = registry.agents[parent]
        parent = parent_node.parent_agent_id
    if not any(registry.agents[identity].role == AgentRole.MAIN for identity in visited):
        raise ExecutionAuthorityError("Specialist has no MainAgent supervision lineage")
    if taskboard is not None:
        assigned = taskboard.get_task(task_id)
        if actor_id not in {assigned.owner_agent_id, *assigned.assigned_agent_ids, *assigned.assigned_subagent_ids}:
            raise ExecutionAuthorityError("Specialist is not assigned to this TaskBoard task")
        execution_root = assigned.execution_repo_root or assigned.managed_worktree_path
        if execution_root and Path(execution_root).resolve() != workspace.worktree_path.resolve():
            raise ExecutionAuthorityError("Workspace differs from the TaskBoard execution assignment")
    worker = registry.agents.get(worker_id) if worker_id else next((candidate for candidate in registry.agents.values()
        if candidate.role == AgentRole.TOOL_WORKER and candidate.parent_agent_id == actor_id and candidate.state != AgentState.DONE), None)
    if worker_id and (worker is None or worker.role != AgentRole.TOOL_WORKER or worker.parent_agent_id != actor_id or worker.state == AgentState.DONE):
        raise ExecutionAuthorityError("ToolRunner is not registered under the assigned specialist")
    if worker is None:
        worker = registry.register(AgentRole.TOOL_WORKER, parent_agent_id=actor_id)
    context = current_activity_context()
    if context is not None and context.task_id and context.task_id != task_id:
        raise ExecutionAuthorityError("Tool task does not match the admitted execution")
    return ToolExecutionAuthority(task_id=task_id, actor_id=actor_id, actor_role=node.role, worker_id=worker.agent_id,
        workspace=workspace, policies=MappingProxyType(dict(policies)), schemas=MappingProxyType(dict(schemas)),
        activity=context, cancel=cancel, authorize=authorize, taskboard=taskboard, registry=registry,
        externally_locked=externally_locked, seal=_SEAL)


def current_operation() -> "RunningOperation | None":
    return _ACTIVE.get()


def assert_tool_runner(tool_name: str = "") -> None:
    operation = current_operation()
    if operation is None or tool_name and operation.request.tool_name != tool_name:
        raise ExecutionAuthorityError("Native tool implementations require ToolRunner execution")


def _public_value(value: Any, protected: tuple[str, ...]) -> Any:
    from mana_agent.utils.redaction import redact_secrets
    if isinstance(value, dict):
        return {str(k): _public_value(v, protected) for k, v in redact_secrets(value).items()
                if str(k).lower() not in {"environment", "env", "lease_token", "protected_context", "prompt", "messages"}}
    if isinstance(value, (list, tuple)):
        return [_public_value(item, protected) for item in value]
    return safe_text(value, None, protected=protected) if isinstance(value, str) else value


def _protected_text(authority: ToolExecutionAuthority) -> tuple[str, ...]:
    import os
    from mana_agent.utils.redaction import SENSITIVE_KEYS
    environment = tuple(value for key, value in os.environ.items()
        if value and len(value) >= 8 and (key.lower() in SENSITIVE_KEYS or any(key.lower().endswith("_" + suffix) for suffix in SENSITIVE_KEYS)))
    return (*tuple(getattr(authority.activity, "protected_text", ())), *environment)


def model_result(result: ToolOperationResult, authority: ToolExecutionAuthority) -> dict[str, Any]:
    """Bound/redact model context; full execution evidence stays in scoped artifacts."""
    remaining = 64_000
    truncated = False
    def bounded(value, depth=0):
        nonlocal remaining, truncated
        if depth > 12 or remaining <= 0:
            truncated = True
            return "[retained detail omitted from model context]"
        if isinstance(value, str):
            length = min(len(value), remaining, 16_000)
            remaining -= length
            if length < len(value):
                truncated = True
            return value[:length] + ("\n[open retained details for remaining content]" if length < len(value) else "")
        if isinstance(value, dict):
            truncated = truncated or len(value) > 100
            return {key: bounded(item, depth + 1) for key, item in list(value.items())[:100]}
        if isinstance(value, list):
            truncated = truncated or len(value) > 100
            return [bounded(item, depth + 1) for item in value[:100]]
        return value
    public = bounded(_public_value(result.result, _protected_text(authority)))
    if truncated:
        public["model_context_truncated"] = True
        public["retained_detail_identity"] = {"execution_id": result.execution_id,
            "operation_id": result.operation_id, "operation_attempt_id": result.operation_attempt_id,
            "attempt_id": result.attempt_id}
    return public


class RunningOperation:
    def __init__(self, request: ToolOperationRequest, authority: ToolExecutionAuthority, runner: "ToolRunnerAgent"):
        self.request, self.authority, self.runner = request, authority, runner
        self.sequence = 0
        self.offsets: dict[str, int] = {}
        self.references: dict[str, list[dict[str, Any]]] = {}
        self.retained_bytes = 0
        self.omitted_bytes = 0
        self.trace_complete = True
        self.lock = threading.RLock()
        self.protected = _protected_text(authority)
        def argument_secrets(value):
            from mana_agent.utils.redaction import SENSITIVE_KEYS
            if isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, str) and (str(key).lower() in SENSITIVE_KEYS or any(str(key).lower().endswith("_" + suffix) for suffix in SENSITIVE_KEYS)):
                        yield item
                    else:
                        yield from argument_secrets(item)
            elif isinstance(value, list):
                for item in value:
                    yield from argument_secrets(item)
        self.protected = (*self.protected, *tuple(argument_secrets(request.arguments)))
        self.started = time.monotonic()
        self.started_at = datetime.now(timezone.utc)
        self.public = self._details()

    def _details(self) -> dict[str, Any]:
        from mana_agent.utils.shell_argv import local_shell_argv
        import shlex
        args = self.request.arguments
        action = args.get("action") if isinstance(args.get("action"), dict) else {}
        commands = action.get("commands", args.get("commands", []))
        command = args.get("command") or args.get("script") or args.get("cmd") or (commands[0] if len(commands) == 1 else "")
        argv = args.get("argv")
        if isinstance(argv, list):
            command = shlex.join(str(arg) for arg in argv)
        elif command:
            argv = local_shell_argv(str(command))
        cwd = (Path(self.request.workspace_root) / str(args.get("cwd") or ".")).resolve()
        path = args.get("path") or args.get("file_path") or ""
        return _public_value({"command": str(command), "display_argv": argv or [], "cwd": str(cwd),
            "path": str(path), "display_arguments": {key: value for key, value in args.items() if key in {
                "path", "paths", "file_path", "query", "glob", "regex", "start_line", "end_line", "limit",
                "command", "script", "cwd", "argv", "timeout", "timeout_ms", "revision", "staged"}}, "timeout_seconds": self.request.timeout_seconds,
            "provenance": "native_tool_runner", "streaming": True,
            "delegation_path": [self.request.requested_by_agent_id, self.request.agent_id]}, self.protected)

    def validate_live(self) -> None:
        registry = self.authority.registry
        owner, worker = registry.agents.get(self.authority.actor_id), registry.agents.get(self.authority.worker_id)
        if owner is None or worker is None or owner.role != self.authority.actor_role or worker.role != AgentRole.TOOL_WORKER or owner.state == AgentState.DONE or worker.state == AgentState.DONE or worker.parent_agent_id != owner.agent_id:
            raise ExecutionAuthorityError("Execution owner or ToolRunner registration was revoked")
        if self.authority.cancel():
            raise InterruptedError("Execution cancelled by its owner")
        if self.authority.taskboard is not None:
            assigned = self.authority.taskboard.get_task(self.request.task_id)
            if self.authority.actor_id not in {assigned.owner_agent_id, *assigned.assigned_agent_ids, *assigned.assigned_subagent_ids}:
                raise ExecutionAuthorityError("Execution task assignment was revoked")
            root = assigned.execution_repo_root or assigned.managed_worktree_path
            if root and Path(root).resolve() != Path(self.request.workspace_root):
                raise ExecutionAuthorityError("Execution workspace assignment changed")
        if self.request.deadline and datetime.now(timezone.utc) >= self.request.deadline:
            raise TimeoutError("Execution deadline expired")
        context = self.authority.activity
        if context is not None:
            with context.publication():
                pass

    @contextmanager
    def scope(self) -> Iterator[None]:
        token = _ACTIVE.set(self)
        context = self.authority.activity
        scoped = replace(context, agent_id=self.request.agent_id, parent_agent_id=self.request.requested_by_agent_id) if context else None
        try:
            with activity_scope(scoped) if scoped else nullcontext():
                yield
        finally:
            _ACTIVE.reset(token)

    def emit(self, event_type: str, *, status: str = "running", **details: Any) -> None:
        from mana_agent.coding.live_events import publish_coding_event
        details = _public_value(details, self.protected)
        with self.lock:
            self.sequence += 1
            metadata = {**self.public, **details, "parent_execution_id": self.request.parent_execution_id,
                "workspace_id": self.runner.workspace_id,
                "execution_level": 3, "agent_role": "tool_worker", "operation_attempt_id": self.request.operation_attempt_id,
                "operation_sequence": self.sequence, "tool_call_id": self.request.tool_call_id,
                "trace_complete": self.trace_complete, "retained_bytes": self.retained_bytes,
                "omitted_bytes": self.omitted_bytes}
            event = AgentEvent(event_type=event_type, task_id=self.request.task_id,
                backend="internal" if self.authority.actor_role == AgentRole.CODING else "tools",
                operation_id=self.request.operation_id, operation_attempt_id=self.request.operation_attempt_id,
                agent_id=self.request.agent_id, parent_agent_id=self.request.requested_by_agent_id,
                root_task_id=self.request.root_task_id, parent_task_id=self.request.parent_task_id,
                attempt_id=self.request.attempt_id, attempt_generation=self.request.attempt_generation,
                producer_sequence=self.sequence, status=status,
                title=str(self.public.get("command") or self.public.get("path") or self.request.tool_name),
                tool_name=self.request.tool_name, command=str(self.public.get("command") or ""),
                path=str(self.public.get("path") or ""), output_preview=str(details.get("output_preview") or ""),
                duration_ms=int((time.monotonic() - self.started) * 1000),
                visibility="progress", semantic_kind="tool_execution", payload={**metadata, "status": status})
            if publish_coding_event(event) is False:
                self.trace_complete = False
            if self.authority.taskboard is not None and event_type in {"tool.completed", "tool.failed", "tool.interrupted", "tool.cancelled", "tool.waiting"}:
                try:
                    context = self.authority.activity
                    with context.publication() if context else nullcontext():
                        self.authority.taskboard.add_evidence(self.request.task_id,
                            f"ToolRunner {self.request.operation_id}: {self.request.tool_name} {status}; execution evidence does not establish goal verification.")
                        changed = details.get("changed_files") or []
                        if changed:
                            self.authority.taskboard.add_files_touched(self.request.task_id, list(changed))
                except Exception:
                    self.trace_complete = False
                    logger.warning("TaskBoard execution evidence publication unavailable")

    def retain(self, channel: str, text: str) -> None:
        """Only sanitized content enters context artifacts or public journal events."""
        from mana_agent.context_cost.artifact_store import ContextArtifactStore
        text = str(_public_value(text, self.protected))
        with self.lock:
            body = text.encode("utf-8")
            room = self.runner.trace_budget.claim(min(len(body), max(0, self.runner.capture_limit - self.retained_bytes)), self.runner.trace_limit)
            retained = body[:room].decode("utf-8", errors="ignore")
            byte_count = len(retained.encode("utf-8"))
            self.omitted_bytes += len(body) - byte_count
            if self.omitted_bytes:
                self.trace_complete = False
            if not retained:
                return
            offset = self.offsets.get(channel, 0)
            reference = None
            context = self.authority.activity
            if context and context.conversation_id and context.repository_id:
                try:
                    reference = ContextArtifactStore(retention_days=self.runner.retention_days).put(retained,
                        session_id=context.conversation_id, repository_id=context.repository_id,
                        workspace_id=self.runner.workspace_id, content_type=f"execution/{channel}").artifact_id
                except Exception:
                    self.trace_complete = False
                    logger.warning("Execution trace persistence unavailable")
            else:
                self.trace_complete = False
            entry = {"reference": reference or "", "offset": offset, "length": len(retained), "bytes": byte_count}
            self.references.setdefault(channel, []).append(entry)
            self.offsets[channel] = offset + len(retained)
            self.retained_bytes += byte_count
            self.emit(f"tool.{channel}" if channel in {"stdout", "stderr"} else "tool.output",
                stream_channel=channel, stream_offset=offset, output_preview=retained[:3000],
                content_reference=reference or "", content_length=len(retained), content_bytes=byte_count,
                preview_truncated=len(retained) > 3000)


class ToolRunnerAgent:
    """One scoped execution service, sharing existing state and recovery machinery."""
    def __init__(self, authority: ToolExecutionAuthority, *, capture_limit: int = 10_485_760,
                 trace_limit: int = 104_857_600, retention_days: int = 30, workspace_id: str = ""):
        if authority.seal is not _SEAL:
            raise ExecutionAuthorityError("Execution authority was not bound by the runtime")
        self.authority = authority
        self.capture_limit, self.trace_limit, self.retention_days = capture_limit, trace_limit, retention_days
        self.workspace_id = workspace_id
        from mana_agent.services.operation_trace import execution_trace_budget
        self.trace_budget = execution_trace_budget(authority.activity)
        self.lock = threading.RLock()
        self.results: OrderedDict[str, tuple[str, ToolOperationResult]] = OrderedDict()

    def request(self, name: str, arguments: dict[str, Any], *, call_id: str, decision_id: str,
                timeout_seconds: float = 120) -> ToolOperationRequest:
        policy = self.authority.policies.get(name)
        if policy is None:
            raise ExecutionAuthorityError(f"Unregistered execution capability: {name}")
        context = self.authority.activity
        execution_id = str(getattr(context, "execution_id", "") or self.authority.task_id)
        return ToolOperationRequest(task_id=self.authority.task_id,
            root_task_id=str(getattr(context, "root_task_id", "") or self.authority.task_id),
            parent_task_id=str(getattr(context, "parent_task_id", "") or ""), execution_id=execution_id,
            parent_execution_id=execution_id, requested_by_agent_id=self.authority.actor_id,
            agent_id=self.authority.worker_id, decision_id=decision_id, tool_call_id=call_id,
            operation_id=call_id, operation_attempt_id=call_id,
            attempt_id=str(getattr(context, "attempt_id", "") or ""),
            attempt_generation=int(getattr(context, "attempt_generation", 0)), tool_name=name, arguments=copy.deepcopy(arguments),
            required_capability=policy.capability, authorization_scope=self.authority.scope_id,
            workspace_root=str(self.authority.workspace.worktree_path.resolve()), timeout_seconds=timeout_seconds,
            deadline=datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds), cancellation_id=execution_id)

    def execute(self, request: ToolOperationRequest, implementation: Callable[[dict[str, Any]], dict[str, Any]]) -> ToolOperationResult:
        a = self.authority
        policy = a.policies.get(request.tool_name)
        if (request.authorization_scope != a.scope_id or request.task_id != a.task_id
                or request.requested_by_agent_id != a.actor_id or request.agent_id != a.worker_id
                or request.workspace_root != str(a.workspace.worktree_path.resolve())
                or policy is None or request.required_capability != policy.capability):
            raise ExecutionAuthorityError("Tool request differs from its bound execution authority")
        context = a.activity
        execution_id = str(getattr(context, "execution_id", "") or a.task_id)
        if (request.execution_id != execution_id or request.parent_execution_id != execution_id
                or request.root_task_id != str(getattr(context, "root_task_id", "") or a.task_id)
                or request.parent_task_id != str(getattr(context, "parent_task_id", "") or "")
                or request.cancellation_id != execution_id or request.execution_level != 3):
            raise ExecutionAuthorityError("Tool correlation differs from its admitted execution")
        if a.taskboard is not None:
            task = a.taskboard.get_task(a.task_id)
            if a.actor_id not in {task.owner_agent_id, *task.assigned_agent_ids, *task.assigned_subagent_ids}:
                raise ExecutionAuthorityError("Specialist task assignment was revoked")
            if request.tool_name in task.forbidden_tools or task.allowed_tools and request.tool_name not in task.allowed_tools:
                raise ExecutionAuthorityError("Tool is outside the TaskBoard capability scope")
        node = a.registry.agents.get(a.actor_id)
        if node is None or node.allowed_tools and request.tool_name not in node.allowed_tools:
            raise ExecutionAuthorityError("Tool is outside the registered actor capability scope")
        if request.attempt_id != str(getattr(context, "attempt_id", "") or "") or request.attempt_generation != int(getattr(context, "attempt_generation", 0)):
            raise ExecutionAuthorityError("Tool request belongs to another execution attempt")
        validate_arguments(request.arguments, a.schemas.get(request.tool_name, {"type": "object"}))
        if a.authorize is not None:
            a.authorize(request.tool_name)
        if policy.effect == "command" and request.tool_name in {"git", "git_generic"}:
            from mana_agent.multi_agent.tools.git_tools import classify_git_risk
            arguments = request.arguments.get("args")
            argv = arguments.get("args") if isinstance(arguments, dict) else arguments
            if isinstance(argv, list) and classify_git_risk(argv).value == "READ_ONLY":
                policy = OperationPolicy(policy.capability, "read_only")
        if a.workspace.sandbox == "readOnly" and (policy.effect not in {"read_only", "external"} or policy.repository and policy.effect != "read_only"):
            raise ExecutionAuthorityError("Tool capability is not allowed in the read-only execution scope")
        cwd = (a.workspace.worktree_path / str(request.arguments.get("cwd") or ".")).resolve()
        if not cwd.is_relative_to(a.workspace.worktree_path.resolve()):
            raise ExecutionAuthorityError("Tool cwd escapes its assigned workspace")
        if policy.repository:
            def validate_paths(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in {"path", "file_path", "repo_path", "repo_root", "execution_repo_root", "source", "destination", "src", "dst"} and isinstance(item, str) and item:
                            target = (a.workspace.worktree_path / item).resolve()
                            if not target.is_relative_to(a.workspace.worktree_path.resolve()):
                                raise ExecutionAuthorityError("Tool target escapes its assigned workspace")
                            if any(part.startswith(".env") or part in {".ssh", "secrets.toml", "id_rsa", "id_ed25519"} for part in target.relative_to(a.workspace.worktree_path.resolve()).parts):
                                raise ExecutionAuthorityError("Tool target is protected credential material")
                        elif key in {"paths", "files"} and isinstance(item, list):
                            for entry in item:
                                validate_paths({"path": entry} if isinstance(entry, str) else entry)
                        elif key in {"writes", "moves", "patches", "args"}:
                            validate_paths(item)
                        elif key == "patch" and isinstance(item, str):
                            for line in item.splitlines():
                                for prefix in ("*** Add File: ", "*** Update File: ", "*** Delete File: ", "*** Move to: "):
                                    if line.startswith(prefix):
                                        validate_paths({"path": line[len(prefix):]})
                elif isinstance(value, list):
                    for item in value:
                        validate_paths(item)
            validate_paths(request.arguments)
        fingerprint = hashlib.sha256(json.dumps({"tool": request.tool_name, "arguments": request.arguments}, sort_keys=True, default=str).encode()).hexdigest()
        # Duplicate delivery is serialized with execution, including read calls.
        with _LOCKS.lock_for(f"request:{a.scope_id}:{request.tool_call_id}"):
            if request.tool_call_id in self.results:
                old_fingerprint, result = self.results[request.tool_call_id]
                if old_fingerprint != fingerprint:
                    raise ExecutionAuthorityError("A tool-call identity cannot be reused with different arguments")
                return result
            running = RunningOperation(request, a, self)
            with running.scope():
                running.validate_live()
                running.emit("tool.requested", status="queued")
                lock = _LOCKS.workspace_lock(request.workspace_root, validate=running.validate_live) if policy.repository and policy.effect != "read_only" and not a.externally_locked else nullcontext()
                with lock:
                    running.validate_live()
                    try:
                        result = self._execute(running, implementation, policy, fingerprint)
                    except Exception as exc:
                        running.emit("tool.interrupted", status="interrupted", error=safe_text(str(exc), protected=running.protected),
                            termination_reason="Execution stopped; reconcile any durable action receipt before retry")
                        raise
            if result.status == "success":
                self.results[request.tool_call_id] = (fingerprint, result)
            return result

    def _execute(self, operation: RunningOperation, implementation, policy, fingerprint: str) -> ToolOperationResult:
        from mana_agent.execution_supervisor.models import ActionRequestState, SideEffectClassification
        request, context = operation.request, self.authority.activity
        supervisor = getattr(context, "supervisor", None)
        action = None
        if supervisor is not None and policy.effect != "read_only":
            action = supervisor.prepare_action(request.task_id, attempt_id=request.attempt_id,
                lease_token=context.lease_token, tool_name=request.tool_name,
                action_fingerprint=fingerprint, classification=SideEffectClassification.UNKNOWN,
                idempotency_key=request.tool_call_id)
            supervisor.update_action(action.action_id, request_state=ActionRequestState.STARTED,
                attempt_id=request.attempt_id, lease_token=context.lease_token)
        before = _snapshot(Path(request.workspace_root), request.arguments, request.tool_name, self.capture_limit) if policy.effect in {"mutation", "command"} else {}
        operation.emit("tool.started")
        raw: dict[str, Any]
        try:
            operation.validate_live()
            raw = implementation(copy.deepcopy(request.arguments))
            if not isinstance(raw, dict):
                raw = {"ok": True, "result": raw}
            if raw.get("permission_required") or raw.get("status") in {"awaiting_approval", "waiting_approval"}:
                status = "waiting_approval"
            elif raw.get("cancelled") or raw.get("error_code") == "CANCELLED":
                status = "cancelled"
            elif raw.get("timed_out"):
                status = "timed_out"
            else:
                observed_ok = not bool(raw.get("error")) and raw.get("returncode", raw.get("exit_code")) in {None, 0}
                status = "success" if raw.get("ok", observed_ok) else "failed"
            raw.setdefault("ok", status == "success")
        except (InterruptedError, TimeoutError) as exc:
            status = "cancelled" if isinstance(exc, InterruptedError) else "timed_out"
            raw = {"ok": False, "error": str(exc), status: True,
                   "stdout": getattr(exc, "stdout", ""), "stderr": getattr(exc, "stderr", "")}
        except Exception as exc:
            import subprocess
            status = "interrupted" if getattr(exc, "outcome_unknown", False) else "timed_out" if isinstance(exc, subprocess.TimeoutExpired) else "failed"
            raw = {"ok": False, "error": safe_text(str(exc), protected=operation.protected), "error_code": type(exc).__name__,
                   "stdout": str(getattr(exc, "stdout", "") or ""), "stderr": str(getattr(exc, "stderr", "") or "")}
        after = _snapshot(Path(request.workspace_root), request.arguments, request.tool_name, self.capture_limit) if policy.effect in {"mutation", "command"} else {}
        mutations = _mutations(before, after, partial=status != "success")
        if not getattr(before, "complete", True) or not getattr(after, "complete", True) or any(value is None for value in (*before.values(), *after.values())):
            operation.trace_complete = False
        for mutation in mutations:
            operation.retain("diff", mutation.diff)
            operation.emit("file.edit.applied", status="running",
                mutation=mutation.model_dump(exclude={"diff"}), path=mutation.path,
                changed_files=[mutation.path], diff=mutation.diff[:3000], evidence_status="observed")
        # Streaming runners already recorded these channels; buffered adapters get
        # truthful terminal output, never a fabricated live stream.
        for channel in ("stdout", "stderr"):
            if raw.get(channel) and not operation.offsets.get(channel) and not operation.public.get("child_operation_ids"):
                operation.retain(channel, str(raw[channel]))
        public_result = _public_value(raw, operation.protected)
        operation.retain("result", json.dumps(public_result, ensure_ascii=False, default=str))
        result = ToolOperationResult(tool_call_id=request.tool_call_id, operation_id=request.operation_id,
            operation_attempt_id=request.operation_attempt_id, task_id=request.task_id, attempt_id=request.attempt_id,
            root_task_id=request.root_task_id, parent_task_id=request.parent_task_id,
            execution_id=request.execution_id, parent_execution_id=request.parent_execution_id,
            requested_by_agent_id=request.requested_by_agent_id, agent_id=request.agent_id,
            decision_id=request.decision_id, tool_name=request.tool_name,
            started_at=operation.started_at, ended_at=datetime.now(timezone.utc), output_references=operation.references,
            attempt_generation=request.attempt_generation, status=status, result=public_result,
            duration_ms=int((time.monotonic() - operation.started) * 1000),
            exit_code=raw.get("returncode", raw.get("exit_code")),
            termination_reason=str(raw.get("error") or "") if status != "success" else "",
            mutations=mutations, retained_bytes=operation.retained_bytes, omitted_bytes=operation.omitted_bytes,
            trace_complete=operation.trace_complete)
        if action is not None:
            supervisor.update_action(action.action_id,
                attempt_id=request.attempt_id, lease_token=context.lease_token,
                request_state=ActionRequestState.SUCCEEDED if status == "success" else ActionRequestState.OUTCOME_UNKNOWN if status in {"cancelled", "timed_out", "interrupted"} or mutations else ActionRequestState.FAILED,
                result_reference=next((item["reference"] for item in operation.references.get("result", []) if item["reference"]), ""),
                verification_state={"operation_attempt_id": request.operation_attempt_id, "status": status})
        operation.emit("tool.completed" if status == "success" else "tool.waiting" if status == "waiting_approval" else "tool.cancelled" if status == "cancelled" else "tool.failed",
            status=status, exit_code=result.exit_code, termination_reason=result.termination_reason,
            result_summary=str(public_result.get("summary") or public_result.get("error") or status),
            changed_files=[m.path for m in mutations], output_references=operation.references)
        result.trace_complete = operation.trace_complete
        result.result["execution"] = result.model_dump(mode="json", exclude={"result", "mutations"})
        result.result["mutations"] = _public_value([m.model_dump() for m in mutations], operation.protected)
        return result


class FileSnapshot(dict):
    complete: bool = True


def _snapshot(root: Path, arguments: dict[str, Any], tool_name: str, limit: int) -> dict[str, bytes | None]:
    """Bounded observation within the worker lock; never execute another tool."""
    import os
    paths: set[str] = set()
    complete = True
    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"path", "file_path", "source", "destination", "src", "dst", "from", "to"} and isinstance(item, str):
                    paths.add(item)
                elif key == "paths" and isinstance(item, list):
                    paths.update(p for p in item if isinstance(p, str))
                elif key in {"writes", "moves", "patches"}:
                    collect(item)
                elif key == "patch" and isinstance(item, str):
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, str):
            for line in value.splitlines():
                for prefix in ("*** Add File: ", "*** Update File: ", "*** Delete File: ", "*** Move to: "):
                    if line.startswith(prefix):
                        paths.add(line[len(prefix):])
    collect(arguments)
    if not paths:
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in {".git", ".mana", "node_modules", "venv", ".venv", "__pycache__"} and not (Path(directory) / d).is_symlink()]
            for name in files:
                paths.add(str((Path(directory) / name).relative_to(root)))
                if len(paths) >= 5000:
                    complete = False
                    break
            if len(paths) >= 5000:
                break
    snapshot = FileSnapshot()
    snapshot.complete = complete
    remaining = limit
    for path in sorted(paths):
        target = (root / path).resolve()
        if not target.is_relative_to(root) or any(p in {".git", ".mana"} or p.startswith(".env") for p in Path(path).parts):
            snapshot.complete = False
            continue
        try:
            if target.is_file() and target.stat().st_size <= remaining:
                body = target.read_bytes()
                snapshot[str(target.relative_to(root))] = body
                remaining -= len(body)
            elif target.exists():
                snapshot[str(target.relative_to(root))] = None
                snapshot.complete = False
        except OSError:
            snapshot[str(target.relative_to(root))] = None
    return snapshot


def _mutations(before: dict[str, bytes | None], after: dict[str, bytes | None], *, partial: bool) -> list[FileMutationEvidence]:
    result = []
    for path in sorted(before.keys() | after.keys()):
        old, new = before.get(path, b""), after.get(path, b"")
        if old == new and (path in before) == (path in after):
            continue
        unknown = old is None or new is None or (path not in before and not getattr(before, "complete", True)) or (path not in after and not getattr(after, "complete", True))
        binary = unknown or b"\0" in (old or b"") or b"\0" in (new or b"")
        try:
            (old or b"").decode("utf-8")
            (new or b"").decode("utf-8")
        except UnicodeDecodeError:
            binary = True
        old_lines = (old or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        new_lines = (new or b"").decode("utf-8", errors="replace").splitlines(keepends=True)
        diff = "" if binary else "".join(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{path}" if path in before else "/dev/null", tofile=f"b/{path}" if path in after else "/dev/null"))
        result.append(FileMutationEvidence(path=path, operation="delete" if path not in after else "create" if path not in before else "edit",
            before_sha256=hashlib.sha256(old).hexdigest() if old is not None and path in before else "",
            after_sha256=hashlib.sha256(new).hexdigest() if new is not None and path in after else "",
            diff=diff, added_lines=sum(line.startswith("+") and not line.startswith("+++") for line in diff.splitlines()),
            removed_lines=sum(line.startswith("-") and not line.startswith("---") for line in diff.splitlines()),
            binary=binary, status="unknown" if unknown else "partial" if partial else "applied"))
    deleted = [item for item in result if item.operation == "delete" and item.before_sha256]
    for created in list(result):
        if created.operation != "create":
            continue
        match = next((item for item in deleted if item.before_sha256 == created.after_sha256), None)
        if match is not None:
            created.operation = "move"
            created.old_path = match.path
            created.before_sha256 = match.before_sha256
            created.diff = f"rename from {match.path}\nrename to {created.path}\n"
            created.added_lines = created.removed_lines = 0
            result.remove(match)
            deleted.remove(match)
    return result
