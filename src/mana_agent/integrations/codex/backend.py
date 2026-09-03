"""Provider-neutral coding backend powered by the official Codex app-server."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import logging
import os
import subprocess
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from mana_agent.coding.models import AgentEvent, CodingTask, CodingTaskResult, WorkspaceContext
from mana_agent.integrations.codex.client import AsyncCodexAppServer, CodexCancellationOutcome
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.event_adapter import adapt_codex_event
from mana_agent.integrations.codex.exceptions import (
    CodexError,
    CodexExecutionError,
    CodexInterruptionError,
    CodexTimeoutError,
    CodexUnavailableError,
)
from mana_agent.integrations.codex.health import check_codex_health
from mana_agent.integrations.codex.prompt_builder import build_codex_prompt
from mana_agent.integrations.codex.result_parser import parse_codex_result
from mana_agent.integrations.codex.runtime_config import CodexRuntimeConfigBuilder
from mana_agent.integrations.codex.runtime_environment import CodexRuntimeContext, CodexRuntimeEnvironment
from mana_agent.context_cost import ContextCostGovernor
from mana_agent.context_cost.estimator import estimate_value_tokens
from mana_agent.context_cost.models import ContextSegment
from mana_agent.utils.path_safety import safe_cwd

ClientFactory = Callable[[tuple[str, ...]], AsyncCodexAppServer]

_CODEX_SANDBOX_VALUES = {
    "readOnly": "read-only",
    "workspaceWrite": "workspace-write",
}


class CodexCodingBackend:
    name = "codex"

    def __init__(
        self,
        settings: CodexSettings,
        *,
        client_factory: ClientFactory | None = None,
        worker_id: str | None = None,
        resume_thread_id: str = "",
        context_cost_governor: ContextCostGovernor | None = None,
    ) -> None:
        self.settings = settings
        self.worker_id = worker_id or f"codex-{uuid.uuid4().hex[:8]}"
        self.resume_thread_id = str(resume_thread_id or "").strip()
        self.context_cost_governor = context_cost_governor
        self._uses_default_client = client_factory is None
        self._client_factory = client_factory or (lambda command: AsyncCodexAppServer(command))
        self._client: AsyncCodexAppServer | None = None
        self._runtime_context: CodexRuntimeContext | None = None
        self._active: dict[str, tuple[str, str]] = {}
        self._results: dict[str, CodingTaskResult] = {}
        self._run_lock = asyncio.Lock()

    async def start(
        self,
        repository_path: str | Path | None = None,
        *,
        sandbox_mode: str = "workspace-write",
    ) -> None:
        if self._client is not None and self._client.running:
            return
        if not self.settings.enabled:
            raise CodexUnavailableError("Codex integration is disabled. No fallback backend was executed.")
        executable = self.settings.codex_bin
        health_root = _existing_directory(repository_path) or safe_cwd()
        if self._uses_default_client:
            report = await asyncio.to_thread(
                check_codex_health,
                self.settings,
                health_root,
            )
            if not report.healthy or report.executable is None:
                detail = "; ".join(report.errors) or "unknown health-check failure"
                raise CodexUnavailableError(
                    "Codex preflight failed. No fallback backend was executed. "
                    f"Reason: {detail}"
                )
            executable = report.executable
            runtime_config = CodexRuntimeConfigBuilder.build(
                self.settings,
                sandbox_mode=sandbox_mode,
            )
        command = (executable, "app-server")
        if self._uses_default_client:
            self._runtime_context = CodexRuntimeEnvironment.create(runtime_config)
            # Prefer the repository as the child CWD so Codex project discovery
            # is correct; fall back to the isolated CODEX_HOME when the repo was
            # deleted under a live process (SWE-bench thrash).
            child_cwd = _existing_directory(repository_path) or self._runtime_context.home
            self._client = AsyncCodexAppServer(
                command,
                environment=self._runtime_context.environment,
                provider_name=runtime_config.provider_display_name,
                model=runtime_config.model,
                cwd=child_cwd,
            )
        else:
            self._client = self._client_factory(command)
        try:
            await self._client.start()
        except BaseException:
            await self.close()
            raise

    async def execute(self, task: CodingTask, workspace: WorkspaceContext) -> CodingTaskResult:
        async for _event in self.stream(task, workspace):
            pass
        result = self._results.get(task.task_id)
        if result is None:
            raise CodexExecutionError(f"Codex task produced no result: {task.task_id}")
        return result

    def result_for(self, task_id: str) -> CodingTaskResult:
        result = self._results.get(str(task_id))
        if result is None:
            raise CodexExecutionError(f"Codex task produced no result: {task_id}")
        return result

    async def stream(self, task: CodingTask, workspace: WorkspaceContext) -> AsyncIterator[AgentEvent]:
        self._validate_workspace(task, workspace)
        execution_dir = _execution_directory(workspace)
        if not execution_dir.is_dir():
            raise CodexExecutionError(
                f"Codex execution directory does not exist: {execution_dir}. "
                "The worktree may have been removed while the agent was running."
            )
        await self.start(
            execution_dir,
            sandbox_mode=_codex_sandbox(workspace),
        )
        if self._client is None:
            raise CodexUnavailableError("Codex app-server did not start")
        task_created_at = getattr(task, "task_created_at", None) or datetime.now(timezone.utc)
        scheduled_at = datetime.now(timezone.utc)
        worker_claimed_at = datetime.now(timezone.utc)
        provider_started_at: datetime | None = None
        provider_completed_at: datetime | None = None
        async with self._run_lock:
            baseline_changes = (
                await asyncio.to_thread(_git_changed_file_state, workspace.worktree_path)
                if task.requires_repository_write
                else {}
            )
            notifications: list[dict[str, Any]] = []
            seen_event_ids: set[str] = set()
            sequence = 0
            thread_id = ""
            turn_id = ""
            governor_call_id = ""
            last_usage: dict[str, Any] | None = None
            first_output_at: datetime | None = None
            last_output_at: datetime | None = None
            output_chunks_count: int = 0
            prompt = build_codex_prompt(task, workspace)
            sequence += 1
            transport = getattr(self.settings, "codex_transport", None)
            transport_value = getattr(transport, "value", str(transport or ""))
            yield AgentEvent(
                event_type="backend.selected",
                task_id=task.task_id,
                backend="codex",
                sequence=sequence,
                title="Codex backend selected",
                summary=self.worker_id,
                model=self.settings.model or "",
                payload={
                    "provider": self.settings.provider,
                    "execution_mode": getattr(self.settings.execution_mode, "value", str(self.settings.execution_mode)),
                    "transport": transport_value or (
                        "codex_responses_bridge"
                        if not self.settings.supports_responses_api
                        else "direct_responses"
                    ),
                    "task_created_at": task_created_at.isoformat(),
                    "scheduled_at": scheduled_at.isoformat(),
                    "worker_claimed_at": worker_claimed_at.isoformat(),
                },
            )
            try:
                if self.resume_thread_id:
                    thread_response = await self._client.request(
                        "thread/resume",
                        {"threadId": self.resume_thread_id, **self._thread_params(workspace)},
                    )
                else:
                    thread_response = await self._client.request("thread/start", self._thread_params(workspace))
                thread_id = _response_id(thread_response, "thread")
                if not thread_id and self.resume_thread_id:
                    thread_id = self.resume_thread_id
                if not thread_id:
                    raise CodexExecutionError("Codex thread/start returned no thread id")
                sequence += 1
                yield AgentEvent(
                    event_type="turn.starting",
                    task_id=task.task_id,
                    backend="codex",
                    sequence=sequence,
                    title="Starting Codex turn",
                    thread_id=thread_id,
                    model=self.settings.model or "",
                )
                if self.context_cost_governor is not None and self.context_cost_governor.enabled:
                    governor_call_id, governor_decision = self.context_cost_governor.before_model_call(
                        (
                            ContextSegment(
                                kind="system",
                                content="Mana-Agent Codex worker safety and output contract",
                                token_estimate=estimate_value_tokens("Mana-Agent Codex worker safety and output contract"),
                                protected=True,
                                source_id="codex:contract",
                            ),
                            ContextSegment(
                                kind="user",
                                content=task.goal,
                                token_estimate=estimate_value_tokens(task.goal),
                                protected=True,
                                source_id="codex:goal",
                            ),
                            ContextSegment(
                                kind="repository",
                                content=prompt,
                                token_estimate=estimate_value_tokens(prompt),
                                protected=True,
                                source_id="codex:mana-prompt",
                            ),
                        ),
                        model=self.settings.model or "app-server-default",
                        provider=self.settings.provider,  # real inference provider, not the bridge
                        turn_id=task.task_id,
                        task_id=task.task_id,
                        agent_id=self.worker_id,
                    )
                    sequence += 1
                    yield AgentEvent(
                        event_type="context.budget",
                        task_id=task.task_id,
                        backend="codex",
                        sequence=sequence,
                        title="Codex context budget",
                        summary=f"{governor_decision.snapshot.used_tokens}/{governor_decision.snapshot.budget.context_window} tokens",
                        model=self.settings.model or "",
                        payload=governor_decision.snapshot.as_dict(),
                    )
                provider_started_at = datetime.now(timezone.utc)
                turn_response = await self._client.request(
                    "turn/start",
                    {
                        "threadId": thread_id,
                        "input": [{"type": "text", "text": prompt}],
                        "cwd": str(_execution_directory(workspace)),
                        "approvalPolicy": self.settings.approval_policy,
                        "sandbox": _codex_sandbox(workspace),
                        **({"model": self.settings.model} if self.settings.model else {}),
                    },
                )
                turn_id = _response_id(turn_response, "turn")
                if not turn_id:
                    raise CodexExecutionError("Codex turn/start returned no turn id")
                self._active[task.task_id] = (thread_id, turn_id)
                sequence += 1
                yield AgentEvent(
                    event_type="turn.started",
                    task_id=task.task_id,
                    backend="codex",
                    sequence=sequence,
                    title="Codex turn started",
                    thread_id=thread_id,
                    turn_id=turn_id,
                    model=self.settings.model or "",
                    payload={
                        "provider_started_at": provider_started_at.isoformat(),
                    },
                )
                logger.info(
                    "codex_stream.attached task_id=%s thread_id=%s turn_id=%s",
                    task.task_id,
                    thread_id,
                    turn_id,
                )
                iterator = self._client.notifications(thread_id).__aiter__()
                deadline = asyncio.get_running_loop().time() + self.settings.task_timeout_seconds
                try:
                    while True:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            raise asyncio.TimeoutError
                        try:
                            notification = await asyncio.wait_for(anext(iterator), timeout=remaining)
                        except StopAsyncIteration:
                            provider_completed_at = datetime.now(timezone.utc)
                            logger.info(
                                "codex_stream.completed task_id=%s thread_id=%s turn_id=%s output_chunks=%d",
                                task.task_id,
                                thread_id,
                                turn_id,
                                output_chunks_count,
                            )
                            break
                        event = adapt_codex_event(
                            task.task_id,
                            notification,
                            sequence=sequence + 1,
                            model=self.settings.model or "",
                        )
                        if event.event_id in seen_event_ids:
                            continue
                        seen_event_ids.add(event.event_id)
                        sequence += 1
                        notifications.append(notification)
                        if event.output_preview or event.event_type.startswith("command.output"):
                            if first_output_at is None:
                                first_output_at = datetime.now(timezone.utc)
                                logger.info(
                                    "codex_stream.first_output task_id=%s thread_id=%s turn_id=%s event_id=%s",
                                    task.task_id,
                                    thread_id,
                                    turn_id,
                                    event.event_id,
                                )
                            last_output_at = datetime.now(timezone.utc)
                            output_chunks_count += 1
                        # A provider-level turn/completed notification only means
                        # Codex has stopped streaming. Mana still has to parse the
                        # trace, verify the repository outcome, and publish the
                        # validated terminal answer. Keep the UI in an explicit
                        # finalizing state until that work is done.
                        if event.event_type == "turn.completed":
                            event = event.model_copy(
                                update={
                                    "event_type": "turn.finalizing",
                                    "status": "running",
                                    "title": "Codex response received — preparing result",
                                }
                            )
                        if event.token_usage:
                            last_usage = dict(event.token_usage)
                            hard_reason = self.context_cost_governor.active_hard_limit_reason(
                                last_usage,
                                provider=self.settings.provider,
                                model=self.settings.model or "app-server-default",
                                context_window=(int(last_usage["context_window"]) if last_usage.get("context_window") is not None else None),
                            ) if self.context_cost_governor is not None else None
                            if hard_reason:
                                await self._client.interrupt(thread_id=thread_id, turn_id=turn_id)
                                raise CodexExecutionError(
                                    f"Codex turn interrupted because the enforce-mode {hard_reason} was reached."
                                )
                        if event.event_type == "warning" and "approval" in str(notification.get("method") or "").lower():
                            await self._client.deny_server_request(notification)
                            raise CodexExecutionError(
                                "Codex requested approval. Mana-Agent denied the request and did not elevate permissions."
                            )
                        yield event
                finally:
                    logger.info(
                        "codex_stream.detached task_id=%s thread_id=%s turn_id=%s output_chunks=%d",
                        task.task_id,
                        thread_id,
                        turn_id,
                        output_chunks_count,
                    )
            except (asyncio.TimeoutError, CodexTimeoutError) as exc:
                logger.warning(
                    "codex_stream.timeout task_id=%s thread_id=%s turn_id=%s output_chunks=%d",
                    task.task_id,
                    thread_id,
                    turn_id,
                    output_chunks_count,
                )
                if provider_completed_at is None:
                    provider_completed_at = datetime.now(timezone.utc)
                if thread_id and turn_id:
                    try:
                        await self._client.interrupt(thread_id=thread_id, turn_id=turn_id)
                    except Exception:
                        pass
                err_code = getattr(exc, "error_code", "CODING_PROVIDER_TIMEOUT") if isinstance(exc, CodexTimeoutError) else "CODING_TIMEOUT"
                err_msg = str(exc) if str(exc).strip() else "Codex task timed out"
                notifications.append(
                    {"method": "turn/failed", "params": {"message": err_msg, "error_code": err_code, "reason": "timeout"}}
                )
                yield AgentEvent(
                    event_type="error",
                    task_id=task.task_id,
                    backend="codex",
                    sequence=sequence + 1,
                    status="failed",
                    title="Codex task timed out",
                    error=err_msg,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    payload={"error_code": err_code, "error_category": "timeout"},
                )
            except CodexInterruptionError as exc:
                logger.warning(
                    "codex_stream.interrupted task_id=%s thread_id=%s turn_id=%s reason=%s output_chunks=%d",
                    task.task_id,
                    thread_id,
                    turn_id,
                    exc.reason,
                    output_chunks_count,
                )
                if provider_completed_at is None:
                    provider_completed_at = datetime.now(timezone.utc)
                if thread_id and turn_id:
                    try:
                        await self._client.interrupt(thread_id=thread_id, turn_id=turn_id)
                    except Exception:
                        pass
                err_code = exc.error_code or "MODEL_INTERRUPTED"
                err_msg = str(exc) or "Codex turn interrupted"
                notifications.append(
                    {"method": "turn/cancelled", "params": {"message": err_msg, "error_code": err_code, "reason": exc.reason}}
                )
                yield AgentEvent(
                    event_type="error",
                    task_id=task.task_id,
                    backend="codex",
                    sequence=sequence + 1,
                    status="cancelled",
                    title="Codex turn interrupted",
                    summary=err_msg,
                    error=err_msg,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    payload={"error_code": err_code, "error_category": "interruption", "interruption_reason": exc.reason},
                )
            except CodexError as exc:
                if provider_completed_at is None:
                    provider_completed_at = datetime.now(timezone.utc)
                err_code = getattr(exc, "error_code", "") or "CODING_AGENT_FAILED"
                http_status = getattr(exc, "http_status", None)
                orig_err = getattr(exc, "original_error", "") or str(exc)
                provider = getattr(exc, "provider", "") or self.settings.provider
                model = getattr(exc, "model", "") or (self.settings.model or "app-server-default")
                transport = getattr(exc, "transport", "")
                if not transport and hasattr(self.settings, "codex_transport"):
                    transport = getattr(self.settings.codex_transport, "value", str(self.settings.codex_transport))

                notifications.append(
                    {
                        "method": "turn/failed",
                        "params": {
                            "message": str(exc),
                            "error_code": err_code,
                            "http_status": http_status,
                            "original_error": orig_err,
                            "provider": provider,
                            "model": model,
                            "transport": transport,
                        },
                    }
                )
                yield AgentEvent(
                    event_type="error",
                    task_id=task.task_id,
                    backend="codex",
                    sequence=sequence + 1,
                    status="failed",
                    title="Codex task failed",
                    summary=str(exc),
                    error=str(exc),
                    thread_id=thread_id,
                    turn_id=turn_id,
                    payload={
                        "error_code": err_code,
                        "http_status": http_status,
                        "original_error": orig_err,
                        "provider": provider,
                        "model": model,
                        "transport": transport,
                    },
                )
            finally:
                if provider_completed_at is None and provider_started_at is not None:
                    provider_completed_at = datetime.now(timezone.utc)
                task_completed_at = datetime.now(timezone.utc)
                if self.context_cost_governor is not None and governor_call_id:
                    self.context_cost_governor.record_model_call(
                        governor_call_id,
                        usage=last_usage,
                        provider=self.settings.provider,
                        model=self.settings.model or "app-server-default",
                        estimated_input=prompt,
                        turn_id=task.task_id,
                        task_id=task.task_id,
                        agent_id=self.worker_id,
                    )
                self._active.pop(task.task_id, None)
                changed_files = (
                    await asyncio.to_thread(
                        _git_changed_files,
                        workspace.worktree_path,
                        baseline=baseline_changes,
                    )
                    if task.requires_repository_write
                    else []
                )
                self._results[task.task_id] = parse_codex_result(
                    task=task,
                    workspace=workspace,
                    worker_id=self.worker_id,
                    thread_id=thread_id,
                    turn_id=turn_id,
                    notifications=notifications,
                    changed_files=changed_files,
                    task_created_at=task_created_at,
                    scheduled_at=scheduled_at,
                    worker_claimed_at=worker_claimed_at,
                    provider_started_at=provider_started_at,
                    provider_completed_at=provider_completed_at,
                    task_completed_at=task_completed_at,
                )

    async def cancel(self, task_id: str, *, timeout_seconds: float = 2.0) -> CodexCancellationOutcome:
        active = self._active.get(str(task_id))
        if active is None or self._client is None:
            return CodexCancellationOutcome(
                acknowledged=False,
                status="closed" if self._client is None else "not_found",
                thread_id=active[0] if active else "",
                turn_id=active[1] if active else "",
                error=f"No active Codex task: {task_id}" if active is None else "Codex client is not running",
            )
        outcome = await self._client.interrupt(
            thread_id=active[0],
            turn_id=active[1],
            timeout_seconds=timeout_seconds,
        )
        if not outcome.acknowledged:
            await self.close()
        return outcome

    async def close(self, *, wait_timeout: float = 1.0) -> None:
        try:
            if self._client is not None:
                close_func = getattr(self._client, "close", None)
                if callable(close_func):
                    try:
                        await self._client.close(wait_timeout=wait_timeout)
                    except TypeError:
                        await self._client.close()
        finally:
            self._client = None
            self._active.clear()
            if self._runtime_context is not None:
                self._runtime_context.close()
                self._runtime_context = None

    def health(self, repository_path: str | Path):
        return check_codex_health(self.settings, repository_path)

    def _thread_params(self, workspace: WorkspaceContext) -> dict[str, Any]:
        return {
            "cwd": str(_execution_directory(workspace)),
            "approvalPolicy": self.settings.approval_policy,
            "sandbox": _codex_sandbox(workspace),
            **({"model": self.settings.model} if self.settings.model else {}),
        }

    def _validate_workspace(self, task: CodingTask, workspace: WorkspaceContext) -> None:
        if not task.requires_repository_write:
            return
        repository_root = workspace.repository_path.resolve()
        execution_root = workspace.worktree_path.resolve()
        if (
            self.settings.worktree_isolation
            and repository_root == execution_root
            and not workspace.allow_in_place_write
        ):
            raise CodexExecutionError("Codex writing task was not assigned an isolated worktree")
        if not self.settings.worktree_isolation and repository_root == execution_root and not workspace.allow_in_place_write:
            raise CodexExecutionError("Codex in-place writing was not explicitly authorized")
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=execution_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise CodexExecutionError("Codex worktree is not a readable Git checkout")
        if (
            self.settings.worktree_isolation
            and not workspace.allow_in_place_write
            and completed.stdout.strip()
        ):
            raise CodexExecutionError("Codex worktree must be clean before execution")


def _execution_directory(workspace: WorkspaceContext) -> Path:
    return (workspace.working_directory or workspace.worktree_path).resolve()


def _existing_directory(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    try:
        path = Path(value).expanduser().resolve(strict=False)
    except OSError:
        return None
    try:
        return path if path.is_dir() else None
    except OSError:
        return None


def _response_id(response: dict[str, Any], key: str) -> str:
    value = response.get(key)
    if isinstance(value, dict) and value.get("id"):
        return str(value["id"])
    direct = response.get(f"{key}Id") or response.get("id")
    return str(direct or "")


def _codex_sandbox(workspace: WorkspaceContext) -> str:
    """Translate Mana's typed sandbox value to the Codex app-server protocol."""

    try:
        return _CODEX_SANDBOX_VALUES[workspace.sandbox]
    except KeyError as exc:
        raise CodexExecutionError(
            f"Unsupported Codex sandbox value: {workspace.sandbox}"
        ) from exc


def _git_changed_file_state(worktree: Path) -> dict[str, str]:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=worktree,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {}
    root = worktree.expanduser().resolve()
    records = completed.stdout.split(b"\0")
    changed: dict[str, str] = {}
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if len(record) < 4:
            continue
        status = record[:2].decode("ascii", errors="replace")
        value = os.fsdecode(record[3:])
        if status[0] in {"R", "C"} or status[1] in {"R", "C"}:
            # Porcelain v1 -z emits the destination first and the source as a
            # second NUL-delimited field. The destination is the final artifact.
            index += 1
        if not value:
            continue
        target = (root / value).resolve(strict=False)
        try:
            target.relative_to(root)
        except ValueError:
            continue
        changed[value] = f"{status}:{_path_state_fingerprint(target)}"
    return changed


def _git_changed_files(worktree: Path, *, baseline: dict[str, str] | None = None) -> list[str]:
    current = _git_changed_file_state(worktree)
    if baseline is None:
        return sorted(current)
    return sorted(
        path
        for path, fingerprint in current.items()
        if baseline.get(path) != fingerprint
    )


def _path_state_fingerprint(path: Path) -> str:
    try:
        if path.is_symlink():
            return "symlink:" + hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return "file:" + digest.hexdigest()
        if path.is_dir():
            return f"directory:{path.stat().st_mtime_ns}"
    except OSError as exc:
        return f"unreadable:{type(exc).__name__}"
    return "missing"


__all__ = ["CodexCodingBackend"]
