"""Provider-neutral coding backend powered by the official Codex app-server."""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from mana_agent.coding.models import AgentEvent, CodingTask, CodingTaskResult, WorkspaceContext
from mana_agent.integrations.codex.client import AsyncCodexAppServer
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.event_adapter import adapt_codex_event
from mana_agent.integrations.codex.exceptions import CodexError, CodexExecutionError, CodexUnavailableError
from mana_agent.integrations.codex.health import check_codex_health
from mana_agent.integrations.codex.prompt_builder import build_codex_prompt
from mana_agent.integrations.codex.result_parser import parse_codex_result
from mana_agent.integrations.codex.runtime_config import CodexRuntimeConfigBuilder
from mana_agent.integrations.codex.runtime_environment import CodexRuntimeContext, CodexRuntimeEnvironment
from mana_agent.context_cost import ContextCostGovernor
from mana_agent.context_cost.estimator import estimate_value_tokens
from mana_agent.context_cost.models import ContextSegment

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
        if self._uses_default_client:
            report = await asyncio.to_thread(
                check_codex_health,
                self.settings,
                repository_path or Path.cwd(),
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
            self._client = AsyncCodexAppServer(
                command,
                environment=self._runtime_context.environment,
                provider_name=runtime_config.provider_display_name,
                model=runtime_config.model,
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
        await self.start(
            workspace.repository_path,
            sandbox_mode=_codex_sandbox(workspace),
        )
        if self._client is None:
            raise CodexUnavailableError("Codex app-server did not start")
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
            prompt = build_codex_prompt(task, workspace)
            sequence += 1
            yield AgentEvent(
                event_type="backend.selected",
                task_id=task.task_id,
                backend="codex",
                sequence=sequence,
                title="Codex backend selected",
                summary=self.worker_id,
                model=self.settings.model or "",
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
                    event_type="turn.started",
                    task_id=task.task_id,
                    backend="codex",
                    sequence=sequence,
                    title="Codex turn started",
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
                        provider=self.settings.provider,
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
                iterator = self._client.notifications(thread_id).__aiter__()
                deadline = asyncio.get_running_loop().time() + self.settings.task_timeout_seconds
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    try:
                        notification = await asyncio.wait_for(anext(iterator), timeout=remaining)
                    except StopAsyncIteration:
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
                    if event.token_usage:
                        last_usage = dict(event.token_usage)
                        hard_reason = self.context_cost_governor.active_hard_limit_reason(
                            last_usage,
                            provider=self.settings.provider,
                            model=self.settings.model or "app-server-default",
                            context_window=int(last_usage.get("context_window") or 128_000),
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
            except asyncio.TimeoutError:
                if thread_id and turn_id:
                    await self._client.interrupt(thread_id=thread_id, turn_id=turn_id)
                notifications.append(
                    {"method": "turn/failed", "params": {"message": "Codex task timed out"}}
                )
                yield AgentEvent(
                    event_type="error",
                    task_id=task.task_id,
                    backend="codex",
                    sequence=sequence + 1,
                    status="failed",
                    title="Codex task timed out",
                    error="Codex task timed out",
                    thread_id=thread_id,
                    turn_id=turn_id,
                )
            except CodexError as exc:
                notifications.append({"method": "turn/failed", "params": {"message": str(exc)}})
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
                )
            finally:
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
                )

    async def cancel(self, task_id: str) -> None:
        active = self._active.get(str(task_id))
        if active is None or self._client is None:
            raise CodexExecutionError(f"No active Codex task: {task_id}")
        await self._client.interrupt(thread_id=active[0], turn_id=active[1])

    async def close(self) -> None:
        try:
            if self._client is not None:
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
