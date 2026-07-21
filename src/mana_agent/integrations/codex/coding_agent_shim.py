"""Compatibility surface that delegates complete coding turns to Codex.

Frontends still call the historical ``CodingAgent`` methods while Codex owns
planning, repository inspection, mutation, and task-specific verification in a
single app-server turn. Mana-Agent retains routing, workspace isolation,
permission enforcement, and result normalization.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Callable

from mana_agent.coding.models import AgentEvent, CodingTask, CodingTaskResult, WorkspaceContext
from mana_agent.integrations.codex.backend import CodexCodingBackend
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.multi_agent.worktrees import WorkspaceManager, WorkspaceStatus
from mana_agent.evals.recorder import record_current

BackendFactory = Callable[[], CodexCodingBackend]
WorkspaceManagerFactory = Callable[[], WorkspaceManager]


class CodexCodingAgentShim:
    """Preserve the frontend coding-agent API with Codex as sole executor."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        codex_settings: CodexSettings,
        repository_id: str | None = None,
        session_id: str = "",
        event_sink: Callable[..., Any] | None = None,
        backend_factory: BackendFactory | None = None,
        workspace_manager_factory: WorkspaceManagerFactory | None = None,
        workspace_task_id: str = "",
        resume_thread_id: str = "",
        **_legacy_kwargs: Any,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.codex_settings = codex_settings
        self.repository_id = str(repository_id or "").strip() or None
        self.session_id = str(session_id or "").strip()
        self.event_sink = event_sink
        self.workspace_task_id = str(workspace_task_id or "").strip()
        self.resume_thread_id = str(resume_thread_id or "").strip()
        self._backend_factory = backend_factory or (
            lambda: CodexCodingBackend(self.codex_settings, resume_thread_id=self.resume_thread_id)
        )
        self._workspace_manager_factory = workspace_manager_factory or (
            lambda: WorkspaceManager(
                self.repo_root,
                repository_id=self.repository_id,
                enabled=self.codex_settings.worktree_isolation,
            )
        )
        self._flow_results: dict[str, dict[str, Any]] = {}
        self._active_flow_id: str | None = None

    def preview_execution_checklist(
        self,
        request: str,
        *,
        flow_id: str | None = None,
        flow_context: str | None = None,
    ) -> dict[str, Any]:
        """Do not run a second planner before Codex's authoritative turn."""

        _ = (request, flow_context)
        return {
            "flow_id": flow_id,
            "prechecklist": None,
            "prechecklist_source": "codex_turn",
            "prechecklist_warning": "",
        }

    def generate(self, request: str, **kwargs: Any) -> dict[str, Any]:
        mode = str(kwargs.get("auto_chat_mode") or "").strip().lower()
        requires_write = mode not in {"plan", "plan_only"}
        return self._execute_turn(
            request,
            requires_repository_write=requires_write,
            flow_id=kwargs.get("flow_id"),
        )

    def generate_dir_mode(self, request: str, **kwargs: Any) -> dict[str, Any]:
        return self.generate(request, **kwargs)

    def generate_auto_execute(self, request: str, **kwargs: Any) -> dict[str, Any]:
        mode = str(kwargs.get("auto_chat_mode") or "").strip().lower()
        return self._execute_turn(
            request,
            requires_repository_write=mode not in {"plan", "plan_only"},
            flow_id=kwargs.get("flow_id"),
        )

    def flow_summary(self, flow_id: str | None = None) -> dict[str, Any] | None:
        selected = str(flow_id or self._active_flow_id or "").strip()
        return dict(self._flow_results[selected]) if selected in self._flow_results else None

    def get_active_flow_id(self) -> str | None:
        """Return the Codex thread-backed flow identifier used by frontends."""

        return self._active_flow_id

    def checkpoint_flow(self, flow_id: str | None = None) -> str | None:
        """Acknowledge an already persisted Codex result without running a planner."""

        selected = str(flow_id or self._active_flow_id or "").strip()
        return selected if selected in self._flow_results else None

    def reset_flow(self, flow_id: str | None = None) -> str | None:
        selected = str(flow_id or self._active_flow_id or "").strip()
        if selected:
            self._flow_results.pop(selected, None)
        if not flow_id or selected == self._active_flow_id:
            self._active_flow_id = None
        return selected or None

    def update_model(self, model_name: str) -> None:
        self.codex_settings = self.codex_settings.model_copy(
            update={"model": str(model_name or "").strip() or None}
        )

    def _tool_policy_for_request(self, _request: str, **_kwargs: Any) -> dict[str, Any]:
        """Reject legacy queue planning instead of manufacturing a tool policy."""

        raise RuntimeError(
            "Codex owns coding tool selection; no legacy QueueManager tool policy is available."
        )

    def set_tools_manager_orchestrator(self, _manager: Any) -> None:
        raise RuntimeError(
            "Codex owns coding planning and execution; the legacy tools orchestrator cannot be attached."
        )

    def _execute_turn(
        self,
        request: str,
        *,
        requires_repository_write: bool,
        flow_id: Any = None,
    ) -> dict[str, Any]:
        goal = str(request or "").strip()
        if not goal:
            raise ValueError("Codex coding request is required")
        task_id = f"codex_task_{uuid.uuid4().hex[:16]}"
        record_current(
            "codex.turn.started",
            {
                "task_id": task_id,
                "model": self.codex_settings.model,
                "sandbox": "workspaceWrite" if requires_repository_write else "readOnly",
                "approval_policy": self.codex_settings.approval_policy,
                "repository_identity": str(self.repo_root),
            },
        )
        task = CodingTask(
            task_id=task_id,
            goal=goal,
            requirements=(
                [
                    "Own the complete coding decision: inspect, plan, implement, and verify.",
                    "Do not invent repository changes when the requested outcome is underspecified.",
                    "Ask for required clarification instead of applying an arbitrary edit.",
                ]
                if requires_repository_write
                else [
                    "Inspect the repository and produce a decision-complete plan.",
                    "Do not modify repository files.",
                ]
            ),
            acceptance_criteria=[
                "The response directly satisfies the user's stated goal.",
                "All claims and changes are grounded in current repository evidence.",
            ],
            relevant_context=(
                "This is the authoritative Codex turn. There is no separate Mana coding planner "
                "or native coding executor before or after this turn."
            ),
            requires_repository_write=requires_repository_write,
        )

        manager: WorkspaceManager | None = None
        managed_workspace: Any = None
        if requires_repository_write and self.codex_settings.worktree_isolation:
            manager = self._workspace_manager_factory()
            workspace_task_id = self.workspace_task_id or task_id
            managed_workspace = manager.create_for_task(
                workspace_task_id,
                title=goal,
                assigned_agent_id="codex",
                session_id=self.session_id,
                reuse_existing=bool(self.workspace_task_id),
            )
            manager.transition(
                workspace_task_id,
                WorkspaceStatus.RUNNING,
                agent_id="codex",
                force=bool(self.workspace_task_id),
            )
            workspace = WorkspaceContext(
                repository_path=self.repo_root,
                worktree_path=Path(managed_workspace.worktree_path),
                branch_name=managed_workspace.branch_name,
                sandbox="workspaceWrite",
                approval_policy=self.codex_settings.approval_policy,
            )
        elif requires_repository_write:
            workspace = WorkspaceContext(
                repository_path=self.repo_root,
                worktree_path=self.repo_root,
                sandbox="workspaceWrite",
                approval_policy=self.codex_settings.approval_policy,
                allow_in_place_write=True,
            )
        else:
            workspace = WorkspaceContext(
                repository_path=self.repo_root,
                worktree_path=self.repo_root,
                sandbox="readOnly",
                approval_policy=self.codex_settings.approval_policy,
            )

        events: list[AgentEvent] = []
        backend = self._backend_factory()

        async def run() -> CodingTaskResult:
            try:
                async for event in backend.stream(task, workspace):
                    events.append(event)
                    self._emit_event(event)
                return backend.result_for(task_id)
            finally:
                await backend.close()

        try:
            result = asyncio.run(run())
        except Exception as exc:
            record_current("codex.turn.failed", {"task_id": task_id, "error_type": type(exc).__name__, "error": str(exc)})
            if manager is not None:
                manager.transition(
                    self.workspace_task_id or task_id,
                    WorkspaceStatus.FAILED,
                    agent_id="codex",
                    error=str(exc),
                )
            raise

        if manager is not None:
            if result.status == "completed":
                manager.transition(
                    self.workspace_task_id or task_id,
                    WorkspaceStatus.MERGE_CANDIDATE,
                    agent_id="codex",
                    notes=["Codex completed planning, implementation, and verification."],
                )
            elif result.status == "cancelled":
                manager.transition(self.workspace_task_id or task_id, WorkspaceStatus.INTERRUPTED, agent_id="codex")
            else:
                manager.transition(
                    self.workspace_task_id or task_id,
                    WorkspaceStatus.FAILED,
                    agent_id="codex",
                    error="; ".join(result.errors),
                )

        payload = self._result_payload(
            result,
            events=events,
            workspace_path=(str(workspace.worktree_path) if requires_repository_write else ""),
        )
        selected_flow_id = str(result.thread_id or flow_id or task_id).strip()
        payload["flow_id"] = selected_flow_id
        self._active_flow_id = selected_flow_id
        self._flow_results[selected_flow_id] = dict(payload)
        record_current("codex.turn.finished", {"task_id": task_id, "result": result.model_dump(mode="json"), "workspace_path": str(workspace.worktree_path)})
        return payload

    def _emit_event(self, event: AgentEvent) -> None:
        record_current(event.event_type, event.model_dump(mode="json"))
        if self.event_sink is None:
            return
        payload = event.model_dump(mode="json")
        try:
            self.event_sink(event.event_type, payload)
        except TypeError:
            self.event_sink(payload)

    @staticmethod
    def _result_payload(
        result: CodingTaskResult,
        *,
        events: list[AgentEvent],
        workspace_path: str,
    ) -> dict[str, Any]:
        terminal_reason = {
            "completed": "completed",
            "failed": "codex_failed",
            "cancelled": "codex_cancelled",
        }[result.status]
        answer = result.summary
        if result.status == "failed" and result.errors:
            answer = f"{result.summary} Reason: {result.errors[0]}".strip()
        return {
            "answer": answer,
            "backend": result.backend,
            "status": result.status,
            "run_status": result.status,
            "run_id": result.task_id,
            "auto_execute_terminal_reason": terminal_reason,
            "changed_files": list(result.changed_files),
            "warnings": [*result.warnings, *result.errors],
            "tests_run": list(result.tests_run),
            "tests_passed": result.tests_passed,
            "commands_run": list(result.commands_run),
            "thread_id": result.thread_id,
            "turn_id": result.turn_id,
            "branch_name": result.branch_name,
            "workspace_path": workspace_path,
            "trace": [event.model_dump(mode="json") for event in events],
            "actions_taken": [event.model_dump(mode="json") for event in events],
            "token_usage": result.token_usage,
        }


__all__ = ["CodexCodingAgentShim"]
