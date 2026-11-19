"""Unified Coding Agent Facade.

Provides a unified interface to both Internal and External coding engines,
subclassing and preserving full backward compatibility with CodexCodingAgentShim
call surface across CLI, TUI, Chat Gateway, and Autonomous workflows.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mana_agent.coding.internal_engine import InternalCodingEngine
from mana_agent.batch.policy import TaskExecutionPolicy
from mana_agent.coding.models import (
    CodingEngineMode,
    CodingTask,
    CodingTaskResult,
    WorkspaceContext,
)
from mana_agent.integrations.codex.coding_agent_shim import CodexCodingAgentShim
from mana_agent.integrations.codex.config import CodexSettings

logger = logging.getLogger(__name__)


class UnifiedCodingAgentFacade(CodexCodingAgentShim):
    """Unified facade managing selectable Internal and External coding execution engines."""

    supports_gateway_task_identity = True

    def __init__(
        self,
        *,
        repo_root: str | Path,
        working_directory: str | Path | None = None,
        default_mode: CodingEngineMode = "external",
        codex_settings: CodexSettings | None = None,
        internal_engine: InternalCodingEngine | None = None,
        repository_id: str | None = None,
        session_id: str = "",
        event_sink: Callable[..., Any] | None = None,
        workspace_id: str | None = None,
        context_cost_governor: Any | None = None,
        **kwargs: Any,
    ) -> None:
        effective_settings = codex_settings or CodexSettings()
        super().__init__(
            repo_root=repo_root,
            working_directory=working_directory,
            codex_settings=effective_settings,
            repository_id=repository_id,
            session_id=session_id,
            event_sink=event_sink,
            workspace_id=workspace_id,
            context_cost_governor=context_cost_governor,
            **kwargs,
        )
        norm_mode: CodingEngineMode = "internal" if str(default_mode).strip().lower() == "internal" else "external"
        self._active_mode: CodingEngineMode = norm_mode
        self._internal_engine = internal_engine or InternalCodingEngine()
        self._task_in_progress = False
        self._active_task_id: str | None = None
        self._running_internal_engine = None
        self._assignment_workers = {}
        self._assignment_parent = None

    def for_assignment(self, *, root: Path, task_id: str = ""):
        """Task-local execution state; the lane remains the capacity authority."""
        import copy
        from mana_agent.services.activity_context import current_activity_context
        task_id = task_id or str(getattr(current_activity_context(), "task_id", "") or "")
        if not task_id:
            raise PermissionError("Task-local coding execution requires an admitted task identity")
        # The admitted lane has already selected and isolated this exact checkout.
        settings = self.codex_settings.model_copy(update={"worktree_isolation": False})
        worker = UnifiedCodingAgentFacade(repo_root=root, working_directory=root,
            default_mode=self._active_mode, codex_settings=settings,
            internal_engine=copy.copy(self._internal_engine), repository_id=self.repository_id,
            session_id=f"{self.session_id or 'assignment'}-{task_id}", event_sink=self.event_sink, workspace_id=self.workspace_id,
            context_cost_governor=self.context_cost_governor, routing_authority=self.routing_authority)
        worker._assignment_parent = self
        activity = current_activity_context()
        worker._assignment_scope = {task_id, str(getattr(activity, "root_task_id", "") or ""),
                                    str(getattr(activity, "parent_task_id", "") or "")}
        with self._execution_lock:
            self._assignment_workers[id(worker)] = worker
        return worker

    @property
    def active_engine_mode(self) -> CodingEngineMode:
        return self._active_mode

    @active_engine_mode.setter
    def active_engine_mode(self, mode: CodingEngineMode) -> None:
        self.set_engine_mode(mode)

    def set_engine_mode(self, mode: str) -> None:
        """Explicitly switch active coding engine mode, preventing changes during active runs."""
        norm = str(mode or "").strip().lower()
        if norm == "codex":
            norm = "external"
        if norm not in {"internal", "external"}:
            raise ValueError(f"Invalid coding engine mode '{mode}'. Must be 'internal' or 'external'.")

        with self._execution_lock:
            if self._active_backend is not None or self._task_in_progress or self._assignment_workers:
                raise RuntimeError(
                    f"Cannot switch coding engine to '{norm}' while a coding task is actively executing."
                )
            self._active_mode = norm  # type: ignore[assignment]
            logger.info("Coding engine switched to: %s", self._active_mode)

    def cancel(self, task_id: str | None = None) -> bool:
        with self._execution_lock:
            workers = list(self._assignment_workers.values())
        cancelled = any([worker.cancel() for worker in workers
                         if task_id is None or task_id in worker._assignment_scope])
        with self._execution_lock:
            if task_id and self._active_task_id and task_id != self._active_task_id:
                return cancelled
            active_id = task_id or self._active_task_id or ""
            c_ext = super().cancel(task_id)
            c_int = False
            engine = self._running_internal_engine or self._internal_engine
            if engine is not None and active_id:
                try:
                    asyncio.run(engine.cancel(active_id))
                    c_int = True
                except Exception:
                    pass
            self._task_in_progress = False
            self._active_task_id = None
            return cancelled or c_ext or c_int

    def close(self) -> None:
        with self._execution_lock:
            workers = list(self._assignment_workers.values())
        for worker in workers:
            worker.close()
        super().close()
        engine = self._running_internal_engine or self._internal_engine
        if engine is not None:
            try:
                asyncio.run(engine.close())
            except Exception:
                pass
        if self._assignment_parent is not None:
            with self._assignment_parent._execution_lock:
                self._assignment_parent._assignment_workers.pop(id(self), None)

    def generate(self, request: str, **kwargs: Any) -> dict[str, Any]:
        return self._dispatch("generate", request, kwargs)

    def generate_auto_execute(self, request: str, **kwargs: Any) -> dict[str, Any]:
        return self._dispatch("generate_auto_execute", request, kwargs)

    def generate_dir_mode(self, request: str, **kwargs: Any) -> dict[str, Any]:
        return self._dispatch("generate_dir_mode", request, kwargs)

    def _dispatch(self, method: str, request: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        if kwargs.get("coding_mode_proposal") and (
            kwargs.get("execution_mode") != "read_only" or not kwargs.get("coding_mode_files")
        ):
            raise PermissionError("Coding patch proposals require read-only execution and explicit file targets")
        with self._execution_lock:
            if self._task_in_progress:
                raise RuntimeError(
                    f"A coding task ({self._active_task_id}) is already executing in this workspace."
                )
            self._task_in_progress = True
            task_id = str(kwargs.get("gateway_task_id") or f"task_{uuid.uuid4().hex[:12]}")
            self._active_task_id = task_id
            prior_directory = self.working_directory

        try:
            runtime_board = (kwargs.get("execution_runtime") or {}).get("taskboard") or (kwargs.get("batch_runtime") or {}).get("taskboard")
            if runtime_board is not None:
                from mana_agent.repository_evidence.tasks import validate_assignment
                validate_assignment(runtime_board, task_id)
            policy = TaskExecutionPolicy.model_validate(kwargs.get("execution_policy") or {})
            if kwargs.get("repository_investigation") and policy.required and self._active_mode == "external":
                raise PermissionError("The selected external investigator cannot execute an internal-only Batch policy")
            if kwargs.get("coding_mode_proposal") and (self._active_mode != "external" or policy.required):
                proposal_root = Path(kwargs["coding_mode_root"]).resolve()
                if proposal_root not in {self.repo_root.resolve(), self.working_directory.resolve()}:
                    raise PermissionError("The coding proposal belongs to a different checkout")
                self.working_directory = proposal_root
            if self._active_mode == "external" and not policy.required:
                charge_model = (kwargs.get("execution_runtime") or {}).get("charge_model")
                if callable(charge_model):
                    charge_model()
                func = getattr(super(), method)
                result = func(request, **kwargs)
                if runtime_board is not None:
                    validate_assignment(runtime_board, task_id)
                if isinstance(result, dict):
                    result.setdefault("backend", "external")
                    runtime = kwargs.get("execution_runtime") or kwargs.get("batch_runtime") or {}
                    from mana_agent.services.activity_context import current_activity_context
                    activity = current_activity_context()
                    board = runtime.get("taskboard") or getattr(activity, "taskboard", None)
                    if board is not None and result.get("changed_files"):
                        paths = []
                        for value in result["changed_files"]:
                            target = (self.working_directory / value).resolve()
                            if not target.is_relative_to(self.working_directory.resolve()):
                                raise PermissionError("The coding result contains a change outside its assigned checkout")
                            paths.append(target.relative_to(self.working_directory.resolve()).as_posix())
                        board.evidence_board(root=self.working_directory).invalidate_sources(paths,
                            task_id=task_id, reason="Observed external coding result")
                        scope = board.get_task(task_id).coding_assignment.get("allowed_scope", [])
                        if scope and "." not in scope and any(not any(path == selected or path.startswith(selected + "/")
                                for selected in scope) for path in paths):
                            raise PermissionError("The coding result changed files outside its assigned scope; reconcile the retained checkout")
                return result

            # Internal Native Execution
            return self._execute_internal(task_id, request, kwargs)
        finally:
            with self._execution_lock:
                if kwargs.get("coding_mode_proposal"):
                    self.working_directory = prior_directory
                self._task_in_progress = False
                self._active_task_id = None

    def _execute_internal(self, task_id: str, request: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        import copy
        engine = copy.copy(self._internal_engine)
        engine.context_cost_governor = self.context_cost_governor
        engine.extra_tools = dict(self._internal_engine.extra_tools)
        engine._active_sessions = {}
        engine._cancellation_events = {}
        engine._pending_model_calls = {}
        engine._tool_runners = {}
        self._running_internal_engine = engine
        runtime = {**(kwargs.get("execution_runtime") or {}), **(kwargs.get("batch_runtime") or {})}
        for name in ("execution_registry", "tool_authorize", "execution_supervisor", "taskboard", "batch_manager", "batch_authorize",
                     "batch_authority", "batch_autonomous_config", "model", "provider", "charge_model", "charge_tool"):
            if name in runtime:
                setattr(engine, name, runtime[name])
        engine.max_tool_calls = min(engine.max_tool_calls, int(runtime.get("remaining_tool_calls", engine.max_tool_calls)))
        engine.max_model_turns = int(runtime.get("remaining_model_turns", 120))
        if "max_no_progress_cycles" in runtime:
            engine.max_no_progress_cycles = runtime["max_no_progress_cycles"]
        from mana_agent.llm.engine import NativeLLMEngine
        if isinstance(engine.llm_client, NativeLLMEngine) and (engine.llm_client.provider, engine.llm_client.model) != (engine.provider, engine.model):
            engine.llm_client = None
            engine.api_key = None
            engine.base_url = None
        evidence_tools = []
        if runtime.get("taskboard") is not None:
            from mana_agent.repository_evidence.context_tools import context_tools
            evidence_tools = context_tools(runtime["taskboard"].evidence_board(root=self.working_directory))
        for tool in [*runtime.get("context_tools", []), *evidence_tools]:
            def invoke_context(_tool=tool, **arguments):
                return _tool.call(arguments)
            invoke_context.input_model = tool.input_model
            from mana_agent.tools.contracts import ToolContract
            invoke_context.tool_contract = ToolContract(name=tool.name, description=tool.description,
                input_schema=tool.input_model.model_json_schema(), output_schema={}, error_format={},
                required_capability="context", effect="read_only", repository_operation=False)
            engine.extra_tools[tool.name] = invoke_context
        policy = TaskExecutionPolicy.model_validate(kwargs.get("execution_policy") or {})
        allowed_files: list[str] = list(kwargs.get("coding_mode_files") or [])
        prechecklist = kwargs.get("prechecklist_payload")
        if isinstance(prechecklist, dict):
            pre_data = prechecklist.get("prechecklist")
            if isinstance(pre_data, dict):
                allowed_files.extend(pre_data.get("target_files") or [])

        coding_task = CodingTask(
            task_id=task_id,
            goal=str(kwargs.get("coding_original_goal") or request),
            allowed_files=allowed_files,
            requires_repository_write=kwargs.get("execution_mode") not in {"plan", "read_only"},
            execution_policy=policy,
            execution_mode="batch" if policy.required else str(kwargs.get("execution_mode") or "auto"),
            relevant_context=str(request if kwargs.get("repository_investigation") else kwargs.get("relevant_context") or ""),
            context={"batch_resume": kwargs.get("batch_resume") or {},
                     "coding_resume_context": kwargs.get("coding_resume_context") or {},
                     "coding_research_context": kwargs.get("coding_research_context") or {},
                     "mode_patch_proposal": bool(kwargs.get("coding_mode_proposal")),
                     "repository_investigation": kwargs.get("repository_investigation"),
                     "coding_continuation_id": kwargs.get("coding_continuation_id") or "",
                     "conversation_context_in_request": bool(kwargs.get("conversation_context_in_request"))},
        )
        workspace = WorkspaceContext(
            repository_path=self.repo_root,
            worktree_path=self.working_directory,
            working_directory=self.working_directory,
            sandbox="readOnly" if kwargs.get("execution_mode") in {"plan", "read_only"} else "workspaceWrite",
            allow_in_place_write=True,
        )

        if not engine.model and self.codex_settings.model:
            engine.model = self.codex_settings.model
        if not engine.api_key and getattr(self.codex_settings, "api_key", None):
            engine.api_key = self.codex_settings.api_key
        if not engine.base_url and getattr(self.codex_settings, "base_url", None):
            engine.base_url = self.codex_settings.base_url
        if not engine.provider and getattr(self.codex_settings, "provider", None):
            engine.provider = self.codex_settings.provider

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                from contextvars import copy_context
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    res: CodingTaskResult = pool.submit(
                        copy_context().run, asyncio.run, engine.execute(coding_task, workspace)
                    ).result()
            else:
                res: CodingTaskResult = asyncio.run(engine.execute(coding_task, workspace))
        except Exception as exc:
            return {
                "answer": f"Internal coding execution failed: {exc}",
                "status": "failed",
                "error_code": "INTERNAL_EXECUTION_FAILED",
                "changed_files": [],
                "warnings": [str(exc)],
                "backend": "internal",
                "run_id": task_id,
            }

        return {
            "answer": res.summary,
            "producing_agent_id": res.worker_id,
            "status": res.status,
            "pending_operation": res.pending_operation.model_dump(mode="json") if res.pending_operation else None,
            "human_input": res.human_input,
            "debugging_report": res.debugging_report.model_dump(mode="json") if res.debugging_report else None,
            "continuation_context": res.continuation_context,
            "investigation_result": res.investigation_result,
            "missing_evidence": res.missing_evidence,
            "goal_satisfied": False if res.status == "waiting" else None,
            "changed_files": res.changed_files,
            "commands_run": res.commands_run,
            "tests_run": res.tests_run,
            "tests_passed": res.tests_passed,
            "duration_breakdown": res.duration_breakdown,
            "error_code": "INTERNAL_EXECUTION_FAILED" if res.status in {"failed", "cancelled"} else None,
            "warnings": res.warnings,
            "errors": res.errors,
            "backend": "internal",
            "run_id": task_id,
            "flow_id": self._active_flow_id,
            "workspace_path": str(workspace.worktree_path),
        }


__all__ = ["UnifiedCodingAgentFacade"]
