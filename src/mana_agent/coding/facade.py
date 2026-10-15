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
            if self._active_backend is not None or self._task_in_progress:
                raise RuntimeError(
                    f"Cannot switch coding engine to '{norm}' while a coding task is actively executing."
                )
            self._active_mode = norm  # type: ignore[assignment]
            logger.info("Coding engine switched to: %s", self._active_mode)

    def cancel(self, task_id: str | None = None) -> bool:
        with self._execution_lock:
            active_id = task_id or self._active_task_id or ""
            c_ext = super().cancel(task_id)
            c_int = False
            if self._internal_engine is not None and active_id:
                try:
                    asyncio.run(self._internal_engine.cancel(active_id))
                    c_int = True
                except Exception:
                    pass
            self._task_in_progress = False
            self._active_task_id = None
            return c_ext or c_int

    def close(self) -> None:
        super().close()
        if self._internal_engine is not None:
            try:
                asyncio.run(self._internal_engine.close())
            except Exception:
                pass

    def generate(self, request: str, **kwargs: Any) -> dict[str, Any]:
        return self._dispatch("generate", request, kwargs)

    def generate_auto_execute(self, request: str, **kwargs: Any) -> dict[str, Any]:
        return self._dispatch("generate_auto_execute", request, kwargs)

    def generate_dir_mode(self, request: str, **kwargs: Any) -> dict[str, Any]:
        return self._dispatch("generate_dir_mode", request, kwargs)

    def _dispatch(self, method: str, request: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        with self._execution_lock:
            if self._task_in_progress:
                raise RuntimeError(
                    f"A coding task ({self._active_task_id}) is already executing in this workspace."
                )
            self._task_in_progress = True
            task_id = str(kwargs.get("gateway_task_id") or f"task_{uuid.uuid4().hex[:12]}")
            self._active_task_id = task_id

        try:
            if self._active_mode == "external":
                func = getattr(super(), method)
                result = func(request, **kwargs)
                if isinstance(result, dict):
                    result.setdefault("backend", "external")
                return result

            # Internal Native Execution
            return self._execute_internal(task_id, request, kwargs)
        finally:
            with self._execution_lock:
                self._task_in_progress = False
                self._active_task_id = None

    def _execute_internal(self, task_id: str, request: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        allowed_files: list[str] = []
        prechecklist = kwargs.get("prechecklist_payload")
        if isinstance(prechecklist, dict):
            pre_data = prechecklist.get("prechecklist")
            if isinstance(pre_data, dict):
                allowed_files.extend(pre_data.get("target_files") or [])

        coding_task = CodingTask(
            task_id=task_id,
            goal=request,
            allowed_files=allowed_files,
            requires_repository_write=True,
        )
        workspace = WorkspaceContext(
            repository_path=self.repo_root,
            worktree_path=self.working_directory,
            working_directory=self.working_directory,
            sandbox="workspaceWrite",
            allow_in_place_write=True,
        )

        if not self._internal_engine.model and self.codex_settings.model:
            self._internal_engine.model = self.codex_settings.model
        if not self._internal_engine.api_key and getattr(self.codex_settings, "api_key", None):
            self._internal_engine.api_key = self.codex_settings.api_key
        if not self._internal_engine.base_url and getattr(self.codex_settings, "base_url", None):
            self._internal_engine.base_url = self.codex_settings.base_url
        if not self._internal_engine.provider and getattr(self.codex_settings, "provider", None):
            self._internal_engine.provider = self.codex_settings.provider

        try:
            res: CodingTaskResult = asyncio.run(self._internal_engine.execute(coding_task, workspace))
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
            "status": res.status,
            "changed_files": res.changed_files,
            "commands_run": res.commands_run,
            "warnings": res.warnings,
            "errors": res.errors,
            "backend": "internal",
            "run_id": task_id,
            "flow_id": self._active_flow_id,
        }


__all__ = ["UnifiedCodingAgentFacade"]
