"""Normalized-event wrapper around Mana-Agent's native coding runtime."""

from __future__ import annotations

import uuid
from typing import Any

from mana_agent.coding.live_events import coding_execution_context, publish_coding_event
from mana_agent.coding.models import AgentEvent
from mana_agent.multi_agent.runtime.coding_agent import CodingAgent as NativeCodingAgent


class InternalCodingAgentShim:
    name = "internal"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Gateway compatibility supplies both the repository root and its
        # selected working directory. The native agent's existing contract uses
        # ``repo_root`` and resolves all tools beneath it.
        kwargs.pop("project_root", None)
        self._agent = NativeCodingAgent(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)

    def generate(self, request: str, **kwargs: Any) -> Any:
        return self._execute("generate", request, kwargs)

    def generate_dir_mode(self, request: str, **kwargs: Any) -> Any:
        return self._execute("generate_dir_mode", request, kwargs)

    def generate_auto_execute(self, request: str, **kwargs: Any) -> Any:
        return self._execute("generate_auto_execute", request, kwargs)

    def _execute(self, method: str, request: str, kwargs: dict[str, Any]) -> Any:
        task_id = f"internal_task_{uuid.uuid4().hex[:16]}"
        model = str(getattr(getattr(self._agent, "ask_agent", None), "model", "") or "")
        with coding_execution_context(task_id=task_id, backend="internal", model=model) as state:
            self._emit(state, "backend.selected", "Internal backend selected", model=model)
            self._emit(state, "turn.started", "Internal coding turn started", model=model)
            try:
                result = getattr(self._agent, method)(request, **kwargs)
            except BaseException as exc:
                self._emit(
                    state,
                    "error",
                    "Internal coding turn failed",
                    status="cancelled" if type(exc).__name__ == "CancelledError" else "failed",
                    error=str(exc),
                    model=model,
                )
                raise
            self._emit(state, "turn.completed", "Internal coding turn completed", status="success", model=model)
            if isinstance(result, dict):
                result.setdefault("backend", "internal")
                result.setdefault("run_id", task_id)
            return result

    @staticmethod
    def _emit(
        state: dict[str, object],
        event_type: str,
        title: str,
        *,
        status: str = "running",
        error: str = "",
        model: str = "",
    ) -> None:
        sequence = int(state.get("sequence") or 0) + 1
        state["sequence"] = sequence
        publish_coding_event(AgentEvent(
            event_type=event_type,
            task_id=str(state["task_id"]),
            backend="internal",
            sequence=sequence,
            status=status,  # type: ignore[arg-type]
            title=title,
            error=error,
            model=model,
        ))


__all__ = ["InternalCodingAgentShim"]
