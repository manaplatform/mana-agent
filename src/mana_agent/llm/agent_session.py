from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field


AgentRouteKind = Literal["coding_agent", "queue_manager", "classic", "readonly"]


class AgentRoute(BaseModel):
    """Internal routing record for a single agent turn."""

    route: AgentRouteKind
    reason: str = ""

    @property
    def uses_coding_agent(self) -> bool:
        return self.route == "coding_agent"


class AgentSession(BaseModel):
    """Normalized execution context shared by CodingAgent, QueueManager, and tools."""

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    flow_id: str | None = None
    run_id: str
    repo_root: str
    index_dir: str | None = None
    index_dirs: list[str] | None = None
    tool_policy: dict[str, Any] = Field(default_factory=dict)
    execution_backend: str = "local"

    @classmethod
    def from_queue_run(
        cls,
        *,
        repo_root: str | Path,
        run_id: str,
        flow_id: str | None = None,
        index_dir: str | Path | None = None,
        index_dirs: Sequence[str | Path] | None = None,
        tool_policy: dict[str, Any] | None = None,
        execution_backend: str = "local",
    ) -> "AgentSession":
        return cls(
            flow_id=flow_id or None,
            run_id=str(run_id),
            repo_root=str(Path(repo_root).resolve()),
            index_dir=str(Path(index_dir).resolve()) if index_dir is not None else None,
            index_dirs=[str(Path(item).resolve()) for item in (index_dirs or []) if str(item).strip()] or None,
            tool_policy=dict(tool_policy or {}),
            execution_backend=str(execution_backend or "local"),
        )

    def with_tool_policy(self, tool_policy: dict[str, Any] | None) -> "AgentSession":
        return self.model_copy(update={"tool_policy": dict(tool_policy or {})})


def route_for_turn(
    *,
    coding_agent_available: bool,
    agent_tools: bool,
    coding_agent_is_custom: bool = False,
    readonly: bool = False,
    reason: str = "",
) -> AgentRoute:
    if readonly:
        return AgentRoute(route="readonly", reason=reason or "read-only mode")
    if coding_agent_available and (agent_tools or coding_agent_is_custom):
        return AgentRoute(route="coding_agent", reason=reason or "coding agent available with tool routing")
    return AgentRoute(route="classic", reason=reason or "coding agent unavailable")


__all__ = ["AgentRoute", "AgentRouteKind", "AgentSession", "route_for_turn"]
