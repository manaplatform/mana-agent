from mana_agent.multi_agent.core.types import ToolResult

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class FileMutationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    old_path: str = ""
    operation: Literal["create", "edit", "delete", "move"]
    before_sha256: str = ""
    after_sha256: str = ""
    diff: str = ""
    added_lines: int = 0
    removed_lines: int = 0
    binary: bool = False
    status: Literal["applied", "rolled_back", "failed", "partial", "unknown"] = "applied"


class ToolOperationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    tool_call_id: str
    operation_id: str
    operation_attempt_id: str
    task_id: str
    root_task_id: str = ""
    parent_task_id: str = ""
    execution_id: str = ""
    parent_execution_id: str = ""
    requested_by_agent_id: str = ""
    agent_id: str = ""
    execution_level: Literal[3] = 3
    decision_id: str = ""
    tool_name: str = ""
    attempt_id: str = ""
    attempt_generation: int = 0
    status: Literal["success", "failed", "cancelled", "timed_out", "waiting_approval", "interrupted"]
    result: dict[str, Any] = Field(default_factory=dict, repr=False)
    duration_ms: int = 0
    started_at: datetime | None = None
    ended_at: datetime | None = None
    exit_code: int | None = None
    termination_reason: str = ""
    mutations: list[FileMutationEvidence] = Field(default_factory=list)
    retained_bytes: int = 0
    omitted_bytes: int = 0
    trace_complete: bool = True
    output_references: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

__all__ = ["ToolResult", "ToolOperationResult", "FileMutationEvidence"]
