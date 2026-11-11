from mana_agent.multi_agent.core.types import ToolRequest

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class ToolOperationRequest(BaseModel):
    """An invocation, never a permission grant. Authority is bound by the host."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    task_id: str = Field(min_length=1)
    root_task_id: str = Field(min_length=1)
    parent_task_id: str = ""
    execution_id: str = Field(min_length=1)
    parent_execution_id: str = Field(min_length=1)
    requested_by_agent_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    execution_level: Literal[3] = 3
    decision_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    operation_id: str = Field(min_length=1)
    operation_attempt_id: str = Field(min_length=1)
    attempt_id: str = ""
    attempt_generation: int = Field(default=0, ge=0)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict, repr=False)
    required_capability: str = Field(min_length=1)
    authorization_scope: str = Field(min_length=1)
    workspace_root: str = Field(min_length=1)
    timeout_seconds: float = Field(default=120, gt=0)
    deadline: datetime | None = None
    cancellation_id: str = Field(min_length=1)
    expected_result_type: Literal["structured"] = "structured"

__all__ = ["ToolRequest", "ToolOperationRequest"]
