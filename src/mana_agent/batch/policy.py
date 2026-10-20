"""Task-scoped execution intent. Capability checks never select a fallback."""

from __future__ import annotations

from typing import Literal
from contextlib import contextmanager
from contextvars import ContextVar

from pydantic import BaseModel, ConfigDict, model_validator


class TaskExecutionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["auto", "realtime", "batch"] = "auto"
    required: bool = False
    source_message_id: str = ""
    decision_id: str = ""

    @model_validator(mode="after")
    def validate_requirement(self) -> "TaskExecutionPolicy":
        if self.required and self.mode != "batch":
            raise ValueError("required Batch execution must select mode=batch")
        if self.mode == "batch" and not self.required:
            raise ValueError("OpenAI Batch selection requires an explicit task requirement")
        return self


_execution_policy: ContextVar[TaskExecutionPolicy | None] = ContextVar("mana_task_execution_policy", default=None)
_tool_call_id: ContextVar[str] = ContextVar("mana_batch_tool_call_id", default="")


def current_execution_policy() -> TaskExecutionPolicy | None:
    return _execution_policy.get()


def current_batch_tool_call_id() -> str:
    return _tool_call_id.get()


@contextmanager
def batch_tool_call_context(call_id: str):
    token = _tool_call_id.set(call_id)
    try:
        yield
    finally:
        _tool_call_id.reset(token)


@contextmanager
def execution_policy_context(policy: TaskExecutionPolicy):
    token = _execution_policy.set(policy)
    try:
        yield
    finally:
        _execution_policy.reset(token)


class PendingBatchOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation_id: str
    batch_id: str
    task_id: str
    phase: str
    checkpoint_id: str = ""


class BatchPending(RuntimeError):
    def __init__(self, operation: PendingBatchOperation) -> None:
        self.operation = operation
        super().__init__(f"Waiting for OpenAI Batch {operation.batch_id or operation.operation_id}")


class BatchHumanInput(RuntimeError):
    def __init__(self, question: str, inbox_item_id: str = "") -> None:
        self.inbox_item_id = inbox_item_id
        super().__init__(question)
