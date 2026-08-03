"""Durable execution identity shared by consequential runtime boundaries."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DurableExecutionContext(BaseModel):
    """Trusted correlation data; model decisions are evidence, not task IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = ""
    branch_id: str = ""
    parent_task_id: str = ""
    checkpoint_id: str = ""
    execution_attempt_id: str = ""
    session_id: str = ""
    conversation_id: str = ""
    turn_id: str = ""
    source_decision_id: str = Field(min_length=1)
    originating_agent_id: str = "model_tool"

    def redacted(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "branch_id": self.branch_id,
            "checkpoint_id": self.checkpoint_id,
            "execution_attempt_id": self.execution_attempt_id,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "source_decision_id": self.source_decision_id,
            "originating_agent_id": self.originating_agent_id,
        }
