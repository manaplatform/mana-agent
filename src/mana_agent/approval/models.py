from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ApprovalMode(str, Enum):
    """Supported execution approval modes."""

    DEFAULT = "default"
    AUTO_APPROVE_SAFE = "auto_approve_safe"
    FULL_ACCESS = "full_access"

    @classmethod
    def from_str(cls, value: str | None) -> "ApprovalMode":
        if not value:
            return cls.DEFAULT
        normalized = str(value).strip().lower().replace("-", "_")
        for mode in cls:
            if mode.value == normalized:
                return mode
        raise ValueError(
            f"Invalid approval mode: {value!r}. Supported modes: "
            f"{', '.join(m.value for m in cls)}"
        )


class ApprovalDecision(BaseModel):
    """Authoritative decision produced by the Centralized Approval Engine."""

    approved: bool
    mode: ApprovalMode
    source: str  # "full_access", "safe_auto_approval", "human", "policy_allowed", "policy_denied"
    interactive: bool = False
    reason: str = ""
    approval_id: str = ""
    inbox_item_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalRequest(BaseModel):
    """Context and payload for an action or tool requiring an approval evaluation."""

    action_type: str = ""  # "transactional", "shell", "git", "file", "browser", "email", "server", "api", "plan", "steering", "budget"
    action_id: str = ""
    command: str = ""
    tool_name: str = ""
    intent: Any = None
    risk_level: str = "medium"
    task_id: str = ""
    attempt_id: str = ""
    execution_id: str = ""
    agent_id: str = ""
    reversibility: str = "reversible"
    blast_radius: str = "workspace"
    metadata: dict[str, Any] = Field(default_factory=dict)
