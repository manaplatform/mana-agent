"""Configuration for Universal Model-Directed Autonomous Execution."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class AutoRenewConfig(BaseModel):
    """Configuration for accounting-based resource budget auto-renewals."""

    enabled: bool = True
    strategy: Literal["accounting_based", "fixed", "disabled"] = "accounting_based"
    renew_model_turns: int = Field(default=40, ge=1)
    renew_tool_calls: int = Field(default=60, ge=1)
    max_auto_renewals: int = Field(default=3, ge=0)
    require_progress: bool = True
    respect_account_budget: bool = True
    require_approval_above_budget: bool = True
    stop_on_no_progress: bool = True


class AutonomousExecutionConfig(BaseModel):
    """Configuration for the unified model-directed execution controller.
    
    Mana-Agent operates in a single, universal mode: 'model_directed'.
    Execution continues in an autonomous loop until an explicit model decision
    (FINALIZE, WAIT_FOR_USER, or BLOCKED) is validated with accounting enforcement.
    """

    mode: Literal["model_directed"] = "model_directed"
    max_model_turns: int = Field(default=120, ge=1)
    max_tool_calls: int = Field(default=200, ge=1)
    require_verification: bool = True
    auto_renew: AutoRenewConfig = Field(default_factory=AutoRenewConfig)
    max_no_progress_cycles: int = Field(default=3, ge=1)
    soft_limit_turns_headroom: int = Field(default=2, ge=0)
    soft_limit_tools_headroom: int = Field(default=5, ge=0)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AutonomousExecutionConfig":
        if not data:
            return cls()
        return cls.model_validate(data)
