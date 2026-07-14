"""Bounded, task-specific parent-to-agent delegation contracts."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field


class DelegationRequest(BaseModel):
    user_goal: str
    delegated_objective: str
    known_repository_facts: list[str] = Field(default_factory=list)
    canonical_target_paths: list[str] = Field(default_factory=list)
    evidence_references: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    max_tool_calls: int = Field(ge=1, le=32)
    max_tokens: int = Field(ge=128, le=32000)
    out_of_bounds: list[str] = Field(default_factory=list)
    expected_result: str
    success_conditions: list[str] = Field(min_length=1)
    stop_conditions: list[str] = Field(min_length=1)
    parent_agent_id: str
    task_id: str
    root_task_id: str

    def ephemeral_prompt(self) -> str:
        """Serialize only task-specific state; stable roles live in system prompts."""
        return json.dumps(self.model_dump(), ensure_ascii=False, sort_keys=True)


class DelegationResult(BaseModel):
    status: Literal["completed", "blocked", "escalation_requested"]
    summary: str
    evidence_references: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    requested_action: str = ""


__all__ = ["DelegationRequest", "DelegationResult"]
