"""Strict model decision contracts for autonomous continuation and completion."""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import Field, model_validator

from mana_agent.execution_supervisor.models import StrictModel, stable_id
from mana_agent.coding_research_contracts import CodingResearchPlan, CodingResearchRecommendation


class ContinuationDecisionError(RuntimeError):
    """Raised when an autonomous continuation decision is missing, invalid, or violates policy."""


class ContinuationAction(str, Enum):
    CONTINUE = "CONTINUE"
    FINALIZE = "FINALIZE"
    REPLAN = "REPLAN"
    WAIT_FOR_USER = "WAIT_FOR_USER"
    BLOCKED = "BLOCKED"


class ContinuationDecision(StrictModel):
    """Authoritative model decision evaluated after meaningful execution steps.
    
    Mana-Agent operates in a single mode: 'model_directed'. The model controls
    semantic continuation decisions until the goal is satisfied, blocked, or
    requires user input.
    """

    decision_id: str = Field(default_factory=lambda: stable_id("continuation_dec"))
    task_id: str = Field(default="task_id", min_length=1)
    action: ContinuationAction
    goal_satisfied: bool
    progress_detected: bool
    evidence_summary: str = Field(min_length=1)
    remaining_work: str = Field(default="")
    user_message: str = Field(default="")
    public_summary: str = Field(default="", max_length=240)
    verification_passed: bool = False
    safe_to_continue: bool
    renewal_requested: bool = False
    renewal_reason: str = ""
    reason: str = Field(min_length=1)
    research_plan: CodingResearchPlan | None = None
    research_recommendation: CodingResearchRecommendation | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if isinstance(payload.get("action"), str):
            try:
                payload["action"] = ContinuationAction(payload["action"])
            except ValueError:
                pass
        if not payload.get("task_id"):
            payload["task_id"] = "task_id"
        if not payload.get("reason"):
            for candidate in ("reasoning", "rationale", "explanation", "justification"):
                if payload.get(candidate):
                    payload["reason"] = str(payload[candidate]).strip()
                    break
        if not payload.get("evidence_summary"):
            for candidate in ("evidence", "summary", "observed_evidence"):
                if payload.get(candidate):
                    payload["evidence_summary"] = str(payload[candidate]).strip()
                    break
        if not payload.get("evidence_summary") and payload.get("reason"):
            payload["evidence_summary"] = payload["reason"]
        if "progress_detected" not in payload:
            payload["progress_detected"] = True
        if not payload.get("remaining_work") and payload.get("suggested_next_steps"):
            steps = payload["suggested_next_steps"]
            payload["remaining_work"] = ", ".join(steps) if isinstance(steps, list) else str(steps)
        if not payload.get("task_id"):
            payload["task_id"] = "task_id"
        if not payload.get("reason"):
            payload["reason"] = payload.get("evidence_summary") or "continuation evaluated"

        # Filter out extra keys not defined on model to remain resilient to extraneous model outputs
        known_keys = set(cls.model_fields.keys())
        filtered = {k: v for k, v in payload.items() if k in known_keys}
        return filtered

    @model_validator(mode="after")
    def validate_semantic_invariants(self) -> "ContinuationDecision":
        if self.research_plan is not None and self.action not in {ContinuationAction.CONTINUE, ContinuationAction.REPLAN}:
            raise ValueError("research_plan requires CONTINUE or REPLAN")
        if self.action == ContinuationAction.FINALIZE:
            if not self.goal_satisfied:
                raise ValueError("FINALIZE action requires goal_satisfied=True")
            if not self.safe_to_continue:
                raise ValueError("FINALIZE action requires safe_to_continue=True")
        elif self.action in {ContinuationAction.CONTINUE, ContinuationAction.REPLAN}:
            if self.goal_satisfied:
                raise ValueError(f"{self.action.value} action cannot have goal_satisfied=True")
            if not self.safe_to_continue:
                raise ValueError(f"{self.action.value} action requires safe_to_continue=True")
        elif self.action == ContinuationAction.BLOCKED:
            if self.safe_to_continue:
                raise ValueError("BLOCKED action cannot have safe_to_continue=True")
            if self.goal_satisfied:
                raise ValueError("BLOCKED action cannot have goal_satisfied=True")
        elif self.action == ContinuationAction.WAIT_FOR_USER:
            if self.goal_satisfied:
                raise ValueError("WAIT_FOR_USER action cannot have goal_satisfied=True")

        if self.renewal_requested and not self.renewal_reason.strip():
            raise ValueError("renewal_requested=True requires non-empty renewal_reason")

        return self


class CodingContinuationDecision(ContinuationDecision):
    """Validate raw coding decisions before legacy alias/default normalization."""

    @model_validator(mode="before")
    @classmethod
    def require_explicit_decision(cls, data: Any) -> Any:
        if isinstance(data, dict):
            payload = dict(data)
            required = {"task_id", "action", "goal_satisfied", "progress_detected", "evidence_summary", "safe_to_continue", "reason"}
            if required - set(payload) or set(payload) - set(cls.model_fields):
                raise ValueError("Coding continuation must explicitly match the decision schema")
            for key in ("goal_satisfied", "progress_detected", "safe_to_continue"):
                if type(payload[key]) is not bool:
                    raise ValueError(f"Coding continuation {key} must be boolean")
            if isinstance(payload.get("action"), str):
                try:
                    payload["action"] = ContinuationAction(payload["action"])
                except ValueError:
                    pass
            return payload
        return data
