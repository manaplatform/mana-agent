"""Execution modes, structured plan models, and mode-based permission controls."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence
from pydantic import BaseModel, ConfigDict, Field


class ExecutionMode(str, Enum):
    """Execution modes for mana-agent.

    PLAN: Read-only exploration, context gathering, dependency inspection,
          and structured plan generation without modifying any files or executing
          destructive commands.
    ACT:  Authorized implementation phase that executes changes and actions
          according to an approved plan.
    """

    PLAN = "plan"
    ACT = "act"

    @property
    def is_plan(self) -> bool:
        return self == ExecutionMode.PLAN

    @property
    def is_act(self) -> bool:
        return self == ExecutionMode.ACT

    @property
    def allows_mutation(self) -> bool:
        return self == ExecutionMode.ACT

    @property
    def allows_destructive_commands(self) -> bool:
        return self == ExecutionMode.ACT


def normalize_execution_mode(value: Any, default: ExecutionMode = ExecutionMode.PLAN) -> ExecutionMode:
    """Normalize strings, enums, or AutoChatMode into an ExecutionMode."""
    if isinstance(value, ExecutionMode):
        return value
    text = str(getattr(value, "value", value) or "").strip().lower().replace("-", "_")
    if text in {"act", "edit", "execute", "run", "apply", "mutating", "auto"}:
        return ExecutionMode.ACT
    if text in {"plan", "plan_only", "planning", "read_only", "readonly", "inspect"}:
        return ExecutionMode.PLAN
    return default


class StructuredPlan(BaseModel):
    """Structured, actionable implementation plan."""

    model_config = ConfigDict(frozen=True, extra="allow")

    task_id: str
    goal: str
    objectives: list[str] = Field(default_factory=list)
    context_summary: str = ""
    affected_files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    implementation_steps: list[str] = Field(default_factory=list)
    potential_risks: list[str] = Field(default_factory=list)
    validation_strategies: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    approved: bool = False
    approved_by: str = ""
    approved_at: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def with_approval(self, approved: bool = True, approver: str = "user") -> StructuredPlan:
        """Return a copy of the plan with updated approval state."""
        return self.model_copy(
            update={
                "approved": approved,
                "approved_by": approver if approved else "",
                "approved_at": (
                    datetime.now(timezone.utc).isoformat() if approved else None
                ),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the structured plan as a dictionary."""
        return self.model_dump()

    def to_markdown(self) -> str:
        """Render a formatted markdown implementation plan."""
        lines = [
            f"# Implementation Plan: {self.goal}",
            "",
            "## 1. Objectives",
        ]
        if self.objectives:
            for obj in self.objectives:
                lines.append(f"- {obj}")
        else:
            lines.append(f"- {self.goal}")

        lines.extend([
            "",
            "## 2. Context & Dependencies",
            self.context_summary.strip() or "- Repository context explored; ready for implementation.",
            "",
            "### Identified Dependencies",
        ])
        if self.dependencies:
            for dep in self.dependencies:
                lines.append(f"- `{dep}`")
        else:
            lines.append("- No external dependencies or migrations required.")

        lines.extend([
            "",
            "## 3. Affected Files",
        ])
        if self.affected_files:
            for item in self.affected_files:
                lines.append(f"- `{item}`")
        else:
            lines.append("- Candidate files will be resolved during execution.")

        lines.extend([
            "",
            "## 4. Implementation Steps",
        ])
        if self.implementation_steps:
            for i, step in enumerate(self.implementation_steps, 1):
                lines.append(f"{i}. {step}")
        else:
            lines.append("1. Inspect current implementation\n2. Apply changes\n3. Verify results")

        lines.extend([
            "",
            "## 5. Potential Risks & Mitigations",
        ])
        if self.potential_risks:
            for risk in self.potential_risks:
                lines.append(f"- {risk}")
        else:
            lines.append("- Low risk change; adheres to established module patterns.")

        lines.extend([
            "",
            "## 6. Validation Strategies & Verification Commands",
        ])
        if self.validation_strategies:
            for val in self.validation_strategies:
                lines.append(f"- `{val}`")
        else:
            lines.append("- Run targeted unit tests and smoke tests.")

        lines.extend([
            "",
            "## 7. Approval Status",
            f"- Status: {'Approved' if self.approved else 'Pending User Approval'}",
        ])
        if self.approved and self.approved_by:
            lines.append(f"- Approved By: {self.approved_by}")
        if self.approved_at:
            lines.append(f"- Approved At: {self.approved_at}")

        lines.append("")
        return "\n".join(lines)


class PlanApprovalState(BaseModel):
    """Tracks plan approval status before execution in Act Mode."""

    plan_id: str
    approved: bool = False
    approved_by: str = ""
    approved_at: str | None = None
    rejection_reason: str = ""

    def approve(self, approver: str = "user") -> None:
        self.approved = True
        self.approved_by = approver
        self.approved_at = datetime.now(timezone.utc).isoformat()
        self.rejection_reason = ""

    def reject(self, reason: str = "") -> None:
        self.approved = False
        self.rejection_reason = reason
        self.approved_at = None
