"""Execution modes, structured plan models, and mode-based permission controls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler, GetJsonSchemaHandler, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import PydanticCustomError, core_schema

from mana_agent.human_inbox.models import canonical_digest


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
    DEBUG = "debug"
    AUTO = "auto"

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
    if text in {"act", "edit", "execute", "run", "apply", "mutating"}:
        return ExecutionMode.ACT
    if text in {"debug", "auto"}:
        return ExecutionMode(text)
    if text in {"plan", "plan_only", "planning", "read_only", "readonly", "inspect"}:
        return ExecutionMode.PLAN
    return default


class ModePhase(str, Enum):
    UNDERSTANDING = "understanding"
    INVESTIGATING = "investigating"
    CLARIFICATION_REQUIRED = "clarification_required"
    DRAFTING_PLAN = "drafting_plan"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


def repository_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or path.as_posix() != value or value == "."
            or "\\" in value or ":" in value or any(ord(c) < 32 for c in value)
            or any(p == ".." or p in {".git", ".mana", ".ssh", "secrets.toml", "credentials.json"}
                   or p.startswith(".env") for p in path.parts)):
        raise PydanticCustomError("mode_repository_path", "use an unprotected, canonical repository-relative path")
    return value


class ModeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CodingWorkflowDecision(ModeContract):
    """The model's coding choice. Access is still checked at dispatch and at each write."""

    mode: Literal["auto", "debug"]
    objective: str = Field(pattern=r"\S", max_length=12000)
    reason: str = Field(pattern=r"\S", max_length=240)
    investigation_scope: str = Field(pattern=r"\S", max_length=2000)
    expected_result: str = Field(pattern=r"\S", max_length=2000)
    risk_and_uncertainty: str = Field(pattern=r"\S", max_length=2000)
    verification_requirements: tuple[str, ...] = Field(min_length=1, max_length=12)
    requested_effect: Literal["read", "write"]
    clarification: str = Field(max_length=1000)

    @model_validator(mode="after")
    def validate_requirements(self):
        if any(not item.strip() or len(item) > 1000 for item in self.verification_requirements):
            raise ValueError("verification requirements must be bounded nonempty statements")
        if self.mode == "auto" and self.requested_effect != "write":
            raise ValueError("AUTO requires an explicit implementation request")
        return self


@dataclass(frozen=True)
class DecisionEntries:
    """Advertise fixed-key records to models and retain dictionaries in saved state."""

    key: str
    value: str

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        mapping = handler(source)
        record = core_schema.typed_dict_schema({
            self.key: core_schema.typed_dict_field(mapping["keys_schema"]),
            self.value: core_schema.typed_dict_field(mapping["values_schema"]),
        }, extra_behavior="forbid")
        entries = core_schema.no_info_after_validator_function(self._to_mapping, core_schema.list_schema(record))
        return core_schema.union_schema([entries, mapping])

    def __get_pydantic_json_schema__(self, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        entries, mapping = schema["choices"]
        return handler(entries if handler.mode == "validation" else mapping)

    def _to_mapping(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        mapping = {record[self.key]: record[self.value] for record in records}
        if len(mapping) != len(records):
            raise PydanticCustomError("mode_duplicate_entry", f"entries must use unique {self.key} values")
        return mapping


class VerificationCommand(ModeContract):
    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str = "."
    expected: Literal["pass", "fail"] = "pass"
    output_marker: str = ""
    image: str = ""

    @model_validator(mode="after")
    def validate_command(self):
        if self.cwd != ".":
            repository_relative_path(self.cwd)
        if not self.argv[0].strip() or any("\x00" in arg for arg in self.argv):
            raise PydanticCustomError("mode_verification_argv", "verification requires literal argv")
        if self.expected == "fail" and not self.output_marker:
            raise PydanticCustomError("mode_verification_diagnostic", "verification expecting failure requires an output_marker")
        if self.image and (self.image.startswith("-") or any(c.isspace() for c in self.image)):
            raise PydanticCustomError("mode_verification_image", "verification image must be a local image name")
        return self


FileOperations = Annotated[tuple[Literal["add", "update", "delete"], ...], Field(min_length=1)]


class PlanStep(ModeContract):
    step_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    instruction: str = Field(min_length=1)
    depends_on: tuple[str, ...] = ()
    changes: Annotated[dict[str, FileOperations], DecisionEntries("path", "operations")] = Field(default_factory=dict)
    verification: tuple[VerificationCommand, ...] = ()
    acceptance_criteria: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_scope(self):
        if not self.changes and not self.verification:
            raise PydanticCustomError("mode_step_scope", "each step needs file operations or a verification command")
        for path in self.changes:
            repository_relative_path(path)
        return self


class PlanDraft(ModeContract):
    objectives: tuple[str, ...] = Field(min_length=1)
    findings: str = Field(min_length=1)
    scope: tuple[str, ...] = Field(min_length=1)
    exclusions: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    recovery: str = ""
    steps: tuple[PlanStep, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_order(self):
        seen = set()
        for step in self.steps:
            if step.step_id in seen or not set(step.depends_on).issubset(seen):
                raise PydanticCustomError("mode_step_order", "step IDs must be unique and dependencies must refer to earlier steps")
            seen.add(step.step_id)
        if not any(check.expected == "pass" for step in self.steps for check in step.verification):
            raise PydanticCustomError("mode_passing_check", "include at least one verification command that expects to pass")
        return self


class PlanRevision(ModeContract):
    plan_id: str
    revision: int = Field(ge=1)
    session_id: str
    workspace_id: str
    repository_id: str
    repository_root: str
    goal: str
    constraints: dict[str, Any]
    base_file_digests: dict[str, str] = Field(default_factory=dict)
    base_checkout: str = ""
    coding_backend: Literal["", "internal", "external"] = ""
    execution_policy: dict[str, Any] = Field(default_factory=dict)
    draft: PlanDraft

    @property
    def digest(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))

    def to_markdown(self) -> str:
        lines = [f"# Implementation Plan: {self.goal}", "", f"Revision {self.revision} · `{self.digest}`",
                 "", "## Findings", self.draft.findings]
        if self.coding_backend:
            lines.extend(["", f"Coding backend: {self.coding_backend}"])
        for title, entries in (
            ("Objectives", self.draft.objectives), ("Scope", self.draft.scope),
            ("Out of scope", self.draft.exclusions), ("Dependencies", self.draft.dependencies),
            ("Risks", self.draft.risks), ("Assumptions", self.draft.assumptions),
            ("User decisions", tuple(f"{key}: {value}" for key, value in self.constraints.items())),
            ("Recovery", (self.draft.recovery,) if self.draft.recovery else ()),
        ):
            if entries:
                lines.extend(["", f"## {title}", *(f"- {entry}" for entry in entries)])
        lines.extend(["", "## Steps"])
        for step in self.draft.steps:
            lines.extend(["", f"### {step.step_id} · {step.instruction}"])
            if step.depends_on:
                lines.append(f"Depends on {', '.join(step.depends_on)}")
            lines.extend(f"- `{path}` ({', '.join(ops)})" for path, ops in step.changes.items())
            for check in step.verification:
                image = f", image {check.image}" if check.image else ""
                marker = f", diagnostic {check.output_marker}" if check.output_marker else ""
                lines.append(f"- Verify `{list(check.argv)}` in `{check.cwd}` (expect {check.expected}{image}{marker})")
            lines.extend(f"- Acceptance: {criterion}" for criterion in step.acceptance_criteria)
        return "\n".join(lines) + "\n"


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

    @property
    def is_approved(self) -> bool:
        if self.approved:
            return True
        from mana_agent.approval.policy import get_approval_policy

        return get_approval_policy().is_full_access()

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

    @property
    def is_approved(self) -> bool:
        if self.approved:
            return True
        from mana_agent.approval.policy import get_approval_policy

        return get_approval_policy().is_full_access()
