"""Validated, model-selected execution scope for one coding turn.

The model owns semantic choices (intent, scope, evidence, mutation, verification,
and delegation).  Runtime code only validates and enforces the resulting
contract; it never widens the scope when the contract is absent or invalid.
"""

from __future__ import annotations

from enum import IntEnum
import hashlib
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, field_validator, model_validator

from mana_agent.multi_agent.runtime.edit_scope import resolve_repo_path


class ScopeLevel(IntEnum):
    DIRECT = 0
    BOUNDED = 1
    IMPACT = 2
    BROAD = 3


TaskType = Literal["answer", "inspect", "edit", "verify", "plan"]
RiskLevel = Literal["low", "medium", "high"]
SearchScope = Literal["none", "named_files", "bounded", "dependency", "repository"]
MutationStrategy = Literal["none", "single_patch", "bounded_patch", "multi_file_patch"]
VerificationStrategy = Literal["none", "artifact", "targeted", "related", "full"]


_KNOWN_TOOL_FAMILIES = {
    "read",
    "search",
    "symbols",
    "mutation",
    "verification",
    "git_read",
    "agents",
}


class ExecutionScopeDecision(BaseModel):
    """Complete semantic execution contract selected once by a model."""

    decision_id: str
    task_type: TaskType
    scope_level: ScopeLevel
    complexity: Literal["trivial", "small", "medium", "large"]
    risk: RiskLevel
    explicit_target_files: list[str] = Field(default_factory=list)
    related_files: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    allowed_tool_families: list[str] = Field(default_factory=list)
    search_scope: SearchScope
    max_search_operations: int = Field(ge=0, le=16)
    max_unique_file_reads: int = Field(ge=0, le=64)
    mutation_strategy: MutationStrategy
    verification_strategy: VerificationStrategy
    verification_commands: list[list[str]] = Field(default_factory=list, max_length=12)
    delegated_agents: list[str] = Field(default_factory=list, max_length=8)
    stop_conditions: list[str] = Field(min_length=1, max_length=12)
    confidence: float = Field(ge=0.0, le=1.0)
    escalation_reason: str = ""
    unresolved_questions: list[str] = Field(default_factory=list, max_length=12)
    out_of_bounds: list[str] = Field(default_factory=list, max_length=24)

    @field_validator(
        "decision_id",
        "explicit_target_files",
        "related_files",
        "required_evidence",
        "allowed_tool_families",
        "delegated_agents",
        "stop_conditions",
        "unresolved_questions",
        "out_of_bounds",
        mode="before",
    )
    @classmethod
    def _reject_blank_values(cls, value: Any) -> Any:
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("value must not be blank")
            return value.strip()
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value]
            if any(not item for item in cleaned):
                raise ValueError("list values must not be blank")
            return list(dict.fromkeys(cleaned))
        return value

    @model_validator(mode="after")
    def _validate_contract(self) -> "ExecutionScopeDecision":
        unknown = sorted(set(self.allowed_tool_families).difference(_KNOWN_TOOL_FAMILIES))
        if unknown:
            raise ValueError(f"unknown tool families: {', '.join(unknown)}")
        if self.scope_level == ScopeLevel.DIRECT:
            if not self.explicit_target_files:
                raise ValueError("level 0 requires explicit target files")
            if self.search_scope != "none" or self.max_search_operations != 0:
                raise ValueError("level 0 cannot search")
            if self.delegated_agents:
                raise ValueError("level 0 cannot delegate")
        if self.search_scope == "none" and self.max_search_operations:
            raise ValueError("search budget must be zero when search_scope is none")
        if self.task_type == "edit" and self.mutation_strategy == "none":
            raise ValueError("edit task requires a mutation strategy")
        if self.task_type != "edit" and self.mutation_strategy != "none":
            raise ValueError("non-edit task cannot select a mutation strategy")
        if self.scope_level < ScopeLevel.IMPACT and self.verification_strategy == "full":
            raise ValueError("full verification requires impact or broad scope")
        if self.verification_strategy in {"targeted", "related", "full"} and not self.verification_commands:
            raise ValueError("selected verification strategy requires explicit argv commands")
        if self.verification_strategy in {"none", "artifact"} and self.verification_commands:
            raise ValueError("none/artifact verification cannot include shell commands")
        for command in self.verification_commands:
            if not command or any(not str(arg).strip() for arg in command):
                raise ValueError("verification commands must be non-empty argv lists")
        if self.scope_level > ScopeLevel.DIRECT and not self.escalation_reason:
            raise ValueError("non-direct scope requires an escalation reason")
        return self

    @property
    def all_evidence_files(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([*self.explicit_target_files, *self.related_files]))

    def trace_row(self) -> dict[str, Any]:
        return {
            "layer": "execution_scope",
            "decision_id": self.decision_id,
            "task_type": self.task_type,
            "scope_level": int(self.scope_level),
            "risk": self.risk,
            "explicit_target_files": list(self.explicit_target_files),
            "related_files": list(self.related_files),
            "search_scope": self.search_scope,
            "budgets": {
                "searches": self.max_search_operations,
                "unique_file_reads": self.max_unique_file_reads,
            },
            "mutation_strategy": self.mutation_strategy,
            "verification_strategy": self.verification_strategy,
            "verification_commands": [list(command) for command in self.verification_commands],
            "delegated_agents": list(self.delegated_agents),
            "confidence": self.confidence,
            "escalation_reason": self.escalation_reason,
        }


class ExecutionScopeDecisionError(RuntimeError):
    """Raised before tools run when a scope decision cannot be validated."""


class ScopeEscalationRequest(BaseModel):
    current_decision_id: str
    requested_level: ScopeLevel
    requested_by_agent_id: str
    missing_evidence: list[str] = Field(min_length=1, max_length=12)
    evidence_references: list[str] = Field(min_length=1, max_length=24)
    reason: str = Field(min_length=1, max_length=600)
    requested_search_operations: int = Field(ge=0, le=16)
    requested_unique_file_reads: int = Field(ge=1, le=64)


def approve_scope_escalation(
    current: ExecutionScopeDecision,
    request: ScopeEscalationRequest | dict[str, Any],
    *,
    approved_by_agent_id: str,
) -> ExecutionScopeDecision:
    """Apply one parent-approved escalation step; workers cannot widen directly."""

    escalation = request if isinstance(request, ScopeEscalationRequest) else ScopeEscalationRequest.model_validate(request)
    if escalation.current_decision_id != current.decision_id:
        raise ExecutionScopeDecisionError("scope escalation references a stale decision")
    if not str(approved_by_agent_id).strip():
        raise ExecutionScopeDecisionError("scope escalation requires a parent approver")
    if int(escalation.requested_level) != int(current.scope_level) + 1:
        raise ExecutionScopeDecisionError("scope escalation must advance exactly one level")
    search_scope: SearchScope = {
        ScopeLevel.BOUNDED: "bounded",
        ScopeLevel.IMPACT: "dependency",
        ScopeLevel.BROAD: "repository",
    }[escalation.requested_level]
    payload = current.model_dump(mode="python")
    payload.update(
        {
            "decision_id": f"{current.decision_id}:L{int(escalation.requested_level)}",
            "scope_level": escalation.requested_level,
            "search_scope": search_scope,
            "max_search_operations": escalation.requested_search_operations,
            "max_unique_file_reads": escalation.requested_unique_file_reads,
            "escalation_reason": escalation.reason,
            "unresolved_questions": list(escalation.missing_evidence),
            "allowed_tool_families": list(dict.fromkeys([*current.allowed_tool_families, "search"])),
        }
    )
    return ExecutionScopeDecision.model_validate(payload)


def validate_execution_scope(
    value: ExecutionScopeDecision | dict[str, Any],
    *,
    repo_root: str | Path,
) -> ExecutionScopeDecision:
    """Validate schema, canonicalize paths, and enforce repository boundaries."""

    try:
        decision = value if isinstance(value, ExecutionScopeDecision) else ExecutionScopeDecision.model_validate(value)
    except Exception as exc:
        raise ExecutionScopeDecisionError(
            f"Model decision failed: execution_scope. No action executed. Reason: {exc}"
        ) from exc

    root = Path(repo_root).resolve()

    def _canonicalize(paths: Sequence[str], *, must_exist: bool) -> list[str]:
        output: list[str] = []
        for raw in paths:
            resolution = resolve_repo_path(root, raw)
            if resolution.ok:
                canonical = resolution.resolved_path
            elif must_exist:
                raise ExecutionScopeDecisionError(
                    "Model decision failed: execution_scope. No action executed. "
                    f"Reason: selected file cannot be resolved safely: {raw} ({resolution.reason})."
                )
            else:
                candidate = str(raw).replace("\\", "/").lstrip("./")
                absolute = (root / candidate).resolve()
                try:
                    absolute.relative_to(root)
                except ValueError as exc:
                    raise ExecutionScopeDecisionError(
                        "Model decision failed: execution_scope. No action executed. "
                        f"Reason: selected path is outside the repository: {raw}."
                    ) from exc
                canonical = candidate
            folded = canonical.casefold()
            if all(item.casefold() != folded for item in output):
                output.append(canonical)
        return output

    explicit = _canonicalize(decision.explicit_target_files, must_exist=decision.task_type != "edit")
    related = _canonicalize(decision.related_files, must_exist=True)
    if len(set([path.casefold() for path in [*explicit, *related]])) > decision.max_unique_file_reads:
        raise ExecutionScopeDecisionError(
            "Model decision failed: execution_scope. No action executed. "
            "Reason: selected evidence files exceed the unique-read budget."
        )
    return decision.model_copy(update={"explicit_target_files": explicit, "related_files": related})


def scope_from_explicit_contract(
    *,
    user_goal: str,
    target_files: Sequence[str],
    requires_edit: bool,
) -> ExecutionScopeDecision:
    """Adapt an already-structured caller decision for legacy API clients.

    This adapter does not inspect user text.  It preserves only choices the
    caller made explicitly (`requires_edit` and `target_files`) and selects the
    smallest mechanically valid budget.  New model callers should pass the
    full :class:`ExecutionScopeDecision` instead.
    """

    targets = list(dict.fromkeys(str(path).strip() for path in target_files if str(path).strip()))
    direct = len(targets) == 1
    has_targets = bool(targets)
    digest = hashlib.sha1(
        f"{requires_edit}|{','.join(targets)}|{user_goal}".encode("utf-8")
    ).hexdigest()[:12]
    return ExecutionScopeDecision(
        decision_id=f"scope_caller_{digest}",
        task_type="edit" if requires_edit else "inspect",
        scope_level=ScopeLevel.DIRECT if direct else ScopeLevel.BOUNDED,
        complexity="small" if has_targets else "medium",
        risk="low" if has_targets else "medium",
        explicit_target_files=targets,
        related_files=[],
        required_evidence=["current content of each caller-selected target"] if targets else ["model-selected target location"],
        allowed_tool_families=(
            ["read", "mutation", "verification"] if requires_edit else ["read", "search"]
        ),
        search_scope="none" if has_targets else "bounded",
        max_search_operations=0 if has_targets else 1,
        max_unique_file_reads=max(1, len(targets)) if has_targets else 8,
        mutation_strategy=("single_patch" if len(targets) <= 1 else "bounded_patch") if requires_edit else "none",
        verification_strategy="artifact" if requires_edit else "none",
        verification_commands=[],
        delegated_agents=[],
        stop_conditions=["caller-selected deliverables are satisfied", "required verification is complete or validly skipped"],
        confidence=1.0,
        escalation_reason=(
            "" if direct else "caller selected a bounded multi-file change" if has_targets else "caller did not select an explicit target"
        ),
        out_of_bounds=["files outside the caller-selected targets"],
    )


__all__ = [
    "ExecutionScopeDecision",
    "ExecutionScopeDecisionError",
    "ScopeLevel",
    "ScopeEscalationRequest",
    "approve_scope_escalation",
    "scope_from_explicit_contract",
    "validate_execution_scope",
]
