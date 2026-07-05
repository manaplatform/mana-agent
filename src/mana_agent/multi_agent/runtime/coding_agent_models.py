from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

from pydantic import BaseModel, Field


class FlowStep(BaseModel):
    """Represents a planned step with tooling guidance and execution status."""

    id: str
    title: str
    reason: str
    status: Literal["pending", "in_progress", "done", "blocked"] = "pending"
    requires_tools: list[str] = Field(default_factory=list)


class FlowChecklist(BaseModel):
    """Structured plan capturing objective, constraints, acceptance criteria, and steps."""

    objective: str
    requires_edit: bool = False
    target_files: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    acceptance: list[str] = Field(default_factory=list)
    steps: list[FlowStep] = Field(default_factory=list)
    next_action: str = ""


class ExecutionDecision(BaseModel):
    phase: Literal["discover", "inspect", "edit", "verify", "answer", "blocked"]
    tool_call_allowed: bool
    why: str


CodingAgentPhase = Literal[
    "understand",
    "plan",
    "search",
    "read",
    "patch",
    "verify",
    "revise",
    "finalize",
]


class CodingAgentStateMachine(BaseModel):
    """Small explicit phase machine used by coding-agent orchestration tests."""

    phase: CodingAgentPhase = "understand"
    files_read: set[str] = Field(default_factory=set)
    patched_files: set[str] = Field(default_factory=set)
    verification_run: bool = False
    transitions: list[dict[str, str]] = Field(default_factory=list)

    _ORDER: ClassVar[tuple[CodingAgentPhase, ...]] = (
        "understand",
        "plan",
        "search",
        "read",
        "patch",
        "verify",
        "revise",
        "finalize",
    )

    def mark_read(self, path: str) -> None:
        cleaned = str(path or "").strip()
        if cleaned:
            self.files_read.add(cleaned)

    def can_patch(self, targets: list[str]) -> bool:
        return all(str(item).strip() in self.files_read for item in targets if str(item).strip())

    def transition(self, next_phase: CodingAgentPhase, *, reason: str = "", targets: list[str] | None = None) -> None:
        if next_phase == "patch" and not self.can_patch(targets or []):
            raise ValueError("cannot enter patch phase before target files are read")
        if next_phase not in self._ORDER:
            raise ValueError(f"unknown coding-agent phase: {next_phase}")
        self.transitions.append({"from_phase": self.phase, "to_phase": next_phase, "reason": reason})
        self.phase = next_phase


class DynamicReadPolicy(BaseModel):
    """LLM-selected read policy for one coding turn (full-auto only)."""

    read_budget: int
    read_line_window: int
    reason: str = ""


class AskAgentLike(Protocol):
    tools: list[Any]

    def ask(self, question: str, **kwargs: Any) -> Any:  # pragma: no cover
        ...


def as_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): as_jsonable(v) for k, v in obj.items()}
    return obj
