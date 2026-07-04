from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mana_agent.agent.selection import AgentPhase


@dataclass(frozen=True, slots=True)
class TaskContext:
    request: str
    mode: str = "answer_only"
    phase: AgentPhase = AgentPhase.DISCOVER
    goal: str = ""
    explicit_requirements: tuple[str, ...] = ()
    implicit_requirements: tuple[str, ...] = (
        "preserve existing behavior",
        "avoid unrelated changes",
        "verify when code changes",
    )
    repo_root: Path | None = None
    constraints: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    candidate_files: tuple[str, ...] = ()
    files_read: tuple[str, ...] = ()
    candidate_search_terms: tuple[str, ...] = ()
    done_criteria: tuple[str, ...] = ()
    verification_plan: tuple[str, ...] = ()
    flow_context: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


def _format_list(label: str, values: tuple[str, ...]) -> list[str]:
    if not values:
        return [f"- {label}: none"]
    return [f"- {label}: {', '.join(str(item) for item in values)}"]


def render_task_context(context: TaskContext) -> str:
    """Render the current task as a compact prompt layer."""
    goal = context.goal or context.request.strip()[:180] or "Handle the current user request."
    done_criteria = context.done_criteria or (
        "all explicit requirements are addressed",
        "repository changes are verified when verification is available",
        "final response reports changes and verification honestly",
    )
    lines = [
        "Current Task Context",
        "current_task:",
        f"  raw_user_request: {context.request.strip()}",
        f"- detected_mode: {context.mode}",
        f"- goal: {goal}",
        f"- current_phase: {context.phase.value}",
    ]
    if context.repo_root is not None:
        lines.append(f"- repo_root: {context.repo_root}")
    lines.extend(_format_list("explicit_requirements", context.explicit_requirements or (context.request.strip(),)))
    lines.extend(_format_list("implicit_requirements", context.implicit_requirements))
    lines.extend(_format_list("constraints", context.constraints))
    lines.extend(_format_list("assumptions", context.assumptions))
    lines.extend(_format_list("candidate_files", context.candidate_files))
    lines.extend(_format_list("files_already_read", context.files_read))
    lines.extend(_format_list("candidate_search_terms", context.candidate_search_terms))
    lines.extend(_format_list("done_criteria", done_criteria))
    lines.extend(_format_list("verification_plan", context.verification_plan))
    for key, value in sorted(context.metadata.items()):
        lines.append(f"- {key}: {value}")
    if context.flow_context:
        lines.extend(["", "Active Flow Context", context.flow_context.strip()])
    return "\n".join(lines).strip()
