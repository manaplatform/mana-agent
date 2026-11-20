from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

from mana_agent.multi_agent.runtime.edit_scope import resolve_repo_path


TaskType = Literal[
    "answer_only",
    "repo_inspection",
    "mutation_required",
    "verification_only",
    "continue_previous_flow",
    "slash_command",
    "unknown_needs_clarification",
]
TaskScope = Literal["single_file", "single_file_section", "multi_file", "project_wide", "unknown"]

_FILE_RE = re.compile(r"(?i)(?<![\w/.-])([A-Za-z0-9_.\-/]+\.[A-Za-z0-9_]+)(?![\w/.-])")
_HEADING_RE = re.compile(r"(?m)(#{1,6})\s+([^\n#][^\n]*)")
@dataclass(frozen=True, slots=True)
class TaskDecision:
    task_type: TaskType
    target_files: tuple[str, ...] = ()
    target_sections: tuple[str, ...] = ()
    mutation_intent: str = ""
    constraints: tuple[str, ...] = ()
    needs_repo_search: bool = False
    needs_file_read: bool = False
    needs_mutation: bool = False
    needs_verification: bool = False
    scope: TaskScope = "unknown"
    confidence: float = 0.0
    trace: tuple[dict[str, object], ...] = field(default_factory=tuple)

    def as_trace_row(self) -> dict[str, object]:
        return {
            "layer": "task_classifier",
            "decision": self.task_type,
            "reason": self.mutation_intent or "classified user request",
            "target_files": list(self.target_files),
            "target_sections": list(self.target_sections),
            "scope": self.scope,
            "confidence": self.confidence,
        }


def _repo_relative(path: str, *, repo_root: Path) -> str:
    text = str(path or "").strip().replace("\\", "/").lstrip("./")
    if not text:
        return ""
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(repo_root).as_posix()
        except Exception:
            return candidate.as_posix()
    resolution = resolve_repo_path(repo_root, text)
    if resolution.ok:
        return resolution.resolved_path
    return text


def _extract_target_files(request: str, *, repo_root: Path, explicit_targets: Sequence[str]) -> tuple[str, ...]:
    found: list[str] = []
    for raw in explicit_targets:
        rel = _repo_relative(str(raw), repo_root=repo_root)
        if rel:
            found.append(rel)
    for match in _FILE_RE.findall(request):
        rel = _repo_relative(match, repo_root=repo_root)
        if rel:
            found.append(rel)
    return tuple(dict.fromkeys(found))


def _extract_target_sections(request: str) -> tuple[str, ...]:
    sections: list[str] = []
    for _marks, title in _HEADING_RE.findall(request):
        cleaned = " ".join(title.strip().strip("#").split())
        if cleaned:
            sections.append(cleaned)
    return tuple(dict.fromkeys(sections))


def classify_task(
    request: str,
    *,
    repo_root: str | Path,
    target_files: Sequence[str] = (),
    requires_edit: bool | None = None,
) -> TaskDecision:
    root = Path(repo_root).resolve()
    text = str(request or "").strip()
    if requires_edit is None and not text.startswith("/"):
        raise ValueError("Task classification requires an explicit repository effect")
    explicit_files = _extract_target_files(text, repo_root=root, explicit_targets=target_files)
    sections = _extract_target_sections(text)
    mutation = requires_edit is True
    scope: TaskScope = "unknown"
    if len(explicit_files) == 1:
        scope = "single_file_section" if sections else "single_file"
    elif len(explicit_files) > 1:
        scope = "multi_file"

    if text.startswith("/"):
        task_type: TaskType = "slash_command"
    elif mutation:
        task_type = "mutation_required"
    else:
        task_type = "repo_inspection"

    needs_file_read = bool(explicit_files) and task_type in {"mutation_required", "repo_inspection", "verification_only"}
    needs_repo_search = (not explicit_files) and task_type in {"mutation_required", "repo_inspection"}
    confidence = 0.45
    if explicit_files:
        confidence += 0.25
    if sections:
        confidence += 0.15
    if mutation:
        confidence += 0.1
    confidence = min(confidence, 0.95)

    if mutation and explicit_files:
        section_text = f" {' / '.join(sections)} section" if sections else ""
        mutation_intent = f"update {', '.join(explicit_files)}{section_text}".strip()
    elif mutation:
        mutation_intent = "apply requested repository change"
    else:
        mutation_intent = ""

    decision = TaskDecision(
        task_type=task_type,
        target_files=explicit_files,
        target_sections=sections,
        mutation_intent=mutation_intent,
        needs_repo_search=needs_repo_search,
        needs_file_read=needs_file_read,
        needs_mutation=mutation,
        needs_verification=mutation,
        scope=scope,
        confidence=confidence,
    )
    return decision


__all__ = ["TaskDecision", "TaskScope", "TaskType", "classify_task"]
