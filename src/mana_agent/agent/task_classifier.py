from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


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
_MUTATION_WORD_RE = re.compile(
    r"(?i)\b(task\s*:\s*)?(update|edit|write|fix|change|modify|replace|refactor|add|remove|delete|create)\b"
)
_VERIFY_WORD_RE = re.compile(r"(?i)\b(test|verify|check|pytest|lint|typecheck)\b")


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
    try:
        wanted_name = Path(text).name.lower()
        wanted_path = text.lower()
        for found in repo_root.rglob("*"):
            if not found.is_file() or found.name.lower() != wanted_name:
                continue
            rel = found.resolve().relative_to(repo_root).as_posix()
            if rel.lower() == wanted_path or found.name.lower() == wanted_name:
                return rel
    except OSError:
        pass
    direct = repo_root / text
    if direct.exists():
        return direct.resolve().relative_to(repo_root).as_posix()
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
    lowered = text.lower()
    explicit_files = _extract_target_files(text, repo_root=root, explicit_targets=target_files)
    sections = _extract_target_sections(text)
    mutation = bool(requires_edit) if requires_edit is not None else bool(_MUTATION_WORD_RE.search(text))
    verification_only = (not mutation) and bool(_VERIFY_WORD_RE.search(text))
    slash = lowered.startswith("/")
    scope: TaskScope
    architecture_source_update = bool(explicit_files) and "architecture" in lowered and "src" in lowered
    if architecture_source_update:
        scope = "multi_file"
    elif len(explicit_files) == 1 and sections:
        scope = "single_file_section"
    elif len(explicit_files) == 1:
        scope = "single_file"
    elif len(explicit_files) > 1:
        scope = "multi_file"
    elif any(token in lowered for token in ("repo", "project", "everywhere", "all files")):
        scope = "project_wide"
    else:
        scope = "unknown"

    if slash:
        task_type: TaskType = "slash_command"
    elif mutation:
        task_type = "mutation_required"
    elif verification_only:
        task_type = "verification_only"
    elif explicit_files or any(token in lowered for token in ("inspect", "read", "find", "where")):
        task_type = "repo_inspection"
    elif not text:
        task_type = "unknown_needs_clarification"
    else:
        task_type = "answer_only"

    needs_file_read = bool(explicit_files) and task_type in {"mutation_required", "repo_inspection", "verification_only"}
    needs_repo_search = (not explicit_files or architecture_source_update) and task_type in {"mutation_required", "repo_inspection"}
    confidence = 0.45
    if explicit_files:
        confidence += 0.25
    if sections:
        confidence += 0.15
    if mutation or verification_only:
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
        needs_verification=mutation or verification_only,
        scope=scope,
        confidence=confidence,
    )
    return decision


__all__ = ["TaskDecision", "TaskScope", "TaskType", "classify_task"]
