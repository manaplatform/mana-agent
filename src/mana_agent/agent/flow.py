from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from mana_agent.agent.selection import AgentPhase, select_task
from mana_agent.agent.task_context import TaskContext
from mana_agent.agent.verification import VerificationPlan, default_verification_plan


FLOW_ORDER: tuple[AgentPhase, ...] = (
    AgentPhase.DISCOVER,
    AgentPhase.SELECT,
    AgentPhase.READ,
    AgentPhase.ACT,
    AgentPhase.VERIFY,
    AgentPhase.SUMMARIZE,
)


@dataclass(frozen=True, slots=True)
class AgentFlow:
    context: TaskContext
    verification: VerificationPlan


def _short_goal(request: str) -> str:
    cleaned = " ".join(str(request or "").split())
    if not cleaned:
        return "Handle the current user request."
    return cleaned[:180]


def _candidate_search_terms(request: str, candidate_files: tuple[str, ...]) -> tuple[str, ...]:
    terms: list[str] = []
    terms.extend(candidate_files[:3])
    for match in re.findall(r"`([^`]{2,120})`", request):
        terms.append(match.strip())
    for match in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", request):
        if match.lower() not in {"the", "and", "for", "with", "this", "that", "from", "into", "when"}:
            terms.append(match)
        if len(terms) >= 8:
            break
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            unique.append(term)
    return tuple(unique[:8])


def _done_criteria(mode: str) -> tuple[str, ...]:
    if mode == "plan_only":
        return (
            "plan includes goal, requirements, implementation steps, verification, and done criteria",
            "no repository files are edited",
        )
    if mode == "analyze":
        return (
            "analysis is backed by repository evidence",
            "risks and recommended next steps are explicit",
        )
    if mode == "verify":
        return (
            "relevant checks are run or a concrete blocker is reported",
            "final response includes exact command outcomes",
        )
    if mode == "edit":
        return (
            "requested code changes are implemented",
            "changed files are reviewed",
            "relevant verification is run or a blocker is reported",
        )
    return ("the user receives a direct, useful, honest answer",)


def build_agent_flow(
    request: str,
    *,
    repo_root: str | Path | None = None,
    explicit_mode: str | None = None,
    candidate_files: tuple[str, ...] = (),
    files_read: tuple[str, ...] = (),
    flow_context: str | None = None,
) -> AgentFlow:
    selection = select_task(
        request,
        explicit_mode=explicit_mode,
        candidate_files=candidate_files,
        files_read=files_read,
    )
    verification = default_verification_plan(mode=selection.mode.value, request=request)
    context = TaskContext(
        request=request,
        mode=selection.mode.value,
        phase=selection.phase,
        goal=_short_goal(request),
        explicit_requirements=(request.strip(),) if request.strip() else (),
        repo_root=Path(repo_root).resolve() if repo_root is not None else None,
        candidate_files=selection.candidate_files,
        files_read=tuple(files_read),
        candidate_search_terms=_candidate_search_terms(request, selection.candidate_files),
        done_criteria=_done_criteria(selection.mode.value),
        verification_plan=verification.commands or verification.notes,
        flow_context=flow_context,
    )
    return AgentFlow(context=context, verification=verification)
