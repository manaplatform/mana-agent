from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mana_agent.multi_agent.runtime.auto_chat import AutoChatMode, classify_auto_chat_intent


class AgentPhase(str, Enum):
    DISCOVER = "discover"
    SELECT = "select"
    READ = "read"
    ACT = "act"
    VERIFY = "verify"
    SUMMARIZE = "summarize"


@dataclass(frozen=True, slots=True)
class SelectionResult:
    mode: AutoChatMode
    phase: AgentPhase
    candidate_files: tuple[str, ...] = ()


def select_mode(request: str, *, explicit_mode: str | None = None) -> AutoChatMode:
    """Resolve the task mode using the same bounded classifier as chat."""
    if explicit_mode:
        normalized = str(explicit_mode).strip().lower().replace("-", "_")
        for mode in AutoChatMode:
            if normalized in {mode.value, mode.name.lower()}:
                return mode
    return classify_auto_chat_intent(request)


def select_initial_phase(*, candidate_files: tuple[str, ...] = (), files_read: tuple[str, ...] = ()) -> AgentPhase:
    """Choose the prompt-visible phase from known task context."""
    if files_read:
        return AgentPhase.ACT
    if candidate_files:
        return AgentPhase.READ
    return AgentPhase.DISCOVER


def select_task(
    request: str,
    *,
    explicit_mode: str | None = None,
    candidate_files: tuple[str, ...] = (),
    files_read: tuple[str, ...] = (),
) -> SelectionResult:
    return SelectionResult(
        mode=select_mode(request, explicit_mode=explicit_mode),
        phase=select_initial_phase(candidate_files=candidate_files, files_read=files_read),
        candidate_files=tuple(candidate_files),
    )

