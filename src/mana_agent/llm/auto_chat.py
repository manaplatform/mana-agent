"""Bounded normal-chat routing helpers.

Slash commands are handled by the CLI before this module is consulted. These
helpers only classify and constrain ordinary chat messages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import re
from pathlib import Path
from typing import Any


class AutoChatMode(Enum):
    ANSWER_ONLY = "answer_only"
    PLAN_ONLY = "plan_only"
    EDIT = "edit"
    REVIEW = "review"
    VERIFY = "verify"
    ANALYZE = "analyze"


AUTO_MAX_SEARCH_QUERIES = 4
AUTO_MAX_CANDIDATE_FILES = 12
AUTO_MAX_FILES_TO_READ = 6
AUTO_MAX_LINES_PER_FILE = 220
AUTO_MAX_DISCOVERY_ROUNDS = 2
AUTO_MAX_TOOL_CALLS_BEFORE_DECISION = 8

MUTATION_TOOLS = frozenset({"edit_file", "multi_edit_file", "apply_patch", "apply_patch_batch", "write_file", "create_file", "delete_file", "move_file"})
READ_TOOLS = [
    "semantic_search",
    "read_file",
    "repo_batch_read",
    "chunk_file",
    "list_tools",
    "ls",
    "repo_search",
    "repo_batch_search",
    "list_files",
    "find_symbols",
    "call_graph",
    "git_status",
    "git_diff",
    "tool_contracts",
    "read_skill",
]

EDIT_TOOLS = [
    *READ_TOOLS,
    "run_command",
    "verify_project",
    "edit_file",
    "multi_edit_file",
    "apply_patch",
    "apply_patch_batch",
    "create_file",
    "write_file",
    "delete_file",
]

VERIFY_TOOLS = ["run_command", "run_script_once", "verify_project", "git_status", "git_diff", "read_file", "repo_batch_read", "ls", "list_files"]
REVIEW_TOOLS = ["git_status", "git_diff", "read_file", "repo_batch_read", "ls", "list_files", "repo_search", "repo_batch_search"]


_EDIT_RE = re.compile(
    r"\b(fix|add|implement|change|update|rename|refactor|create|delete|remove|migrate|patch|edit|modify|rewrite|build)\b",
    re.IGNORECASE,
)
_PLAN_RE = re.compile(
    r"\b(plan|prompt|strategy|approach|design|architecture|roadmap|implementation steps|how should we implement)\b",
    re.IGNORECASE,
)
_EXECUTE_PLAN_RE = re.compile(
    r"\b(implement|execute|run|apply)\s+(?:the\s+|last\s+|that\s+|current\s+)?plan\b",
    re.IGNORECASE,
)
_REVIEW_RE = re.compile(r"\b(review|check my changes|inspect diff|what is wrong|quality|code review)\b", re.IGNORECASE)
_VERIFY_RE = re.compile(r"\b(run tests|verify|run checks|pytest|lint|typecheck|test this|check this)\b", re.IGNORECASE)
_ANALYZE_RE = re.compile(r"\b(analyze project|analyze module|full analysis|project analysis|report)\b", re.IGNORECASE)
_ANSWER_RE = re.compile(r"\b(where|what|why|explain|how does|find|locate|show me|which file)\b", re.IGNORECASE)
_FOLLOWUP_RE = re.compile(r"^\s*(continue|do it|now implement|verify|fix it|proceed|go ahead)\s*[.!]?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class AutoChatLimits:
    max_search_queries: int = AUTO_MAX_SEARCH_QUERIES
    max_candidate_files: int = AUTO_MAX_CANDIDATE_FILES
    max_files_to_read: int = AUTO_MAX_FILES_TO_READ
    max_lines_per_file: int = AUTO_MAX_LINES_PER_FILE
    max_discovery_rounds: int = AUTO_MAX_DISCOVERY_ROUNDS
    max_tool_calls_before_decision: int = AUTO_MAX_TOOL_CALLS_BEFORE_DECISION


@dataclass
class AutoChatSessionState:
    last_mode: str = ""
    last_task: str = ""
    relevant_files: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    verification: str = ""
    summary: str = ""


def classify_auto_chat_intent(message: str) -> AutoChatMode:
    """Classify a non-slash chat message into a bounded execution mode."""
    text = str(message or "").strip()
    if not text:
        return AutoChatMode.ANSWER_ONLY
    if text.startswith("/"):
        return AutoChatMode.ANSWER_ONLY
    if _VERIFY_RE.search(text):
        return AutoChatMode.VERIFY
    if _REVIEW_RE.search(text):
        return AutoChatMode.REVIEW
    if _ANALYZE_RE.search(text):
        return AutoChatMode.ANALYZE
    if _EXECUTE_PLAN_RE.search(text):
        return AutoChatMode.EDIT
    if _PLAN_RE.search(text) and not _EDIT_RE.search(text):
        return AutoChatMode.PLAN_ONLY
    if _EDIT_RE.search(text):
        return AutoChatMode.EDIT
    if _ANSWER_RE.search(text):
        return AutoChatMode.ANSWER_ONLY
    return AutoChatMode.ANSWER_ONLY


def is_plan_execution_request(message: str) -> bool:
    return bool(_EXECUTE_PLAN_RE.search(str(message or "")))


def is_followup_auto_message(message: str) -> bool:
    return bool(_FOLLOWUP_RE.match(str(message or "").strip()))


def resolve_auto_followup(message: str, state: AutoChatSessionState | None) -> str:
    """Reuse compact prior context for short follow-up turns."""
    text = str(message or "").strip()
    if not text or not state or not is_followup_auto_message(text):
        return text
    if not state.last_task:
        return text
    lines = [
        text,
        "",
        "Previous auto-chat task context:",
        f"- last_mode: {state.last_mode or 'unknown'}",
        f"- last_task: {state.last_task}",
    ]
    if state.relevant_files:
        lines.append(f"- relevant_files: {', '.join(state.relevant_files[:AUTO_MAX_CANDIDATE_FILES])}")
    if state.changed_files:
        lines.append(f"- changed_files: {', '.join(state.changed_files[:AUTO_MAX_CANDIDATE_FILES])}")
    if state.summary:
        lines.append(f"- summary: {state.summary[:600]}")
    return "\n".join(lines)


def mode_allows_mutation(mode: AutoChatMode | str | None) -> bool:
    return normalize_mode(mode) == AutoChatMode.EDIT


def normalize_mode(mode: AutoChatMode | str | None) -> AutoChatMode:
    if isinstance(mode, AutoChatMode):
        return mode
    text = str(mode or "").strip().lower()
    for item in AutoChatMode:
        if text == item.value:
            return item
    return AutoChatMode.ANSWER_ONLY


def apply_auto_chat_tool_policy(
    policy: dict[str, Any],
    mode: AutoChatMode | str | None,
    *,
    limits: AutoChatLimits | None = None,
) -> dict[str, Any]:
    """Return a policy constrained by auto-chat mode and discovery limits."""
    resolved_mode = normalize_mode(mode)
    resolved_limits = limits or AutoChatLimits()
    constrained = dict(policy or {})

    if resolved_mode == AutoChatMode.EDIT:
        allowed_tools = list(constrained.get("allowed_tools") or EDIT_TOOLS)
        for tool in ("edit_file", "multi_edit_file", "apply_patch", "apply_patch_batch", "create_file", "write_file", "delete_file"):
            if tool not in allowed_tools:
                allowed_tools.append(tool)
    elif resolved_mode == AutoChatMode.VERIFY:
        allowed_tools = VERIFY_TOOLS
    elif resolved_mode == AutoChatMode.REVIEW:
        allowed_tools = REVIEW_TOOLS
    else:
        allowed_tools = READ_TOOLS

    if not mode_allows_mutation(resolved_mode):
        allowed_tools = [tool for tool in allowed_tools if tool not in MUTATION_TOOLS]
        constrained["mutation_allowed"] = False
    else:
        constrained["mutation_allowed"] = True

    constrained["allowed_tools"] = allowed_tools
    constrained["auto_chat_mode"] = resolved_mode.value
    constrained["search_budget"] = min(
        int(constrained.get("search_budget", resolved_limits.max_search_queries) or 1),
        resolved_limits.max_search_queries,
    )
    constrained["read_budget"] = min(
        int(constrained.get("read_budget", resolved_limits.max_files_to_read) or 1),
        resolved_limits.max_files_to_read,
    )
    constrained["read_budget_cap"] = min(
        int(constrained.get("read_budget_cap", resolved_limits.max_files_to_read) or 1),
        resolved_limits.max_files_to_read,
    )
    constrained["read_line_window"] = min(
        int(constrained.get("read_line_window", resolved_limits.max_lines_per_file) or 1),
        resolved_limits.max_lines_per_file,
    )
    constrained["max_candidate_files"] = resolved_limits.max_candidate_files
    constrained["max_discovery_rounds"] = resolved_limits.max_discovery_rounds
    constrained["max_tool_calls_before_decision"] = resolved_limits.max_tool_calls_before_decision
    constrained["search_repeat_limit"] = min(int(constrained.get("search_repeat_limit", 1) or 1), 1)
    constrained["stop_on_repeated_evidence"] = True
    constrained["no_full_repo_scan_by_default"] = True
    if resolved_mode != AutoChatMode.EDIT:
        constrained["require_read_files"] = min(int(constrained.get("require_read_files", 1) or 0), 1)
    return constrained


def tool_allowed_for_mode(tool_name: str, mode: AutoChatMode | str | None) -> bool:
    if str(tool_name or "").strip() in MUTATION_TOOLS and not mode_allows_mutation(mode):
        return False
    return True


def compact_auto_state_path(root: str | Path) -> Path:
    return Path(root) / ".mana" / "chat" / "auto_state.json"


def load_auto_chat_state(root: str | Path) -> AutoChatSessionState:
    path = compact_auto_state_path(root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return AutoChatSessionState()
    if not isinstance(payload, dict):
        return AutoChatSessionState()
    return AutoChatSessionState(
        last_mode=str(payload.get("last_mode", "") or ""),
        last_task=str(payload.get("last_task", "") or ""),
        relevant_files=[str(item) for item in payload.get("relevant_files", []) if str(item).strip()][:AUTO_MAX_CANDIDATE_FILES],
        changed_files=[str(item) for item in payload.get("changed_files", []) if str(item).strip()][:AUTO_MAX_CANDIDATE_FILES],
        verification=str(payload.get("verification", "") or ""),
        summary=str(payload.get("summary", "") or ""),
    )


def save_auto_chat_state(root: str | Path, state: AutoChatSessionState) -> None:
    path = compact_auto_state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
