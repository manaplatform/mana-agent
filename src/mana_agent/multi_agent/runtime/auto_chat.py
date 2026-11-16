"""Bounded normal-chat routing helpers.

Slash commands are handled by the CLI before this module is consulted. These
helpers constrain tools after a route has been selected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from pathlib import Path
from mana_agent.workspaces.paths import repository_dir, repository_id_for_path
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

MUTATION_TOOLS = frozenset(
    {
        "edit_file",
        "multi_edit_file",
        "apply_patch",
        "apply_patch_batch",
        "write_file",
        "create_file",
        "delete_file",
        "document_create",
        "document_update",
        "document_delete",
        "move_file",
        "batch_write_files",
        "batch_apply_patches",
        "batch_move_files",
        "batch_delete_files",
        "git_create_branch",
        "git_switch",
        "git_checkout",
        "git_add",
        "git_restore",
        "git_stash",
        "git_commit",
        "git_push",
        "git_pull",
        "git_fetch",
        "git_tag",
        "git_merge",
        "git_rebase",
        "git_revert",
        "git_reset",
        "git_clean",
        "git_config",
        "git_generic",
        "run_command",
        "run_script_once",
        "verify_project",
        "email_draft_create",
        "email_draft_update",
        "email_draft_delete",
        "email_send",
        "email_draft_send",
        "email_reply",
        "email_reply_all",
        "email_forward",
        "email_trash",
        "email_restore",
        "email_labels_update",
        "api_request_execute",
        "api_integration_update",
        "api_integration_delete",
        "canvas_create_surface",
        "canvas_update_components",
        "canvas_update_data",
        "canvas_delete_surface",
    }
)
READ_TOOLS = [
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
    "git_help",
    "git_log",
    "git_branch",
    "git_remote",
    "tool_contracts",
    "read_skill",
]

EDIT_TOOLS = [
    *READ_TOOLS,
    "run_command",
    "verify_project",
    "git_create_branch",
    "git_switch",
    "git_add",
    "git_commit",
    "git_push",
    "git_checkout",
    "git_restore",
    "git_stash",
    "git_pull",
    "git_fetch",
    "git_tag",
    "git_merge",
    "git_rebase",
    "git_revert",
    "git_reset",
    "git_clean",
    "git_config",
    "git_generic",
    "edit_file",
    "multi_edit_file",
    "apply_patch",
    "apply_patch_batch",
    "create_file",
    "write_file",
    "delete_file",
]

VERIFY_TOOLS = ["run_command", "run_script_once", "verify_project", "git_status", "git_diff", "git_log", "read_file", "repo_batch_read", "ls", "list_files"]
REVIEW_TOOLS = ["git_status", "git_diff", "git_log", "git_branch", "git_remote", "read_file", "repo_batch_read", "ls", "list_files", "repo_search", "repo_batch_search"]


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
    execution_mode: str = "plan"
    active_plan: dict[str, Any] | None = None


def mode_allows_mutation(mode: AutoChatMode | str | Any | None) -> bool:
    if hasattr(mode, "allows_mutation"):
        return bool(mode.allows_mutation)
    return normalize_mode(mode) == AutoChatMode.EDIT


def normalize_mode(mode: AutoChatMode | str | Any | None) -> AutoChatMode:
    if isinstance(mode, AutoChatMode):
        return mode
    text = str(getattr(mode, "value", mode) or "").strip().lower().replace("-", "_")
    if text in {"act", "edit"}:
        return AutoChatMode.EDIT
    if text in {"plan", "plan_only"}:
        return AutoChatMode.PLAN_ONLY
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
        for tool in (
            "edit_file",
            "multi_edit_file",
            "apply_patch",
            "apply_patch_batch",
            "create_file",
            "write_file",
            "delete_file",
            "document_create",
            "document_update",
            "document_delete",
        ):
            if tool not in allowed_tools:
                allowed_tools.append(tool)
    elif resolved_mode == AutoChatMode.VERIFY:
        allowed_tools = VERIFY_TOOLS
    elif resolved_mode == AutoChatMode.REVIEW:
        allowed_tools = REVIEW_TOOLS
    else:
        allowed_tools = READ_TOOLS

    if not mode_allows_mutation(resolved_mode):
        # Verification is an explicit execution intent (e.g. running tests),
        # unlike Plan Mode's PLAN_ONLY route. Keep its bounded verification
        # commands available while excluding them from every other read-only
        # route.
        verification_tools = set(VERIFY_TOOLS) if resolved_mode == AutoChatMode.VERIFY else set()
        allowed_tools = [
            tool
            for tool in allowed_tools
            if tool not in MUTATION_TOOLS or tool in verification_tools
        ]
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
    tool = str(tool_name or "").strip().lower()
    resolved_mode = normalize_mode(mode)
    if tool in MUTATION_TOOLS and not mode_allows_mutation(resolved_mode):
        if resolved_mode == AutoChatMode.VERIFY and tool in VERIFY_TOOLS:
            return True
        return False
    return True


def compact_auto_state_path(root: str | Path) -> Path:
    resolved = Path(root).resolve()
    return repository_dir(repository_id_for_path(resolved)) / "chat" / "auto_state.json"


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
        execution_mode=str(payload.get("execution_mode", "plan") or "plan"),
        active_plan=payload.get("active_plan") if isinstance(payload.get("active_plan"), dict) else None,
    )


def save_auto_chat_state(root: str | Path, state: AutoChatSessionState) -> None:
    path = compact_auto_state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
