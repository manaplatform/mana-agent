"""
mana_agent.utils.tool_policy

Helpers for resolving high-level tool-policy aliases (e.g. ``file_system``)
into the concrete tool names the worker actually registers.

Upstream callers (and LLM-produced policies) sometimes emit grouped aliases
such as ``allowed_tools=["file_system"]``. The worker only understands real
tool names like ``ls`` / ``read_file`` / ``repo_search``, so an unexpanded
alias silently blocks every real tool. This module expands aliases up front
and reports any names that are still unknown after expansion.
"""

from __future__ import annotations

from typing import Iterable

# High-level aliases → concrete registered tool names.
TOOL_GROUPS: dict[str, list[str]] = {
    "file_system": ["ls", "list_files", "read_file", "repo_search"],
    "inspect": ["ls", "list_files", "read_file", "repo_search", "find_symbols", "call_graph"],
    "search": ["semantic_search", "repo_search", "list_files", "find_symbols", "call_graph"],
    "edit": ["edit_file", "multi_edit_file", "apply_patch", "write_file", "create_file", "delete_file", "git_diff"],
    "verify": ["verify_project", "run_command"],
}

# The full set of real tool names the AskAgent worker registers. Kept here so
# policy validation does not need to instantiate an agent. Must stay in sync
# with AskAgent._build_tools / the externally-registered tools.
REGISTERED_TOOLS: frozenset[str] = frozenset(
    {
        "semantic_search",
        "read_file",
        "run_command",
        "chunk_file",
        "list_tools",
        "ls",
        "repo_search",
        "list_files",
        "find_symbols",
        "call_graph",
        "git_status",
        "git_diff",
        "verify_project",
        "tool_contracts",
        "edit_file",
        "multi_edit_file",
        "apply_patch",
        "write_file",
        "create_file",
        "delete_file",
        "github_search",
    }
)


class InvalidToolPolicyError(ValueError):
    """Raised when a tool policy references names that cannot be resolved."""

    def __init__(self, unknown: Iterable[str]) -> None:
        self.unknown = sorted({str(item) for item in unknown})
        super().__init__(
            "invalid tool policy: unknown tool(s) "
            + ", ".join(self.unknown)
            + ". Use real tool names or known aliases: "
            + ", ".join(sorted(TOOL_GROUPS))
        )

    def to_error_payload(self) -> dict[str, object]:
        return {
            "ok": False,
            "error_code": "invalid_tool_policy",
            "error": str(self),
            "unknown_tools": list(self.unknown),
        }


def expand_tool_aliases(
    names: Iterable[str],
    *,
    registered: Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Expand alias groups into concrete tool names.

    Returns ``(expanded, unknown)`` where ``expanded`` is the sorted, de-duped
    list of concrete tool names and ``unknown`` is any name that is neither a
    known alias nor a registered tool. Names already valid pass through
    unchanged, so this is safe to call on policies that never used aliases.
    """
    registered_set = frozenset(registered) if registered is not None else REGISTERED_TOOLS
    expanded: set[str] = set()
    unknown: list[str] = []
    for raw in names:
        name = str(raw or "").strip()
        if not name:
            continue
        if name in TOOL_GROUPS:
            expanded.update(TOOL_GROUPS[name])
            continue
        if name in registered_set:
            expanded.add(name)
            continue
        unknown.append(name)
    return sorted(expanded), unknown


def resolve_allowed_tools(
    names: Iterable[str],
    *,
    registered: Iterable[str] | None = None,
    strict: bool = False,
) -> list[str]:
    """Resolve ``allowed_tools`` aliases to concrete names.

    With ``strict=True`` an unknown tool raises :class:`InvalidToolPolicyError`
    so callers can surface a structured policy error before a worker run
    begins. With ``strict=False`` unknown names are dropped (the legacy,
    permissive behavior) but aliases are still expanded.
    """
    expanded, unknown = expand_tool_aliases(names, registered=registered)
    if unknown and strict:
        raise InvalidToolPolicyError(unknown)
    return expanded
