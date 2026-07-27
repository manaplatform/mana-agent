"""Catalog of auto-chat tools for CLI/TUI visibility.

This module discovers tool *metadata* (name + short description + category)
for display in the chat CLI TUI. It does not execute tools and does not start
MCP servers unless ``include_mcp_discovery`` is explicitly requested.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ToolCatalogEntry:
    """One tool visible to the chat interface."""

    name: str
    description: str
    category: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


# Stable first-party auto-chat surface. Descriptions match runtime tools where
# those tools are registered on AskAgent / ChatService (not every contract-only
# name). Grouped by category for readable TUI output.
_BUILTIN_AUTO_CHAT_TOOLS: tuple[tuple[str, str, str], ...] = (
    # Search / research
    (
        "web_search",
        "Search the public web for current facts, docs, and topics outside the local repo.",
        "search",
    ),
    (
        "github_search",
        "Search public GitHub repositories, code, issues, and project metadata.",
        "search",
    ),
    (
        "semantic_search",
        "Search indexed code chunks semantically when a local vector index is available.",
        "search",
    ),
    (
        "repo_search",
        "Search repository text with regex or literal matching (read-only).",
        "search",
    ),
    (
        "repo_batch_search",
        "Run multiple repository text searches in one call.",
        "search",
    ),
    # Repository inspection
    (
        "read_file",
        "Read a repository file (full file or line range).",
        "repository",
    ),
    (
        "repo_batch_read",
        "Read multiple repository files in one call.",
        "repository",
    ),
    (
        "list_files",
        "List repository files with optional glob filtering.",
        "repository",
    ),
    (
        "ls",
        "List project directories relative to the repository root.",
        "repository",
    ),
    (
        "find_symbols",
        "Find Python classes, functions, and methods via AST.",
        "repository",
    ),
    (
        "call_graph",
        "Inspect Python call edges by caller, callee, or file.",
        "repository",
    ),
    (
        "chunk_file",
        "Chunk a large file into text parts when a full read is blocked.",
        "repository",
    ),
    (
        "list_tools",
        "List available tool names for the current agent session.",
        "repository",
    ),
    (
        "tool_contracts",
        "Return strict contracts (schema, safety, examples) for coding tools.",
        "repository",
    ),
    (
        "read_skill",
        "Load one skills/<name>/SKILL.md body after the skill is selected.",
        "repository",
    ),
    # Email connector (wired on AskAgent via build_email_langchain_tools)
    (
        "email_accounts_list",
        "List connected non-secret email accounts and capabilities.",
        "email",
    ),
    (
        "email_search",
        "Search a connected email account; results are untrusted external data.",
        "email",
    ),
    (
        "email_read",
        "Read a normalized email message via message_ref from email_search.",
        "email",
    ),
    (
        "email_thread_read",
        "Read a normalized email thread (untrusted external data).",
        "email",
    ),
    # Documents
    (
        "document_detect",
        "Detect supported document files by path, extension, and MIME.",
        "document",
    ),
    (
        "document_read",
        "Read DOCX, PDF, XLSX/XLSM, or CSV into normalized chunks.",
        "document",
    ),
    (
        "document_analyze",
        "Analyze document structure, tables, OCR needs, and workbook schemas.",
        "document",
    ),
    (
        "document_query",
        "Search parsed document chunks with optional filters.",
        "document",
    ),
    (
        "document_create",
        "Create DOCX, XLSX/XLSM, CSV, or simple PDF artifacts.",
        "document",
    ),
    (
        "document_update",
        "Safely update supported documents (backup by default).",
        "document",
    ),
    (
        "document_delete",
        "Delete a document only when explicit delete intent is validated.",
        "document",
    ),
    # Git (common read + mutation surface)
    ("git_status", "Return git status --short (read-only).", "git"),
    ("git_diff", "Return git diff, optionally for one path (read-only).", "git"),
    ("git_log", "Inspect recent commit history (read-only).", "git"),
    ("git_branch", "List branches (read-only).", "git"),
    ("git_remote", "Inspect configured remotes (read-only).", "git"),
    ("git_help", "Git help or commands discovered from git help -a.", "git"),
    ("git_generic", "Run a model-selected Git argv list through the safe executor.", "git"),
    # Verification / shell
    (
        "run_command",
        "Run a non-destructive shell command in the project root.",
        "verify",
    ),
    (
        "run_script_once",
        "Run one grouped, non-destructive shell script and return output.",
        "verify",
    ),
    (
        "verify_project",
        "Run standard pytest/ruff/mypy/import/CLI smoke checks.",
        "verify",
    ),
    # Edit / mutation (auto-chat edit mode)
    ("edit_file", "Replace one exact string in a repository file.", "edit"),
    ("multi_edit_file", "Apply several exact-string replacements atomically.", "edit"),
    ("apply_patch", "Apply a Codex-style patch with context recovery.", "edit"),
    ("apply_patch_batch", "Validate and apply multiple related patches.", "edit"),
    ("write_file", "Write full file content with overwrite guards.", "edit"),
    ("create_file", "Create a new file without overwriting an existing target.", "edit"),
    ("delete_file", "Delete one existing repository file.", "edit"),
)

# Preferred display order for categories in the TUI.
CATEGORY_ORDER: tuple[str, ...] = (
    "search",
    "email",
    "computer",
    "mcp",
    "browser",
    "repository",
    "document",
    "git",
    "verify",
    "edit",
    "other",
)

CATEGORY_LABELS: dict[str, str] = {
    "search": "Search & research",
    "email": "Email",
    "computer": "Computer control",
    "mcp": "MCP connectors",
    "browser": "Browser",
    "repository": "Repository",
    "document": "Documents",
    "git": "Git",
    "verify": "Verify & shell",
    "edit": "Edit",
    "other": "Other",
}


def _one_line(text: str, *, limit: int = 100) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def _category_for_name(name: str) -> str:
    n = str(name or "").strip()
    if not n:
        return "other"
    if n.startswith("email_"):
        return "email"
    if n.startswith(("computer_", "calendar_", "media_", "notes_", "clipboard_")):
        return "computer"
    if n in {"browser_get_active_page", "browser_read_page", "browser_list_tabs", "browser_open_url", "browser_activate_tab", "browser_close_tab"}:
        return "computer"
    if n.startswith("browser_"):
        return "browser"
    if n.startswith("document_"):
        return "document"
    if n.startswith("git_") or n.startswith("git."):
        return "git"
    if n.startswith("mcp__") or n.startswith("mcp.") or n == "mcp":
        return "mcp"
    if n in {"web_search", "github_search", "semantic_search", "repo_search", "repo_batch_search"}:
        return "search"
    if n in {
        "edit_file",
        "multi_edit_file",
        "apply_patch",
        "apply_patch_batch",
        "write_file",
        "create_file",
        "delete_file",
    }:
        return "edit"
    if n in {"run_command", "run_script_once", "verify_project"}:
        return "verify"
    return "repository"


def _merge_entry(
    by_name: dict[str, ToolCatalogEntry],
    name: str,
    description: str,
    category: str | None = None,
) -> None:
    cleaned_name = str(name or "").strip()
    if not cleaned_name:
        return
    cat = category or _category_for_name(cleaned_name)
    desc = _one_line(description) or f"{cleaned_name} tool"
    existing = by_name.get(cleaned_name)
    # Prefer a more specific description over a generic placeholder.
    if existing is not None and len(existing.description) >= len(desc):
        return
    by_name[cleaned_name] = ToolCatalogEntry(
        name=cleaned_name,
        description=desc,
        category=cat,
    )


def _add_browser_tools(by_name: dict[str, ToolCatalogEntry]) -> None:
    from mana_agent.config.user_config import get_setting

    if not bool(get_setting("MANA_BROWSER_ENABLED", True)):
        return
    try:
        from mana_agent.connectors.browser.contracts import browser_tool_contracts
    except ImportError:
        return
    for contract in browser_tool_contracts():
        _merge_entry(by_name, contract.name, contract.description, "browser")


def _add_computer_tools(by_name: dict[str, ToolCatalogEntry]) -> None:
    from mana_agent.integrations.computer_control.config import ComputerControlSettings

    try:
        settings = ComputerControlSettings.load()
    except ValueError:
        return
    if not settings.enabled:
        return
    from mana_agent.integrations.computer_control.tool_contracts import computer_tool_contracts

    for contract in computer_tool_contracts():
        _merge_entry(by_name, contract.name, contract.description, "computer")


def _resolve_mcp_overrides(mcp_overrides: list[str] | None) -> list[str]:
    """Merge explicit overrides with MANA_MCP_SERVER_OVERRIDES session env."""
    selected = list(mcp_overrides or [])
    if selected:
        return selected
    import json
    import os

    raw_env = str(os.getenv("MANA_MCP_SERVER_OVERRIDES") or "").strip()
    if not raw_env:
        return []
    try:
        raw = json.loads(raw_env)
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        return []
    return list(raw)


def _add_mcp_entries(
    by_name: dict[str, ToolCatalogEntry],
    *,
    include_mcp_discovery: bool,
    mcp_overrides: list[str] | None,
) -> list[str]:
    """Add MCP connector/tool entries. Returns non-fatal warning strings."""
    warnings: list[str] = []
    try:
        from mana_agent.mcp.config import McpConfigError, load_mcp_servers
    except ImportError:
        return warnings

    selected_overrides = _resolve_mcp_overrides(mcp_overrides)
    try:
        servers = load_mcp_servers(overrides=selected_overrides)
    except (McpConfigError, ValueError, OSError) as exc:
        warnings.append(f"MCP config not loaded: {exc}")
        return warnings

    if not servers:
        # Still surface the MCP capability so users know connectors exist.
        _merge_entry(
            by_name,
            "mcp",
            "Mana MCP protocol connectors (configure providers with mana-agent mcp).",
            "mcp",
        )
        return warnings

    for server in servers:
        _merge_entry(
            by_name,
            f"mcp:{server.id}",
            (
                f"MCP connector '{server.id}' ({server.transport}); "
                "tools are discovered when the model selects this provider."
            ),
            "mcp",
        )

    if not include_mcp_discovery:
        return warnings

    try:
        from mana_agent.mcp.tools import discovered_mcp_langchain_tools
    except ImportError:
        return warnings

    tools, mcp_warnings = discovered_mcp_langchain_tools(overrides=selected_overrides)
    warnings.extend(str(item) for item in mcp_warnings if str(item).strip())
    for tool in tools:
        name = str(getattr(tool, "name", "") or "").strip()
        desc = str(getattr(tool, "description", "") or "").strip()
        _merge_entry(by_name, name, desc or f"MCP tool {name}", "mcp")
    return warnings


def list_auto_chat_tools(
    *,
    include_mcp_discovery: bool = False,
    mcp_overrides: list[str] | None = None,
) -> list[ToolCatalogEntry]:
    """Return auto-chat tool metadata for CLI/TUI display.

    Always includes first-party tools (email, web_search, repo, browser when
    enabled). MCP providers are listed from config without starting servers
    unless ``include_mcp_discovery`` is True.
    """
    by_name: dict[str, ToolCatalogEntry] = {}

    for name, description, category in _BUILTIN_AUTO_CHAT_TOOLS:
        _merge_entry(by_name, name, description, category)

    # Enrich descriptions for tools already in the catalog. Do not add
    # contract-only names that are not wired into auto-chat AskAgent.
    known = set(by_name)
    try:
        from mana_agent.multi_agent.routing.agent_decision import agent_tool_descriptions

        for item in agent_tool_descriptions():
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name not in known:
                continue
            _merge_entry(
                by_name,
                name,
                str(item.get("description") or ""),
                by_name[name].category,
            )
    except Exception:
        pass

    _add_browser_tools(by_name)
    _add_computer_tools(by_name)
    _add_mcp_entries(
        by_name,
        include_mcp_discovery=include_mcp_discovery,
        mcp_overrides=mcp_overrides,
    )

    return sort_tool_catalog(by_name.values())


def sort_tool_catalog(entries: Iterable[ToolCatalogEntry]) -> list[ToolCatalogEntry]:
    order = {name: index for index, name in enumerate(CATEGORY_ORDER)}

    def _key(entry: ToolCatalogEntry) -> tuple[int, str]:
        return (order.get(entry.category, len(CATEGORY_ORDER)), entry.name.lower())

    # Deduplicate by name while preserving best description already chosen.
    by_name: dict[str, ToolCatalogEntry] = {}
    for entry in entries:
        if entry.name not in by_name:
            by_name[entry.name] = entry
    return sorted(by_name.values(), key=_key)


def group_tool_catalog(
    entries: Iterable[ToolCatalogEntry],
) -> list[tuple[str, list[ToolCatalogEntry]]]:
    """Group catalog entries by category in display order."""
    buckets: dict[str, list[ToolCatalogEntry]] = {}
    for entry in sort_tool_catalog(entries):
        buckets.setdefault(entry.category, []).append(entry)

    grouped: list[tuple[str, list[ToolCatalogEntry]]] = []
    for category in CATEGORY_ORDER:
        items = buckets.pop(category, None)
        if items:
            grouped.append((category, items))
    for category in sorted(buckets):
        grouped.append((category, buckets[category]))
    return grouped


def format_tool_catalog_plain(
    entries: Iterable[ToolCatalogEntry],
    *,
    max_per_category: int | None = None,
) -> str:
    """Plain-text grouped catalog suitable for console or TUI welcome text."""
    items = sort_tool_catalog(entries)
    if not items:
        return "No auto-chat tools are registered."
    lines: list[str] = [f"Auto-chat tools ({len(items)} available)"]
    for category, group in group_tool_catalog(items):
        label = CATEGORY_LABELS.get(category, category.title())
        lines.append(f"{label} ({len(group)})")
        shown = group if max_per_category is None else group[: max(0, int(max_per_category))]
        for entry in shown:
            lines.append(f"  {entry.name} — {entry.description}")
        if max_per_category is not None and len(group) > int(max_per_category):
            lines.append(f"  … +{len(group) - int(max_per_category)} more")
    return "\n".join(lines)


def format_tool_catalog_summary(entries: Iterable[ToolCatalogEntry]) -> str:
    """One-line summary for startup chrome (counts + key categories)."""
    items = sort_tool_catalog(entries)
    if not items:
        return "none"
    counts: dict[str, int] = {}
    for entry in items:
        counts[entry.category] = counts.get(entry.category, 0) + 1
    parts: list[str] = [f"{len(items)} available"]
    for category in ("email", "mcp", "search", "browser"):
        if counts.get(category):
            parts.append(f"{category}×{counts[category]}")
    return " · ".join(parts)


def catalog_as_dicts(entries: Iterable[ToolCatalogEntry]) -> list[dict[str, str]]:
    return [entry.to_dict() for entry in sort_tool_catalog(entries)]


def catalog_names(entries: Iterable[ToolCatalogEntry]) -> list[str]:
    return [entry.name for entry in sort_tool_catalog(entries)]
