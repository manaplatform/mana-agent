"""
mana_agent.tools

Tool implementations used by agentic components.
"""

from .apply_patch import build_apply_patch_tool, safe_apply_patch, extract_patch_touched_files  # noqa: F401
from .contracts import coding_tool_contracts, coding_tool_contracts_payload  # noqa: F401
from .edit_file import (  # noqa: F401
    build_edit_file_tool,
    build_multi_edit_file_tool,
    safe_edit_file,
    safe_multi_edit_file,
)
from .repository import (  # noqa: F401
    call_graph,
    explore_src,
    find_symbols,
    git_diff,
    git_status,
    inspect_project_structure,
    inspect_tests,
    list_files,
    repo_search,
    verify_file_created,
    verify_project,
)
from .write_file import (  # noqa: F401
    build_create_file_tool,
    build_delete_file_tool,
    build_write_file_tool,
    safe_create_file,
    safe_delete_file,
    safe_write_file,
)

__all__ = [
    "build_apply_patch_tool",
    "build_edit_file_tool",
    "build_multi_edit_file_tool",
    "coding_tool_contracts",
    "coding_tool_contracts_payload",
    "call_graph",
    "extract_patch_touched_files",
    "explore_src",
    "find_symbols",
    "git_diff",
    "git_status",
    "inspect_project_structure",
    "inspect_tests",
    "list_files",
    "repo_search",
    "safe_apply_patch",
    "safe_edit_file",
    "safe_multi_edit_file",
    "verify_file_created",
    "verify_project",
    "build_write_file_tool",
    "build_create_file_tool",
    "build_delete_file_tool",
    "safe_write_file",
    "safe_create_file",
    "safe_delete_file",
]
