from __future__ import annotations

import pytest

from mana_agent.utils.tool_policy import (
    InvalidToolPolicyError,
    expand_tool_aliases,
    resolve_allowed_tools,
)


def test_file_system_alias_expands_to_real_tools() -> None:
    expanded, unknown = expand_tool_aliases(["file_system"])
    assert unknown == []
    assert set(expanded) == {"ls", "list_files", "read_file", "repo_search"}


def test_real_tool_names_pass_through_unchanged() -> None:
    expanded, unknown = expand_tool_aliases(["ls", "read_file", "create_file"])
    assert unknown == []
    assert set(expanded) == {"ls", "read_file", "create_file"}


def test_edit_alias_includes_create_and_delete_file() -> None:
    expanded, unknown = expand_tool_aliases(["edit"])
    assert unknown == []
    assert {"edit_file", "multi_edit_file", "apply_patch", "create_file", "delete_file", "write_file"} <= set(expanded)


def test_unknown_alias_is_reported() -> None:
    expanded, unknown = expand_tool_aliases(["file_system", "totally_made_up"])
    assert "totally_made_up" in unknown
    # The valid alias still expands.
    assert "read_file" in expanded


def test_resolve_strict_raises_on_unknown() -> None:
    with pytest.raises(InvalidToolPolicyError) as excinfo:
        resolve_allowed_tools(["file_system", "nope"], strict=True)
    assert "nope" in excinfo.value.unknown
    payload = excinfo.value.to_error_payload()
    assert payload["error_code"] == "invalid_tool_policy"
    assert "nope" in payload["unknown_tools"]


def test_resolve_non_strict_drops_unknown_but_expands() -> None:
    resolved = resolve_allowed_tools(["file_system", "nope"], strict=False)
    assert "read_file" in resolved
    assert "nope" not in resolved


def test_blank_and_duplicate_names_are_normalized() -> None:
    expanded, unknown = expand_tool_aliases(["", "  ", "ls", "ls", "inspect"])
    assert unknown == []
    # inspect group includes AST/callgraph tools on top of file_system tools.
    assert "find_symbols" in expanded
    assert "call_graph" in expanded
    assert expanded == sorted(set(expanded))
