from __future__ import annotations

import json

from mana_agent.multi_agent.runtime.auto_chat import (
    AUTO_MAX_CANDIDATE_FILES,
    AUTO_MAX_DISCOVERY_ROUNDS,
    AUTO_MAX_FILES_TO_READ,
    AUTO_MAX_LINES_PER_FILE,
    AUTO_MAX_SEARCH_QUERIES,
    AUTO_MAX_TOOL_CALLS_BEFORE_DECISION,
    AutoChatMode,
    AutoChatSessionState,
    apply_auto_chat_tool_policy,
    classify_auto_chat_intent,
    load_auto_chat_state,
    mode_allows_mutation,
    resolve_auto_followup,
    save_auto_chat_state,
    tool_allowed_for_mode,
)


def test_auto_chat_intent_classifier_modes() -> None:
    assert classify_auto_chat_intent("Where is transaction approval handled?") == AutoChatMode.ANSWER_ONLY
    assert classify_auto_chat_intent("Give me a plan for duplicate detection") == AutoChatMode.PLAN_ONLY
    assert classify_auto_chat_intent("Fix duplicate transaction approval") == AutoChatMode.EDIT
    assert classify_auto_chat_intent("review my changes") == AutoChatMode.REVIEW
    assert classify_auto_chat_intent("run tests") == AutoChatMode.VERIFY
    assert classify_auto_chat_intent("analyze project") == AutoChatMode.ANALYZE
    assert classify_auto_chat_intent("execute the plan") == AutoChatMode.EDIT


def test_auto_chat_mutation_guard_blocks_non_edit_modes() -> None:
    for mode in (
        AutoChatMode.ANSWER_ONLY,
        AutoChatMode.PLAN_ONLY,
        AutoChatMode.REVIEW,
        AutoChatMode.VERIFY,
        AutoChatMode.ANALYZE,
    ):
        assert mode_allows_mutation(mode) is False
        assert tool_allowed_for_mode("edit_file", mode) is False
        assert tool_allowed_for_mode("multi_edit_file", mode) is False
        assert tool_allowed_for_mode("apply_patch", mode) is False

    assert mode_allows_mutation(AutoChatMode.EDIT) is True
    assert tool_allowed_for_mode("edit_file", AutoChatMode.EDIT) is True
    assert tool_allowed_for_mode("multi_edit_file", AutoChatMode.EDIT) is True
    assert tool_allowed_for_mode("apply_patch", AutoChatMode.EDIT) is True


def test_auto_chat_tool_policy_applies_limits_and_read_only_allowlist() -> None:
    policy = apply_auto_chat_tool_policy(
        {
            "allowed_tools": ["repo_search", "read_file", "apply_patch", "write_file", "verify_project"],
            "search_budget": 50,
            "read_budget": 50,
            "read_budget_cap": 50,
            "read_line_window": 5000,
            "require_read_files": 4,
        },
        AutoChatMode.ANSWER_ONLY,
    )

    assert "apply_patch" not in policy["allowed_tools"]
    assert "write_file" not in policy["allowed_tools"]
    assert policy["mutation_allowed"] is False
    assert policy["search_budget"] == AUTO_MAX_SEARCH_QUERIES
    assert policy["read_budget"] == AUTO_MAX_FILES_TO_READ
    assert policy["read_budget_cap"] == AUTO_MAX_FILES_TO_READ
    assert policy["read_line_window"] == AUTO_MAX_LINES_PER_FILE
    assert policy["max_candidate_files"] == AUTO_MAX_CANDIDATE_FILES
    assert policy["max_discovery_rounds"] == AUTO_MAX_DISCOVERY_ROUNDS
    assert policy["max_tool_calls_before_decision"] == AUTO_MAX_TOOL_CALLS_BEFORE_DECISION
    assert policy["require_read_files"] == 1


def test_auto_chat_edit_policy_keeps_mutation_tools_but_still_bounded() -> None:
    policy = apply_auto_chat_tool_policy(
        {"allowed_tools": ["repo_search", "read_file", "apply_patch"], "search_budget": 99, "read_budget": 99},
        AutoChatMode.EDIT,
    )

    assert "apply_patch" in policy["allowed_tools"]
    assert {"edit_file", "multi_edit_file", "apply_patch", "create_file", "write_file", "delete_file"} <= set(policy["allowed_tools"])
    assert policy["mutation_allowed"] is True
    assert policy["search_budget"] == AUTO_MAX_SEARCH_QUERIES
    assert policy["read_budget"] == AUTO_MAX_FILES_TO_READ


def test_auto_chat_followup_reuses_compact_state() -> None:
    state = AutoChatSessionState(
        last_mode="answer_only",
        last_task="Where is approval handled?",
        relevant_files=["src/payments.py"],
        summary="Approval is handled by approve_payment.",
    )

    resolved = resolve_auto_followup("do it", state)

    assert "Previous auto-chat task context" in resolved
    assert "Where is approval handled?" in resolved
    assert "src/payments.py" in resolved


def test_auto_chat_state_roundtrip(tmp_path) -> None:
    state = AutoChatSessionState(
        last_mode="edit",
        last_task="fix bug",
        relevant_files=["a.py"],
        changed_files=["a.py"],
        verification="pytest a",
        summary="done",
    )

    save_auto_chat_state(tmp_path, state)
    loaded = load_auto_chat_state(tmp_path)

    assert loaded.last_mode == "edit"
    assert loaded.relevant_files == ["a.py"]
    assert json.loads((tmp_path / ".mana" / "chat" / "auto_state.json").read_text())["summary"] == "done"
