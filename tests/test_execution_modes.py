from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mana_agent.agent.mode import (
    ExecutionMode,
    PlanApprovalState,
    StructuredPlan,
    normalize_execution_mode,
)
from mana_agent.agent.plan_service import PlanService
from mana_agent.chat_commands.builtins import (
    _act_command,
    _approve_command,
    _execution_mode_command,
    _plan_command,
)
from mana_agent.chat_commands.models import CommandContext
from mana_agent.coding.models import CodingTask
from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.multi_agent.runtime.auto_chat import (
    AutoChatMode,
    mode_allows_mutation,
    tool_allowed_for_mode,
)
from mana_agent.tools.shell_exec import ShellExecutor, is_plan_mode_read_only_command


def test_execution_mode_enum_and_normalization() -> None:
    plan_mode = ExecutionMode.PLAN
    assert plan_mode.is_plan is True
    assert plan_mode.is_act is False
    assert plan_mode.allows_mutation is False
    assert plan_mode.allows_destructive_commands is False

    act_mode = ExecutionMode.ACT
    assert act_mode.is_plan is False
    assert act_mode.is_act is True
    assert act_mode.allows_mutation is True
    assert act_mode.allows_destructive_commands is True

    # Test normalization
    assert normalize_execution_mode("plan") == ExecutionMode.PLAN
    assert normalize_execution_mode("PLAN") == ExecutionMode.PLAN
    assert normalize_execution_mode("plan_only") == ExecutionMode.PLAN
    assert normalize_execution_mode("act") == ExecutionMode.ACT
    assert normalize_execution_mode("edit") == ExecutionMode.ACT
    assert normalize_execution_mode(AutoChatMode.PLAN_ONLY) == ExecutionMode.PLAN
    assert normalize_execution_mode(AutoChatMode.EDIT) == ExecutionMode.ACT
    assert normalize_execution_mode(ExecutionMode.ACT) == ExecutionMode.ACT


def test_structured_plan_model_and_markdown() -> None:
    plan = StructuredPlan(
        task_id="plan-1234",
        goal="Add Plan Mode and Act Mode",
        objectives=["Add execution modes", "Enforce read-only in Plan Mode"],
        context_summary="Repository exploration complete",
        affected_files=["src/mana_agent/agent/mode.py", "src/mana_agent/tools/shell_exec.py"],
        dependencies=["pytest", "typer"],
        implementation_steps=[
            "Create mode.py with ExecutionMode and StructuredPlan",
            "Update tool gates to check execution mode",
        ],
        potential_risks=["Backwards compatibility with AutoChatMode"],
        validation_strategies=["pytest tests/test_execution_modes.py -v"],
        assumptions=["User will approve before Act Mode"],
        approved=False,
    )

    md = plan.to_markdown()
    assert "# Implementation Plan: Add Plan Mode and Act Mode" in md
    assert "## 1. Objectives" in md
    assert "## 2. Context & Dependencies" in md
    assert "`pytest`" in md
    assert "## 3. Affected Files" in md
    assert "`src/mana_agent/agent/mode.py`" in md
    assert "## 4. Implementation Steps" in md
    assert "1. Create mode.py with ExecutionMode and StructuredPlan" in md
    assert "## 5. Potential Risks & Mitigations" in md
    assert "## 6. Validation Strategies & Verification Commands" in md
    assert "Pending User Approval" in md

    # Test approval update
    approved_plan = plan.with_approval(approved=True, approver="alice")
    assert approved_plan.approved is True
    assert approved_plan.approved_by == "alice"
    assert approved_plan.approved_at is not None
    assert "Approved By: alice" in approved_plan.to_markdown()


def test_plan_approval_state() -> None:
    state = PlanApprovalState(plan_id="p-1")
    assert state.approved is False
    assert state.approved_at is None

    state.approve(approver="tester")
    assert state.approved is True
    assert state.approved_by == "tester"
    assert state.approved_at is not None

    state.reject(reason="Needs more tests")
    assert state.approved is False
    assert state.rejection_reason == "Needs more tests"
    assert state.approved_at is None


def test_plan_mode_blocks_mutation_tools() -> None:
    for mode in (ExecutionMode.PLAN, AutoChatMode.PLAN_ONLY, "plan", "plan_only"):
        assert mode_allows_mutation(mode) is False
        for mutation_tool in (
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
            "batch_write_files",
            "batch_apply_patches",
            "git_commit",
            "git_push",
        ):
            assert tool_allowed_for_mode(mutation_tool, mode) is False, f"Tool {mutation_tool} must be blocked in {mode}"

    # Additional write-capable tools exposed by the tool registry must not
    # pass through Plan Mode's per-tool guard.
    for tool in (
        "git_generic",
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
        "run_command",
        "run_script_once",
        "verify_project",
        "email_send",
        "api_request_execute",
        "canvas_update_data",
    ):
        assert tool_allowed_for_mode(tool, "plan") is False, f"Tool {tool} must be blocked in Plan Mode"

    # An explicit verification request still has access to bounded test tools.
    assert tool_allowed_for_mode("run_command", AutoChatMode.VERIFY) is True


def test_act_mode_allows_mutation_tools() -> None:
    for mode in (ExecutionMode.ACT, AutoChatMode.EDIT, "act", "edit"):
        assert mode_allows_mutation(mode) is True
        for mutation_tool in (
            "edit_file",
            "multi_edit_file",
            "apply_patch",
            "write_file",
            "create_file",
        ):
            assert tool_allowed_for_mode(mutation_tool, mode) is True


def test_shell_execution_engine_blocks_destructive_commands_in_plan_mode(tmp_path: Path) -> None:
    engine = ShellExecutor(workspace_root=tmp_path, needs_approval=False)

    # In Plan Mode, mutating commands are rejected with PlanModeError
    result_touch = engine.run("touch bad_file.txt", execution_mode="plan")
    assert result_touch.outcome.exit_code == 1
    assert "PlanModeError" in result_touch.stderr
    assert not (tmp_path / "bad_file.txt").exists()

    result_rm = engine.run("rm -f non_existent", execution_mode="plan")
    assert result_rm.outcome.exit_code == 1
    assert "PlanModeError" in result_rm.stderr

    # Read-only-looking commands with write-capable options must fail closed.
    for command in (
        "find . -delete",
        r"find . -exec touch plan_mode_bypass \;",
        "sh -c 'ls && touch plan_mode_bypass'",
        "git branch created-in-plan",
        "git tag created-in-plan",
        "git remote add plan-mode-remote https://example.invalid/repo.git",
        "git diff --output=plan_mode_bypass",
        "pytest tests/test_example.py",
    ):
        assert not is_plan_mode_read_only_command(command), command
        result = engine.run(command, execution_mode="plan")
        assert result.outcome.exit_code == 1, command
        assert "PlanModeError" in result.stderr, command
    assert not (tmp_path / "plan_mode_bypass").exists()

    # In Plan Mode, read-only commands succeed
    result_ls = engine.run("ls -la", execution_mode="plan")
    assert result_ls.outcome.exit_code == 0
    assert result_ls.outcome.type == "exit"

    # In execute_action API, plan mode blocks mutating commands
    action_res = engine.execute(
        commands=["echo 'read only'", "touch another_bad.txt"],
        execution_mode="plan",
    )
    assert len(action_res["output"]) == 2
    assert action_res["output"][0]["outcome"]["exit_code"] == 0
    assert "PlanModeError" in action_res["output"][1]["stderr"]
    assert not (tmp_path / "another_bad.txt").exists()

    # Model-controlled action fields cannot elevate a Plan Mode call.
    forged_mode_res = engine.execute_action(
        {"commands": ["touch forged_mode.txt"], "execution_mode": "act"},
        execution_mode="plan",
    )
    assert "PlanModeError" in forged_mode_res["output"][0]["stderr"]
    assert not (tmp_path / "forged_mode.txt").exists()

    unknown_mode_res = engine.run("touch unknown_mode.txt", execution_mode="unexpected")
    assert "PlanModeError" in unknown_mode_res.stderr
    assert not (tmp_path / "unknown_mode.txt").exists()


def test_plan_service_requires_a_model_and_reads_only_selected_lines(tmp_path: Path) -> None:
    from mana_agent.agent.plan_service import EvidenceRequest

    (tmp_path / "app.py").write_text("before\nfunction\nafter\n")
    service = PlanService(tmp_path, llm=None)
    read = EvidenceRequest(tool="read_file", level=1, paths=("app.py",),
                           start_line=2, end_line=2, reason="Reported location")
    assert service.inspect(read)["content"] == "function\n"
    with pytest.raises(RuntimeError, match="native decision model"):
        service.next_decision("plan", {"request": "Refactor this function"})


def test_gateway_session_mode_is_auto_until_user_selects_plan(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("mana_agent.commands.cli_internal.build_ask_service", lambda *a, **k: None)
    gw = AgentChatGateway(root=tmp_path, coding_agent=False)
    session = gw.create_session(frontend="cli")
    assert gw.get_session_execution_mode(session) == "auto"
    switched = gw.dispatch_command("/mode plan", session_id=session, frontend="cli")
    assert switched.status == "success"
    assert gw.get_session_execution_mode(session) == "plan"
    assert gw.get_session_active_plan(session) is None
    no_revision = gw.dispatch_command("/approve", session_id=session, frontend="cli")
    assert no_revision.status == "error"
    assert "inbox-id" in no_revision.message


def test_coding_task_plan_mode_read_only_default() -> None:
    # Plan mode tasks must not require repository write by default
    task_plan = CodingTask(task_id="t-plan", goal="Analyze module", execution_mode="plan")
    assert task_plan.requires_repository_write is False

    task_act = CodingTask(task_id="t-act", goal="Fix bug", execution_mode="act")
    assert task_act.requires_repository_write is True


def test_plan_service_validates_structured_model_decision(tmp_path: Path) -> None:
    mock_llm = MagicMock()
    mock_llm.decide_sync.return_value = {"decision": {
        "action": "investigate", "reason": "Investigate the target", "scope": "app.py",
        "objective": "Find the source of the handler behavior",
    }}
    service = PlanService(tmp_path, llm=mock_llm)
    decision = service.next_decision("plan", {"request": "Inspect app.py"})
    assert decision.scope == "app.py"
    mock_llm.decide_sync.return_value = {"decision": {"action": "draft", "reason": "missing material"}}
    with pytest.raises(RuntimeError, match="ModeDecision"):
        service.next_decision("plan", {"request": "Inspect app.py"})
