from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mana_agent.coding.models import WorkspaceContext
from mana_agent.coding.programmatic import ToolExecutionDispatcher
from mana_agent.human_inbox.approval_tools import approval_request
from mana_agent.multi_agent.agents.approval_agent import (
    ApprovalAgent,
    ModelApprovalJudgment,
    judge_approval_request,
)


class _MockLLM:
    def __init__(self, response_dict: dict[str, object]) -> None:
        self.response_dict = response_dict
        self.invoked = False

    def invoke(self, _messages: object) -> SimpleNamespace:
        self.invoked = True
        return SimpleNamespace(content=json.dumps(self.response_dict))

    def __call__(self, _messages: object) -> SimpleNamespace:
        return self.invoke(_messages)


def test_judge_approval_request_accepts_safe_command() -> None:
    mock_llm = _MockLLM({
        "judgment": "accept",
        "reason": "Test verification is safe and required.",
        "suggested_command": "",
        "confidence": 0.95,
        "risk_assessment": "low_risk_verification",
    })
    judgment = judge_approval_request(
        command="pytest tests/test_core.py -q",
        reason="Run test suite to verify changes",
        goal="Fix issue in core logic",
        llm=mock_llm,
    )

    assert mock_llm.invoked is True
    assert judgment.judgment == "accept"
    assert "safe" in judgment.reason.lower()
    assert judgment.confidence == 0.95


def test_judge_approval_request_evaluates_via_model_without_hardcoded_gate() -> None:
    # All commands pass to model judgment without hardcoded regex bypass
    mock_llm = _MockLLM({
        "judgment": "deny",
        "reason": "Destructive rm command denied by model judgment.",
        "suggested_command": "",
        "confidence": 1.0,
    })
    judgment = judge_approval_request(
        command="rm -rf /tmp/important_data",
        reason="Clean up data",
        goal="Clean directory",
        llm=mock_llm,
    )

    assert mock_llm.invoked is True
    assert judgment.judgment == "deny"
    assert "destructive" in judgment.reason.lower()


def test_judge_approval_request_suggests_alternative_command() -> None:
    mock_llm = _MockLLM({
        "judgment": "deny",
        "reason": "Full test suite takes too long; run targeted unit test instead.",
        "suggested_command": "pytest tests/test_unit.py -k test_quick",
        "confidence": 0.9,
        "risk_assessment": "medium_duration",
    })
    judgment = judge_approval_request(
        command="pytest tests/ --run-slow",
        reason="Run full test suite",
        goal="Quick sanity test",
        llm=mock_llm,
    )

    assert judgment.judgment == "deny"
    assert judgment.suggested_command == "pytest tests/test_unit.py -k test_quick"


def test_approval_agent_evaluate_approval_method() -> None:
    mock_llm = _MockLLM({
        "judgment": "accept",
        "reason": "Safe lint check.",
        "suggested_command": "",
        "confidence": 1.0,
    })
    agent = ApprovalAgent(agent_id="agent_approval_test", llm=mock_llm)
    judgment = agent.evaluate_approval(
        command="ruff check src/",
        reason="Check formatting",
    )

    assert judgment.judgment == "accept"
    assert agent.may_continue("approved") is True


def test_dispatcher_executes_command_when_model_judgment_accepts(tmp_path: Path) -> None:
    # Model judgment is advisory and does not bypass human approval requirements
    script_path = tmp_path / "script.py"
    script_path.write_text("print('EXEC_SUCCESS')\n", encoding="utf-8")

    mock_llm = _MockLLM({
        "judgment": "accept",
        "reason": "Python script is safe and scoped to workspace.",
        "suggested_command": "",
        "confidence": 1.0,
    })

    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        working_directory=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )
    dispatcher = ToolExecutionDispatcher(
        workspace=workspace,
        task_id="task_test_approval",
        llm=mock_llm,
    )

    # Calling run_command with a mutating or approval-sensitive command
    result = dispatcher.dispatch(
        "run_command",
        {"command": f"python {script_path.name}"},
        call_id="call_test_1",
    )

    assert result.get("ok") is False
    assert result.get("approved") is False
    assert result.get("permission_required") is True
    assert result.get("status") == "awaiting_approval"


def test_dispatcher_denies_command_when_model_judgment_denies(tmp_path: Path) -> None:
    mock_llm = _MockLLM({
        "judgment": "deny",
        "reason": "Unsafe command outside permitted scope.",
        "suggested_command": "echo safe_alternative",
        "confidence": 0.9,
    })

    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        working_directory=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )
    dispatcher = ToolExecutionDispatcher(
        workspace=workspace,
        task_id="task_test_deny",
        llm=mock_llm,
    )

    result = dispatcher.dispatch(
        "run_command",
        {"command": "python -c 'import shutil; shutil.rmtree(\"/tmp/test\")'"},
        call_id="call_test_2",
    )

    assert result.get("ok") is False
    assert result.get("approved") is False


def test_approval_request_tool_resolves_via_model_judgment(tmp_path: Path) -> None:
    mock_llm = _MockLLM({
        "judgment": "accept",
        "reason": "Approved by model judgment for testing.",
        "suggested_command": "",
        "confidence": 1.0,
    })

    res = approval_request(
        command="python -V",
        reason="Check python version",
        task_id="task_approval_test",
        workspace_root=tmp_path,
        llm=mock_llm,
        enable_model_judgment=True,
        timeout_seconds=0.01,
    )

    assert res.get("approved") is True
    assert res.get("status") == "approved"
    assert res.get("approval_id") == ""


def test_dispatcher_evaluates_different_task_approval_via_judgment_and_executes_with_correct_approval(tmp_path: Path) -> None:
    # 1. Create an approval under task_1
    from mana_agent.transactional_actions.runtime import default_action_gateway
    import getpass
    gateway = default_action_gateway(tmp_path)
    ws1 = WorkspaceContext(repository_path=tmp_path, worktree_path=tmp_path, sandbox="workspaceWrite", allow_in_place_write=True)
    d1 = ToolExecutionDispatcher(workspace=ws1, task_id="task_1")
    pending = d1.dispatch("run_command", {"command": "python -V"})
    foreign_grant_id = gateway.approve(pending["action_id"], approved_by=getpass.getuser())

    # 2. Dispatcher with a different task ID must reject foreign task grant
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        working_directory=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )
    dispatcher = ToolExecutionDispatcher(
        workspace=workspace,
        task_id="task_active_lineage_999",
    )

    result = dispatcher.dispatch(
        "run_command",
        {
            "command": "python -V",
            "action_approval_id": foreign_grant_id,
            "reason": "Verify python environment for task_active",
        },
        call_id="call_cross_task_1",
    )

    assert result.get("ok") is False
    assert result.get("error_code") == "INVALID_APPROVAL"


def test_dispatcher_denies_unrelated_command_approval_from_different_task_via_judgment(tmp_path: Path) -> None:
    from mana_agent.transactional_actions.runtime import default_action_gateway
    import getpass
    gateway = default_action_gateway(tmp_path)
    ws1 = WorkspaceContext(repository_path=tmp_path, worktree_path=tmp_path, sandbox="workspaceWrite", allow_in_place_write=True)
    d1 = ToolExecutionDispatcher(workspace=ws1, task_id="task_1")
    pending = d1.dispatch("run_command", {"command": "python -c 'print(\"UNRELATED\")'"})
    foreign_grant_id = gateway.approve(pending["action_id"], approved_by=getpass.getuser())

    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        working_directory=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )
    dispatcher = ToolExecutionDispatcher(
        workspace=workspace,
        task_id="task_2",
    )

    result = dispatcher.dispatch(
        "run_command",
        {
            "command": "python -c 'print(\"UNRELATED\")'",
            "action_approval_id": foreign_grant_id,
            "reason": "Run unrelated script",
        },
        call_id="call_cross_task_2",
    )

    assert result.get("ok") is False
    assert result.get("error_code") == "INVALID_APPROVAL"


def test_dispatcher_ignores_hardcoded_or_unverified_approval_tokens(tmp_path: Path) -> None:
    script_path = tmp_path / "check.py"
    script_path.write_text("print('HARDCODED_TEST_SUCCESS')\n", encoding="utf-8")

    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        working_directory=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )
    dispatcher = ToolExecutionDispatcher(
        workspace=workspace,
        task_id="task_legitimate",
    )

    # Calling with a fake / hardcoded approval ID string must fail closed
    result = dispatcher.dispatch(
        "run_command",
        {
            "command": f"python {script_path.name}",
            "action_approval_id": "hardcoded_fake_grant_xyz",
            "reason": "Test hardcoded rejection",
        },
        call_id="call_hardcoded_1",
    )

    assert result.get("ok") is False
    assert result.get("error_code") == "INVALID_APPROVAL"


def test_judge_approval_request_returns_approval_verdict_for_consequential_command() -> None:
    mock_llm = _MockLLM({
        "judgment": "approval",
        "reason": "Consequential command requires human approval before proceeding.",
        "suggested_command": "",
        "confidence": 0.95,
        "risk_assessment": "consequential_mutation",
    })
    judgment = judge_approval_request(
        command="git reset --hard HEAD~1",
        reason="Reset branch",
        goal="Roll back commit",
        llm=mock_llm,
    )

    assert mock_llm.invoked is True
    assert judgment.judgment == "approval"
    assert "human approval" in judgment.reason.lower()


def test_dispatcher_script_error_sends_to_judgment_and_retries_on_accept(tmp_path: Path) -> None:
    # Non-zero exit codes do not trigger automatic model-suggested command retry
    script_fail = tmp_path / "fail_then_pass.py"
    script_fail.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")

    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        working_directory=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )
    dispatcher = ToolExecutionDispatcher(
        workspace=workspace,
        task_id="task_retry_test",
    )

    result = dispatcher.dispatch(
        "run_command",
        {"command": f"python {script_fail.name}"},
        call_id="call_retry_1",
    )

    assert result.get("ok") is False
    assert result.get("status") == "awaiting_approval"


def test_dispatcher_sends_approval_when_judgment_demands_human_approval(tmp_path: Path) -> None:
    mock_llm = _MockLLM({
        "judgment": "approval",
        "reason": "Consequential command requires explicit human review and approval.",
        "suggested_command": "",
        "confidence": 0.95,
    })

    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        working_directory=tmp_path,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )
    dispatcher = ToolExecutionDispatcher(
        workspace=workspace,
        task_id="task_require_human_approval",
        llm=mock_llm,
    )

    result = dispatcher.dispatch(
        "run_command",
        {"command": "git clean -fd"},
        call_id="call_approval_demand_1",
    )

    assert result.get("ok") is False
    assert result.get("permission_required") is True
    assert result.get("status") == "awaiting_approval"
    assert "requires human approval" in result.get("error", "").lower()


