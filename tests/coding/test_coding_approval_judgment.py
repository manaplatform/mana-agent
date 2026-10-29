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
    # Create a small script to execute
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

    assert result.get("ok") is True
    assert "EXEC_SUCCESS" in str(result.get("stdout") or "")
    assert "model_judgment" in result
    assert result["model_judgment"]["judgment"] == "accept"


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
    assert result.get("status") == "denied"
    assert "suggested_command" in result
    assert result["suggested_command"] == "echo safe_alternative"


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
    )

    assert res.get("approved") is True
    assert bool(res.get("approval_id")) is True
    assert res.get("status") == "approved"


def test_dispatcher_evaluates_different_task_approval_via_judgment_and_executes_with_correct_approval(tmp_path: Path) -> None:
    # 1. Create an approval under a foreign task ID (e.g. 'shell_exec' or 'task_foreign')
    mock_llm_setup = _MockLLM({
        "judgment": "accept",
        "reason": "Foreign setup command approved.",
        "confidence": 1.0,
    })
    foreign_approval = approval_request(
        command="python -V",
        reason="Verify python version in initial task",
        task_id="shell_exec",
        workspace_root=tmp_path,
        llm=mock_llm_setup,
        enable_model_judgment=True,
    )
    foreign_grant_id = str(foreign_approval.get("approval_id") or "")
    assert foreign_grant_id

    # 2. Set up dispatcher with a different active task ID
    mock_judge_llm = _MockLLM({
        "judgment": "accept",
        "reason": "Python check is directly related to active task goal.",
        "suggested_command": "",
        "confidence": 0.98,
        "risk_assessment": "safe_verification",
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
        task_id="task_active_lineage_999",
        llm=mock_judge_llm,
    )

    # 3. Dispatch run_command with foreign approval ID.
    # It must NOT raise ValueError("Command approval belongs to a different task").
    # It must invoke judgment, decide it is related, issue the correct approval, and execute.
    result = dispatcher.dispatch(
        "run_command",
        {
            "command": "python -V",
            "action_approval_id": foreign_grant_id,
            "reason": "Verify python environment for task_active",
        },
        call_id="call_cross_task_1",
    )

    assert result.get("ok") is True
    assert mock_judge_llm.invoked is True
    assert "model_judgment" in result
    assert result["model_judgment"]["judgment"] == "accept"
    assert "Python check is directly related" in result["model_judgment"]["reason"]


def test_dispatcher_denies_unrelated_command_approval_from_different_task_via_judgment(tmp_path: Path) -> None:
    # 1. Create an approval under task_1
    mock_llm_setup = _MockLLM({
        "judgment": "accept",
        "reason": "Command approved in task_1.",
        "confidence": 1.0,
    })
    foreign_approval = approval_request(
        command="python -c 'print(\"UNRELATED\")'",
        reason="Run script in task_1",
        task_id="task_1",
        workspace_root=tmp_path,
        llm=mock_llm_setup,
        enable_model_judgment=True,
    )
    foreign_grant_id = str(foreign_approval.get("approval_id") or "")
    assert foreign_grant_id

    # 2. Dispatch under task_2 where judgment decides it is unrelated
    mock_judge_llm = _MockLLM({
        "judgment": "deny",
        "reason": "Command from task_1 is completely unrelated to task_2 goals.",
        "suggested_command": "pytest tests/",
        "confidence": 0.9,
        "risk_assessment": "unrelated_execution",
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
        task_id="task_2",
        llm=mock_judge_llm,
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
    assert result.get("status") == "denied"
    assert result.get("error_code") == "APPROVAL_DENIED_BY_MODEL_JUDGMENT"
    assert "completely unrelated" in result.get("reason", "")
    assert result.get("suggested_command") == "pytest tests/"


def test_dispatcher_ignores_hardcoded_or_unverified_approval_tokens(tmp_path: Path) -> None:
    script_path = tmp_path / "check.py"
    script_path.write_text("print('HARDCODED_TEST_SUCCESS')\n", encoding="utf-8")

    mock_judge_llm = _MockLLM({
        "judgment": "accept",
        "reason": "Safe script execution accepted by judgment.",
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
        task_id="task_legitimate",
        llm=mock_judge_llm,
    )

    # Calling with a fake / hardcoded approval ID string
    result = dispatcher.dispatch(
        "run_command",
        {
            "command": f"python {script_path.name}",
            "action_approval_id": "hardcoded_fake_grant_xyz",
            "reason": "Test hardcoded rejection",
        },
        call_id="call_hardcoded_1",
    )

    assert result.get("ok") is True
    assert "HARDCODED_TEST_SUCCESS" in str(result.get("stdout") or "")
    assert mock_judge_llm.invoked is True
    assert result.get("model_judgment", {}).get("judgment") == "accept"


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
    script_fail = tmp_path / "fail_then_pass.py"
    script_fail.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")

    script_ok = tmp_path / "success.py"
    script_ok.write_text("print('RETRY_FIX_SUCCESS')\n", encoding="utf-8")

    mock_llm = _MockLLM({
        "judgment": "accept",
        "reason": "Script failed with error; retry with suggested fixed script.",
        "suggested_command": f"python {script_ok.name}",
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
        task_id="task_retry_test",
        llm=mock_llm,
    )

    result = dispatcher.dispatch(
        "run_command",
        {"command": f"python {script_fail.name}"},
        call_id="call_retry_1",
    )

    assert result.get("ok") is True
    assert "RETRY_FIX_SUCCESS" in str(result.get("stdout") or "")
    assert mock_llm.invoked is True
    assert result.get("model_judgment", {}).get("judgment") == "accept"


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


