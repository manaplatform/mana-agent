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


def test_judge_approval_request_denies_destructive_command_deterministically() -> None:
    # Model should not even be called for deterministic hard-denial
    mock_llm = _MockLLM({"judgment": "accept"})
    judgment = judge_approval_request(
        command="rm -rf /tmp/important_data",
        reason="Clean up data",
        goal="Clean directory",
        llm=mock_llm,
    )

    assert mock_llm.invoked is False
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
