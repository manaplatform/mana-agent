"""Tests for transactional action approval in AgentChatGateway."""

from __future__ import annotations

import getpass
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from mana_agent.gateway import AgentChatGateway
from mana_agent.transactional_actions.adapters import McpActionAdapter, ShellActionAdapter
from mana_agent.transactional_actions.runtime import create_transactional_runtime


def _build_test_gateway(tmp_path: Path) -> AgentChatGateway:
    gw = AgentChatGateway.__new__(AgentChatGateway)
    gw.workspace_root = tmp_path
    gw.project_root = tmp_path
    gw._transactional_runtime = create_transactional_runtime(tmp_path)
    gw.human_inbox_service = gw._transactional_runtime.inbox_service
    return gw


def test_transactional_action_approval_command_shell_action(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    gw = _build_test_gateway(tmp_path)

    adapter = ShellActionAdapter(
        argv=["nmap", "-p", "443", "manadev.net"],
        cwd=tmp_path,
        environment={},
        expected_outputs=[],
        parent_task_id="shell_exec",
        actor="user",
        originating_agent="shell_executor",
        idempotency_key="shell_test_approval_1",
        allow_command_result_verification=True,
    )
    action_intent = gw._transactional_runtime.gateway.propose(adapter)
    inbox_item_id = action_intent.inbox_item_id
    assert inbox_item_id

    result = gw.transactional_action_approval_command(inbox_item_id)
    assert result["status"] == "approved"
    assert "shell action approved" in result["message"].lower()
    assert "legacy mcp" not in result["message"].lower()
    assert result["approval_id"] != ""


def test_transactional_action_approval_command_mcp_action_executes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    gw = _build_test_gateway(tmp_path)

    executed = False

    def fake_mcp_call(args: dict[str, Any]) -> str:
        nonlocal executed
        executed = True
        return json.dumps({"ok": True, "output": "mcp_call_success"})

    mock_tool = MagicMock()
    mock_tool.metadata = {"mcp_provider_id": "test_server", "mcp_tool_name": "scan_port"}
    mock_tool.invoke = MagicMock(side_effect=fake_mcp_call)

    import mana_agent.mcp.tools as mcp_tools_mod
    monkeypatch.setattr(mcp_tools_mod, "discovered_mcp_langchain_tools", lambda server_ids: ([mock_tool], []))

    adapter = McpActionAdapter(
        provider_id="test_server",
        tool_name="scan_port",
        arguments={"host": "manadev.net", "port": 443},
        invoke=lambda: fake_mcp_call({"host": "manadev.net", "port": 443}),
        parent_task_id="ask-mcp",
        actor="user",
        originating_agent="ask_agent",
    )
    action_intent = gw._transactional_runtime.gateway.propose(adapter)
    inbox_item_id = action_intent.inbox_item_id
    assert inbox_item_id

    result = gw.transactional_action_approval_command(inbox_item_id)
    assert result["status"] == "approved"
    assert "mcp action approved once and executed successfully" in result["message"].lower()
    assert executed is True


def test_transactional_action_approval_by_action_id(tmp_path: Path, monkeypatch) -> None:
    """Verifies approval succeeds when passed action_id (e.g. act_...) instead of inbox_item_id."""
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    gw = _build_test_gateway(tmp_path)

    adapter = ShellActionAdapter(
        argv=["nmap", "-p", "443", "manadev.net"],
        cwd=tmp_path,
        environment={},
        expected_outputs=[],
        parent_task_id="shell_exec",
        actor="user",
        originating_agent="shell_executor",
        idempotency_key="shell_test_approval_by_action_id",
        allow_command_result_verification=True,
    )
    action_intent = gw._transactional_runtime.gateway.propose(adapter)
    action_id = action_intent.action_id
    assert action_id.startswith("act_")

    # Approve using action_id directly
    result = gw.transactional_action_approval_command(action_id)
    assert result["status"] == "approved"
    assert result["action_id"] == action_id
    assert result["approval_id"] != ""

    # Check grant was issued for this exact action
    grant = gw._transactional_runtime.gateway.approvals.find_valid(action_intent)
    assert grant is not None
    assert grant.approval_id == result["approval_id"]

