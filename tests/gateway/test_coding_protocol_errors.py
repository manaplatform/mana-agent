from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
import pytest

from mana_agent.config.model_capabilities import (
    ModelCapabilityDescriptor,
    clear_capability_cache,
    clear_model_capability_overrides,
    resolve_model_capability,
)
from mana_agent.config.provider_registry import CodexTransport
from mana_agent.gateway.config import ChatGatewayConfig
from mana_agent.gateway.turn_engine import ChatTurnResult, process_chat_turn
from mana_agent.integrations.codex.exceptions import (
    CodexBadRequestError,
    CodexCapabilityError,
    CodexProtocolError,
    CodexTimeoutError,
    CodexToolProtocolError,
)
from mana_agent.integrations.codex.result_parser import parse_codex_result
from mana_agent.coding.models import CodingTask, WorkspaceContext
from mana_agent.multi_agent.routing.agent_decision import AgentDecision


class MockCodingAgent:
    def __init__(self, side_effect: Any = None) -> None:
        self.side_effect = side_effect
        self.calls: list[dict[str, Any]] = []

    def generate(self, request: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"request": request, "kwargs": kwargs})
        if isinstance(self.side_effect, Exception):
            raise self.side_effect
        if callable(self.side_effect):
            return self.side_effect(request, **kwargs)
        if isinstance(self.side_effect, dict):
            return dict(self.side_effect)
        return {"answer": "done", "status": "completed", "changed_files": ["app.py"]}


def _test_decision() -> AgentDecision:
    return AgentDecision(
        intent="edit",
        code_editing_needed=True,
        selected_tools=["apply_patch"],
        tool_inputs={},
        flow_action="none",
        reasoning_summary="coding is required",
        confidence=0.99,
        verifier_passed=True,
        runtime_capability_change=False,
    )


@pytest.fixture(autouse=True)
def reset_capabilities():
    clear_capability_cache()
    clear_model_capability_overrides()
    yield
    clear_capability_cache()
    clear_model_capability_overrides()


def test_result_parser_classifies_tool_protocol_error(tmp_path: Path):
    """Verify parse_codex_result classifies HTTP 400 server-tool error into CODING_PROVIDER_TOOL_PROTOCOL_ERROR."""
    task = CodingTask(
        task_id="task_1",
        goal="Write code",
        requires_repository_write=True,
    )
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="readOnly",
    )
    notifications = [
        {
            "method": "turn/failed",
            "params": {
                "message": "OpenRouter rejected the request (HTTP 400). Server tools not supported for model x-ai/grok-4.6.",
                "http_status": 400,
                "error": "server-tool failure: unrecognized tool type",
            },
        }
    ]
    result = parse_codex_result(
        task=task,
        workspace=workspace,
        worker_id="worker_1",
        thread_id="th_1",
        turn_id="tu_1",
        notifications=notifications,
        changed_files=[],
    )
    assert result.status == "failed"
    assert result.codex_metadata["http_status"] == 400
    assert result.codex_metadata["error_code"] == "CODING_PROVIDER_TOOL_PROTOCOL_ERROR"
    assert any("CODING_PROVIDER_TOOL_PROTOCOL_ERROR" in e for e in result.errors)
    assert "CODING_PROVIDER_TIMEOUT" not in str(result.errors)
    assert "CODING_TIMEOUT" not in str(result.errors)


def test_result_parser_classifies_bad_request_error(tmp_path: Path):
    """Verify parse_codex_result classifies general HTTP 400 error into CODING_PROVIDER_BAD_REQUEST."""
    task = CodingTask(
        task_id="task_2",
        goal="Write code",
        requires_repository_write=True,
    )
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="readOnly",
    )
    notifications = [
        {
            "method": "turn/failed",
            "params": {
                "message": "OpenRouter rejected the request (HTTP 400). Invalid parameter 'top_k'.",
                "http_status": 400,
            },
        }
    ]
    result = parse_codex_result(
        task=task,
        workspace=workspace,
        worker_id="worker_1",
        thread_id="th_1",
        turn_id="tu_1",
        notifications=notifications,
        changed_files=[],
    )
    assert result.status == "failed"
    assert result.codex_metadata["http_status"] == 400
    assert result.codex_metadata["error_code"] == "CODING_PROVIDER_BAD_REQUEST"
    assert any("CODING_PROVIDER_BAD_REQUEST" in e for e in result.errors)


def test_result_parser_classifies_real_timeout(tmp_path: Path):
    """Verify parse_codex_result classifies actual timeout as CODING_PROVIDER_TIMEOUT."""
    task = CodingTask(
        task_id="task_3",
        goal="Write code",
        requires_repository_write=True,
    )
    workspace = WorkspaceContext(
        repository_path=tmp_path,
        worktree_path=tmp_path,
        sandbox="readOnly",
    )
    notifications = [
        {
            "method": "turn/failed",
            "params": {
                "message": "Request timed out after 300 seconds",
                "reason": "timeout",
                "error_code": "CODING_PROVIDER_TIMEOUT",
            },
        }
    ]
    result = parse_codex_result(
        task=task,
        workspace=workspace,
        worker_id="worker_1",
        thread_id="th_1",
        turn_id="tu_1",
        notifications=notifications,
        changed_files=[],
    )
    assert result.status == "failed"
    assert result.codex_metadata["error_code"] == "CODING_PROVIDER_TIMEOUT"
    assert any("CODING_PROVIDER_TIMEOUT" in e for e in result.errors)


def test_gateway_coding_turn_tool_protocol_error_not_reported_as_timeout(tmp_path: Path):
    """Verify gateway process_chat_turn does NOT report tool protocol error as timeout."""
    session_state: dict[str, Any] = {}
    coding_agent = MockCodingAgent(
        side_effect=CodexToolProtocolError(
            "OpenRouter rejected the request (HTTP 400). Server tools not supported for model x-ai/grok-4.6.",
            provider="openrouter",
            model="x-ai/grok-4.6",
            transport="direct_responses",
            http_status=400,
            original_error="server-tool rejection",
        )
    )

    turn_result = process_chat_turn(
        root=tmp_path,
        text="fix the bug",
        chat_service=SimpleNamespace(),
        ask_service=SimpleNamespace(),
        coding_agent=coding_agent,
        config=ChatGatewayConfig().normalized(),
        session_state=session_state,
        agent_decision=_test_decision(),
        gateway_task_id="task_tool_protocol",
    )

    assert turn_result.error_code == "CODING_PROVIDER_TOOL_PROTOCOL_ERROR"
    assert turn_result.error_category == "protocol"
    assert turn_result.error_code != "CODING_PROVIDER_TIMEOUT"
    assert turn_result.error_code != "CODING_TIMEOUT"
    assert turn_result.error_category != "timeout"
    assert turn_result.payload.get("http_status") == 400
    assert turn_result.payload.get("provider") == "openrouter"
    assert turn_result.payload.get("model") == "x-ai/grok-4.6"
    assert turn_result.payload.get("transport") == "direct_responses"
    assert turn_result.payload.get("original_error") == "server-tool rejection"


def test_gateway_coding_turn_capability_error(tmp_path: Path):
    """Verify gateway process_chat_turn maps capability error to CODING_CAPABILITY_ERROR."""
    session_state: dict[str, Any] = {}
    coding_agent = MockCodingAgent(
        side_effect=CodexCapabilityError(
            "Write-required Codex turn rejected: model capabilities are unknown",
            provider="openrouter",
            model="unknown-custom-model",
            transport="direct_responses",
        )
    )

    turn_result = process_chat_turn(
        root=tmp_path,
        text="write feature",
        chat_service=SimpleNamespace(),
        ask_service=SimpleNamespace(),
        coding_agent=coding_agent,
        config=ChatGatewayConfig().normalized(),
        session_state=session_state,
        agent_decision=_test_decision(),
        gateway_task_id="task_capability_error",
    )

    assert turn_result.error_code == "CODING_CAPABILITY_ERROR"
    assert turn_result.error_category == "capability"
    assert turn_result.payload.get("provider") == "openrouter"
    assert turn_result.payload.get("model") == "unknown-custom-model"


def test_gateway_coding_turn_real_timeout(tmp_path: Path):
    """Verify gateway process_chat_turn maps real timeout to CODING_PROVIDER_TIMEOUT."""
    session_state: dict[str, Any] = {}
    coding_agent = MockCodingAgent(
        side_effect=CodexTimeoutError(
            "Codex request timed out after 300 seconds",
            provider="openai",
            model="gpt-4.1",
            transport="direct_responses",
            timeout_seconds=300.0,
        )
    )

    turn_result = process_chat_turn(
        root=tmp_path,
        text="long running edit",
        chat_service=SimpleNamespace(),
        ask_service=SimpleNamespace(),
        coding_agent=coding_agent,
        config=ChatGatewayConfig().normalized(),
        session_state=session_state,
        agent_decision=_test_decision(),
        gateway_task_id="task_real_timeout",
    )

    assert turn_result.error_code == "CODING_PROVIDER_TIMEOUT"
    assert turn_result.error_category == "timeout"
    assert turn_result.payload.get("provider") == "openai"
    assert turn_result.payload.get("model") == "gpt-4.1"
