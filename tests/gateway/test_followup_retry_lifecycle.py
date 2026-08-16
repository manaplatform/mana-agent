"""Focused regression coverage for durable follow-up/retry boundaries."""

from types import SimpleNamespace

import pytest

from mana_agent._version import get_runtime_git_sha, get_version
from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.gateway.checkpoint_resume import CheckpointResumeError
from mana_agent.services.chat_session_history import ChatSessionMessage


def test_canonical_retry_request_uses_trigger_turn_user_message() -> None:
    gateway = AgentChatGateway.__new__(AgentChatGateway)
    gateway._lane_coordinator = SimpleNamespace(
        execution_supervisor=SimpleNamespace(
            store=SimpleNamespace(
                get_task_or_none=lambda task_id: SimpleNamespace(
                    task_id=task_id,
                    trigger_turn_id="turn_original",
                )
            )
        )
    )
    gateway._history_store = SimpleNamespace(
        list=lambda session_id, limit: [
            ChatSessionMessage(
                message_id="msg_original",
                session_id=session_id,
                conversation_id=session_id,
                turn_id="turn_original",
                role="user",
                content="original request ABC",
            ),
            ChatSessionMessage(
                message_id="msg_retry",
                session_id=session_id,
                conversation_id=session_id,
                turn_id="turn_retry",
                role="user",
                content="retry",
            ),
        ]
    )

    assert gateway._canonical_task_request("task_1", "session_1") == "original request ABC"


def test_canonical_retry_request_fails_closed_without_linkage() -> None:
    gateway = AgentChatGateway.__new__(AgentChatGateway)
    gateway._lane_coordinator = SimpleNamespace(
        execution_supervisor=SimpleNamespace(
            store=SimpleNamespace(get_task_or_none=lambda task_id: SimpleNamespace())
        )
    )
    gateway._history_store = SimpleNamespace(list=lambda session_id, limit: [])

    with pytest.raises(CheckpointResumeError, match="trigger turn linkage is missing"):
        gateway._canonical_task_request("task_1", "session_1")


def test_execution_report_revision_sources_are_observability_only() -> None:
    assert get_version()
    assert isinstance(get_runtime_git_sha(), str)
