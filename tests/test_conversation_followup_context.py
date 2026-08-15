"""Unit tests for conversation context and dialogue follow-up handling."""
from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from mana_agent.multi_agent.runtime.qna_chain import QnAChain
from mana_agent.services.chat_service import ChatService


def test_qna_chain_chat_includes_recent_history() -> None:
    chain = object.__new__(QnAChain)
    chain.provider = "mock"
    chain.model = "mock-model"
    mock_llm = MagicMock()
    mock_response = AIMessage(content="test is 1")
    mock_llm.invoke.return_value = mock_response
    chain.llm = mock_llm

    recent_history = [
        {"role": "user", "content": "remember that test = 1"},
        {"role": "assistant", "content": "Got it, test = 1."},
    ]

    answer = chain.chat("what is test?", recent_history=recent_history)
    assert answer == "test is 1"

    assert mock_llm.invoke.called
    messages_passed = mock_llm.invoke.call_args[0][0]
    # Check messages structure: SystemMessage, HumanMessage(history), AIMessage(history), HumanMessage(current)
    assert len(messages_passed) == 4
    assert isinstance(messages_passed[1], HumanMessage)
    assert messages_passed[1].content == "remember that test = 1"
    assert isinstance(messages_passed[2], AIMessage)
    assert messages_passed[2].content == "Got it, test = 1."
    assert isinstance(messages_passed[3], HumanMessage)
    assert messages_passed[3].content == "what is test?"


def test_chat_service_ask_conversation_forwards_recent_history() -> None:
    mock_ask_service = MagicMock()
    mock_qna = MagicMock()
    mock_qna.chat.return_value = "The test value is 1."
    mock_ask_service.qna_chain = mock_qna

    service = ChatService(ask_service=mock_ask_service, settings=MagicMock())
    recent_history = [
        {"role": "user", "content": "test = 1"},
        {"role": "assistant", "content": "noted"},
    ]
    res = service.ask_conversation("what is test?", recent_history=recent_history)
    assert res == "The test value is 1."
    assert mock_qna.chat.called
    _, kwargs = mock_qna.chat.call_args
    assert kwargs.get("recent_history") == recent_history
