"""Unit tests for conversation context and dialogue follow-up handling."""
from __future__ import annotations

from unittest.mock import MagicMock
import json
import pytest

from langchain_core.messages import AIMessage, HumanMessage

from mana_agent.multi_agent.runtime.qna_chain import QnAChain
from mana_agent.services.chat_service import ChatService
from mana_agent.gateway.conversation_context import recent_conversation_context
from mana_agent.tools.context_retrieval import execute_conversation_context_read


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


def test_direct_chat_service_remembers_the_assistant_offer() -> None:
    ask = MagicMock()
    offer = "Explanation. " * 60 + "I can add CSV export. Would you like that?"
    ask.qna_chain.chat.side_effect = [offer, "I will add CSV export."]
    service = ChatService(ask_service=ask, settings=MagicMock())
    service.ask_conversation("Explain the available export options")
    service.ask_conversation("yes, I want it")
    history = ask.qna_chain.chat.call_args.kwargs["recent_history"]
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == offer
    assert history[-2]["role"] == "user"


def test_explicit_empty_history_does_not_leak_another_session() -> None:
    ask = MagicMock()
    ask.qna_chain.chat.return_value = "Private reply"
    service = ChatService(ask_service=ask, settings=MagicMock())
    service.ask_conversation("Private question")
    service.ask_conversation("New conversation", recent_history=[])
    assert ask.qna_chain.chat.call_args.kwargs["recent_history"] == []


def test_recent_context_preserves_offer_beyond_300_characters() -> None:
    offer = "Background. " * 50 + "Shall I add a download button?"
    context = recent_conversation_context([
        {"turn_id": "previous", "message_id": "user-1", "role": "user", "content": "What can we improve?"},
        {"turn_id": "previous", "message_id": "assistant-1", "role": "assistant", "content": offer},
        {"turn_id": "current", "message_id": "user-2", "role": "user", "content": "yes, I want it"},
    ], current_turn_id="current")
    assert context["messages"][-1]["content"] == offer
    assert context["latest_message_ids"] == ["user-1", "assistant-1"]
    assert not context["retrieval_required"]


def test_budget_does_not_leave_an_orphan_assistant_answer() -> None:
    context = recent_conversation_context([
        {"turn_id": "turn", "message_id": "u", "role": "user", "content": "Question " * 500},
        {"turn_id": "turn", "message_id": "a", "role": "assistant", "content": "Yes"},
    ], max_tokens=10)
    assert context["messages"] == []
    assert context["truncated"]
    assert context["retrieval_required"]
    assert context["latest_message_ids"] == ["u", "a"]


def test_reference_retrieval_uses_observed_ids_instead_of_confirmation_keywords() -> None:
    history = [{"turn_id": "turn-1", "message_id": "assistant-1", "role": "assistant", "content": "I can add CSV export."}]
    payload = json.loads(execute_conversation_context_read(
        session_id="session", conversation_id="session", authenticated_user_id="owner",
        history_store=history, message_ids=["assistant-1"], query="yes, do it",
    ))
    assert payload["turns"][0]["content"] == history[0]["content"]
    assert payload["turns"][0]["role"] == "assistant"
    with pytest.raises(PermissionError):
        execute_conversation_context_read(session_id="session", conversation_id="session",
            authenticated_user_id="owner", history_store=history, message_ids=["another-session-message"])
