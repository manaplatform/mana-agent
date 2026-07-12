from __future__ import annotations

import pytest

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from mana_agent.multi_agent.runtime.compatibility import (
    CompatibleChatOpenAI,
    ModelCapabilities,
    create_chat_model,
)


TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}


def _payload(llm: CompatibleChatOpenAI, *, tools: bool = False) -> dict:
    kwargs = {"tools": [TOOL]} if tools else {}
    return llm._get_request_payload([HumanMessage(content="hello")], **kwargs)


def test_tools_and_reasoning_use_responses_when_supported(monkeypatch) -> None:
    monkeypatch.setenv("MANA_LLM_REASONING_EFFORT", "high")
    llm = create_chat_model(api_key="test", model="gpt-5", base_url="https://api.openai.com/v1")
    payload = _payload(llm, tools=True)
    assert "input" in payload
    assert "messages" not in payload
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["tools"][0]["name"] == "read_file"


def test_tools_use_responses_when_supported_without_explicit_reasoning() -> None:
    llm = create_chat_model(api_key="test", model="gpt-5.6-luna", base_url="https://api.openai.com/v1")
    payload = _payload(llm, tools=True)
    assert "input" in payload
    assert "messages" not in payload
    assert payload["tools"][0]["name"] == "read_file"


def test_chat_tools_normalize_reasoning_when_combination_is_unsupported() -> None:
    llm = CompatibleChatOpenAI(api_key="test", model="gateway-model", reasoning_effort="high", compatibility_capabilities=ModelCapabilities(supports_responses_api=False))
    payload = _payload(llm, tools=True)
    assert "messages" in payload
    assert "input" not in payload
    assert payload["reasoning_effort"] == "none"
    assert payload["tools"][0]["function"]["name"] == "read_file"


def test_tools_without_reasoning_keep_chat_completions_tools() -> None:
    llm = CompatibleChatOpenAI(api_key="test", model="gateway-model", compatibility_capabilities=ModelCapabilities(False))
    payload = _payload(llm, tools=True)
    assert "messages" in payload
    assert payload["tools"][0]["function"]["name"] == "read_file"
    assert "reasoning_effort" not in payload


def test_reasoning_without_tools_remains_chat_completions_compatible() -> None:
    llm = CompatibleChatOpenAI(api_key="test", model="gateway-model", reasoning_effort="high", compatibility_capabilities=ModelCapabilities(False))
    payload = _payload(llm)
    assert "messages" in payload
    assert payload["reasoning_effort"] == "high"


def test_plain_request_is_unchanged() -> None:
    llm = CompatibleChatOpenAI(api_key="test", model="gateway-model", compatibility_capabilities=ModelCapabilities(False))
    payload = _payload(llm)
    assert set(payload).issuperset({"model", "messages"})
    assert "tools" not in payload
    assert "reasoning_effort" not in payload


def test_explicit_gateway_capability_override_selects_responses(monkeypatch) -> None:
    monkeypatch.setenv("MANA_LLM_SUPPORTS_RESPONSES_API", "true")
    monkeypatch.setenv("MANA_LLM_REASONING_EFFORT", "high")
    llm = create_chat_model(api_key="test", model="gateway-model", base_url="https://gateway.example/v1")
    assert "input" in _payload(llm, tools=True)


def test_custom_gateway_does_not_assume_responses_api(monkeypatch) -> None:
    monkeypatch.setenv("MANA_LLM_REASONING_EFFORT", "high")
    llm = create_chat_model(api_key="test", model="gateway-model", base_url="https://gateway.example/v1")
    payload = _payload(llm, tools=True)
    assert "messages" in payload
    assert payload["reasoning_effort"] == "none"


def test_unsupported_tools_reasoning_error_retries_once_with_changed_payload(monkeypatch) -> None:
    llm = CompatibleChatOpenAI(api_key="test", model="gpt-5.6-luna", reasoning_effort="high", compatibility_capabilities=ModelCapabilities(supports_responses_api=True))
    payloads: list[dict] = []

    def fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
        payloads.append(self._get_request_payload(messages, stop=stop, **kwargs))
        if len(payloads) == 1:
            raise RuntimeError("Function tools with reasoning_effort are not supported for gpt-5.6-luna in /v1/chat/completions")
        return "retried"

    monkeypatch.setattr(ChatOpenAI, "_generate", fake_generate)
    assert llm._generate([HumanMessage(content="hello")], tools=[TOOL]) == "retried"
    assert len(payloads) == 2
    assert "input" in payloads[0]
    assert "messages" in payloads[1]
    assert payloads[1]["reasoning_effort"] == "none"


def test_retry_forces_none_even_when_the_initial_metadata_claimed_chat_reasoning_support(monkeypatch) -> None:
    llm = CompatibleChatOpenAI(
        api_key="test",
        model="gateway-model",
        reasoning_effort="high",
        compatibility_capabilities=ModelCapabilities(
            supports_responses_api=False,
            supports_tools_with_chat_reasoning=True,
        ),
    )
    payloads: list[dict] = []

    def fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
        payloads.append(self._get_request_payload(messages, stop=stop, **kwargs))
        if len(payloads) == 1:
            raise RuntimeError("Function tools with reasoning_effort are not supported")
        return "retried"

    monkeypatch.setattr(ChatOpenAI, "_generate", fake_generate)
    assert llm._generate([HumanMessage(content="hello")], tools=[TOOL]) == "retried"
    assert payloads[0]["reasoning_effort"] == "high"
    assert payloads[1]["reasoning_effort"] == "none"


def test_insufficient_permission_error_does_not_retry(monkeypatch) -> None:
    llm = CompatibleChatOpenAI(api_key="test", model="gpt-5.6-luna")
    calls = 0

    def fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("Error code: 401 - You have insufficient permissions for this operation.")

    monkeypatch.setattr(ChatOpenAI, "_generate", fake_generate)
    with pytest.raises(RuntimeError, match="insufficient permissions"):
        llm._generate([HumanMessage(content="hello")])
    assert calls == 1


def test_streaming_uses_the_same_compatibility_decision(monkeypatch) -> None:
    llm = CompatibleChatOpenAI(api_key="test", model="gateway-model", reasoning_effort="high", compatibility_capabilities=ModelCapabilities(False))
    payloads: list[dict] = []

    def fake_stream(self, *args, **kwargs):
        payloads.append(self._get_request_payload(args[0], **kwargs))
        yield "chunk"

    monkeypatch.setattr(ChatOpenAI, "_stream", fake_stream)
    assert list(llm._stream([HumanMessage(content="hello")], tools=[TOOL])) == ["chunk"]
    assert payloads[0]["reasoning_effort"] == "none"
