"""Native Anthropic conversion and incomplete-output regressions."""
from __future__ import annotations
import pytest
from mana_agent.llm.messages import UserMessage, SystemMessage, AssistantMessage, ToolCall, ToolResultMessage
from mana_agent.llm.providers.anthropic import AnthropicProvider
from mana_agent.llm.types import InferenceRequest
from mana_agent.llm.errors import InvalidResponseError

def test_anthropic_authentication_is_provider_isolated():
    adapter = AnthropicProvider(provider="anthropic", api_key="anthropic-test", base_url="https://example.org", endpoint="anthropic_messages")
    assert adapter.options["api_key"] == "anthropic-test"
    assert adapter.options["max_retries"] == 0

def test_anthropic_preserves_system_content_and_tool_correlation():
    adapter = AnthropicProvider(provider="anthropic", api_key="test", base_url="https://example.org", endpoint="anthropic_messages")
    request = InferenceRequest(provider="anthropic", model="configured", parameters={"max_tokens":128}, messages=[SystemMessage("rules"), UserMessage("hello"), AssistantMessage(tool_calls=[ToolCall(id="one",name="echo",args={})]), ToolResultMessage("exact",tool_call_id="one")])
    body = adapter.encode(request)
    assert body["system"] == [{"type":"text","text":"rules"}]
    assert body["messages"][-1]["content"][0] == {"type":"tool_result","tool_use_id":"one","content":"exact"}

@pytest.mark.parametrize("stop_reason", ["max_tokens", "pause_turn", "refusal"])
def test_anthropic_does_not_promote_partial_output_to_success(stop_reason):
    adapter = AnthropicProvider(provider="anthropic", api_key="test", base_url="https://example.org", endpoint="anthropic_messages")
    request = InferenceRequest(provider="anthropic",model="configured",messages=[UserMessage("hello")])
    with pytest.raises(InvalidResponseError):
        adapter.normalize({"stop_reason":stop_reason,"content":[]},request)
