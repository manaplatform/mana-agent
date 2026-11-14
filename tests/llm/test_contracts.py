from __future__ import annotations
import pytest
from pydantic import BaseModel, ValidationError
from mana_agent.llm.messages import AssistantMessage, ContentBlock, UserMessage, ToolCall, ToolResultMessage, messages_from_records, validate_conversation
from mana_agent.llm.tools import ToolBinding
from mana_agent.llm.types import ModelResponse, ContinuationState

def test_native_history_retains_multimodal_content_and_correlated_results():
    messages = [UserMessage(content=[ContentBlock(type="image_url", image_url={"url": "https://example.org/image"})]),
        AssistantMessage(tool_calls=[ToolCall(id="call-1", name="echo", args={"text": "yes"})]),
        ToolResultMessage(content='{"value":"yes"}', tool_call_id="call-1")]
    restored = messages_from_records([message.model_dump(mode="json") for message in messages])
    validate_conversation(restored)
    assert restored == messages


def test_malformed_content_block_requires_a_format():
    with pytest.raises(ValidationError):
        UserMessage(content=[{"text": "missing type"}])

@pytest.mark.parametrize("messages", [
    [ToolResultMessage(content="orphan", tool_call_id="absent")],
    [AssistantMessage(tool_calls=[ToolCall(id="call-1", name="echo", args={})]), UserMessage("interrupted")],
])
def test_missing_tool_correlation_fails_without_synthetic_calls(messages):
    with pytest.raises(ValueError):
        validate_conversation(messages)

def test_provider_continuation_never_enters_visible_message():
    response = ModelResponse(content="answer", continuation=ContinuationState(provider="openai", model="test-model", endpoint="responses", items=[{"type": "reasoning", "encrypted_content": "opaque"}]))
    assert "continuation" not in response.message.model_dump()
    assert "opaque" not in response.message.model_dump_json()

def test_native_tool_validates_exact_arguments_and_typed_output():
    class Output(BaseModel):
        length: int
    calls = []
    def count(text: str) -> dict:
        calls.append(text)
        return {"length": len(text)}
    tool = ToolBinding.from_callable(count, output_model=Output)
    assert tool.call({"text": "mana"}) == {"length": 4}
    with pytest.raises(ValueError):
        tool.call({"text": "mana", "unregistered": True})
    assert calls == ["mana"]

def test_native_tool_does_not_coerce_wrong_input_types():
    def count(value: int):
        return value
    tool = ToolBinding.from_callable(count)
    with pytest.raises(ValueError):
        tool.call({"value": "3"})
