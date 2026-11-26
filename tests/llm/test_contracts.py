from __future__ import annotations
import asyncio
import json
from datetime import datetime
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


@pytest.mark.parametrize("async_call", [False, True])
def test_tool_observer_serializes_typed_arguments_without_changing_handler_values(async_call):
    from mana_agent.automations.runtime_tools import _Create
    from mana_agent.automations.service import AgentPromptJob, OnceTrigger

    class Observer:
        def __init__(self):
            self.started = []
            self.finished = []

        def tool_started(self, serialized, input_str, **kwargs):
            self.started.append((serialized, json.loads(input_str), kwargs))

        def tool_finished(self, output, **kwargs):
            self.finished.append((output, kwargs))

        def tool_failed(self, error, **kwargs):
            pytest.fail(f"Unexpected tool failure: {error}")

    received = []

    def handler(**values):
        received.append(values)
        return {"ok": True}

    async def async_handler(**values):
        return handler(**values)

    tool = ToolBinding.from_callable(handler, name="typed", input_model=_Create, coroutine=async_handler)
    observer = Observer()
    args = {
        "source_decision_id": "decision", "name": "Once", "timezone": "UTC",
        "trigger": {"type": "once", "run_at": "2030-01-01T00:00:00Z", "timezone": "UTC"},
        "job": {"type": "agent_prompt", "prompt": "Run once"},
        "idempotency_key": "test", "permission_references": ["permission"],
    }
    config = {"callbacks": [observer], "tool_call_id": "call-1"}
    result = asyncio.run(tool.call_async(args, config=config)) if async_call else tool.call(args, config=config)

    assert result == {"ok": True}
    assert isinstance(received[0]["trigger"], OnceTrigger)
    assert isinstance(received[0]["trigger"].run_at, datetime)
    assert isinstance(received[0]["job"], AgentPromptJob)
    assert observer.started[0][1]["trigger"]["type"] == "once"
    assert observer.started[0][1]["trigger"]["run_at"].startswith("2030-01-01T00:00:00")
    assert observer.started[0][2]["run_id"] == "call-1"
    assert observer.finished == [({"ok": True}, {"run_id": "call-1"})]


def test_tool_observer_redacts_nested_typed_secrets():
    class Payload(BaseModel):
        api_key: str

    class Input(BaseModel):
        payload: Payload

    class Observer:
        def tool_started(self, serialized, input_str, **kwargs):
            self.payload = json.loads(input_str)

        def tool_finished(self, output, **kwargs):
            pass

    observer = Observer()
    tool = ToolBinding.from_callable(lambda payload: payload.api_key, name="typed", input_model=Input)
    assert tool.call({"payload": {"api_key": "private-value"}}, config={"callbacks": [observer]}) == "private-value"
    assert observer.payload["payload"]["api_key"] == "***REDACTED***"


def test_user_message_accepts_message_id_and_turn_id_and_ignores_extra():
    msg = UserMessage(
        content="hello",
        message_id="user-7512573df588",
        turn_id="turn-62fbf1316d5d",
        created_at="2026-11-14T19:00:00Z",
    )
    assert msg.message_id == "user-7512573df588"
    assert msg.turn_id == "turn-62fbf1316d5d"
    assert msg.content == "hello"

    restored = messages_from_records([
        {
            "role": "user",
            "content": "restored query",
            "message_id": "user-7512573df588",
            "turn_id": "turn-62fbf1316d5d",
            "timestamp": "2026-11-14T19:00:00Z",
        }
    ])
    assert len(restored) == 1
    assert restored[0].message_id == "user-7512573df588"
    assert restored[0].turn_id == "turn-62fbf1316d5d"
    assert restored[0].content == "restored query"

