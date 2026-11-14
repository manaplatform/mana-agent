from __future__ import annotations
import asyncio
import pytest
from pydantic import BaseModel
from mana_agent.llm.capabilities import observation
from mana_agent.llm.engine import NativeLLMEngine
from mana_agent.llm.errors import CapabilityError, InvalidResponseError, InferenceTimeoutError
from mana_agent.llm.messages import UserMessage
from mana_agent.llm.types import ModelResponse

def engine(profile, monkeypatch, adapter):
    value = NativeLLMEngine(provider="openai", model="test-model", api_key="test", base_url="https://example.org", endpoint="responses", adapter=adapter)
    monkeypatch.setattr(value, "profile", lambda *args, **kwargs: profile)
    return value

def test_unknown_capability_blocks_before_provider_call(native_profile, monkeypatch):
    calls = []
    class Adapter:
        async def generate(self, request, **_native_options):
            calls.append(request)
    native_profile.facts.pop("tool_calling")
    value = engine(native_profile, monkeypatch, Adapter()).with_tools([{"name": "echo", "parameters": {"type": "object"}}])
    with pytest.raises(CapabilityError):
        value.generate_sync([UserMessage("hello")])
    assert calls == []

def test_invalid_native_structured_output_never_retries_as_text(native_profile, monkeypatch):
    class Decision(BaseModel):
        approved: bool
    class Adapter:
        calls = 0
        async def generate(self, request, **_native_options):
            self.calls += 1
            assert request.response_schema
            return ModelResponse(content="not JSON")
    adapter = Adapter()
    value = engine(native_profile, monkeypatch, adapter)
    with pytest.raises(ValueError):
        value.decide_sync("hello", Decision)
    assert adapter.calls == 1

def test_timeout_cancels_selected_request_without_other_backend(native_profile, monkeypatch):
    class Adapter:
        cancelled = False
        async def generate(self, request, **_native_options):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
    adapter = Adapter()
    value = engine(native_profile, monkeypatch, adapter)
    value.timeout = .001
    with pytest.raises(InferenceTimeoutError):
        value.generate_sync("hello")
    assert adapter.cancelled

def test_empty_response_does_not_prove_completion(native_profile, monkeypatch):
    class Adapter:
        async def generate(self, request, **_native_options):
            return ModelResponse()
    with pytest.raises(InvalidResponseError):
        engine(native_profile, monkeypatch, Adapter()).generate_sync("hello")


def test_conversation_receipt_cannot_replay_with_changed_history(native_profile, monkeypatch):
    from mana_agent.llm.types import ContinuationState
    class Adapter:
        calls = 0
        async def generate(self, request):
            self.calls += 1
            return ModelResponse(content="answer", continuation=ContinuationState(
                provider="openai", model="test-model", endpoint="responses", message_count=2))
    adapter = Adapter()
    model = engine(native_profile, monkeypatch, adapter)
    response = model.generate_sync("original")
    with pytest.raises(InvalidResponseError, match="history changed"):
        model.generate_sync([UserMessage("changed"), response.message], continuation=response.continuation)
    assert adapter.calls == 1


def test_native_chat_does_not_probe_unknown_features(native_profile, monkeypatch):
    from mana_agent.llm import probes
    async def forbidden(*args, **kwargs):
        pytest.fail("Chat must never probe capabilities")
    monkeypatch.setattr(probes, "run_capability_probe", forbidden)
    class Adapter:
        async def generate(self, request):
            return ModelResponse(content="answer")
    assert engine(native_profile, monkeypatch, Adapter()).generate_sync("hello").content == "answer"


def test_model_response_rejects_unregistered_and_invalid_tool_calls(native_profile, monkeypatch):
    from mana_agent.llm.messages import ToolCall
    class Adapter:
        async def generate(self, request):
            return ModelResponse(tool_calls=[ToolCall(id="call", name="echo", args={"value": "wrong type"})])
    model = engine(native_profile, monkeypatch, Adapter()).with_tools([
        {"name": "echo", "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}},
    ])
    with pytest.raises(InvalidResponseError, match="registered schema"):
        model.generate_sync("hello")
