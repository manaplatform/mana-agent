"""Native transport selection and explicit configuration regressions."""
from __future__ import annotations
import pytest
from mana_agent.llm.engine import create_model
from mana_agent.llm.providers.openai import OpenAIProvider
from mana_agent.llm.types import InferenceRequest
from mana_agent.llm.messages import UserMessage

@pytest.mark.parametrize("provider,endpoint", [("openai", "responses"), ("nvidia", "chat_completions"), ("xai", "chat_completions"), ("openrouter", "chat_completions"), ("custom", "chat_completions"), ("anthropic", "anthropic_messages")])
def test_native_provider_protocol_selection(provider, endpoint, monkeypatch):
    model = create_model(provider=provider, model="explicit-test-model", api_key="test", base_url="https://example.org/v1")
    assert model.endpoint == endpoint
    assert model.provider == provider

def test_retired_sdk_construction_kwargs_fail_explicitly():
    with pytest.raises(ValueError, match="model_kwargs"):
        create_model(provider="openai", model="test", api_key="test", model_configuration={"model_kwargs": {"x": True}})

def test_chat_tools_do_not_silently_disable_reasoning():
    adapter = OpenAIProvider(provider="custom", api_key="test", base_url="https://example.org", endpoint="chat_completions")
    request = InferenceRequest(provider="custom", model="test", messages=[UserMessage("hello")], tools=[{"name":"echo","parameters":{"type":"object"}}], parameters={"reasoning_effort":"high"})
    assert adapter.encode(request)["reasoning_effort"] == "high"
