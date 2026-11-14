"""NVIDIA identities, credentials and explicit native request parameters."""
from __future__ import annotations
from types import SimpleNamespace
from mana_agent.config.inference_provider import resolve_inference_connection
from mana_agent.config.provider_registry import qualify_model_id, split_qualified_model_id
from mana_agent.llm.providers.openai import OpenAIProvider
from mana_agent.llm.messages import UserMessage
from mana_agent.llm.types import InferenceRequest

def test_nvidia_credentials_never_use_openai_key():
    settings = SimpleNamespace(mana_ai_provider="nvidia",nvidia_api_key="nvidia-key",nvidia_base_url="https://example.org/v1",openai_api_key="wrong-key")
    connection = resolve_inference_connection(settings,provider="nvidia")
    assert connection.api_key == "nvidia-key"
    assert connection.provider == "nvidia"

def test_nvidia_nested_org_identity_roundtrips():
    for upstream in ("nvidia/nemotron-test", "deepseek-ai/test"):
        qualified = qualify_model_id("nvidia",upstream)
        assert split_qualified_model_id(qualified,default_provider="nvidia") == ("nvidia",upstream)

def test_nvidia_explicit_template_settings_are_forwarded_without_name_guesses():
    adapter = OpenAIProvider(provider="nvidia",api_key="test",base_url="https://example.org",endpoint="chat_completions")
    parameters = {"extra_body":{"chat_template_kwargs":{"thinking":False,"reasoning_effort":"none"}}}
    request = InferenceRequest(provider="nvidia",model="arbitrary-configured-model",messages=[UserMessage("hello")],parameters=parameters)
    body = adapter.encode(request)
    assert body["extra_body"] == parameters["extra_body"]
    assert "reasoning_effort" not in body
