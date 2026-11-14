from __future__ import annotations
import pytest
from mana_agent.llm.capabilities import ModelProfile, observation
from mana_agent.llm.registry import GLOBAL_CAPABILITY_REGISTRY

@pytest.fixture
def native_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(GLOBAL_CAPABILITY_REGISTRY, "_cache_path", tmp_path / "capabilities.json")
    monkeypatch.setattr(GLOBAL_CAPABILITY_REGISTRY, "_profiles", {})
    monkeypatch.setattr(GLOBAL_CAPABILITY_REGISTRY, "_loaded", True)
    facts = {name: observation(value, source="provider", confidence=1) for name, value in {
        "input_text": True, "output_text": True, "context_window": 65536, "max_output_tokens": 1024,
        "tool_calling": True, "tool_choice": ["required"], "native_structured_output": True,
        "streaming": True, "tool_call_streaming": True, "reasoning": True, "reasoning_effort": ["low"],
    }.items()}
    profile = ModelProfile(provider="openai", model_id="test-model", facts=facts)
    GLOBAL_CAPABILITY_REGISTRY.register(profile)
    return profile
