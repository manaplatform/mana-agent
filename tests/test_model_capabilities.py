"""Capability evidence replaces maintained model-name tables."""
from __future__ import annotations
import pytest
from mana_agent.llm.capabilities import CapabilityState, ModelProfile, metadata_facts, observation
from mana_agent.llm.errors import CapabilityError

@pytest.mark.parametrize("name", ["gpt-6-astra", "grok-4", "deepseek-v4", "claude-reasoning", "unknown-tools"])
def test_names_cannot_establish_capabilities(name):
    profile = ModelProfile(provider="custom",model_id=name,facts=metadata_facts({"id":name},source="provider"))
    assert profile.fact("tool_calling").state is CapabilityState.UNKNOWN
    with pytest.raises(CapabilityError):
        profile.require("tool_calling")

def test_exact_metadata_grants_only_reported_features():
    profile = ModelProfile(provider="custom",model_id="model",facts=metadata_facts({"tool_call":True,"reasoning":False,"limit":{"context":1000,"output":100}},source="provider"))
    assert profile.supports_tool_calls
    assert not profile.supports_reasoning
    assert profile.context_window == 1000
    assert profile.fact("tool_choice").state is CapabilityState.UNKNOWN

def test_explicit_unknown_is_distinct_from_unsupported():
    profile = ModelProfile(provider="custom",model_id="model",facts={"tool_calling":observation(False,source="provider")})
    assert profile.fact("tool_calling").state is CapabilityState.UNSUPPORTED
    assert profile.fact("streaming").state is CapabilityState.UNKNOWN
