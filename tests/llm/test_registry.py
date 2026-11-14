from __future__ import annotations
from datetime import datetime, timedelta, timezone
from mana_agent.llm.capabilities import CapabilityState, ModelProfile, metadata_facts, observation
from mana_agent.llm.registry import ModelCapabilityRegistry, scope_key

def test_model_names_and_missing_fields_do_not_grant_capabilities(tmp_path):
    profile = ModelProfile(provider="custom", model_id="gpt-reasoning-tools-vision-anything")
    assert profile.fact("tool_calling").state is CapabilityState.UNKNOWN
    assert metadata_facts({"id": profile.model_id}, source="provider") == {}

def test_metadata_does_not_conflate_related_capabilities():
    facts = metadata_facts({"tool_call": True, "reasoning": True, "streaming": True}, source="models.dev")
    assert "tool_choice" not in facts
    assert "reasoning_effort" not in facts
    assert "tool_call_streaming" not in facts

def test_field_precedence_retains_conflicts_and_expiry(tmp_path):
    registry = ModelCapabilityRegistry(tmp_path / "registry.json")
    profile = ModelProfile(provider="custom", model_id="model")
    for source, value in [("models.dev", True), ("provider", False), ("probe", True), ("override", False)]:
        profile = registry.merge(profile, {"tool_calling": observation(value, source=source, confidence=1)})
    assert profile.fact("tool_calling").source == "override"
    assert profile.fact("tool_calling").state is CapabilityState.UNSUPPORTED
    assert len(profile.observations["tool_calling"]) == 4
    registry.clear_overrides()
    assert registry.get("custom", "model").supports("tool_calling")
    expired = observation(True, source="probe").model_copy(update={"expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)})
    profile = registry.merge(profile, {"tool_calling": expired})
    assert profile.fact("tool_calling").source == "provider"
    assert not profile.supports("tool_calling")

def test_probe_scope_includes_endpoint_version_configuration_and_account_revision():
    base = dict(provider="custom", model="model", endpoint="https://example.org/v1")
    first = scope_key(**base)
    assert first != scope_key(**{**base, "endpoint": "https://other.example.org/v1"})
    assert first != scope_key(**base, api_version="v2")
    assert first != scope_key(**base, configuration={"reasoning_effort": "low"})
    assert first != scope_key(**base, credential_revision="new-account")

def test_scoped_probe_is_not_reused_for_another_configuration(tmp_path):
    registry = ModelCapabilityRegistry(tmp_path / "registry.json")
    profile = ModelProfile(provider="custom", model_id="model", configuration_fingerprint="scope-a")
    registry.merge(profile, {"tool_calling": observation(True, source="probe", scope="scope-a")}, persist=True)
    assert registry.get("custom", "model", "scope-b") is None
    reloaded = ModelCapabilityRegistry(tmp_path / "registry.json")
    assert reloaded.get("custom", "model", "scope-a").supports("tool_calling")


def test_value_evidence_does_not_promote_untested_levels_and_override_wins(tmp_path):
    registry = ModelCapabilityRegistry(tmp_path / "registry.json")
    profile = ModelProfile(provider="custom", model_id="model")
    profile = registry.merge(profile, {"reasoning_effort": observation(["low", "high"], source="provider")})
    profile = registry.merge(profile, {"reasoning_effort.low": observation(False, source="probe")})
    assert not profile.supports_value("reasoning_effort", "low")
    assert profile.supports_value("reasoning_effort", "high")
    profile = registry.merge(profile, {"reasoning_effort": observation(["low"], source="override")})
    assert profile.supports_value("reasoning_effort", "low")
    assert not profile.supports_value("reasoning_effort", "high")


def test_deleted_settings_override_does_not_survive_in_memory(native_profile, monkeypatch):
    from mana_agent.config import user_config
    from mana_agent.llm.registry import resolve_profile

    settings = {"MANA_MODEL_CAPABILITY_OVERRIDES": [{
        "provider": "openai", "model_id": "test-model", "endpoint": "https://provider.example/v1",
        "capabilities": {"streaming": True},
    }]}
    monkeypatch.setattr(user_config, "get_setting", lambda name, default=None: settings.get(name, default))
    resolve = lambda: resolve_profile("openai", "test-model", endpoint="https://provider.example/v1", catalog_records=[])
    assert resolve().supports("streaming")
    settings.clear()
    assert resolve().fact("streaming").state is CapabilityState.UNKNOWN
