from mana_agent.config.session import ConfigurationDraft

def test_provider_can_be_tested_before_any_model_is_selected():
    draft = ConfigurationDraft(original={},values={"OPENAI_CHAT_MODEL":"previous"})
    draft.set_models(provider="nvidia",high="",coding="",fast="")
    assert draft.values["MANA_AI_PROVIDER"] == "nvidia"
    assert draft.values["OPENAI_CHAT_MODEL"] == "previous"

def test_chat_model_selection_does_not_require_embeddings():
    draft = ConfigurationDraft(original={},values={})
    draft.set_models(provider="nvidia",high="org/model",coding="org/code",fast="org/fast")
    assert draft.values["MANA_PRIMARY_MODEL"] == "nvidia/org/model"
    assert "MANA_EMBEDDING_MODEL" not in draft.values
    assert "OPENAI_EMBED_MODEL" not in draft.values


def test_capability_override_roundtrips_as_structured_toml(monkeypatch, tmp_path):
    from mana_agent.config.user_config import load_user_config, save_user_config
    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    override = {"provider": "openai", "model_id": "selected", "capabilities": {"tool_choice": ["required"], "context_window": 4096}}
    save_user_config({"MANA_MODEL_CAPABILITY_OVERRIDES": [override]}, merge=False)
    assert load_user_config()["MANA_MODEL_CAPABILITY_OVERRIDES"] == [override]


def test_settings_offer_unclassified_discovered_models_without_selecting_one(monkeypatch, tmp_path):
    from mana_agent.config.model_catalog import ModelDescriptor
    from mana_agent.tui.configuration_app import ManaConfigurationApp

    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    app = ManaConfigurationApp(draft=ConfigurationDraft(original={}, values={}))
    unknown = ModelDescriptor(provider="openai", id="org/model", capabilities=frozenset())
    inactive = ModelDescriptor(provider="openai", id="inactive", capabilities=frozenset(), available=False)
    assert app._inference_model_options([unknown, inactive], current="configured") == [
        ("Select an inference model", ""), ("org/model  · Unknown", "org/model"), ("configured · configured", "configured"),
    ]
