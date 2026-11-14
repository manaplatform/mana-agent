from __future__ import annotations

from pathlib import Path

import pytest

from mana_agent.config import user_config
from mana_agent.config.inference_provider import (
    ProviderConfigurationError,
    credentials_from_mapping,
    resolve_inference_connection,
)
from mana_agent.config.provider_registry import (
    CodexTransport,
    PROVIDERS,
    provider_credential_env_names,
    qualify_model_id,
    split_qualified_model_id,
)
from mana_agent.config.session import ConfigurationDraft
from mana_agent.config.settings import Settings
from mana_agent.config.catalog_service import ModelCatalogService
from mana_agent.config.model_catalog import (
    ModelCapability,
    ModelPurpose,
    descriptors_from_catalog,
    filter_models,
    cached_token_limits,
    normalize_capabilities,
)
from mana_agent.llm.engine import create_model
from mana_agent.tui.model_management import configured_agent_models
from mana_agent.tui.model_picker import parse_openai_compatible_model_records


@pytest.fixture()
def isolated_xai_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config_dir = tmp_path / ".mana"
    monkeypatch.setenv("MANA_HOME", str(config_dir))
    for key in (
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "OPENAI_BASE_URL",
        "XAI_BASE_URL",
        "MANA_AI_PROVIDER",
        "MANA_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    return config_dir


def test_xai_is_registered_for_cli_and_openai_compatible_runtime() -> None:
    provider = PROVIDERS.get("xai")
    assert provider.display_name == "xAI (Grok)"
    assert provider.default_base_url == "https://api.x.ai/v1"
    assert provider.api_key_env == "XAI_API_KEY"
    assert provider.supports_responses_api is False
    assert provider.codex_transport is CodexTransport.RESPONSES_BRIDGE
    assert provider_credential_env_names("xai") == ("XAI_API_KEY", "XAI_BASE_URL")


def test_xai_model_ids_preserve_upstream_id() -> None:
    assert qualify_model_id("xai", "grok-4") == "xai/grok-4"
    assert split_qualified_model_id("xai/grok-4", default_provider="xai") == (
        "xai",
        "grok-4",
    )
    assert split_qualified_model_id("grok-4", default_provider="xai") == (
        "xai",
        "grok-4",
    )


def test_xai_connection_uses_isolated_credentials(
    isolated_xai_config: Path,
) -> None:
    user_config.save_effective_user_config(
        {
            "MANA_AI_PROVIDER": "xai",
            "XAI_API_KEY": "xai-test-key",
            "XAI_BASE_URL": "https://api.x.ai/v1",
            "OPENAI_API_KEY": "openai-different-key",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        },
        merge=False,
    )

    connection = resolve_inference_connection(Settings())
    assert connection.provider == "xai"
    assert connection.api_key == "xai-test-key"
    assert connection.base_url == "https://api.x.ai/v1"
    assert not connection.supports_responses_api


def test_xai_missing_key_does_not_fall_back_to_openai() -> None:
    with pytest.raises(ProviderConfigurationError, match="XAI_API_KEY"):
        resolve_inference_connection(
            Settings(openai_api_key="openai-only"), provider="xai"
        )


def test_internal_engine_does_not_use_openai_key_for_xai(
    isolated_xai_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_config.save_effective_user_config(
        {"MANA_AI_PROVIDER": "xai", "OPENAI_API_KEY": "openai-only"},
        merge=False,
    )
    calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        "mana_agent.llm.engine.create_model",
        lambda **kwargs: calls.append(kwargs),
    )

    from mana_agent.coding.internal_engine import InternalCodingEngine

    engine = InternalCodingEngine(model="grok-4", provider="xai")
    with pytest.raises(ProviderConfigurationError):
        engine._get_or_create_llm()
    assert calls == []


def test_xai_credentials_from_mapping_are_isolated() -> None:
    key, base_url = credentials_from_mapping(
        {
            "XAI_API_KEY": "xai-key",
            "XAI_BASE_URL": "https://api.x.ai/v1",
            "OPENAI_API_KEY": "openai-key",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        },
        provider="xai",
    )
    assert key == "xai-key"
    assert base_url == "https://api.x.ai/v1"


def test_xai_configuration_saves_secret_separately(
    isolated_xai_config: Path,
) -> None:
    draft = ConfigurationDraft.load()
    draft.set_secret("XAI_API_KEY", "xai-private-test-key")
    draft.values["XAI_BASE_URL"] = "https://api.x.ai/v1"
    draft.set_models(
        provider="xai",
        high="grok-4",
        coding="grok-4",
        fast="grok-4",
        
    )
    draft.save()

    values = user_config.load_effective_settings(include_env=False)
    assert values["MANA_AI_PROVIDER"] == "xai"
    assert values["MANA_PRIMARY_MODEL"] == "xai/grok-4"
    assert user_config.load_user_secrets()["XAI_API_KEY"] == "xai-private-test-key"
    assert "xai-private-test-key" not in (isolated_xai_config / "config.toml").read_text()


def test_xai_catalog_and_chat_transport_use_openai_compatible_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = parse_openai_compatible_model_records(
        {"data": [{"id": "grok-4"}, {"id": "grok-3-mini"}]}
    )
    assert [record["id"] for record in records] == ["grok-3-mini", "grok-4"]

    monkeypatch.setattr(
        "mana_agent.config.user_config.get_setting",
        lambda _name, default=None: default,
    )
    model = create_model(
        api_key="xai-test-key",
        model="grok-4",
        base_url="https://api.x.ai/v1",
        provider="xai",
    )
    assert model.selected_provider == "xai"
    assert model.endpoint == "chat_completions"
    assert model.model_name == "grok-4"


def test_xai_missing_credentials_in_mapping_do_not_use_openai() -> None:
    key, base_url = credentials_from_mapping(
        {
            "MANA_AI_PROVIDER": "xai",
            "OPENAI_API_KEY": "openai-key",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        },
        provider="xai",
    )
    assert key == ""
    assert base_url == "https://api.x.ai/v1"


def test_xai_model_registry_defaults_are_settings_backed() -> None:
    assert Settings.model_fields["xai_api_key"].alias == "XAI_API_KEY"
    assert Settings.model_fields["xai_base_url"].default == "https://api.x.ai/v1"


def test_xai_model_caching_and_cached_returns_models(
    isolated_xai_config: Path,
) -> None:
    records = parse_openai_compatible_model_records(
        {"data": [{"id": "grok-4"}, {"id": "grok-3-mini"}]},
        provider="xai",
    )
    user_config.save_model_cache("xai", "https://api.x.ai/v1", records)

    # 1. Loading under "xai"
    cached = user_config.load_model_cache("xai", "https://api.x.ai/v1")
    assert cached is not None
    assert len(cached.models) == 2

    # 2. Loading under alias "x-ai"
    cached_alias = user_config.load_model_cache("x-ai", "https://api.x.ai/v1")
    assert cached_alias is not None
    assert len(cached_alias.models) == 2

    # 3. ModelCatalogService.cached() under "xai" and "x-ai"
    svc = ModelCatalogService()
    descriptors = svc.cached(provider="xai", base_url="https://api.x.ai/v1")
    assert len(descriptors) == 2
    assert [d.id for d in descriptors] == ["grok-3-mini", "grok-4"]
    for d in descriptors:
        assert ModelCapability.TEXT_GENERATION in d.capabilities
        assert ModelCapability.TOOL_CALLING in d.capabilities

    descriptors_alias = svc.cached(provider="x-ai", base_url="https://api.x.ai/v1")
    assert len(descriptors_alias) == 2
    assert [d.id for d in descriptors_alias] == ["grok-3-mini", "grok-4"]

    # 4. filter_models purpose=AGENT does not return 0 models
    agent_models = filter_models(descriptors, ModelPurpose.AGENT)
    assert len(agent_models) == 2
    assert [m.id for m in agent_models] == ["grok-3-mini", "grok-4"]


def test_xai_model_catalog_capabilities_and_token_limits() -> None:
    assert not normalize_capabilities("xai", "grok-4")
    assert cached_token_limits("xai", "grok-4") is None


def test_xai_parse_openai_compatible_model_records_populates_capabilities() -> None:
    records = parse_openai_compatible_model_records(
        {"data": [{"id": "grok-4"}, {"id": "grok-2-vision-1212"}]},
        provider="xai",
    )
    assert len(records) == 2
    grok4_rec = next(r for r in records if r["id"] == "grok-4")
    assert "capabilities" in grok4_rec
    assert "text_generation" in grok4_rec["capabilities"]
    assert "tool_calling" in grok4_rec["capabilities"]

    vis_rec = next(r for r in records if r["id"] == "grok-2-vision-1212")
    assert "image_input" in vis_rec["capabilities"]

    descriptors = descriptors_from_catalog("xai", records)
    agent_models = filter_models(descriptors, ModelPurpose.AGENT)
    assert [m.id for m in agent_models] == ["grok-2-vision-1212", "grok-4"]


def test_xai_provider_alias_interoperability() -> None:
    assert PROVIDERS.get("x-ai").id == "xai"
    assert qualify_model_id("x-ai", "grok-4") == "xai/grok-4"
    assert split_qualified_model_id("x-ai/grok-4") == ("xai", "grok-4")
    assert provider_credential_env_names("x-ai") == ("XAI_API_KEY", "XAI_BASE_URL")
    key, base_url = credentials_from_mapping(
        {"XAI_API_KEY": "test-key", "XAI_BASE_URL": "https://api.x.ai/v1"},
        provider="x-ai",
    )
    assert key == "test-key"
    assert base_url == "https://api.x.ai/v1"


def test_xai_configured_agent_models_returns_cached_models(
    isolated_xai_config: Path,
) -> None:
    records = parse_openai_compatible_model_records(
        {"data": [{"id": "grok-4"}]},
        provider="xai",
    )
    user_config.save_model_cache("xai", "https://api.x.ai/v1", records)

    models_xai = configured_agent_models(
        {
            "MANA_AI_PROVIDER": "xai",
            "XAI_API_KEY": "test-key",
            "XAI_BASE_URL": "https://api.x.ai/v1",
        }
    )
    assert len(models_xai) >= 1
    assert any(m.id == "grok-4" for m in models_xai)

    models_xai_alias = configured_agent_models(
        {
            "MANA_AI_PROVIDER": "x-ai",
            "XAI_API_KEY": "test-key",
            "XAI_BASE_URL": "https://api.x.ai/v1",
        }
    )
    assert len(models_xai_alias) >= 1
    assert any(m.id == "grok-4" for m in models_xai_alias)


def test_xai_set_models_decoupled_from_embedding() -> None:
    draft = ConfigurationDraft(
        original={"MANA_AI_PROVIDER": "xai"},
        values={"MANA_AI_PROVIDER": "xai"},
    )
    draft.set_models(
        provider="xai",
        high="grok-4",
        coding="grok-4",
        fast="grok-4",
        
    )
    assert draft.values["MANA_PRIMARY_MODEL"] == "xai/grok-4"
    assert draft.values["OPENAI_CHAT_MODEL"] == "grok-4"
    assert draft.values["LLM_MODEL"] == "grok-4"
    assert draft.values["OPENAI_CODING_PLANNER_MODEL"] == "grok-4"
    assert draft.values["OPENAI_TOOL_WORKER_MODEL"] == "grok-4"
    assert draft.values["MODEL_LEVEL_3_HIGH_REASONING"] == "grok-4"
    assert "MANA_EMBEDDING_MODEL" not in draft.values


def test_xai_names_cannot_grant_structured_output() -> None:
    for name in ("grok-build-0.1", "grok-2", "grok-4", "grok-beta"):
        assert ModelCapability.STRUCTURED_OUTPUT not in normalize_capabilities("xai", name)


def test_xai_missing_evidence_is_unknown(tmp_path) -> None:
    from mana_agent.llm.registry import ModelCapabilityRegistry
    from mana_agent.llm.discovery import ModelMetadataResolver
    registry = ModelCapabilityRegistry(tmp_path / "capabilities.json")
    profile = ModelMetadataResolver(registry).resolve("grok-build-0.1", "xai")
    assert not profile.is_known
    assert not profile.supports_structured_output


def test_default_model_configuration_requires_explicit_choices() -> None:
    from mana_agent.config.user_config import DEFAULT_USER_CONFIG
    for key in ("OPENAI_CHAT_MODEL", "OPENAI_TOOL_WORKER_MODEL", "OPENAI_CODING_PLANNER_MODEL", "MANA_DECISIONS_MODEL"):
        assert DEFAULT_USER_CONFIG[key] == ""


def test_configuration_app_keeps_manual_selection_without_invented_options() -> None:
    from mana_agent.tui.configuration_app import ManaConfigurationApp
    draft = ConfigurationDraft(original={}, values={"MANA_AI_PROVIDER": "xai", "OPENAI_CHAT_MODEL": "operator-model"})
    app = ManaConfigurationApp(draft=draft)
    assert app.draft.values["OPENAI_CHAT_MODEL"] == "operator-model"

