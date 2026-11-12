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
    maintained_token_limits,
    normalize_capabilities,
)
from mana_agent.multi_agent.runtime.compatibility import create_chat_model
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
        "mana_agent.multi_agent.runtime.compatibility.create_chat_model",
        lambda **kwargs: calls.append(kwargs),
    )

    from mana_agent.coding.internal_engine import InternalCodingEngine

    engine = InternalCodingEngine(model="grok-4", provider="xai")
    assert engine._get_or_create_llm() is None
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
        embedding="text-embedding-3-small",
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
        "mana_agent.multi_agent.runtime.compatibility.get_setting",
        lambda _name, default=None: default,
    )
    model = create_chat_model(
        api_key="xai-test-key",
        model="grok-4",
        base_url="https://api.x.ai/v1",
        provider="xai",
    )
    assert model.selected_provider == "xai"
    assert model.compatibility_api_mode == "chat_completions"
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
    caps_grok4 = normalize_capabilities("xai", "grok-4")
    assert ModelCapability.TEXT_GENERATION in caps_grok4
    assert ModelCapability.TOOL_CALLING in caps_grok4
    assert ModelCapability.CODE in caps_grok4
    assert ModelCapability.REASONING in caps_grok4
    assert ModelCapability.STRUCTURED_OUTPUT in caps_grok4

    caps_grok_vis = normalize_capabilities("x-ai", "grok-2-vision-1212")
    assert ModelCapability.TEXT_GENERATION in caps_grok_vis
    assert ModelCapability.IMAGE_INPUT in caps_grok_vis

    assert maintained_token_limits("xai", "grok-4") == (500_000, 65_536)
    assert maintained_token_limits("x-ai", "grok-2") == (128_000, 8_192)


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
        embedding="",
    )
    assert draft.values["MANA_PRIMARY_MODEL"] == "xai/grok-4"
    assert draft.values["OPENAI_CHAT_MODEL"] == "grok-4"
    assert draft.values["LLM_MODEL"] == "grok-4"
    assert draft.values["OPENAI_CODING_PLANNER_MODEL"] == "grok-4"
    assert draft.values["OPENAI_TOOL_WORKER_MODEL"] == "grok-4"
    assert draft.values["MODEL_LEVEL_3_HIGH_REASONING"] == "grok-4"
    assert "MANA_EMBEDDING_MODEL" not in draft.values


def test_xai_grok_models_have_structured_output() -> None:
    for model_id in ("grok-build-0.1", "grok-2", "grok-4", "grok-beta"):
        caps = normalize_capabilities("xai", model_id)
        assert ModelCapability.STRUCTURED_OUTPUT in caps

    records = parse_openai_compatible_model_records(
        {"data": [{"id": "grok-build-0.1"}]},
        provider="xai",
    )
    assert "structured_output" in records[0]["capabilities"]


def test_xai_grok_capability_descriptor_resolution() -> None:
    from mana_agent.config.model_capabilities import ModelCapabilityRegistry

    registry = ModelCapabilityRegistry()
    desc = registry.resolve("xai", "grok-build-0.1")
    assert desc.is_known
    assert desc.supports_structured_output
    assert desc.supports_tool_calls

    desc_qualified = registry.resolve("xai", "xai/grok-build-0.1")
    assert desc_qualified.is_known
    assert desc_qualified.supports_structured_output


def test_default_model_configuration_and_routing_defaults() -> None:
    from mana_agent.config.settings import Settings
    from mana_agent.config.user_config import DEFAULT_USER_CONFIG

    assert DEFAULT_USER_CONFIG["OPENAI_CHAT_MODEL"] == "gpt-4.1-mini"
    assert DEFAULT_USER_CONFIG["OPENAI_TOOL_WORKER_MODEL"] == "gpt-6-luna"
    assert DEFAULT_USER_CONFIG["OPENAI_CODING_PLANNER_MODEL"] == "gpt-6-luna"

    settings = Settings()
    assert settings.openai_chat_model == "gpt-4.1-mini"
    assert settings.openai_tool_worker_model == "gpt-6-luna"
    assert settings.openai_coding_planner_model == "gpt-6-luna"


def test_configuration_app_model_options_and_selection() -> None:
    import asyncio
    from mana_agent.tui.configuration_app import ManaConfigurationApp

    draft = ConfigurationDraft(
        original={"MANA_AI_PROVIDER": "openai"},
        values={"MANA_AI_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"},
    )
    app = ManaConfigurationApp(draft=draft)

    async def run() -> None:
        async with app.run_test():
            high_select = app.query_one("#high-model")
            options = {val for _, val in high_select._options}
            assert "gpt-4.1-mini" in options
            assert "gpt-6-luna" in options
            assert high_select.value == "gpt-4.1-mini"

            coding_select = app.query_one("#coding-model")
            assert coding_select.value == "gpt-6-luna"

            fast_select = app.query_one("#fast-model")
            assert fast_select.value == "gpt-6-luna"

    asyncio.run(run())

