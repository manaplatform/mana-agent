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
from mana_agent.multi_agent.runtime.compatibility import create_chat_model
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


def test_xai_cli_configuration_persists_dedicated_settings(
    isolated_xai_config: Path,
) -> None:
    draft = ConfigurationDraft.load()
    draft.set_secret("XAI_API_KEY", "xai-cli-key")
    draft.values["XAI_BASE_URL"] = "https://api.x.ai/v1"
    draft.values["MANA_AI_PROVIDER"] = "xai"
    draft.save()

    assert user_config.load_effective_settings(include_env=False)["XAI_BASE_URL"] == (
        "https://api.x.ai/v1"
    )
    assert user_config.load_user_secrets()["XAI_API_KEY"] == "xai-cli-key"
    assert "xai-cli-key" not in (isolated_xai_config / "config.toml").read_text()


def test_xai_model_registry_keeps_provider_choice_separate_from_openai() -> None:
    assert provider_credential_env_names("openai") == (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    )
    assert provider_credential_env_names("xai") == ("XAI_API_KEY", "XAI_BASE_URL")
    assert "xai" != "openai"


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
