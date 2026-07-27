from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mana_agent.commands.cli import app
from mana_agent.commands.chat_cli import _apply_session_model_change
from mana_agent.config import user_config
from mana_agent.config.catalog_service import ModelCatalogService
from mana_agent.config.model_catalog import (
    ModelPurpose,
    descriptors_from_catalog,
    filter_models,
)
from mana_agent.config.session import ConfigurationDraft
from mana_agent.search.config import SearchConfig
from mana_agent.tui.model_management import ModelSelection, plain_models_command, save_default_model
from mana_agent.tui.configuration_app import validate_github_connection, validate_search_connection


@pytest.fixture()
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    return tmp_path


def test_root_dispatches_chat_without_mode_menu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import mana_agent.commands.main_cli as main_cli

    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(main_cli, "ensure_setup", lambda **_kwargs: True)
    monkeypatch.setattr(main_cli, "_is_interactive_terminal", lambda: True)
    monkeypatch.setattr(
        main_cli,
        "_invoke_with_multi_agent_route",
        lambda _ctx, name, args, **_kwargs: calls.append((name, args)),
    )

    result = CliRunner().invoke(app, ["--repo", str(tmp_path), "--no-banner"])

    assert result.exit_code == 0, result.output
    assert calls == [("chat", ["--root-dir", str(tmp_path.resolve())])]


def test_chat_alias_is_registered_to_chat_callback() -> None:
    assert any(command.name == "chat" for command in app.registered_commands)


def test_configure_flag_launches_configuration_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    import mana_agent.commands.main_cli as main_cli

    calls: list[bool] = []
    monkeypatch.setattr(main_cli, "_is_interactive_terminal", lambda: True)
    monkeypatch.setattr(main_cli, "run_setup_wizard", lambda **_kwargs: calls.append(True) or True)

    result = CliRunner().invoke(app, ["--configure"])

    assert result.exit_code == 0, result.output
    assert calls == [True]


def test_root_non_tty_never_launches_textual(monkeypatch: pytest.MonkeyPatch) -> None:
    import mana_agent.commands.main_cli as main_cli

    monkeypatch.setattr(main_cli, "ensure_setup", lambda **_kwargs: True)
    monkeypatch.setattr(main_cli, "_is_interactive_terminal", lambda: False)
    result = CliRunner().invoke(app, ["--no-banner"])
    assert result.exit_code != 0
    assert "interactive" in result.output.lower()


def test_model_capability_filters_separate_agent_and_embedding_models() -> None:
    records = [
        {"id": "chat", "capabilities": ["text_generation", "tool_calling"]},
        {"id": "vectors", "capabilities": ["embedding"]},
        {"id": "pictures", "capabilities": ["image_generation"]},
        {"id": "speech", "capabilities": ["text_to_speech"]},
        {"id": "movie", "capabilities": ["video_generation"]},
        {"id": "unknown"},
    ]
    models = descriptors_from_catalog("custom", records)

    assert [model.id for model in filter_models(models, ModelPurpose.AGENT)] == ["chat"]
    assert [model.id for model in filter_models(models, ModelPurpose.EMBEDDING)] == ["vectors"]
    assert next(model for model in models if model.id == "unknown").capabilities == frozenset()


def test_provider_qualified_model_ids_do_not_collide() -> None:
    openai = descriptors_from_catalog("openai", [{"id": "same", "capabilities": ["text_generation"]}])[0]
    custom = descriptors_from_catalog("custom", [{"id": "same", "capabilities": ["text_generation"]}])[0]
    assert openai.qualified_id == "openai/same"
    assert custom.qualified_id == "custom/same"
    assert openai.qualified_id != custom.qualified_id


def test_reauthentication_preserves_selected_model(isolated_config: Path) -> None:
    user_config.save_effective_user_config(
        {"OPENAI_API_KEY": "old", "OPENAI_CHAT_MODEL": "chosen", "MANA_PRIMARY_MODEL": "openai/chosen"},
        merge=False,
    )
    draft = ConfigurationDraft.load()
    draft.set_secret("OPENAI_API_KEY", "new")
    draft.save()

    effective = user_config.load_effective_settings(include_env=False)
    assert effective["OPENAI_API_KEY"] == "new"
    assert effective["OPENAI_CHAT_MODEL"] == "chosen"
    assert effective["MANA_PRIMARY_MODEL"] == "openai/chosen"


def test_unchanged_masked_secret_is_preserved_and_removal_is_explicit(isolated_config: Path) -> None:
    user_config.save_effective_user_config({"OPENAI_API_KEY": "secret"}, merge=False)
    draft = ConfigurationDraft.load()
    draft.set_secret("OPENAI_API_KEY", "")
    draft.save()
    assert user_config.load_user_secrets()["OPENAI_API_KEY"] == "secret"

    draft.remove_secret("OPENAI_API_KEY")
    draft.save()
    assert user_config.load_user_secrets()["OPENAI_API_KEY"] == ""


def test_atomic_write_preserves_previous_file_on_replace_failure(isolated_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user_config.save_user_config({"OPENAI_CHAT_MODEL": "old"}, merge=False)
    previous = user_config.config_file().read_text(encoding="utf-8")
    monkeypatch.setattr(user_config.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("disk")))

    with pytest.raises(user_config.UserConfigError):
        user_config.save_user_config({"OPENAI_CHAT_MODEL": "new"}, merge=False)

    assert user_config.config_file().read_text(encoding="utf-8") == previous


def test_legacy_migration_is_idempotent_and_moves_secrets(isolated_config: Path) -> None:
    user_config.config_file().write_text('OPENAI_API_KEY = "legacy-secret"\nOPENAI_CHAT_MODEL = "legacy-model"\nUNKNOWN = "keep"\n')

    first = user_config.migrate_legacy_config()
    once = user_config.config_file().read_text(encoding="utf-8")
    second = user_config.migrate_legacy_config()

    assert first and second == []
    assert user_config.config_file().read_text(encoding="utf-8") == once
    assert "OPENAI_API_KEY" not in user_config.load_user_config()
    assert user_config.load_user_secrets()["OPENAI_API_KEY"] == "legacy-secret"
    assert user_config.load_user_config()["UNKNOWN"] == "keep"
    assert (isolated_config / "config.toml.bak").exists()


def test_catalog_service_uses_mock_adapter_without_network(isolated_config: Path) -> None:
    service = ModelCatalogService(fetcher=lambda **_kwargs: ["gpt-4.1-mini", "text-embedding-3-small"])
    models = service.refresh(provider="openai", base_url="https://example.test/v1", api_key="secret")
    assert {model.id for model in models} == {"gpt-4.1-mini", "text-embedding-3-small"}
    assert json.loads(user_config.model_cache_file().read_text(encoding="utf-8"))


def test_models_plain_set_allows_only_configured_compatible_models(isolated_config: Path) -> None:
    user_config.save_effective_user_config(
        {
            "OPENAI_API_KEY": "secret",
            "MANA_AI_PROVIDER": "openai",
            "MANA_CONFIGURED_PROVIDERS": ["openai"],
            "OPENAI_BASE_URL": "https://example.test/v1",
            "OPENAI_CHAT_MODEL": "gpt-4.1-mini",
        },
        merge=False,
    )
    user_config.save_model_cache("openai", "https://example.test/v1", ["gpt-4.1-mini", "text-embedding-3-small"])

    message, selection = plain_models_command("/models set openai/gpt-4.1-mini", current_model="gpt-4.1-mini")
    assert selection == ModelSelection("openai", "gpt-4.1-mini", persist=False)
    assert "changed" in message
    _, rejected = plain_models_command("/models set custom/gpt-4.1-mini", current_model="gpt-4.1-mini")
    assert rejected is None


def test_session_and_persistent_model_changes_are_distinct(isolated_config: Path) -> None:
    user_config.save_effective_user_config({"OPENAI_CHAT_MODEL": "old", "MANA_PRIMARY_MODEL": "openai/old"}, merge=False)
    session = ModelSelection("openai", "new", persist=False)
    assert user_config.load_user_config()["OPENAI_CHAT_MODEL"] == "old"
    save_default_model(ModelSelection("openai", "new", persist=True))
    assert user_config.load_user_config()["OPENAI_CHAT_MODEL"] == "new"
    assert session.persist is False


def test_session_model_change_updates_active_clients_and_rolls_back_on_failure() -> None:
    class Client:
        def __init__(self, model: str, *, fail: bool = False) -> None:
            self.model = model
            self.fail = fail

        def update_model(self, model: str) -> None:
            if self.fail:
                raise RuntimeError("no")
            self.model = model

    first = Client("old")
    second = Client("old")
    _apply_session_model_change("new", first, second)
    assert first.model == second.model == "new"

    failing = Client("old", fail=True)
    with pytest.raises(RuntimeError):
        _apply_session_model_change("other", first, failing)
    assert first.model == "new"


def test_search_and_github_credential_sources_remain_separate(isolated_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user_config.save_effective_user_config(
        {
            "MANA_SEARCH_ENABLE_WEB": True,
            "MANA_WEB_SEARCH_PROVIDER": "brave",
            "MANA_WEB_SEARCH_API_KEY": "search-secret",
            "MANA_SEARCH_ENABLE_GITHUB": True,
            "MANA_GITHUB_CREDENTIAL_SOURCE": "environment",
            "MANA_GITHUB_SECRET_REF": "TEST_GH_TOKEN",
        },
        merge=False,
    )
    monkeypatch.setenv("TEST_GH_TOKEN", "github-secret")
    config = SearchConfig.from_env()
    assert config.web_provider == "brave"
    assert config.web_api_key == "search-secret"
    assert config.github_credential_source == "environment"
    assert config.github_token == "github-secret"


def test_search_and_github_connection_tests_use_existing_provider_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    search_calls: list[str] = []
    monkeypatch.setattr(
        "mana_agent.search.web_provider.ConfiguredWebSearchProvider.search_sync",
        lambda self, query, max_results=1: search_calls.append(f"{self.provider}:{query}:{max_results}") or [],
    )
    validate_search_connection(
        {
            "MANA_WEB_SEARCH_PROVIDER": "brave",
            "MANA_WEB_SEARCH_API_KEY": "secret",
            "MANA_SEARCH_TIMEOUT_SECONDS": 1,
        }
    )
    assert search_calls and search_calls[0].startswith("brave:")

    monkeypatch.setattr(
        "mana_agent.search.github_provider.GitHubSearchProvider._get_json",
        lambda _self, url: {"login": "mana-user", "url": url},
    )
    assert validate_github_connection(
        {"MANA_GITHUB_CREDENTIAL_SOURCE": "token", "MANA_GITHUB_TOKEN": "secret"}
    ) == "mana-user"


def test_models_slash_command_opens_modal() -> None:
    import asyncio
    from mana_agent.tui.app import ManaChatApp
    from mana_agent.tui.model_management import ModelManagementScreen

    async def run() -> None:
        chat = ManaChatApp(model="gpt-4.1-mini")
        async with chat.run_test() as pilot:
            await pilot.pause()
            input_widget = chat.query_one("#chat-input")
            input_widget.value = "/models"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(chat.screen, ModelManagementScreen)

    asyncio.run(run())
