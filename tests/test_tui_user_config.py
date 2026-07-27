from __future__ import annotations

import json
from pathlib import Path

import pytest

from mana_agent.config import user_config
from mana_agent.config.settings import Settings
from mana_agent.multi_agent.core.types import AgentRole
from mana_agent.multi_agent.runtime.model_levels import resolve_model_for_role
from mana_agent.search.config import SearchConfig
from mana_agent.tui.menu import NonInteractivePromptError
from mana_agent.tui.model_picker import ModelFetchError, choose_models, fetch_openai_compatible_models, parse_model_ids
from mana_agent.tui.wizard import ensure_setup


@pytest.fixture()
def isolated_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_dir = tmp_path / ".mana"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANA_HOME", str(config_dir))
    for key in set(user_config.DEFAULT_USER_CONFIG) | set(user_config.FIELD_NAME_BY_ENV):
        monkeypatch.delenv(key, raising=False)
    for key in user_config.SECRET_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
    return config_dir


def test_user_config_save_load_and_settings_source(isolated_user_config: Path) -> None:
    user_config.save_effective_user_config(
        {
            "OPENAI_API_KEY": "secret-key",
            "OPENAI_BASE_URL": "https://api.example.test/v1",
            "OPENAI_CHAT_MODEL": "model-a",
            "MANA_SEARCH_ENABLE_WEB": False,
        },
        merge=False,
    )

    settings = Settings()

    assert settings.openai_api_key == "secret-key"
    assert settings.openai_base_url == "https://api.example.test/v1"
    assert settings.openai_chat_model == "model-a"
    assert settings.mana_search_enable_web is False
    assert (isolated_user_config / "secrets.toml").exists()


def test_environment_and_repository_dotenv_do_not_override_user_config(
    isolated_user_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_config.save_effective_user_config({"OPENAI_API_KEY": "file-key", "OPENAI_CHAT_MODEL": "file-model"}, merge=False)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "env-model")
    (isolated_user_config.parent / ".env").write_text(
        'OPENAI_API_KEY="dotenv-key"\nOPENAI_CHAT_MODEL="dotenv-model"\n',
        encoding="utf-8",
    )

    settings = Settings()

    assert settings.openai_api_key == "file-key"
    assert settings.openai_chat_model == "file-model"


def test_runtime_compatibility_settings_ignore_environment(
    isolated_user_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mana_agent.multi_agent.runtime.compatibility import create_chat_model

    user_config.save_effective_user_config(
        {
            "MANA_LLM_API_MODE": "chat_completions",
            "MANA_LLM_REASONING_EFFORT": "low",
            "MANA_LLM_SUPPORTS_RESPONSES_API": False,
        },
        merge=False,
    )
    monkeypatch.setenv("MANA_LLM_API_MODE", "responses")
    monkeypatch.setenv("MANA_LLM_REASONING_EFFORT", "high")
    monkeypatch.setenv("MANA_LLM_SUPPORTS_RESPONSES_API", "true")

    llm = create_chat_model(api_key="saved-key", model="saved-model", base_url="https://gateway.example/v1")

    assert llm.compatibility_api_mode == "chat_completions"
    assert llm.reasoning_effort == "low"
    assert llm.compatibility_capabilities.supports_responses_api is False


def test_tool_worker_connection_settings_ignore_environment(
    isolated_user_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mana_agent.multi_agent.runtime.tool_worker_process import ToolWorkerClient

    user_config.save_effective_user_config(
        {
            "OPENAI_API_KEY": "file-key",
            "OPENAI_BASE_URL": "https://saved.example/v1",
            "OPENAI_TOOL_WORKER_MODEL": "saved-worker-model",
        },
        merge=False,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("OPENAI_TOOL_WORKER_MODEL", "env-worker-model")
    settings = Settings()

    worker = ToolWorkerClient(
        api_key=settings.openai_api_key,
        model=settings.openai_tool_worker_model or settings.openai_chat_model,
        repo_root=isolated_user_config.parent,
        project_root=isolated_user_config.parent,
    )

    payload = worker.init_payload_dict()
    assert payload["api_key"] == "file-key"
    assert payload["base_url"] == "https://saved.example/v1"
    assert payload["model"] == "saved-worker-model"


def test_llm_model_alias_applies_when_saved_chat_model_is_missing(isolated_user_config: Path) -> None:
    user_config.save_effective_user_config({"LLM_MODEL": "alias-model"}, merge=False)

    settings = Settings()

    assert settings.openai_chat_model == "alias-model"


def test_masked_summary_hides_secrets(isolated_user_config: Path) -> None:
    user_config.save_effective_user_config(
        {
            "OPENAI_API_KEY": "sk-test-secret",
            "MANA_WEB_SEARCH_API_KEY": "search-secret",
            "MANA_GITHUB_TOKEN": "github-secret",
        },
        merge=False,
    )

    summary = user_config.masked_config_summary()

    assert summary["OPENAI_API_KEY"] != "sk-test-secret"
    assert summary["MANA_WEB_SEARCH_API_KEY"] != "search-secret"
    assert summary["MANA_GITHUB_TOKEN"] != "github-secret"


def test_parse_model_ids_sorts_and_filters() -> None:
    assert parse_model_ids({"data": [{"id": "z-model"}, {"id": ""}, {"id": "a-model"}]}) == [
        "a-model",
        "z-model",
    ]


def test_model_fetch_failure_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_urlopen(*args, **kwargs):  # noqa: ANN001
        raise OSError("network unavailable")

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)

    with pytest.raises(ModelFetchError, match="Model fetch failed"):
        fetch_openai_compatible_models(base_url="https://api.example.test/v1", api_key="key", timeout_seconds=1)


def test_model_cache_round_trip(isolated_user_config: Path) -> None:
    user_config.save_model_cache("openai", "https://api.example.test/v1", ["b", "a"])

    cached = user_config.load_model_cache("openai", "https://api.example.test/v1")

    assert cached is not None
    assert cached.models == ["a", "b"]
    assert json.loads((isolated_user_config / "model_cache.json").read_text(encoding="utf-8"))


def test_choose_models_persists_model_level_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mana_agent.tui.model_picker.load_or_fetch_models", lambda **_kwargs: ["embed", "main", "planner", "tool"])
    selections = iter(["main", "tool", "planner", "embed"])
    monkeypatch.setattr("mana_agent.tui.model_picker.select_option", lambda **_kwargs: next(selections))

    values = choose_models(
        provider="openai",
        base_url="https://api.example.test/v1",
        api_key="key",
        timeout_seconds=1,
        current={},
    )

    assert values["OPENAI_CHAT_MODEL"] == "main"
    assert values["OPENAI_TOOL_WORKER_MODEL"] == "tool"
    assert values["OPENAI_CODING_PLANNER_MODEL"] == "planner"
    assert values["MODEL_LEVEL_3_HIGH_REASONING"] == "main"
    assert values["MODEL_LEVEL_2_CODING"] == "planner"
    assert values["MODEL_LEVEL_1_FAST_TOOL"] == "tool"


def test_user_config_writes_grouped_model_order(isolated_user_config: Path) -> None:
    user_config.save_effective_user_config(
        {
            "MANA_SEARCH_ENABLE_WEB": True,
            "OPENAI_CHAT_MODEL": "main",
            "MODEL_LEVEL_3_HIGH_REASONING": "main",
            "OPENAI_BASE_URL": "https://api.example.test/v1",
        },
        merge=False,
    )

    lines = (isolated_user_config / "config.toml").read_text(encoding="utf-8").splitlines()

    assert lines[:3] == [
        'OPENAI_BASE_URL = "https://api.example.test/v1"',
        'OPENAI_CHAT_MODEL = "main"',
        'MODEL_LEVEL_3_HIGH_REASONING = "main"',
    ]


def test_model_level_targets_resolve_from_user_config(
    isolated_user_config: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_config.save_effective_user_config(
        {
            "OPENAI_CHAT_MODEL": "main",
            "MODEL_LEVEL_3_HIGH_REASONING": "high",
            "MODEL_LEVEL_2_CODING": "coding",
            "MODEL_LEVEL_1_FAST_TOOL": "fast",
            "MANA_MODEL_MAIN": "MODEL_LEVEL_3_HIGH_REASONING",
            "MANA_MODEL_CODING": "MODEL_LEVEL_2_CODING",
            "MANA_MODEL_TOOL": "MODEL_LEVEL_1_FAST_TOOL",
        },
        merge=False,
    )
    monkeypatch.setenv("MODEL_LEVEL_3_HIGH_REASONING", "environment-high")

    assert resolve_model_for_role(AgentRole.MAIN, global_model="fallback").resolved_model == "high"
    assert resolve_model_for_role(AgentRole.CODING, global_model="fallback").resolved_model == "coding"
    assert resolve_model_for_role(AgentRole.TOOL, global_model="fallback").resolved_model == "fast"


def test_search_provider_config_serialization(isolated_user_config: Path) -> None:
    user_config.save_effective_user_config(
        {
            "OPENAI_API_KEY": "key",
            "MANA_SEARCH_ENABLE_WEB": True,
            "MANA_WEB_SEARCH_PROVIDER": "google_cse",
            "MANA_WEB_SEARCH_API_KEY": "search-key",
            "MANA_WEB_SEARCH_ENGINE_ID": "cx-id",
            "MANA_WEB_SEARCH_MAX_RESULTS": 5,
        },
        merge=False,
    )

    config = SearchConfig.from_env()

    assert config.enable_web is True
    assert config.web_provider == "google_cse"
    assert config.web_api_key == "search-key"
    assert config.web_engine_id == "cx-id"
    assert config.web_max_results == 5


def test_non_interactive_setup_does_not_prompt(isolated_user_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)

    with pytest.raises(NonInteractivePromptError):
        ensure_setup(no_interactive=True, command_needs_llm=True)


def test_main_menu_helpers_importable() -> None:
    from mana_agent.tui.menu import MenuOption, select_option

    selected = select_option(
        title="test",
        text="test",
        options=[MenuOption("chat", "Chat")],
        input_func=lambda _: "1",
    )

    assert selected == "chat"
