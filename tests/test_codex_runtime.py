from __future__ import annotations

import asyncio
import os
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
    import tomli as tomllib
from pathlib import Path

import pytest

from mana_agent.commands.codex_cli import codex_app
from mana_agent.integrations.codex.backend import CodexCodingBackend
from mana_agent.integrations.codex.client import AsyncCodexAppServer
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.integrations.codex.exceptions import CodexConfigurationError
from mana_agent.integrations.codex.runtime_config import (
    CodexRuntimeConfigBuilder,
    resolve_codex_base_url,
)
from mana_agent.integrations.codex.runtime_environment import CodexRuntimeEnvironment


def _settings(**updates: object) -> CodexSettings:
    values = {
        "enabled": True,
        "provider": "openai",
        "provider_display_name": "OpenAI",
        "api_key": "mana-secret-key",
        "base_url": "https://api.example.test/v1/responses/",
        "model": 'model-"quoted"',
        "supports_responses_api": True,
        "http_headers": {"X-Client": 'Mana "Agent"'},
        "query_params": {"region": "eu west"},
    }
    values.update(updates)
    return CodexSettings(**values)


def test_codex_cli_exposes_only_read_only_operational_commands() -> None:
    assert {command.name for command in codex_app.registered_commands} == {"status", "doctor"}


def test_runtime_config_maps_provider_without_persisting_secret() -> None:
    runtime = CodexRuntimeConfigBuilder.build(_settings(), sandbox_mode="workspace-write")
    rendered = runtime.to_toml()
    parsed = tomllib.loads(rendered)

    assert parsed["model"] == 'model-"quoted"'
    assert parsed["model_provider"] == "mana_runtime"
    assert parsed["forced_login_method"] == "api"
    assert parsed["model_providers"]["mana_runtime"] == {
        "name": "Mana-Agent runtime provider",
        "base_url": "https://api.example.test/v1",
        "env_key": "MANA_CODEX_API_KEY",
        "wire_api": "responses",
        "request_max_retries": 4,
        "stream_max_retries": 5,
        "stream_idle_timeout_ms": 300000,
        "http_headers": {"X-Client": 'Mana "Agent"'},
        "query_params": {"region": "eu west"},
    }
    assert "mana-secret-key" not in rendered
    assert runtime.credential_fingerprint.startswith("sha256:")
    assert "mana-secret-key" not in repr(runtime)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://api.openai.com/v1", "https://api.openai.com/v1"),
        ("https://provider.example/v1/", "https://provider.example/v1"),
        ("https://provider.example/v1/responses", "https://provider.example/v1"),
    ],
)
def test_resolve_codex_base_url(value: str, expected: str) -> None:
    assert resolve_codex_base_url(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not-a-url",
        "ftp://provider.example/v1",
        "https://provider.example/v1?api-version=1",
        "https://provider.example/v1/chat/completions",
    ],
)
def test_resolve_codex_base_url_rejects_invalid_or_chat_only_endpoint(value: str) -> None:
    with pytest.raises(CodexConfigurationError):
        resolve_codex_base_url(value)


def test_unsupported_provider_and_missing_values_stop_before_runtime_creation() -> None:
    with pytest.raises(CodexConfigurationError, match="Responses-compatible"):
        CodexRuntimeConfigBuilder.build(
            _settings(supports_responses_api=False), sandbox_mode="read-only"
        )
    for field in ("provider", "api_key", "model", "base_url"):
        with pytest.raises(CodexConfigurationError):
            CodexRuntimeConfigBuilder.build(
                _settings(**{field: ""}), sandbox_mode="read-only"
            )
    with pytest.raises(CodexConfigurationError, match="Unsafe"):
        CodexRuntimeConfigBuilder.build(
            _settings(http_headers={"Authorization": "Bearer secret"}),
            sandbox_mode="read-only",
        )


def test_runtime_environment_is_unique_isolated_and_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mana_root = tmp_path / "mana"
    fake_home = tmp_path / "home"
    global_codex = fake_home / ".codex"
    global_codex.mkdir(parents=True)
    global_config = global_codex / "config.toml"
    global_auth = global_codex / "auth.json"
    global_config.write_text('model_provider = "global"\n', encoding="utf-8")
    global_auth.write_text('{"auth":"chatgpt"}', encoding="utf-8")
    before_config = global_config.read_bytes()
    before_auth = global_auth.read_bytes()
    monkeypatch.setenv("MANA_HOME", str(mana_root))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("CODEX_HOME", str(global_codex))
    monkeypatch.setenv("OPENAI_API_KEY", "global-key")
    monkeypatch.setenv("CODEX_API_KEY", "global-codex-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://global.invalid/v1")
    parent_before = os.environ.copy()
    runtime = CodexRuntimeConfigBuilder.build(_settings(), sandbox_mode="read-only")

    first = CodexRuntimeEnvironment.create(runtime)
    second = CodexRuntimeEnvironment.create(runtime)
    try:
        assert first.home != second.home
        assert first.home.parent == mana_root / "runtime" / "codex"
        assert first.environment["CODEX_HOME"] == str(first.home)
        assert first.environment["MANA_CODEX_API_KEY"] == "mana-secret-key"
        assert "OPENAI_API_KEY" not in first.environment
        assert "CODEX_API_KEY" not in first.environment
        assert "OPENAI_BASE_URL" not in first.environment
        assert "mana-secret-key" not in (first.home / "config.toml").read_text(encoding="utf-8")
        assert not (first.home / "auth.json").exists()
        assert os.environ == parent_before
    finally:
        first_home = first.home
        second_home = second.home
        first.close()
        second.close()

    assert not first_home.exists()
    assert not second_home.exists()
    assert global_config.read_bytes() == before_config
    assert global_auth.read_bytes() == before_auth


def test_parallel_runtime_contexts_do_not_cross_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    one = CodexRuntimeEnvironment.create(
        CodexRuntimeConfigBuilder.build(
            _settings(api_key="key-one", model="model-one", base_url="https://one.test/v1"),
            sandbox_mode="read-only",
        )
    )
    two = CodexRuntimeEnvironment.create(
        CodexRuntimeConfigBuilder.build(
            _settings(api_key="key-two", model="model-two", base_url="https://two.test/v1"),
            sandbox_mode="workspace-write",
        )
    )
    try:
        assert one.environment["MANA_CODEX_API_KEY"] == "key-one"
        assert two.environment["MANA_CODEX_API_KEY"] == "key-two"
        assert "key-two" not in str(one.environment)
        assert "key-one" not in str(two.environment)
        assert 'model = "model-one"' in (one.home / "config.toml").read_text(encoding="utf-8")
        assert 'model = "model-two"' in (two.home / "config.toml").read_text(encoding="utf-8")
    finally:
        one.close()
        two.close()


def test_backend_cleans_runtime_after_app_server_startup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_home: Path | None = None

    class FailingClient:
        running = False

        def __init__(self, _command: object, *, environment: dict[str, str], **_kwargs: object) -> None:
            nonlocal captured_home
            captured_home = Path(environment["CODEX_HOME"])

        async def start(self) -> None:
            raise OSError("startup failed")

        async def close(self) -> None:
            return None

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    monkeypatch.setattr(
        "mana_agent.integrations.codex.backend.check_codex_health",
        lambda _settings, _repository: type(
            "Report", (), {"healthy": True, "executable": "/fake/codex", "errors": []}
        )(),
    )
    monkeypatch.setattr("mana_agent.integrations.codex.backend.AsyncCodexAppServer", FailingClient)
    backend = CodexCodingBackend(_settings())

    with pytest.raises(OSError, match="startup failed"):
        asyncio.run(backend.start(tmp_path))

    assert captured_home is not None
    assert not captured_home.exists()


def test_provider_errors_are_actionable_and_credentials_are_redacted() -> None:
    client = AsyncCodexAppServer(
        ("codex", "app-server"),
        environment={"MANA_CODEX_API_KEY": "unusually-shaped-secret"},
        provider_name="Example Provider",
        model="example-model",
    )

    authentication = client._format_provider_error(
        "HTTP 401 Authorization: Bearer unusually-shaped-secret"
    )
    missing_model = client._format_provider_error("model_not_found: unusually-shaped-secret")

    assert "Provider authentication failed for Example Provider" in authentication
    assert "could not find configured model example-model" in missing_model
    assert "unusually-shaped-secret" not in authentication + missing_model
