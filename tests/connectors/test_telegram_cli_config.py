from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
    import tomli as tomllib
from pathlib import Path

from typer.testing import CliRunner

from mana_agent.commands.cli import app
from mana_agent.config import user_config
from mana_agent.connectors.telegram.config import TelegramConfig, load_telegram_config, save_telegram_config


def test_nested_telegram_config_round_trip_without_secret(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    repository = tmp_path / "repo"
    repository.mkdir()
    config = TelegramConfig(
        enabled=True, transport="auto", bot_token_env="BOT_ENV",
        allowed_users=[123], default_repository=str(repository),
        allowed_repository_roots=[str(tmp_path)], webhook={"public_url": "https://example.test"},
    )
    save_telegram_config(config)
    text = user_config.config_file().read_text(encoding="utf-8")
    assert "[telegram]" in text and "[telegram.webhook]" in text
    assert "BOT_ENV" in text and "bot-token-value" not in text
    assert tomllib.loads(text)["telegram"]["allowed_users"] == [123]
    loaded = load_telegram_config()
    assert loaded.allowed_users == [123]
    assert loaded.webhook.public_url == "https://example.test"


def test_cli_discovers_telegram_commands(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    result = CliRunner().invoke(app, ["connector", "telegram", "--help"])
    assert result.exit_code == 0
    for command in ("setup", "start", "status", "stop", "test", "webhook", "info"):
        assert command in result.output
