"""SWE-bench runner resolves provider/model from ~/.mana/config.toml."""

from __future__ import annotations

from pathlib import Path

from scripts.swe_bench.runner import (
    RunnerConfig,
    _benchmark_config_overrides,
    load_operator_inference_defaults,
    resolve_agent_inference,
    resolve_model_name_or_path,
)


def _write_config(path: Path, body: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config = path / "config.toml"
    config.write_text(body, encoding="utf-8")
    return path


def test_load_operator_defaults_from_mana_config(tmp_path: Path) -> None:
    home = _write_config(
        tmp_path / "mana",
        """
MANA_AI_PROVIDER = "nvidia"
MANA_PRIMARY_MODEL = "nvidia/deepseek-ai/deepseek-v4-flash-0731"
OPENAI_CHAT_MODEL = "deepseek-ai/deepseek-v4-flash-0731"
""".strip()
        + "\n",
    )
    defaults = load_operator_inference_defaults(home)
    assert defaults.provider == "nvidia"
    assert defaults.model == "deepseek-ai/deepseek-v4-flash-0731"
    assert "MANA_PRIMARY_MODEL" in defaults.source


def test_no_cli_model_uses_config(tmp_path: Path) -> None:
    home = _write_config(
        tmp_path / "mana",
        'MANA_AI_PROVIDER = "nvidia"\n'
        'MANA_PRIMARY_MODEL = "nvidia/deepseek-ai/deepseek-v4-flash-0731"\n',
    )
    provider, model, provider_source, model_source = resolve_agent_inference(
        cli_model=None,
        cli_provider=None,
        mana_home=home,
    )
    assert provider == "nvidia"
    assert model == "deepseek-ai/deepseek-v4-flash-0731"
    assert "MANA_AI_PROVIDER" in provider_source
    assert "MANA_PRIMARY_MODEL" in model_source


def test_cli_model_keeps_configured_provider(tmp_path: Path) -> None:
    home = _write_config(
        tmp_path / "mana",
        'MANA_AI_PROVIDER = "nvidia"\n'
        'MANA_PRIMARY_MODEL = "nvidia/deepseek-ai/deepseek-v4-flash-0731"\n',
    )
    provider, model, _, model_source = resolve_agent_inference(
        cli_model="deepseek-ai/deepseek-v4-pro",
        cli_provider=None,
        mana_home=home,
    )
    assert provider == "nvidia"
    assert model == "deepseek-ai/deepseek-v4-pro"
    assert model_source == "cli --model"


def test_cli_provider_overrides_config(tmp_path: Path) -> None:
    home = _write_config(
        tmp_path / "mana",
        'MANA_AI_PROVIDER = "nvidia"\nMANA_PRIMARY_MODEL = "nvidia/deepseek-ai/x"\n',
    )
    provider, model, provider_source, _ = resolve_agent_inference(
        cli_model="gpt-4o-mini",
        cli_provider="openai",
        mana_home=home,
    )
    assert provider == "openai"
    assert model == "gpt-4o-mini"
    assert provider_source == "cli --provider"


def test_benchmark_overrides_pin_provider_and_model() -> None:
    overrides = _benchmark_config_overrides(
        "deepseek-ai/deepseek-v4-flash-0731",
        agent_provider="nvidia",
    )
    assert overrides["MANA_AI_PROVIDER"] == "nvidia"
    assert overrides["OPENAI_CHAT_MODEL"] == "deepseek-ai/deepseek-v4-flash-0731"
    assert overrides["MANA_PRIMARY_MODEL"] == "nvidia/deepseek-ai/deepseek-v4-flash-0731"
    assert overrides["MANA_MEMORY_MODE"] == "internal"


def test_model_name_or_path_includes_non_openai_provider() -> None:
    cfg = RunnerConfig(
        agent_name="mana-agent",
        agent_provider="nvidia",
        agent_model="deepseek-ai/deepseek-v4-flash-0731",
    )
    assert (
        resolve_model_name_or_path(cfg)
        == "mana-agent__nvidia__deepseek-ai__deepseek-v4-flash-0731"
    )
