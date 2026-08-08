"""SWE-bench runner resolves provider/model from ~/.mana/config.toml."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

from scripts.swe_bench.runner import (
    RunnerConfig,
    WorktreeChangeSummary,
    _benchmark_config_overrides,
    _disable_non_coding_integrations,
    _set_toml_table_key,
    build_prompt,
    capture_model_patch,
    format_timeout_label,
    load_operator_inference_defaults,
    prepare_agent_python_path,
    resolve_agent_inference,
    resolve_model_name_or_path,
    resolve_runner_timeout_seconds,
    summarize_worktree_changes,
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
        timeout_seconds=3600,
    )
    assert overrides["MANA_AI_PROVIDER"] == "nvidia"
    assert overrides["OPENAI_CHAT_MODEL"] == "deepseek-ai/deepseek-v4-flash-0731"
    assert overrides["MANA_PRIMARY_MODEL"] == "nvidia/deepseek-ai/deepseek-v4-flash-0731"
    assert overrides["MANA_MEMORY_MODE"] == "internal"
    # Coding-only isolation for SWE-bench.
    assert overrides["MANA_BROWSER_ENABLED"] is False
    assert overrides["MANA_COMPUTER_CONTROL_ENABLED"] is False
    assert overrides["MANA_CANVAS_ENABLED"] is False
    assert overrides["MANA_SEARCH_ENABLE_WEB"] is False
    assert overrides["MANA_CODEX_TASK_TIMEOUT_SECONDS"] == 3600


def test_resolve_runner_timeout_cli_env_and_unlimited() -> None:
    seconds, source = resolve_runner_timeout_seconds(999999999)
    assert seconds == 999999999
    assert source == "cli --timeout"
    assert format_timeout_label(seconds) == "999999999s"

    seconds, source = resolve_runner_timeout_seconds(0)
    assert seconds == 0
    assert format_timeout_label(seconds) == "unlimited"

    seconds, source = resolve_runner_timeout_seconds(
        None, env={"MANA_SWE_BENCH_TIMEOUT": "7200"}
    )
    assert seconds == 7200
    assert "MANA_SWE_BENCH_TIMEOUT" in source

    seconds, source = resolve_runner_timeout_seconds(None, env={})
    assert seconds == 600
    assert "built-in" in source


def test_disable_nested_computer_control_for_isolation(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "MANA_AI_PROVIDER = \"nvidia\"\n\n"
        "[computer_control]\n"
        "enabled = true\n"
        "timeout_seconds = \"30.0\"\n",
        encoding="utf-8",
    )
    _disable_non_coding_integrations(config)
    text = config.read_text(encoding="utf-8")
    assert "enabled = false" in text
    # helper also works for creating missing tables
    _set_toml_table_key(config, "telegram", "enabled", False)
    assert "[telegram]" in config.read_text(encoding="utf-8")


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


def test_build_prompt_prefers_source_edits_and_python3() -> None:
    prompt = build_prompt(
        {
            "instance_id": "astropy__astropy-12907",
            "repo": "astropy/astropy",
            "problem_statement": "separability_matrix nested compound bug",
        },
        worktree=Path("/tmp/worktree"),
        max_chars=48_000,
    )
    assert "python3" in prompt
    assert "never bare `python`" in prompt
    assert "may not be installed or importable" in prompt
    assert "production-source edits" in prompt
    assert "separability_matrix nested compound bug" in prompt


def test_prepare_agent_python_path_shims_python_to_python3(tmp_path: Path) -> None:
    env: dict[str, str] = {"PATH": "/usr/bin"}
    bin_dir = prepare_agent_python_path(run_dir=tmp_path, env=env)
    python_shim = bin_dir / "python"
    assert python_shim.is_file()
    assert python_shim.stat().st_mode & stat.S_IXUSR
    assert str(bin_dir) in env["PATH"].split(os.pathsep)
    assert env["PATH"].split(os.pathsep)[0] == str(bin_dir)
    # Shim must invoke Python 3, not host Python 2.7.
    probe = subprocess.run(
        [str(python_shim), "-c", "import sys; print(sys.version_info[0])"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0
    assert probe.stdout.strip() == "3"


def test_mass_delete_only_summary_and_capture_rejection(tmp_path: Path) -> None:
    summary = WorktreeChangeSummary(modified=0, added=0, deleted=42)
    assert summary.is_mass_delete_only is True
    assert WorktreeChangeSummary(modified=1, added=0, deleted=42).is_mass_delete_only is False

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    for i in range(25):
        path = repo / f"file_{i}.txt"
        path.write_text(f"content {i}\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    for i in range(25):
        (repo / f"file_{i}.txt").unlink()

    counted = summarize_worktree_changes(repo)
    assert counted.deleted >= 20
    assert counted.modified == 0
    assert counted.is_mass_delete_only is True

    patch, reason = capture_model_patch(repo, exclude_test_files=False)
    assert patch == ""
    assert "mass-delete" in reason
