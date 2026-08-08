"""SWE-bench runner resolves provider/model from ~/.mana/config.toml."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.swe_bench.runner import (
    RunnerConfig,
    SweBenchRunnerError,
    WorktreeChangeSummary,
    _benchmark_config_overrides,
    _disable_non_coding_integrations,
    _set_toml_table_key,
    _upsert_toml_keys,
    acquire_worktree_lock,
    build_prompt,
    capture_model_patch,
    format_timeout_label,
    load_operator_inference_defaults,
    prepare_agent_python_path,
    release_worktree_lock,
    resolve_agent_inference,
    resolve_model_name_or_path,
    resolve_runner_timeout_seconds,
    summarize_worktree_changes,
    worktree_lock_holder,
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
    assert overrides["MANA_MANAGED_WORKTREES_ENABLED"] is False
    assert overrides["MANA_CODEX_WORKTREE_ISOLATION"] is False
    assert overrides["MANA_TRANSACTIONAL_ALWAYS_APPROVE"] is True
    assert overrides["MANA_CODEX_TASK_TIMEOUT_SECONDS"] == 3600
    # Stale operator MANA_CODEX_MODEL=gpt-5.6-luna must not survive isolation.
    assert overrides["MANA_CODEX_MODEL"] == "deepseek-ai/deepseek-v4-flash-0731"
    assert overrides["MANA_CODEX_ENABLED"] is True
    assert overrides["MANA_CODING_BACKEND"] == "codex"
    assert overrides["MANA_AUTO_CHAT_TOOL_SURFACE"] == "coding"
    assert overrides["MANA_CONTEXT_UNKNOWN_MODEL_CONTEXT_WINDOW"] == 1_000_000
    assert overrides["MANA_CONTEXT_UNKNOWN_MODEL_MAX_OUTPUT_TOKENS"] == 65_536


def test_resolve_runner_timeout_cli_env_and_unlimited(tmp_path: Path) -> None:
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

    seconds, source = resolve_runner_timeout_seconds(None, env={}, work_dir=tmp_path)
    assert seconds == 600
    assert "built-in" in source

    # File config under work-dir (shell-safe alternative to multi-line --timeout).
    (tmp_path / "runner.toml").write_text("timeout = 0\n", encoding="utf-8")
    seconds, source = resolve_runner_timeout_seconds(None, env={}, work_dir=tmp_path)
    assert seconds == 0
    assert "runner.toml" in source


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


def test_upsert_toml_keys_inserts_before_nested_tables(tmp_path: Path) -> None:
    """Isolation overrides must stay top-level; EOF append nests under last table."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        import tomli as tomllib  # type: ignore[no-redef]

    config = tmp_path / "config.toml"
    config.write_text(
        'MANA_AI_PROVIDER = "nvidia"\n'
        "MANA_BROWSER_ENABLED = true\n\n"
        "[telegram.attachments]\n"
        "enabled = true\n"
        "max_bytes = 10\n",
        encoding="utf-8",
    )
    _upsert_toml_keys(
        config,
        {
            "MANA_BROWSER_ENABLED": False,
            "MANA_MANAGED_WORKTREES_ENABLED": False,
            "MANA_TRANSACTIONAL_ALWAYS_APPROVE": True,
            "MANA_CODEX_WORKTREE_ISOLATION": False,
        },
    )
    text = config.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    # Top-level (not nested under telegram.attachments).
    assert data["MANA_BROWSER_ENABLED"] is False
    assert data["MANA_MANAGED_WORKTREES_ENABLED"] is False
    assert data["MANA_TRANSACTIONAL_ALWAYS_APPROVE"] is True
    assert data["MANA_CODEX_WORKTREE_ISOLATION"] is False
    nested = data["telegram"]["attachments"]
    assert "MANA_TRANSACTIONAL_ALWAYS_APPROVE" not in nested
    assert "MANA_MANAGED_WORKTREES_ENABLED" not in nested
    # Override block appears before the first table header.
    assert text.index("MANA_TRANSACTIONAL_ALWAYS_APPROVE") < text.index("[telegram.attachments]")


def test_benchmark_isolation_settings_reach_settings_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SWE-bench isolation is useless if Settings ignores the rewritten keys."""
    from mana_agent.config.settings import Settings

    home = tmp_path / "mana_home"
    home.mkdir()
    config = home / "config.toml"
    config.write_text(
        'MANA_AI_PROVIDER = "nvidia"\n\n'
        "[telegram.attachments]\n"
        "enabled = false\n",
        encoding="utf-8",
    )
    _upsert_toml_keys(config, _benchmark_config_overrides("m", agent_provider="nvidia"))

    monkeypatch.setenv("MANA_HOME", str(home))
    for key in (
        "MANA_MANAGED_WORKTREES_ENABLED",
        "MANA_TRANSACTIONAL_ALWAYS_APPROVE",
        "MANA_CODEX_WORKTREE_ISOLATION",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = Settings()
    assert settings.mana_managed_worktrees_enabled is False
    assert settings.mana_transactional_always_approve is True
    assert settings.mana_codex_worktree_isolation is False


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
    # POSIX: executable shell script named ``python``.
    # Windows: PATHEXT resolves ``python.cmd`` (chmod/S_IXUSR is not reliable).
    if os.name == "nt":
        python_shim = bin_dir / "python.cmd"
        python3_shim = bin_dir / "python3.cmd"
    else:
        python_shim = bin_dir / "python"
        python3_shim = bin_dir / "python3"
    assert python_shim.is_file()
    assert python3_shim.is_file()
    if os.name != "nt":
        assert python_shim.stat().st_mode & stat.S_IXUSR
        assert python3_shim.stat().st_mode & stat.S_IXUSR
    assert str(bin_dir) in env["PATH"].split(os.pathsep)
    assert env["PATH"].split(os.pathsep)[0] == str(bin_dir)
    assert env.get("PYTHON")
    # Shim must invoke Python 3, not host Python 2.7.
    # Windows cannot CreateProcess a .cmd file without going through cmd.exe.
    argv = [str(python_shim), "-c", "import sys; print(sys.version_info[0])"]
    if os.name == "nt":
        argv = ["cmd", "/c", *argv]
    probe = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, (probe.stdout, probe.stderr)
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


def test_worktree_lock_blocks_second_holder(tmp_path: Path, monkeypatch) -> None:
    worktrees = tmp_path / "worktrees"
    lock = acquire_worktree_lock(worktrees, "astropy__astropy-13033")
    assert lock.is_file()
    assert worktree_lock_holder(worktrees, "astropy__astropy-13033") == os.getpid()

    # Simulate another process by claiming a foreign live pid.
    lock.write_text("1\n", encoding="utf-8")
    monkeypatch.setattr(
        "scripts.swe_bench.runner._pid_is_alive",
        lambda pid: pid == 1,
    )
    with pytest.raises(SweBenchRunnerError, match="locked by live process"):
        acquire_worktree_lock(worktrees, "astropy__astropy-13033")

    # Stale lock from a dead pid is reclaimable.
    monkeypatch.setattr(
        "scripts.swe_bench.runner._pid_is_alive",
        lambda pid: False,
    )
    reclaimed = acquire_worktree_lock(worktrees, "astropy__astropy-13033")
    assert reclaimed.is_file()
    release_worktree_lock(reclaimed)
    assert worktree_lock_holder(worktrees, "astropy__astropy-13033") is None
