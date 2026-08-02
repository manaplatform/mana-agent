from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from mana_agent.multi_agent.core.types import QueueJob, QueueJobType
from mana_agent.multi_agent.tools import git_tools
from mana_agent.multi_agent.tools.tool_manager import ToolsManager
from mana_agent.transactional_actions.runtime import approve_action


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    assert _git(path, "init").returncode == 0
    assert _git(path, "config", "user.name", "Mana Agent Test").returncode == 0
    assert _git(path, "config", "user.email", "test@example.com").returncode == 0
    (path / "README.md").write_text("hello\n", encoding="utf-8")
    assert _git(path, "add", "README.md").returncode == 0
    assert _git(path, "commit", "-m", "test: initial commit").returncode == 0
    return path


def _approve_and_retry(repo: Path, call: Any) -> dict[str, Any]:
    pending = call()
    assert pending["permission_required"] is True
    approval_id = approve_action(
        repo,
        pending["action_id"],
        approved_by="pytest-local-user",
    )
    assert approval_id
    return call()


def test_dynamic_git_command_discovery_uses_local_help(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")

    commands = git_tools.discover_git_commands(repo, refresh=True)

    assert "status" in commands
    assert "commit" in commands
    assert commands == sorted(commands)


def test_generic_git_command_execution_resolves_repo_root_and_returns_structured_result(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    nested = repo / "src" / "pkg"
    nested.mkdir(parents=True)

    result = git_tools.generic(args=["status", "--short"], repo_path=nested)

    assert result["ok"] is True
    assert result["command"] == ["git", "status", "--short"]
    assert result["repo_root"] == str(repo.resolve())
    assert result["risk_level"] == "READ_ONLY"
    assert result["returncode"] == 0
    assert isinstance(result["duration_ms"], float)


def test_status_and_diff_wrappers(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "README.md").write_text("hello\nchanged\n", encoding="utf-8")

    status = git_tools.status(repo_path=repo)
    diff = git_tools.diff(repo_path=repo)

    assert status["ok"] is True
    assert "README.md" in status["stdout"]
    assert diff["ok"] is True
    assert "+changed" in diff["stdout"]


def test_branch_creation_wrapper_switches_to_new_branch(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")

    result = _approve_and_retry(
        repo,
        lambda: git_tools.create_branch(repo_path=repo, branch_name="feature/git-tools"),
    )

    assert result["ok"] is True
    assert result["preflight"]["status"]["command"] == ["git", "status", "--short"]
    assert result["preflight"]["branches"]["command"] == ["git", "branch", "--list"]
    assert _git(repo, "branch", "--show-current").stdout.strip() == "feature/git-tools"


def test_commit_wrapper_commits_staged_files_with_generated_message(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "README.md").write_text("hello\ncommit me\n", encoding="utf-8")
    assert _approve_and_retry(
        repo,
        lambda: git_tools.add(repo_path=repo, paths=["README.md"]),
    )["ok"] is True

    result = _approve_and_retry(
        repo,
        lambda: git_tools.commit(repo_path=repo, message="feat: add git tools test fixture"),
    )

    assert result["ok"] is True
    assert result["preflight"]["status_short"]["command"] == ["git", "status", "--short"]
    assert result["preflight"]["diff"]["command"] == ["git", "diff"]
    assert result["preflight"]["diff_staged"]["command"] == ["git", "diff", "--staged"]
    assert result["preflight"]["diff_staged_stat"]["command"] == ["git", "diff", "--staged", "--stat"]
    assert "feat: add git tools test fixture" in _git(repo, "log", "-1", "--pretty=%s").stdout


def test_push_wrapper_sets_upstream_when_missing(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    assert subprocess.run(["git", "init", "--bare", str(remote)], text=True, capture_output=True, check=False).returncode == 0
    repo = _init_repo(tmp_path / "repo")
    assert _git(repo, "remote", "add", "origin", str(remote)).returncode == 0

    result = _approve_and_retry(repo, lambda: git_tools.push(repo_path=repo))

    assert result["ok"] is True
    assert result["preflight"]["status_short"]["command"] == ["git", "status", "--short"]
    assert result["preflight"]["current_branch"]["command"] == ["git", "branch", "--show-current"]
    assert result["preflight"]["remotes"]["command"] == ["git", "remote", "-v"]
    assert result["preflight"]["upstream"]["command"] == ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]
    assert result["command"][-3:] == ["-u", "origin", _git(repo, "branch", "--show-current").stdout.strip()]


def test_secret_redaction_from_git_output(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    assert _approve_and_retry(
        repo,
        lambda: git_tools.generic(
            args=["config", "user.name", "sk-testsecret123"], repo_path=repo
        ),
    )["ok"] is True

    result = git_tools.generic(args=["config", "user.name"], repo_path=repo)

    assert result["ok"] is True
    assert "sk-testsecret123" not in result["stdout"]
    assert "sk-testsecret123" not in result["stderr"]


def test_destructive_and_force_push_commands_are_blocked(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")

    reset = git_tools.generic(args=["reset", "--hard", "HEAD"], repo_path=repo)
    push = git_tools.push(repo_path=repo, force=True)

    assert reset["ok"] is False
    assert reset["blocked"] is True
    assert reset["risk_level"] == "DESTRUCTIVE"
    assert push["ok"] is False
    assert push["blocked"] is True
    assert push["risk_level"] == "HISTORY_REWRITE"


def test_mutating_branch_and_remote_subcommands_are_not_classified_read_only() -> None:
    assert git_tools.classify_git_risk(["branch", "-d", "topic"]) is git_tools.GitRiskLevel.LOCAL_SAFE_WRITE
    assert git_tools.classify_git_risk(["branch", "-D", "topic"]) is git_tools.GitRiskLevel.DESTRUCTIVE
    assert git_tools.classify_git_risk(["remote", "add", "origin", "example"]) is git_tools.GitRiskLevel.LOCAL_SAFE_WRITE


def test_generic_git_executor_never_uses_shell_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_repo(tmp_path / "repo")
    calls: list[dict[str, Any]] = []
    real_run = subprocess.run

    def spy_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"args": args, "kwargs": kwargs})
        return real_run(*args, **kwargs)

    monkeypatch.setattr(git_tools.subprocess, "run", spy_run)

    result = git_tools.generic(args=["status", "--short"], repo_path=repo)

    assert result["ok"] is True
    assert calls
    assert all(call["kwargs"].get("shell") is not True for call in calls)
    assert all(isinstance(call["args"][0], list) for call in calls)


def test_timeout_returns_structured_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_repo(tmp_path / "repo")

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if args and args[0][:2] == ["git", "status"]:
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=1, output="partial")
        return subprocess.CompletedProcess(args[0], 0, stdout=str(repo), stderr="")

    monkeypatch.setattr(git_tools.subprocess, "run", fake_run)

    result = git_tools.generic(args=["status"], repo_path=repo, timeout=1)

    assert result["ok"] is False
    assert result["error"] == "timeout"
    assert "timed out" in result["stderr"]


def test_git_state_memory_invalidates_on_status_change(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    memory = git_tools.GitStateMemory()

    first = git_tools.generic(args=["status", "--porcelain=v1"], repo_path=repo, memory=memory)
    second = git_tools.generic(args=["status", "--porcelain=v1"], repo_path=repo, memory=memory)
    (repo / "README.md").write_text("hello\nmemory change\n", encoding="utf-8")
    third = git_tools.generic(args=["status", "--porcelain=v1"], repo_path=repo, memory=memory)

    assert first["state"]["memory"]["cache_hit"] is False
    assert second["state"]["memory"]["cache_hit"] is True
    assert third["state"]["memory"]["invalidated"] is True


def test_queue_tools_manager_executes_git_namespace(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    manager = ToolsManager(repo)
    job = QueueJob(
        job_id="job-1",
        task_id="task-1",
        requested_by_agent_id="agent-1",
        job_type=QueueJobType.GIT,
        payload={"tool": "git.status", "args": {}},
    )

    result = manager.execute_job(job)

    assert result.ok is True
    assert result.result["command"] == ["git", "status", "--short"]
