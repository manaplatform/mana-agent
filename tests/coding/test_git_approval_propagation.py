from __future__ import annotations

import getpass
import subprocess

import pytest

from mana_agent.coding.models import WorkspaceContext
from mana_agent.coding.programmatic import ToolExecutionDispatcher
from mana_agent.human_inbox.approval_tools import approval_request
from mana_agent.multi_agent.tools import git_tools
from mana_agent.transactional_actions.runtime import default_action_gateway


def _dispatcher(root, task="task-a", sandbox="workspaceWrite"):
    return ToolExecutionDispatcher(
        workspace=WorkspaceContext(repository_path=root, worktree_path=root, working_directory=root,
                                   sandbox=sandbox, allow_in_place_write=True),
        task_id=task,
    )


def test_git_grant_is_scoped_to_task_and_exact_action(tmp_path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "sample").write_text("sample\n")
    subprocess.run(["git", "add", "sample"], cwd=repo, check=True)
    original = git_tools._run_git_direct
    commits = []

    def fake_git(args, **kwargs):
        if args[0] == "commit":
            commits.append(args)
            return {"ok": True, "state": git_tools.observe_git_state(repo).to_dict()}
        return original(args, **kwargs)

    monkeypatch.setattr(git_tools, "_run_git_direct", fake_git)
    agent = _dispatcher(repo)
    args = {"message": "example"}
    pending = agent.dispatch("git_commit", args)
    assert pending["permission_required"] and pending["inbox_item_id"]
    gateway = default_action_gateway(repo)
    assert gateway.store.get_action(pending["action_id"]).parent_task_id == "task-a"
    grant = gateway.approve(pending["action_id"], approved_by=getpass.getuser())
    changed = agent.dispatch("git_commit", {"message": "different", "action_approval_id": grant})
    foreign = _dispatcher(repo, "task-b").dispatch("git_commit", {**args, "action_approval_id": grant})
    assert changed["error_code"] == foreign["error_code"] == "INVALID_APPROVAL"
    assert not commits and gateway.approvals.get(grant).consumed_at is None
    result = agent.dispatch("git_commit", {**args, "action_approval_id": grant})
    assert result["ok"] and result["action_id"] == pending["action_id"]
    assert commits == [["commit", "-m", "example"]]
    assert gateway.approvals.get(grant).consumed_at is not None


@pytest.mark.parametrize("index_result", ["changed", "unreadable", "timeout"])
def test_commit_approval_fails_closed_if_index_changes_or_cannot_be_read(tmp_path, monkeypatch, index_result):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "sample").write_text("sample\n")
    subprocess.run(["git", "add", "sample"], cwd=repo, check=True)
    raw_git = git_tools._run_raw_git
    run_git = git_tools._run_git_direct
    changed = False

    def fake_raw(args, **kwargs):
        result = raw_git(args, **kwargs)
        if changed and args == ["ls-files", "--stage", "--full-name", "-z"]:
            if index_result == "timeout":
                raise subprocess.TimeoutExpired(args, 30)
            if index_result == "unreadable":
                return subprocess.CompletedProcess(args, 1, "", "index error")
            # Simulate an index entry omitted by a presentation-oriented diff.
            return subprocess.CompletedProcess(args, 0, result.stdout + "160000 " + "a" * 40 + " 0\tmodule\0", "")
        return result

    def no_commit(args, **kwargs):
        if args[0] == "commit":
            pytest.fail("unverified commit must not execute")
        return run_git(args, **kwargs)

    monkeypatch.setattr(git_tools, "_run_raw_git", fake_raw)
    monkeypatch.setattr(git_tools, "_run_git_direct", no_commit)
    agent = _dispatcher(repo)
    pending = agent.dispatch("git_commit", {"message": "example"})
    gateway = default_action_gateway(repo)
    grant = gateway.approve(pending["action_id"], approved_by=getpass.getuser())
    changed = True
    result = agent.dispatch("git_commit", {"message": "example", "action_approval_id": grant})
    assert not result["ok"]
    assert "index" in result["error"] or "staged changes" in result["error"]
    assert gateway.store.get_action(pending["action_id"]).state.value == "failed"


def test_request_rejects_unknown_explicit_inbox_and_reports_action(tmp_path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    pending = _dispatcher(tmp_path).dispatch("run_command", {"command": "python -c 'print(1)'"})
    with pytest.raises(LookupError, match="unknown inbox"):
        approval_request(action_id=pending["action_id"], inbox_item_id="missing", workspace_root=tmp_path)
    result = approval_request(action_id=pending["action_id"], workspace_root=tmp_path, timeout_seconds=0.01)
    assert result["action_id"] == pending["action_id"] and result["status"] == "pending"


def test_commit_worktree_modes_and_git_sandbox_are_blocked(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.setattr(git_tools, "_run_git_direct", lambda *args, **kwargs: pytest.fail("must not execute"))
    assert git_tools.run_git(["commit", "-a", "-m", "example"], repo_path=tmp_path)["error_code"] == "UNSUPPORTED_COMMIT_MODE"
    agent = _dispatcher(tmp_path, sandbox="readOnly")
    assert agent.dispatch("git_commit", {"message": "example"})["error_code"] == "SANDBOX_VIOLATION"
    assert agent.dispatch("git_generic", {"args": ["branch", "new-branch"]})["error_code"] == "SANDBOX_VIOLATION"
    assert agent.dispatch("git_config", {"key": "--unset"})["error_code"] == "SANDBOX_VIOLATION"
    assert git_tools.classify_git_risk(["branch", "new-branch"]) is git_tools.GitRiskLevel.LOCAL_SAFE_WRITE
    assert git_tools.classify_git_risk(["config", "--unset"]) is git_tools.GitRiskLevel.LOCAL_SAFE_WRITE
    assert git_tools.classify_git_risk(["diff", "--output=patch.txt"]) is git_tools.GitRiskLevel.LOCAL_SAFE_WRITE
    assert git_tools.classify_git_risk(["branch", "--show-current"]) is git_tools.GitRiskLevel.READ_ONLY


def test_show_revision_cannot_smuggle_git_options_into_read_only_sandbox(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.setattr(git_tools, "_run_git_direct", lambda *args, **kwargs: pytest.fail("must not execute"))
    result = _dispatcher(tmp_path, sandbox="readOnly").dispatch(
        "git_show", {"revision": "--output=outside.txt"}
    )
    assert not result["ok"]
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.parametrize("tool", ["git.commit", "git.generic"])
def test_queue_git_propagates_task_and_exact_grant(tmp_path, monkeypatch, tool):
    from mana_agent.multi_agent.core.types import QueueJob, QueueJobType
    from mana_agent.multi_agent.tools.tool_manager import ToolsManager

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "sample").write_text("sample\n")
    subprocess.run(["git", "add", "sample"], cwd=repo, check=True)
    original = git_tools._run_git_direct
    commits = []

    def fake_git(args, **kwargs):
        if args[0] == "commit":
            commits.append(args)
            return {"ok": True, "state": git_tools.observe_git_state(repo).to_dict()}
        return original(args, **kwargs)

    monkeypatch.setattr(git_tools, "_run_git_direct", fake_git)
    manager = ToolsManager(repo)

    def job(task, message, grant=""):
        args = ({"message": message} if tool == "git.commit"
                else {"args": ["commit", "-m", message]})
        return QueueJob(job_id="job-1", task_id=task, requested_by_agent_id="agent-1",
                        job_type=QueueJobType.GIT,
                        payload={"tool": tool, "args": {**args, "action_approval_id": grant}})

    pending = manager.execute_job(job("task-1", "example"))
    assert not pending.ok and pending.result["permission_required"] and not commits
    gateway = default_action_gateway(repo)
    assert gateway.store.get_action(pending.result["action_id"]).parent_task_id == "task-1"
    grant = gateway.approve(pending.result["action_id"], approved_by=getpass.getuser())
    assert not manager.execute_job(job("task-2", "example", grant)).ok
    assert not manager.execute_job(job("task-1", "different", grant)).ok
    assert not commits
    completed = manager.execute_job(job("task-1", "example", grant))
    assert completed.ok and completed.result["action_id"] == pending.result["action_id"]
    assert commits == [["commit", "-m", "example"]]
