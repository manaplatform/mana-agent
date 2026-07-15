"""Focused tests for Mana-managed agent worktrees."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mana_agent.commands.cli import app
from mana_agent.multi_agent.core.types import QueueJob, QueueJobType, RiskLevel, TaskStatus
from mana_agent.multi_agent.queue.queue_manager import QueueManager
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.multi_agent.tools.tool_manager import ToolsManager
from mana_agent.multi_agent.worktrees import (
    WorkspaceError,
    WorkspaceManager,
    WorkspaceStatus,
    review_task_branch,
)
from mana_agent.multi_agent.worktrees.manager import coding_route_requires_worktree
from mana_agent.multi_agent.worktrees.store import ManagedWorkspaceStore
from mana_agent.workspaces.paths import repository_id_for_path


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    init = _git(path, "init", "-b", "main")
    if init.returncode != 0:
        assert _git(path, "init").returncode == 0
        assert _git(path, "branch", "-M", "main").returncode == 0
    assert _git(path, "config", "user.name", "Mana Agent Test").returncode == 0
    assert _git(path, "config", "user.email", "test@example.com").returncode == 0
    (path / "README.md").write_text("hello\n", encoding="utf-8")
    (path / "src").mkdir(exist_ok=True)
    (path / "src" / "app.py").write_text("value = 1\n", encoding="utf-8")
    assert _git(path, "add", ".").returncode == 0
    assert _git(path, "commit", "-m", "test: initial commit").returncode == 0
    return path


@pytest.fixture
def mana_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "mana-home"
    home.mkdir()
    monkeypatch.setenv("MANA_HOME", str(home))
    return home


@pytest.fixture
def repo(tmp_path: Path, mana_home: Path) -> Path:
    _ = mana_home
    return _init_repo(tmp_path / "source-repo")


def test_coding_route_requires_worktree() -> None:
    assert coding_route_requires_worktree("coding")
    assert coding_route_requires_worktree("tool")
    assert coding_route_requires_worktree("high_risk_tool")
    assert not coding_route_requires_worktree("analyze")
    assert not coding_route_requires_worktree("chat")


def test_deterministic_task_to_workspace_mapping(repo: Path) -> None:
    manager = WorkspaceManager(repo)
    slug_a = manager.task_slug("task_abc", title="Fix login")
    slug_b = manager.task_slug("task_abc", title="Fix login")
    slug_c = manager.task_slug("task_xyz", title="Fix login")
    assert slug_a == slug_b
    assert slug_a != slug_c
    assert slug_a.startswith("fix-login-")


def test_workspace_isolation_and_branch_creation(repo: Path) -> None:
    manager = WorkspaceManager(repo)
    base = _git(repo, "rev-parse", "HEAD").stdout.strip()
    ws = manager.create_for_task("task_iso_1", title="Isolate edits", assigned_agent_id="agent_coding_1")

    assert ws.status == WorkspaceStatus.READY
    assert ws.base_revision == base
    assert ws.branch_name.startswith("mana/")
    assert Path(ws.worktree_path).is_dir()
    assert Path(ws.worktree_path).resolve() != repo.resolve()
    assert _git(Path(ws.worktree_path), "branch", "--show-current").stdout.strip() == ws.branch_name
    # Source checkout stays on main and is clean.
    assert _git(repo, "branch", "--show-current").stdout.strip() in {"main", "master"}
    assert _git(repo, "status", "--porcelain").stdout.strip() == ""


def test_parallel_tasks_never_share_checkout(repo: Path) -> None:
    manager = WorkspaceManager(repo)
    first = manager.create_for_task("task_parallel_a", title="Task A")
    second = manager.create_for_task("task_parallel_b", title="Task B")

    assert first.worktree_path != second.worktree_path
    assert first.branch_name != second.branch_name

    (Path(first.worktree_path) / "src" / "app.py").write_text("value = 11\n", encoding="utf-8")
    (Path(second.worktree_path) / "src" / "app.py").write_text("value = 22\n", encoding="utf-8")

    assert (repo / "src" / "app.py").read_text(encoding="utf-8") == "value = 1\n"
    assert (Path(first.worktree_path) / "src" / "app.py").read_text(encoding="utf-8") == "value = 11\n"
    assert (Path(second.worktree_path) / "src" / "app.py").read_text(encoding="utf-8") == "value = 22\n"


def test_branch_collision_gets_unique_suffix(repo: Path) -> None:
    manager = WorkspaceManager(repo)
    first = manager.create_for_task("task_coll_1", title="Same Title")
    # Force a second workspace with the same preferred slug base by creating a branch that collides.
    # Use the same slug path uniqueness by pre-creating branch name of second task.
    slug = manager.task_slug("task_coll_2", title="Same Title")
    preferred = f"mana/{slug}"
    assert _git(repo, "branch", preferred).returncode == 0
    second = manager.create_for_task("task_coll_2", title="Same Title")
    assert second.branch_name != first.branch_name
    assert second.branch_name.startswith("mana/")
    assert preferred == first.branch_name or preferred != second.branch_name


def test_tools_manager_uses_execution_repo_root_not_primary(repo: Path) -> None:
    manager = WorkspaceManager(repo)
    ws = manager.create_for_task("task_tools_root", title="Tools root")
    (Path(ws.worktree_path) / "src" / "app.py").write_text("value = 99\n", encoding="utf-8")

    tools = ToolsManager(repo)
    job = QueueJob(
        job_id="job_1",
        task_id="task_tools_root",
        requested_by_agent_id="agent_coding",
        job_type=QueueJobType.REPO_READ,
        payload={"path": "src/app.py"},
        execution_repo_root=ws.worktree_path,
    )
    result = tools.execute_job(job)
    assert result.ok
    assert result.result["content"] == "value = 99\n"
    assert result.result["execution_repo_root"] == str(Path(ws.worktree_path).resolve())
    # Primary checkout untouched.
    assert (repo / "src" / "app.py").read_text(encoding="utf-8") == "value = 1\n"


def test_queue_manager_stamps_execution_root_from_task(repo: Path) -> None:
    manager = WorkspaceManager(repo)
    ws = manager.create_for_task("task_queue_root", title="Queue root")
    board = TaskBoard(repo)
    task = board.create_task(
        title="Queue root",
        user_request="edit something",
        owner_agent_id="agent_main_0001",
    )
    # Force known task id mapping by attaching workspace fields onto the created task.
    task.task_id  # ensure created
    # Attach managed workspace identity onto this task for queue scope.
    manager.attach_to_taskboard(task, ws)
    # Re-key store: create_for_task used a different task id; rebind metadata for this test.
    ws.task_id = task.task_id
    ManagedWorkspaceStore(manager.repository_id).save(ws)
    board.save()

    queue = QueueManager(repo, taskboard=board)
    job = queue.enqueue(
        task_id=task.task_id,
        requested_by_agent_id="agent_coding_0001",
        approved_by_agent_id="agent_main_0001",
        job_type=QueueJobType.REPO_READ,
        payload={"path": "src/app.py"},
        purpose="read isolated",
    )
    assert job.execution_repo_root == ws.worktree_path
    assert job.payload.get("execution_repo_root") == ws.worktree_path


def test_verification_runs_in_worktree_root(repo: Path) -> None:
    manager = WorkspaceManager(repo)
    ws = manager.create_for_task("task_verify_root", title="Verify root")
    board = TaskBoard(repo)
    task = board.create_task(title="Verify", user_request="verify", owner_agent_id="agent_main_0001")
    ws.task_id = task.task_id
    manager.attach_to_taskboard(task, ws)
    ManagedWorkspaceStore(manager.repository_id).save(ws)
    board.save()

    queue = QueueManager(repo, taskboard=board)
    job = queue.enqueue(
        task_id=task.task_id,
        requested_by_agent_id="agent_verifier_0001",
        approved_by_agent_id="agent_main_0001",
        job_type=QueueJobType.SHELL,
        payload={"command": "git rev-parse --show-toplevel"},
        purpose="verify inside worktree",
    )
    ran = queue.run_next(worker_agent_id=job.assigned_worker_agent_id)
    assert ran is not None
    assert ran.status.value == "done", ran.error or ran.result
    stdout = str((ran.result or {}).get("stdout") or "").strip()
    expected = str(Path(ws.worktree_path).resolve())
    assert expected in stdout or Path(stdout).resolve() == Path(expected).resolve()
    assert Path((ran.result or {}).get("execution_repo_root") or expected).resolve() == Path(expected).resolve()


def test_reviewer_diff_base_correctness(repo: Path) -> None:
    manager = WorkspaceManager(repo)
    ws = manager.create_for_task("task_review_base", title="Review base")
    manager.mark_running(ws.task_id)
    wt = Path(ws.worktree_path)
    (wt / "src" / "app.py").write_text("value = 7\n", encoding="utf-8")
    assert _git(wt, "add", "src/app.py").returncode == 0
    assert _git(wt, "commit", "-m", "feat: change value").returncode == 0

    diff = manager.diff(ws.task_id)
    assert diff["ok"] is True
    assert "+value = 7" in str(diff.get("stdout") or "")
    assert diff["base_revision"] == ws.base_revision

    review = review_task_branch(manager, ws.task_id, reviewer_agent_id="agent_reviewer", verification_passed=True)
    assert review["approved"] is True
    assert review["base_revision"] == ws.base_revision
    assert manager.get_for_task(ws.task_id).status == WorkspaceStatus.MERGE_CANDIDATE


def test_dirty_workspace_retained_and_not_auto_deleted(repo: Path) -> None:
    manager = WorkspaceManager(repo)
    ws = manager.create_for_task("task_dirty", title="Dirty")
    (Path(ws.worktree_path) / "src" / "app.py").write_text("dirty\n", encoding="utf-8")
    manager._refresh_git_flags(ws, persist=True)  # noqa: SLF001
    assert manager.get_for_task(ws.task_id).dirty is True

    with pytest.raises(WorkspaceError, match="dirty or unmerged"):
        manager.remove(ws.task_id)

    assert Path(ws.worktree_path).is_dir()
    # Force with intent still works.
    removed = manager.remove(ws.task_id, force=True, explicit_user_intent=True)
    assert removed["ok"] is True


def test_interrupted_task_resume(repo: Path) -> None:
    manager = WorkspaceManager(repo)
    ws = manager.create_for_task("task_resume", title="Resume me")
    manager.mark_running(ws.task_id)
    manager.mark_interrupted(ws.task_id, error="process killed")
    assert manager.get_for_task(ws.task_id).status == WorkspaceStatus.INTERRUPTED

    resumed = manager.resume(ws.task_id, assigned_agent_id="agent_coding_2")
    assert resumed.status == WorkspaceStatus.READY
    assert resumed.assigned_agent_id == "agent_coding_2"
    assert Path(resumed.worktree_path).is_dir()


def test_restart_recovery_reconciles_orphaned_metadata(repo: Path) -> None:
    manager = WorkspaceManager(repo)
    ws = manager.create_for_task("task_orphan", title="Orphan")
    # Simulate deleted worktree directory without git cleanup.
    path = Path(ws.worktree_path)
    # Properly remove via git first then delete metadata mismatch by re-saving path.
    _git(repo, "worktree", "remove", "--force", str(path))
    report = manager.reconcile(task_id=ws.task_id)
    assert ws.task_id in report["orphaned_metadata_tasks"]
    stale = manager.get_for_task(ws.task_id)
    assert stale.status == WorkspaceStatus.STALE
    assert stale.orphaned_metadata is True


def test_merge_requires_explicit_intent_and_never_force(repo: Path) -> None:
    manager = WorkspaceManager(repo)
    ws = manager.create_for_task("task_merge", title="Merge me")
    manager.mark_running(ws.task_id)
    wt = Path(ws.worktree_path)
    (wt / "src" / "app.py").write_text("value = 3\n", encoding="utf-8")
    assert _git(wt, "add", "src/app.py").returncode == 0
    assert _git(wt, "commit", "-m", "feat: merge candidate").returncode == 0
    review_task_branch(manager, ws.task_id, verification_passed=True)

    with pytest.raises(WorkspaceError, match="explicit user intent"):
        manager.merge(ws.task_id, explicit_user_intent=False)

    result = manager.merge(ws.task_id, explicit_user_intent=True)
    assert result["ok"] is True
    assert manager.get_for_task(ws.task_id).status == WorkspaceStatus.MERGED
    assert (repo / "src" / "app.py").read_text(encoding="utf-8") == "value = 3\n"


def test_path_escaping_rejected_in_tools_manager(repo: Path) -> None:
    manager = WorkspaceManager(repo)
    ws = manager.create_for_task("task_escape", title="Escape")
    tools = ToolsManager(repo)
    job = QueueJob(
        job_id="job_escape",
        task_id=ws.task_id,
        requested_by_agent_id="agent_coding",
        job_type=QueueJobType.REPO_READ,
        payload={"path": "../outside.txt"},
        execution_repo_root=ws.worktree_path,
    )
    result = tools.execute_job(job)
    assert result.ok is False
    assert "escape" in str(result.error or "").lower()


def test_cli_worktree_list_create_status_diff(repo: Path, mana_home: Path) -> None:
    _ = mana_home
    runner = CliRunner()
    create = runner.invoke(
        app,
        ["worktree", "create", "task_cli_1", "--root-dir", str(repo), "--title", "CLI Task"],
    )
    assert create.exit_code == 0, create.output
    payload = json.loads(create.output)
    assert payload["task_id"] == "task_cli_1"
    assert payload["branch"].startswith("mana/")

    listed = runner.invoke(app, ["worktree", "list", "--root-dir", str(repo)])
    assert listed.exit_code == 0, listed.output
    rows = json.loads(listed.output)
    assert any(row["task_id"] == "task_cli_1" for row in rows)

    status = runner.invoke(app, ["worktree", "status", "task_cli_1", "--root-dir", str(repo)])
    assert status.exit_code == 0, status.output
    status_payload = json.loads(status.output)
    assert status_payload["base_revision"]
    assert "git" in status_payload

    diff = runner.invoke(app, ["worktree", "diff", "task_cli_1", "--root-dir", str(repo)])
    assert diff.exit_code == 0, diff.output


def test_cli_merge_refuses_without_yes(repo: Path, mana_home: Path) -> None:
    _ = mana_home
    runner = CliRunner()
    runner.invoke(app, ["worktree", "create", "task_cli_merge", "--root-dir", str(repo)])
    merge = runner.invoke(app, ["worktree", "merge", "task_cli_merge", "--root-dir", str(repo)])
    assert merge.exit_code == 2
    assert "explicit user intent" in merge.output.lower() or "pass --yes" in merge.output.lower()


def test_non_worktree_primary_checkout_flow_still_works(repo: Path) -> None:
    """Existing non-worktree coding tools continue against the primary root."""

    tools = ToolsManager(repo)
    job = QueueJob(
        job_id="job_primary",
        task_id="task_primary",
        requested_by_agent_id="agent_coding",
        job_type=QueueJobType.REPO_READ,
        payload={"path": "src/app.py"},
    )
    result = tools.execute_job(job)
    assert result.ok
    assert result.result["content"] == "value = 1\n"
    assert Path(result.result["execution_repo_root"]).resolve() == repo.resolve()


def test_write_lock_keys_are_per_worktree(repo: Path) -> None:
    manager = WorkspaceManager(repo)
    a = manager.create_for_task("task_lock_a", title="Lock A")
    b = manager.create_for_task("task_lock_b", title="Lock B")
    board = TaskBoard(repo)
    ta = board.create_task(title="A", user_request="a", owner_agent_id="agent_main_0001")
    tb = board.create_task(title="B", user_request="b", owner_agent_id="agent_main_0001")
    a.task_id = ta.task_id
    b.task_id = tb.task_id
    manager.attach_to_taskboard(ta, a)
    manager.attach_to_taskboard(tb, b)
    ManagedWorkspaceStore(manager.repository_id).save(a)
    ManagedWorkspaceStore(manager.repository_id).save(b)
    board.save()

    queue = QueueManager(repo, taskboard=board)
    ja = queue.enqueue(
        task_id=ta.task_id,
        requested_by_agent_id="agent_coding_0001",
        approved_by_agent_id="agent_main_0001",
        job_type=QueueJobType.APPLY_PATCH,
        payload={"patch": ""},
        purpose="patch a",
        requires_write_lock=True,
    )
    jb = queue.enqueue(
        task_id=tb.task_id,
        requested_by_agent_id="agent_coding_0001",
        approved_by_agent_id="agent_main_0001",
        job_type=QueueJobType.APPLY_PATCH,
        payload={"patch": ""},
        purpose="patch b",
        requires_write_lock=True,
    )
    assert ja.lock_key != jb.lock_key
    assert ja.lock_key.startswith("worktree:")
    assert jb.lock_key.startswith("worktree:")


def test_repository_id_stable_for_worktree_paths(repo: Path) -> None:
    manager = WorkspaceManager(repo)
    assert manager.repository_id == repository_id_for_path(repo)
    ws = manager.create_for_task("task_id_stable", title="Stable")
    assert ws.repository_id == manager.repository_id
