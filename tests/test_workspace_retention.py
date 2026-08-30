"""Regression tests for Durable Workspace/Repository Retention and Reference-Aware GC."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from mana_agent.config.settings import Settings
from mana_agent.execution_supervisor.models import ExecutionState, TaskRecord, CheckpointRecord
from mana_agent.execution_supervisor.store import LocalExecutionStore
from mana_agent.workspaces.models import (
    RepositoryRecord,
    SessionRecord,
    WorkspaceDiscoveryConfig,
    WorkspaceRecord,
)
from mana_agent.workspaces.paths import repository_dir, workspace_dir
from mana_agent.workspaces.retention import (
    ReferenceAwareGC,
    RetentionClass,
    RetentionPolicy,
    TombstoneRecord,
)
from mana_agent.workspaces.service import WorkspaceService
from mana_agent.workspaces.store import WorkspaceStore


@pytest.fixture
def test_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[WorkspaceStore, LocalExecutionStore, ReferenceAwareGC]:
    home = tmp_path / "mana-home"
    monkeypatch.setenv("MANA_HOME", str(home))
    ws_store = WorkspaceStore()
    exec_store = LocalExecutionStore(home / "execution")
    policy = RetentionPolicy(
        max_completed_tasks_per_workspace=3,
        temp_workspace_ttl_seconds=1,  # 1 second for fast test
        max_diagnostic_log_bytes=100,
    )
    gc = ReferenceAwareGC(workspace_store=ws_store, execution_store=exec_store, policy=policy)
    return ws_store, exec_store, gc


def test_referenced_workspaces_never_deleted(test_env: tuple[WorkspaceStore, LocalExecutionStore, ReferenceAwareGC]) -> None:
    """Workspaces referenced by active tasks, sessions, or checkpoints are never pruned by GC."""
    ws_store, exec_store, gc = test_env

    # 1. Workspace referenced by active task
    ws1 = ws_store.save_workspace(
        WorkspaceRecord(
            name="workspace_with_active_task",
            implicit=True,
            created_at=(datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
            updated_at=(datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
        )
    )
    task1 = TaskRecord(
        task_id="active_task_1",
        workspace_id=ws1.workspace_id,
        state=ExecutionState.RUNNING,
    )
    exec_store.create_task(task1)

    # 2. Workspace referenced by active session
    ws2 = ws_store.save_workspace(
        WorkspaceRecord(
            name="workspace_with_active_session",
            implicit=True,
            created_at=(datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
            updated_at=(datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
        )
    )
    repo = ws_store.save_repository(
        RepositoryRecord(name="repo", canonical_path=str(Path("/tmp/repo_test").resolve()))
    )
    ws_store.save_session(
        SessionRecord(
            session_id="session_active_1",
            workspace_id=ws2.workspace_id,
            primary_repository_id=repo.repository_id,
            cwd="/tmp/repo_test",
            status="active",
        )
    )

    # 3. Explicit (user-configured) workspace
    ws3 = ws_store.save_workspace(
        WorkspaceRecord(
            name="explicit_user_workspace",
            implicit=False,
            created_at=(datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
            updated_at=(datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
        )
    )

    metrics = gc.run_retention_pass()

    assert metrics.workspace_records_pruned == 0
    assert ws_store.get_workspace(ws1.workspace_id) is not None
    assert ws_store.get_workspace(ws2.workspace_id) is not None
    assert ws_store.get_workspace(ws3.workspace_id) is not None


def test_orphaned_stale_workspaces_removed_with_tombstones(test_env: tuple[WorkspaceStore, LocalExecutionStore, ReferenceAwareGC]) -> None:
    """Stale implicit workspaces past TTL without references are safely deleted and tombstoned."""
    ws_store, exec_store, gc = test_env

    # Create stale implicit workspace with no active references
    stale_ws = ws_store.save_workspace(
        WorkspaceRecord(
            name="stale_temp_ws",
            implicit=True,
            created_at=(datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
            updated_at=(datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
        )
    )

    metrics = gc.run_retention_pass()

    assert metrics.workspace_records_pruned == 1
    assert metrics.tombstones_created >= 1

    # Workspace file should be gone
    with pytest.raises(Exception):
        ws_store.get_workspace(stale_ws.workspace_id)

    # Tombstone record should be recorded
    tombstone = gc.get_tombstone(stale_ws.workspace_id)
    assert tombstone is not None
    assert tombstone.id == stale_ws.workspace_id
    assert tombstone.type == "workspace"
    assert tombstone.terminal_status == "deleted_expired"


def test_repository_records_canonical_and_do_not_duplicate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Registering the same repository multiple times updates the canonical record in place without duplicating."""
    home = tmp_path / "mana-home"
    monkeypatch.setenv("MANA_HOME", str(home))
    service = WorkspaceService()

    repo_dir = tmp_path / "my_project"
    repo_dir.mkdir(parents=True, exist_ok=True)

    r1 = service.register_repository(repo_dir)
    r2 = service.register_repository(repo_dir)
    r3 = service.register_repository(repo_dir, refresh=True)

    assert r1.repository_id == r2.repository_id == r3.repository_id
    all_repos = service.store.list_repositories()
    assert len(all_repos) == 1
    assert all_repos[0].repository_id == r1.repository_id


def test_completed_tasks_compacted_to_tombstones(test_env: tuple[WorkspaceStore, LocalExecutionStore, ReferenceAwareGC]) -> None:
    """Excess completed tasks are compacted into tombstones and pruned from active execution store."""
    ws_store, exec_store, gc = test_env
    # Max completed tasks is set to 3 in fixture policy

    for i in range(6):
        t = TaskRecord(
            task_id=f"completed_task_{i}",
            workspace_id="ws_main",
            state=ExecutionState.COMPLETED,
            normalized_intent=f"Intent for completed task {i}",
            created_at=datetime.now(timezone.utc) - timedelta(minutes=10 - i),
            updated_at=datetime.now(timezone.utc) - timedelta(minutes=10 - i),
        )
        exec_store.create_task(t)

    metrics = gc.run_retention_pass()

    # 6 tasks - max 3 = 3 excess tasks compacted
    assert metrics.tasks_compacted == 3
    remaining_tasks = exec_store.list_tasks()
    assert len(remaining_tasks) == 3

    # Check that tombstones exist for pruned tasks
    tombstones = gc.list_tombstones()
    task_tombstones = [t for t in tombstones if t.type == "task"]
    assert len(task_tombstones) == 3


def test_diagnostic_log_size_truncation(test_env: tuple[WorkspaceStore, LocalExecutionStore, ReferenceAwareGC]) -> None:
    """Diagnostic log files exceeding limit are truncated using rolling tail truncation."""
    ws_store, exec_store, gc = test_env

    logs_dir = exec_store.root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    test_log = logs_dir / "audit.log"
    # Write 500 bytes (limit in fixture is 100 bytes)
    test_log.write_bytes(b"LOG LINE DATA " * 35)

    metrics = gc.run_retention_pass()

    assert metrics.logs_truncated_bytes > 0
    assert test_log.stat().st_size <= 150
    content = test_log.read_bytes()
    assert b"truncated by retention policy" in content


def test_gc_idempotency_and_startup(test_env: tuple[WorkspaceStore, LocalExecutionStore, ReferenceAwareGC]) -> None:
    """Running GC multiple times is safe, idempotent, and maintains invariants."""
    ws_store, exec_store, gc = test_env

    # First pass
    m1 = gc.run_retention_pass()
    # Second pass immediately after
    m2 = gc.run_retention_pass()

    assert m2.workspace_records_pruned == 0
    assert m2.tasks_compacted == 0
    assert m2.stale_records_pruned == 0
