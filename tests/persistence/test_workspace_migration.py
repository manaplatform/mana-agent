"""Tests for TaskboardGatewayMigrator (restart-safe migration)."""

import json
from pathlib import Path

import pytest

from mana_agent.gateway.lanes import LaneId, LanePriority, LaneTaskState, LockMode
from mana_agent.multi_agent.core.types import RiskLevel, TaskStatus, to_jsonable, utc_now
from mana_agent.persistence.migration import TaskboardGatewayMigrator
from mana_agent.persistence.workspace_db import WorkspaceDatabase
from mana_agent.persistence.workspace_repository import WorkspaceRepository


@pytest.fixture
def migration_setup(tmp_path: Path, monkeypatch):
    workspace_id = "test-migrated-workspace"
    ws_dir = tmp_path / "workspaces" / workspace_id
    ws_dir.mkdir(parents=True, exist_ok=True)

    # Monkeypatch workspace_dir to return our tmp path
    monkeypatch.setattr(
        "mana_agent.persistence.migration.workspace_dir",
        lambda wid: tmp_path / "workspaces" / wid,
    )
    monkeypatch.setattr(
        "mana_agent.persistence.workspace_db.workspace_dir",
        lambda wid: tmp_path / "workspaces" / wid,
    )

    db_path = ws_dir / "state.db"
    db = WorkspaceDatabase(workspace_id, db_path=db_path)
    migrator = TaskboardGatewayMigrator(workspace_id, db=db)
    yield migrator, ws_dir, db
    db.close()


def test_migrate_taskboard_legacy_files(migration_setup):
    migrator, ws_dir, db = migration_setup
    taskboard_dir = ws_dir / "taskboard"
    taskboard_dir.mkdir(parents=True, exist_ok=True)

    # Create legacy state.json
    state_payload = {
        "schema_version": 2,
        "tasks": {
            "task-001": {
                "task_id": "task-001",
                "root_task_id": "task-001",
                "parent_task_id": None,
                "title": "Legacy task 1",
                "user_request": "Do legacy work",
                "normalized_goal": "Legacy work goal",
                "status": "in_progress",
                "priority": 100,
                "risk_level": "low",
                "workspace_id": "test-migrated-workspace",
                "repository_ids": ["repo-1"],
                "depends_on": [],
                "created_at": utc_now().isoformat(),
                "updated_at": utc_now().isoformat(),
            }
        },
    }
    (taskboard_dir / "state.json").write_text(json.dumps(state_payload), encoding="utf-8")

    # Create legacy history.jsonl
    event_payload = {"event_type": "task.created", "payload": {"task_id": "task-001"}, "created_at": utc_now().isoformat()}
    (taskboard_dir / "history.jsonl").write_text(json.dumps(event_payload) + "\n", encoding="utf-8")

    # Create legacy decisions.json
    dec_payload = {
        "dec-1": {
            "decision_id": "dec-1",
            "task_id": "task-001",
            "discussion_id": None,
            "made_by_agent_id": "agent-1",
            "decision_status": "approved",
            "summary": "Legacy decision",
            "rationale_summary": "Legacy rationale",
            "selected_route": "legacy",
            "assigned_agent_ids": ["agent-1"],
            "required_verification": [],
            "created_at": utc_now().isoformat(),
        }
    }
    (taskboard_dir / "decisions.json").write_text(json.dumps(dec_payload), encoding="utf-8")

    # Run migration
    res = migrator.migrate_taskboard(cleanup_legacy=True)
    assert res["status"] == "completed"
    assert res["tasks"] == 1
    assert res["events"] == 1
    assert res["decisions"] == 1

    # Verify migration is marked complete
    assert migrator.is_taskboard_migrated() is True

    # Check imported task in SQLite repository
    repo = WorkspaceRepository(migrator.workspace_id, db=db)
    task = repo.get_task("task-001")
    assert task.title == "Legacy task 1"
    assert task.status == TaskStatus.IN_PROGRESS

    # Check decisions
    decision = repo.get_decision("dec-1")
    assert decision is not None
    assert decision.summary == "Legacy decision"

    # Subsequent run returns already_migrated
    res2 = migrator.migrate_taskboard()
    assert res2["status"] == "already_migrated"


def test_migrate_gateway_legacy_files(migration_setup):
    migrator, ws_dir, db = migration_setup
    gateway_dir = ws_dir / "gateway"
    gateway_dir.mkdir(parents=True, exist_ok=True)

    # Create legacy lane_coordinator.json
    gateway_payload = {
        "schema_version": 2,
        "updated_at": utc_now().isoformat(),
        "executions": [
            {
                "task_id": "lane-task-1",
                "root_task_id": "lane-task-1",
                "owning_lane": "code",
                "state": "running",
                "normalized_intent": "Code something",
                "repository_id": "repo-1",
                "workspace_id": "test-migrated-workspace",
                "session_id": "session-1",
                "priority": "normal",
                "budget": {
                    "reserved_input_tokens": 500,
                    "reserved_output_tokens": 250,
                },
                "handoffs": [],
                "target_files": [],
                "capabilities": [],
                "changed_files": [],
                "verification_state": {},
                "last_heartbeat": utc_now().isoformat(),
                "created_at": utc_now().isoformat(),
                "updated_at": utc_now().isoformat(),
            }
        ],
        "waiters": [],
    }
    (gateway_dir / "lane_coordinator.json").write_text(json.dumps(gateway_payload), encoding="utf-8")

    # Run migration
    res = migrator.migrate_gateway(cleanup_legacy=True)
    assert res["status"] == "completed"
    assert res["executions"] == 1

    assert migrator.is_gateway_migrated() is True

    # Verify execution in SQLite repository
    repo = WorkspaceRepository(migrator.workspace_id, db=db)
    ex = repo.get_execution("lane-task-1")
    assert ex is not None
    assert ex.owning_lane == LaneId.CODING
    assert ex.state == LaneTaskState.RUNNING


def test_migration_error_on_corrupt_file(migration_setup):
    migrator, ws_dir, _ = migration_setup
    taskboard_dir = ws_dir / "taskboard"
    taskboard_dir.mkdir(parents=True, exist_ok=True)

    (taskboard_dir / "state.json").write_text("{ corrupt json ...", encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupt taskboard state.json"):
        migrator.migrate_taskboard()
