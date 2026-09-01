"""Atomic, restart-safe migration from legacy taskboard and gateway JSON files to SQLite."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mana_agent.gateway.lane_models import LaneBudget, LaneExecution, LaneHandoff, LockLease
from mana_agent.gateway.lanes import LaneId, LanePriority, LaneTaskState, LockMode
from mana_agent.multi_agent.taskboard.deserializers import (
    decision_from_dict,
    discussion_from_dict,
    message_from_dict,
    task_from_dict,
)
from mana_agent.persistence.workspace_db import WorkspaceDatabase, get_workspace_db
from mana_agent.persistence.workspace_repository import WorkspaceRepository
from mana_agent.workspaces.paths import workspace_dir


def _file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _normalize_lane_id(value: Any) -> LaneId:
    if isinstance(value, LaneId):
        return value
    val = str(value).lower().strip()
    if val == "code":
        return LaneId.CODING
    if val == "plan":
        return LaneId.RESEARCH
    return LaneId(val)


class TaskboardGatewayMigrator:
    """Migrates existing taskboard/* and gateway/* directory files into the workspace SQLite database."""

    def __init__(self, workspace_id: str, db: WorkspaceDatabase | None = None) -> None:
        self.workspace_id = str(workspace_id)
        self.db = db or get_workspace_db(self.workspace_id)
        self.repository = WorkspaceRepository(self.workspace_id, db=self.db)
        self.base_dir = workspace_dir(self.workspace_id)
        self.taskboard_dir = self.base_dir / "taskboard"
        self.gateway_dir = self.base_dir / "gateway"

    def is_taskboard_migrated(self) -> bool:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT status FROM workspace_migration_state WHERE source_type = 'taskboard';",
            ).fetchone()
            return bool(row and row["status"] == "completed")

    def is_gateway_migrated(self) -> bool:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT status FROM workspace_migration_state WHERE source_type = 'gateway';",
            ).fetchone()
            return bool(row and row["status"] == "completed")

    def migrate_all(self, *, cleanup_legacy: bool = False) -> dict[str, Any]:
        results = {
            "taskboard": self.migrate_taskboard(cleanup_legacy=cleanup_legacy),
            "gateway": self.migrate_gateway(cleanup_legacy=cleanup_legacy),
        }
        return results

    def migrate_taskboard(self, *, cleanup_legacy: bool = False) -> dict[str, Any]:
        if self.is_taskboard_migrated():
            return {"status": "already_migrated"}

        state_file = self.taskboard_dir / "state.json"
        history_file = self.taskboard_dir / "history.jsonl"
        decisions_file = self.taskboard_dir / "decisions.json"
        discussions_file = self.taskboard_dir / "discussions.json"
        messages_file = self.taskboard_dir / "messages.jsonl"

        has_legacy = any(
            p.exists() for p in (state_file, history_file, decisions_file, discussions_file, messages_file)
        )
        if not has_legacy:
            # Mark clean migration for new workspace
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO workspace_migration_state (
                        source_type, migrated_at, source_checksum, row_counts, status
                    ) VALUES ('taskboard', ?, 'none', '{}', 'completed');
                    """,
                    (datetime.now(timezone.utc).isoformat(),),
                )
            return {"status": "completed", "tasks": 0, "events": 0, "discussions": 0, "decisions": 0, "messages": 0}

        checksums = {
            "state_json": _file_sha256(state_file),
            "history_jsonl": _file_sha256(history_file),
            "decisions_json": _file_sha256(decisions_file),
            "discussions_json": _file_sha256(discussions_file),
            "messages_jsonl": _file_sha256(messages_file),
        }

        task_count = 0
        event_count = 0
        discussion_count = 0
        decision_count = 0
        message_count = 0

        # Read all legacy payloads into memory before starting transaction
        tasks_to_import = []
        if state_file.exists():
            try:
                payload = json.loads(state_file.read_text(encoding="utf-8"))
                raw_tasks = payload.get("tasks", {}) if isinstance(payload, dict) else {}
                for task_dict in raw_tasks.values():
                    if isinstance(task_dict, dict):
                        tasks_to_import.append(task_from_dict(task_dict))
            except Exception as exc:
                raise ValueError(f"Corrupt taskboard state.json: {state_file}") from exc

        events_to_import = []
        if history_file.exists():
            for line in history_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    events_to_import.append(json.loads(line))
                except Exception:
                    continue

        discussions_to_import = []
        if discussions_file.exists():
            try:
                raw_discussions = json.loads(discussions_file.read_text(encoding="utf-8"))
                for d_dict in raw_discussions.values():
                    if isinstance(d_dict, dict):
                        discussions_to_import.append(discussion_from_dict(d_dict))
            except Exception:
                pass

        decisions_to_import = []
        if decisions_file.exists():
            try:
                raw_decisions = json.loads(decisions_file.read_text(encoding="utf-8"))
                for dec_dict in raw_decisions.values():
                    if isinstance(dec_dict, dict):
                        decisions_to_import.append(decision_from_dict(dec_dict))
            except Exception:
                pass

        messages_to_import = []
        if messages_file.exists():
            for line in messages_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    messages_to_import.append(message_from_dict(json.loads(line)))
                except Exception:
                    continue

        # Atomic commit
        for t in tasks_to_import:
            self.repository.save_task(t)
            task_count += 1

        for ev in events_to_import:
            self.repository.append_task_event(
                task_id=ev.get("payload", {}).get("task_id") if isinstance(ev.get("payload"), dict) else None,
                event_type=str(ev.get("event_type") or "task.event"),
                payload=ev.get("payload", {}),
                created_at=None,
            )
            event_count += 1

        for d in discussions_to_import:
            self.repository.save_discussion(d)
            discussion_count += 1

        for dec in decisions_to_import:
            self.repository.save_decision(dec)
            decision_count += 1

        for m in messages_to_import:
            self.repository.save_message(m)
            message_count += 1

        counts = {
            "tasks": task_count,
            "events": event_count,
            "discussions": discussion_count,
            "decisions": decision_count,
            "messages": message_count,
        }

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO workspace_migration_state (
                    source_type, migrated_at, source_checksum, row_counts, status
                ) VALUES ('taskboard', ?, ?, ?, 'completed');
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(checksums, sort_keys=True),
                    json.dumps(counts, sort_keys=True),
                ),
            )

        if cleanup_legacy:
            for p in (state_file, history_file, decisions_file, discussions_file, messages_file):
                if p.exists():
                    p.unlink(missing_ok=True)

        return {"status": "completed", **counts}

    def migrate_gateway(self, *, cleanup_legacy: bool = False) -> dict[str, Any]:
        if self.is_gateway_migrated():
            return {"status": "already_migrated"}

        state_file = self.gateway_dir / "lane_coordinator.json"
        locks_file = self.gateway_dir / "lane_locks.json"

        has_legacy = state_file.exists() or locks_file.exists()
        if not has_legacy:
            with self.db.transaction() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO workspace_migration_state (
                        source_type, migrated_at, source_checksum, row_counts, status
                    ) VALUES ('gateway', ?, 'none', '{}', 'completed');
                    """,
                    (datetime.now(timezone.utc).isoformat(),),
                )
            return {"status": "completed", "executions": 0, "locks": 0, "waiters": 0}

        checksums = {
            "lane_coordinator_json": _file_sha256(state_file),
            "lane_locks_json": _file_sha256(locks_file),
        }

        executions_to_import = []
        waiters_to_import = []
        locks_to_import = []

        if state_file.exists():
            try:
                payload = json.loads(state_file.read_text(encoding="utf-8"))
                for raw in payload.get("executions", []):
                    item = dict(raw)
                    raw_budget = item.pop("budget", {})
                    budget_kwargs = {k: v for k, v in raw_budget.items() if hasattr(LaneBudget, k)} if isinstance(raw_budget, dict) else {}
                    budget = LaneBudget(**budget_kwargs)
                    handoffs = []
                    for h_raw in item.pop("handoffs", []):
                        h_dict = dict(h_raw)
                        h_dict["source_lane"] = _normalize_lane_id(h_dict["source_lane"])
                        h_dict["target_lane"] = _normalize_lane_id(h_dict["target_lane"])
                        h_dict["budget_consumed"] = LaneBudget(**h_dict.get("budget_consumed", {}))
                        handoffs.append(LaneHandoff(**h_dict))
                    item["owning_lane"] = _normalize_lane_id(item["owning_lane"])
                    item["state"] = LaneTaskState(item["state"])
                    item["priority"] = LanePriority(item["priority"])
                    execution = LaneExecution(budget=budget, handoffs=handoffs, **item)
                    execution.supervisor_lease_token = ""
                    executions_to_import.append(execution)

                waiters_to_import = [dict(w) for w in payload.get("waiters", []) if isinstance(w, dict)]
            except Exception:
                pass

        if locks_file.exists():
            try:
                lock_rows = json.loads(locks_file.read_text(encoding="utf-8")).get("locks", [])
                for raw in lock_rows:
                    item = dict(raw)
                    item["mode"] = LockMode(item["mode"])
                    locks_to_import.append(LockLease(**item))
            except Exception:
                pass

        exec_count = 0
        lock_count = 0

        for ex in executions_to_import:
            self.repository.save_execution(ex)
            exec_count += 1

        for lock in locks_to_import:
            self.repository.save_lock(lock)
            lock_count += 1

        if waiters_to_import:
            self.repository.save_waiters(waiters_to_import)

        counts = {
            "executions": exec_count,
            "locks": lock_count,
            "waiters": len(waiters_to_import),
        }

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO workspace_migration_state (
                    source_type, migrated_at, source_checksum, row_counts, status
                ) VALUES ('gateway', ?, ?, ?, 'completed');
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(checksums, sort_keys=True),
                    json.dumps(counts, sort_keys=True),
                ),
            )

        if cleanup_legacy:
            for p in (state_file, locks_file):
                if p.exists():
                    p.unlink(missing_ok=True)

        return {"status": "completed", **counts}
