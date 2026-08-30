"""Durable workspace and repository retention with reference-aware garbage collection."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from mana_agent.config.settings import Settings
from mana_agent.execution_supervisor.models import ExecutionState, TERMINAL_STATES, TaskRecord
from mana_agent.execution_supervisor.store import LocalExecutionStore
from mana_agent.workspaces.models import RepositoryRecord, SessionRecord, WorkspaceRecord
from mana_agent.workspaces.paths import ensure_home_layout, repository_dir, workspace_dir
from mana_agent.workspaces.store import WorkspaceStore, atomic_write_json


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


class RetentionClass(str, Enum):
    KEEP = "KEEP"
    COMPACT = "COMPACT"
    DELETE = "DELETE"


class TombstoneRecord(BaseModel):
    """Summarized tombstone record preserving audit trail when detailed state is pruned."""

    schema_version: int = 1
    id: str
    type: str  # "task", "execution", "workspace", "scan", "session"
    terminal_status: str
    created_at: str
    completed_at: str | None = None
    repository_reference: str | None = None
    workspace_reference: str | None = None
    concise_result_summary: str = ""
    terminal_error_code: str | None = None
    tombstoned_at: str = Field(default_factory=utc_iso)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class RetentionPolicy:
    max_completed_tasks_per_workspace: int = 50
    max_execution_traces: int = 100
    max_provider_events: int = 200
    temp_workspace_ttl_seconds: int = 604800  # 7 days
    stale_cache_ttl_seconds: int = 259200  # 3 days
    max_diagnostic_log_bytes: int = 10485760  # 10MB
    max_recovery_candidates: int = 8
    max_route_history: int = 10

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RetentionPolicy:
        if settings is None:
            settings = Settings()
        return cls(
            max_completed_tasks_per_workspace=int(
                getattr(settings, "mana_retention_max_completed_tasks_per_workspace", 50)
            ),
            max_execution_traces=int(
                getattr(settings, "mana_retention_max_execution_traces", 100)
            ),
            max_provider_events=int(
                getattr(settings, "mana_retention_max_provider_events", 200)
            ),
            temp_workspace_ttl_seconds=int(
                getattr(settings, "mana_retention_temp_workspace_ttl_seconds", 604800)
            ),
            stale_cache_ttl_seconds=int(
                getattr(settings, "mana_retention_stale_cache_ttl_seconds", 259200)
            ),
            max_diagnostic_log_bytes=int(
                getattr(settings, "mana_retention_max_diagnostic_log_bytes", 10485760)
            ),
            max_recovery_candidates=int(
                getattr(settings, "mana_retention_max_recovery_candidates", 8)
            ),
            max_route_history=int(
                getattr(settings, "mana_retention_max_route_history", 10)
            ),
        )


@dataclass
class RetentionMetrics:
    stale_records_pruned: int = 0
    workspace_records_pruned: int = 0
    repository_records_compacted: int = 0
    tasks_compacted: int = 0
    tombstones_created: int = 0
    logs_truncated_bytes: int = 0


class ReferenceAwareGC:
    """Reference-aware garbage collector for workspaces, repositories, and execution records."""

    def __init__(
        self,
        workspace_store: WorkspaceStore | None = None,
        execution_store: LocalExecutionStore | None = None,
        policy: RetentionPolicy | None = None,
    ) -> None:
        self.workspace_store = workspace_store or WorkspaceStore()
        self.home = self.workspace_store.home
        self.execution_store = (
            execution_store or LocalExecutionStore(self.home / "execution")
        )
        self.policy = policy or RetentionPolicy()
        self.tombstone_dir = self.home / "tombstones"
        self.tombstone_dir.mkdir(parents=True, exist_ok=True)

    def save_tombstone(self, tombstone: TombstoneRecord) -> TombstoneRecord:
        path = self.tombstone_dir / f"{tombstone.id}.json"
        atomic_write_json(path, tombstone.to_dict())
        return tombstone

    def get_tombstone(self, record_id: str) -> TombstoneRecord | None:
        path = self.tombstone_dir / f"{record_id}.json"
        if not path.exists():
            return None
        try:
            return TombstoneRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list_tombstones(self) -> list[TombstoneRecord]:
        rows: list[TombstoneRecord] = []
        for path in sorted(self.tombstone_dir.glob("*.json")):
            try:
                rows.append(
                    TombstoneRecord.model_validate_json(path.read_text(encoding="utf-8"))
                )
            except Exception:
                continue
        return rows

    def collect_active_references(self) -> dict[str, set[str]]:
        """Collect all IDs currently referenced by active work."""
        active_tasks = set()
        active_workspaces = set()
        active_repositories = set()
        active_checkpoints = set()
        active_executions = set()

        # 1. Active supervisor tasks
        all_tasks = self.execution_store.list_tasks()
        for task in all_tasks:
            is_active = task.state not in TERMINAL_STATES
            if is_active:
                active_tasks.add(task.task_id)
                if task.workspace_id:
                    active_workspaces.add(task.workspace_id)
                if task.repository_id:
                    active_repositories.add(task.repository_id)
                if task.checkpoint_id:
                    active_checkpoints.add(task.checkpoint_id)
                for child_id in task.child_task_ids:
                    active_tasks.add(child_id)
                for dep_id in task.dependency_task_ids:
                    active_tasks.add(dep_id)
            # Pending recovery candidates or active checkpoints
            if task.state == ExecutionState.WAITING or task.recovery_intervention_id:
                active_tasks.add(task.task_id)
                if task.workspace_id:
                    active_workspaces.add(task.workspace_id)
                if task.repository_id:
                    active_repositories.add(task.repository_id)

        # 2. Checkpoints referenced by active tasks or recent non-terminal states
        for task_id in list(active_tasks):
            for cp in self.execution_store.checkpoints_for_task(task_id):
                active_checkpoints.add(cp.checkpoint_id)
                if cp.workspace_reference:
                    active_workspaces.add(cp.workspace_reference)

        # 3. Active chat sessions
        for session in self.workspace_store.list_sessions():
            if session.status == "active":
                active_workspaces.add(session.workspace_id)
                active_repositories.add(session.primary_repository_id)
                for repo_id in session.attached_repository_ids:
                    active_repositories.add(repo_id)

        # 4. Primary repositories in existing workspaces
        for ws in self.workspace_store.list_workspaces():
            if not ws.implicit:
                # Explicit user-configured workspaces are protected
                active_workspaces.add(ws.workspace_id)
                for repo_id in ws.repository_ids:
                    active_repositories.add(repo_id)

        return {
            "tasks": active_tasks,
            "workspaces": active_workspaces,
            "repositories": active_repositories,
            "checkpoints": active_checkpoints,
            "executions": active_executions,
        }

    def prune_workspaces(
        self, active_refs: dict[str, set[str]], metrics: RetentionMetrics
    ) -> list[str]:
        """Safely prune expired temporary workspaces with zero active references."""
        now = datetime.now(timezone.utc)
        pruned: list[str] = []
        for ws in self.workspace_store.list_workspaces():
            # Never delete explicit/user workspaces or workspaces with active references
            if not ws.implicit:
                continue
            if ws.workspace_id in active_refs["workspaces"]:
                continue

            created = _parse_iso(ws.created_at)
            updated = _parse_iso(ws.updated_at) or created
            age_seconds = (now - (updated or now)).total_seconds()

            if age_seconds >= self.policy.temp_workspace_ttl_seconds:
                # Create tombstone before deleting
                tombstone = TombstoneRecord(
                    id=ws.workspace_id,
                    type="workspace",
                    terminal_status="deleted_expired",
                    created_at=ws.created_at,
                    completed_at=utc_iso(),
                    workspace_reference=ws.workspace_id,
                    repository_reference=ws.primary_repository_id,
                    concise_result_summary=f"Expired temporary workspace '{ws.name}' pruned by GC",
                )
                self.save_tombstone(tombstone)
                self.workspace_store.delete_workspace(ws.workspace_id)
                pruned.append(ws.workspace_id)
                metrics.workspace_records_pruned += 1
                metrics.tombstones_created += 1

        return pruned

    def compact_repositories(
        self, active_refs: dict[str, set[str]], metrics: RetentionMetrics
    ) -> list[str]:
        """Ensure canonical repository records per path and clean orphaned analysis caches."""
        compacted: list[str] = []
        repos = self.workspace_store.list_repositories()
        repos_by_canonical: dict[str, list[RepositoryRecord]] = {}

        for repo in repos:
            canonical = os.path.normcase(str(Path(repo.canonical_path).resolve()))
            repos_by_canonical.setdefault(canonical, []).append(repo)

        # Clean duplicate records for the same canonical path
        for canonical, duplicates in repos_by_canonical.items():
            if len(duplicates) > 1:
                # Keep the primary/most recently updated one
                primary = max(
                    duplicates,
                    key=lambda r: (
                        r.repository_id in active_refs["repositories"],
                        r.updated_at,
                    ),
                )
                for dup in duplicates:
                    if dup.repository_id != primary.repository_id:
                        if dup.repository_id not in active_refs["repositories"]:
                            dup_path = repository_dir(dup.repository_id)
                            if dup_path.exists():
                                shutil.rmtree(dup_path, ignore_errors=True)
                            metrics.repository_records_compacted += 1
                            metrics.stale_records_pruned += 1

        # Clean stale analysis/index caches for deleted repositories
        repo_dirs = list((self.home / "repositories").glob("*"))
        known_repo_ids = {r.repository_id for r in repos}
        for path in repo_dirs:
            if not path.is_dir():
                continue
            repo_id = path.name
            if repo_id not in known_repo_ids and repo_id not in active_refs["repositories"]:
                shutil.rmtree(path, ignore_errors=True)
                metrics.stale_records_pruned += 1

        return compacted

    def compact_completed_tasks(
        self, active_refs: dict[str, set[str]], metrics: RetentionMetrics
    ) -> list[str]:
        """Compact excess completed tasks per workspace into tombstones."""
        compacted: list[str] = []
        all_tasks = self.execution_store.list_tasks()

        # Group completed tasks by workspace
        tasks_by_ws: dict[str, list[TaskRecord]] = {}
        for task in all_tasks:
            if task.state in TERMINAL_STATES and task.task_id not in active_refs["tasks"]:
                ws_id = str(task.workspace_id or "default")
                tasks_by_ws.setdefault(ws_id, []).append(task)

        for ws_id, completed_tasks in tasks_by_ws.items():
            # Sort newest first
            completed_tasks.sort(key=lambda t: t.updated_at, reverse=True)
            excess = completed_tasks[self.policy.max_completed_tasks_per_workspace:]

            for task in excess:
                # Never delete if referenced by an active checkpoint
                if task.checkpoint_id in active_refs["checkpoints"]:
                    continue

                # Create concise tombstone
                error_code = task.failure_reason if task.state != ExecutionState.COMPLETED else None
                summary = (
                    task.normalized_intent
                    or task.requested_operation
                    or f"Task {task.task_id} completed with state {task.state.value}"
                )
                tombstone = TombstoneRecord(
                    id=task.task_id,
                    type="task",
                    terminal_status=task.state.value,
                    created_at=task.created_at.isoformat() if hasattr(task.created_at, "isoformat") else str(task.created_at),
                    completed_at=task.finished_at.isoformat() if task.finished_at and hasattr(task.finished_at, "isoformat") else str(task.finished_at or ""),
                    repository_reference=task.repository_id or None,
                    workspace_reference=task.workspace_id or None,
                    concise_result_summary=summary[:200],
                    terminal_error_code=error_code,
                )
                self.save_tombstone(tombstone)

                # Prune attempt and event files for this task
                task_file = self.execution_store.root / "tasks" / f"{task.task_id}.json"
                if task_file.exists():
                    task_file.unlink()

                for attempt_id in task.attempt_ids:
                    att_file = self.execution_store.root / "attempts" / f"{attempt_id}.json"
                    if att_file.exists():
                        att_file.unlink()

                events_file = self.execution_store.root / "events" / f"{task.task_id}.jsonl"
                if events_file.exists():
                    events_file.unlink()

                compacted.append(task.task_id)
                metrics.tasks_compacted += 1
                metrics.tombstones_created += 1
                metrics.stale_records_pruned += 1

        return compacted

    def prune_diagnostic_logs(self, metrics: RetentionMetrics) -> int:
        """Enforce maximum diagnostic log file size using rolling truncation."""
        logs_dir = self.execution_store.root / "logs"
        if not logs_dir.exists():
            return 0

        truncated_bytes = 0
        for log_file in logs_dir.glob("*.log"):
            try:
                size = log_file.stat().st_size
                if size > self.policy.max_diagnostic_log_bytes:
                    excess = size - self.policy.max_diagnostic_log_bytes
                    # Keep the trailing half of the max allowed bytes
                    keep_bytes = self.policy.max_diagnostic_log_bytes // 2
                    with open(log_file, "rb") as f:
                        f.seek(-keep_bytes, os.SEEK_END)
                        tail = f.read()
                    with open(log_file, "wb") as f:
                        f.write(b"... [truncated by retention policy] ...\n")
                        f.write(tail)
                    truncated_bytes += excess
            except Exception:
                continue

        metrics.logs_truncated_bytes += truncated_bytes
        return truncated_bytes

    def run_retention_pass(self) -> RetentionMetrics:
        """Run a full, incremental and idempotent retention and cleanup pass."""
        metrics = RetentionMetrics()
        active_refs = self.collect_active_references()

        self.prune_workspaces(active_refs, metrics)
        self.compact_repositories(active_refs, metrics)
        self.compact_completed_tasks(active_refs, metrics)
        self.prune_diagnostic_logs(metrics)

        return metrics


__all__ = [
    "ReferenceAwareGC",
    "RetentionClass",
    "RetentionMetrics",
    "RetentionPolicy",
    "TombstoneRecord",
]
