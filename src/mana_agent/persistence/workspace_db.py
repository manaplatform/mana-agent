"""Workspace SQLite database manager with WAL mode, foreign keys, and migration tracking."""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from mana_agent.workspaces.paths import workspace_dir


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA_V1_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL,
        migration_name TEXT NOT NULL,
        details TEXT NOT NULL DEFAULT '{}'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS workspace_migration_state (
        source_type TEXT PRIMARY KEY,
        migrated_at TEXT NOT NULL,
        source_checksum TEXT NOT NULL,
        row_counts TEXT NOT NULL,
        status TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        parent_task_id TEXT,
        root_task_id TEXT NOT NULL,
        title TEXT NOT NULL,
        user_request TEXT NOT NULL,
        normalized_goal TEXT NOT NULL,
        status TEXT NOT NULL,
        priority INTEGER NOT NULL DEFAULT 100,
        risk_level TEXT NOT NULL DEFAULT 'low',
        workspace_id TEXT NOT NULL,
        session_id TEXT NOT NULL DEFAULT '',
        trigger_turn_id TEXT NOT NULL DEFAULT '',
        relation_type TEXT NOT NULL DEFAULT 'independent',
        previous_task_id TEXT NOT NULL DEFAULT '',
        primary_repository_id TEXT NOT NULL DEFAULT '',
        owner_agent_id TEXT,
        supervisor_agent_id TEXT,
        delegated_by_agent_id TEXT,
        accepted_by_agent_id TEXT,
        executed_by_worker_agent_id TEXT,
        reviewed_by_agent_id TEXT,
        approved_by_agent_id TEXT,
        entry_route TEXT NOT NULL DEFAULT '',
        owning_lane TEXT NOT NULL DEFAULT '',
        result_summary TEXT NOT NULL DEFAULT '',
        verification_status TEXT NOT NULL DEFAULT '',
        supervisor_execution_id TEXT NOT NULL DEFAULT '',
        supervisor_state TEXT NOT NULL DEFAULT '',
        supervisor_state_version INTEGER NOT NULL DEFAULT 0,
        aggregate_progress TEXT NOT NULL DEFAULT '',
        wiring_required INTEGER NOT NULL DEFAULT 0,
        wiring_reason TEXT,
        wiring_outcome TEXT NOT NULL DEFAULT 'pending',
        wiring_outcome_reason TEXT NOT NULL DEFAULT '',
        integration_role TEXT NOT NULL DEFAULT '',
        implementation_verified INTEGER NOT NULL DEFAULT 0,
        integration_verified INTEGER NOT NULL DEFAULT 0,
        runtime_reachability_verified INTEGER NOT NULL DEFAULT 0,
        integration_stage TEXT NOT NULL DEFAULT '',
        waiting_kind TEXT NOT NULL DEFAULT '',
        waiting_reason TEXT NOT NULL DEFAULT '',
        wake_up_source TEXT NOT NULL DEFAULT '',
        wake_up_reference TEXT NOT NULL DEFAULT '',
        resume_checkpoint_id TEXT NOT NULL DEFAULT '',
        resume_operation TEXT NOT NULL DEFAULT '',
        decomposition_local_id TEXT NOT NULL DEFAULT '',
        preferred_parallelism TEXT NOT NULL DEFAULT 'automatic',
        managed_workspace_id TEXT NOT NULL DEFAULT '',
        managed_branch TEXT NOT NULL DEFAULT '',
        managed_worktree_path TEXT NOT NULL DEFAULT '',
        workspace_status TEXT NOT NULL DEFAULT '',
        base_revision TEXT NOT NULL DEFAULT '',
        execution_repo_root TEXT NOT NULL DEFAULT '',
        budget_reserved_tokens INTEGER NOT NULL DEFAULT 0,
        budget_used_tokens INTEGER NOT NULL DEFAULT 0,
        budget_remaining_tokens INTEGER NOT NULL DEFAULT 0,
        budget_reserved_ms INTEGER NOT NULL DEFAULT 0,
        budget_used_ms INTEGER NOT NULL DEFAULT 0,
        max_agents INTEGER NOT NULL DEFAULT 8,
        max_subagents INTEGER NOT NULL DEFAULT 4,
        max_queue_jobs INTEGER NOT NULL DEFAULT 32,
        max_tool_calls INTEGER NOT NULL DEFAULT 32,
        task_created_at TEXT,
        scheduled_at TEXT,
        worker_claimed_at TEXT,
        provider_started_at TEXT,
        provider_completed_at TEXT,
        task_completed_at TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        extra_attributes TEXT NOT NULL DEFAULT '{}'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS task_dependencies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        dependency_task_id TEXT NOT NULL,
        dependency_type TEXT NOT NULL DEFAULT 'depends_on',
        UNIQUE(task_id, dependency_task_id, dependency_type)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS task_collections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        collection_type TEXT NOT NULL,
        item_value TEXT NOT NULL,
        sort_order INTEGER NOT NULL DEFAULT 0,
        UNIQUE(task_id, collection_type, item_value, sort_order)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS task_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT,
        event_type TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS task_handoffs (
        handoff_id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        from_agent_id TEXT NOT NULL,
        to_agent_id TEXT NOT NULL,
        reason TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS task_verifications (
        verification_id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        verified_by_agent_id TEXT NOT NULL,
        commands_run TEXT NOT NULL,
        passed INTEGER NOT NULL,
        summary TEXT NOT NULL,
        failures TEXT NOT NULL DEFAULT '[]',
        risks TEXT NOT NULL DEFAULT '[]',
        execution_job_ids TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS task_budgets (
        budget_record_id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        agent_id TEXT NOT NULL DEFAULT '',
        queue_job_id TEXT NOT NULL DEFAULT '',
        budget_reserved_tokens INTEGER NOT NULL DEFAULT 0,
        budget_used_tokens INTEGER NOT NULL DEFAULT 0,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS task_approvals (
        approval_id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        requested_by_agent_id TEXT NOT NULL,
        approved_by_agent_id TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        reason TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        resolved_at TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS task_integration_evidence (
        evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        entrypoint TEXT NOT NULL,
        evidence_path TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        source_references TEXT NOT NULL,
        observable_result TEXT NOT NULL,
        verification_source TEXT NOT NULL DEFAULT '',
        reviewer TEXT NOT NULL DEFAULT '',
        recorded_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS discussions (
        discussion_id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        title TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'open',
        participant_agent_ids TEXT NOT NULL,
        message_ids TEXT NOT NULL,
        created_by_agent_id TEXT NOT NULL DEFAULT '',
        final_decision_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS decisions (
        decision_id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        discussion_id TEXT,
        made_by_agent_id TEXT NOT NULL,
        decision_status TEXT NOT NULL DEFAULT 'proposed',
        summary TEXT NOT NULL,
        rationale_summary TEXT NOT NULL,
        selected_route TEXT NOT NULL,
        assigned_agent_ids TEXT NOT NULL,
        required_verification TEXT NOT NULL,
        risks TEXT NOT NULL DEFAULT '[]',
        assumptions TEXT NOT NULL DEFAULT '[]',
        rejected_options TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        message_id TEXT PRIMARY KEY,
        discussion_id TEXT,
        from_agent_id TEXT NOT NULL,
        to_agent_id TEXT,
        task_id TEXT NOT NULL,
        message_type TEXT NOT NULL,
        content TEXT NOT NULL,
        root_task_id TEXT NOT NULL DEFAULT '',
        evidence_references TEXT NOT NULL DEFAULT '[]',
        confidence REAL NOT NULL DEFAULT 0.0,
        requested_action TEXT NOT NULL DEFAULT '',
        metadata TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS gateway_executions (
        task_id TEXT PRIMARY KEY,
        root_task_id TEXT NOT NULL,
        parent_task_id TEXT,
        owning_lane TEXT NOT NULL,
        state TEXT NOT NULL,
        normalized_intent TEXT NOT NULL,
        repository_id TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        target_files TEXT NOT NULL,
        priority TEXT NOT NULL,
        taskboard_task_id TEXT NOT NULL DEFAULT '',
        worker_id TEXT NOT NULL DEFAULT '',
        model TEXT NOT NULL DEFAULT '',
        provider TEXT NOT NULL DEFAULT '',
        routing_decision_id TEXT NOT NULL DEFAULT '',
        task_type TEXT NOT NULL DEFAULT 'single',
        capabilities TEXT NOT NULL DEFAULT '[]',
        changed_files TEXT NOT NULL DEFAULT '[]',
        verification_state TEXT NOT NULL DEFAULT '{}',
        duplicate_of TEXT,
        last_heartbeat TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        task_created_at TEXT NOT NULL DEFAULT '',
        scheduled_at TEXT NOT NULL DEFAULT '',
        worker_claimed_at TEXT NOT NULL DEFAULT '',
        provider_started_at TEXT NOT NULL DEFAULT '',
        provider_completed_at TEXT NOT NULL DEFAULT '',
        task_completed_at TEXT NOT NULL DEFAULT '',
        duration_breakdown TEXT NOT NULL DEFAULT '{}',
        error TEXT NOT NULL DEFAULT '',
        heartbeat_failure TEXT NOT NULL DEFAULT '',
        progress_summary TEXT NOT NULL DEFAULT '',
        current_tool_activity TEXT NOT NULL DEFAULT '{}',
        evidence TEXT NOT NULL DEFAULT '[]',
        cancellation_state TEXT NOT NULL DEFAULT '{}',
        final_result TEXT NOT NULL DEFAULT '{}',
        supervisor_attempt_id TEXT NOT NULL DEFAULT '',
        checkpoint_id TEXT NOT NULL DEFAULT '',
        trigger_turn_id TEXT NOT NULL DEFAULT '',
        relation_type TEXT NOT NULL DEFAULT 'independent',
        previous_task_id TEXT NOT NULL DEFAULT '',
        user_message_id TEXT NOT NULL DEFAULT '',
        accounting_reservation_ids TEXT NOT NULL DEFAULT '[]',
        lane_history TEXT NOT NULL DEFAULT '[]',
        budget_reserved_input_tokens INTEGER NOT NULL DEFAULT 0,
        budget_reserved_output_tokens INTEGER NOT NULL DEFAULT 0,
        budget_consumed_input_tokens INTEGER NOT NULL DEFAULT 0,
        budget_consumed_output_tokens INTEGER NOT NULL DEFAULT 0,
        budget_estimated_cost REAL NOT NULL DEFAULT 0.0,
        budget_actual_cost REAL NOT NULL DEFAULT 0.0,
        budget_estimated_cost_known INTEGER NOT NULL DEFAULT 0,
        budget_actual_cost_known INTEGER NOT NULL DEFAULT 0,
        budget_model_context_window INTEGER NOT NULL DEFAULT 0,
        budget_model_max_output_tokens INTEGER NOT NULL DEFAULT 0,
        budget_estimate_confidence TEXT NOT NULL DEFAULT '',
        budget_estimate_source TEXT NOT NULL DEFAULT '',
        budget_turn_budget_tokens INTEGER NOT NULL DEFAULT 0,
        budget_turn_consumed_tokens INTEGER NOT NULL DEFAULT 0,
        budget_turn_reserved_tokens INTEGER NOT NULL DEFAULT 0,
        budget_revisions TEXT NOT NULL DEFAULT '[]'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS gateway_handoffs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        source_lane TEXT NOT NULL,
        target_lane TEXT NOT NULL,
        reason TEXT NOT NULL,
        artifacts TEXT NOT NULL DEFAULT '[]',
        changed_files TEXT NOT NULL DEFAULT '[]',
        remaining_work TEXT NOT NULL DEFAULT '[]',
        verification_state TEXT NOT NULL DEFAULT '{}',
        budget_consumed TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS gateway_locks (
        lease_id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        mode TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        repository_id TEXT NOT NULL,
        paths TEXT NOT NULL,
        owner_pid INTEGER NOT NULL,
        acquired_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS gateway_waiters (
        sequence INTEGER PRIMARY KEY,
        task_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        lane_id TEXT NOT NULL,
        priority TEXT NOT NULL,
        normalized_intent TEXT NOT NULL,
        waiter_data TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS gateway_turns (
        turn_id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        user_message_id TEXT NOT NULL,
        message_fingerprint TEXT NOT NULL,
        received_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'received',
        normalized_intent TEXT NOT NULL DEFAULT '',
        routing_decision_id TEXT NOT NULL DEFAULT '',
        related_task_ids TEXT NOT NULL DEFAULT '[]',
        created_task_ids TEXT NOT NULL DEFAULT '[]',
        response_message_id TEXT NOT NULL DEFAULT '',
        response_execution_id TEXT NOT NULL DEFAULT '',
        response TEXT NOT NULL DEFAULT '{}',
        workspace_id TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(conversation_id, user_message_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS checkpoints (
        checkpoint_id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        boundary TEXT NOT NULL,
        completed_steps TEXT NOT NULL DEFAULT '[]',
        pending_steps TEXT NOT NULL DEFAULT '[]',
        workspace_reference TEXT NOT NULL DEFAULT '',
        state_snapshot TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS escrow_results (
        result_id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        execution_id TEXT NOT NULL,
        trigger_turn_id TEXT NOT NULL DEFAULT '',
        session_id TEXT NOT NULL DEFAULT '',
        parent_task_id TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'verified',
        result_data TEXT NOT NULL DEFAULT '{}',
        acknowledged INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    );
    """,
    # Indexes
    "CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_task_id);",
    "CREATE INDEX IF NOT EXISTS idx_tasks_root ON tasks(root_task_id);",
    "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);",
    "CREATE INDEX IF NOT EXISTS idx_tasks_workspace ON tasks(workspace_id);",
    "CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_tasks_turn ON tasks(trigger_turn_id);",
    "CREATE INDEX IF NOT EXISTS idx_tasks_supervisor_exec ON tasks(supervisor_execution_id);",
    "CREATE INDEX IF NOT EXISTS idx_task_events_task ON task_events(task_id);",
    "CREATE INDEX IF NOT EXISTS idx_task_collections_lookup ON task_collections(task_id, collection_type);",
    "CREATE INDEX IF NOT EXISTS idx_task_deps_task ON task_dependencies(task_id);",
    "CREATE INDEX IF NOT EXISTS idx_discussions_task ON discussions(task_id);",
    "CREATE INDEX IF NOT EXISTS idx_decisions_task ON decisions(task_id);",
    "CREATE INDEX IF NOT EXISTS idx_messages_task ON messages(task_id);",
    "CREATE INDEX IF NOT EXISTS idx_messages_discussion ON messages(discussion_id);",
    "CREATE INDEX IF NOT EXISTS idx_gateway_executions_state ON gateway_executions(state);",
    "CREATE INDEX IF NOT EXISTS idx_gateway_executions_session ON gateway_executions(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_gateway_executions_taskboard ON gateway_executions(taskboard_task_id);",
    "CREATE INDEX IF NOT EXISTS idx_gateway_turns_conv_msg ON gateway_turns(conversation_id, user_message_id);",
    "CREATE INDEX IF NOT EXISTS idx_checkpoints_task ON checkpoints(task_id);",
    "CREATE INDEX IF NOT EXISTS idx_escrow_results_task ON escrow_results(task_id);",
    "CREATE INDEX IF NOT EXISTS idx_escrow_results_exec ON escrow_results(execution_id);",
    "CREATE INDEX IF NOT EXISTS idx_escrow_results_turn ON escrow_results(trigger_turn_id);",
    "CREATE INDEX IF NOT EXISTS idx_escrow_results_session ON escrow_results(session_id);",
]


class WorkspaceDatabase:
    """Thread-safe SQLite database manager for workspace runtime state."""

    def __init__(self, workspace_id: str, db_path: Path | str | None = None) -> None:
        self.workspace_id = str(workspace_id)
        if db_path is None:
            self.db_path = workspace_dir(self.workspace_id) / "state.db"
        elif str(db_path) == ":memory:":
            self.db_path = Path(":memory:")
        else:
            self.db_path = Path(db_path).resolve()

        self._init_db()

    def _create_connection(self) -> sqlite3.Connection:
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        else:
            conn = sqlite3.connect(":memory:", timeout=30.0, check_same_thread=False)

        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 30000;")
        if str(self.db_path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        return conn

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = self._create_connection()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._create_connection()
        conn.execute("BEGIN IMMEDIATE;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.transaction() as conn:
            for stmt in SCHEMA_V1_STATEMENTS:
                conn.execute(stmt)

            row = conn.execute("SELECT version FROM schema_migrations WHERE version = 1;").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at, migration_name, details) VALUES (?, ?, ?, ?);",
                    (1, _utc_now_iso(), "initial_normalized_workspace_schema", "{}"),
                )


_DATABASES: dict[str, WorkspaceDatabase] = {}
_DATABASES_LOCK = threading.RLock()


def get_workspace_db(workspace_id: str, db_path: Path | str | None = None) -> WorkspaceDatabase:
    key = f"{workspace_id}:{db_path}"
    with _DATABASES_LOCK:
        db = _DATABASES.get(key)
        if db is None:
            db = WorkspaceDatabase(workspace_id, db_path=db_path)
            _DATABASES[key] = db
        return db
