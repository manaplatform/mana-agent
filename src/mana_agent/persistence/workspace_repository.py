"""Targeted workspace repository backed by SQLite."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mana_agent.gateway.chat_turn_store import ChatTurnRecord
from mana_agent.gateway.lanes import (
    ACTIVE_LANE_STATES,
    LaneId,
    LanePriority,
    LaneTaskState,
    LockMode,
)
from mana_agent.gateway.lane_models import (
    LaneBudget,
    LaneExecution,
    LaneHandoff,
    LockLease,
)
from mana_agent.execution_supervisor.models import CheckpointRecord, EscrowResult
from mana_agent.multi_agent.core.types import (
    AgentMessage,
    DecisionRecord,
    DecisionStatus,
    DiscussionStatus,
    DiscussionThread,
    HandoffRecord,
    MessageType,
    RiskLevel,
    TaskBoardItem,
    TaskStatus,
    VerificationResult,
    parse_dt,
    to_jsonable,
    utc_now,
)
from mana_agent.persistence.workspace_db import WorkspaceDatabase, get_workspace_db
from mana_agent.utils.tool_results import json_safe_tool_payload


def _dt_to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _iso_to_dt(iso_str: str | None) -> datetime | None:
    if not iso_str:
        return None
    return parse_dt(iso_str)


COLLECTION_FIELDS = (
    "repository_ids",
    "assigned_agent_ids",
    "assigned_subagent_ids",
    "required_capabilities",
    "allowed_tools",
    "forbidden_tools",
    "files_to_inspect",
    "files_touched",
    "queue_job_ids",
    "verification_queue_job_ids",
    "acceptance_criteria",
    "plan",
    "depends_on",
    "output_artifacts",
    "approval_request_ids",
    "child_task_ids",
    "evidence",
    "blockers",
    "assumptions",
    "implementation_targets",
    "wiring_targets",
    "registration_points",
    "runtime_entrypoints",
    "configuration_targets",
    "export_targets",
    "integration_verification",
    "required_wiring_task_ids",
    "integration_evidence",
    "discussion_ids",
    "decision_ids",
    "verification_commands",
)


class WorkspaceRepository:
    """Targeted repository providing isolated, granular read/write operations against workspace SQLite database."""

    def __init__(self, workspace_id: str, db: WorkspaceDatabase | None = None, db_path: Path | str | None = None) -> None:
        self.workspace_id = str(workspace_id)
        self.db = db or get_workspace_db(self.workspace_id, db_path=db_path)

    # -------------------------------------------------------------------------
    # TaskBoard / Tasks
    # -------------------------------------------------------------------------

    def save_task(self, task: TaskBoardItem) -> TaskBoardItem:
        extra = {
            "routing_evidence": task.routing_evidence,
            "supervisor_verification_evidence": task.supervisor_verification_evidence,
            "decomposition_id_map": task.decomposition_id_map,
            "reachability_edges": task.reachability_edges,
            "verification_provenance": task.verification_provenance,
            "hierarchy_violations": task.hierarchy_violations,
            "actual_tool_events": task.actual_tool_events,
            "cost_by_agent_id": task.cost_by_agent_id,
            "cost_by_queue_job_id": task.cost_by_queue_job_id,
            "memory_status": task.memory_status,
            "duration_breakdown": task.duration_breakdown,
        }
        extra_json = json.dumps(json_safe_tool_payload(extra), sort_keys=True, ensure_ascii=False)

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tasks (
                    task_id, parent_task_id, root_task_id, title, user_request, normalized_goal,
                    status, priority, risk_level, workspace_id, session_id, trigger_turn_id,
                    relation_type, previous_task_id, primary_repository_id, owner_agent_id,
                    supervisor_agent_id, delegated_by_agent_id, accepted_by_agent_id,
                    executed_by_worker_agent_id, reviewed_by_agent_id, approved_by_agent_id,
                    entry_route, owning_lane, result_summary, verification_status,
                    supervisor_execution_id, supervisor_state, supervisor_state_version,
                    aggregate_progress, wiring_required, wiring_reason, wiring_outcome,
                    wiring_outcome_reason, integration_role, implementation_verified,
                    integration_verified, runtime_reachability_verified, integration_stage,
                    waiting_kind, waiting_reason, wake_up_source, wake_up_reference,
                    resume_checkpoint_id, resume_operation, decomposition_local_id,
                    preferred_parallelism, managed_workspace_id, managed_branch,
                    managed_worktree_path, workspace_status, base_revision, execution_repo_root,
                    budget_reserved_tokens, budget_used_tokens, budget_remaining_tokens,
                    budget_reserved_ms, budget_used_ms, max_agents, max_subagents,
                    max_queue_jobs, max_tool_calls, task_created_at, scheduled_at,
                    worker_claimed_at, provider_started_at, provider_completed_at,
                    task_completed_at, created_at, updated_at, extra_attributes
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?
                );
                """,
                (
                    task.task_id,
                    task.parent_task_id,
                    task.root_task_id,
                    task.title,
                    task.user_request,
                    task.normalized_goal,
                    task.status.value if isinstance(task.status, TaskStatus) else str(task.status),
                    int(task.priority),
                    task.risk_level.value if isinstance(task.risk_level, RiskLevel) else str(task.risk_level),
                    task.workspace_id or self.workspace_id,
                    task.session_id,
                    task.trigger_turn_id,
                    task.relation_type,
                    task.previous_task_id,
                    task.primary_repository_id,
                    task.owner_agent_id,
                    task.supervisor_agent_id,
                    task.delegated_by_agent_id,
                    task.accepted_by_agent_id,
                    task.executed_by_worker_agent_id,
                    task.reviewed_by_agent_id,
                    task.approved_by_agent_id,
                    task.entry_route,
                    task.owning_lane,
                    task.result_summary,
                    task.verification_status,
                    task.supervisor_execution_id,
                    task.supervisor_state,
                    int(task.supervisor_state_version),
                    task.aggregate_progress,
                    1 if task.wiring_required else 0,
                    task.wiring_reason,
                    task.wiring_outcome,
                    task.wiring_outcome_reason,
                    task.integration_role,
                    1 if task.implementation_verified else 0,
                    1 if task.integration_verified else 0,
                    1 if task.runtime_reachability_verified else 0,
                    task.integration_stage,
                    task.waiting_kind,
                    task.waiting_reason,
                    task.wake_up_source,
                    task.wake_up_reference,
                    task.resume_checkpoint_id,
                    task.resume_operation,
                    task.decomposition_local_id,
                    task.preferred_parallelism,
                    task.managed_workspace_id,
                    task.managed_branch,
                    task.managed_worktree_path,
                    task.workspace_status,
                    task.base_revision,
                    task.execution_repo_root,
                    int(task.budget_reserved_tokens),
                    int(task.budget_used_tokens),
                    int(task.budget_remaining_tokens),
                    int(task.budget_reserved_ms),
                    int(task.budget_used_ms),
                    int(task.max_agents),
                    int(task.max_subagents),
                    int(task.max_queue_jobs),
                    int(task.max_tool_calls),
                    _dt_to_iso(task.task_created_at),
                    _dt_to_iso(task.scheduled_at),
                    _dt_to_iso(task.worker_claimed_at),
                    _dt_to_iso(task.provider_started_at),
                    _dt_to_iso(task.provider_completed_at),
                    _dt_to_iso(task.task_completed_at),
                    _dt_to_iso(task.created_at) or _dt_to_iso(utc_now()),
                    _dt_to_iso(task.updated_at) or _dt_to_iso(utc_now()),
                    extra_json,
                ),
            )

            # Collections: delete old and insert fresh
            conn.execute("DELETE FROM task_collections WHERE task_id = ?;", (task.task_id,))
            for col in COLLECTION_FIELDS:
                items = getattr(task, col, [])
                if items:
                    for idx, val in enumerate(items):
                        val_str = str(val).strip()
                        if val_str:
                            conn.execute(
                                "INSERT INTO task_collections (task_id, collection_type, item_value, sort_order) VALUES (?, ?, ?, ?);",
                                (task.task_id, col, val_str, idx),
                            )

            # Dependencies
            conn.execute("DELETE FROM task_dependencies WHERE task_id = ?;", (task.task_id,))
            for dep in task.depends_on:
                dep_str = str(dep).strip()
                if dep_str:
                    conn.execute(
                        "INSERT OR IGNORE INTO task_dependencies (task_id, dependency_task_id, dependency_type) VALUES (?, ?, 'depends_on');",
                        (task.task_id, dep_str),
                    )
            for wiring_dep in task.required_wiring_task_ids:
                w_str = str(wiring_dep).strip()
                if w_str:
                    conn.execute(
                        "INSERT OR IGNORE INTO task_dependencies (task_id, dependency_task_id, dependency_type) VALUES (?, ?, 'required_wiring');",
                        (task.task_id, w_str),
                    )

            # Handoffs
            conn.execute("DELETE FROM task_handoffs WHERE task_id = ?;", (task.task_id,))
            for handoff in task.handoff_records:
                conn.execute(
                    "INSERT INTO task_handoffs (task_id, from_agent_id, to_agent_id, reason, created_at) VALUES (?, ?, ?, ?, ?);",
                    (task.task_id, handoff.from_agent_id, handoff.to_agent_id, handoff.reason, _dt_to_iso(handoff.created_at)),
                )

            # Verifications
            conn.execute("DELETE FROM task_verifications WHERE task_id = ?;", (task.task_id,))
            for v in task.verification_results:
                conn.execute(
                    """
                    INSERT INTO task_verifications (
                        verification_id, task_id, verified_by_agent_id, commands_run,
                        passed, summary, failures, risks, execution_job_ids, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        v.verification_id,
                        task.task_id,
                        v.verified_by_agent_id,
                        json.dumps(v.commands_run, ensure_ascii=False),
                        1 if v.passed else 0,
                        v.summary,
                        json.dumps(v.failures, ensure_ascii=False),
                        json.dumps(v.risks, ensure_ascii=False),
                        json.dumps(v.execution_job_ids, ensure_ascii=False),
                        _dt_to_iso(v.created_at),
                    ),
                )

            # Budgets
            conn.execute("DELETE FROM task_budgets WHERE task_id = ?;", (task.task_id,))
            for b in task.budget_records:
                agent_id = str(b.get("agent_id") or b.get("requested_by_agent_id") or "")
                queue_job_id = str(b.get("queue_job_id") or "")
                reserved = int(b.get("budget_reserved_tokens") or b.get("budget_reserved") or 0)
                used = int(b.get("budget_used_tokens") or b.get("budget_used") or 0)
                conn.execute(
                    "INSERT INTO task_budgets (task_id, agent_id, queue_job_id, budget_reserved_tokens, budget_used_tokens, payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                    (task.task_id, agent_id, queue_job_id, reserved, used, json.dumps(json_safe_tool_payload(b), ensure_ascii=False), _dt_to_iso(utc_now())),
                )

            # Integration evidence records
            conn.execute("DELETE FROM task_integration_evidence WHERE task_id = ?;", (task.task_id,))
            for ev in task.integration_evidence_records:
                conn.execute(
                    """
                    INSERT INTO task_integration_evidence (
                        task_id, entrypoint, evidence_path, summary, source_references,
                        observable_result, verification_source, reviewer, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        task.task_id,
                        str(ev.get("entrypoint") or ""),
                        json.dumps(ev.get("evidence_path") or ev.get("path") or [], ensure_ascii=False),
                        str(ev.get("summary") or ""),
                        json.dumps(ev.get("source_references") or [], ensure_ascii=False),
                        str(ev.get("observable_result") or ""),
                        str(ev.get("verification_source") or ""),
                        str(ev.get("reviewer") or ""),
                        str(ev.get("recorded_at") or _dt_to_iso(utc_now())),
                    ),
                )

        return task

    def _hydrate_task(self, row: sqlite3.Row, conn: sqlite3.Connection) -> TaskBoardItem:
        task_id = str(row["task_id"])
        # Fetch collections
        col_rows = conn.execute(
            "SELECT collection_type, item_value FROM task_collections WHERE task_id = ? ORDER BY sort_order ASC;",
            (task_id,),
        ).fetchall()
        collections: dict[str, list[str]] = {col: [] for col in COLLECTION_FIELDS}
        for c_row in col_rows:
            c_type = str(c_row["collection_type"])
            if c_type in collections:
                collections[c_type].append(str(c_row["item_value"]))

        # Fetch handoffs
        h_rows = conn.execute(
            "SELECT from_agent_id, to_agent_id, task_id, reason, created_at FROM task_handoffs WHERE task_id = ? ORDER BY handoff_id ASC;",
            (task_id,),
        ).fetchall()
        handoffs = [
            HandoffRecord(
                from_agent_id=str(h["from_agent_id"]),
                to_agent_id=str(h["to_agent_id"]),
                task_id=str(h["task_id"]),
                reason=str(h["reason"]),
                created_at=parse_dt(h["created_at"]),
            )
            for h in h_rows
        ]

        # Fetch verifications
        v_rows = conn.execute(
            "SELECT verification_id, task_id, verified_by_agent_id, commands_run, passed, summary, failures, risks, execution_job_ids, created_at FROM task_verifications WHERE task_id = ? ORDER BY verification_id ASC;",
            (task_id,),
        ).fetchall()
        verifications = []
        for v in v_rows:
            try:
                commands_run = json.loads(v["commands_run"])
            except Exception:
                commands_run = []
            try:
                failures = json.loads(v["failures"])
            except Exception:
                failures = []
            try:
                risks = json.loads(v["risks"])
            except Exception:
                risks = []
            try:
                execution_job_ids = json.loads(v["execution_job_ids"])
            except Exception:
                execution_job_ids = []
            verifications.append(
                VerificationResult(
                    verification_id=str(v["verification_id"]),
                    task_id=str(v["task_id"]),
                    verified_by_agent_id=str(v["verified_by_agent_id"]),
                    commands_run=commands_run,
                    passed=bool(v["passed"]),
                    summary=str(v["summary"]),
                    failures=failures,
                    risks=risks,
                    execution_job_ids=execution_job_ids,
                    created_at=parse_dt(v["created_at"]),
                )
            )

        # Fetch budgets
        b_rows = conn.execute(
            "SELECT payload FROM task_budgets WHERE task_id = ? ORDER BY budget_record_id ASC;",
            (task_id,),
        ).fetchall()
        budget_records = []
        for b in b_rows:
            try:
                budget_records.append(json.loads(b["payload"]))
            except Exception:
                pass

        # Fetch integration evidence
        ie_rows = conn.execute(
            "SELECT entrypoint, evidence_path, summary, source_references, observable_result, verification_source, reviewer, recorded_at FROM task_integration_evidence WHERE task_id = ? ORDER BY evidence_id ASC;",
            (task_id,),
        ).fetchall()
        integration_evidence_records = []
        for ie in ie_rows:
            try:
                ev_path = json.loads(ie["evidence_path"])
            except Exception:
                ev_path = []
            try:
                src_refs = json.loads(ie["source_references"])
            except Exception:
                src_refs = []
            integration_evidence_records.append({
                "entrypoint": str(ie["entrypoint"]),
                "evidence_path": ev_path,
                "path": ev_path,
                "summary": str(ie["summary"]),
                "source_references": src_refs,
                "observable_result": str(ie["observable_result"]),
                "verification_source": str(ie["verification_source"]),
                "reviewer": str(ie["reviewer"]),
                "recorded_at": str(ie["recorded_at"]),
            })

        extra_json = str(row["extra_attributes"] or "{}")
        try:
            extra = json.loads(extra_json)
        except Exception:
            extra = {}

        return TaskBoardItem(
            task_id=task_id,
            parent_task_id=str(row["parent_task_id"]) if row["parent_task_id"] else None,
            root_task_id=str(row["root_task_id"]),
            title=str(row["title"]),
            user_request=str(row["user_request"]),
            normalized_goal=str(row["normalized_goal"]),
            status=TaskStatus(row["status"]),
            priority=int(row["priority"]),
            risk_level=RiskLevel(row["risk_level"]),
            workspace_id=str(row["workspace_id"]),
            session_id=str(row["session_id"]),
            trigger_turn_id=str(row["trigger_turn_id"]),
            relation_type=str(row["relation_type"]),
            previous_task_id=str(row["previous_task_id"]),
            primary_repository_id=str(row["primary_repository_id"]),
            repository_ids=collections["repository_ids"],
            managed_workspace_id=str(row["managed_workspace_id"]),
            managed_branch=str(row["managed_branch"]),
            managed_worktree_path=str(row["managed_worktree_path"]),
            workspace_status=str(row["workspace_status"]),
            base_revision=str(row["base_revision"]),
            execution_repo_root=str(row["execution_repo_root"]),
            owner_agent_id=str(row["owner_agent_id"]) if row["owner_agent_id"] else None,
            supervisor_agent_id=str(row["supervisor_agent_id"]) if row["supervisor_agent_id"] else None,
            delegated_by_agent_id=str(row["delegated_by_agent_id"]) if row["delegated_by_agent_id"] else None,
            accepted_by_agent_id=str(row["accepted_by_agent_id"]) if row["accepted_by_agent_id"] else None,
            executed_by_worker_agent_id=str(row["executed_by_worker_agent_id"]) if row["executed_by_worker_agent_id"] else None,
            reviewed_by_agent_id=str(row["reviewed_by_agent_id"]) if row["reviewed_by_agent_id"] else None,
            approved_by_agent_id=str(row["approved_by_agent_id"]) if row["approved_by_agent_id"] else None,
            assigned_agent_ids=collections["assigned_agent_ids"],
            assigned_subagent_ids=collections["assigned_subagent_ids"],
            required_capabilities=collections["required_capabilities"],
            allowed_tools=collections["allowed_tools"],
            forbidden_tools=collections["forbidden_tools"],
            files_to_inspect=collections["files_to_inspect"],
            files_touched=collections["files_touched"],
            queue_job_ids=collections["queue_job_ids"],
            verification_queue_job_ids=collections["verification_queue_job_ids"],
            acceptance_criteria=collections["acceptance_criteria"],
            plan=collections["plan"],
            depends_on=collections["depends_on"],
            decomposition_local_id=str(row["decomposition_local_id"]),
            preferred_parallelism=str(row["preferred_parallelism"]),
            entry_route=str(row["entry_route"]),
            owning_lane=str(row["owning_lane"]),
            routing_evidence=extra.get("routing_evidence", {}),
            result_summary=str(row["result_summary"]),
            verification_status=str(row["verification_status"]),
            supervisor_execution_id=str(row["supervisor_execution_id"]),
            supervisor_state=str(row["supervisor_state"]),
            supervisor_state_version=int(row["supervisor_state_version"]),
            supervisor_verification_evidence=extra.get("supervisor_verification_evidence", {}),
            output_artifacts=collections["output_artifacts"],
            approval_request_ids=collections["approval_request_ids"],
            child_task_ids=collections["child_task_ids"],
            decomposition_id_map=extra.get("decomposition_id_map", {}),
            aggregate_progress=str(row["aggregate_progress"]),
            evidence=collections["evidence"],
            blockers=collections["blockers"],
            assumptions=collections["assumptions"],
            implementation_targets=collections["implementation_targets"],
            wiring_targets=collections["wiring_targets"],
            registration_points=collections["registration_points"],
            runtime_entrypoints=collections["runtime_entrypoints"],
            configuration_targets=collections["configuration_targets"],
            export_targets=collections["export_targets"],
            integration_verification=collections["integration_verification"],
            wiring_required=bool(row["wiring_required"]),
            wiring_reason=str(row["wiring_reason"]) if row["wiring_reason"] is not None else None,
            wiring_outcome=str(row["wiring_outcome"]),
            wiring_outcome_reason=str(row["wiring_outcome_reason"]),
            reachability_edges=extra.get("reachability_edges", []),
            verification_provenance=extra.get("verification_provenance", {}),
            integration_role=str(row["integration_role"]),
            required_wiring_task_ids=collections["required_wiring_task_ids"],
            implementation_verified=bool(row["implementation_verified"]),
            integration_verified=bool(row["integration_verified"]),
            runtime_reachability_verified=bool(row["runtime_reachability_verified"]),
            integration_evidence=collections["integration_evidence"],
            integration_evidence_records=integration_evidence_records,
            integration_stage=str(row["integration_stage"]),
            waiting_kind=str(row["waiting_kind"]),
            waiting_reason=str(row["waiting_reason"]),
            wake_up_source=str(row["wake_up_source"]),
            wake_up_reference=str(row["wake_up_reference"]),
            resume_checkpoint_id=str(row["resume_checkpoint_id"]),
            resume_operation=str(row["resume_operation"]),
            discussion_ids=collections["discussion_ids"],
            decision_ids=collections["decision_ids"],
            handoff_records=handoffs,
            budget_records=budget_records,
            hierarchy_violations=extra.get("hierarchy_violations", []),
            actual_tool_events=extra.get("actual_tool_events", []),
            budget_reserved_tokens=int(row["budget_reserved_tokens"]),
            budget_used_tokens=int(row["budget_used_tokens"]),
            budget_remaining_tokens=int(row["budget_remaining_tokens"]),
            budget_reserved_ms=int(row["budget_reserved_ms"]),
            budget_used_ms=int(row["budget_used_ms"]),
            max_agents=int(row["max_agents"]),
            max_subagents=int(row["max_subagents"]),
            max_queue_jobs=int(row["max_queue_jobs"]),
            max_tool_calls=int(row["max_tool_calls"]),
            cost_by_agent_id=extra.get("cost_by_agent_id", {}),
            cost_by_queue_job_id=extra.get("cost_by_queue_job_id", {}),
            verification_commands=collections["verification_commands"],
            verification_results=verifications,
            memory_status=extra.get("memory_status", {}),
            task_created_at=_iso_to_dt(row["task_created_at"]),
            scheduled_at=_iso_to_dt(row["scheduled_at"]),
            worker_claimed_at=_iso_to_dt(row["worker_claimed_at"]),
            provider_started_at=_iso_to_dt(row["provider_started_at"]),
            provider_completed_at=_iso_to_dt(row["provider_completed_at"]),
            task_completed_at=_iso_to_dt(row["task_completed_at"]),
            duration_breakdown=extra.get("duration_breakdown", {}),
            created_at=parse_dt(row["created_at"]),
            updated_at=parse_dt(row["updated_at"]),
        )

    def get_task(self, task_id: str) -> TaskBoardItem:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?;", (task_id,)).fetchone()
            if row is None:
                raise KeyError(f"Task not found: {task_id}")
            return self._hydrate_task(row, conn)

    def get_task_or_none(self, task_id: str) -> TaskBoardItem | None:
        try:
            return self.get_task(task_id)
        except KeyError:
            return None

    def task_exists(self, task_id: str) -> bool:
        with self.db.connect() as conn:
            row = conn.execute("SELECT 1 FROM tasks WHERE task_id = ?;", (task_id,)).fetchone()
            return row is not None

    def list_tasks(self, status: TaskStatus | None = None, session_id: str | None = None) -> list[TaskBoardItem]:
        query = "SELECT * FROM tasks WHERE 1=1"
        params: list[Any] = []
        if status is not None:
            query += " AND status = ?"
            params.append(status.value if isinstance(status, TaskStatus) else str(status))
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(str(session_id))
        query += " ORDER BY created_at ASC;"

        with self.db.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._hydrate_task(row, conn) for row in rows]

    def count_tasks(self) -> int:
        with self.db.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM tasks;").fetchone()
            return int(row["total"]) if row else 0

    def delete_task(self, task_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM task_collections WHERE task_id = ?;", (task_id,))
            conn.execute("DELETE FROM task_dependencies WHERE task_id = ?;", (task_id,))
            conn.execute("DELETE FROM task_handoffs WHERE task_id = ?;", (task_id,))
            conn.execute("DELETE FROM task_verifications WHERE task_id = ?;", (task_id,))
            conn.execute("DELETE FROM task_budgets WHERE task_id = ?;", (task_id,))
            conn.execute("DELETE FROM task_approvals WHERE task_id = ?;", (task_id,))
            conn.execute("DELETE FROM task_integration_evidence WHERE task_id = ?;", (task_id,))
            conn.execute("DELETE FROM task_events WHERE task_id = ?;", (task_id,))
            conn.execute("DELETE FROM tasks WHERE task_id = ?;", (task_id,))

    # -------------------------------------------------------------------------
    # Task Events
    # -------------------------------------------------------------------------

    def append_task_event(self, task_id: str | None, event_type: str, payload: Any, created_at: datetime | None = None) -> None:
        dt = created_at or utc_now()
        payload_json = json.dumps(json_safe_tool_payload(payload), sort_keys=True, ensure_ascii=False)
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO task_events (task_id, event_type, payload, created_at) VALUES (?, ?, ?, ?);",
                (task_id, str(event_type), payload_json, _dt_to_iso(dt)),
            )

    def list_task_events(self, task_id: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        query = "SELECT event_type, payload, created_at FROM task_events"
        params: list[Any] = []
        if task_id is not None:
            query += " WHERE task_id = ?"
            params.append(str(task_id))
        query += " ORDER BY event_id ASC LIMIT ?;"
        params.append(max(1, int(limit)))

        with self.db.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for r in rows:
                try:
                    payload = json.loads(r["payload"])
                except Exception:
                    payload = {}
                results.append({
                    "event_type": str(r["event_type"]),
                    "payload": payload,
                    "created_at": str(r["created_at"]),
                })
            return results

    # -------------------------------------------------------------------------
    # Discussions
    # -------------------------------------------------------------------------

    def save_discussion(self, discussion: DiscussionThread) -> DiscussionThread:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO discussions (
                    discussion_id, task_id, title, status, participant_agent_ids,
                    message_ids, created_by_agent_id, final_decision_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    discussion.discussion_id,
                    discussion.task_id,
                    discussion.title,
                    discussion.status.value if isinstance(discussion.status, DiscussionStatus) else str(discussion.status),
                    json.dumps(discussion.participant_agent_ids, ensure_ascii=False),
                    json.dumps(discussion.message_ids, ensure_ascii=False),
                    discussion.created_by_agent_id,
                    discussion.final_decision_id,
                    _dt_to_iso(discussion.created_at),
                    _dt_to_iso(discussion.updated_at),
                ),
            )
        return discussion

    def get_discussion(self, discussion_id: str) -> DiscussionThread | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM discussions WHERE discussion_id = ?;", (discussion_id,)).fetchone()
            if row is None:
                return None
            try:
                participants = json.loads(row["participant_agent_ids"])
            except Exception:
                participants = []
            try:
                message_ids = json.loads(row["message_ids"])
            except Exception:
                message_ids = []
            return DiscussionThread(
                discussion_id=str(row["discussion_id"]),
                task_id=str(row["task_id"]),
                title=str(row["title"]),
                status=DiscussionStatus(row["status"]),
                participant_agent_ids=participants,
                message_ids=message_ids,
                created_by_agent_id=str(row["created_by_agent_id"] or ""),
                final_decision_id=str(row["final_decision_id"]) if row["final_decision_id"] else None,
                created_at=parse_dt(row["created_at"]),
                updated_at=parse_dt(row["updated_at"]),
            )

    def list_discussions(self, task_id: str | None = None) -> list[DiscussionThread]:
        query = "SELECT * FROM discussions"
        params: list[Any] = []
        if task_id is not None:
            query += " WHERE task_id = ?"
            params.append(str(task_id))
        query += " ORDER BY created_at ASC;"

        with self.db.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                try:
                    participants = json.loads(row["participant_agent_ids"])
                except Exception:
                    participants = []
                try:
                    message_ids = json.loads(row["message_ids"])
                except Exception:
                    message_ids = []
                results.append(
                    DiscussionThread(
                        discussion_id=str(row["discussion_id"]),
                        task_id=str(row["task_id"]),
                        title=str(row["title"]),
                        status=DiscussionStatus(row["status"]),
                        participant_agent_ids=participants,
                        message_ids=message_ids,
                        created_by_agent_id=str(row["created_by_agent_id"] or ""),
                        final_decision_id=str(row["final_decision_id"]) if row["final_decision_id"] else None,
                        created_at=parse_dt(row["created_at"]),
                        updated_at=parse_dt(row["updated_at"]),
                    )
                )
            return results

    # -------------------------------------------------------------------------
    # Decisions
    # -------------------------------------------------------------------------

    def save_decision(self, decision: DecisionRecord) -> DecisionRecord:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO decisions (
                    decision_id, task_id, discussion_id, made_by_agent_id,
                    decision_status, summary, rationale_summary, selected_route,
                    assigned_agent_ids, required_verification, risks, assumptions,
                    rejected_options, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    decision.decision_id,
                    decision.task_id,
                    decision.discussion_id,
                    decision.made_by_agent_id,
                    decision.decision_status.value if isinstance(decision.decision_status, DecisionStatus) else str(decision.decision_status),
                    decision.summary,
                    decision.rationale_summary,
                    decision.selected_route,
                    json.dumps(decision.assigned_agent_ids, ensure_ascii=False),
                    json.dumps(decision.required_verification, ensure_ascii=False),
                    json.dumps(decision.risks, ensure_ascii=False),
                    json.dumps(decision.assumptions, ensure_ascii=False),
                    json.dumps(decision.rejected_options, ensure_ascii=False),
                    _dt_to_iso(decision.created_at),
                ),
            )
        return decision

    def get_decision(self, decision_id: str) -> DecisionRecord | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM decisions WHERE decision_id = ?;", (decision_id,)).fetchone()
            if row is None:
                return None
            try:
                assigned = json.loads(row["assigned_agent_ids"])
            except Exception:
                assigned = []
            try:
                req_ver = json.loads(row["required_verification"])
            except Exception:
                req_ver = []
            try:
                risks = json.loads(row["risks"])
            except Exception:
                risks = []
            try:
                assumptions = json.loads(row["assumptions"])
            except Exception:
                assumptions = []
            try:
                rejected = json.loads(row["rejected_options"])
            except Exception:
                rejected = []
            return DecisionRecord(
                decision_id=str(row["decision_id"]),
                task_id=str(row["task_id"]),
                discussion_id=str(row["discussion_id"]) if row["discussion_id"] else None,
                made_by_agent_id=str(row["made_by_agent_id"]),
                decision_status=DecisionStatus(row["decision_status"]),
                summary=str(row["summary"]),
                rationale_summary=str(row["rationale_summary"]),
                selected_route=str(row["selected_route"]),
                assigned_agent_ids=assigned,
                required_verification=req_ver,
                risks=risks,
                assumptions=assumptions,
                rejected_options=rejected,
                created_at=parse_dt(row["created_at"]),
            )

    def list_decisions(self, task_id: str | None = None) -> list[DecisionRecord]:
        query = "SELECT * FROM decisions"
        params: list[Any] = []
        if task_id is not None:
            query += " WHERE task_id = ?"
            params.append(str(task_id))
        query += " ORDER BY created_at ASC;"

        with self.db.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                try:
                    assigned = json.loads(row["assigned_agent_ids"])
                except Exception:
                    assigned = []
                try:
                    req_ver = json.loads(row["required_verification"])
                except Exception:
                    req_ver = []
                try:
                    risks = json.loads(row["risks"])
                except Exception:
                    risks = []
                try:
                    assumptions = json.loads(row["assumptions"])
                except Exception:
                    assumptions = []
                try:
                    rejected = json.loads(row["rejected_options"])
                except Exception:
                    rejected = []
                results.append(
                    DecisionRecord(
                        decision_id=str(row["decision_id"]),
                        task_id=str(row["task_id"]),
                        discussion_id=str(row["discussion_id"]) if row["discussion_id"] else None,
                        made_by_agent_id=str(row["made_by_agent_id"]),
                        decision_status=DecisionStatus(row["decision_status"]),
                        summary=str(row["summary"]),
                        rationale_summary=str(row["rationale_summary"]),
                        selected_route=str(row["selected_route"]),
                        assigned_agent_ids=assigned,
                        required_verification=req_ver,
                        risks=risks,
                        assumptions=assumptions,
                        rejected_options=rejected,
                        created_at=parse_dt(row["created_at"]),
                    )
                )
            return results

    # -------------------------------------------------------------------------
    # Messages
    # -------------------------------------------------------------------------

    def save_message(self, message: AgentMessage) -> AgentMessage:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO messages (
                    message_id, discussion_id, from_agent_id, to_agent_id,
                    task_id, message_type, content, root_task_id,
                    evidence_references, confidence, requested_action, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    message.message_id,
                    message.discussion_id,
                    message.from_agent_id,
                    message.to_agent_id,
                    message.task_id,
                    message.message_type.value if isinstance(message.message_type, MessageType) else str(message.message_type),
                    message.content,
                    message.root_task_id,
                    json.dumps(message.evidence_references, ensure_ascii=False),
                    float(message.confidence),
                    message.requested_action,
                    json.dumps(json_safe_tool_payload(message.metadata), ensure_ascii=False),
                    _dt_to_iso(message.created_at),
                ),
            )
        return message

    def get_message(self, message_id: str) -> AgentMessage | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM messages WHERE message_id = ?;", (message_id,)).fetchone()
            if row is None:
                return None
            return self._hydrate_message(row)

    def _hydrate_message(self, row: sqlite3.Row) -> AgentMessage:
        try:
            evidence_references = json.loads(row["evidence_references"])
        except Exception:
            evidence_references = []
        try:
            metadata = json.loads(row["metadata"])
        except Exception:
            metadata = {}
        return AgentMessage(
            message_id=str(row["message_id"]),
            discussion_id=str(row["discussion_id"]) if row["discussion_id"] else None,
            from_agent_id=str(row["from_agent_id"]),
            to_agent_id=str(row["to_agent_id"]) if row["to_agent_id"] else None,
            task_id=str(row["task_id"]),
            message_type=MessageType(row["message_type"]),
            content=str(row["content"]),
            root_task_id=str(row["root_task_id"]),
            evidence_references=evidence_references,
            confidence=float(row["confidence"]),
            requested_action=str(row["requested_action"]),
            metadata=metadata,
            created_at=parse_dt(row["created_at"]),
        )

    def list_messages_for_task(self, task_id: str) -> list[AgentMessage]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE task_id = ? ORDER BY created_at ASC;",
                (task_id,),
            ).fetchall()
            return [self._hydrate_message(r) for r in rows]

    def list_messages_for_thread(self, discussion_id: str) -> list[AgentMessage]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE discussion_id = ? ORDER BY created_at ASC;",
                (discussion_id,),
            ).fetchall()
            return [self._hydrate_message(r) for r in rows]

    def list_messages_for_inbox(self, agent_id: str) -> list[AgentMessage]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE to_agent_id = ? OR to_agent_id IS NULL ORDER BY created_at ASC;",
                (agent_id,),
            ).fetchall()
            return [self._hydrate_message(r) for r in rows]

    def list_all_messages(self) -> list[AgentMessage]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM messages ORDER BY created_at ASC;").fetchall()
            return [self._hydrate_message(r) for r in rows]

    # -------------------------------------------------------------------------
    # Gateway Executions
    # -------------------------------------------------------------------------

    def save_execution(self, execution: LaneExecution) -> LaneExecution:
        b = execution.budget
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO gateway_executions (
                    task_id, root_task_id, parent_task_id, owning_lane, state,
                    normalized_intent, repository_id, workspace_id, session_id,
                    target_files, priority, taskboard_task_id, worker_id, model,
                    provider, routing_decision_id, task_type, capabilities, changed_files,
                    verification_state, duplicate_of, last_heartbeat, created_at, updated_at,
                    task_created_at, scheduled_at, worker_claimed_at, provider_started_at,
                    provider_completed_at, task_completed_at, duration_breakdown, error,
                    heartbeat_failure, progress_summary, current_tool_activity, evidence,
                    cancellation_state, final_result, supervisor_attempt_id, checkpoint_id,
                    trigger_turn_id, relation_type, previous_task_id, user_message_id,
                    accounting_reservation_ids, lane_history,
                    budget_reserved_input_tokens, budget_reserved_output_tokens,
                    budget_consumed_input_tokens, budget_consumed_output_tokens,
                    budget_estimated_cost, budget_actual_cost,
                    budget_estimated_cost_known, budget_actual_cost_known,
                    budget_model_context_window, budget_model_max_output_tokens,
                    budget_estimate_confidence, budget_estimate_source,
                    budget_turn_budget_tokens, budget_turn_consumed_tokens,
                    budget_turn_reserved_tokens, budget_revisions
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?
                );
                """,
                (
                    execution.task_id,
                    execution.root_task_id,
                    execution.parent_task_id,
                    execution.owning_lane.value if isinstance(execution.owning_lane, LaneId) else str(execution.owning_lane),
                    execution.state.value if isinstance(execution.state, LaneTaskState) else str(execution.state),
                    execution.normalized_intent,
                    execution.repository_id,
                    execution.workspace_id,
                    execution.session_id,
                    json.dumps(execution.target_files, ensure_ascii=False),
                    execution.priority.value if isinstance(execution.priority, LanePriority) else str(execution.priority),
                    execution.taskboard_task_id,
                    execution.worker_id,
                    execution.model,
                    execution.provider,
                    execution.routing_decision_id,
                    execution.task_type,
                    json.dumps(execution.capabilities, ensure_ascii=False),
                    json.dumps(execution.changed_files, ensure_ascii=False),
                    json.dumps(json_safe_tool_payload(execution.verification_state), ensure_ascii=False),
                    execution.duplicate_of,
                    execution.last_heartbeat,
                    execution.created_at,
                    execution.updated_at,
                    execution.task_created_at,
                    execution.scheduled_at,
                    execution.worker_claimed_at,
                    execution.provider_started_at,
                    execution.provider_completed_at,
                    execution.task_completed_at,
                    json.dumps(execution.duration_breakdown, ensure_ascii=False),
                    execution.error,
                    execution.heartbeat_failure,
                    execution.progress_summary,
                    json.dumps(json_safe_tool_payload(execution.current_tool_activity), ensure_ascii=False),
                    json.dumps(json_safe_tool_payload(execution.evidence), ensure_ascii=False),
                    json.dumps(json_safe_tool_payload(execution.cancellation_state), ensure_ascii=False),
                    json.dumps(json_safe_tool_payload(execution.final_result), ensure_ascii=False),
                    execution.supervisor_attempt_id,
                    execution.checkpoint_id,
                    execution.trigger_turn_id,
                    execution.relation_type,
                    execution.previous_task_id,
                    execution.user_message_id,
                    json.dumps(execution.accounting_reservation_ids, ensure_ascii=False),
                    json.dumps(json_safe_tool_payload(execution.lane_history), ensure_ascii=False),
                    int(b.reserved_input_tokens),
                    int(b.reserved_output_tokens),
                    int(b.consumed_input_tokens),
                    int(b.consumed_output_tokens),
                    float(b.estimated_cost),
                    float(b.actual_cost),
                    1 if b.estimated_cost_known else 0,
                    1 if b.actual_cost_known else 0,
                    int(b.model_context_window),
                    int(b.model_max_output_tokens),
                    str(b.estimate_confidence),
                    str(b.estimate_source),
                    int(b.turn_budget_tokens),
                    int(b.turn_consumed_tokens),
                    int(b.turn_reserved_tokens),
                    json.dumps(json_safe_tool_payload(b.revisions), ensure_ascii=False),
                ),
            )

            # Handoffs
            conn.execute("DELETE FROM gateway_handoffs WHERE task_id = ?;", (execution.task_id,))
            for h in execution.handoffs:
                conn.execute(
                    """
                    INSERT INTO gateway_handoffs (
                        task_id, source_lane, target_lane, reason, artifacts,
                        changed_files, remaining_work, verification_state, budget_consumed, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        execution.task_id,
                        h.source_lane.value if isinstance(h.source_lane, LaneId) else str(h.source_lane),
                        h.target_lane.value if isinstance(h.target_lane, LaneId) else str(h.target_lane),
                        h.reason,
                        json.dumps(h.artifacts, ensure_ascii=False),
                        json.dumps(h.changed_files, ensure_ascii=False),
                        json.dumps(h.remaining_work, ensure_ascii=False),
                        json.dumps(json_safe_tool_payload(h.verification_state), ensure_ascii=False),
                        json.dumps(json_safe_tool_payload(asdict(h.budget_consumed)), ensure_ascii=False),
                        h.created_at,
                    ),
                )
        return execution

    def _hydrate_execution(self, row: sqlite3.Row, conn: sqlite3.Connection) -> LaneExecution:
        task_id = str(row["task_id"])
        # Fetch handoffs
        h_rows = conn.execute(
            "SELECT source_lane, target_lane, task_id, reason, artifacts, changed_files, remaining_work, verification_state, budget_consumed, created_at FROM gateway_handoffs WHERE task_id = ? ORDER BY id ASC;",
            (task_id,),
        ).fetchall()
        handoffs = []
        for h in h_rows:
            try:
                artifacts = json.loads(h["artifacts"])
            except Exception:
                artifacts = []
            try:
                changed_files = json.loads(h["changed_files"])
            except Exception:
                changed_files = []
            try:
                remaining_work = json.loads(h["remaining_work"])
            except Exception:
                remaining_work = []
            try:
                verification_state = json.loads(h["verification_state"])
            except Exception:
                verification_state = {}
            try:
                b_raw = json.loads(h["budget_consumed"])
                b_consumed = LaneBudget(**b_raw)
            except Exception:
                b_consumed = LaneBudget()
            handoffs.append(
                LaneHandoff(
                    source_lane=LaneId(h["source_lane"]),
                    target_lane=LaneId(h["target_lane"]),
                    task_id=task_id,
                    reason=str(h["reason"]),
                    artifacts=artifacts,
                    changed_files=changed_files,
                    remaining_work=remaining_work,
                    verification_state=verification_state,
                    budget_consumed=b_consumed,
                    created_at=str(h["created_at"]),
                )
            )

        try:
            target_files = json.loads(row["target_files"])
        except Exception:
            target_files = []
        try:
            capabilities = json.loads(row["capabilities"])
        except Exception:
            capabilities = []
        try:
            changed_files = json.loads(row["changed_files"])
        except Exception:
            changed_files = []
        try:
            verification_state = json.loads(row["verification_state"])
        except Exception:
            verification_state = {}
        try:
            duration_breakdown = json.loads(row["duration_breakdown"])
        except Exception:
            duration_breakdown = {}
        try:
            current_tool_activity = json.loads(row["current_tool_activity"])
        except Exception:
            current_tool_activity = {}
        try:
            evidence = json.loads(row["evidence"])
        except Exception:
            evidence = []
        try:
            cancellation_state = json.loads(row["cancellation_state"])
        except Exception:
            cancellation_state = {}
        try:
            final_result = json.loads(row["final_result"])
        except Exception:
            final_result = {}
        try:
            accounting_reservation_ids = json.loads(row["accounting_reservation_ids"])
        except Exception:
            accounting_reservation_ids = []
        try:
            lane_history = json.loads(row["lane_history"])
        except Exception:
            lane_history = []
        try:
            budget_revisions = json.loads(row["budget_revisions"])
        except Exception:
            budget_revisions = []

        budget = LaneBudget(
            reserved_input_tokens=int(row["budget_reserved_input_tokens"]),
            reserved_output_tokens=int(row["budget_reserved_output_tokens"]),
            consumed_input_tokens=int(row["budget_consumed_input_tokens"]),
            consumed_output_tokens=int(row["budget_consumed_output_tokens"]),
            estimated_cost=float(row["budget_estimated_cost"]),
            actual_cost=float(row["budget_actual_cost"]),
            estimated_cost_known=bool(row["budget_estimated_cost_known"]),
            actual_cost_known=bool(row["budget_actual_cost_known"]),
            model_context_window=int(row["budget_model_context_window"]),
            model_max_output_tokens=int(row["budget_model_max_output_tokens"]),
            estimate_confidence=str(row["budget_estimate_confidence"]),
            estimate_source=str(row["budget_estimate_source"]),
            revisions=budget_revisions,
            turn_budget_tokens=int(row["budget_turn_budget_tokens"]),
            turn_consumed_tokens=int(row["budget_turn_consumed_tokens"]),
            turn_reserved_tokens=int(row["budget_turn_reserved_tokens"]),
        )

        return LaneExecution(
            task_id=task_id,
            root_task_id=str(row["root_task_id"]),
            parent_task_id=str(row["parent_task_id"]) if row["parent_task_id"] else None,
            owning_lane=LaneId(row["owning_lane"]),
            state=LaneTaskState(row["state"]),
            normalized_intent=str(row["normalized_intent"]),
            repository_id=str(row["repository_id"]),
            workspace_id=str(row["workspace_id"]),
            session_id=str(row["session_id"]),
            target_files=target_files,
            priority=LanePriority(row["priority"]),
            budget=budget,
            taskboard_task_id=str(row["taskboard_task_id"]),
            worker_id=str(row["worker_id"]),
            model=str(row["model"]),
            provider=str(row["provider"]),
            routing_decision_id=str(row["routing_decision_id"]),
            accounting_reservation_ids=accounting_reservation_ids,
            task_type=str(row["task_type"]),
            capabilities=capabilities,
            changed_files=changed_files,
            verification_state=verification_state,
            lane_history=lane_history,
            handoffs=handoffs,
            duplicate_of=str(row["duplicate_of"]) if row["duplicate_of"] else None,
            last_heartbeat=str(row["last_heartbeat"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            task_created_at=str(row["task_created_at"]),
            scheduled_at=str(row["scheduled_at"]),
            worker_claimed_at=str(row["worker_claimed_at"]),
            provider_started_at=str(row["provider_started_at"]),
            provider_completed_at=str(row["provider_completed_at"]),
            task_completed_at=str(row["task_completed_at"]),
            duration_breakdown=duration_breakdown,
            error=str(row["error"]),
            heartbeat_failure=str(row["heartbeat_failure"]),
            progress_summary=str(row["progress_summary"]),
            current_tool_activity=current_tool_activity,
            evidence=evidence,
            cancellation_state=cancellation_state,
            final_result=final_result,
            supervisor_attempt_id=str(row["supervisor_attempt_id"]),
            supervisor_lease_token="",
            checkpoint_id=str(row["checkpoint_id"]),
            trigger_turn_id=str(row["trigger_turn_id"]),
            relation_type=str(row["relation_type"]),
            previous_task_id=str(row["previous_task_id"]),
            user_message_id=str(row["user_message_id"]),
        )

    def get_execution(self, task_id: str) -> LaneExecution | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM gateway_executions WHERE task_id = ?;", (task_id,)).fetchone()
            if row is None:
                return None
            return self._hydrate_execution(row, conn)

    def list_executions(self, active_only: bool = False, session_id: str | None = None) -> list[LaneExecution]:
        query = "SELECT * FROM gateway_executions WHERE 1=1"
        params: list[Any] = []
        if active_only:
            placeholders = ", ".join("?" for _ in ACTIVE_LANE_STATES)
            query += f" AND state IN ({placeholders})"
            params.extend(s.value for s in ACTIVE_LANE_STATES)
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(str(session_id))
        query += " ORDER BY created_at ASC;"

        with self.db.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._hydrate_execution(r, conn) for r in rows]

    def delete_execution(self, task_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM gateway_handoffs WHERE task_id = ?;", (task_id,))
            conn.execute("DELETE FROM gateway_executions WHERE task_id = ?;", (task_id,))

    # -------------------------------------------------------------------------
    # Gateway Locks & Waiters
    # -------------------------------------------------------------------------

    def save_lock(self, lock: LockLease) -> LockLease:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO gateway_locks (
                    lease_id, task_id, mode, workspace_id, repository_id,
                    paths, owner_pid, acquired_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    lock.lease_id,
                    lock.task_id,
                    lock.mode.value if isinstance(lock.mode, LockMode) else str(lock.mode),
                    lock.workspace_id,
                    lock.repository_id,
                    json.dumps(lock.paths, ensure_ascii=False),
                    int(lock.owner_pid),
                    lock.acquired_at,
                    lock.expires_at,
                ),
            )
        return lock

    def delete_lock(self, lease_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM gateway_locks WHERE lease_id = ?;", (lease_id,))

    def list_locks(self, workspace_id: str | None = None) -> list[LockLease]:
        query = "SELECT * FROM gateway_locks"
        params: list[Any] = []
        if workspace_id is not None:
            query += " WHERE workspace_id = ?"
            params.append(str(workspace_id))
        query += " ORDER BY acquired_at ASC;"

        with self.db.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for r in rows:
                try:
                    paths = json.loads(r["paths"])
                except Exception:
                    paths = []
                results.append(
                    LockLease(
                        lease_id=str(r["lease_id"]),
                        task_id=str(r["task_id"]),
                        mode=LockMode(r["mode"]),
                        workspace_id=str(r["workspace_id"]),
                        repository_id=str(r["repository_id"]),
                        paths=paths,
                        owner_pid=int(r["owner_pid"]),
                        acquired_at=str(r["acquired_at"]),
                        expires_at=str(r["expires_at"]),
                    )
                )
            return results

    def clear_locks(self) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM gateway_locks;")

    def save_waiters(self, waiters: list[dict[str, Any]]) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM gateway_waiters;")
            for w in waiters:
                seq = int(w.get("sequence", 0))
                task_id = str(w.get("task_id") or "")
                session_id = str(w.get("session_id") or "")
                lane_id = str(w.get("lane_id") or "")
                priority = str(w.get("priority") or "")
                intent = str(w.get("normalized_intent") or "")
                conn.execute(
                    "INSERT INTO gateway_waiters (sequence, task_id, session_id, lane_id, priority, normalized_intent, waiter_data, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                    (seq, task_id, session_id, lane_id, priority, intent, json.dumps(json_safe_tool_payload(w), ensure_ascii=False), _dt_to_iso(utc_now())),
                )

    def list_waiters(self) -> list[dict[str, Any]]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT waiter_data FROM gateway_waiters ORDER BY sequence ASC;").fetchall()
            results = []
            for r in rows:
                try:
                    results.append(json.loads(r["waiter_data"]))
                except Exception:
                    pass
            return results

    def clear_waiters(self) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM gateway_waiters;")

    # -------------------------------------------------------------------------
    # Gateway Turns (Idempotency & Turn ledger)
    # -------------------------------------------------------------------------

    def create_or_get_turn(
        self,
        *,
        conversation_id: str,
        user_message_id: str,
        turn_id: str,
        text: str,
    ) -> tuple[ChatTurnRecord, bool]:
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        now_str = _dt_to_iso(utc_now())

        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM gateway_turns WHERE conversation_id = ? AND user_message_id = ?;",
                (conversation_id, user_message_id),
            ).fetchone()
            if row is not None:
                record = self._hydrate_turn(row)
                if record.message_fingerprint != fingerprint:
                    raise ValueError("user_message_id already belongs to a different message")
                return record, True

            record = ChatTurnRecord(
                turn_id=turn_id,
                conversation_id=conversation_id,
                user_message_id=user_message_id,
                message_fingerprint=fingerprint,
                received_at=now_str,
            )
            conn.execute(
                """
                INSERT INTO gateway_turns (
                    turn_id, conversation_id, user_message_id, message_fingerprint,
                    received_at, status, normalized_intent, routing_decision_id,
                    related_task_ids, created_task_ids, response_message_id,
                    response_execution_id, response, workspace_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.turn_id,
                    record.conversation_id,
                    record.user_message_id,
                    record.message_fingerprint,
                    record.received_at,
                    record.status,
                    record.normalized_intent,
                    record.routing_decision_id,
                    json.dumps(record.related_task_ids, ensure_ascii=False),
                    json.dumps(record.created_task_ids, ensure_ascii=False),
                    record.response_message_id,
                    record.response_execution_id,
                    json.dumps(json_safe_tool_payload(record.response), ensure_ascii=False),
                    self.workspace_id,
                    now_str,
                    now_str,
                ),
            )
            return record, False

    def update_turn(self, record: ChatTurnRecord) -> ChatTurnRecord:
        now_str = _dt_to_iso(utc_now())
        safe_response = json_safe_tool_payload(record.response)
        record.response = safe_response
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO gateway_turns (
                    turn_id, conversation_id, user_message_id, message_fingerprint,
                    received_at, status, normalized_intent, routing_decision_id,
                    related_task_ids, created_task_ids, response_message_id,
                    response_execution_id, response, workspace_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.turn_id,
                    record.conversation_id,
                    record.user_message_id,
                    record.message_fingerprint,
                    record.received_at,
                    record.status,
                    record.normalized_intent,
                    record.routing_decision_id,
                    json.dumps(record.related_task_ids, ensure_ascii=False),
                    json.dumps(record.created_task_ids, ensure_ascii=False),
                    record.response_message_id,
                    record.response_execution_id,
                    json.dumps(safe_response, ensure_ascii=False),
                    self.workspace_id,
                    record.received_at,
                    now_str,
                ),
            )
        return record

    def get_turn(self, turn_id: str) -> ChatTurnRecord | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM gateway_turns WHERE turn_id = ?;", (turn_id,)).fetchone()
            if row is None:
                return None
            return self._hydrate_turn(row)

    def _hydrate_turn(self, row: sqlite3.Row) -> ChatTurnRecord:
        try:
            related_task_ids = json.loads(row["related_task_ids"])
        except Exception:
            related_task_ids = []
        try:
            created_task_ids = json.loads(row["created_task_ids"])
        except Exception:
            created_task_ids = []
        try:
            response = json.loads(row["response"])
        except Exception:
            response = {}

        return ChatTurnRecord(
            turn_id=str(row["turn_id"]),
            conversation_id=str(row["conversation_id"]),
            user_message_id=str(row["user_message_id"]),
            message_fingerprint=str(row["message_fingerprint"]),
            received_at=str(row["received_at"]),
            status=str(row["status"]),
            normalized_intent=str(row["normalized_intent"] or ""),
            routing_decision_id=str(row["routing_decision_id"] or ""),
            related_task_ids=related_task_ids,
            created_task_ids=created_task_ids,
            response_message_id=str(row["response_message_id"] or ""),
            response_execution_id=str(row["response_execution_id"] or ""),
            response=response,
        )

    # -------------------------------------------------------------------------
    # Checkpoints
    # -------------------------------------------------------------------------

    def save_checkpoint(self, checkpoint: CheckpointRecord) -> CheckpointRecord:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints (
                    checkpoint_id, task_id, boundary, completed_steps,
                    pending_steps, workspace_reference, state_snapshot, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    checkpoint.checkpoint_id,
                    checkpoint.task_id,
                    checkpoint.boundary,
                    json.dumps(list(checkpoint.completed_steps), ensure_ascii=False),
                    json.dumps(list(checkpoint.pending_steps), ensure_ascii=False),
                    checkpoint.workspace_reference or "",
                    json.dumps(json_safe_tool_payload(checkpoint.state_snapshot or {}), ensure_ascii=False),
                    _dt_to_iso(checkpoint.created_at),
                ),
            )
        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM checkpoints WHERE checkpoint_id = ?;", (checkpoint_id,)).fetchone()
            if row is None:
                return None
            try:
                completed = tuple(json.loads(row["completed_steps"]))
            except Exception:
                completed = ()
            try:
                pending = tuple(json.loads(row["pending_steps"]))
            except Exception:
                pending = ()
            try:
                snapshot = json.loads(row["state_snapshot"])
            except Exception:
                snapshot = {}
            return CheckpointRecord(
                checkpoint_id=str(row["checkpoint_id"]),
                task_id=str(row["task_id"]),
                boundary=str(row["boundary"]),
                completed_steps=completed,
                pending_steps=pending,
                workspace_reference=str(row["workspace_reference"]),
                state_snapshot=snapshot,
                created_at=parse_dt(row["created_at"]),
            )

    def checkpoints_for_task(self, task_id: str) -> list[CheckpointRecord]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM checkpoints WHERE task_id = ? ORDER BY created_at ASC;", (task_id,)).fetchall()
            results = []
            for row in rows:
                try:
                    completed = tuple(json.loads(row["completed_steps"]))
                except Exception:
                    completed = ()
                try:
                    pending = tuple(json.loads(row["pending_steps"]))
                except Exception:
                    pending = ()
                try:
                    snapshot = json.loads(row["state_snapshot"])
                except Exception:
                    snapshot = {}
                results.append(
                    CheckpointRecord(
                        checkpoint_id=str(row["checkpoint_id"]),
                        task_id=str(row["task_id"]),
                        boundary=str(row["boundary"]),
                        completed_steps=completed,
                        pending_steps=pending,
                        workspace_reference=str(row["workspace_reference"]),
                        state_snapshot=snapshot,
                        created_at=parse_dt(row["created_at"]),
                    )
                )
            return results

    # -------------------------------------------------------------------------
    # Escrow Results
    # -------------------------------------------------------------------------

    def save_escrow_result(self, result: EscrowResult) -> EscrowResult:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO escrow_results (
                    result_id, task_id, execution_id, trigger_turn_id,
                    session_id, parent_task_id, status, result_data, acknowledged, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    result.result_id,
                    result.task_id,
                    result.execution_id,
                    result.trigger_turn_id,
                    result.session_id,
                    result.parent_task_id,
                    str(result.status.value if hasattr(result.status, "value") else result.status),
                    json.dumps(result.model_dump(mode="json"), ensure_ascii=False),
                    1 if getattr(result, "acknowledged", False) else 0,
                    _dt_to_iso(result.created_at),
                ),
            )
        return result

    def get_escrow_result(self, result_id: str) -> EscrowResult | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT result_data FROM escrow_results WHERE result_id = ?;", (result_id,)).fetchone()
            if row is None:
                return None
            try:
                return EscrowResult.model_validate_json(row["result_data"])
            except Exception:
                return None

    def get_escrow_result_by_execution_id(self, execution_id: str) -> EscrowResult | None:
        with self.db.connect() as conn:
            row = conn.execute("SELECT result_data FROM escrow_results WHERE execution_id = ?;", (execution_id,)).fetchone()
            if row is None:
                return None
            try:
                return EscrowResult.model_validate_json(row["result_data"])
            except Exception:
                return None

    def results_for_task(self, task_id: str) -> list[EscrowResult]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT result_data FROM escrow_results WHERE task_id = ? ORDER BY created_at ASC;", (task_id,)).fetchall()
            results = []
            for r in rows:
                try:
                    results.append(EscrowResult.model_validate_json(r["result_data"]))
                except Exception:
                    pass
            return results

    def results_for_turn(self, trigger_turn_id: str) -> list[EscrowResult]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT result_data FROM escrow_results WHERE trigger_turn_id = ? ORDER BY created_at ASC;", (trigger_turn_id,)).fetchall()
            results = []
            for r in rows:
                try:
                    results.append(EscrowResult.model_validate_json(r["result_data"]))
                except Exception:
                    pass
            return results

    def results_for_session(self, session_id: str) -> list[EscrowResult]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT result_data FROM escrow_results WHERE session_id = ? ORDER BY created_at ASC;", (session_id,)).fetchall()
            results = []
            for r in rows:
                try:
                    results.append(EscrowResult.model_validate_json(r["result_data"]))
                except Exception:
                    pass
            return results
