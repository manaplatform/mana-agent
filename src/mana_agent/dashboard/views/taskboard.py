from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from mana_agent.config.settings import Settings
from mana_agent.execution_supervisor import ExecutionSupervisor, ExecutionSupervisorConfig
from mana_agent.multi_agent.core.types import TaskStatus
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.ui.streamlit_helpers import (
    find_mana_root,
    load_recent_traces,
    load_taskboard_state,
)


def render(root: Path | None = None) -> None:
    root = root or find_mana_root()
    st.header("TaskBoard & Execution Workspace")
    st.caption("Supporting task-management workspace and state authority for Mana-Agent.")

    tb: TaskBoard | None = None
    try:
        tb = TaskBoard(root)
    except Exception as exc:
        st.caption(f"TaskBoard direct storage note: {exc}")

    tab_tasks, tab_supervisor, tab_traces = st.tabs([
        "📋 TaskBoard Workspace",
        "🛡️ Execution Supervisor",
        "📜 Activity Logs & Traces",
    ])

    with tab_tasks:
        _render_taskboard_workspace(root, tb)

    with tab_supervisor:
        _render_supervisor_state(root)

    with tab_traces:
        _render_traces(root)


def _render_taskboard_workspace(root: Path, tb: TaskBoard | None) -> None:
    tasks: list[Any] = []
    if tb is not None:
        try:
            tasks = tb.list_tasks()
        except Exception:
            tasks = []

    if not tasks:
        raw_state = load_taskboard_state(root)
        raw_tasks_dict = raw_state.get("tasks", {}) if isinstance(raw_state, dict) else {}
        if isinstance(raw_tasks_dict, dict):
            tasks = list(raw_tasks_dict.values())

    # Progress Summary
    if tb is not None:
        try:
            progress = tb.get_progress()
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Tasks", progress.total_tasks)
            col2.metric("Completed", progress.completed_tasks)
            col3.metric("Verified", progress.verified_tasks)
            col4.metric("Crossed Off", progress.crossed_off_tasks)
            col5.metric("Blocked", progress.blocked_tasks)

            st.progress(
                progress.percent_complete / 100.0,
                text=f"Progress: {progress.percent_complete:.1f}% ({progress.completed_tasks}/{progress.total_tasks} completed, {progress.verified_tasks} verified)",
            )
        except Exception:
            pass
    elif tasks:
        total = len(tasks)
        done = sum(1 for t in tasks if (getattr(t, "status", None) == TaskStatus.DONE or (isinstance(t, dict) and t.get("status") == "done")))
        col1, col2 = st.columns(2)
        col1.metric("Total Tasks", total)
        col2.metric("Done", done)
        st.progress(done / max(total, 1))

    # Form to create new task
    with st.expander("➕ Create New Task", expanded=False):
        with st.form("create_task_form", clear_on_submit=True):
            new_title = st.text_input("Title", placeholder="e.g. Implement user login API")
            new_desc = st.text_area("Description / Goal", placeholder="Detailed requirements or acceptance criteria...")
            new_parent = st.text_input("Parent Task ID (optional)", placeholder="task_...")
            submitted = st.form_submit_button("Create Task")
            if submitted and new_title.strip():
                if tb is not None:
                    try:
                        created = tb.create_task(
                            title=new_title.strip(),
                            user_request=new_desc.strip(),
                            parent_task_id=new_parent.strip() if new_parent.strip() else None,
                        )
                        st.success(f"Task created: {created.task_id}")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to create task: {exc}")
                else:
                    st.error("TaskBoard is not available for writes in this environment.")

    if not tasks:
        st.info("No tasks recorded yet on the TaskBoard.")
        return

    # Filter controls
    st.subheader("Tasks")
    filter_col1, filter_col2 = st.columns([2, 3])
    with filter_col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Active (Routed / In Progress)", "Verifying", "Completed (Done)", "Blocked", "Crossed Off"],
            index=0,
        )
    with filter_col2:
        search_query = st.text_input("Search tasks", placeholder="Search by title or ID...").lower().strip()

    filtered_tasks: list[Any] = []
    for t in tasks:
        t_id = getattr(t, "task_id", "") if not isinstance(t, dict) else t.get("task_id", "")
        t_title = getattr(t, "title", "") if not isinstance(t, dict) else t.get("title", "")
        t_status = getattr(t, "status", None)
        t_status_val = t_status.value if hasattr(t_status, "value") else str(t_status or (t.get("status") if isinstance(t, dict) else ""))
        t_crossed = bool(getattr(t, "crossed_off", False) if not isinstance(t, dict) else t.get("crossed_off", False))

        if search_query and search_query not in t_title.lower() and search_query not in t_id.lower():
            continue

        if status_filter == "Active (Routed / In Progress)" and t_status_val not in {"routed", "in_progress"}:
            continue
        elif status_filter == "Verifying" and t_status_val != "verifying":
            continue
        elif status_filter == "Completed (Done)" and t_status_val != "done":
            continue
        elif status_filter == "Blocked" and t_status_val != "blocked":
            continue
        elif status_filter == "Crossed Off" and not t_crossed:
            continue

        filtered_tasks.append(t)

    st.caption(f"Showing {len(filtered_tasks)} of {len(tasks)} tasks")

    # Render Task Items
    for t in filtered_tasks:
        t_id = getattr(t, "task_id", "") if not isinstance(t, dict) else t.get("task_id", "")
        t_title = getattr(t, "title", "") if not isinstance(t, dict) else t.get("title", "")
        t_request = getattr(t, "user_request", "") if not isinstance(t, dict) else t.get("user_request", "")
        t_status = getattr(t, "status", None)
        t_status_val = t_status.value if hasattr(t_status, "value") else str(t_status or (t.get("status") if isinstance(t, dict) else ""))
        t_crossed = bool(getattr(t, "crossed_off", False) if not isinstance(t, dict) else t.get("crossed_off", False))
        t_plan = getattr(t, "plan", []) if not isinstance(t, dict) else t.get("plan", [])
        t_depends = getattr(t, "depends_on", []) if not isinstance(t, dict) else t.get("depends_on", [])
        t_subtasks = getattr(t, "subtasks", []) if not isinstance(t, dict) else t.get("subtasks", [])
        t_blockers = getattr(t, "blockers", []) if not isinstance(t, dict) else t.get("blockers", [])
        t_evidence = getattr(t, "evidence", []) if not isinstance(t, dict) else t.get("evidence", [])
        t_verifications = getattr(t, "verification_results", []) if not isinstance(t, dict) else t.get("verification_results", [])

        is_verified = bool(t_verifications and any(
            (v.passed if hasattr(v, "passed") else v.get("passed"))
            for v in t_verifications
        ))

        container = st.container(border=True)
        with container:
            hdr_col1, hdr_col2, hdr_col3 = st.columns([6, 2, 2])
            with hdr_col1:
                # Checkbox for crossing off
                new_cross = st.checkbox(
                    f"**{t_title}** `{t_id[:12]}`",
                    value=t_crossed,
                    key=f"chk_cross_{t_id}",
                    help="Cross off task on TaskBoard",
                )
                if new_cross != t_crossed and tb is not None:
                    try:
                        tb.mark_crossed_off(t_id, crossed_off=new_cross)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error crossing off task: {exc}")

            with hdr_col2:
                # Status badge
                status_color = {
                    "done": "🟢 Done",
                    "in_progress": "🔵 In Progress",
                    "verifying": "🟣 Verifying",
                    "blocked": "🔴 Blocked",
                    "skipped": "⚪ Skipped",
                    "cancelled": "⚫ Cancelled",
                    "routed": "🟡 Routed",
                }.get(t_status_val.lower(), f"⚪ {t_status_val}")
                st.write(status_color)

            with hdr_col3:
                if is_verified:
                    st.write("🛡️ **Verified**")
                else:
                    st.write("⏳ Unverified")

            # Expandable Details
            with st.expander("Details, Dependencies & Verification", expanded=False):
                if t_request:
                    st.markdown(f"**Goal / Description:** {t_request}")

                if t_blockers:
                    st.error(f"**Blockers:** {', '.join(t_blockers)}")

                if t_depends:
                    st.info(f"**Dependencies:** {', '.join(t_depends)}")

                if t_subtasks:
                    st.markdown(f"**Subtasks ({len(t_subtasks)}):**")
                    for s_id in t_subtasks:
                        st.markdown(f"- `{s_id}`")

                if t_plan:
                    st.markdown("**Plan Steps:**")
                    for idx, step in enumerate(t_plan, 1):
                        st.markdown(f"{idx}. {step}")

                if t_verifications:
                    st.markdown("**Verification Results:**")
                    for v in t_verifications:
                        v_passed = v.passed if hasattr(v, "passed") else v.get("passed", False)
                        v_summary = v.summary if hasattr(v, "summary") else v.get("summary", "")
                        v_icon = "✅" if v_passed else "❌"
                        st.markdown(f"- {v_icon} {v_summary}")

                if t_evidence:
                    st.markdown("**Execution Evidence:**")
                    for ev in t_evidence[-5:]:
                        st.caption(f"• {ev}")

                # Action buttons
                btn_col1, btn_col2, btn_col3 = st.columns(3)
                with btn_col1:
                    if st.button("🔄 Reopen Task", key=f"reopen_{t_id}"):
                        if tb is not None:
                            try:
                                tb.reopen(t_id, reason="Reopened via dashboard")
                                st.success("Task reopened.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Error reopening task: {exc}")

                with btn_col2:
                    if t_status_val.lower() != "done":
                        if st.button("✅ Mark Completed", key=f"complete_{t_id}"):
                            if tb is not None:
                                try:
                                    tb.mark_completed(t_id, result_summary="Completed via dashboard")
                                    st.success("Task marked completed.")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Error completing task: {exc}")

                with btn_col3:
                    if st.button("🗑️ Delete Task", key=f"del_{t_id}"):
                        if tb is not None:
                            try:
                                tb.delete_task(t_id)
                                st.success("Task deleted.")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Error deleting task: {exc}")


def _render_supervisor_state(root: Path) -> None:
    st.subheader("Durable Execution State")
    st.caption("Execution Supervisor controls runtime lifecycle, leases, fences, checkpoints, and retries.")

    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig.from_settings(Settings()))
    try:
        supervised = supervisor.store.list_tasks()
    except Exception as exc:
        st.error(f"Durable execution state is unavailable: {exc}")
        supervised = []

    if supervised:
        try:
            import pandas as pd

            execution_rows = [
                {
                    "task": item.task_id,
                    "attempt": item.attempt_id,
                    "state": item.state.value,
                    "elapsed seconds": max(
                        0,
                        int(
                            (
                                (item.finished_at or datetime.now(timezone.utc))
                                - (item.started_at or item.created_at)
                            ).total_seconds()
                        ),
                    ),
                    "agent": item.assigned_agent,
                    "model": item.assigned_model,
                    "worker": item.assigned_worker,
                    "heartbeat": item.heartbeat_at,
                    "lease expiry": item.lease_expires_at,
                    "retries": f"{item.retry_count}/{sum(item.retry_budget.model_dump().values())}",
                    "checkpoints": item.checkpoint_count,
                    "children": len(item.child_task_ids),
                    "tokens": item.token_usage,
                    "cost": item.actual_cost,
                    "verification": item.verification_status.value,
                    "artefacts": ", ".join(
                        artifact.path or artifact.artifact_type
                        for artifact in item.completion_artefacts
                    ),
                    "reason": item.failure_reason or item.recovery_reason,
                }
                for item in supervised
            ]
            st.dataframe(pd.DataFrame(execution_rows), width="stretch", hide_index=True)
        except Exception:
            st.json([item.model_dump(mode="json") for item in supervised])
    else:
        st.info("No durable supervised executions recorded yet.")


def _render_traces(root: Path) -> None:
    st.subheader("Recent Execution Activity & Traces")
    st.caption("Detailed event stream and execution logs from sessions and CLI runs.")

    traces = load_recent_traces(root, limit=12)
    if not traces:
        st.info("No execution traces recorded yet.")
        return

    for t in traces:
        key = f"{t.get('_file', 'trace')} - {t.get('kind', t.get('event_type', t.get('event', 'event')))}"
        with st.expander(key):
            st.json(t)
