from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from mana_agent.ui.streamlit_helpers import (
    find_mana_root,
    load_recent_traces,
    load_taskboard_state,
)
from mana_agent.config.settings import Settings
from mana_agent.execution_supervisor import ExecutionSupervisor, ExecutionSupervisorConfig


def render(root: Path | None = None) -> None:
    root = root or find_mana_root()
    st.header("Taskboard & Traces")
    st.caption("Loaded from workspace taskboard state and recent session traces.")
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig.from_settings(Settings()))
    try:
        supervised = supervisor.store.list_tasks()
    except Exception as exc:
        st.error(f"Durable execution state is unavailable: {exc}")
        supervised = []
    if supervised:
        st.subheader("Durable execution state")
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
            st.dataframe(pd.DataFrame(execution_rows), use_container_width=True, hide_index=True)
        except Exception:
            st.json([item.model_dump(mode="json") for item in supervised])
    else:
        st.info("No durable supervised executions yet.")
    tb = load_taskboard_state(root)
    tasks_dict = tb.get("tasks", {}) if isinstance(tb, dict) else {}
    tasks = list(tasks_dict.values()) if isinstance(tasks_dict, dict) else []
    if tasks:
        try:
            import pandas as pd

            rows = [
                {
                    "id": str(t.get("task_id", ""))[:12],
                    "title": str(t.get("title", ""))[:60],
                    "status": t.get("status"),
                    "owner": t.get("owner_agent_id"),
                    "updated": str(t.get("updated_at", ""))[:19],
                }
                for t in tasks
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except Exception:
            st.json({"task_count": len(tasks), "sample": tasks[:2]})
    else:
        st.info("No tasks yet.")
    with st.expander("Raw taskboard state"):
        st.json(tb, expanded=False)
    st.subheader("Recent traces")
    for t in load_recent_traces(root, limit=8)[:10]:
        key = f"{t.get('_file', 'trace')} - {t.get('kind', t.get('event_type', t.get('event', 'event')))}"
        with st.expander(key):
            st.json(t)
