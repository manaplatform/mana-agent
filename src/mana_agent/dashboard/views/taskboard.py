from __future__ import annotations

from pathlib import Path

import streamlit as st

from mana_agent.ui.streamlit_helpers import (
    find_mana_root,
    load_recent_traces,
    load_taskboard_state,
)


def render(root: Path | None = None) -> None:
    root = root or find_mana_root()
    st.header("Taskboard & Traces")
    st.caption("Loaded from workspace taskboard state and recent session traces.")
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
