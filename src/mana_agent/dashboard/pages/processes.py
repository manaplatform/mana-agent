from __future__ import annotations

from pathlib import Path

import streamlit as st

from mana_agent.background import BackgroundProcessManager


def render(_root: Path | None = None) -> None:
    manager = BackgroundProcessManager()
    st.header("Background processes")
    st.caption("Persistent registered services are independent from dashboard lifetime and agent tasks.")
    rows = manager.list()
    if not rows:
        st.info("No background processes are recorded.")
        return
    for row in rows:
        with st.container(border=True):
            cols = st.columns([3, 2, 2, 2])
            cols[0].markdown(f"**{row.process_type}** · `{row.process_id}`")
            cols[1].metric("State", row.state)
            cols[2].metric("Health", row.health)
            cols[3].metric("Restarts", row.restart_count)
            st.caption(f"Command: `{row.command_identifier}` · PID: `{row.os_pid or 'none'}` · owner: `{row.ownership}`")
            with st.expander("Managed logs"):
                st.code(manager.logs(row.process_id) or "No log output.")
            actions = st.columns(2)
            if actions[0].button("Stop", key=f"stop_{row.process_id}", disabled=row.state not in {"starting", "running"}):
                manager.stop(row.process_id)
                st.rerun()
            if actions[1].button("Restart", key=f"restart_{row.process_id}", disabled=row.state not in {"stopped", "failed", "stale", "running"}):
                manager.restart(row.process_id)
                st.rerun()
