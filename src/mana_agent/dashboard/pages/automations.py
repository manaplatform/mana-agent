"""Read-only inspection and deletion for canonical automations."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from mana_agent.automations.service import AutomationService, human_trigger
from mana_agent.ui.streamlit_helpers import find_mana_root


def render(root: Path | None = None) -> None:
    root = root or find_mana_root()
    service = AutomationService(root)
    st.header("Automations")
    st.caption("Create or update automations in chat. This page is for inspection and deletion.")
    records = service.list()
    if not records:
        st.info("No automations yet. Ask Mana-Agent in chat to create one.")
        return
    for automation in records:
        status = automation.deployment.status
        with st.expander(f"{automation.name} · {status}", expanded=False):
            st.write(f"**ID:** `{automation.id}`")
            st.write(f"**Trigger:** {human_trigger(automation.trigger)}")
            st.write(f"**Timezone:** {automation.timezone}")
            st.write(f"**Source:** {automation.source}")
            st.write(f"**Next run:** {automation.next_run_at or '—'}")
            st.write(f"**Last run:** {automation.last_run_at or '—'}")
            st.write(f"**Job:** {automation.job.type}")
            st.write(f"**Deployment:** {automation.deployment.status} ({automation.deployment.backend or 'unassigned'})")
            if automation.deployment.blocked_reason:
                st.warning(automation.deployment.blocked_reason)
            if automation.recent_execution:
                st.write(
                    f"**Last result:** {automation.recent_execution.status} — "
                    f"{automation.recent_execution.output_summary or automation.recent_execution.error or 'no summary'}"
                )
            runs = service.status(automation.id).get("recent_runs", [])
            if runs:
                st.dataframe(runs, use_container_width=True)
            if st.button("Delete", key=f"delete_{automation.id}", type="secondary"):
                service.delete(automation.id)
                st.rerun()
