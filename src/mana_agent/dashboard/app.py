"""Mana-Agent Streamlit dashboard entrypoint.

Uses Streamlit multipage navigation (`st.navigation` / `st.Page`) so sidebar
items are real app routes with active state — not checkbox/radio forms.

Launch:
  mana-agent dashboard
  streamlit run src/mana_agent/dashboard/app.py
"""

from __future__ import annotations

import sys

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    print("ERROR: streamlit is required. pip install 'mana-agent[dashboard]'", file=sys.stderr)
    raise SystemExit(1) from exc

from mana_agent.dashboard.pages import (
    analyze,
    automations,
    chat,
    connectors,
    cron,
    metrics,
    observability,
    overview,
    processes,
    reports,
    skills,
    taskboard,
)
from mana_agent.ui.streamlit_helpers import find_mana_root

st.set_page_config(
    page_title="Mana Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

root = find_mana_root()
st.session_state.setdefault("mana_dashboard_root", str(root))
st.session_state.setdefault("mana_api_base", "http://127.0.0.1:8000")


def _page(fn, *, title: str, icon: str, url_path: str, default: bool = False):
    def _runner() -> None:
        st.sidebar.markdown("## 🧠 Mana Agent")
        st.sidebar.caption(f"Root: `{root.name}`")
        st.sidebar.text_input("API base (for sockets)", key="mana_api_base")
        st.sidebar.caption("Navigation uses Streamlit page routes with active highlighting.")
        fn(root)

    return st.Page(_runner, title=title, icon=icon, url_path=url_path, default=default)


pages = {
    "Workspace": [
        _page(overview.render, title="Overview", icon="🏠", url_path="overview", default=True),
        _page(chat.render, title="Chat", icon="💬", url_path="chat"),
        _page(analyze.render, title="Analyze", icon="🔬", url_path="analyze"),
        _page(reports.render, title="Reports", icon="📄", url_path="reports"),
        _page(skills.render, title="Skill Workshop", icon="🧩", url_path="skill-workshop"),
    ],
    "Runtime": [
        _page(taskboard.render, title="Taskboard & Traces", icon="🗂️", url_path="taskboard"),
        _page(observability.render, title="Observability", icon="📡", url_path="observability"),
        _page(metrics.render, title="Metrics", icon="📊", url_path="metrics"),
        _page(processes.render, title="Processes", icon="⚙️", url_path="processes"),
    ],
    "Operations": [
        _page(connectors.render, title="Connectors", icon="🔌", url_path="connectors"),
        _page(automations.render, title="Automations", icon="⚡", url_path="automations"),
        _page(cron.render, title="Cron Jobs", icon="⏰", url_path="cron"),
    ],
}

nav = st.navigation(pages, position="sidebar")
nav.run()
