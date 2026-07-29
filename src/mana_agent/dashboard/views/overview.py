from __future__ import annotations

from pathlib import Path

import streamlit as st

from mana_agent.ui.streamlit_helpers import (
    find_mana_root,
    get_index_stats,
    get_last_analysis_summary,
    get_metrics_summary,
)


def render(root: Path | None = None) -> None:
    root = root or find_mana_root()
    st.header("Project Overview")
    st.caption(f"Repository root: `{root}`")

    m = get_metrics_summary(root)
    idx = get_index_stats(root)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Index Ready", "✅" if idx.get("ready") else "❌", idx.get("chunks", 0))
    c2.metric("Sessions / Traces", m.get("sessions", 0))
    c3.metric("Success Rate", f"{m.get('success_rate', 0)}%")
    c4.metric("Tokens", m.get("total_tokens", 0))

    st.subheader("Last Analysis")
    analysis = get_last_analysis_summary(root)
    if analysis.get("type") == "md":
        st.markdown(analysis.get("preview", "")[:1500])
        st.caption(f"Source: {analysis.get('path')}")
    elif analysis.get("type") == "json":
        st.json(analysis.get("data", {}))
    else:
        st.info(analysis.get("message", "No analysis yet. Open Analyze to start one."))

    st.subheader("Quick links")
    st.markdown(
        "- **Chat** — multi-conversation chat with live execution events\n"
        "- **Analyze** — start repository analysis and inspect artifacts\n"
        "- **Taskboard & Traces** — multi-agent task state\n"
        "- **Observability** — redacted span explorer"
    )
