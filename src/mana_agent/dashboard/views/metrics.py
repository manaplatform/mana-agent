from __future__ import annotations

from pathlib import Path

import streamlit as st

from mana_agent.ui.streamlit_helpers import (
    find_mana_root,
    get_metrics_summary,
    get_observability_overview,
)


def render(root: Path | None = None) -> None:
    root = root or find_mana_root()
    st.header("Metrics")
    m = get_metrics_summary(root)
    c1, c2, c3 = st.columns(3)
    c1.metric("Sessions / Traces", m.get("sessions", 0))
    c2.metric("Avg Tokens / span", m.get("avg_tokens", 0))
    c3.metric("Success rate", f"{m.get('success_rate', 0)}%")
    overview = get_observability_overview(root)
    by_kind = overview.get("by_kind", [])
    if by_kind:
        st.bar_chart({row["kind"]: row["tokens"] for row in by_kind})
    else:
        st.info("No observability spans yet.")
