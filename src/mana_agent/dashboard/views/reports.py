from __future__ import annotations

from pathlib import Path

import streamlit as st

from mana_agent.ui.streamlit_helpers import (
    find_mana_root,
    list_analysis_artifacts,
    safe_read_json,
)


def render(root: Path | None = None) -> None:
    root = root or find_mana_root()
    st.header("Reports & Artifacts")
    st.caption(
        "Read-only view of analysis and report artifacts. Use Analyze to generate new ones."
    )
    arts = list_analysis_artifacts(root)
    if not arts:
        st.info("No artifacts yet.")
        return
    for artifact in arts:
        with st.expander(artifact["name"]):
            path = Path(artifact["path"])
            st.caption(str(path))
            if artifact["type"] == "json":
                st.json(safe_read_json(path) or {})
            else:
                st.code(path.read_text(encoding="utf-8", errors="replace")[:4000])
