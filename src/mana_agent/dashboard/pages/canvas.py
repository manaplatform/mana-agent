from __future__ import annotations

from pathlib import Path

import streamlit as st

from mana_agent.dashboard.components.live_canvas import render_live_canvas
from mana_agent.services.conversation_service import conversation_service_for_root
from mana_agent.ui.streamlit_helpers import find_mana_root


def render(root: Path | None = None) -> None:
    root = root or find_mana_root()
    st.header("Live Canvas")
    st.caption("Durable A2UI surfaces and workflow activity from the shared authenticated event stream.")
    service = conversation_service_for_root(root)
    conversations = service.list(limit=50)
    if not conversations:
        st.info("Create a chat before opening Live Canvas.")
        return
    labels = {f"{item.title[:48]} · {item.conversation_id[-8:]}": item.conversation_id for item in conversations}
    selected = st.selectbox("Conversation", list(labels), key="canvas_conversation")
    conversation_id = labels[selected]
    api_base = str(st.session_state.get("mana_api_base") or "").strip().rstrip("/")
    if not api_base:
        st.warning("Live Canvas requires the dashboard API base configured in the sidebar.")
        return
    render_live_canvas(conversation_id=conversation_id, root=root, api_base=api_base)
