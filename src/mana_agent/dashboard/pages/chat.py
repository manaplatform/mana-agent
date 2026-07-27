from __future__ import annotations

from pathlib import Path

import streamlit as st

from mana_agent.dashboard.components.live_chat import render_live_chat
from mana_agent.dashboard.components.chat_timeline import render_timeline
from mana_agent.services.conversation_service import conversation_service_for_root
from mana_agent.ui.streamlit_helpers import find_mana_root


def _api_base() -> str:
    return str(st.session_state.get("mana_api_base") or "").strip().rstrip("/")


def render(root: Path | None = None) -> None:
    root = root or find_mana_root()
    service = conversation_service_for_root(root)
    st.header("Chat")
    st.caption(
        "Persistent multi-conversation chat over the Mana-Agent Ask/chat stack. "
        "Runtime events use the shared ChatEvent model and live socket channel."
    )

    # Sidebar conversation controls (page-local)
    with st.sidebar:
        st.markdown("### Conversations")
        if st.button("➕ New conversation", use_container_width=True, key="chat_new_conv"):
            created = service.create(title="New conversation")
            st.session_state.active_conversation_id = created.conversation_id
            st.rerun()
        conversations = service.list(limit=50)
        labels = {
            f"{item.title[:40]} · {item.conversation_id[-8:]}": item.conversation_id
            for item in conversations
        }
        if not labels:
            created = service.create(title="New conversation")
            st.session_state.active_conversation_id = created.conversation_id
            conversations = [created]
            labels = {f"{created.title} · {created.conversation_id[-8:]}": created.conversation_id}
        active = st.session_state.get("active_conversation_id")
        options = list(labels.keys())
        default_idx = 0
        if active:
            for i, key in enumerate(options):
                if labels[key] == active:
                    default_idx = i
                    break
        selected_label = st.selectbox("Open conversation", options, index=default_idx, key="chat_conv_select")
        conversation_id = labels[selected_label]
        st.session_state.active_conversation_id = conversation_id
        rename_title = st.text_input("Rename chat", value=next(item.title for item in conversations if item.conversation_id == conversation_id), key=f"rename_{conversation_id}")
        if st.button("Rename", use_container_width=True, key=f"rename_button_{conversation_id}"):
            service.rename(conversation_id, rename_title)
            st.rerun()
        confirm_delete = st.checkbox("Confirm permanent deletion", key=f"confirm_delete_{conversation_id}")
        if st.button("Delete chat", type="secondary", use_container_width=True, disabled=not confirm_delete, key=f"delete_{conversation_id}"):
            service.delete(conversation_id)
            st.session_state.pop("active_conversation_id", None)
            st.rerun()

    conversation_id = st.session_state.active_conversation_id
    try:
        full = service.get_full(conversation_id)
    except FileNotFoundError:
        st.warning("Conversation not found. Creating a new one.")
        created = service.create()
        st.session_state.active_conversation_id = created.conversation_id
        st.rerun()
        return

    record = full["conversation"]
    messages = full["messages"]
    events = full["events"]

    top = st.columns([3, 1, 1])
    top[0].markdown(f"**{record.get('title', 'Conversation')}**")
    top[1].metric("Status", record.get("status", "idle"))
    top[2].metric("Messages", record.get("message_count", 0))
    st.caption(f"ID `{conversation_id}` · repo `{record.get('repository_id')}`")

    api_base = _api_base()
    if api_base:
        render_live_chat(
            conversation_id=conversation_id,
            root=root,
            api_base=api_base,
            messages=messages,
            events=events,
        )
    else:
        st.warning(
            "Live chat requires the dashboard API. Launch with `mana-agent dashboard`, "
            "or configure API base after starting `mana-agent api`."
        )
        render_timeline(messages, events)
