from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from mana_agent.dashboard.components.chat_timeline import render_timeline
from mana_agent.services.conversation_service import conversation_service_for_root
from mana_agent.services.execution_event_hub import get_execution_event_hub
from mana_agent.ui.streamlit_helpers import find_mana_root


def _api_base() -> str:
    return str(st.session_state.get("mana_api_base") or "http://127.0.0.1:8000").rstrip("/")


def _ws_url(conversation_id: str, root: Path) -> str:
    base = _api_base().replace("https://", "wss://").replace("http://", "ws://")
    return f"{base}/api/v1/ws/conversations/{conversation_id}?root={root}"


def _socket_bridge(conversation_id: str, root: Path, height: int = 120) -> None:
    """Browser WebSocket client that displays live connection status and last events."""
    url = _ws_url(conversation_id, root)
    html = f"""
    <div id="mana-ws" style="font-family:ui-sans-serif,system-ui;font-size:13px;padding:8px;border:1px solid #3333;border-radius:8px;">
      <div><strong>Live socket</strong>: <span id="st">connecting…</span></div>
      <div id="log" style="max-height:70px;overflow:auto;color:#666;margin-top:4px;"></div>
    </div>
    <script>
      const statusEl = document.getElementById('st');
      const logEl = document.getElementById('log');
      let ws;
      let retries = 0;
      function line(msg) {{
        const d = document.createElement('div');
        d.textContent = msg;
        logEl.prepend(d);
        while (logEl.childElementCount > 8) logEl.removeChild(logEl.lastChild);
      }}
      function connect() {{
        statusEl.textContent = 'connecting…';
        try {{
          ws = new WebSocket({json.dumps(url)});
        }} catch (e) {{
          statusEl.textContent = 'error';
          line(String(e));
          return;
        }}
        ws.onopen = () => {{ statusEl.textContent = 'connected'; retries = 0; line('socket ready'); }};
        ws.onclose = () => {{
          statusEl.textContent = 'disconnected — reconnecting';
          const delay = Math.min(10000, 500 * Math.pow(2, retries++));
          setTimeout(connect, delay);
        }};
        ws.onerror = () => {{ statusEl.textContent = 'error'; }};
        ws.onmessage = (ev) => {{
          try {{
            const data = JSON.parse(ev.data);
            if (data.type === 'event' || data.type === 'event.replay') {{
              const e = data.event || {{}};
              line((e.type || data.type) + ' · ' + (e.title || '') + ' · ' + (e.status || ''));
            }} else if (data.type === 'socket.ready') {{
              line('replay starting');
            }} else if (data.type === 'socket.replay_complete') {{
              line('replay complete (' + (data.count || 0) + ')');
            }} else if (data.type === 'pong') {{
              // ignore
            }} else {{
              line(data.type || 'message');
            }}
          }} catch (err) {{ line(String(ev.data).slice(0, 120)); }}
        }};
      }}
      connect();
      setInterval(() => {{ if (ws && ws.readyState === 1) ws.send('ping'); }}, 15000);
    </script>
    """
    components.html(html, height=height)


def _run_chat(root: Path, conversation_id: str, content: str) -> None:
    service = conversation_service_for_root(root)
    try:
        service.send_message(conversation_id, content)
    except Exception as exc:  # ensure status recovers
        try:
            service.set_status(conversation_id, "failed")
            get_execution_event_hub().emit(
                "error",
                title="Chat execution failed",
                conversation_id=conversation_id,
                repository_id=service.repository_id,
                message=str(exc),
                status="failed",
            )
        except Exception:
            pass


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

    with st.expander("Live socket connection", expanded=record.get("status") == "running"):
        st.caption(f"WebSocket: `{_ws_url(conversation_id, root)}`")
        _socket_bridge(conversation_id, root)
        st.caption("Events are also polled from durable conversation storage for reconnect recovery.")

    render_timeline(messages, events)

    if record.get("status") == "running":
        st.info("Execution in progress… timeline refreshes automatically.")
        time.sleep(1.0)
        st.rerun()

    if prompt := st.chat_input("Message this conversation"):
        # The canonical service owns execution; no frontend-owned daemon thread.
        with st.spinner("Mana-Agent is working…"):
            _run_chat(root, conversation_id, prompt)
        st.rerun()
