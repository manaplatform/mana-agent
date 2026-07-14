"""Structured chat timeline rendering for messages and runtime events.

Pure helpers (``merge_timeline``) import without Streamlit so core CI can test
timeline ordering. Streamlit is imported lazily only inside render functions.
"""

from __future__ import annotations

from typing import Any


_ROLE_AVATAR = {
    "user": "🧑",
    "assistant": "🤖",
    "system": "⚙️",
    "tool": "🛠️",
    "agent": "🧩",
}

_EVENT_ICONS = {
    "routing": "🧭",
    "plan_step": "📋",
    "reasoning": "💭",
    "tool": "🛠️",
    "subagent": "🧩",
    "user_request": "➡️",
    "response": "✅",
    "error": "⚠️",
    "session": "🔗",
}


def _streamlit():
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional dashboard extra
        raise ImportError(
            "streamlit is required for dashboard timeline rendering. "
            "Install with: pip install 'mana-agent[dashboard]'"
        ) from exc
    return st


def render_message(message: dict[str, Any]) -> None:
    st = _streamlit()
    role = str(message.get("role") or "system").lower()
    avatar = _ROLE_AVATAR.get(role, "💬")
    with st.chat_message(role if role in {"user", "assistant"} else "assistant", avatar=avatar):
        content = str(message.get("content") or "")
        if role in {"tool", "agent", "system"}:
            st.caption(f"{role} · {message.get('created_at', '')}")
            st.markdown(content)
        else:
            st.markdown(content)
        meta = message.get("metadata") or {}
        sources = meta.get("sources") or []
        if sources:
            with st.expander("Sources", expanded=False):
                for source in sources[:8]:
                    st.write(f"- {source.get('file_path') or source.get('path') or source}")


def render_event(event: dict[str, Any], *, compact: bool = True) -> None:
    st = _streamlit()
    kind = str(event.get("kind") or "reasoning")
    status = str(event.get("status") or "running")
    icon = _EVENT_ICONS.get(kind, "•")
    title = str(event.get("title") or event.get("type") or kind)
    summary = str(event.get("summary") or event.get("message") or "")
    duration = event.get("duration_ms")
    duration_txt = f" · {duration}ms" if duration not in (None, "") else ""
    label = f"{icon} **{title}** · `{status}`{duration_txt}"
    if compact and not summary and not (event.get("metadata") or {}):
        st.markdown(label)
        return
    with st.expander(f"{icon} {title} · {status}{duration_txt}", expanded=False):
        if summary:
            st.write(summary)
        meta = event.get("metadata") or event.get("details") or {}
        technical = {
            "type": event.get("type"),
            "kind": kind,
            "event_id": event.get("event_id") or event.get("id"),
            "execution_id": event.get("execution_id") or event.get("turn_id"),
            "agent_id": event.get("agent_id"),
            "subagent_id": event.get("subagent_id"),
            "started_at": event.get("started_at") or event.get("timestamp"),
            "ended_at": event.get("ended_at"),
            "metadata": meta,
        }
        st.json(technical)


def merge_timeline(
    messages: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Interleave messages and events by timestamp for ordered rendering."""
    rows: list[dict[str, Any]] = []
    for message in messages:
        rows.append(
            {
                "kind": "message",
                "ts": str(message.get("created_at") or ""),
                "payload": message,
            }
        )
    for event in events:
        # Keep final assistant text as messages; show tool/agent activity as events.
        event_type = str(event.get("type") or "")
        if event_type in {"turn.finished", "assistant.delta"} and str(event.get("status")) == "success":
            # Still show completion as compact event; message holds full answer.
            pass
        rows.append(
            {
                "kind": "event",
                "ts": str(event.get("started_at") or event.get("timestamp") or ""),
                "payload": event,
            }
        )
    rows.sort(key=lambda item: item["ts"])
    return rows


def render_timeline(messages: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    st = _streamlit()
    timeline = merge_timeline(messages, events)
    if not timeline:
        st.info("No messages yet. Start a conversation.")
        return
    for row in timeline:
        if row["kind"] == "message":
            render_message(row["payload"])
        else:
            # Skip noisy pure turn.started duplicates of the user message content.
            event = row["payload"]
            if str(event.get("type") or "") == "turn.started":
                continue
            render_event(event)
