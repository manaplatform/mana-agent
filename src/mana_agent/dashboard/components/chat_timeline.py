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
    raw_status = str(event.get("status") or "running")
    status = raw_status
    icon = _EVENT_ICONS.get(kind, "•")
    title = str(event.get("title") or event.get("type") or kind)
    summary = str(event.get("summary") or event.get("message") or "")
    meta = event.get("metadata") or event.get("details") or {}
    duration = event.get("duration_ms")
    duration_txt = f" · {duration}ms" if duration not in (None, "") else ""

    # Special compact live-style rendering for tool activity items.
    # Uses the same event_id correlation as CLI so start updates to finished in place.
    if kind == "tool" or str(event.get("type") or "").startswith("tool."):
        tool_name = str(meta.get("tool_name") or title or "tool")
        action = str(meta.get("args_summary") or meta.get("result_summary") or summary or "").strip()
        if raw_status == "running":
            spinner = "⏳"
            label = f"{spinner} **{tool_name}** · running"
            if action:
                label += f" — {action[:80]}"
            st.markdown(label)
            return
        fin_icon = "✅" if raw_status in {"success", "done"} else "❌"
        dur = f" ({duration}ms)" if duration else ""
        label = f"{fin_icon} **{tool_name}**{dur}"
        if action:
            label += f" — {action[:100]}"
        st.markdown(label)
        if summary and summary != action:
            st.caption(summary[:120])
        return

    label = f"{icon} **{title}** · `{status}`{duration_txt}"
    if compact and not summary and not meta:
        st.markdown(label)
        return
    with st.expander(f"{icon} {title} · {status}{duration_txt}", expanded=False):
        if summary:
            st.write(summary)
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
    """Interleave messages and events by timestamp for ordered rendering.

    For tool.* events, collapse by event_id keeping the latest status (supports
    start + in-place terminal update over the shared event/socket architecture).
    """
    rows: list[dict[str, Any]] = []
    for message in messages:
        rows.append(
            {
                "kind": "message",
                "ts": str(message.get("created_at") or ""),
                "payload": message,
            }
        )

    # Collapse tool events by event_id (latest wins) while preserving first-seen ts order.
    tool_latest: dict[str, dict[str, Any]] = {}
    tool_first_ts: dict[str, str] = {}
    other_events: list[dict[str, Any]] = []

    for event in events:
        et = str(event.get("type") or "")
        if et.startswith("tool."):
            meta = event.get("metadata") or event.get("details") or {}
            eid = str(
                event.get("tool_call_id")
                or meta.get("tool_call_id")
                or event.get("event_id")
                or event.get("id")
                or ""
            ).strip()
            ts = str(event.get("started_at") or event.get("timestamp") or "")
            if eid:
                if eid not in tool_first_ts:
                    tool_first_ts[eid] = ts
                tool_latest[eid] = event  # overwrite with later (terminal) version
            else:
                other_events.append(event)
        else:
            other_events.append(event)

    for event in other_events:
        event_type = str(event.get("type") or "")
        if event_type in {"turn.finished", "assistant.delta"} and str(event.get("status")) == "success":
            pass
        rows.append(
            {
                "kind": "event",
                "ts": str(event.get("started_at") or event.get("timestamp") or ""),
                "payload": event,
            }
        )

    for eid, ev in tool_latest.items():
        ts = tool_first_ts.get(eid, str(ev.get("started_at") or ev.get("timestamp") or ""))
        rows.append({"kind": "event", "ts": ts, "payload": ev})

    rows.sort(
        key=lambda item: (
            item["ts"],
            int(item["payload"].get("sequence") or 0),
        )
    )
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
