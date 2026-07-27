"""Browser-side live chat component backed by the shared REST/WebSocket API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


def live_chat_html(
    *,
    conversation_id: str,
    root: Path,
    api_base: str,
    messages: list[dict[str, Any]],
    events: list[dict[str, Any]],
    height: int = 680,
) -> str:
    """Build the self-contained same-origin live chat document."""
    script = Path(__file__).with_name("live_chat.js").read_text(encoding="utf-8")
    config = {
        "mountId": "mana-live-chat",
        "sessionId": conversation_id,
        "root": str(root),
        "apiBase": api_base.rstrip("/"),
        "wsBase": api_base.rstrip("/").replace("https://", "wss://").replace("http://", "ws://"),
        "token": str(os.getenv("MANA_DASHBOARD_API_TOKEN") or ""),
        "messages": messages,
        "events": events,
        "height": height,
    }
    safe_config = json.dumps(config, ensure_ascii=False).replace("</", "<\\/")
    return (
        '<!doctype html><html><head><meta charset="utf-8"></head>'
        '<body style="margin:0"><div id="mana-live-chat"></div><script>'
        + script
        + f"\nManaLiveChat.init({safe_config});</script></body></html>"
    )


def render_live_chat(
    *,
    conversation_id: str,
    root: Path,
    api_base: str,
    messages: list[dict[str, Any]],
    events: list[dict[str, Any]],
    height: int = 680,
) -> None:
    import streamlit as st

    del messages, events  # The same-origin endpoint hydrates fresh persisted state.
    query = urlencode(
        {
            "conversation_id": conversation_id,
            "root": str(root),
            "height": height,
        }
    )
    st.iframe(
        f"{api_base.rstrip('/')}/api/v1/dashboard/live-chat?{query}",
        height=height + 4,
        width="stretch",
    )


__all__ = ["live_chat_html", "render_live_chat"]
