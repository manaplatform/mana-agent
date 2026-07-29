"""Same-origin browser component for the A2UI Live Canvas workspace."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlencode


def live_canvas_html(
    *, conversation_id: str, root: Path, api_base: str, surface_id: str = "", height: int = 760
) -> str:
    script = Path(__file__).with_name("live_canvas.js").read_text(encoding="utf-8")
    config = {
        "mountId": "mana-live-canvas",
        "sessionId": conversation_id,
        "root": str(root),
        "surfaceId": surface_id,
        "apiBase": api_base.rstrip("/"),
        "wsBase": api_base.rstrip("/").replace("https://", "wss://").replace("http://", "ws://"),
        "token": str(os.getenv("MANA_API_TOKEN") or ""),
        "height": height,
    }
    safe = json.dumps(config, ensure_ascii=False).replace("</", "<\\/")
    return (
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"></head><body style="margin:0">'
        '<div id="mana-live-canvas"></div><script>' + script
        + f"\nManaLiveCanvas.init({safe});</script></body></html>"
    )


def render_live_canvas(
    *, conversation_id: str, root: Path, api_base: str, surface_id: str = "", height: int = 760
) -> None:
    import streamlit as st

    query = urlencode({
        "conversation_id": conversation_id, "root": str(root),
        "surface_id": surface_id, "height": height,
    })
    st.iframe(
        f"{api_base.rstrip('/')}/api/v1/dashboard/live-canvas?{query}",
        height=height + 4, width="stretch",
    )


__all__ = ["live_canvas_html", "render_live_canvas"]
