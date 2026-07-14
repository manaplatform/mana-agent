from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mana_agent.dashboard.components.chat_timeline import merge_timeline


def test_timeline_merge_orders_messages_and_events() -> None:
    messages = [
        {"role": "user", "content": "hi", "created_at": "2026-01-01T00:00:02"},
        {"role": "assistant", "content": "hello", "created_at": "2026-01-01T00:00:04"},
    ]
    events = [
        {"type": "tool.started", "kind": "tool", "title": "search", "started_at": "2026-01-01T00:00:03", "status": "running"},
        {"type": "turn.started", "kind": "user_request", "title": "user", "started_at": "2026-01-01T00:00:01", "status": "running"},
    ]
    timeline = merge_timeline(messages, events)
    assert [row["kind"] for row in timeline] == ["event", "message", "event", "message"]
    assert timeline[0]["payload"]["type"] == "turn.started"


def test_dashboard_page_modules_are_discoverable() -> None:
    """Page modules exist on the package path without importing Streamlit."""
    for name in ("overview", "chat", "analyze", "reports", "taskboard", "observability", "metrics", "automations", "cron"):
        spec = importlib.util.find_spec(f"mana_agent.dashboard.pages.{name}")
        assert spec is not None and spec.origin
        assert Path(spec.origin).name == f"{name}.py"


def test_packaged_dashboard_app_module_is_discoverable() -> None:
    spec = importlib.util.find_spec("mana_agent.dashboard.app")
    assert spec is not None and spec.origin
    assert Path(spec.origin).name == "app.py"


@pytest.mark.skipif(importlib.util.find_spec("streamlit") is None, reason="streamlit optional extra not installed")
def test_dashboard_pages_export_render_callables() -> None:
    from mana_agent.dashboard.pages import analyze, chat, overview

    assert callable(overview.render)
    assert callable(chat.render)
    assert callable(analyze.render)


@pytest.mark.skipif(importlib.util.find_spec("streamlit") is None, reason="streamlit optional extra not installed")
def test_packaged_dashboard_app_is_importable() -> None:
    import mana_agent.dashboard.app as app_module

    assert Path(app_module.__file__).name == "app.py"
