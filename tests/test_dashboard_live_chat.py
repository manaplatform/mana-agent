from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from mana_agent.dashboard.components.chat_timeline import merge_timeline
from mana_agent.services.execution_event_hub import ExecutionEventHub


def test_correlated_tool_revisions_persist_and_merge_as_one_card(tmp_path: Path) -> None:
    hub = ExecutionEventHub()
    common = {
        "conversation_id": "session_chat",
        "repository_id": "repo_chat",
        "execution_id": "run-1",
        "event_id": "tool-call-1",
        "metadata": {"tool_call_id": "tool-call-1", "tool_name": "search"},
    }
    started = hub.publish(
        {**common, "type": "tool.started", "status": "running", "summary": "query"},
        persist=False,
    )
    completed = hub.publish(
        {**common, "type": "tool.finished", "status": "success", "summary": "done"},
        persist=False,
    )
    history = hub.history(conversation_id="session_chat", repository_id="repo_chat")
    assert [row["type"] for row in history] == ["tool.started", "tool.finished"]
    assert completed["sequence"] > started["sequence"]
    merged = merge_timeline([], history)
    assert len(merged) == 1
    assert merged[0]["payload"]["status"] == "success"


def test_event_history_cursor_and_exact_duplicate_delivery() -> None:
    hub = ExecutionEventHub()
    received: list[dict] = []
    hub.subscribe("session_chat", received.append)
    first = hub.publish(
        {"event_id": "one", "type": "log.info", "summary": "one"},
        conversation_id="session_chat",
        persist=False,
    )
    duplicate = hub.publish(
        {"event_id": "one", "type": "log.info", "summary": "one"},
        conversation_id="session_chat",
        persist=False,
    )
    second = hub.publish(
        {"event_id": "two", "type": "log.info", "summary": "two"},
        conversation_id="session_chat",
        persist=False,
    )
    assert duplicate["sequence"] == first["sequence"]
    assert len(received) == 2
    assert hub.history(
        conversation_id="session_chat",
        after_sequence=first["sequence"],
    ) == [second]


def test_event_payload_redacts_secret_fields_and_free_text_tokens(tmp_path: Path) -> None:
    hub = ExecutionEventHub()
    payload = hub.publish(
        {
            "event_id": "secret-event",
            "type": "tool.started",
            "summary": "Authorization: Bearer abc.def",
            "metadata": {
                "authorization": "Bearer abc.def",
                "api_key": "sk-abcdefghijklmnopqrstuvwxyz",
            },
        },
        conversation_id="session_chat",
        repository_id="repo_chat",
        persist=False,
    )
    encoded = json.dumps(payload)
    assert "abc.def" not in encoded
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in encoded
    assert "REDACTED" in encoded


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_browser_reducer_suite() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["node", "--test", "tests/dashboard/live_chat_reducer.test.mjs"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
