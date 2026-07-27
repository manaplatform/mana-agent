from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mana_agent.api.app import create_app
from mana_agent.services.conversation_service import ConversationService
from mana_agent.services.execution_event_hub import get_execution_event_hub, reset_execution_event_hub_for_tests


@pytest.fixture()
def setup_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    monkeypatch.delenv("MANA_API_TOKEN", raising=False)
    reset_execution_event_hub_for_tests()
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("# demo\n", encoding="utf-8")
    monkeypatch.setenv("MANA_DASHBOARD_ROOT", str(root))
    return root


def test_websocket_connect_replay_and_live_event(setup_env: Path) -> None:
    root = setup_env
    service = ConversationService(root=root)
    conv = service.create(title="WS")
    hub = get_execution_event_hub()
    hub.emit(
        "tool.started",
        title="repo_search",
        conversation_id=conv.conversation_id,
        execution_id="exec_ws",
        repository_id=service.repository_id,
        message="seed",
        status="running",
    )
    client = TestClient(create_app())
    with client.websocket_connect(
        f"/api/v1/ws/conversations/{conv.conversation_id}?root={root}&replay_limit=50"
    ) as ws:
        ready = ws.receive_json()
        assert ready["type"] == "socket.ready"
        assert ready["conversation_id"] == conv.conversation_id
        # Replay at least the seeded event.
        seen_replay = False
        for _ in range(5):
            msg = ws.receive_json()
            if msg.get("type") == "event.replay":
                seen_replay = True
                assert msg["event"]["conversation_id"] == conv.conversation_id
            if msg.get("type") == "socket.replay_complete":
                break
        assert seen_replay
        hub.emit(
            "tool.finished",
            title="repo_search",
            conversation_id=conv.conversation_id,
            execution_id="exec_ws",
            repository_id=service.repository_id,
            message="done",
            status="success",
        )
        live = ws.receive_json()
        assert live["type"] == "event"
        assert live["event"]["type"] == "tool.finished"
        assert live["event"]["conversation_id"] == conv.conversation_id
        ws.send_text("ping")
        pong = ws.receive_json()
        assert pong["type"] == "pong"


def test_websocket_missing_conversation(setup_env: Path) -> None:
    client = TestClient(create_app())
    with client.websocket_connect(f"/api/v1/ws/conversations/conv_missing000000?root={setup_env}") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_websocket_cursor_replays_only_missed_events(setup_env: Path) -> None:
    service = ConversationService(root=setup_env)
    conversation = service.create(title="Resume")
    hub = get_execution_event_hub()
    first = hub.emit(
        "log.info",
        title="first",
        conversation_id=conversation.conversation_id,
        repository_id=service.repository_id,
        status="success",
    )
    second = hub.emit(
        "log.info",
        title="second",
        conversation_id=conversation.conversation_id,
        repository_id=service.repository_id,
        status="success",
    )
    client = TestClient(create_app())
    with client.websocket_connect(
        f"/api/v1/ws/conversations/{conversation.conversation_id}"
        f"?root={setup_env}&after_sequence={first['sequence']}&replay_limit=50"
    ) as ws:
        assert ws.receive_json()["type"] == "socket.ready"
        replay = ws.receive_json()
        assert replay["type"] == "event.replay"
        assert replay["event"]["sequence"] == second["sequence"]
        complete = ws.receive_json()
        assert complete["type"] == "socket.replay_complete"
        assert complete["last_sequence"] == second["sequence"]
