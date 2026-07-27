from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mana_agent.api.app import create_app
from mana_agent.services.execution_event_hub import reset_execution_event_hub_for_tests


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    monkeypatch.delenv("MANA_API_TOKEN", raising=False)
    reset_execution_event_hub_for_tests()
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    monkeypatch.setenv("MANA_DASHBOARD_ROOT", str(root))
    return TestClient(create_app())


def test_conversation_create_list_get_and_message(client: TestClient, tmp_path: Path) -> None:
    root = str(tmp_path / "repo")
    created = client.post("/api/v1/conversations", json={"title": "Demo", "root": root})
    assert created.status_code == 201
    conversation_id = created.json()["conversation"]["conversation_id"]

    listed = client.get("/api/v1/conversations", params={"root": root})
    assert listed.status_code == 200
    assert any(item["conversation_id"] == conversation_id for item in listed.json()["conversations"])

    detail = client.get(f"/api/v1/conversations/{conversation_id}", params={"root": root})
    assert detail.status_code == 200
    assert detail.json()["conversation"]["title"] == "Demo"

    # Patch chat runner by overriding ConversationService.send_message path via monkeypatch on helper
    from mana_agent.services import conversation_service as cs

    def fake_send(self, conversation_id, content, **kwargs):  # noqa: ANN001
        self.append_message(conversation_id, role="user", content=content, execution_id="exec_test")
        self.append_message(conversation_id, role="assistant", content=f"got:{content}", execution_id="exec_test")
        self.set_status(conversation_id, "idle", execution_id="exec_test")
        return {
            "ok": True,
            "conversation_id": conversation_id,
            "execution_id": "exec_test",
            "user_message": {"role": "user", "content": content},
            "assistant_message": {"role": "assistant", "content": f"got:{content}"},
            "result": {"answer": f"got:{content}", "mode": "preview"},
            "events": [],
        }

    # Use dependency-free path: call messages endpoint with patched send_message
    import mana_agent.api.routes.conversations as routes

    original = cs.ConversationService.send_message
    cs.ConversationService.send_message = fake_send  # type: ignore[method-assign]
    try:
        sent = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={"content": "hello", "root": root},
        )
        assert sent.status_code == 201
        assert sent.json()["assistant_message"]["content"] == "got:hello"
    finally:
        cs.ConversationService.send_message = original  # type: ignore[method-assign]

    history = client.get(f"/api/v1/conversations/{conversation_id}/messages", params={"root": root})
    assert history.status_code == 200
    roles = [m["role"] for m in history.json()["messages"]]
    assert roles == ["user", "assistant"]


def test_missing_conversation_returns_404(client: TestClient, tmp_path: Path) -> None:
    root = str(tmp_path / "repo")
    response = client.get("/api/v1/conversations/conv_doesnotexist123", params={"root": root})
    assert response.status_code == 404


def test_live_chat_document_hydrates_same_origin_reducer(
    client: TestClient,
    tmp_path: Path,
) -> None:
    root = str(tmp_path / "repo")
    created = client.post(
        "/api/v1/conversations",
        json={"title": "Live", "root": root},
    ).json()
    conversation_id = created["conversation"]["conversation_id"]
    response = client.get(
        "/api/v1/dashboard/live-chat",
        params={"conversation_id": conversation_id, "root": root},
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "Content-Security-Policy" in response.headers
    assert "ManaLiveChat.init" in response.text
    assert conversation_id in response.text
    assert "/api/v1/ws/conversations/" in response.text


def test_event_serialization_shape(client: TestClient, tmp_path: Path) -> None:
    root = str(tmp_path / "repo")
    created = client.post("/api/v1/conversations", json={"title": "Events", "root": root}).json()
    conversation_id = created["conversation"]["conversation_id"]
    from mana_agent.services.execution_event_hub import get_execution_event_hub
    from mana_agent.services.conversation_service import ConversationService

    service = ConversationService(root=root)
    hub = get_execution_event_hub()
    hub.emit(
        "agent.routing",
        title="Routing",
        conversation_id=conversation_id,
        execution_id="exec_1",
        repository_id=service.repository_id,
        message="decision",
        status="running",
    )
    events = client.get(
        f"/api/v1/conversations/{conversation_id}/events",
        params={"root": root, "execution_id": "exec_1"},
    )
    assert events.status_code == 200
    payload = events.json()["events"]
    assert payload
    event = payload[0]
    for key in ("event_id", "type", "kind", "status", "conversation_id", "execution_id", "started_at"):
        assert key in event
    assert event["conversation_id"] == conversation_id
    assert event["execution_id"] == "exec_1"
