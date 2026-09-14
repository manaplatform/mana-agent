"""Tests for conversation attachments API endpoints."""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from mana_agent.api.app import create_app
from mana_agent.services.execution_event_hub import reset_execution_event_hub_for_tests


MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, str]:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    monkeypatch.delenv("MANA_API_TOKEN", raising=False)
    reset_execution_event_hub_for_tests()
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    monkeypatch.setenv("MANA_DASHBOARD_ROOT", str(root))
    return TestClient(create_app()), str(root)


def test_attachment_upload_and_download(client: tuple[TestClient, str]) -> None:
    api, root = client

    # Create conversation
    res = api.post("/api/v1/conversations", json={"title": "Attachment Test", "root": root})
    assert res.status_code == 201
    conversation_id = res.json()["conversation"]["conversation_id"]

    # Upload attachment
    files = {"file": ("test_diagram.png", io.BytesIO(MINIMAL_PNG), "image/png")}
    upload_res = api.post(
        f"/api/v1/conversations/{conversation_id}/attachments",
        params={"root": root},
        files=files,
    )
    assert upload_res.status_code == 201
    data = upload_res.json()
    assert data["ok"] is True
    att = data["attachment"]
    assert att["filename"] == "test_diagram.png"
    assert att["category"] == "image"
    attachment_id = att["attachment_id"]

    # Download attachment
    download_res = api.get(
        f"/api/v1/conversations/{conversation_id}/attachments/{attachment_id}",
        params={"root": root},
    )
    assert download_res.status_code == 200
    assert download_res.content == MINIMAL_PNG
    assert download_res.headers["content-type"] == "image/png"


def test_send_message_with_attachments_and_empty_text(
    client: tuple[TestClient, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    api, root = client

    # Create conversation
    res = api.post("/api/v1/conversations", json={"title": "Chat With Attachment", "root": root})
    assert res.status_code == 201
    conversation_id = res.json()["conversation"]["conversation_id"]

    # Mock send_message runner on ConversationService to simulate execution
    from mana_agent.services import conversation_service as cs

    recorded_calls = []

    def mock_send_message(self, conv_id, content, **kwargs):  # noqa: ANN001
        recorded_calls.append({"conv_id": conv_id, "content": content, "kwargs": kwargs})
        attachments = kwargs.get("attachments") or []
        self.append_message(
            conv_id,
            role="user",
            content=content,
            execution_id="exec_1",
            attachments=attachments,
        )
        self.append_message(
            conv_id,
            role="assistant",
            content="Received message with attachment",
            execution_id="exec_1",
        )
        self.set_status(conv_id, "idle", execution_id="exec_1")
        return {
            "conversation_id": conv_id,
            "status": "idle",
            "user_message": {"role": "user", "content": content, "attachments": attachments},
            "assistant_message": {"role": "assistant", "content": "Received message with attachment"},
            "events": [],
        }

    monkeypatch.setattr(cs.ConversationService, "send_message", mock_send_message)

    att_payload = {
        "attachment_id": "att_123",
        "filename": "chart.png",
        "mime_type": "image/png",
        "size_bytes": len(MINIMAL_PNG),
        "category": "image",
    }

    # 1. Send message with empty content and attachment
    send_res = api.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "content": "",
            "attachments": [att_payload],
            "root": root,
        },
    )
    assert send_res.status_code == 201
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["content"] == ""
    assert recorded_calls[0]["kwargs"]["attachments"] == [att_payload]

    # 2. Send message with both content and attachment
    send_res2 = api.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "content": "Analyze this chart please",
            "attachments": [att_payload],
            "root": root,
        },
    )
    assert send_res2.status_code == 201
    assert len(recorded_calls) == 2
    assert recorded_calls[1]["content"] == "Analyze this chart please"

    # 3. Send message with neither content nor attachments -> 422 rejected
    bad_res = api.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={
            "content": "   ",
            "attachments": [],
            "root": root,
        },
    )
    assert bad_res.status_code == 422

    # 4. Verify conversation messages list includes attachments
    detail = api.get(f"/api/v1/conversations/{conversation_id}", params={"root": root})
    assert detail.status_code == 200
    messages = detail.json()["messages"]
    user_msgs = [m for m in messages if m["role"] == "user"]
    assert len(user_msgs) == 2
    assert user_msgs[0]["attachments"][0]["filename"] == "chart.png"
