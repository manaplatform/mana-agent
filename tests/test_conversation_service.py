from __future__ import annotations

from pathlib import Path

import pytest

from mana_agent.services.conversation_service import ConversationService
from mana_agent.services.execution_event_hub import reset_execution_event_hub_for_tests


@pytest.fixture()
def conv_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    reset_execution_event_hub_for_tests()
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("# demo\n", encoding="utf-8")
    return root


def test_create_list_and_load_conversation(conv_root: Path) -> None:
    service = ConversationService(root=conv_root)
    created = service.create(title="First")
    assert created.conversation_id.startswith("session_")
    listed = service.list()
    assert any(item.conversation_id == created.conversation_id for item in listed)
    loaded = service.get(created.conversation_id)
    assert loaded.title == "First"
    assert loaded.repository_id == service.repository_id


def test_message_history_and_send(conv_root: Path) -> None:
    service = ConversationService(root=conv_root)

    def fake_chat(prompt: str, **kwargs):  # noqa: ANN001
        sink = kwargs.get("event_sink")
        if callable(sink):
            sink("tool.started", "repo_search", message="q", status="running", metadata={"tool_name": "repo_search"})
            sink("tool.finished", "repo_search", message="ok", status="success", metadata={"tool_name": "repo_search"})
        return {"answer": f"echo:{prompt}", "mode": "preview", "sources": []}

    conv = service.create(title="Chat")
    result = service.send_message(conv.conversation_id, "hello world", chat_runner=fake_chat)
    assert result["ok"] is True
    assert result["assistant_message"]["content"] == "echo:hello world"
    messages = service.list_messages(conv.conversation_id)
    assert [m.role for m in messages] == ["user", "assistant"]
    events = service.list_events(conv.conversation_id, execution_id=result["execution_id"])
    types = {e.get("type") for e in events}
    assert "turn.started" in types
    assert "tool.started" in types
    assert "turn.finished" in types
    full = service.get_full(conv.conversation_id)
    assert full["conversation"]["message_count"] == 2


def test_new_command_replaces_conversation_before_persisting_message(conv_root: Path) -> None:
    service = ConversationService(root=conv_root)
    current = service.create(title="Old chat")
    service.append_message(current.conversation_id, role="user", content="private history")

    result = service.send_message(current.conversation_id, "/new")

    replacement_id = result["conversation_id"]
    assert result["ok"] is True
    assert replacement_id != current.conversation_id
    assert result["command_result"]["data"]["session_id"] == replacement_id
    assert service.list_messages(replacement_id) == []
    with pytest.raises(FileNotFoundError):
        service.get(current.conversation_id)
    assert current.conversation_id not in {
        item.conversation_id for item in service.list()
    }


def test_event_routing_isolates_conversations(conv_root: Path) -> None:
    service = ConversationService(root=conv_root)
    a = service.create(title="A")
    b = service.create(title="B")
    hub = reset_execution_event_hub_for_tests()
    hub.emit(
        "tool.started",
        title="read_file",
        conversation_id=a.conversation_id,
        execution_id="exec_a",
        repository_id=service.repository_id,
        message="a.py",
        status="running",
    )
    hub.emit(
        "tool.started",
        title="read_file",
        conversation_id=b.conversation_id,
        execution_id="exec_b",
        repository_id=service.repository_id,
        message="b.py",
        status="running",
    )
    events_a = hub.history(conversation_id=a.conversation_id, repository_id=service.repository_id)
    events_b = hub.history(conversation_id=b.conversation_id, repository_id=service.repository_id)
    assert all(e["conversation_id"] == a.conversation_id for e in events_a)
    assert all(e["conversation_id"] == b.conversation_id for e in events_b)
    assert hub.history(conversation_id=a.conversation_id, execution_id="exec_b", repository_id=service.repository_id) == []


def test_client_message_id_is_persisted_and_retry_is_idempotent(conv_root: Path) -> None:
    service = ConversationService(root=conv_root)
    conversation = service.create(title="Optimistic")
    calls: list[str] = []

    def fake_chat(prompt: str, **_kwargs):  # noqa: ANN001
        calls.append(prompt)
        return {"answer": "done", "mode": "test", "sources": []}

    first = service.send_message(
        conversation.conversation_id,
        "hello",
        chat_runner=fake_chat,
        client_message_id="client_stable_1",
    )
    second = service.send_message(
        conversation.conversation_id,
        "hello",
        chat_runner=fake_chat,
        client_message_id="client_stable_1",
    )

    assert first["user_message"]["message_id"] == "client_stable_1"
    assert second["duplicate"] is True
    assert calls == ["hello"]
    assert [row.message_id for row in service.list_messages(conversation.conversation_id)].count(
        "client_stable_1"
    ) == 1
