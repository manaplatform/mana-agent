from pathlib import Path

from mana_agent.gateway.chat_turn_store import ChatTurnStore


def test_message_id_is_the_durable_turn_idempotency_boundary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    store = ChatTurnStore("session_turns")

    created, duplicate = store.create_or_get(
        conversation_id="session_turns",
        user_message_id="msg_1",
        turn_id="turn_1",
        text="add dashboard support",
    )
    replayed, replay_duplicate = store.create_or_get(
        conversation_id="session_turns",
        user_message_id="msg_1",
        turn_id="turn_ignored",
        text="add dashboard support",
    )
    next_turn, next_duplicate = store.create_or_get(
        conversation_id="session_turns",
        user_message_id="msg_2",
        turn_id="turn_2",
        text="find AI agent jobs",
    )

    assert duplicate is False
    assert replay_duplicate is True
    assert replayed.turn_id == created.turn_id
    assert next_duplicate is False
    assert next_turn.turn_id == "turn_2"


def test_chat_turn_store_persists_records_with_sets_in_response_and_trace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    store = ChatTurnStore("session_with_sets")

    record, duplicate = store.create_or_get(
        conversation_id="session_with_sets",
        user_message_id="msg_set_1",
        turn_id="turn_set_1",
        text="check email",
    )
    assert duplicate is False

    # Attach response payload containing sets in trace, labels, and result fields
    record.response = {
        "answer": "Here are your emails",
        "trace": [
            {
                "tool_name": "email_search",
                "result": {
                    "labels": {"INBOX", "UNREAD"},
                    "tags": frozenset({"important", "work"}),
                },
                "status": "ok",
            }
        ],
        "metadata": {"permission_set": {"email.read", "email.metadata"}},
    }

    updated = store.update(record)
    assert updated.turn_id == "turn_set_1"

    # Reload from disk and verify clean serialization without TypeError
    reloaded_records = store._load()
    reloaded = reloaded_records.get("session_with_sets:msg_set_1")
    assert reloaded is not None
    assert reloaded["response"]["trace"][0]["result"]["labels"] == ["INBOX", "UNREAD"]
    assert reloaded["response"]["trace"][0]["result"]["tags"] == ["important", "work"]
    assert sorted(reloaded["response"]["metadata"]["permission_set"]) == ["email.metadata", "email.read"]

