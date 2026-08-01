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
