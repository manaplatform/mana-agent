from __future__ import annotations

import pytest
from mana_agent.context_cost.artifact_store import ContextArtifactStore, ArtifactAccessError
from mana_agent.services.operation_trace import operation_content, operation_detail


class Conversation:
    repository_id = "repo"

    def __init__(self, rows):
        self.rows = rows

    def get_or_raise(self, conversation_id):
        if conversation_id != "session":
            raise FileNotFoundError(conversation_id)

    def list_events(self, conversation_id, *, execution_id, limit=200, after_sequence=0, forward=False):
        rows = [r for r in self.rows if r["execution_id"] == execution_id and r["sequence"] > after_sequence]
        return rows[:limit] if forward or after_sequence else rows[-limit:]


def output(sequence, reference, **changes):
    return {"event_id": f"event-{sequence}", "sequence": sequence, "type": "tool.stdout", "status": "running",
        "task_id": "task", "execution_id": "execution", "operation_id": "operation", "operation_attempt_id": "invocation",
        "attempt_id": "supervisor-attempt", "agent_id": "worker", "output_preview": "retained output",
        "metadata": {"stream_channel": "stdout", "stream_offset": 0, "content_reference": reference,
            "content_length": 15, "trace_complete": True}, **changes}


def test_owned_content_is_paginated_and_survives_more_than_one_event_page(tmp_path, monkeypatch):
    store = ContextArtifactStore(tmp_path / "artifacts")
    reference = store.put("retained output", session_id="session", repository_id="repo", workspace_id="", content_type="execution/stdout")
    monkeypatch.setattr("mana_agent.services.operation_trace.ContextArtifactStore", lambda: store)
    rows = [output(1, reference.artifact_id)]
    rows.extend(output(i, "", operation_id=f"other-{i}") for i in range(2, 1205))
    service = Conversation(rows)
    first = operation_content(service, "session", "execution", "operation", "supervisor-attempt", channel="stdout", limit=8)
    second = operation_content(service, "session", "execution", "operation", "supervisor-attempt", channel="stdout", offset=first["next_offset"], limit=8)
    assert first["content"] + second["content"] == "retained output"
    assert first["has_more"] is True and second["has_more"] is False


def test_artifact_reference_cannot_cross_session_or_workspace(tmp_path, monkeypatch):
    store = ContextArtifactStore(tmp_path / "artifacts")
    reference = store.put("private output", session_id="other-session", repository_id="repo", workspace_id="", content_type="execution/stdout")
    monkeypatch.setattr("mana_agent.services.operation_trace.ContextArtifactStore", lambda: store)
    with pytest.raises(ArtifactAccessError):
        operation_content(Conversation([output(1, reference.artifact_id)]), "session", "execution", "operation", "supervisor-attempt", channel="stdout")


def test_missing_or_wrong_attempt_and_invalid_ranges_are_rejected():
    service = Conversation([output(1, "")])
    with pytest.raises(ValueError):
        operation_detail(service, "session", "execution", "operation", "foreign-attempt")
    with pytest.raises(ValueError):
        operation_content(service, "session", "execution", "operation", "supervisor-attempt", channel="stdout", limit=64001)
    with pytest.raises(FileNotFoundError):
        operation_detail(service, "foreign-session", "execution", "operation", "supervisor-attempt")


def test_detail_endpoint_requires_existing_bearer_guard(monkeypatch):
    from mana_agent.api.routes.conversations import get_operation
    from mana_agent.api.exceptions import ManaApiError
    monkeypatch.setenv("MANA_API_TOKEN", "test-token")
    with pytest.raises(ManaApiError):
        get_operation("session", "execution", "operation", authorization=None)
