from __future__ import annotations

from types import SimpleNamespace

import pytest

from mana_agent.coding.live_events import coding_event_scope
from mana_agent.coding.models import AgentEvent
from mana_agent.coding.selection import (
    CodingBackendConfigurationError,
    resolve_coding_backend,
)
from mana_agent.integrations.codex.event_adapter import adapt_codex_event
from mana_agent.services.execution_event_hub import ExecutionEventHub


def test_legacy_backend_selection_uses_codex_only_while_enabled() -> None:
    assert resolve_coding_backend(SimpleNamespace(mana_codex_enabled=True)).backend == "codex"
    assert resolve_coding_backend(SimpleNamespace(mana_codex_enabled=False)).backend == "internal"


def test_explicit_internal_backend_does_not_require_codex() -> None:
    selection = resolve_coding_backend(SimpleNamespace(
        mana_coding_backend="internal",
        mana_codex_enabled=False,
    ))
    assert selection.backend == "internal"
    assert selection.source == "explicit"


def test_explicit_disabled_codex_configuration_is_rejected() -> None:
    with pytest.raises(CodingBackendConfigurationError, match="MANA_CODEX_ENABLED is false"):
        resolve_coding_backend(SimpleNamespace(
            mana_coding_backend="codex",
            mana_codex_enabled=False,
        ))


def test_codex_event_normalizes_usage_and_redacts_secrets() -> None:
    event = adapt_codex_event(
        "task-1",
        {
            "id": "provider-1",
            "method": "turn/completed",
            "params": {
                "turn": {"id": "turn-1"},
                "usage": {"inputTokens": 10, "cachedInputTokens": 4},
                "authorization": "Bearer top-secret",
            },
        },
        sequence=9,
        model="gpt-test",
    )
    assert event.event_id == "codex-provider-1"
    assert event.sequence == 9
    assert event.token_usage == {"input_tokens": 10, "cached_tokens": 4}
    assert event.payload["authorization"] == "[REDACTED]"


def test_event_hub_ignores_duplicate_ids_for_live_subscribers() -> None:
    hub = ExecutionEventHub()
    received: list[dict] = []
    hub.subscribe("conversation-1", received.append)
    event = {
        "event_id": "same-id",
        "type": "tool.started",
        "conversation_id": "conversation-1",
        "sequence": 1,
    }
    hub.publish(event, conversation_id="conversation-1", persist=False)
    hub.publish(event, conversation_id="conversation-1", persist=False)
    assert len(received) == 1


def test_scoped_live_events_are_isolated() -> None:
    first: list[str] = []
    second: list[str] = []
    event = AgentEvent(event_type="turn.started", task_id="task", backend="internal")
    from mana_agent.coding.live_events import publish_coding_event

    with coding_event_scope(lambda item: first.append(item.task_id)):
        publish_coding_event(event)
    with coding_event_scope(lambda item: second.append(item.task_id)):
        publish_coding_event(event)
    assert first == ["task"]
    assert second == ["task"]
