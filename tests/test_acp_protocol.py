from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("acp")

from mana_agent.chat.events import AssistantMessageEvent, ToolCallEvent, ToolResultEvent
from mana_agent.gateway.turn_engine import ChatTurnResult
from mana_agent.services.chat_session_history import ChatSessionHistory
from mana_agent.workspaces.service import WorkspaceService
from mana_agent.protocols.acp.agent import ManaAcpAgent
from mana_agent.protocols.acp.permissions import AcpPermissionBroker
from mana_agent.protocols.acp.event_mapper import AcpEventMapper
from mana_agent.protocols.acp.server import acp_sdk_info


def test_acp_sdk_and_stable_event_ids() -> None:
    assert acp_sdk_info()["protocol_version"] == "1"
    mapper = AcpEventMapper()
    start = mapper.map(ToolCallEvent(tool_name="read_file", call_id="call-stable"))[0]
    finish = mapper.map(ToolResultEvent(call_id="call-stable", tool_name="read_file", success=True))[0]
    assert start.tool_call_id == finish.tool_call_id == "call-stable"
    message = mapper.map(AssistantMessageEvent(content="done"))[0]
    assert message.content.text == "done"


class _Gateway:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._workspaces = WorkspaceService()
        self.history = ChatSessionHistory()
        self.ask = type("Ask", (), {"mcp_server_overrides": []})()
        self.closed: list[str] = []

    def create_new_session(self, *, frontend: str) -> str:
        assert frontend == "acp"
        return self._workspaces.create_session(self.root).session_id

    def create_session(self, *, frontend: str, session_id: str) -> str:
        assert frontend == "acp"
        self._workspaces.store.get_session(session_id)
        return session_id

    def close_session(self, session_id: str) -> None:
        self._workspaces.close_session(session_id)
        self.closed.append(session_id)

    def session_messages(self, session_id: str) -> list[dict]:
        return [item.to_dict() for item in self.history.list(session_id)]

    def get_ask_service(self):
        return self.ask

    async def process_turn_async(self, session_id: str, text: str, **kwargs) -> ChatTurnResult:
        assert kwargs["protocol_read_only"] is False
        self.history.append(session_id, role="user", content=text, turn_id="turn-1")
        self.history.append(session_id, role="assistant", content="answer", turn_id="turn-1")
        return ChatTurnResult(answer="answer", mode="test")

    def cancel(self, session_id: str) -> bool:
        return bool(session_id)


class _Client:
    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []

    async def session_update(self, session_id: str, update: object) -> None:
        self.updates.append((session_id, update))


def test_acp_sessions_prompt_and_history_replay_are_isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    root = tmp_path / "repo"
    root.mkdir()
    gateway = _Gateway(root)
    agent = ManaAcpAgent(gateway)  # type: ignore[arg-type]
    client = _Client()
    agent.on_connect(client)

    async def scenario() -> None:
        from acp.schema import TextContentBlock

        initialized = await agent.initialize(1)
        assert initialized.protocol_version == 1
        assert initialized.agent_capabilities.session_capabilities.close is not None
        first = await agent.new_session(str(root), mcp_servers=[])
        second = await agent.new_session(str(root), mcp_servers=[])
        assert first.session_id != second.session_id
        response = await agent.prompt(first.session_id, [TextContentBlock(type="text", text="hello")])
        assert response.stop_reason == "end_turn"
        client.updates.clear()
        await agent.load_session(str(root), first.session_id, mcp_servers=[])
        replay_types = [getattr(update, "session_update", "") for _, update in client.updates]
        assert "user_message_chunk" in replay_types
        assert "agent_message_chunk" in replay_types
        assert all(session_id == first.session_id for session_id, _ in client.updates)
        await agent.close_session(first.session_id)
        await agent.close_session(first.session_id)

    asyncio.run(scenario())
    assert gateway.closed.count(agent.sessions.get(next(iter(agent.sessions.sessions))).mana_session_id) == 1


def test_acp_permission_broker_is_fail_closed_and_scopes_session_grants() -> None:
    from acp.schema import AllowedOutcome, DeniedOutcome, RequestPermissionResponse

    class PermissionClient:
        responses = [
            RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", option_id="allow-session")),
            RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled")),
        ]
        calls = 0

        async def request_permission(self, **kwargs):
            _ = kwargs
            self.calls += 1
            return self.responses.pop(0)

    client = PermissionClient()
    broker = AcpPermissionBroker(client)

    async def scenario() -> None:
        assert await broker.request(session_id="one", call_id="1", title="Edit", tool_name="edit") is True
        assert await broker.request(session_id="one", call_id="2", title="Edit", tool_name="edit") is True
        assert client.calls == 1
        assert await broker.request(session_id="two", call_id="3", title="Edit", tool_name="edit") is False

    asyncio.run(scenario())
