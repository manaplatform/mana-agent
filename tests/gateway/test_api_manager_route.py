from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mana_agent.api_manager.runtime_tools import API_MANAGER_TOOL_NAMES
from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.gateway.entry_routing import EntryRouteContext, EntryRoutingDecision


def test_api_route_uses_only_narrow_manager_tools(tmp_path: Path) -> None:
    class ModelToolExecutor:
        def run(self, *, question: str, system_prompt: str, tool_policy: dict, **kwargs):
            assert question == "Get contact 123 from Acme CRM."
            assert tool_policy["allowed_tools"] == list(API_MANAGER_TOOL_NAMES)
            assert tool_policy["disable_external_search"] is True
            assert "api_operations_search first" in system_prompt
            assert "Never claim an API call succeeded" in system_prompt
            assert kwargs["flow_id"] == "session-api"
            return SimpleNamespace(
                answer="Operation search requires one missing credential reference.",
                sources=[],
                warnings=[],
                tool_traces=[],
            )

    gateway = object.__new__(AgentChatGateway)
    gateway.root = tmp_path
    gateway._index_dir = None
    gateway._resolved_k = 4
    gateway._agent_timeout_seconds = 30
    gateway._event_sink = None
    gateway.config = SimpleNamespace(agent_max_steps=8)
    result = gateway._execute_api_route(
        decision=EntryRoutingDecision(
            route="api",
            confidence=0.99,
            reason="Use the saved integration.",
            required_sources=("api",),
        ),
        context=EntryRouteContext(
            session_id="session-api",
            conversation_id="session-api",
            turn_id="turn-api",
        ),
        text="Get contact 123 from Acme CRM.",
        ask_service=SimpleNamespace(ask_agent=ModelToolExecutor()),
        callbacks=None,
    )
    assert result.mode == "route-api"
    assert result.payload["route"] == "api"

