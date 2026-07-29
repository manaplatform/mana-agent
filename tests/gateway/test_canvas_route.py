"""Canvas route completion-contract coverage."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mana_agent.canvas.models import RendererAction
from mana_agent.canvas.runtime_tools import build_canvas_langchain_tools
from mana_agent.canvas.service import canvas_service_for_root
from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.gateway.entry_routing import EntryRouteContext, EntryRoutingDecision


def _gateway(root: Path) -> AgentChatGateway:
    gateway = object.__new__(AgentChatGateway)
    gateway.root = root
    gateway._index_dir = None
    gateway._resolved_k = 4
    gateway._agent_timeout_seconds = 30
    gateway.config = SimpleNamespace(agent_max_steps=8)
    return gateway


def _decision() -> EntryRoutingDecision:
    return EntryRoutingDecision(
        route="canvas",
        confidence=0.99,
        reason="The user requested a visual workspace.",
        required_sources=("canvas",),
    )


def _context() -> EntryRouteContext:
    return EntryRouteContext(
        session_id="session-canvas",
        conversation_id="session-canvas",
        turn_id="turn-canvas",
    )


def test_canvas_route_creates_complete_surface_atomically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    tools = {item.name: item for item in build_canvas_langchain_tools(tmp_path)}

    class Executor:
        calls = 0

        def run(self, *, question: str, tool_policy: dict, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                assert "canvas_create_surface" in tool_policy["allowed_tools"]
                tools["canvas_create_surface"].invoke(
                    {
                        "source_decision_id": "turn-canvas",
                        "session_id": "session-canvas",
                        "conversation_id": "session-canvas",
                        "surface_id": "pet",
                        "owner": {"agent_id": "main", "task_id": "turn-canvas"},
                        "components": [
                            {
                                "id": "root",
                                "type": "Column",
                                "children": ["title"],
                            },
                            {"id": "title", "type": "Heading", "text": "Blue pet"},
                        ],
                        "data_model": {},
                    }
                )
            return SimpleNamespace(answer="Canvas updated.", sources=[], warnings=[])

    executor = Executor()
    result = _gateway(tmp_path)._execute_canvas_route(
        decision=_decision(),
        context=_context(),
        text="Create a blue pet.",
        ask_service=SimpleNamespace(ask_agent=executor),
        callbacks=None,
    )

    assert result.mode == "route-canvas"
    assert executor.calls == 1
    snapshot = canvas_service_for_root(tmp_path).get_surface("session-canvas", "pet")
    assert snapshot.components[0].id == "root"


def test_canvas_route_rolls_back_surface_when_model_correction_stays_incomplete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    class Executor:
        calls = 0

        def run(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                canvas_service_for_root(tmp_path).create_surface(
                    session_id="session-canvas",
                    conversation_id="session-canvas",
                    surface_id="empty",
                    owner={"agent_id": "main", "task_id": "turn-canvas"},
                    correlation_id="turn-canvas",
                )
            else:
                service = canvas_service_for_root(tmp_path)
                invalid = [
                    {
                        "id": "root",
                        "component": "Text",
                        "text": "invalid",
                        "html": "not allowed",
                    }
                ]
                for _ in range(2):
                    try:
                        service.update_components(
                            session_id="session-canvas",
                            conversation_id="session-canvas",
                            surface_id="empty",
                            components=invalid,
                            correlation_id="turn-canvas",
                        )
                    except ValueError:
                        pass
            return SimpleNamespace(answer="Done.", sources=[], warnings=[])

    executor = Executor()
    result = _gateway(tmp_path)._execute_canvas_route(
        decision=_decision(),
        context=_context(),
        text="Create a surface.",
        ask_service=SimpleNamespace(ask_agent=executor),
        callbacks=None,
    )

    assert result.mode == "route-canvas-error"
    assert "No fallback UI" in result.answer
    assert executor.calls == 2
    snapshot = canvas_service_for_root(tmp_path).get_surface("session-canvas", "empty")
    assert snapshot.deleted is True


def test_canvas_renderer_action_resumes_owning_model_and_updates_surface(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    tools = {item.name: item for item in build_canvas_langchain_tools(tmp_path)}

    class Executor:
        calls = 0

        def run(self, *, question: str, tool_policy: dict, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                tools["canvas_create_surface"].invoke(
                    {
                        "source_decision_id": "turn-canvas",
                        "session_id": "session-canvas",
                        "conversation_id": "session-canvas",
                        "surface_id": "counter",
                        "owner": {"agent_id": "main", "task_id": "turn-canvas"},
                        "components": [
                            {
                                "id": "root",
                                "component": {
                                    "type": "Column",
                                    "children": ["count", "increment"],
                                },
                            },
                            {
                                "id": "count",
                                "component": {
                                    "type": "Text",
                                    "text": {"path": "/count"},
                                },
                            },
                            {
                                "id": "increment",
                                "component": {
                                    "type": "Button",
                                    "label": "Increase counter",
                                    "actions": [
                                        {
                                            "name": "counter.press",
                                            "context": {
                                                "count": {"path": "/count"}
                                            },
                                        }
                                    ],
                                },
                            },
                        ],
                        "data_model": {"count": 0},
                        "retain_on_complete": True,
                    }
                )
            else:
                assert "validated renderer action" in question
                assert "canvas_create_surface" not in tool_policy["allowed_tools"]
                tools["canvas_update_data"].invoke(
                    {
                        "source_decision_id": "action-counter",
                        "session_id": "session-canvas",
                        "conversation_id": "session-canvas",
                        "surface_id": "counter",
                        "value": {"count": 1},
                        "path": "/",
                    }
                )
            return SimpleNamespace(answer="Canvas updated.", sources=[], warnings=[])

    executor = Executor()
    result = _gateway(tmp_path)._execute_canvas_route(
        decision=_decision(),
        context=_context(),
        text="Create a counter.",
        ask_service=SimpleNamespace(ask_agent=executor),
        callbacks=None,
    )
    assert result.mode == "route-canvas"

    service = canvas_service_for_root(tmp_path)
    service.submit_action(
        RendererAction(
            action_id="canvas_action_counter",
            session_id="session-canvas",
            conversation_id="session-canvas",
            surface_id="counter",
            source_component_id="increment",
            name="counter.press",
            correlation_id="action-counter",
            context={"count": 0},
        )
    )

    assert executor.calls == 2
    assert service.get_surface("session-canvas", "counter").data_model == {"count": 1}
