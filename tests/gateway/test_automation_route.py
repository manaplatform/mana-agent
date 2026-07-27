"""Integration coverage for model-driven automation chat authoring."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from mana_agent.automations.runtime_tools import build_automation_langchain_tools
from mana_agent.automations.service import AutomationService
from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.gateway.entry_routing import EntryRouteContext, EntryRoutingDecision


def test_natural_language_automation_route_creates_persisted_exact_interval(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    monkeypatch.setattr(
        "mana_agent.automations.service.reconcile_deployment",
        lambda root, automation_id: AutomationService(root).get(automation_id),
    )
    tools = {item.name: item for item in build_automation_langchain_tools(tmp_path)}

    class ModelToolExecutor:
        def run(self, *, question: str, tool_policy: dict, **_kwargs):
            assert question == "Check my email every 5 hours."
            assert set(tool_policy["allowed_tools"]) >= {"automation_create", "automation_status"}
            output = tools["automation_create"].invoke({
                "name": "Check email",
                "description": "Check the connected inbox and summarize new actionable mail.",
                "trigger": {
                    "type": "interval",
                    "every_seconds": 18_000,
                    "anchor_at": datetime.now(timezone.utc).isoformat(),
                },
                "job": {
                    "type": "connector_action",
                    "connector": "gmail",
                    "action": "check_inbox",
                    "arguments": {},
                    "prompt": "Summarize new actionable messages.",
                },
                "timezone": "UTC",
                "target_runtime": "local",
                "permission_references": ["account://gmail/default"],
                "retry_policy": {},
                "misfire_policy": {"mode": "run_once"},
                "idempotency_key": "turn-create-email-five-hours",
                "source_decision_id": "decision-1",
            })
            return SimpleNamespace(answer=output, sources=[], warnings=[], tool_traces=[])

    gateway = object.__new__(AgentChatGateway)
    gateway.root = tmp_path
    gateway._index_dir = None
    gateway._resolved_k = 4
    gateway._agent_timeout_seconds = 30
    gateway.config = SimpleNamespace(agent_max_steps=8)
    result = gateway._execute_automation_route(
        decision=EntryRoutingDecision(
            route="automation",
            confidence=0.99,
            reason="Create a durable automation.",
            required_sources=("none",),
        ),
        context=EntryRouteContext(
            session_id="session-test",
            conversation_id="session-test",
            turn_id="turn-test",
        ),
        text="Check my email every 5 hours.",
        ask_service=SimpleNamespace(ask_agent=ModelToolExecutor()),
        callbacks=None,
    )
    persisted = AutomationService(tmp_path).list()
    assert result.mode == "route-automation"
    assert len(persisted) == 1
    assert persisted[0].trigger.type == "interval"
    assert persisted[0].trigger.every_seconds == 18_000
    assert persisted[0].job.type == "connector_action"
