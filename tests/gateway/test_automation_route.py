"""Integration coverage for model-driven automation chat authoring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mana_agent.automations.runtime_tools import build_automation_langchain_tools
from mana_agent.automations.service import AutomationService
from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.gateway.entry_routing import EntryRouteContext, EntryRoutingDecision
from mana_agent.gateway.turn_engine import ChatTurnResult


def test_automation_create_schema_reports_only_relevant_payload_errors(
    tmp_path: Path,
) -> None:
    tool = {item.name: item for item in build_automation_langchain_tools(tmp_path)}[
        "automation_create"
    ]
    payload = {
        "source_decision_id": "decision-invalid-shape",
        "name": "Gmail latest messages",
        "description": "Check Gmail once.",
        "trigger": {
            "type": "once",
            "run_at": "2027-07-28T01:30:00+03:30",
            "timezone": "Asia/Tehran",
        },
        "job": {
            "type": "connector_action",
            "connector": "gmail",
            "action": "check_messages",
            "input": {"account": "ahdr1277@gmail.com", "limit": 5},
        },
        "timezone": "Asia/Tehran",
        "target_runtime": "local",
        "permission_references": ["account://gmail/ahdr1277@gmail.com"],
        "retry_policy": {"max_attempts": 3, "backoff_seconds": 60},
        "misfire_policy": {"type": "run_once_as_soon_as_possible"},
        "idempotency_key": "invalid-shape",
    }

    with pytest.raises(ValidationError) as captured:
        tool.invoke(payload)

    errors = captured.value.errors(include_url=False)
    assert len(errors) == 3
    assert {error["loc"] for error in errors} == {
        ("job", "connector_action", "input"),
        ("retry_policy", "max_attempts"),
        ("misfire_policy", "type"),
    }
    job_schema = tool.args_schema.model_json_schema()["properties"]["job"]
    assert job_schema["discriminator"]["propertyName"] == "type"


def test_natural_language_automation_route_creates_persisted_exact_interval(
    tmp_path: Path,
    monkeypatch,
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
            assert tool_policy["allowed_tools"] == ["automation_create"]
            output = tools["automation_create"].invoke(
                {
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
                }
            )
            return SimpleNamespace(
                answer=output, sources=[], warnings=[], tool_traces=[]
            )

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
            automation_operation="create",
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


def test_one_time_gmail_request_is_authored_without_recurrence_question(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    monkeypatch.setattr(
        "mana_agent.automations.service.reconcile_deployment",
        lambda root, automation_id: AutomationService(root).get(automation_id),
    )
    tools = {item.name: item for item in build_automation_langchain_tools(tmp_path)}
    run_at = (datetime.now(timezone.utc) + timedelta(days=1)).replace(
        hour=1, minute=8, second=0, microsecond=0
    )

    class ModelToolExecutor:
        def run(
            self,
            *,
            question: str,
            system_prompt: str,
            tool_policy: dict,
            **_kwargs,
        ):
            assert question == (
                "Create an automation that checks ahdr1277@gmail.com once at 1:08 UTC."
            )
            assert "recurrence is not a missing field" in system_prompt
            assert (
                "never execute the connector action during the authoring turn"
                in system_prompt
            )
            assert "do not ask the user to choose" in system_prompt
            assert str(tmp_path) in system_prompt
            assert "current_utc=" in system_prompt
            assert "strictly after current_utc" in system_prompt
            assert "with `arguments` (never `input`)" in system_prompt
            assert tool_policy["allowed_tools"] == ["automation_create"]
            assert "never call automation_list" in system_prompt
            output = tools["automation_create"].invoke(
                {
                    "name": "One-time Gmail check",
                    "description": "Check the requested Gmail account once at the requested time.",
                    "trigger": {
                        "type": "once",
                        "run_at": run_at.isoformat(),
                        "timezone": "UTC",
                    },
                    "job": {
                        "type": "connector_action",
                        "connector": "gmail",
                        "action": "check_inbox",
                        "arguments": {"account": "ahdr1277@gmail.com"},
                        "prompt": "Capture the Gmail check information.",
                    },
                    "timezone": "UTC",
                    "target_runtime": "local",
                    "permission_references": ["account://gmail/ahdr1277@gmail.com"],
                    "retry_policy": {},
                    "misfire_policy": {"mode": "run_once"},
                    "idempotency_key": "turn-create-one-time-gmail-check",
                    "source_decision_id": "decision-one-time-gmail",
                }
            )
            return SimpleNamespace(
                answer=output, sources=[], warnings=[], tool_traces=[]
            )

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
            reason="Create a one-time Gmail automation without running Gmail now.",
            required_sources=("repository",),
            automation_operation="create",
        ),
        context=EntryRouteContext(
            session_id="session-test",
            conversation_id="session-test",
            turn_id="turn-one-time-gmail",
        ),
        text="Create an automation that checks ahdr1277@gmail.com once at 1:08 UTC.",
        ask_service=SimpleNamespace(ask_agent=ModelToolExecutor()),
        callbacks=None,
    )

    persisted = AutomationService(tmp_path).list()
    assert result.mode == "route-automation"
    assert len(persisted) == 1
    assert persisted[0].trigger.type == "once"
    assert persisted[0].trigger.run_at == run_at
    assert persisted[0].job.type == "connector_action"
    assert persisted[0].job.connector == "gmail"
    assert persisted[0].job.arguments == {"account": "ahdr1277@gmail.com"}


def test_registered_automation_route_dispatches_from_multi_task_child_lane() -> None:
    automation_tools = (
        "automation_create",
        "automation_get",
        "automation_list",
        "automation_status",
        "automation_update",
        "automation_delete",
        "automation_enable",
        "automation_disable",
        "automation_run_now",
    )
    authorized: list[tuple[str, str]] = []
    executed: dict[str, object] = {}
    gateway = object.__new__(AgentChatGateway)
    gateway._entry_route_registry = SimpleNamespace(
        get=lambda route: SimpleNamespace(
            tools=automation_tools,
            availability=lambda: SimpleNamespace(available=True),
        )
    )
    gateway._lane_coordinator = SimpleNamespace(
        authorize_tool=lambda task_id, tool_name: authorized.append(
            (task_id, tool_name)
        )
    )

    def execute_automation_route(**kwargs):
        executed.update(kwargs)
        return ChatTurnResult(
            answer="Automation created.",
            mode="route-automation",
            payload={"route": "automation"},
        )

    gateway._execute_automation_route = execute_automation_route
    decision = EntryRoutingDecision(
        route="automation",
        confidence=0.99,
        reason="Create the model-authored schedule.",
        required_sources=("none",),
        automation_operation="create",
    )
    context = EntryRouteContext(
        session_id="session-test",
        conversation_id="session-test",
        turn_id="turn-test:automation-child",
        atomic_child=True,
        orchestration_parent_task_id="root-task",
    )

    result = gateway._execute_entry_route(
        decision=decision,
        context=context,
        text="Create the requested automation.",
        state={"history": ["parent conversation must stay isolated"]},
        ask_service=SimpleNamespace(),
        sink=None,
        options={
            "_lane_task_id": "automation-child-lane",
            "_isolated_child_prompt": True,
        },
    )

    assert result.mode == "route-automation"
    assert result.error is None
    assert executed["decision"] is decision
    assert executed["context"] is context
    assert executed["text"] == "Create the requested automation."
    assert authorized == [
        ("automation-child-lane", tool_name) for tool_name in automation_tools
    ]
