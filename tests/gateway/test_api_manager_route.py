from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mana_agent.api_manager.runtime_tools import API_MANAGER_TOOL_NAMES
from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.gateway.entry_routing import EntryRouteContext, EntryRoutingDecision


def test_api_route_uses_only_narrow_manager_tools(tmp_path: Path) -> None:
    class ModelToolExecutor:
        def run(self, *, question: str, system_prompt: str, tool_policy: dict, **kwargs):
            assert question == "Get contact 123 from Acme CRM."
            assert tool_policy["allowed_tools"] == [
                *API_MANAGER_TOOL_NAMES,
                "browser_open",
                "browser_inspect",
                "browser_click",
                "browser_wait",
                "browser_scroll",
                "browser_close",
            ]
            assert tool_policy["disable_external_search"] is True
            assert "api_operations_search first" in system_prompt
            assert "api_docs_import_semantic" in system_prompt
            assert "Never type, submit forms, sign in" in system_prompt
            assert "Never claim an API call succeeded" in system_prompt
            assert kwargs["flow_id"] == "session-api"
            return SimpleNamespace(
                answer="Contact 123 was returned by the saved API operation.",
                sources=[],
                warnings=[],
                trace=[
                    {
                        "tool_name": "api_workflow_decide",
                        "status": "ok",
                        "output_preview": json.dumps(
                            {
                                "ok": True,
                                "result": {
                                    "task_intent": "retrieve contact 123",
                                    "required_actions": [
                                        "operation_search",
                                        "request_execution",
                                    ],
                                    "reason": "The operation must be selected and executed.",
                                    "safe_to_continue": True,
                                },
                            }
                        ),
                    },
                    {
                        "tool_name": "api_operations_search",
                        "status": "ok",
                        "output_preview": '{"ok":true,"result":[{"operation_id":"getContact"}]}',
                    },
                    {
                        "tool_name": "api_request_execute",
                        "status": "ok",
                        "output_preview": '{"ok":true,"result":{"executed":true}}',
                    },
                ],
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


def test_api_route_surfaces_network_approval_in_result_payload(tmp_path: Path) -> None:
    permission = {
        "ok": False,
        "error_code": "permission_required",
        "permission_request_id": "api_approval_http_1",
        "permission_scope": "api.request.execute",
        "session_id": "session-api",
        "preview": {"method": "GET", "approval_required": True},
    }

    class ModelToolExecutor:
        def run(self, **kwargs):
            return SimpleNamespace(
                answer="The exact HTTP request is waiting for local approval.",
                sources=[],
                warnings=[],
                trace=[
                    {
                        "tool_name": "api_workflow_decide",
                        "status": "ok",
                        "output_preview": json.dumps(
                            {
                                "ok": True,
                                "result": {
                                    "task_intent": "execute API request",
                                    "required_actions": ["request_execution"],
                                    "reason": "The user requested execution.",
                                    "safe_to_continue": True,
                                },
                            }
                        ),
                    },
                    {
                        "tool_name": "api_request_execute",
                        "status": "ok",
                        "output_preview": json.dumps(permission),
                    }
                ],
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
            reason="Use the API lifecycle.",
            required_sources=("api",),
        ),
        context=EntryRouteContext(
            session_id="session-api",
            conversation_id="session-api",
            turn_id="turn-api",
        ),
        text="Execute the saved API request using the configured HTTP endpoint.",
        ask_service=SimpleNamespace(ask_agent=ModelToolExecutor()),
        callbacks=None,
    )

    assert result.mode == "route-api-awaiting-approval"
    assert result.payload["permission_requests"][0]["permission_request_id"] == (
        "api_approval_http_1"
    )


def test_api_route_does_not_complete_without_required_execution_evidence(
    tmp_path: Path,
) -> None:
    class ModelToolExecutor:
        def run(self, **kwargs):
            return SimpleNamespace(
                answer="The operation was identified, but no API call was made.",
                sources=[],
                warnings=[],
                trace=[
                    {
                        "tool_name": "api_workflow_decide",
                        "status": "ok",
                        "output_preview": json.dumps(
                            {
                                "ok": True,
                                "result": {
                                    "task_intent": "inspect, import, and call API",
                                    "required_actions": [
                                        "documentation_inspection",
                                        "integration_import",
                                        "operation_search",
                                        "request_execution",
                                    ],
                                    "reason": "All stages are required by the user.",
                                    "safe_to_continue": True,
                                },
                            }
                        ),
                    },
                    {
                        "tool_name": "browser_inspect",
                        "status": "ok",
                        "output_preview": '{"ok":true,"text":"GET /{ip}"}',
                    },
                    {
                        "tool_name": "api_docs_import_semantic",
                        "status": "ok",
                        "output_preview": '{"ok":true,"result":{"saved":true}}',
                    },
                ],
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
            reason="Use the complete API lifecycle.",
            required_sources=("api",),
        ),
        context=EntryRouteContext(
            session_id="session-api",
            conversation_id="session-api",
            turn_id="turn-api",
        ),
        text="Inspect documentation and execute the API request.",
        ask_service=SimpleNamespace(ask_agent=ModelToolExecutor()),
        callbacks=None,
    )

    assert result.mode == "route-api-incomplete"
    assert result.error == "api_workflow_incomplete"
    assert result.payload["workflow_completion"]["missing_actions"] == [
        "operation_search",
        "request_execution",
    ]
