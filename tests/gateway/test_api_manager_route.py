from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mana_agent.api_manager.runtime_tools import API_MANAGER_TOOL_NAMES
from mana_agent.gateway.chat_gateway import (
    AgentChatGateway,
    _api_workflow_completion_from_trace,
)
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
            assert tool_policy["capability_discovery_required"] is True
            assert tool_policy["initial_tools"] == ["api_workflow_decide"]
            assert "model_max_tokens" not in tool_policy
            assert "api_operations_search first" in system_prompt
            assert "capability_search and capability_load" in system_prompt
            assert "api_docs_import_semantic" in system_prompt
            assert "retry the same import" in system_prompt
            assert "refresh_integration_id" in system_prompt
            assert "redacted saved-integration snapshot" in system_prompt
            assert "declare only operation_search, request_preview, and request_execution" in system_prompt
            assert "Never type, submit forms, sign in" in system_prompt
            assert "Never claim an API call succeeded" in system_prompt
            assert "explicit risk=read_only declaration" in system_prompt
            assert kwargs["flow_id"] == "session-api"
            assert kwargs["max_steps"] >= 32
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
                                        "request_preview",
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
                        "tool_name": "api_request_preview",
                        "status": "ok",
                        "output_preview": '{"ok":true,"result":{"risk_level":"read_only"}}',
                    },
                    {
                        "tool_name": "api_request_execute",
                        "status": "ok",
                        "output_preview": '{"ok":true,"result":{"executed":true,"upstream_ok":true,"status_code":200}}',
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
    assert result.payload["workflow_completion"]["execution_evidence"]["status_code"] == 200
    assert '"status_code": 200' in result.answer


def test_api_route_treats_documentation_url_as_optional_when_integration_is_saved(
    tmp_path: Path,
) -> None:
    class ModelToolExecutor:
        def run(self, *, system_prompt: str, **kwargs):
            assert (
                "A supplied documentation URL is not, by itself, evidence that import or refresh "
                "is required." in system_prompt
            )
            assert "Immediately after the workflow decision, list saved integrations." in system_prompt
            return SimpleNamespace(
                answer="The saved news integration returned current reporting.",
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
                                    "task_intent": "retrieve current Iran-US news",
                                    "required_actions": [
                                        "operation_search",
                                        "request_preview",
                                        "request_execution",
                                    ],
                                    "reason": "A suitable saved news integration exists.",
                                    "safe_to_continue": True,
                                },
                            }
                        ),
                    },
                    {
                        "tool_name": "api_operations_search",
                        "status": "ok",
                        "output_preview": '{"ok":true,"result":[{"operation_id":"get_news"}]}',
                    },
                    {
                        "tool_name": "api_request_preview",
                        "status": "ok",
                        "output_preview": '{"ok":true,"result":{"risk_level":"read_only"}}',
                    },
                    {
                        "tool_name": "api_request_execute",
                        "status": "ok",
                        "output_preview": (
                            '{"ok":true,"result":{"executed":true,"upstream_ok":true,'
                            '"status_code":200}}'
                        ),
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
            reason="Use the saved Mediastack integration.",
            required_sources=("api",),
        ),
        context=EntryRouteContext(
            session_id="session-api",
            conversation_id="session-api",
            turn_id="turn-api",
        ),
        text="Use this documentation URL and call the saved news API.",
        ask_service=SimpleNamespace(ask_agent=ModelToolExecutor()),
        callbacks=None,
    )

    assert result.mode == "route-api"
    assert result.payload["workflow_completion"]["required_actions"] == [
        "operation_search",
        "request_preview",
        "request_execution",
    ]


def test_api_route_stops_at_preview_when_network_approval_is_required(tmp_path: Path) -> None:
    permission = {
        "ok": True,
        "result": {
            "permission_required": True,
            "permission_request_id": "api_approval_http_1",
            "permission_scope": "api.request.execute",
            "session_id": "session-api",
            "preview": {"method": "GET", "approval_required": True},
        },
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
                                    "required_actions": [
                                        "operation_search",
                                        "request_preview",
                                        "request_execution",
                                    ],
                                    "reason": "The user requested execution.",
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
                        "tool_name": "api_request_preview",
                        "status": "ok",
                        "output_preview": json.dumps(permission),
                    },
                ],
            )

    gateway = object.__new__(AgentChatGateway)
    gateway.root = tmp_path
    gateway._index_dir = None
    gateway._resolved_k = 4
    gateway._agent_timeout_seconds = 30
    observed_events: list[tuple[str, str, dict]] = []
    gateway._event_sink = lambda event_type, title, metadata: observed_events.append(
        (event_type, title, metadata)
    )
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
    assert result.payload["permission_requests"][0]["permission_request_id"] == "api_approval_http_1"
    assert observed_events[0][0] == "api.waiting_approval"
    assert observed_events[0][2]["permission_request_id"] == "api_approval_http_1"


def test_api_approval_completion_includes_validated_response_evidence() -> None:
    message = AgentChatGateway._api_approval_completion_message(
        {
            "method": "GET",
            "redacted_url": "http://api.example.test/5.216.25.186?access_key=[REDACTED]",
            "status_code": 200,
            "content_type": "application/json",
            "body_kind": "json",
            "json_body": {"city": "Tehran", "country_name": "Iran"},
            "latency_ms": 42.5,
        },
        200,
    )

    assert "HTTP status 200" in message
    assert "Validated API result" in message
    assert "**City:** Tehran" in message
    assert "**Country Name:** Iran" in message
    assert "**Endpoint:** http://api.example.test/5.216.25.186?access_key=[REDACTED]" in message
    assert "[REDACTED]" in message


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
                                        "request_preview",
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
        "request_preview",
        "request_execution",
    ]


def test_api_route_surfaces_valid_execution_when_import_remains_incomplete(
    tmp_path: Path,
) -> None:
    class ModelToolExecutor:
        def run(self, **kwargs):
            return SimpleNamespace(
                answer="The response body was unavailable.",
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
                                        "request_preview",
                                        "request_execution",
                                    ],
                                    "reason": "Every lifecycle action is required.",
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
                        "status": "error",
                        "output_preview": '{"ok":false,"error_code":"duplicate"}',
                    },
                    {
                        "tool_name": "api_operations_search",
                        "status": "ok",
                        "output_preview": '{"ok":true,"result":[{"operation_id":"lookup"}]}',
                    },
                    {
                        "tool_name": "api_request_preview",
                        "status": "ok",
                        "output_preview": '{"ok":true,"result":{"risk_level":"read_only"}}',
                    },
                    {
                        "tool_name": "api_request_execute",
                        "status": "ok",
                        "output_preview": (
                            '{"ok":true,"result":{"executed":true,"upstream_ok":true,'
                            '"status_code":200,"json_body":{"city":"Shiraz"}}}'
                        ),
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
    assert result.payload["workflow_completion"]["missing_actions"] == [
        "integration_import"
    ]
    assert "overall workflow remains incomplete" in result.answer
    assert '"city": "Shiraz"' in result.answer


def test_api_workflow_accepts_successful_clipped_non_execution_evidence() -> None:
    response = SimpleNamespace(
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
                                "request_preview",
                                "request_execution",
                            ],
                            "reason": "Every selected lifecycle action is required.",
                            "safe_to_continue": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "browser_inspect",
                "status": "ok",
                "output_preview": '{"ok":true,"text":"' + ("x" * 3981),
            },
            {
                "tool_name": "api_docs_import_semantic",
                "status": "ok",
                "output_preview": '{"ok":true,"result":{"saved":true}}',
            },
            {
                "tool_name": "api_operations_search",
                "status": "ok",
                "output_preview": '{"ok":true,"result":[{"operation_id":"lookup"}]}',
            },
            {
                "tool_name": "api_request_preview",
                "status": "ok",
                "output_preview": '{"ok":true,"result":{"risk_level":"read_only"}}',
            },
            {
                "tool_name": "api_request_execute",
                "status": "ok",
                "output_preview": (
                    '{"ok":true,"result":{"executed":true,"upstream_ok":true,'
                    '"status_code":200}}'
                ),
            },
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is True
    assert completion["missing_actions"] == []
    assert "documentation_inspection" in completion["completed_actions"]


def test_api_workflow_rejects_unparseable_non_clipped_evidence() -> None:
    response = SimpleNamespace(
        trace=[
            {
                "tool_name": "api_workflow_decide",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "task_intent": "inspect API documentation",
                            "required_actions": ["documentation_inspection"],
                            "reason": "Documentation evidence is required.",
                            "safe_to_continue": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "browser_inspect",
                "status": "ok",
                "output_preview": "not structured evidence",
            },
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is False
    assert completion["missing_actions"] == ["documentation_inspection"]
