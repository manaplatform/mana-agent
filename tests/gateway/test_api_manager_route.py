from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mana_agent.analysis.models import ToolInvocationTrace
from mana_agent.api_manager.runtime_tools import API_MANAGER_TOOL_NAMES
from mana_agent.gateway.chat_gateway import (
    AgentChatGateway,
    _api_workflow_completion_from_trace,
)
from mana_agent.gateway.entry_routing import EntryRouteContext, EntryRoutingDecision
from mana_agent.gateway.turn_engine import _serialize_tool_traces


def _outcome_trace(executor: str = "api_request_execute", *, method: str = "GET") -> SimpleNamespace:
    return SimpleNamespace(
        answer="The API returned a result.",
        sources=[],
        warnings=[],
        trace=[
            {
                "tool_name": "api_workflow_decide",
                "status": "ok",
                "output_preview": json.dumps({
                    "ok": True,
                    "result": {
                        "task_intent": "look up the requested resource",
                        "required_outcomes": [
                            "api_target_resolved",
                            "api_execution_verified",
                        ],
                        "reason": "A live response is required.",
                        "safe_to_continue": True,
                    },
                }),
            },
            {
                "tool_name": executor,
                "status": "ok",
                "output_preview": json.dumps({
                    "ok": True,
                    "result": {
                        "executed": True,
                        "upstream_ok": True,
                        "status_code": 200,
                        "method": method,
                        "redacted_url": "https://api.example.test/resource",
                        "response_received": True,
                        "json_body": {"value": "result"},
                    },
                }),
            },
        ],
    )


def test_api_workflow_read_only_execution_needs_no_preview_or_documentation() -> None:
    completion = _api_workflow_completion_from_trace(_outcome_trace())

    assert completion["valid"] is True
    assert completion["missing_actions"] == []
    assert completion["execution_evidence"]["execution_verified"] is True
    assert completion["execution_evidence"]["executor"] == "api_request_execute"


def test_api_workflow_accepts_authorized_non_api_manager_executor() -> None:
    completion = _api_workflow_completion_from_trace(
        _outcome_trace("authorized_http_connector")
    )

    assert completion["valid"] is True
    assert completion["execution_evidence"]["executor"] == "authorized_http_connector"


def test_api_workflow_requires_preview_for_mutations() -> None:
    completion = _api_workflow_completion_from_trace(_outcome_trace(method="POST"))

    assert completion["valid"] is False
    assert completion["error_code"] == "api_workflow_incomplete"
    assert "api_execution_verified" in completion["missing_actions"]


def test_api_workflow_does_not_count_documentation_or_model_claim_as_execution() -> None:
    response = _outcome_trace()
    response.trace = response.trace[:1] + [
        {
            "tool_name": "api_docs_inspect",
            "status": "ok",
            "output_preview": '{"ok":true,"result":{"text":"GET /resource"}}',
        },
    ]
    response.answer = "The API returned HTTP 200."

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is False
    assert completion["execution_evidence"] == {}
    assert completion["missing_actions"] == [
        "api_target_resolved",
        "api_execution_verified",
    ]


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
            assert tool_policy.get("capability_discovery_required") is not True
            assert "initial_tools" not in tool_policy
            assert "model_max_tokens" not in tool_policy
            assert "api_operations_search first" in system_prompt
            assert "capability_search and capability_load" not in system_prompt
            assert "api_docs_import_semantic" in system_prompt
            assert "retry the same import" in system_prompt
            assert "refresh_integration_id" in system_prompt
            assert "redacted saved-integration snapshot" in system_prompt
            assert "api_target_resolved, api_execution_verified" in system_prompt
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
                                    "required_outcomes": [
                                        "api_execution_verified",
                                        "user_goal_verified",
                                    ],
                                    "optional_outcomes": [
                                        "operation_resolved",
                                        "request_previewed",
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
                                    "required_outcomes": [
                                        "api_execution_verified",
                                        "user_goal_verified",
                                    ],
                                    "optional_outcomes": [
                                        "operation_resolved",
                                        "request_previewed",
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
    assert result.payload["workflow_completion"]["required_outcomes"] == [
        "api_execution_verified",
        "user_goal_verified",
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
                                    "required_outcomes": [
                                        "api_execution_verified",
                                        "user_goal_verified",
                                    ],
                                    "optional_outcomes": [
                                        "operation_resolved",
                                        "request_previewed",
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
                                    "required_outcomes": [
                                        "api_execution_verified",
                                        "user_goal_verified",
                                    ],
                                    "optional_outcomes": [
                                        "documentation_understood",
                                        "integration_available",
                                        "operation_resolved",
                                        "request_previewed",
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
    assert "api_execution_verified" in result.payload["workflow_completion"]["missing_actions"]


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
                                    "required_outcomes": [
                                        "api_execution_verified",
                                        "user_goal_verified",
                                    ],
                                    "optional_outcomes": [
                                        "documentation_understood",
                                        "integration_available",
                                        "operation_resolved",
                                        "request_previewed",
                                    ],
                                    "reason": "Every lifecycle action is optional support for execution.",
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

    assert result.mode == "route-api"
    assert result.payload["workflow_completion"]["missing_actions"] == []
    assert result.payload["workflow_completion"]["valid"] is True
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
                            "required_outcomes": [
                                "documentation_understood",
                                "integration_available",
                                "operation_resolved",
                                "request_previewed",
                                "api_execution_verified",
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
                            "required_outcomes": ["documentation_understood"],
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
    assert completion["missing_actions"] == ["documentation_understood"]


def test_api_workflow_recovers_when_workflow_decision_retried_after_validation_error() -> None:
    response = SimpleNamespace(
        trace=[
            {
                "tool_name": "api_workflow_decide",
                "status": "error",
                "output_preview": (
                    "1 validation error for _WorkflowDecision\nrequired_outcomes\n"
                    "  Input should be a valid tuple [type=tuple_type, input_value='operation_resolved', input_type=str]"
                ),
            },
            {
                "tool_name": "api_workflow_decide",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "task_intent": "search available operations",
                            "required_outcomes": ["operation_resolved"],
                            "reason": "Search is required to discover matching operations.",
                            "safe_to_continue": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_operations_search",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": [
                            {"operation_id": "lookup_verse", "integration_id": "api_123"}
                        ],
                    }
                ),
            },
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is True
    assert completion["error_code"] == ""
    assert completion["missing_actions"] == []
    assert "operation_resolved" in completion["completed_outcomes"]


def test_api_workflow_rejects_when_first_tool_is_not_workflow_decide() -> None:
    response = SimpleNamespace(
        trace=[
            {
                "tool_name": "api_operations_search",
                "status": "ok",
                "output_preview": '{"ok":true,"result":[]}',
            }
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is False
    assert completion["error_code"] == "api_workflow_decision_missing"


def test_api_workflow_rejects_when_workflow_decision_never_succeeds() -> None:
    response = SimpleNamespace(
        trace=[
            {
                "tool_name": "api_workflow_decide",
                "status": "error",
                "output_preview": "Validation error: invalid field",
            },
            {
                "tool_name": "api_operations_search",
                "status": "ok",
                "output_preview": '{"ok":true,"result":[]}',
            },
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is False
    assert completion["error_code"] == "api_workflow_decision_invalid"


def test_api_workflow_accepts_execution_evidence_when_output_preview_is_truncated() -> None:
    execution_result = {
        "ok": True,
        "result": {
            "integration_id": "api_9de895d68869da96eaf42393",
            "operation_id": "21dc79663be7391405e98eefd6599f93",
            "method": "GET",
            "redacted_url": "https://dummyjson.com/products/search?q=phone",
            "status_code": 200,
            "executed": True,
            "upstream_ok": True,
            "json_body": {"total": 100, "products": [{"title": "sample"}] * 100},
        },
    }

    # Simulate truncated JSON string in output_preview due to 4000-char clipping
    truncated_preview = json.dumps(execution_result)[:4000]

    response = SimpleNamespace(
        trace=[
            ToolInvocationTrace(
                tool_name="api_workflow_decide",
                args_summary="",
                duration_ms=1.0,
                status="ok",
                output_preview=json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "task_intent": "search products and execute API request",
                            "required_outcomes": [
                                "api_execution_verified",
                                "user_goal_verified",
                            ],
                            "optional_outcomes": [
                                "operation_resolved",
                                "request_previewed",
                            ],
                            "reason": "Search, preview, and execution are required.",
                            "safe_to_continue": True,
                        },
                    }
                ),
            ),
            ToolInvocationTrace(
                tool_name="api_operations_search",
                args_summary="",
                duration_ms=10.0,
                status="ok",
                output_preview='{"ok":true,"result":[{"operation_id":"21dc79663be7391405e98eefd6599f93"}]}',
            ),
            ToolInvocationTrace(
                tool_name="api_request_preview",
                args_summary="",
                duration_ms=10.0,
                status="ok",
                output_preview='{"ok":true,"result":{"risk_level":"read_only"}}',
            ),
            ToolInvocationTrace(
                tool_name="api_request_execute",
                args_summary="",
                duration_ms=150.0,
                status="ok",
                output_preview=truncated_preview,
                result=execution_result,
            ),
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is True
    assert completion["missing_actions"] == []
    assert completion["error_code"] == ""
    assert "request_execution" in completion["completed_actions"]
    assert completion["execution_evidence"]["status_code"] == 200
    assert completion["execution_evidence"]["operation_id"] == "21dc79663be7391405e98eefd6599f93"
    

def test_tool_invocation_trace_preserves_structured_result_in_serialization() -> None:
    trace = ToolInvocationTrace(
        tool_name="api_request_execute",
        args_summary="sample",
        duration_ms=50.0,
        status="ok",
        output_preview="clipped preview",
        result={"ok": True, "result": {"executed": True, "upstream_ok": True, "status_code": 200}},
    )

    serialized = _serialize_tool_traces(SimpleNamespace(trace=[trace]))
    assert len(serialized) == 1
    assert serialized[0]["result"]["ok"] is True
    assert serialized[0]["result"]["result"]["status_code"] == 200


def test_api_approval_resumes_continuation_and_completes_model_intent(tmp_path: Path) -> None:
    events: list[tuple[str, str, dict]] = []

    def sink(event_type: str, title: str = "", **kwargs: Any) -> None:
        events.append((event_type, title, kwargs))

    class ContinuationAskAgent:
        def __init__(self) -> None:
            self.calls = []

        def run(self, *, question: str, system_prompt: str, **kwargs: Any) -> SimpleNamespace:
            self.calls.append({"question": question, "kwargs": kwargs})
            assert "Validated API Execution Evidence" in question
            return SimpleNamespace(
                answer="خلاصه فارسی متن سفاریا پیدایش ۱:۱ بر اساس نتیجه اجرای ای‌پی‌آی.",
                sources=[],
                warnings=[],
                trace=[],
            )

    ask_agent = ContinuationAskAgent()
    gateway = object.__new__(AgentChatGateway)
    gateway.root = tmp_path
    gateway._index_dir = None
    gateway._resolved_k = 4
    gateway._agent_timeout_seconds = 30
    gateway._event_sink = sink
    gateway.config = SimpleNamespace(agent_max_steps=8)
    gateway._stack = SimpleNamespace(ask_service=SimpleNamespace(ask_agent=ask_agent))
    gateway._sessions = {}
    gateway._lane_coordinator = None

    from mana_agent.api_manager.runtime_tools import api_manager_service
    service = api_manager_service(tmp_path)
    from mana_agent.api_manager.models import HttpMethod, OperationRiskLevel
    from mana_agent.api_manager.request_builder import BuiltApiRequest, RequestPreview
    from datetime import datetime, timezone, timedelta
    req = BuiltApiRequest(
        integration_id="api_1234567890abcdef12345678",
        operation_id="get_sefaria_text",
        method="GET",
        url="https://www.sefaria.org/api/v3/texts/Genesis.1.1",
        timeout_seconds=30.0,
        risk_level=OperationRiskLevel.UPDATE,
        session_id="conv_sefaria_session",
        routing_task_intent="درخواست متن پیدایش ۱:۱ از سفاریا و ترجمه/خلاصه فارسی",
    )
    prev = RequestPreview(
        integration_id="api_1234567890abcdef12345678",
        integration_name="Sefaria",
        operation_id="get_sefaria_text",
        operation_name="Get Text",
        method="GET",
        redacted_url="https://www.sefaria.org/api/v3/texts/Genesis.1.1",
        redacted_headers={},
        query_parameters={},
        body_summary={},
        expected_side_effects="Read or update Sefaria text.",
        risk_level=OperationRiskLevel.UPDATE,
        approval_required=True,
    )
    details = service.approvals.prepare(
        req,
        prev,
        session_id="conv_sefaria_session",
        conversation_id="conv_sefaria_session",
        turn_id="turn_123",
        task_intent="درخواست متن پیدایش ۱:۱ از سفاریا و ترجمه/خلاصه فارسی",
    )
    approval_id = details["permission_request_id"]

    # Mock low-level execute_prepared_request to simulate successful HTTP call
    from types import MethodType
    def mock_execute(self, request, preview, **kwargs):
        return SimpleNamespace(
            executed=True,
            upstream_ok=True,
            status_code=200,
            json_body={"he": "בְּרֵאשִׁ֖ית", "text": "In the beginning"},
            model_dump=lambda mode="json": {
                "executed": True,
                "upstream_ok": True,
                "status_code": 200,
                "json_body": {"he": "בְּרֵאשִׁ֖ית", "text": "In the beginning"},
            },
        )
    service._execute_prepared_request = MethodType(mock_execute, service)

    result = gateway.api_approval_command(
        approval_id,
        session_id="conv_sefaria_session",
        client_type="dashboard",
    )

    assert result["status"] == "completed"
    assert result["resume"] == "completed"
    assert "خلاصه فارسی" in result["answer"]
    assert len(ask_agent.calls) == 1

    event_types = [e[0] for e in events]
    assert "api.approval_decided" in event_types
    assert "turn.resume_requested" in event_types
    assert "turn.finished" in event_types

    # Duplicate call is idempotent and does not run model or HTTP again
    duplicate_result = gateway.api_approval_command(
        approval_id,
        session_id="conv_sefaria_session",
        client_type="dashboard",
    )
    assert duplicate_result["status"] == "completed"
    assert len(ask_agent.calls) == 1




def test_api_workflow_accepts_unsupported_terminal_after_complete_inspection() -> None:
    documentation_ref = "sha256:" + ("a" * 64)

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
                            "required_outcomes": [
                                "documentation_understood",
                            ],
                            "reason": (
                                "The supplied documentation must be inspected "
                                "before attempting the API lifecycle."
                            ),
                            "safe_to_continue": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_docs_inspect",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "reference": "https://example.test/docs",
                            "documentation_ref": documentation_ref,
                            "content_type": "text/html",
                            "bytes": 2040,
                            "text": "A" * 2000,
                            "offset": 0,
                            "limit": 2000,
                            "next_offset": 2000,
                            "truncated": True,
                            "more_available": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_docs_inspect",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "reference": "https://example.test/docs",
                            "documentation_ref": documentation_ref,
                            "content_type": "text/html",
                            "bytes": 2040,
                            "text": "No usable API definition is documented.",
                            "offset": 2000,
                            "limit": 2000,
                            "next_offset": None,
                            "truncated": False,
                            "more_available": False,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_workflow_terminal",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "outcome": "unsupported_documentation",
                            "documentation_ref": documentation_ref,
                            "reason": (
                                "The complete inspected source contains no "
                                "usable API definition that can be safely imported."
                            ),
                        },
                    }
                ),
            },
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is True
    assert completion["error_code"] == ""

    assert "documentation_understood" in completion["completed_outcomes"]
    assert completion["missing_actions"] == []
    assert completion["unexpected_actions"] == []

    assert completion["terminal_outcome"] == "unsupported_documentation"
    assert completion["terminal_evidence"]["documentation_ref"] == documentation_ref
    assert (
        completion["terminal_evidence"]["outcome"]
        == "unsupported_documentation"
    )


def test_api_workflow_rejects_unsupported_terminal_before_complete_inspection() -> None:
    documentation_ref = "sha256:" + ("b" * 64)

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
                            "required_outcomes": [
                                "documentation_understood",
                            ],
                            "reason": (
                                "The supplied documentation must be inspected "
                                "before attempting the API lifecycle."
                            ),
                            "safe_to_continue": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_docs_inspect",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "reference": "https://example.test/docs",
                        "documentation_ref": documentation_ref,
                        "content_type": "text/html",
                        "bytes": 8000,
                        "text": "A" * 2000,
                        "offset": 0,
                        "limit": 2000,
                        "next_offset": 2000,
                        "truncated": True,
                        "more_available": True,
                    }
                ),
            },
            {
                "tool_name": "api_workflow_terminal",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "outcome": "unsupported_documentation",
                            "documentation_ref": documentation_ref,
                            "reason": "No API definition was found in the preview.",
                        },
                    }
                ),
            },
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is False
    assert completion["error_code"] == "api_workflow_terminal_invalid"

    assert (
        "complete contiguous documentation inspection"
        in completion["message"]
    )

    assert completion["completed_actions"] == []
    assert completion["waived_actions"] == []

    assert completion["missing_actions"] == [
        "documentation_understood",
    ]

    assert completion["terminal_outcome"] == ""
    assert completion["terminal_evidence"] == {}


def test_api_workflow_dummyjson_search_evidence_satisfies_goal() -> None:
    response = SimpleNamespace(
        answer="Found products matching 'phone' in the catalog.",
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
                            "task_intent": "search products matching 'phone'",
                            "required_outcomes": [
                                "api_target_resolved",
                                "api_execution_verified",
                                "user_goal_verified",
                            ],
                            "reason": "Real API execution is required to retrieve product results.",
                            "safe_to_continue": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_operations_search",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": [{"operation_id": "search_products", "path": "/products/search"}],
                    }
                ),
            },
            {
                "tool_name": "api_request_execute",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "executed": True,
                            "upstream_ok": True,
                            "status_code": 200,
                            "method": "GET",
                            "redacted_url": "https://dummyjson.com/products/search?q=phone",
                            "response_received": True,
                            "json_body": {
                                "total": 15,
                                "products": [{"title": "iPhone 9", "description": "An apple mobile phone"}],
                            },
                        },
                    }
                ),
            },
        ],
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is True
    assert completion["goal_satisfied"] is True
    assert completion["missing_actions"] == []
    assert completion["execution_evidence"]["execution_verified"] is True
    assert completion["execution_evidence"]["target_origin"] == "https://dummyjson.com"
    assert completion["execution_evidence"]["status_code"] == 200
    assert len(completion["actual_tool_events"]) == 2

def test_api_workflow_browser_docs_with_authorized_executor() -> None:
    response = SimpleNamespace(
        trace=[
            {
                "tool_name": "api_workflow_decide",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "task_intent": "inspect docs and call endpoint",
                            "required_outcomes": [
                                "documentation_understood",
                                "api_execution_verified",
                            ],
                            "reason": "Inspect documentation and execute request.",
                            "safe_to_continue": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "browser_inspect",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "text": "Endpoint: GET /v1/data. Status: 200 OK.",
                    }
                ),
            },
            {
                "tool_name": "authorized_http_connector",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "executed": True,
                            "upstream_ok": True,
                            "status_code": 200,
                            "method": "GET",
                            "redacted_url": "https://api.example.test/v1/data",
                            "response_received": True,
                            "json_body": {"items": [1, 2, 3]},
                        },
                    }
                ),
            },
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is True
    assert completion["goal_satisfied"] is True
    assert completion["missing_actions"] == []
    assert "documentation_understood" in completion["completed_outcomes"]
    assert completion["execution_evidence"]["executor"] == "authorized_http_connector"
    assert completion["execution_evidence"]["status_code"] == 200


def test_api_workflow_saved_integration_without_reinspection() -> None:
    response = SimpleNamespace(
        trace=[
            {
                "tool_name": "api_workflow_decide",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "task_intent": "call saved integration",
                            "required_outcomes": [
                                "operation_resolved",
                                "api_execution_verified",
                            ],
                            "reason": "Saved operation is already available.",
                            "safe_to_continue": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_operations_search",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": [{"operation_id": "lookup_user"}],
                    }
                ),
            },
            {
                "tool_name": "api_request_execute",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "executed": True,
                            "upstream_ok": True,
                            "status_code": 200,
                            "method": "GET",
                            "redacted_url": "https://api.crm.test/users/42",
                            "response_received": True,
                            "json_body": {"id": 42, "name": "Alice"},
                        },
                    }
                ),
            },
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is True
    assert completion["missing_actions"] == []
    assert "operation_resolved" in completion["completed_outcomes"]
    assert "api_execution_verified" in completion["completed_outcomes"]
    assert completion["execution_evidence"]["status_code"] == 200


def test_api_workflow_mutation_without_preview_fails_policy() -> None:
    response = SimpleNamespace(
        trace=[
            {
                "tool_name": "api_workflow_decide",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "task_intent": "create user record",
                            "required_outcomes": [
                                "api_target_resolved",
                                "api_execution_verified",
                            ],
                            "reason": "POST mutation request.",
                            "safe_to_continue": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_request_execute",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "executed": True,
                            "upstream_ok": True,
                            "status_code": 201,
                            "method": "POST",
                            "redacted_url": "https://api.crm.test/users",
                            "response_received": True,
                            "json_body": {"id": 100},
                        },
                    }
                ),
            },
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    # Mutation without preview or approval fails the preview policy
    assert completion["valid"] is False
    assert completion["error_code"] == "api_workflow_incomplete"
    assert "api_execution_verified" in completion["missing_actions"]


def test_api_workflow_separates_actual_tool_events_from_evidence() -> None:
    response = SimpleNamespace(
        trace=[
            {
                "tool_name": "api_workflow_decide",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "task_intent": "inspect and call",
                            "required_outcomes": [
                                "api_target_resolved",
                                "api_execution_verified",
                            ],
                            "reason": "Execution required.",
                            "safe_to_continue": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_operations_search",
                "status": "ok",
                "output_preview": '{"ok":true,"result":[{"operation_id":"op1"}]}',
            },
            {
                "tool_name": "api_request_execute",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "executed": True,
                            "upstream_ok": True,
                            "status_code": 200,
                            "method": "GET",
                            "redacted_url": "https://api.test/data",
                            "response_received": True,
                        },
                    }
                ),
            },
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is True
    assert isinstance(completion["actual_tool_events"], list)
    assert len(completion["actual_tool_events"]) == 2
    assert completion["actual_tool_events"][0]["tool_name"] == "api_operations_search"
    assert completion["actual_tool_events"][1]["tool_name"] == "api_request_execute"
    assert completion["execution_evidence"]["execution_verified"] is True
    assert completion["execution_evidence"]["executor"] == "api_request_execute"


def test_api_workflow_quran_reproduction_succeeds_without_fixed_tool_sequence() -> None:
    """Read-only API call (e.g. Quran ayah retrieval) succeeds on verified execution."""
    response = SimpleNamespace(
        trace=[
            {
                "tool_name": "api_workflow_decide",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "task_intent": "fetch Ayatul Kursi (2:255) from Quran API",
                            "required_outcomes": [
                                "api_execution_verified",
                                "user_goal_verified",
                            ],
                            "optional_outcomes": [
                                "documentation_understood",
                                "integration_available",
                                "operation_resolved",
                                "request_previewed",
                            ],
                            "reason": "Direct execution of Quran API endpoint.",
                            "safe_to_continue": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_request_execute",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "executed": True,
                            "upstream_ok": True,
                            "status_code": 200,
                            "method": "GET",
                            "redacted_url": "https://api.alquran.cloud/v1/ayah/2:255/en.asad",
                            "response_received": True,
                            "json_body": {
                                "code": 200,
                                "status": "OK",
                                "data": {
                                    "number": 262,
                                    "text": "GOD - there is no deity save Him...",
                                },
                            },
                        },
                    }
                ),
            },
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is True
    assert completion["goal_satisfied"] is True
    assert completion["error_code"] == ""
    assert completion["missing_outcomes"] == []
    assert completion["missing_actions"] == []
    assert "api_execution_verified" in completion["completed_outcomes"]
    assert "user_goal_verified" in completion["completed_outcomes"]
    assert completion["execution_evidence"]["execution_verified"] is True
    assert completion["execution_evidence"]["status_code"] == 200


def test_api_workflow_saved_integration_succeeds_without_docs_or_preview() -> None:
    """Saved integration requires only operation search + execute, no docs or preview."""
    response = SimpleNamespace(
        trace=[
            {
                "tool_name": "api_workflow_decide",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "task_intent": "fetch user profile",
                            "required_outcomes": [
                                "api_execution_verified",
                                "user_goal_verified",
                            ],
                            "optional_outcomes": [
                                "operation_resolved",
                            ],
                            "reason": "Saved integration is available.",
                            "safe_to_continue": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_operations_search",
                "status": "ok",
                "output_preview": '{"ok":true,"result":[{"operation_id":"getUser"}]}',
            },
            {
                "tool_name": "api_request_execute",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "executed": True,
                            "upstream_ok": True,
                            "status_code": 200,
                            "method": "GET",
                            "redacted_url": "https://api.crm.test/v1/users/1",
                            "response_received": True,
                            "json_body": {"id": 1, "name": "Ali"},
                        },
                    }
                ),
            },
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is True
    assert completion["goal_satisfied"] is True
    assert completion["missing_outcomes"] == []
    assert "documentation_understood" not in completion["missing_outcomes"]
    assert "request_previewed" not in completion["missing_outcomes"]


def test_api_workflow_docs_only_task_does_not_require_execution() -> None:
    """Documentation explanation task requires only documentation_understood."""
    response = SimpleNamespace(
        trace=[
            {
                "tool_name": "api_workflow_decide",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "task_intent": "explain API authentication mechanisms",
                            "required_outcomes": [
                                "documentation_understood",
                            ],
                            "reason": "User requested documentation explanation.",
                            "safe_to_continue": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_docs_inspect",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "reference": "https://api.test/docs",
                            "documentation_ref": "sha256:" + ("c" * 64),
                            "text": "Bearer token authentication is supported.",
                            "truncated": False,
                            "more_available": False,
                        },
                    }
                ),
            },
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is True
    assert completion["goal_satisfied"] is True
    assert completion["error_code"] == ""
    assert "api_execution_verified" not in completion["required_outcomes"]
    assert completion["missing_outcomes"] == []


def test_api_workflow_no_execution_fails_for_api_task() -> None:
    """When execution was required, inspecting docs/searching without calling API fails."""
    response = SimpleNamespace(
        trace=[
            {
                "tool_name": "api_workflow_decide",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "task_intent": "fetch contact info",
                            "required_outcomes": [
                                "api_execution_verified",
                                "user_goal_verified",
                            ],
                            "reason": "Execution required.",
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
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is False
    assert completion["goal_satisfied"] is False
    assert completion["error_code"] == "api_workflow_incomplete"
    assert "api_execution_verified" in completion["missing_outcomes"]
    assert "user_goal_verified" in completion["missing_outcomes"]


def test_api_workflow_mutation_requires_preview_or_approval_policy() -> None:
    """Mutations (POST/PUT/PATCH/DELETE) without preview or approval fail execution verification."""
    response = SimpleNamespace(
        trace=[
            {
                "tool_name": "api_workflow_decide",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "task_intent": "delete contact 123",
                            "required_outcomes": [
                                "api_execution_verified",
                                "user_goal_verified",
                            ],
                            "reason": "Delete contact.",
                            "safe_to_continue": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_request_execute",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "executed": True,
                            "upstream_ok": True,
                            "status_code": 204,
                            "method": "DELETE",
                            "redacted_url": "https://api.test/contacts/123",
                        },
                    }
                ),
            },
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    # DELETE without preview or approval fails policy check -> execution not verified
    assert completion["valid"] is False
    assert "api_execution_verified" in completion["missing_outcomes"]


def test_api_workflow_legacy_decisions_rejected_without_explicit_migration() -> None:
    """Historical decision dictionaries with only required_actions fail runtime validation."""
    response = SimpleNamespace(
        trace=[
            {
                "tool_name": "api_workflow_decide",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "task_intent": "legacy task",
                            "required_actions": ["request_execution"],
                            "reason": "Legacy decision.",
                            "safe_to_continue": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_request_execute",
                "status": "ok",
                "output_preview": '{"ok":true,"result":{"executed":true,"status_code":200}}',
            },
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is False
    assert completion["error_code"] == "api_workflow_decision_invalid"


def test_api_workflow_user_goal_verified_fails_when_response_does_not_satisfy() -> None:
    """HTTP 200 response with goal_satisfied=False leaves user_goal_verified incomplete."""
    response = SimpleNamespace(
        trace=[
            {
                "tool_name": "api_workflow_decide",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "task_intent": "fetch verse 2:255",
                            "required_outcomes": [
                                "api_execution_verified",
                                "user_goal_verified",
                            ],
                            "reason": "Execution required.",
                            "safe_to_continue": True,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_request_execute",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "executed": True,
                            "upstream_ok": True,
                            "status_code": 200,
                            "method": "GET",
                            "redacted_url": "https://api.test/data",
                            "goal_satisfied": False,
                            "user_goal_verified": False,
                        },
                    }
                ),
            },
        ]
    )

    completion = _api_workflow_completion_from_trace(response)

    assert completion["valid"] is False
    assert completion["goal_satisfied"] is False
    assert "user_goal_verified" in completion["missing_outcomes"]
    assert "api_execution_verified" in completion["completed_outcomes"]


def test_api_workflow_durable_task_projection_persists_events_and_evidence(
    tmp_path: Path,
) -> None:
    """Lane task coordinator receives projected tool events and normalized outcomes."""
    attached_records = []

    class MockLaneCoordinator:
        def __init__(self):
            self.taskboard = None

        def attach_evidence(self, task_id: str, evidence: dict[str, Any]):
            attached_records.append((task_id, evidence))

        def transition(self, task_id: str, state: Any, reason: str = ""):
            pass

    coordinator = MockLaneCoordinator()

    class ModelToolExecutor:
        def run(self, **kwargs):
            return SimpleNamespace(
                answer="Quran API returned verse 2:255.",
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
                                    "task_intent": "fetch ayah 2:255",
                                    "required_outcomes": [
                                        "api_execution_verified",
                                        "user_goal_verified",
                                    ],
                                    "reason": "Execute Quran API.",
                                    "safe_to_continue": True,
                                },
                            }
                        ),
                    },
                    {
                        "tool_name": "api_request_execute",
                        "status": "ok",
                        "output_preview": json.dumps(
                            {
                                "ok": True,
                                "result": {
                                    "executed": True,
                                    "upstream_ok": True,
                                    "status_code": 200,
                                    "method": "GET",
                                    "redacted_url": "https://api.alquran.cloud/v1/ayah/2:255/en.asad",
                                },
                            }
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
    gateway._lane_coordinator = coordinator
    gateway.config = SimpleNamespace(agent_max_steps=8)

    result = gateway._execute_api_route(
        decision=EntryRoutingDecision(
            route="api",
            confidence=0.99,
            reason="Fetch Quran ayah.",
            required_sources=("api",),
        ),
        context=EntryRouteContext(
            session_id="session-api",
            conversation_id="session-api",
            turn_id="turn-api",
        ),
        text="Fetch verse 2:255 from Quran API.",
        ask_service=SimpleNamespace(ask_agent=ModelToolExecutor()),
        callbacks=None,
        lane_task_id="lane-task-quran-1",
    )

    assert result.mode == "route-api"
    assert len(attached_records) == 1
    task_id, evidence = attached_records[0]
    assert task_id == "lane-task-quran-1"
    assert evidence["required_outcomes"] == ["api_execution_verified", "user_goal_verified"]
    assert "api_execution_verified" in evidence["completed_outcomes"]
    assert evidence["execution_evidence"]["status_code"] == 200
    assert len(evidence["actual_tool_events"]) == 1
    assert evidence["actual_tool_events"][0]["tool_name"] == "api_request_execute"


def test_api_workflow_surfaces_specific_blocker_error_codes() -> None:
    for code in ("validation_failure", "ambiguous_operation", "missing_credential", "blocked_host"):
        response = SimpleNamespace(
            answer="Blocked by specific error.",
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
                                "task_intent": "call api",
                                "required_outcomes": ["api_execution_verified", "user_goal_verified"],
                                "safe_to_continue": True,
                            },
                        }
                    ),
                },
                {
                    "tool_name": "api_request_preview",
                    "status": "error",
                    "output_preview": json.dumps(
                        {
                            "ok": False,
                            "error_code": code,
                            "message": f"Operation failed due to {code}",
                        }
                    ),
                },
            ],
        )
        completion = _api_workflow_completion_from_trace(response)
        assert completion["valid"] is False
        assert completion["error_code"] == code
        assert f"API workflow blocked ({code})" in completion["message"]


def test_api_route_sefaria_post_preview_awaits_approval(tmp_path: Path) -> None:
    class SefariaAskAgent:
        def run(self, *, system_prompt: str, **kwargs):
            return SimpleNamespace(
                answer="The annotation has been previewed and is awaiting local approval before execution.",
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
                                    "task_intent": "create annotation on Sefaria text",
                                    "required_outcomes": [
                                        "api_target_resolved",
                                        "operation_resolved",
                                        "api_execution_verified",
                                        "user_goal_verified",
                                    ],
                                    "reason": "Mutation requires execution evidence and goal verification.",
                                    "safe_to_continue": True,
                                },
                            }
                        ),
                    },
                    {
                        "tool_name": "api_operations_search",
                        "status": "ok",
                        "output_preview": json.dumps(
                            {
                                "ok": True,
                                "result": [
                                    {
                                        "operation_id": "create_annotation",
                                        "method": "POST",
                                        "risk_level": "unknown_high_risk",
                                    }
                                ],
                            }
                        ),
                    },
                    {
                        "tool_name": "api_request_preview",
                        "status": "ok",
                        "output_preview": json.dumps(
                            {
                                "ok": True,
                                "result": {
                                    "risk_level": "unknown_high_risk",
                                    "permission_required": True,
                                    "permission_scope": "api.request.execute",
                                    "permission_request_id": "req_sefaria_approval_001",
                                    "approval_required": True,
                                },
                            }
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
    gateway._lane_coordinator = None
    gateway.config = SimpleNamespace(agent_max_steps=8)

    result = gateway._execute_api_route(
        decision=EntryRoutingDecision(
            route="api",
            confidence=0.99,
            reason="Annotate Sefaria text.",
            required_sources=("api",),
        ),
        context=EntryRouteContext(
            session_id="session-sefaria",
            conversation_id="session-sefaria",
            turn_id="turn-sefaria",
        ),
        text="Create an annotation on Sefaria Genesis 1:1.",
        ask_service=SimpleNamespace(ask_agent=SefariaAskAgent()),
        callbacks=None,
    )

    assert result.mode == "route-api-awaiting-approval"
    assert result.error is None
    assert len(result.payload.get("permission_requests", [])) == 1
    req = result.payload["permission_requests"][0]
    assert req["permission_request_id"] == "req_sefaria_approval_001"
    assert req["permission_scope"] == "api.request.execute"


def _setup_test_approval_environment(tmp_path: Path, monkeypatch: Any | None = None):
    import os
    if monkeypatch is not None:
        monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    else:
        os.environ["MANA_HOME"] = str(tmp_path / "home")
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    from mana_agent.gateway.lanes import ACTIVE_LANE_STATES, LaneId, LaneTaskState
    from mana_agent.gateway.lane_coordinator import LaneCoordinator
    from mana_agent.api_manager.runtime_tools import api_manager_service
    from mana_agent.api_manager.models import HttpMethod, OperationRiskLevel
    from mana_agent.api_manager.request_builder import BuiltApiRequest, RequestPreview

    events: list[tuple[str, str, dict]] = []

    def sink(event_type: str, title: str = "", **kwargs: Any) -> None:
        events.append((event_type, title, kwargs))

    class ContinuationAskAgent:
        def __init__(self):
            self.calls = []

        def run(self, *, question: str, system_prompt: str, **kwargs):
            self.calls.append({"question": question, "kwargs": kwargs})
            return SimpleNamespace(
                answer="خلاصه فارسی: متن با موفقیت دریافت شد.",
                sources=[],
                warnings=[],
                trace=[],
            )

    ask_agent = ContinuationAskAgent()
    coordinator = LaneCoordinator(tmp_path)
    gateway = object.__new__(AgentChatGateway)
    gateway.root = tmp_path
    gateway._index_dir = None
    gateway._resolved_k = 4
    gateway._agent_timeout_seconds = 30
    gateway._event_sink = sink
    gateway.config = SimpleNamespace(agent_max_steps=8)
    gateway._stack = SimpleNamespace(ask_service=SimpleNamespace(ask_agent=ask_agent))
    gateway._sessions = {}
    gateway._lane_coordinator = coordinator
    gateway._synchronize_lane_usage = lambda task_id: {}

    reservation = coordinator.reserve(
        normalized_intent="Sefaria API request",
        lane_id=LaneId.OPERATIONS,
        session_id="conv_sefaria_session",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
        requested_input_tokens=100,
        requested_output_tokens=200,
    )
    lane_task_id = reservation.execution.task_id
    coordinator.transition(
        lane_task_id,
        LaneTaskState.RUNNING,
        reason="API request running",
    )
    coordinator.transition(
        lane_task_id,
        LaneTaskState.WAITING,
        reason="API request waiting for trusted local approval",
    )

    service = api_manager_service(tmp_path)
    req = BuiltApiRequest(
        integration_id="api_sefaria_integration",
        operation_id="post_sefaria_annotation",
        method="POST",
        url="https://www.sefaria.org/api/v3/texts/Genesis.1.1",
        timeout_seconds=30.0,
        risk_level=OperationRiskLevel.UPDATE,
        session_id="conv_sefaria_session",
        routing_task_intent="ایجاد حاشیه‌نویسی در متن سفاریا",
    )
    prev = RequestPreview(
        integration_id="api_sefaria_integration",
        integration_name="Sefaria",
        operation_id="post_sefaria_annotation",
        operation_name="Post Annotation",
        method="POST",
        redacted_url="https://www.sefaria.org/api/v3/texts/Genesis.1.1",
        redacted_headers={},
        query_parameters={},
        body_summary={"text": "annotation body"},
        expected_side_effects="Create annotation.",
        risk_level=OperationRiskLevel.UPDATE,
        approval_required=True,
    )
    details = service.approvals.prepare(
        req,
        prev,
        session_id="conv_sefaria_session",
        conversation_id="conv_sefaria_session",
        turn_id="turn_sefaria_post",
        task_intent="ایجاد حاشیه‌نویسی در متن سفاریا",
        lane_task_id=lane_task_id,
    )
    approval_id = details["permission_request_id"]
    return gateway, coordinator, lane_task_id, service, approval_id, ask_agent, events


def test_api_workflow_post_approval_upstream_failure_finalizes_lane_and_emits_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mana_agent.gateway.lanes import ACTIVE_LANE_STATES, LaneTaskState
    from types import MethodType

    gateway, coordinator, lane_task_id, service, approval_id, ask_agent, events = (
        _setup_test_approval_environment(tmp_path, monkeypatch)
    )

    def mock_execute_fail(self, request, preview, **kwargs):
        return SimpleNamespace(
            executed=True,
            upstream_ok=False,
            status_code=400,
            json_body={"error": "Field 'user_id' is required"},
            text_body="Bad Request",
            model_dump=lambda mode="json": {
                "executed": True,
                "upstream_ok": False,
                "status_code": 400,
                "json_body": {"error": "Field 'user_id' is required"},
                "text_body": "Bad Request",
            },
        )

    service._execute_prepared_request = MethodType(mock_execute_fail, service)

    result = gateway.api_approval_command(
        approval_id,
        session_id="conv_sefaria_session",
        client_type="dashboard",
    )

    assert result["status"] == "failed"
    assert result["upstream_ok"] is False
    assert result["executed"] is True
    assert "400" in result["answer"]
    assert "Field 'user_id' is required" in result["answer"]
    assert len(ask_agent.calls) == 0

    assert coordinator.inspect_task(lane_task_id).state == LaneTaskState.FAILED
    active = [t for t in coordinator.executions if t.state in ACTIVE_LANE_STATES]
    assert len(active) == 0

    event_types = [e[0] for e in events]
    assert "api.call.failed" in event_types
    assert "turn.finished" in event_types

    duplicate_result = gateway.api_approval_command(
        approval_id,
        session_id="conv_sefaria_session",
        client_type="dashboard",
    )
    assert duplicate_result["status"] == "failed"
    assert len(ask_agent.calls) == 0


def test_api_workflow_post_approval_transport_exception_finalizes_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mana_agent.gateway.lanes import ACTIVE_LANE_STATES, LaneTaskState
    from types import MethodType

    gateway, coordinator, lane_task_id, service, approval_id, ask_agent, events = (
        _setup_test_approval_environment(tmp_path, monkeypatch)
    )

    def mock_execute_exc(self, request, preview, **kwargs):
        raise ConnectionError("DNS resolution failed for api.sefaria.org")

    service._execute_prepared_request = MethodType(mock_execute_exc, service)

    result = gateway.api_approval_command(
        approval_id,
        session_id="conv_sefaria_session",
        client_type="dashboard",
    )

    assert result["status"] == "failed"
    assert result["executed"] is False
    assert result["upstream_ok"] is False
    assert "DNS resolution failed" in result["message"]
    assert len(ask_agent.calls) == 0

    assert coordinator.inspect_task(lane_task_id).state == LaneTaskState.FAILED
    active = [t for t in coordinator.executions if t.state in ACTIVE_LANE_STATES]
    assert len(active) == 0

    event_types = [e[0] for e in events]
    assert "api.call.failed" in event_types
    assert "turn.finished" in event_types

    duplicate_result = gateway.api_approval_command(
        approval_id,
        session_id="conv_sefaria_session",
        client_type="dashboard",
    )
    assert duplicate_result["status"] == "failed"


def test_api_workflow_post_approval_denial_cancels_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mana_agent.gateway.lanes import ACTIVE_LANE_STATES, LaneTaskState

    gateway, coordinator, lane_task_id, service, approval_id, ask_agent, events = (
        _setup_test_approval_environment(tmp_path, monkeypatch)
    )

    result = gateway.deny_api_approval_command(
        approval_id,
        session_id="conv_sefaria_session",
        client_type="dashboard",
    )

    assert result["status"] == "denied"
    assert len(ask_agent.calls) == 0

    assert coordinator.inspect_task(lane_task_id).state == LaneTaskState.CANCELLED
    active = [t for t in coordinator.executions if t.state in ACTIVE_LANE_STATES]
    assert len(active) == 0

    event_types = [e[0] for e in events]
    assert "api.approval_decided" in event_types
    assert "turn.finished" in event_types

    duplicate = gateway.deny_api_approval_command(
        approval_id,
        session_id="conv_sefaria_session",
        client_type="dashboard",
    )
    assert duplicate["status"] == "denied"


def test_api_workflow_post_approval_continuation_failure_finalizes_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mana_agent.gateway.lanes import ACTIVE_LANE_STATES, LaneTaskState
    from types import MethodType

    gateway, coordinator, lane_task_id, service, approval_id, ask_agent, events = (
        _setup_test_approval_environment(tmp_path, monkeypatch)
    )

    def mock_execute_ok(self, request, preview, **kwargs):
        return SimpleNamespace(
            executed=True,
            upstream_ok=True,
            status_code=200,
            json_body={"ok": True},
            text_body='{"ok": true}',
            model_dump=lambda mode="json": {
                "executed": True,
                "upstream_ok": True,
                "status_code": 200,
                "json_body": {"ok": True},
            },
        )

    service._execute_prepared_request = MethodType(mock_execute_ok, service)

    def failing_run(*args, **kwargs):
        raise RuntimeError("Continuation LLM timeout")

    ask_agent.run = failing_run

    result = gateway.api_approval_command(
        approval_id,
        session_id="conv_sefaria_session",
        client_type="dashboard",
    )

    assert result["status"] == "failed"
    assert result["error"] == "continuation_model_failed"
    assert "Continuation LLM timeout" in result["message"]

    assert coordinator.inspect_task(lane_task_id).state == LaneTaskState.FAILED
    active = [t for t in coordinator.executions if t.state in ACTIVE_LANE_STATES]
    assert len(active) == 0


def test_api_workflow_post_approval_reconciles_waiting_parent_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    from mana_agent.gateway.lanes import ACTIVE_LANE_STATES, LaneId, LaneTaskState
    from mana_agent.api_manager.runtime_tools import api_manager_service
    from mana_agent.api_manager.models import HttpMethod, OperationRiskLevel
    from mana_agent.api_manager.request_builder import BuiltApiRequest, RequestPreview
    from mana_agent.gateway.lane_coordinator import LaneCoordinator
    from types import MethodType

    coordinator = LaneCoordinator(tmp_path)
    gateway = object.__new__(AgentChatGateway)
    gateway.root = tmp_path
    gateway._index_dir = None
    gateway._resolved_k = 4
    gateway._agent_timeout_seconds = 30
    gateway._event_sink = lambda *a, **kw: None
    gateway.config = SimpleNamespace(agent_max_steps=8)
    gateway._stack = SimpleNamespace(ask_service=SimpleNamespace(ask_agent=None))
    gateway._sessions = {}
    gateway._lane_coordinator = coordinator
    gateway._synchronize_lane_usage = lambda task_id: {}

    parent_reservation = coordinator.reserve(
        normalized_intent="Parent workflow intent",
        lane_id=LaneId.RESEARCH,
        session_id="conv_sefaria_session",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
        requested_input_tokens=100,
        requested_output_tokens=200,
    )
    parent_task_id = parent_reservation.execution.task_id
    coordinator.transition(
        parent_task_id,
        LaneTaskState.RUNNING,
        reason="Parent running",
    )
    coordinator.transition(
        parent_task_id,
        LaneTaskState.WAITING,
        reason="Parent waiting on child approval",
    )

    child_reservation = coordinator.reserve(
        normalized_intent="Child API mutation",
        lane_id=LaneId.OPERATIONS,
        session_id="conv_sefaria_session",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
        parent_task_id=parent_task_id,
        requested_input_tokens=100,
        requested_output_tokens=200,
    )
    child_task_id = child_reservation.execution.task_id
    coordinator.transition(
        child_task_id,
        LaneTaskState.RUNNING,
        reason="Child running",
    )
    coordinator.transition(
        child_task_id,
        LaneTaskState.WAITING,
        reason="Child waiting on user approval",
    )

    service = api_manager_service(tmp_path)
    req = BuiltApiRequest(
        integration_id="api_sefaria_integration",
        operation_id="post_annotation",
        method="POST",
        url="https://www.sefaria.org/api/v3/texts/Genesis.1.1",
        timeout_seconds=30.0,
        risk_level=OperationRiskLevel.UPDATE,
        session_id="conv_sefaria_session",
        routing_task_intent="Child mutation",
    )
    prev = RequestPreview(
        integration_id="api_sefaria_integration",
        integration_name="Sefaria",
        operation_id="post_annotation",
        operation_name="Post Annotation",
        method="POST",
        redacted_url="https://www.sefaria.org/api/v3/texts/Genesis.1.1",
        redacted_headers={},
        query_parameters={},
        body_summary={},
        expected_side_effects="Create annotation.",
        risk_level=OperationRiskLevel.UPDATE,
        approval_required=True,
    )
    details = service.approvals.prepare(
        req,
        prev,
        session_id="conv_sefaria_session",
        conversation_id="conv_sefaria_session",
        turn_id="turn_child_post",
        task_intent="Child mutation",
        lane_task_id=child_task_id,
    )
    approval_id = details["permission_request_id"]

    def mock_execute_fail(self, request, preview, **kwargs):
        return SimpleNamespace(
            executed=True,
            upstream_ok=False,
            status_code=400,
            json_body={"error": "Validation error"},
            text_body="Bad Request",
            model_dump=lambda mode="json": {
                "executed": True,
                "upstream_ok": False,
                "status_code": 400,
                "json_body": {"error": "Validation error"},
            },
        )

    service._execute_prepared_request = MethodType(mock_execute_fail, service)

    result = gateway.api_approval_command(
        approval_id,
        session_id="conv_sefaria_session",
        client_type="dashboard",
    )

    assert result["status"] == "failed"
    assert coordinator.inspect_task(child_task_id).state == LaneTaskState.FAILED
    assert coordinator.inspect_task(parent_task_id).state == LaneTaskState.FAILED
    active = [t for t in coordinator.executions if t.state in ACTIVE_LANE_STATES]
    assert len(active) == 0

