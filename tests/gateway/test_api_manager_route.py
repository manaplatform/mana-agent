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


def test_api_workflow_recovers_when_workflow_decision_retried_after_validation_error() -> None:
    response = SimpleNamespace(
        trace=[
            {
                "tool_name": "api_workflow_decide",
                "status": "error",
                "output_preview": (
                    "1 validation error for _WorkflowDecision\nrequired_actions\n"
                    "  Input should be a valid tuple [type=tuple_type, input_value='operation_search', input_type=str]"
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
                            "required_actions": ["operation_search"],
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
    assert completion["completed_actions"] == ["operation_search"]


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
            "redacted_url": "https://api.alquran.cloud/v1/search/joined/all/en.sahih",
            "status_code": 200,
            "executed": True,
            "upstream_ok": True,
            "json_body": {"count": 100, "matches": [{"text": "sample"}] * 100},
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
                            "task_intent": "search text and execute API request",
                            "required_actions": [
                                "operation_search",
                                "request_preview",
                                "request_execution",
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
                            "required_actions": [
                                "documentation_inspection",
                                "integration_import",
                                "operation_search",
                                "request_preview",
                                "request_execution",
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

    assert completion["completed_actions"] == [
        "documentation_inspection"
    ]

    assert completion["waived_actions"] == [
        "integration_import",
        "operation_search",
        "request_preview",
        "request_execution",
    ]

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
                            "required_actions": [
                                "documentation_inspection",
                                "integration_import",
                                "operation_search",
                                "request_preview",
                                "request_execution",
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
                            "bytes": 8000,
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
        "documentation_inspection",
        "integration_import",
        "operation_search",
        "request_preview",
        "request_execution",
    ]

    assert completion["terminal_outcome"] == ""
    assert completion["terminal_evidence"] == {}


def test_regression_case_a_full_import_and_call_lifecycle(tmp_path: Path) -> None:
    doc_ref = "urn:mana:doc:test_case_a"
    response = SimpleNamespace(
        answer="Operation executed successfully.",
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
                            "source_decision_id": "dec-a",
                            "task_intent": "import and call",
                            "required_actions": [
                                "documentation_inspection",
                                "integration_import",
                                "operation_search",
                                "request_preview",
                                "request_execution",
                            ],
                            "reason": "Complete declared lifecycle.",
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
                            "documentation_ref": doc_ref,
                            "offset": 0,
                            "text": "openapi spec content",
                            "truncated": False,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_docs_import",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "saved": True,
                            "integration": {"integration_id": "api_integration_a"},
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
                        "result": [{"operation_id": "op_get_data"}],
                    }
                ),
            },
            {
                "tool_name": "api_request_preview",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {"risk_level": "read_only"},
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
                            "integration_id": "api_integration_a",
                            "operation_id": "op_get_data",
                            "method": "GET",
                            "redacted_url": "https://api.example.com/data",
                        },
                    }
                ),
            },
        ],
    )

    completion = _api_workflow_completion_from_trace(response)
    assert completion["valid"] is True
    assert completion["error_code"] == ""
    assert completion["missing_actions"] == []
    assert completion["completed_actions"] == [
        "documentation_inspection",
        "integration_import",
        "operation_search",
        "request_execution",
        "request_preview",
    ]
    assert completion["execution_evidence"]["executed"] is True
    assert completion["execution_evidence"]["status_code"] == 200


def test_regression_case_b_existing_integration_refresh_succeeds(tmp_path: Path) -> None:
    doc_ref = "urn:mana:doc:test_case_b"
    response = SimpleNamespace(
        answer="Refreshed and executed.",
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
                            "source_decision_id": "dec-b",
                            "task_intent": "refresh and call",
                            "required_actions": [
                                "documentation_inspection",
                                "integration_import",
                                "operation_search",
                                "request_preview",
                                "request_execution",
                            ],
                            "reason": "Refresh workflow.",
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
                            "documentation_ref": doc_ref,
                            "offset": 0,
                            "text": "openapi spec content",
                            "truncated": False,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_docs_import",
                "status": "error",
                "output_preview": json.dumps(
                    {
                        "ok": False,
                        "error_code": "integration_already_exists",
                        "refresh_integration_id": "api_existing_123",
                        "message": "Integration 'api_existing_123' already exists.",
                    }
                ),
            },
            {
                "tool_name": "api_docs_import",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {
                            "saved": True,
                            "integration": {"integration_id": "api_existing_123"},
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
                        "result": [{"operation_id": "op_test"}],
                    }
                ),
            },
            {
                "tool_name": "api_request_preview",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {"risk_level": "read_only"},
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
                        },
                    }
                ),
            },
        ],
    )

    completion = _api_workflow_completion_from_trace(response)
    assert completion["valid"] is True
    assert "integration_import" in completion["completed_actions"]
    assert completion["missing_actions"] == []


def test_regression_case_c_refresh_not_completed_fails_incomplete(tmp_path: Path) -> None:
    doc_ref = "urn:mana:doc:test_case_c"
    response = SimpleNamespace(
        answer="Skipped import after duplicate error.",
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
                            "source_decision_id": "dec-c",
                            "task_intent": "import and call",
                            "required_actions": [
                                "documentation_inspection",
                                "integration_import",
                                "operation_search",
                                "request_preview",
                                "request_execution",
                            ],
                            "reason": "Declared import.",
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
                            "documentation_ref": doc_ref,
                            "offset": 0,
                            "text": "openapi spec content",
                            "truncated": False,
                        },
                    }
                ),
            },
            {
                "tool_name": "api_docs_import",
                "status": "error",
                "output_preview": json.dumps(
                    {
                        "ok": False,
                        "error_code": "integration_already_exists",
                        "refresh_integration_id": "api_dup_456",
                    }
                ),
            },
            {
                "tool_name": "api_operations_search",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": [{"operation_id": "op_test"}],
                    }
                ),
            },
            {
                "tool_name": "api_request_preview",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {"risk_level": "read_only"},
                    }
                ),
            },
            {
                "tool_name": "api_request_execute",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {"executed": True, "upstream_ok": True, "status_code": 200},
                    }
                ),
            },
        ],
    )

    completion = _api_workflow_completion_from_trace(response)
    assert completion["valid"] is False
    assert completion["error_code"] == "api_workflow_incomplete"
    assert "integration_import" in completion["missing_actions"]


def test_regression_case_d_read_only_without_preview_fails_incomplete(tmp_path: Path) -> None:
    response = SimpleNamespace(
        answer="Executed GET directly without preview.",
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
                            "source_decision_id": "dec-d",
                            "task_intent": "read only get",
                            "required_actions": [
                                "operation_search",
                                "request_preview",
                                "request_execution",
                            ],
                            "reason": "Read-only GET call.",
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
                        "result": [{"operation_id": "op_get"}],
                    }
                ),
            },
            {
                "tool_name": "api_request_execute",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {"executed": True, "upstream_ok": True, "status_code": 200},
                    }
                ),
            },
        ],
    )

    completion = _api_workflow_completion_from_trace(response)
    assert completion["valid"] is False
    assert completion["error_code"] == "api_workflow_incomplete"
    assert "request_preview" in completion["missing_actions"]
    assert "request_execution" not in completion["completed_actions"]


def test_regression_case_e_read_only_with_preview_succeeds(tmp_path: Path) -> None:
    response = SimpleNamespace(
        answer="Executed GET with preview.",
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
                            "source_decision_id": "dec-e",
                            "task_intent": "read only get with preview",
                            "required_actions": [
                                "operation_search",
                                "request_preview",
                                "request_execution",
                            ],
                            "reason": "Read-only GET call with preview.",
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
                        "result": [{"operation_id": "op_get"}],
                    }
                ),
            },
            {
                "tool_name": "api_request_preview",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {"risk_level": "read_only"},
                    }
                ),
            },
            {
                "tool_name": "api_request_execute",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {"executed": True, "upstream_ok": True, "status_code": 200},
                    }
                ),
            },
        ],
    )

    completion = _api_workflow_completion_from_trace(response)
    assert completion["valid"] is True
    assert completion["error_code"] == ""
    assert completion["missing_actions"] == []


def test_regression_case_f_waiting_approval_yields_awaiting_approval_mode(tmp_path: Path) -> None:
    class ModelToolExecutor:
        def run(self, **kwargs):
            return SimpleNamespace(
                answer="Waiting for user approval.",
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
                                    "source_decision_id": "dec-f",
                                    "task_intent": "mutate API",
                                    "required_actions": [
                                        "operation_search",
                                        "request_preview",
                                        "request_execution",
                                    ],
                                    "reason": "Mutation requires preview and execute.",
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
                                "result": [{"operation_id": "op_post_data"}],
                            }
                        ),
                    },
                    {
                        "tool_name": "api_request_preview",
                        "status": "error",
                        "output_preview": json.dumps(
                            {
                                "ok": False,
                                "error_code": "permission_required",
                                "message": "Approval required for API mutation.",
                                "details": {
                                    "permission_scope": "api.request.execute",
                                    "permission_request_id": "perm_req_789",
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
    gateway.config = SimpleNamespace(agent_max_steps=8)

    result = gateway._execute_api_route(
        decision=EntryRoutingDecision(
            route="api",
            confidence=0.99,
            reason="Mutate via API.",
            required_sources=("api",),
        ),
        context=EntryRouteContext(
            turn_id="turn-f",
            session_id="session-f",
            conversation_id="session-f",
        ),
        text="Update API record.",
        ask_service=SimpleNamespace(ask_agent=ModelToolExecutor()),
        callbacks=None,
    )

    assert result.mode == "route-api-awaiting-approval"
    assert result.error is None
    assert len(result.payload["permission_requests"]) > 0


def test_regression_case_g_diagnostic_durability_and_projection(tmp_path: Path) -> None:
    response = SimpleNamespace(
        answer="Stopped early.",
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
                            "source_decision_id": "dec-g",
                            "task_intent": "inspect and call",
                            "required_actions": [
                                "documentation_inspection",
                                "integration_import",
                                "operation_search",
                                "request_preview",
                                "request_execution",
                            ],
                            "reason": "Complete lifecycle.",
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
                            "documentation_ref": "urn:mana:doc:g",
                            "offset": 0,
                            "text": "docs",
                            "truncated": False,
                        },
                    }
                ),
            },
        ],
    )

    completion = _api_workflow_completion_from_trace(response)
    assert completion["valid"] is False
    assert completion["error_code"] == "api_workflow_incomplete"
    assert completion["workflow_decision_id"] == "dec-g"
    assert completion["last_successful_action"] == "documentation_inspection"
    assert completion["last_api_tool"] == "api_docs_inspect"
    assert completion["completed_actions"] == ["documentation_inspection"]
    assert completion["missing_actions"] == [
        "integration_import",
        "operation_search",
        "request_preview",
        "request_execution",
    ]


def test_regression_case_h_no_secret_leakage(tmp_path: Path) -> None:
    secret_key = "sk-super-secret-token-12345"
    response = SimpleNamespace(
        answer="Executed with headers.",
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
                            "source_decision_id": "dec-h",
                            "task_intent": "call api",
                            "required_actions": [
                                "operation_search",
                                "request_preview",
                                "request_execution",
                            ],
                            "reason": "Call API.",
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
                        "result": [{"operation_id": "op_h"}],
                    }
                ),
            },
            {
                "tool_name": "api_request_preview",
                "status": "ok",
                "output_preview": json.dumps(
                    {
                        "ok": True,
                        "result": {"risk_level": "read_only"},
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
                            "authorization": f"Bearer {secret_key}",
                            "api_key": secret_key,
                            "headers": {"Authorization": f"Bearer {secret_key}"},
                        },
                    }
                ),
            },
        ],
    )

    completion = _api_workflow_completion_from_trace(response)
    evidence = completion["execution_evidence"]
    dumped = json.dumps(evidence)
    assert secret_key not in dumped
    assert evidence.get("authorization") == "[REDACTED]" or "authorization" not in evidence


def test_regression_quran_api_duplicate_import_and_lifecycle_recovery() -> None:
    """Test full Quran API reproduction: duplicate import recovery, search, preview, and execution."""
    from mana_agent.api_manager.workflow import ApiWorkflowController

    controller = ApiWorkflowController()

    # Step 1: Workflow decision
    controller.record_tool_trace({
        "tool_name": "api_workflow_decide",
        "status": "ok",
        "result": {
            "ok": True,
            "result": {
                "source_decision_id": "dec-quran-001",
                "task_intent": "fetch surah list from quran api",
                "required_actions": [
                    "documentation_inspection",
                    "integration_import",
                    "operation_search",
                    "request_preview",
                    "request_execution",
                ],
                "safe_to_continue": True,
            },
        },
    })
    assert controller.next_required_action() == "documentation_inspection"
    prompt_1 = controller.format_continuation_prompt()
    assert "documentation_inspection" in prompt_1

    # Step 2: Documentation inspection
    controller.record_tool_trace({
        "tool_name": "api_docs_inspect",
        "status": "ok",
        "result": {
            "ok": True,
            "result": {
                "documentation_ref": "https://api.quran.com/docs/v4",
                "offset": 0,
                "text": "Quran.com API v4 documentation contents...",
                "truncated": False,
            },
        },
    })
    assert controller.completed_actions == {"documentation_inspection"}
    assert controller.next_required_action() == "integration_import"

    # Step 3: Duplicate import attempt returns integration_already_exists with refresh_integration_id
    controller.record_tool_trace({
        "tool_name": "api_docs_import_semantic",
        "status": "error",
        "result": {
            "ok": False,
            "error_code": "integration_already_exists",
            "details": {
                "refresh_integration_id": "quran_v4_api",
            },
        },
    })
    assert controller.refresh_integration_id == "quran_v4_api"
    assert "integration_import" not in controller.completed_actions
    assert controller.next_required_action() == "integration_import"
    prompt_retry = controller.format_continuation_prompt()
    assert "refresh_integration_id='quran_v4_api'" in prompt_retry or "quran_v4_api" in prompt_retry

    # Step 4: Refresh import succeeds
    controller.record_tool_trace({
        "tool_name": "api_docs_import_semantic",
        "status": "ok",
        "result": {
            "ok": True,
            "result": {
                "integration": {
                    "integration_id": "quran_v4_api",
                    "name": "Quran API v4",
                },
                "operation_count": 15,
            },
        },
    })
    assert "integration_import" in controller.completed_actions
    assert controller.integration_id == "quran_v4_api"
    assert controller.next_required_action() == "operation_search"

    # Step 5: Operation search succeeds
    controller.record_tool_trace({
        "tool_name": "api_operations_search",
        "status": "ok",
        "result": {
            "ok": True,
            "result": [
                {
                    "operation_id": "get_chapters",
                    "integration_id": "quran_v4_api",
                    "summary": "Get list of chapters",
                }
            ],
        },
    })
    assert "operation_search" in controller.completed_actions
    assert controller.operation_id == "get_chapters"
    assert controller.next_required_action() == "request_preview"

    # Step 6: Request preview succeeds (mandatory before execution)
    controller.record_tool_trace({
        "tool_name": "api_request_preview",
        "status": "ok",
        "result": {
            "ok": True,
            "result": {
                "operation_id": "get_chapters",
                "integration_id": "quran_v4_api",
                "method": "GET",
                "url": "https://api.quran.com/api/v4/chapters",
            },
        },
    })
    assert "request_preview" in controller.completed_actions
    assert controller.next_required_action() == "request_execution"

    # Step 7: Request execution succeeds
    controller.record_tool_trace({
        "tool_name": "api_request_execute",
        "status": "ok",
        "result": {
            "ok": True,
            "result": {
                "executed": True,
                "upstream_ok": True,
                "status_code": 200,
                "integration_id": "quran_v4_api",
                "operation_id": "get_chapters",
                "method": "GET",
                "redacted_url": "https://api.quran.com/api/v4/chapters",
            },
        },
    })
    assert "request_execution" in controller.completed_actions
    assert controller.next_required_action() is None

    result = controller.evaluate()
    assert result["valid"] is True
    assert result["error_code"] == ""
    assert len(result["actual_tool_events"]) >= 6
    assert any(ev["error_code"] == "integration_already_exists" for ev in result["actual_tool_events"])


def test_regression_api_workflow_stalled_on_repeated_no_progress() -> None:
    """Test that repeated no-progress attempts terminate as api_workflow_stalled instead of continuing."""
    from mana_agent.api_manager.workflow import ApiWorkflowController

    controller = ApiWorkflowController()
    controller.record_tool_trace({
        "tool_name": "api_workflow_decide",
        "status": "ok",
        "result": {
            "ok": True,
            "result": {
                "source_decision_id": "dec-stall-001",
                "task_intent": "stall reproduction",
                "required_actions": [
                    "documentation_inspection",
                    "integration_import",
                    "operation_search",
                    "request_preview",
                    "request_execution",
                ],
                "safe_to_continue": True,
            },
        },
    })

    # 3 repeated failing attempts with no progress
    for _ in range(3):
        controller.record_tool_trace({
            "tool_name": "api_docs_inspect",
            "status": "error",
            "result": {
                "ok": False,
                "error_code": "network_timeout",
            },
        })

    assert controller.is_stalled is True
    assert controller.stalled_action == "documentation_inspection"
    evaluation = controller.evaluate()
    assert evaluation["valid"] is False
    assert evaluation["error_code"] == "api_workflow_stalled"
    assert evaluation["stalled"] is True
    assert "stalled at action 'documentation_inspection'" in evaluation["message"]


def test_regression_supervisor_persists_api_workflow_and_actual_tool_events(tmp_path: Path) -> None:
    """Test that ExecutionSupervisor and TaskRecord accurately preserve actual_tool_events and api_workflow."""
    from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
    from mana_agent.execution_supervisor.models import ExecutionState, SideEffectClassification
    from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig

    supervisor = ExecutionSupervisor(config=ExecutionSupervisorConfig(root=tmp_path / "supervisor"))
    task = supervisor.create_task(
        routing_decision_id="dec-task-001",
        side_effect_classification=SideEffectClassification.READ_ONLY,
        workspace_path=tmp_path,
    )

    metadata = {
        "api_workflow": {
            "decision_id": "dec-task-001",
            "completed_actions": ["documentation_inspection"],
            "missing_actions": ["integration_import"],
        },
        "actual_tool_events": [
            {
                "type": "api_tool_event",
                "tool_name": "api_docs_inspect",
                "action": "documentation_inspection",
                "status": "ok",
                "timestamp": "2026-08-27T19:00:00Z",
            }
        ],
    }

    supervisor.persist_provider_metadata(task.task_id, metadata)
    stored_task = supervisor.store.get_task(task.task_id)
    assert stored_task.api_workflow["decision_id"] == "dec-task-001"
    assert len(stored_task.actual_tool_events) == 1
    assert stored_task.actual_tool_events[0]["tool_name"] == "api_docs_inspect"

    escrow = supervisor.record_terminal_result(
        task.task_id,
        state=ExecutionState.FAILED,
        reason="api_workflow_incomplete",
        provider_metadata=metadata,
    )
    assert escrow.payload["api_workflow"]["decision_id"] == "dec-task-001"