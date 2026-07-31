from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

from mana_agent.api_manager.discovery import ApiOperationDiscovery, ApiRouteDecision
from mana_agent.api_manager.documentation import DocumentationImporter
from mana_agent.api_manager.errors import (
    AmbiguousOperationError,
    ApiRateLimitError,
    ApiTimeoutError,
    DocumentationAuthorizationRequiredError,
    PermissionRequiredError,
    RequestValidationError,
    ResponseTooLargeError,
    SsrfPolicyViolationError,
    UnsupportedDocumentationError,
    UpstreamApiError,
)
from mana_agent.api_manager.executor import (
    ApiExecutor,
    NetworkAccessPolicy,
    PendingApiApprovalBroker,
    _RawResponse,
    validate_network_target,
)
from mana_agent.api_manager.models import (
    ApiIntegration,
    ApiOperation,
    ApiParameter,
    ApiServer,
    AuthenticationConfig,
    AuthenticationType,
    DocumentationSource,
    DocumentationSourceType,
    HttpMethod,
    OperationRiskLevel,
    ParameterLocation,
    RequestBody,
    RetryPolicy,
)
from mana_agent.api_manager.redaction import redact_mapping, redact_url
from mana_agent.api_manager.registry import ApiIntegrationRegistry
from mana_agent.api_manager.request_builder import ApiRequestBuilder
from mana_agent.api_manager.runtime_tools import (
    API_MANAGER_TOOL_NAMES,
    _WorkflowDecision,
    build_api_manager_langchain_tools,
)
from mana_agent.api_manager.service import ApiManagerService


OPENAPI: dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {"title": "Acme CRM", "version": "1.0", "description": "Contacts API"},
    "servers": [{"url": "https://api.acme.example/v1"}],
    "components": {
        "securitySchemes": {
            "bearer": {"type": "http", "scheme": "bearer"},
        },
        "schemas": {
            "Contact": {
                "type": "object",
                "required": ["email"],
                "additionalProperties": False,
                "properties": {
                    "email": {"type": "string"},
                    "name": {"type": "string"},
                },
            }
        },
    },
    "security": [{"bearer": []}],
    "paths": {
        "/contacts/{contact_id}": {
            "parameters": [
                {
                    "name": "contact_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
            "get": {
                "operationId": "getContact",
                "summary": "Get contact",
                "responses": {
                    "200": {
                        "description": "Contact",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Contact"}
                            }
                        },
                    }
                },
            },
            "patch": {
                "summary": "Update contact",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Contact"}
                        }
                    },
                },
                "responses": {"200": {"description": "Updated"}},
            },
        },
        "/uploads": {
            "post": {
                "operationId": "uploadFile",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "required": ["file"],
                                "properties": {
                                    "file": {"type": "string", "format": "binary"}
                                },
                            }
                        }
                    },
                },
                "responses": {"201": {"description": "Uploaded"}},
            }
        },
    },
}


def _import() -> ApiIntegration:
    return DocumentationImporter().from_text(
        json.dumps(OPENAPI),
        name="Acme CRM",
        source_decision_id="decision-1",
    )


def _registry(tmp_path: Path) -> tuple[ApiIntegrationRegistry, ApiIntegration]:
    registry = ApiIntegrationRegistry(tmp_path / "integrations")
    integration = _import()
    registry.save(integration)
    return registry, integration


def _public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )


def test_openapi_json_and_yaml_import_have_stable_operation_ids() -> None:
    importer = DocumentationImporter()
    json_integration = importer.from_text(
        json.dumps(OPENAPI),
        name="Acme CRM",
        source_decision_id="decision-json",
    )
    yaml_integration = importer.from_bytes(
        yaml.safe_dump(OPENAPI).encode(),
        name="Acme CRM",
        reference="openapi.yaml",
        source_type=DocumentationSourceType.LOCAL_FILE,
        format_hint="yaml",
        source_decision_id="decision-yaml",
    )
    assert json_integration.integration_id == yaml_integration.integration_id
    assert [item.operation_id for item in json_integration.operations] == [
        item.operation_id for item in yaml_integration.operations
    ]
    assert json_integration.operations[0].operation_id == "getContact"
    generated = next(item for item in json_integration.operations if item.method is HttpMethod.PATCH)
    assert generated.operation_id.startswith("op_")
    assert generated.request_body_schema["required"] == ["email"]


def test_swagger_2_import_normalizes_server_body_and_basic_auth() -> None:
    swagger = {
        "swagger": "2.0",
        "info": {"title": "Legacy", "version": "1"},
        "host": "legacy.example.com",
        "basePath": "/api",
        "schemes": ["https"],
        "securityDefinitions": {"basicAuth": {"type": "basic"}},
        "security": [{"basicAuth": []}],
        "paths": {
            "/items": {
                "post": {
                    "operationId": "createItem",
                    "parameters": [
                        {
                            "name": "body",
                            "in": "body",
                            "required": True,
                            "schema": {
                                "type": "object",
                                "required": ["name"],
                                "properties": {"name": {"type": "string"}},
                            },
                        }
                    ],
                    "responses": {"201": {"description": "Created"}},
                }
            }
        },
    }
    integration = DocumentationImporter().from_text(
        yaml.safe_dump(swagger),
        name="Legacy",
        source_decision_id="swagger-decision",
    )
    operation = integration.operations[0]
    assert operation.base_url == "https://legacy.example.com/api"
    assert operation.request_body is not None and operation.request_body.required
    assert operation.authentication[0].type is AuthenticationType.BASIC


def test_unstructured_import_requires_validated_semantic_definition() -> None:
    importer = DocumentationImporter()
    with pytest.raises(UnsupportedDocumentationError, match="No heuristic or fallback"):
        importer.from_text(
            "GET /contacts returns a contact.",
            name="Acme",
            source_decision_id="decision-1",
        )
    integration = importer.from_text(
        "GET /contacts returns a contact.",
        name="Acme",
        source_decision_id="decision-2",
        semantic_definition={
            "servers": [{"url": "https://api.acme.example"}],
            "operations": [
                {
                    "operation_id": "listContacts",
                    "name": "List contacts",
                    "method": "GET",
                    "path": "/contacts",
                    "base_url": "https://api.acme.example",
                    "risk_level": "read_only",
                    "source_reference": "pasted-text#line-1",
                    "inferred_fields": ["operation_id", "response_schema"],
                    "unresolved_fields": ["response_schema"],
                }
            ],
        },
    )
    assert integration.operations[0].inferred_fields
    assert integration.operations[0].unresolved_fields == ("response_schema",)


def test_unstructured_import_accepts_documented_fields_and_normalizes_auth_parameter(
) -> None:
    source_url = "https://docs.example.com/ip-api/quickstart"
    semantic_definition = {
        "servers": [{"url": "https://api.example.com"}],
        "operations": [
            {
                "operation_id": "ip.standard_lookup",
                "name": "Standard IP Lookup",
                "method": "GET",
                "path": "/{ip}",
                "base_url": "https://api.example.com",
                "parameters": [
                    {
                        "name": "ip",
                        "location": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                    {
                        "name": "access_key",
                        "location": "query",
                        "required": True,
                        "schema": {"type": "string", "format": "password"},
                    },
                ],
                "authentication": [
                    {
                        "type": "api_key_query",
                        "scheme_name": "Access key",
                        "parameter_name": "access_key",
                        "credential_reference": "env://IP_API_TOKEN",
                        "required": True,
                    }
                ],
                "risk_level": "read_only",
                "source_reference": source_url,
                "inferred_fields": [],
                "unresolved_fields": [],
            }
        ],
        "authentication": [
            {
                "type": "api_key_query",
                "scheme_name": "Access key",
                "parameter_name": "access_key",
                "credential_reference": "env://IP_API_TOKEN",
                "required": True,
            }
        ],
    }

    integration = DocumentationImporter().from_text(
        f"Quickstart: {source_url}. GET /{{ip}} uses access_key authentication.",
        name="IP API",
        source_decision_id="documented-semantic-decision",
        semantic_definition=semantic_definition,
    )

    operation = integration.operations[0]
    assert operation.inferred_fields == ()
    assert [parameter.name for parameter in operation.parameters] == ["ip"]
    assert operation.authentication[0].credential_reference == "env://IP_API_TOKEN"


def test_authentication_rejects_bare_credential_reference() -> None:
    with pytest.raises(ValueError, match="credential_reference must use env://"):
        AuthenticationConfig(
            type=AuthenticationType.API_KEY_QUERY,
            parameter_name="access_key",
            credential_reference="IP_API_TOKEN",
        )


def test_documentation_inspection_returns_complete_authorized_evidence(
    tmp_path: Path,
) -> None:
    service = ApiManagerService(
        tmp_path,
        registry=ApiIntegrationRegistry(tmp_path / "integrations"),
    )
    evidence = service.inspect_documentation(
        text="GET /{ip}?access_key=TOKEN returns IP details."
    )

    assert evidence["reference"] == "pasted-text"
    assert evidence["truncated"] is False
    assert evidence["text"].startswith("GET /{ip}")


def test_read_only_http_request_requires_exact_ui_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _public_dns(monkeypatch)
    document = json.loads(json.dumps(OPENAPI))
    document["servers"] = [{"url": "http://api.acme.example/v1"}]
    integration = DocumentationImporter().from_text(
        json.dumps(document),
        name="HTTP API",
        source_decision_id="http-import-decision",
    )
    registry = ApiIntegrationRegistry(tmp_path / "integrations")
    registry.save(integration)
    registry.update(
        integration.integration_id,
        {"authentication": (AuthenticationConfig(),)},
    )
    broker = PendingApiApprovalBroker()
    executor = ApiExecutor(
        network_policy=NetworkAccessPolicy(allow_http=False),
        approval_broker=broker,
        transport=_Transport(
            [_RawResponse(200, {"content-type": "application/json"}, b'{"ok":true}')]
        ),
    )
    service = ApiManagerService(
        tmp_path,
        registry=registry,
        executor=executor,
    )
    route = ApiRouteDecision(
        source_decision_id="http-call-decision",
        task_intent="retrieve one contact",
        workflow="request_execution",
        integration_id=integration.integration_id,
        operation_id="getContact",
        confidence=0.99,
        matched_terms=("contact",),
        reason="The saved read-only operation exactly matches.",
        safe_to_continue=True,
    )

    with pytest.raises(PermissionRequiredError) as raised:
        service.execute_request(
            routing_decision=route,
            integration_id=integration.integration_id,
            operation_id="getContact",
            path_parameters={"contact_id": "123"},
            session_id="http-session",
        )

    details = raised.value.details
    assert details["preview"]["approval_required"] is True
    assert "unencrypted HTTP" in details["preview"]["expected_side_effects"]
    approved = service.decide_approval(
        details["permission_request_id"],
        session_id="http-session",
        approve=True,
        client_type="tui",
    )
    assert approved["executed"] is True


def test_local_documentation_import_is_confined_to_authorized_roots(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    spec = allowed / "openapi.json"
    spec.write_text(json.dumps(OPENAPI), encoding="utf-8")
    importer = DocumentationImporter(allowed_file_roots=(allowed,))
    assert importer.from_file(
        spec,
        name="Acme CRM",
        source_decision_id="file-decision",
    ).operations
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(OPENAPI), encoding="utf-8")
    with pytest.raises(UnsupportedDocumentationError, match="outside the authorized roots"):
        importer.from_file(
            outside,
            name="Outside",
            source_decision_id="outside-decision",
        )


def test_registry_persists_updates_refresh_and_delete(tmp_path: Path) -> None:
    registry, integration = _registry(tmp_path)
    reloaded = ApiIntegrationRegistry(registry.path).get(integration.integration_id)
    assert reloaded.name == "Acme CRM"
    assert registry.disable(integration.integration_id).enabled is False
    refreshed = registry.refresh(integration.integration_id, _import())
    assert refreshed.active_version == 2
    assert len(refreshed.versions) == 2
    assert registry.delete(integration.integration_id, explicit=True)["deleted"] is True
    assert registry.list() == []


def test_registry_rejects_raw_secret_material(tmp_path: Path) -> None:
    integration = _import().model_copy(
        update={"metadata": {"access_token": "raw-secret"}}
    )
    with pytest.raises(ValueError, match="Raw secret material"):
        ApiIntegrationRegistry(tmp_path / "integrations").save(integration)


def test_request_builder_validates_and_serializes_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, integration = _registry(tmp_path)
    monkeypatch.setenv("ACME_TOKEN", "top-secret-token")
    configured = [
        item.model_copy(update={"credential_reference": "env://ACME_TOKEN"})
        for item in integration.authentication
    ]
    registry.update(integration.integration_id, {"authentication": tuple(configured)})
    builder = ApiRequestBuilder(registry)
    request = builder.build(
        integration_id=integration.integration_id,
        operation_id="getContact",
        path_parameters={"contact_id": "a/b"},
    )
    assert "/contacts/a%2Fb" in request.url
    assert request.headers["Authorization"] == "Bearer top-secret-token"
    preview = builder.preview(request)
    assert preview.redacted_headers["Authorization"] == "[REDACTED]"
    with pytest.raises(RequestValidationError, match="Unknown query"):
        builder.build(
            integration_id=integration.integration_id,
            operation_id="getContact",
            path_parameters={"contact_id": "123"},
            query_parameters={"invented": True},
        )


def test_explicit_request_credential_resolves_known_unconfigured_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, integration = _registry(tmp_path)
    monkeypatch.setenv("ACME_TOKEN", "top-secret-token")
    builder = ApiRequestBuilder(registry)

    with pytest.raises(RequestValidationError, match="Supply an explicit"):
        builder.build(
            integration_id=integration.integration_id,
            operation_id="getContact",
            path_parameters={"contact_id": "123"},
        )

    request = builder.build(
        integration_id=integration.integration_id,
        operation_id="getContact",
        path_parameters={"contact_id": "123"},
        headers={"Accept": "application/json"},
        credential_reference="env://ACME_TOKEN",
    )

    assert request.headers["Accept"] == "application/json"
    assert request.headers["Authorization"] == "Bearer top-secret-token"


def test_json_body_validation_and_multipart_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, integration = _registry(tmp_path)
    monkeypatch.setenv("ACME_TOKEN", "token")
    registry.update(
        integration.integration_id,
        {
            "authentication": tuple(
                item.model_copy(update={"credential_reference": "env://ACME_TOKEN"})
                for item in integration.authentication
            )
        },
    )
    builder = ApiRequestBuilder(registry)
    patch_operation = next(
        item for item in integration.operations if item.method is HttpMethod.PATCH
    )
    with pytest.raises(RequestValidationError, match="missing required properties"):
        builder.build(
            integration_id=integration.integration_id,
            operation_id=patch_operation.operation_id,
            path_parameters={"contact_id": "123"},
            body={"name": "No email"},
        )
    upload = builder.build(
        integration_id=integration.integration_id,
        operation_id="uploadFile",
        body={
            "file": {
                "filename": "hello.txt",
                "content_type": "text/plain",
                "content": "hello",
            }
        },
        content_type="multipart/form-data",
    )
    assert upload.content_type.startswith("multipart/form-data; boundary=")
    assert b'filename="hello.txt"' in (upload.body or b"")
    assert b"hello" in (upload.body or b"")


def test_redaction_covers_headers_query_and_nested_values() -> None:
    assert redact_mapping({"client_secret": "value"}) == {"client_secret": "[REDACTED]"}
    assert "api_key=%5BREDACTED%5D" in redact_url(
        "https://example.com/items?api_key=secret&safe=yes"
    )
    assert redact_mapping({"safe": "prefix secret suffix"}, secret_values=("secret",)) == {
        "safe": "prefix [REDACTED] suffix"
    }


class _Transport:
    def __init__(self, responses: list[_RawResponse]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.urls: list[str] = []

    def send(self, *args: Any, **kwargs: Any) -> _RawResponse:
        self.calls += 1
        self.urls.append(str(args[1]))
        return self.responses.pop(0)


class _TimeoutTransport:
    def send(self, *args: Any, **kwargs: Any) -> _RawResponse:
        raise socket.timeout()


class _OversizeTransport:
    def send(self, *args: Any, **kwargs: Any) -> _RawResponse:
        raise ResponseTooLargeError("mocked response exceeded limit")


def test_read_only_execution_retry_and_structured_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _public_dns(monkeypatch)
    registry, integration = _registry(tmp_path)
    operation = registry.operation(integration.integration_id, "getContact")[1]
    registry.update(
        integration.integration_id,
        {
            "authentication": (AuthenticationConfig(),),
            "operations": tuple(
                item.model_copy(
                    update={"authentication": (AuthenticationConfig(),)}
                )
                if item.operation_id == operation.operation_id
                else item
                for item in integration.operations
            ),
        },
    )
    builder = ApiRequestBuilder(registry)
    request = builder.build(
        integration_id=integration.integration_id,
        operation_id="getContact",
        path_parameters={"contact_id": "123"},
    ).model_copy(
        update={
            "retry_maximum_attempts": 2,
            "retry_statuses": (503,),
            "retry_backoff_seconds": 0,
        }
    )
    transport = _Transport(
        [
            _RawResponse(503, {"content-type": "text/plain"}, b"retry"),
            _RawResponse(200, {"content-type": "application/json"}, b'{"email":"a@example.com"}'),
        ]
    )
    executor = ApiExecutor(transport=transport, sleep=lambda _: None)
    result = executor.execute(request, preview=builder.preview(request))
    assert result.executed is True
    assert result.status_code == 200
    assert result.json_body["email"] == "a@example.com"
    assert result.attempts == 2


def test_timeout_is_structured_and_redirect_target_is_revalidated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _public_dns(monkeypatch)
    registry, integration = _registry(tmp_path)
    registry.update(integration.integration_id, {"authentication": (AuthenticationConfig(),)})
    builder = ApiRequestBuilder(registry)
    request = builder.build(
        integration_id=integration.integration_id,
        operation_id="getContact",
        path_parameters={"contact_id": "123"},
    )
    with pytest.raises(ApiTimeoutError):
        ApiExecutor(transport=_TimeoutTransport()).execute(
            request,
            preview=builder.preview(request),
        )

    resolved_hosts: list[str] = []

    def dns(host: str, *args: Any, **kwargs: Any):
        resolved_hosts.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", dns)
    transport = _Transport(
        [
            _RawResponse(
                302,
                {"location": "https://cdn.acme.example/contact", "content-type": "text/plain"},
                b"",
            ),
            _RawResponse(200, {"content-type": "application/json"}, b'{"ok":true}'),
        ]
    )
    result = ApiExecutor(transport=transport).execute(
        request,
        preview=builder.preview(request),
    )
    assert resolved_hosts == ["api.acme.example", "cdn.acme.example"]
    assert result.redirects == ("https://cdn.acme.example/contact",)


def test_documentation_redirect_encodes_spaces_and_rejects_control_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _public_dns(monkeypatch)
    transport = _Transport(
        [
            _RawResponse(
                302,
                {"location": "/authorize?scope=openid profile email"},
                b"",
            ),
            _RawResponse(200, {"content-type": "text/plain"}, b"API documentation"),
        ]
    )
    executor = ApiExecutor(transport=transport)

    body, content_type = executor.fetch_documentation("https://docs.example.com/api")

    assert body == b"API documentation"
    assert content_type == "text/plain"
    assert transport.urls[1] == (
        "https://docs.example.com/authorize?scope=openid%20profile%20email"
    )

    rejected = ApiExecutor(
        transport=_Transport(
            [_RawResponse(302, {"location": "/authorize\r\nX-Injected: yes"}, b"")]
        )
    )
    with pytest.raises(UpstreamApiError, match="forbidden control characters"):
        rejected.fetch_documentation("https://docs.example.com/api")


def test_documentation_oauth_redirect_requests_rendered_browser_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _public_dns(monkeypatch)
    executor = ApiExecutor(
        transport=_Transport(
            [
                _RawResponse(
                    302,
                    {
                        "location": (
                            "https://portal.example.com/authorize?response_type=code"
                            "&scope=openid profile email&client_id=docs-client"
                            "&redirect_uri=https://portal.example.com/redirect"
                        )
                    },
                    b"",
                )
            ]
        )
    )

    with pytest.raises(DocumentationAuthorizationRequiredError) as raised:
        executor.fetch_documentation("https://docs.example.com/api")

    assert raised.value.code == "documentation_authorization_required"
    assert raised.value.details == {
        "authorization_origin": "https://portal.example.com",
        "rendered_browser_inspection_available": True,
    }


def test_rate_limit_response_is_structured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _public_dns(monkeypatch)
    registry, integration = _registry(tmp_path)
    registry.update(integration.integration_id, {"authentication": (AuthenticationConfig(),)})
    builder = ApiRequestBuilder(registry)
    request = builder.build(
        integration_id=integration.integration_id,
        operation_id="getContact",
        path_parameters={"contact_id": "123"},
    )
    transport = _Transport(
        [_RawResponse(429, {"retry-after": "30", "content-type": "application/json"}, b"{}")]
    )
    with pytest.raises(ApiRateLimitError) as raised:
        ApiExecutor(transport=transport).execute(
            request,
            preview=builder.preview(request),
        )
    assert raised.value.details["retry_after"] == "30"


def test_mutation_requires_bound_approval_and_then_executes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _public_dns(monkeypatch)
    registry, integration = _registry(tmp_path)
    monkeypatch.setenv("ACME_TOKEN", "token")
    registry.update(
        integration.integration_id,
        {
            "authentication": tuple(
                item.model_copy(update={"credential_reference": "env://ACME_TOKEN"})
                for item in integration.authentication
            )
        },
    )
    broker = PendingApiApprovalBroker()
    transport = _Transport(
        [_RawResponse(200, {"content-type": "application/json"}, b'{"updated":true}')]
    )
    executor = ApiExecutor(transport=transport, approval_broker=broker)
    builder = ApiRequestBuilder(registry)
    operation = next(item for item in integration.operations if item.method is HttpMethod.PATCH)
    request = builder.build(
        integration_id=integration.integration_id,
        operation_id=operation.operation_id,
        path_parameters={"contact_id": "123"},
        body={"email": "new@example.com"},
    ).model_copy(update={"session_id": "session-1"})
    preview = builder.preview(request)
    with pytest.raises(PermissionRequiredError) as raised:
        executor.execute(request, preview=preview)
    request_id = raised.value.details["permission_request_id"]
    approved_request, approved_preview = broker.approve(
        request_id,
        session_id="session-1",
        client_type="tui",
    )
    result = executor.execute(
        approved_request,
        preview=approved_preview,
        approval_reference=request_id,
    )
    assert result.status_code == 200
    assert transport.calls == 1


def test_ssrf_blocks_private_and_allows_explicit_trusted_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))
        ],
    )
    with pytest.raises(SsrfPolicyViolationError):
        validate_network_target("https://example.test", NetworkAccessPolicy())
    assert (
        validate_network_target(
            "https://internal.example.test",
            NetworkAccessPolicy(trusted_internal_networks=("127.0.0.0/8",)),
        )
        == "127.0.0.1"
    )


def test_cancellation_and_response_limit_are_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _public_dns(monkeypatch)
    registry, integration = _registry(tmp_path)
    registry.update(
        integration.integration_id,
        {"authentication": (AuthenticationConfig(),)},
    )
    request = ApiRequestBuilder(registry).build(
        integration_id=integration.integration_id,
        operation_id="getContact",
        path_parameters={"contact_id": "123"},
    )
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(Exception, match="cancelled"):
        ApiExecutor(transport=_Transport([])).execute(
            request,
            preview=ApiRequestBuilder(registry).preview(request),
            cancellation=cancelled,
        )
    with pytest.raises(ResponseTooLargeError):
        ApiExecutor(transport=_OversizeTransport()).execute(
            request,
            preview=ApiRequestBuilder(registry).preview(request),
        )


def test_dynamic_search_requires_model_decision_and_rejects_ambiguity(
    tmp_path: Path,
) -> None:
    registry, integration = _registry(tmp_path)
    discovery = ApiOperationDiscovery(registry)
    candidates = discovery.search("contact", limit=10)
    assert candidates
    with pytest.raises(AmbiguousOperationError):
        discovery.validate_decision(
            ApiRouteDecision(
                source_decision_id="route-1",
                task_intent="get a contact",
                workflow="request_execution",
                integration_id=integration.integration_id,
                operation_id="getContact",
                confidence=0.4,
                reason="Two candidates remain plausible.",
                safe_to_continue=True,
            ),
            candidates=candidates,
        )
    evidence = discovery.validate_decision(
        ApiRouteDecision(
            source_decision_id="route-2",
            task_intent="get a contact",
            workflow="request_execution",
            integration_id=integration.integration_id,
            operation_id="getContact",
            confidence=0.95,
            matched_terms=("contact",),
            reason="Exact operation name and read-only method match.",
            safe_to_continue=True,
        ),
        candidates=candidates,
    )
    assert evidence.operation_id == "getContact"


def test_gateway_tools_are_narrow_and_registered(tmp_path: Path) -> None:
    service = ApiManagerService(
        tmp_path,
        registry=ApiIntegrationRegistry(tmp_path / "integrations"),
    )
    tools = build_api_manager_langchain_tools(tmp_path, service=service)
    assert tuple(tool.name for tool in tools) == API_MANAGER_TOOL_NAMES
    semantic_import_schema = next(
        tool for tool in tools if tool.name == "api_docs_import_semantic"
    ).args_schema.model_json_schema()
    assert "text" in semantic_import_schema["required"]
    assert "semantic_definition" in semantic_import_schema["required"]
    execute_schema = next(
        tool for tool in tools if tool.name == "api_request_execute"
    ).args_schema.model_json_schema()
    properties = execute_schema["properties"]
    assert "integration_id" in properties
    assert "operation_id" in properties
    assert "url" not in properties
    assert "base_url" not in properties


def test_api_workflow_execution_requires_declared_search_and_preview() -> None:
    with pytest.raises(ValueError, match="request_execution requires declared actions"):
        _WorkflowDecision(
            source_decision_id="decision-api",
            session_id="session-api",
            task_intent="execute saved operation",
            required_actions=("request_execution",),
            reason="The user requested a live API call.",
            safe_to_continue=True,
        )

    decision = _WorkflowDecision(
        source_decision_id="decision-api",
        session_id="session-api",
        task_intent="execute saved operation",
        required_actions=("operation_search", "request_preview", "request_execution"),
        reason="Search, preview, and execute the selected operation.",
        safe_to_continue=True,
    )

    assert decision.required_actions == (
        "operation_search",
        "request_preview",
        "request_execution",
    )


def test_registry_and_executor_emit_redacted_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _public_dns(monkeypatch)
    events: list[tuple[str, dict[str, Any]]] = []
    registry = ApiIntegrationRegistry(
        tmp_path / "integrations",
        event_sink=lambda kind, payload: events.append((kind, payload)),
    )
    integration = _import()
    registry.save(integration)
    registry.update(integration.integration_id, {"authentication": (AuthenticationConfig(),)})
    builder = ApiRequestBuilder(registry)
    request = builder.build(
        integration_id=integration.integration_id,
        operation_id="getContact",
        path_parameters={"contact_id": "123"},
    )
    executor = ApiExecutor(
        transport=_Transport(
            [_RawResponse(200, {"content-type": "application/json"}, b'{"ok":true}')]
        ),
        event_sink=lambda kind, payload: events.append((kind, payload)),
    )
    executor.execute(request, preview=builder.preview(request))
    kinds = [kind for kind, _payload in events]
    assert "api.integration.saved" in kinds
    assert "api.call.started" in kinds
    assert "api.call.completed" in kinds
    assert "Authorization" not in json.dumps(events)


def test_integration_flow_import_search_preview_approval_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    _public_dns(monkeypatch)
    registry = ApiIntegrationRegistry(tmp_path / "integrations")
    broker = PendingApiApprovalBroker()
    transport = _Transport(
        [
            _RawResponse(200, {"content-type": "application/json"}, b'{"email":"old@example.com"}'),
            _RawResponse(200, {"content-type": "application/json"}, b'{"email":"new@example.com"}'),
        ]
    )
    executor = ApiExecutor(transport=transport, approval_broker=broker)
    service = ApiManagerService(tmp_path, registry=registry, executor=executor)
    imported = service.import_documentation(
        name="Acme CRM",
        text=json.dumps(OPENAPI),
        source_decision_id="import-decision",
    )
    integration_id = imported["integration"]["integration_id"]
    service.update_integration(
        integration_id,
        authentication=[AuthenticationConfig()],
    )
    get_route = ApiRouteDecision(
        source_decision_id="get-decision",
        task_intent="get contact",
        workflow="request_execution",
        integration_id=integration_id,
        operation_id="getContact",
        confidence=0.99,
        matched_terms=("get", "contact"),
        reason="Exact read-only contact operation.",
        safe_to_continue=True,
    )
    preview = service.preview_request(
        routing_decision=get_route,
        integration_id=integration_id,
        operation_id="getContact",
        path_parameters={"contact_id": "123"},
    )
    assert preview["approval_required"] is False
    read_result = service.execute_request(
        routing_decision=get_route,
        integration_id=integration_id,
        operation_id="getContact",
        path_parameters={"contact_id": "123"},
        session_id="session-flow",
    )
    assert read_result["executed"] is True

    patch_operation = next(
        item
        for item in registry.get(integration_id).operations
        if item.method is HttpMethod.PATCH
    )
    update_route = ApiRouteDecision(
        source_decision_id="update-decision",
        task_intent="update contact",
        workflow="request_execution",
        integration_id=integration_id,
        operation_id=patch_operation.operation_id,
        confidence=0.99,
        matched_terms=("update", "contact"),
        reason="Exact update operation.",
        safe_to_continue=True,
    )
    with pytest.raises(PermissionRequiredError) as raised:
        service.execute_request(
            routing_decision=update_route,
            integration_id=integration_id,
            operation_id=patch_operation.operation_id,
            path_parameters={"contact_id": "123"},
            body={"email": "new@example.com"},
            session_id="session-flow",
        )
    approved = service.decide_approval(
        raised.value.details["permission_request_id"],
        session_id="session-flow",
        approve=True,
        client_type="tui",
    )
    assert approved["executed"] is True
    assert approved["result"]["status_code"] == 200
