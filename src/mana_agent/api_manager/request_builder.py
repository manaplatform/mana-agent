"""Validated request construction for saved API operations."""

from __future__ import annotations

import json
import re
import secrets
from http.cookies import SimpleCookie
from typing import Any
from urllib.parse import quote, urlencode, urljoin

from pydantic import Field

from mana_agent.api_manager.authentication import (
    CredentialResolver,
    EnvironmentCredentialResolver,
    apply_authentication,
)
from mana_agent.api_manager.errors import RequestValidationError
from mana_agent.api_manager.models import (
    ApiIntegration,
    ApiOperation,
    AuthenticationConfig,
    OperationRiskLevel,
    ParameterLocation,
    StrictModel,
)
from mana_agent.api_manager.redaction import redact_mapping, redact_url
from mana_agent.api_manager.registry import ApiIntegrationRegistry
from mana_agent.api_manager.schemas import validate_json


class BuiltApiRequest(StrictModel):
    integration_id: str
    operation_id: str
    method: str
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    body: bytes | None = None
    content_type: str = ""
    timeout_seconds: float
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 30.0
    risk_level: OperationRiskLevel
    retry_maximum_attempts: int = 1
    retry_statuses: tuple[int, ...] = ()
    retry_non_idempotent: bool = False
    retry_backoff_seconds: float = 0.25
    secret_values: tuple[str, ...] = ()
    session_id: str = ""
    routing_task_intent: str = ""
    approved_network_host: str = ""
    allow_insecure_http_once: bool = False


class RequestPreview(StrictModel):
    integration_id: str
    integration_name: str
    operation_id: str
    operation_name: str
    method: str
    redacted_url: str
    redacted_headers: dict[str, str]
    query_parameters: dict[str, Any]
    body_summary: dict[str, Any]
    expected_side_effects: str
    risk_level: OperationRiskLevel
    approval_required: bool


class ApiRequestBuilder:
    def __init__(
        self,
        registry: ApiIntegrationRegistry,
        *,
        credential_resolver: CredentialResolver | None = None,
    ) -> None:
        self.registry = registry
        self.credential_resolver = credential_resolver or EnvironmentCredentialResolver()

    def build(
        self,
        *,
        integration_id: str,
        operation_id: str,
        path_parameters: dict[str, Any] | None = None,
        query_parameters: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
        body: Any = None,
        credential_reference: str = "",
        content_type: str = "",
    ) -> BuiltApiRequest:
        integration, operation = self.registry.operation(integration_id, operation_id)
        if not integration.enabled:
            raise RequestValidationError("The selected API integration is disabled.")
        unresolved_inputs = [
            f"{parameter.location.value}:{parameter.name}"
            for parameter in operation.parameters
            if parameter.unresolved
        ]
        if operation.request_body and operation.request_body.unresolved:
            unresolved_inputs.append("body")
        if unresolved_inputs:
            raise RequestValidationError(
                "The selected operation has unresolved request requirements.",
                details={"unresolved": unresolved_inputs},
            )
        path_values = dict(path_parameters or {})
        query_values = dict(query_parameters or {})
        header_values = {str(key): str(value) for key, value in (headers or {}).items()}
        cookie_values = {str(key): str(value) for key, value in (cookies or {}).items()}
        defined = {
            location: {
                item.name: item
                for item in operation.parameters
                if item.location is location
            }
            for location in ParameterLocation
        }
        header_names = {name.lower(): name for name in defined[ParameterLocation.HEADER]}
        header_values = {
            header_names.get(name.lower(), name): value
            for name, value in header_values.items()
        }
        self._validate_parameters(path_values, defined[ParameterLocation.PATH], "path", operation)
        self._validate_parameters(query_values, defined[ParameterLocation.QUERY], "query", operation)
        self._validate_parameters(header_values, defined[ParameterLocation.HEADER], "header", operation)
        self._validate_parameters(cookie_values, defined[ParameterLocation.COOKIE], "cookie", operation)

        path = operation.path
        for name, parameter in defined[ParameterLocation.PATH].items():
            if name in path_values:
                serialized = _serialize_path(path_values[name], parameter, name=name)
                path = path.replace("{" + name + "}", quote(serialized, safe=""))
        unresolved_tokens = re.findall(r"\{([^{}]+)\}", path)
        if unresolved_tokens:
            raise RequestValidationError(
                f"Missing required path parameters: {', '.join(sorted(unresolved_tokens))}.",
                details={"missing": sorted(unresolved_tokens), "location": "path"},
            )

        query_items: list[tuple[str, str]] = []
        for name, value in query_values.items():
            parameter = defined[ParameterLocation.QUERY].get(name)
            query_items.extend(_serialize_query(name, value, parameter))

        _validate_headers(header_values)
        if cookie_values:
            _validate_headers(cookie_values)
            cookie = SimpleCookie()
            for name, value in cookie_values.items():
                cookie[name] = value
            header_values["Cookie"] = "; ".join(
                morsel.OutputString() for morsel in cookie.values()
            )

        request_content_type, encoded_body = self._encode_body(
            operation,
            body=body,
            requested_content_type=content_type,
        )
        if request_content_type:
            header_values.setdefault("Content-Type", request_content_type)
        header_values.setdefault("Accept", "application/json, text/plain;q=0.9, */*;q=0.5")

        authentication = self._select_authentication(integration, operation)
        secret_values: tuple[str, ...] = ()
        if authentication is not None:
            header_values, query_items, secret_values = apply_authentication(
                authentication,
                credential_reference=credential_reference,
                headers=header_values,
                query=query_items,
                resolver=self.credential_resolver,
            )
            _validate_headers(header_values)
        base = operation.base_url.rstrip("/") + "/"
        url = urljoin(base, path.lstrip("/"))
        if query_items:
            url = f"{url}?{urlencode(query_items, doseq=True)}"
        return BuiltApiRequest(
            integration_id=integration_id,
            operation_id=operation_id,
            method=operation.method.value,
            url=url,
            headers=header_values,
            body=encoded_body,
            content_type=request_content_type,
            timeout_seconds=operation.timeout.connect_seconds + operation.timeout.read_seconds,
            connect_timeout_seconds=operation.timeout.connect_seconds,
            read_timeout_seconds=operation.timeout.read_seconds,
            risk_level=operation.risk_level,
            retry_maximum_attempts=operation.retry_policy.maximum_attempts,
            retry_statuses=operation.retry_policy.retry_statuses,
            retry_non_idempotent=operation.retry_policy.retry_non_idempotent,
            retry_backoff_seconds=operation.retry_policy.backoff_seconds,
            secret_values=secret_values,
        )

    def preview(self, request: BuiltApiRequest) -> RequestPreview:
        integration, operation = self.registry.operation(
            request.integration_id, request.operation_id
        )
        from urllib.parse import parse_qs, urlsplit

        parsed = urlsplit(request.url)
        auth_query_names = tuple(
            auth.parameter_name
            for auth in operation.authentication
            if auth.parameter_name
        )
        body_summary: dict[str, Any] = {"present": request.body is not None, "bytes": len(request.body or b"")}
        if request.body and request.content_type == "application/json":
            try:
                body_summary["json_shape"] = _body_shape(
                    redact_mapping(json.loads(request.body))
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                body_summary["format"] = "invalid-json"
        elif request.body:
            body_summary["content_type"] = request.content_type
        return RequestPreview(
            integration_id=integration.integration_id,
            integration_name=integration.name,
            operation_id=operation.operation_id,
            operation_name=operation.name,
            method=request.method,
            redacted_url=redact_url(request.url, sensitive_query_names=auth_query_names),
            redacted_headers=redact_mapping(request.headers, secret_values=request.secret_values),
            query_parameters=redact_mapping(parse_qs(parsed.query), secret_values=request.secret_values),
            body_summary=body_summary,
            expected_side_effects=(
                "No side effects are expected."
                if operation.risk_level is OperationRiskLevel.READ_ONLY
                else f"This operation is classified as {operation.risk_level.value} and may change external state."
            ),
            risk_level=operation.risk_level,
            approval_required=operation.risk_level.mutating,
        )

    @staticmethod
    def _validate_parameters(
        values: dict[str, Any],
        definitions: dict[str, Any],
        location: str,
        operation: ApiOperation,
    ) -> None:
        missing = sorted(
            name for name, definition in definitions.items() if definition.required and name not in values
        )
        if missing:
            raise RequestValidationError(
                f"Missing required {location} parameters: {', '.join(missing)}.",
                details={"missing": missing, "location": location},
            )
        unknown = sorted(set(values).difference(definitions))
        if unknown and not operation.allow_unknown_parameters:
            raise RequestValidationError(
                f"Unknown {location} parameters: {', '.join(unknown)}.",
                details={"unknown": unknown, "location": location},
            )
        for name, value in values.items():
            definition = definitions.get(name)
            if definition:
                validate_json(value, definition.schema_, path=f"$.{location}.{name}")

    @staticmethod
    def _select_authentication(
        integration: ApiIntegration,
        operation: ApiOperation,
    ) -> AuthenticationConfig | None:
        if (
            len(integration.authentication) == 1
            and integration.authentication[0].type.value == "none"
        ):
            return integration.authentication[0]
        operation_candidates = list(operation.authentication)
        if not operation_candidates:
            return None
        configured_by_name = {
            item.scheme_name: item
            for item in integration.authentication
            if item.scheme_name
        }
        candidates = [
            configured_by_name.get(item.scheme_name, item)
            for item in operation_candidates
        ]
        explicitly_configured = [
            item
            for item in candidates
            if item.type.value == "none" or bool(item.credential_reference)
        ]
        if len(candidates) > 1 and len(explicitly_configured) == 1:
            candidates = explicitly_configured
        if not candidates:
            return None
        if len(candidates) > 1:
            names = [item.scheme_name or item.type.value for item in candidates]
            raise RequestValidationError(
                "The operation allows multiple authentication alternatives; a configured operation-specific "
                f"choice is required ({', '.join(names)}). No authentication fallback was selected."
            )
        authentication = candidates[0]
        if authentication.unresolved:
            raise RequestValidationError("Authentication requirements remain unresolved.")
        return authentication

    @staticmethod
    def _encode_body(
        operation: ApiOperation,
        *,
        body: Any,
        requested_content_type: str,
    ) -> tuple[str, bytes | None]:
        definition = operation.request_body
        if body is None:
            if definition and definition.required:
                raise RequestValidationError("A request body is required.")
            return "", None
        if definition is None:
            raise RequestValidationError("The selected operation does not document a request body.")
        available = list(definition.content)
        content_type = requested_content_type or (
            "application/json" if "application/json" in available else available[0] if available else ""
        )
        if content_type not in definition.content:
            raise RequestValidationError(
                f"Unsupported request content type {content_type!r}; expected one of {available!r}."
            )
        schema = definition.content.get(content_type) or {}
        validation_body = body
        if content_type.startswith("multipart/form-data") and isinstance(body, dict):
            validation_body = {
                key: (
                    value.get("content", "")
                    if isinstance(value, dict) and "filename" in value
                    else value
                )
                for key, value in body.items()
            }
        validate_json(validation_body, schema, path="$.body")
        if content_type == "application/json" or content_type.endswith("+json"):
            return content_type, json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if content_type == "application/x-www-form-urlencoded":
            if not isinstance(body, dict):
                raise RequestValidationError("Form request body must be an object.")
            return content_type, urlencode(body, doseq=True).encode("utf-8")
        if content_type.startswith("multipart/form-data"):
            if not isinstance(body, dict):
                raise RequestValidationError("Multipart request body must be an object.")
            boundary = f"mana-{secrets.token_hex(12)}"
            chunks: list[bytes] = []
            for name, value in body.items():
                chunks.append(f"--{boundary}\r\n".encode())
                if isinstance(value, dict) and "filename" in value:
                    filename = _safe_disposition(value.get("filename"))
                    media_type = str(value.get("content_type") or "application/octet-stream")
                    if "\r" in media_type or "\n" in media_type:
                        raise RequestValidationError("Multipart content type contains a newline.")
                    chunks.append(
                        (
                            f'Content-Disposition: form-data; name="{_safe_disposition(name)}"; '
                            f'filename="{filename}"\r\nContent-Type: {media_type}\r\n\r\n'
                        ).encode()
                    )
                    content = value.get("content", b"")
                    chunks.append(content if isinstance(content, bytes) else str(content).encode("utf-8"))
                else:
                    chunks.append(
                        f'Content-Disposition: form-data; name="{_safe_disposition(name)}"\r\n\r\n'.encode()
                    )
                    chunks.append(str(value).encode("utf-8"))
                chunks.append(b"\r\n")
            chunks.append(f"--{boundary}--\r\n".encode())
            return f"multipart/form-data; boundary={boundary}", b"".join(chunks)
        if content_type.startswith("text/"):
            if not isinstance(body, str):
                raise RequestValidationError("Text request body must be a string.")
            return content_type, body.encode("utf-8")
        raise RequestValidationError(f"Request content type {content_type!r} is not supported.")


def _serialize_scalar(value: Any, schema: dict[str, Any], *, name: str) -> str:
    if isinstance(value, (dict, list)):
        raise RequestValidationError(f"Path parameter {name!r} must be a scalar value.")
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _serialize_path(value: Any, parameter: Any, *, name: str) -> str:
    if isinstance(value, list):
        return ",".join(
            _serialize_scalar(item, parameter.schema_.get("items") or {}, name=name)
            for item in value
        )
    if isinstance(value, dict):
        flattened = [str(item) for pair in value.items() for item in pair]
        return ",".join(flattened)
    return _serialize_scalar(value, parameter.schema_, name=name)


def _serialize_query(
    name: str,
    value: Any,
    parameter: Any | None,
) -> list[tuple[str, str]]:
    schema = parameter.schema_ if parameter else {}
    style = str(parameter.style or "form") if parameter else "form"
    explode = parameter.explode if parameter and parameter.explode is not None else style == "form"
    if isinstance(value, list):
        delimiter = " " if style == "spaceDelimited" else "|" if style == "pipeDelimited" else ","
        if not explode or style in {"spaceDelimited", "pipeDelimited"}:
            return [
                (
                    name,
                    delimiter.join(
                        _serialize_scalar(item, schema.get("items") or {}, name=name)
                        for item in value
                    ),
                )
            ]
        return [(name, _serialize_scalar(item, schema.get("items") or {}, name=name)) for item in value]
    if isinstance(value, dict):
        if style == "deepObject":
            return [
                (f"{name}[{key}]", _serialize_scalar(item, {}, name=name))
                for key, item in value.items()
            ]
        if explode:
            return [
                (str(key), _serialize_scalar(item, {}, name=name))
                for key, item in value.items()
            ]
        flattened = [
            _serialize_scalar(item, {}, name=name)
            for pair in value.items()
            for item in pair
        ]
        return [(name, ",".join(flattened))]
    return [(name, _serialize_scalar(value, schema, name=name))]


def _validate_headers(headers: dict[str, str]) -> None:
    for name, value in headers.items():
        if not re.fullmatch(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+", name):
            raise RequestValidationError(f"Invalid HTTP header name: {name!r}.")
        if "\r" in value or "\n" in value:
            raise RequestValidationError(f"HTTP header {name!r} contains a newline.")
        if name.lower() in {"host", "content-length", "transfer-encoding", "connection"}:
            raise RequestValidationError(f"HTTP header {name!r} is controlled by the API runtime.")


def _safe_disposition(value: Any) -> str:
    return str(value).replace('"', "").replace("\r", "").replace("\n", "")[:200]


def _body_shape(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return type(value).__name__
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if value[key] == "[REDACTED]"
                else _body_shape(value[key], depth=depth + 1)
            )
            for key in sorted(value)
        }
    if isinstance(value, list):
        return {
            "type": "array",
            "count": len(value),
            "items": _body_shape(value[0], depth=depth + 1) if value else "unknown",
        }
    return type(value).__name__
