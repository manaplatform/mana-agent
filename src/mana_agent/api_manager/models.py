"""Normalized, provider-neutral API integration contracts."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stable_id(prefix: str, *parts: str) -> str:
    canonical = "\x1f".join(str(part or "").strip().lower() for part in parts)
    return f"{prefix}_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class HttpMethod(str, Enum):
    GET = "GET"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class ParameterLocation(str, Enum):
    PATH = "path"
    QUERY = "query"
    HEADER = "header"
    COOKIE = "cookie"


class AuthenticationType(str, Enum):
    NONE = "none"
    API_KEY_HEADER = "api_key_header"
    API_KEY_QUERY = "api_key_query"
    BEARER = "bearer"
    BASIC = "basic"
    CUSTOM_HEADERS = "custom_headers"
    OAUTH2 = "oauth2"


class OperationRiskLevel(str, Enum):
    READ_ONLY = "read_only"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    HIGH = "unknown_high_risk"

    @property
    def mutating(self) -> bool:
        return self is not self.READ_ONLY


class DocumentationSourceType(str, Enum):
    OPENAPI_JSON = "openapi_json"
    OPENAPI_YAML = "openapi_yaml"
    SWAGGER_JSON = "swagger_json"
    SWAGGER_YAML = "swagger_yaml"
    WEBPAGE = "webpage"
    MARKDOWN = "markdown"
    TEXT = "text"
    LOCAL_FILE = "local_file"
    PASTED = "pasted"


class ApiServer(StrictModel):
    url: str = Field(min_length=1)
    description: str = ""
    variables: dict[str, str] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def absolute_http_url(cls, value: str) -> str:
        from urllib.parse import urlsplit

        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("server URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("server URL must not contain credentials")
        return value.rstrip("/")


class ApiParameter(StrictModel):
    name: str = Field(min_length=1)
    location: ParameterLocation
    required: bool = False
    description: str = ""
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")
    style: str = ""
    explode: bool | None = None
    allow_unknown: bool = False
    inferred: bool = False
    unresolved: bool = False

    @model_validator(mode="after")
    def path_is_required(self) -> "ApiParameter":
        if self.location is ParameterLocation.PATH and not self.required:
            self.required = True
        return self


class RequestBody(StrictModel):
    required: bool = False
    content: dict[str, dict[str, Any]] = Field(default_factory=dict)
    description: str = ""
    inferred: bool = False
    unresolved: bool = False


class ResponseDefinition(StrictModel):
    status_code: str
    description: str = ""
    content: dict[str, dict[str, Any]] = Field(default_factory=dict)
    headers: dict[str, dict[str, Any]] = Field(default_factory=dict)


class OAuthFlow(StrictModel):
    authorization_url: str = ""
    token_url: str = ""
    refresh_url: str = ""
    scopes: dict[str, str] = Field(default_factory=dict)


class AuthenticationConfig(StrictModel):
    type: AuthenticationType = AuthenticationType.NONE
    scheme_name: str = ""
    parameter_name: str = ""
    credential_reference: str = ""
    custom_header_names: tuple[str, ...] = ()
    oauth_flows: dict[str, OAuthFlow] = Field(default_factory=dict)
    required: bool = False
    inferred: bool = False
    unresolved: bool = False

    @model_validator(mode="after")
    def validate_reference_and_shape(self) -> "AuthenticationConfig":
        if self.credential_reference and not self.credential_reference.startswith(
            ("env://", "mana-secret://")
        ):
            raise ValueError(
                "credential_reference must use env://<name> or mana-secret://<id>"
            )
        if self.credential_reference:
            identifier = self.credential_reference.split("://", 1)[1]
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", identifier):
                raise ValueError("credential_reference contains an invalid credential ID")
        if self.type in {
            AuthenticationType.API_KEY_HEADER,
            AuthenticationType.API_KEY_QUERY,
        } and not self.parameter_name:
            raise ValueError("API-key authentication requires a parameter_name")
        if self.type is AuthenticationType.CUSTOM_HEADERS and not self.custom_header_names:
            raise ValueError("custom-header authentication requires header names")
        return self


class PaginationPolicy(StrictModel):
    kind: Literal["none", "page", "offset", "cursor", "link_header"] = "none"
    request_parameter: str = ""
    page_size_parameter: str = ""
    response_cursor_path: str = ""
    response_items_path: str = ""
    maximum_pages: int = Field(default=1, ge=1, le=1000)


class RetryPolicy(StrictModel):
    maximum_attempts: int = Field(default=1, ge=1, le=10)
    backoff_seconds: float = Field(default=0.25, ge=0, le=60)
    retry_statuses: tuple[int, ...] = (429, 502, 503, 504)
    retry_non_idempotent: bool = False


class TimeoutPolicy(StrictModel):
    connect_seconds: float = Field(default=10.0, gt=0, le=300)
    read_seconds: float = Field(default=30.0, gt=0, le=900)


class RateLimitMetadata(StrictModel):
    requests: int | None = Field(default=None, gt=0)
    window_seconds: int | None = Field(default=None, gt=0)
    limit_header: str = ""
    remaining_header: str = ""
    reset_header: str = ""


class DocumentationSource(StrictModel):
    type: DocumentationSourceType
    reference: str = Field(min_length=1)
    content_sha256: str = ""
    imported_at: datetime = Field(default_factory=utc_now)
    source_decision_id: str = Field(min_length=1)


class IntegrationVersion(StrictModel):
    number: int = Field(default=1, ge=1)
    source_sha256: str
    created_at: datetime = Field(default_factory=utc_now)
    notes: str = ""


class ApiOperation(StrictModel):
    operation_id: str = Field(min_length=1, max_length=512)
    name: str = Field(min_length=1)
    description: str = ""
    method: HttpMethod
    path: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    tags: tuple[str, ...] = ()
    parameters: tuple[ApiParameter, ...] = ()
    request_body: RequestBody | None = None
    responses: tuple[ResponseDefinition, ...] = ()
    authentication: tuple[AuthenticationConfig, ...] = ()
    required_scopes: tuple[str, ...] = ()
    timeout: TimeoutPolicy = Field(default_factory=TimeoutPolicy)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    pagination: PaginationPolicy = Field(default_factory=PaginationPolicy)
    rate_limit: RateLimitMetadata = Field(default_factory=RateLimitMetadata)
    risk_level: OperationRiskLevel
    source_reference: str = Field(min_length=1)
    inferred_fields: tuple[str, ...] = ()
    unresolved_fields: tuple[str, ...] = ()
    allow_unknown_parameters: bool = False

    @field_validator("path")
    @classmethod
    def path_shape(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("operation path must start with '/'")
        return value

    @model_validator(mode="after")
    def mutation_methods_require_mutation_risk(self) -> "ApiOperation":
        if self.method in {
            HttpMethod.POST,
            HttpMethod.PUT,
            HttpMethod.PATCH,
            HttpMethod.DELETE,
        } and self.risk_level is OperationRiskLevel.READ_ONLY:
            raise ValueError(
                f"{self.method.value} operations cannot be classified as read-only"
            )
        return self

    @property
    def request_body_schema(self) -> dict[str, Any]:
        if not self.request_body:
            return {}
        for media_type in ("application/json", "application/*+json"):
            if media_type in self.request_body.content:
                return self.request_body.content[media_type]
        return next(iter(self.request_body.content.values()), {})

    @property
    def response_schema(self) -> dict[str, Any]:
        for response in self.responses:
            if response.status_code.startswith("2"):
                return next(iter(response.content.values()), {})
        return {}


class ApiIntegration(StrictModel):
    integration_id: str = Field(min_length=1, pattern=r"^api_[a-f0-9]{24}$")
    name: str = Field(min_length=1, max_length=160)
    description: str = ""
    enabled: bool = True
    servers: tuple[ApiServer, ...]
    operations: tuple[ApiOperation, ...]
    authentication: tuple[AuthenticationConfig, ...] = ()
    documentation_sources: tuple[DocumentationSource, ...]
    versions: tuple[IntegrationVersion, ...]
    active_version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    ephemeral: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_members(self) -> "ApiIntegration":
        if not self.servers:
            raise ValueError("integration requires at least one server")
        if not self.operations:
            raise ValueError("integration requires at least one operation")
        ids = [operation.operation_id for operation in self.operations]
        if len(ids) != len(set(ids)):
            raise ValueError("operation IDs must be unique within an integration")
        version_numbers = {version.number for version in self.versions}
        if self.active_version not in version_numbers:
            raise ValueError("active_version must reference a stored version")
        return self

    @classmethod
    def create(
        cls,
        *,
        name: str,
        servers: list[ApiServer],
        operations: list[ApiOperation],
        documentation_source: DocumentationSource,
        description: str = "",
        authentication: list[AuthenticationConfig] | None = None,
        ephemeral: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> "ApiIntegration":
        digest = documentation_source.content_sha256 or hashlib.sha256(
            json.dumps([item.model_dump(mode="json", by_alias=True) for item in operations], sort_keys=True).encode()
        ).hexdigest()
        return cls(
            integration_id=stable_id("api", name, servers[0].url),
            name=name,
            description=description,
            servers=tuple(servers),
            operations=tuple(operations),
            authentication=tuple(authentication or ()),
            documentation_sources=(documentation_source,),
            versions=(IntegrationVersion(number=1, source_sha256=digest),),
            ephemeral=ephemeral,
            metadata=metadata or {},
        )


class RoutingEvidence(StrictModel):
    integration_id: str
    operation_id: str
    confidence: float = Field(ge=0, le=1)
    matched_terms: tuple[str, ...] = ()
    required_missing_inputs: tuple[str, ...] = ()
    risk_classification: OperationRiskLevel
    reason: str
    source_decision_id: str = Field(min_length=1)
