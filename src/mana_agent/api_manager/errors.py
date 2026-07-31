"""Typed, user-actionable API Manager errors."""

from __future__ import annotations

from typing import Any


class ApiManagerError(RuntimeError):
    code = "api_manager_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"error_code": self.code, "message": str(self), "details": self.details}


class UnsupportedDocumentationError(ApiManagerError):
    code = "unsupported_documentation"


class MalformedSpecificationError(ApiManagerError):
    code = "malformed_specification"


class UnresolvedSchemaReferenceError(ApiManagerError):
    code = "unresolved_schema_reference"


class IntegrationNotFoundError(ApiManagerError):
    code = "integration_not_found"


class OperationNotFoundError(ApiManagerError):
    code = "operation_not_found"


class AmbiguousOperationError(ApiManagerError):
    code = "ambiguous_operation"


class MissingCredentialError(ApiManagerError):
    code = "missing_credential"


class RequestValidationError(ApiManagerError):
    code = "validation_failure"


class PermissionRequiredError(ApiManagerError):
    code = "permission_required"


class BlockedHostError(ApiManagerError):
    code = "blocked_host"


class SsrfPolicyViolationError(ApiManagerError):
    code = "ssrf_policy_violation"


class ApiTimeoutError(ApiManagerError):
    code = "timeout"


class ApiRateLimitError(ApiManagerError):
    code = "rate_limit"


class ResponseTooLargeError(ApiManagerError):
    code = "response_too_large"


class UnsupportedContentTypeError(ApiManagerError):
    code = "unsupported_content_type"


class UpstreamApiError(ApiManagerError):
    code = "upstream_api_failure"


class ApiCancelledError(ApiManagerError):
    code = "cancelled"

