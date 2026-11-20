"""Typed inference failures; exceptions never select another semantic action."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

if TYPE_CHECKING:
    from mana_agent.integrations.provider_failure import ProviderFailure

class InferenceError(RuntimeError):
    pass

class CapabilityError(InferenceError):
    pass

class InvalidResponseError(InferenceError):
    def __init__(self, message: str, *, public_detail: str = "", reason_code: str = ""):
        super().__init__(message)
        # Only application-authored descriptions belong in public activity.
        self.public_detail = public_detail
        self.reason_code = reason_code

class StructuredOutputError(InvalidResponseError, ValueError):
    """A schema-constrained response failed validation. The message contains no input values."""


def describe_decision_validation(error: ValidationError, schema: dict[str, Any]) -> str:
    """Name schema fields and failed checks without printing rejected model data."""
    fields: set[str] = set()

    def collect(shape: Any) -> None:
        if isinstance(shape, dict):
            for name, spec in shape.get("properties", {}).items():
                fields.add(name)
                if name in {"action", "tool"} and isinstance(spec, dict):
                    values = spec.get("enum", [spec.get("const")])
                    fields.update(value for value in values if isinstance(value, str))
            for value in shape.values():
                collect(value)
        elif isinstance(shape, list):
            for value in shape:
                collect(value)

    collect(schema)
    errors = error.errors(include_input=False, include_context=False, include_url=False)
    inspection_requests = ("decision", "inspect", "requests")
    invalid_inspection_item = any(issue["loc"][:3] == inspection_requests and len(issue["loc"]) > 3
                                  for issue in errors)
    issues = []
    for issue in errors:
        if len(issues) == 4:
            break
        path = ".".join(str(part) if isinstance(part, int) or part in fields else "<key>"
                        for part in issue["loc"])
        code = issue["type"]
        if code == "too_short" and issue["loc"] == inspection_requests:
            # A failed item can leave the validated tuple empty even when the model supplied a request.
            if invalid_inspection_item:
                continue
            message = "Include at least one inspection request or select another allowed action."
        elif code.startswith(("mode_", "source_citation_", "coding_result_", "runtime_wiring_")) or code in {
            "entry_route_media_request", "media_operation_input",
        }:
            message = issue["msg"]
        elif code in {"union_tag_invalid", "union_tag_not_found"}:
            message = "Select an action or tool declared in the schema."
        else:
            message = code.replace("_", " ")
        issues.append(f"{path or 'decision'}: {message}")
    return "; ".join(issues)

class InferenceTimeoutError(InferenceError):
    pass

class ProviderRequestError(InferenceError):
    def __init__(self, message: str, *, provider: str, status_code: int | None = None,
                 retryable: bool = False, capability_rejection: bool = False,
                 failure: ProviderFailure | None = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.capability_rejection = capability_rejection
        self.failure = failure

    @property
    def public_detail(self) -> str:
        # Native inference classifies by status alone, with no upstream body.
        if self.failure is not None and not self.failure.upstream_body_snippet:
            return self.failure.safe_message
        status = f" (HTTP {self.status_code})" if self.status_code is not None else ""
        return f"The selected provider could not complete the request{status}."


def format_provider_error(exc: BaseException, *, provider: str | None = None, model: str | None = None) -> str:
    from mana_agent.utils.redaction import redact_secrets
    label = provider or "Inference provider"
    status = getattr(exc, "status_code", None)
    suffix = f" (HTTP {status})" if status else ""
    return str(redact_secrets(f"{label} model={model or 'unselected'} request failed{suffix}: {exc}"))
