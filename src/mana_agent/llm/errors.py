"""Typed inference failures; exceptions never select another semantic action."""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

class InferenceError(RuntimeError):
    pass

class CapabilityError(InferenceError):
    pass

class InvalidResponseError(InferenceError):
    def __init__(self, message: str, *, public_detail: str = ""):
        super().__init__(message)
        # Only application-authored descriptions belong in public activity.
        self.public_detail = public_detail

class StructuredOutputError(InvalidResponseError, ValueError):
    """A schema-constrained response failed validation. The message contains no input values."""


def describe_decision_validation(error: ValidationError, schema: dict[str, Any]) -> str:
    """Name schema fields and failed checks without printing rejected model data."""
    fields: set[str] = set()

    def collect(shape: Any) -> None:
        if isinstance(shape, dict):
            fields.update(shape.get("properties", {}))
            for value in shape.values():
                collect(value)
        elif isinstance(shape, list):
            for value in shape:
                collect(value)

    collect(schema)
    issues = []
    for issue in error.errors(include_input=False, include_context=False, include_url=False)[:4]:
        path = ".".join(str(part) if isinstance(part, int) or part in fields else "<key>"
                        for part in issue["loc"])
        code = issue["type"]
        message = issue["msg"] if code.startswith("mode_") else code.replace("_", " ")
        issues.append(f"{path or 'decision'}: {message}")
    return "; ".join(issues)

class InferenceTimeoutError(InferenceError):
    pass

class ProviderRequestError(InferenceError):
    def __init__(self, message: str, *, provider: str, status_code: int | None = None,
                 retryable: bool = False, capability_rejection: bool = False):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.capability_rejection = capability_rejection


def format_provider_error(exc: BaseException, *, provider: str | None = None, model: str | None = None) -> str:
    from mana_agent.utils.redaction import redact_secrets
    label = provider or "Inference provider"
    status = getattr(exc, "status_code", None)
    suffix = f" (HTTP {status})" if status else ""
    return str(redact_secrets(f"{label} model={model or 'unselected'} request failed{suffix}: {exc}"))
