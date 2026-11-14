"""Typed inference failures; exceptions never select another semantic action."""
from __future__ import annotations

class InferenceError(RuntimeError):
    pass

class CapabilityError(InferenceError):
    pass

class InvalidResponseError(InferenceError):
    pass

class StructuredOutputError(InvalidResponseError, ValueError):
    """A schema-constrained response failed validation. The message contains no input values."""

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
