"""Typed, sanitized errors exposed by provider-neutral email connectors."""
from __future__ import annotations

from typing import Any


class EmailConnectorError(RuntimeError):
    code = "email_connector_error"
    retryable = False
    reconnect_required = False

    def __init__(
        self,
        message: str = "The email connector could not complete the request.",
        *,
        provider: str | None = None,
        provider_status: int | None = None,
        diagnostic_context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.provider_status = provider_status
        self.diagnostic_context = diagnostic_context or {}

    def to_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "provider": self.provider,
            "provider_status": self.provider_status,
            "retryable": self.retryable,
            "reconnect_required": self.reconnect_required,
            "diagnostic_context": self.diagnostic_context,
        }


class EmailAuthenticationError(EmailConnectorError):
    code = "email_authentication_failed"
    reconnect_required = True


class EmailAuthorizationError(EmailConnectorError):
    code = "email_authorization_failed"
    reconnect_required = True


class EmailMessageNotFoundError(EmailConnectorError):
    code = "email_message_not_found"


class EmailInvalidMessageReferenceError(EmailConnectorError):
    code = "email_invalid_message_reference"


class EmailProviderError(EmailConnectorError):
    code = "email_provider_error"


class EmailConnectorConfigurationError(EmailConnectorError):
    code = "email_connector_configuration_error"


class EmailTemporaryError(EmailConnectorError):
    code = "email_temporary_error"
    retryable = True


class AuthenticationRequired(EmailAuthenticationError): pass
class CredentialsRevoked(EmailAuthenticationError): pass
class EmailPermissionDenied(EmailAuthorizationError): pass
class ApprovalRequired(EmailConnectorError): pass
class CapabilityUnsupported(EmailConnectorError): pass
class ProviderRateLimited(EmailTemporaryError): pass
class TemporaryProviderFailure(EmailTemporaryError): pass
class PermanentProviderFailure(EmailProviderError): pass
class InvalidMessageIdentifier(EmailInvalidMessageReferenceError): pass
class AttachmentTooLarge(EmailConnectorError): pass
class UnsafeAttachment(EmailConnectorError): pass
class AmbiguousSendResult(EmailConnectorError): pass
class SynchronizationCursorExpired(EmailConnectorError): pass
