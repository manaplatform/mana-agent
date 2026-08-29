from typing import Any


class CodexError(RuntimeError):
    def __init__(
        self,
        message: str = "",
        *,
        provider: str = "",
        model: str = "",
        transport: str = "",
        http_status: int | None = None,
        original_error: str | None = None,
        error_code: str = "",
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.transport = transport
        self.http_status = http_status
        self.original_error = original_error
        self.error_code = error_code


class CodexUnavailableError(CodexError):
    pass


class CodexProtocolError(CodexError):
    def __init__(
        self,
        message: str = "Codex protocol error",
        *,
        provider: str = "",
        model: str = "",
        transport: str = "",
        http_status: int | None = None,
        original_error: str | None = None,
        error_code: str = "CODING_PROVIDER_PROTOCOL_ERROR",
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            transport=transport,
            http_status=http_status,
            original_error=original_error,
            error_code=error_code,
        )


class CodexToolProtocolError(CodexProtocolError):
    """Raised when server-tool calling or tool protocol fails (e.g. HTTP 400 server tools)."""

    def __init__(
        self,
        message: str = "Codex provider tool protocol error",
        *,
        provider: str = "",
        model: str = "",
        transport: str = "",
        http_status: int | None = 400,
        original_error: str | None = None,
        error_code: str = "CODING_PROVIDER_TOOL_PROTOCOL_ERROR",
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            transport=transport,
            http_status=http_status,
            original_error=original_error,
            error_code=error_code,
        )


class CodexBadRequestError(CodexProtocolError):
    """Raised when provider rejects request with HTTP 400."""

    def __init__(
        self,
        message: str = "Codex provider bad request (HTTP 400)",
        *,
        provider: str = "",
        model: str = "",
        transport: str = "",
        http_status: int | None = 400,
        original_error: str | None = None,
        error_code: str = "CODING_PROVIDER_BAD_REQUEST",
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            transport=transport,
            http_status=http_status,
            original_error=original_error,
            error_code=error_code,
        )


class CodexExecutionError(CodexError):
    pass


class CodexConfigurationError(CodexError):
    pass


class CodexCapabilityError(CodexConfigurationError):
    """Selected provider/model/transport cannot satisfy required coding capabilities."""

    def __init__(
        self,
        message: str = "Coding capabilities cannot be satisfied",
        *,
        provider: str = "",
        model: str = "",
        transport: str = "",
        http_status: int | None = None,
        original_error: str | None = None,
        error_code: str = "CODING_CAPABILITY_ERROR",
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            transport=transport,
            http_status=http_status,
            original_error=original_error,
            error_code=error_code,
        )


class CodexTimeoutError(CodexProtocolError):
    """Raised when a Codex app-server request or turn times out."""

    def __init__(
        self,
        message: str = "Codex request timed out",
        *,
        method: str = "",
        timeout_seconds: float | None = None,
        error_code: str = "CODING_PROVIDER_TIMEOUT",
        thread_id: str = "",
        turn_id: str = "",
        provider: str = "",
        model: str = "",
        transport: str = "",
        http_status: int | None = 408,
        original_error: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            transport=transport,
            http_status=http_status,
            original_error=original_error,
            error_code=error_code,
        )
        self.method = method
        self.timeout_seconds = timeout_seconds
        self.thread_id = thread_id
        self.turn_id = turn_id


class CodexInterruptionError(CodexError):
    """Raised when a Codex turn is interrupted (by model, user, provider timeout, or deadline)."""

    def __init__(
        self,
        message: str = "Codex turn interrupted",
        *,
        reason: str = "turn_interrupted",
        error_code: str = "MODEL_INTERRUPTED",
        thread_id: str = "",
        turn_id: str = "",
        changed_files: list[str] | None = None,
        checkpoint: dict[str, Any] | None = None,
        provider: str = "",
        model: str = "",
        transport: str = "",
        http_status: int | None = None,
        original_error: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            transport=transport,
            http_status=http_status,
            original_error=original_error,
            error_code=error_code,
        )
        self.reason = reason
        self.thread_id = thread_id
        self.turn_id = turn_id
        self.changed_files = list(changed_files or [])
        self.checkpoint = checkpoint


__all__ = [
    "CodexBadRequestError",
    "CodexCapabilityError",
    "CodexConfigurationError",
    "CodexError",
    "CodexExecutionError",
    "CodexInterruptionError",
    "CodexProtocolError",
    "CodexTimeoutError",
    "CodexToolProtocolError",
    "CodexUnavailableError",
]

