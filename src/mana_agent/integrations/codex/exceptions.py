from typing import Any


class CodexError(RuntimeError):
    pass


class CodexUnavailableError(CodexError):
    pass


class CodexProtocolError(CodexError):
    pass


class CodexExecutionError(CodexError):
    pass


class CodexConfigurationError(CodexError):
    pass


class CodexCapabilityError(CodexConfigurationError):
    """Selected provider/model/transport cannot satisfy required coding capabilities."""


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
    ) -> None:
        super().__init__(message)
        self.method = method
        self.timeout_seconds = timeout_seconds
        self.error_code = error_code
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
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.error_code = error_code
        self.thread_id = thread_id
        self.turn_id = turn_id
        self.changed_files = list(changed_files or [])
        self.checkpoint = checkpoint


__all__ = [
    "CodexCapabilityError",
    "CodexConfigurationError",
    "CodexError",
    "CodexExecutionError",
    "CodexInterruptionError",
    "CodexProtocolError",
    "CodexTimeoutError",
    "CodexUnavailableError",
]

