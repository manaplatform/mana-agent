"""Codex integration errors."""


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


__all__ = ["CodexConfigurationError", "CodexError", "CodexExecutionError", "CodexProtocolError", "CodexUnavailableError"]
