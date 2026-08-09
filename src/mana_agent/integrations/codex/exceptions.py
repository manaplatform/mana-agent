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


class CodexCapabilityError(CodexConfigurationError):
    """Selected provider/model/transport cannot satisfy required coding capabilities."""


__all__ = [
    "CodexCapabilityError",
    "CodexConfigurationError",
    "CodexError",
    "CodexExecutionError",
    "CodexProtocolError",
    "CodexUnavailableError",
]
