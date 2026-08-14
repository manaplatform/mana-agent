"""Execution-supervisor domain errors."""


class ExecutionSupervisorError(RuntimeError):
    """Base error for durable execution supervision."""


class TaskNotFoundError(ExecutionSupervisorError):
    pass


class InvalidTransitionError(ExecutionSupervisorError):
    pass


class ConcurrentUpdateError(ExecutionSupervisorError):
    pass


class LeaseConflictError(ExecutionSupervisorError):
    pass


class StaleLeaseError(ExecutionSupervisorError):
    pass


class RetrySafetyError(ExecutionSupervisorError):
    pass


class CompletionVerificationError(ExecutionSupervisorError):
    pass


class BudgetExceededError(ExecutionSupervisorError):
    pass


class DecisionRequiredError(ExecutionSupervisorError):
    pass


class EscrowError(ExecutionSupervisorError):
    """Base error for execution result escrow operations."""

    def __init__(self, message: str, *, code: str = "RESULT_ESCROW_ERROR") -> None:
        super().__init__(message)
        self.code = code


class EscrowConflictError(EscrowError):
    """Raised when an escrow write conflicts with an existing immutable result."""

    def __init__(self, message: str, *, code: str = "RESULT_CONFLICT") -> None:
        super().__init__(message, code=code)


class EscrowCorruptError(EscrowError):
    """Raised when an escrow record is corrupted or cannot be deserialized."""

    def __init__(self, message: str, *, code: str = "RESULT_CORRUPT") -> None:
        super().__init__(message, code=code)


class EscrowIncompatibleVersionError(EscrowError):
    """Raised when an escrow record has an unsupported or future schema version."""

    def __init__(self, message: str, *, code: str = "RESULT_SCHEMA_INCOMPATIBLE") -> None:
        super().__init__(message, code=code)


class EscrowNotFoundError(EscrowError):
    """Raised when an expected escrow record does not exist."""

    def __init__(self, message: str, *, code: str = "RESULT_NOT_FOUND") -> None:
        super().__init__(message, code=code)

