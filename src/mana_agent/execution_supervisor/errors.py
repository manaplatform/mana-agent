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
