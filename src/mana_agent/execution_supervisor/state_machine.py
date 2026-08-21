"""Validated execution state transitions."""

from mana_agent.execution_supervisor.errors import InvalidTransitionError
from mana_agent.execution_supervisor.models import ExecutionState, TERMINAL_STATES


TRANSITIONS: dict[ExecutionState, frozenset[ExecutionState]] = {
    ExecutionState.CREATED: frozenset({ExecutionState.QUEUED, ExecutionState.CANCELLING, ExecutionState.FAILED, ExecutionState.BUDGET_EXHAUSTED, ExecutionState.RECOVERY_REVIEW_REQUIRED}),
    ExecutionState.QUEUED: frozenset({ExecutionState.LEASED, ExecutionState.WAITING, ExecutionState.CANCELLING, ExecutionState.FAILED, ExecutionState.BUDGET_EXHAUSTED, ExecutionState.RECOVERY_REVIEW_REQUIRED}),
    ExecutionState.LEASED: frozenset({ExecutionState.RUNNING, ExecutionState.QUEUED, ExecutionState.WAITING, ExecutionState.RETRY_SCHEDULED, ExecutionState.CANCELLING, ExecutionState.FAILED, ExecutionState.BUDGET_EXHAUSTED, ExecutionState.RECOVERY_REVIEW_REQUIRED}),
    ExecutionState.RUNNING: frozenset({ExecutionState.CHECKPOINTING, ExecutionState.WAITING, ExecutionState.RETRY_SCHEDULED, ExecutionState.REPLANNING, ExecutionState.CANCELLING, ExecutionState.FAILED, ExecutionState.BUDGET_EXHAUSTED, ExecutionState.RECOVERY_REVIEW_REQUIRED, ExecutionState.PENDING_BUDGET_DECISION, ExecutionState.COMPLETED_PENDING_VERIFICATION}),
    ExecutionState.CHECKPOINTING: frozenset({ExecutionState.RUNNING, ExecutionState.WAITING, ExecutionState.RETRY_SCHEDULED, ExecutionState.CANCELLING, ExecutionState.FAILED, ExecutionState.BUDGET_EXHAUSTED, ExecutionState.RECOVERY_REVIEW_REQUIRED}),
    ExecutionState.WAITING: frozenset({ExecutionState.QUEUED, ExecutionState.LEASED, ExecutionState.RUNNING, ExecutionState.CHECKPOINTING, ExecutionState.RETRY_SCHEDULED, ExecutionState.REPLANNING, ExecutionState.CANCELLING, ExecutionState.FAILED, ExecutionState.BUDGET_EXHAUSTED, ExecutionState.RECOVERY_REVIEW_REQUIRED, ExecutionState.PENDING_BUDGET_DECISION, ExecutionState.COMPLETED_PENDING_VERIFICATION}),
    ExecutionState.RETRY_SCHEDULED: frozenset({ExecutionState.QUEUED, ExecutionState.WAITING, ExecutionState.CANCELLING, ExecutionState.FAILED, ExecutionState.RECOVERY_REVIEW_REQUIRED}),
    ExecutionState.REPLANNING: frozenset({ExecutionState.QUEUED, ExecutionState.WAITING, ExecutionState.CANCELLING, ExecutionState.FAILED, ExecutionState.RECOVERY_REVIEW_REQUIRED}),
    ExecutionState.CANCELLING: frozenset({ExecutionState.CANCELLED, ExecutionState.FAILED}),
    ExecutionState.COMPLETED_PENDING_VERIFICATION: frozenset({ExecutionState.COMPLETED, ExecutionState.RETRY_SCHEDULED, ExecutionState.REPLANNING, ExecutionState.CANCELLING, ExecutionState.FAILED}),
    ExecutionState.PENDING_BUDGET_DECISION: frozenset({ExecutionState.COMPLETED, ExecutionState.WAITING, ExecutionState.RETRY_SCHEDULED, ExecutionState.REPLANNING, ExecutionState.CANCELLING, ExecutionState.FAILED}),
    ExecutionState.CANCELLED: frozenset(),
    ExecutionState.FAILED: frozenset({ExecutionState.RETRY_SCHEDULED, ExecutionState.REPLANNING}),
    ExecutionState.BUDGET_EXHAUSTED: frozenset(),
    ExecutionState.RECOVERY_REVIEW_REQUIRED: frozenset(),
    # A validated same-task decision may reopen only for re-verification. It
    # cannot resume implementation work from the completed state.
    ExecutionState.COMPLETED: frozenset({ExecutionState.COMPLETED_PENDING_VERIFICATION}),
}


def validate_transition(source: ExecutionState, target: ExecutionState) -> None:
    if source == target:
        return
    if source in TERMINAL_STATES and target not in TRANSITIONS[source]:
        raise InvalidTransitionError(f"terminal task cannot transition: {source.value} -> {target.value}")
    if target not in TRANSITIONS.get(source, frozenset()):
        raise InvalidTransitionError(f"invalid execution transition: {source.value} -> {target.value}")
