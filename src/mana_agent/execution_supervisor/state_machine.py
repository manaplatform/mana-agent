"""Validated execution state transitions."""

from mana_agent.execution_supervisor.errors import InvalidTransitionError
from mana_agent.execution_supervisor.models import ExecutionState, TERMINAL_STATES


TRANSITIONS: dict[ExecutionState, frozenset[ExecutionState]] = {
    ExecutionState.CREATED: frozenset({ExecutionState.QUEUED, ExecutionState.CANCELLING, ExecutionState.FAILED}),
    ExecutionState.QUEUED: frozenset({ExecutionState.LEASED, ExecutionState.WAITING, ExecutionState.CANCELLING, ExecutionState.FAILED}),
    ExecutionState.LEASED: frozenset({ExecutionState.RUNNING, ExecutionState.QUEUED, ExecutionState.RETRY_SCHEDULED, ExecutionState.CANCELLING, ExecutionState.FAILED}),
    ExecutionState.RUNNING: frozenset({ExecutionState.CHECKPOINTING, ExecutionState.WAITING, ExecutionState.RETRY_SCHEDULED, ExecutionState.REPLANNING, ExecutionState.CANCELLING, ExecutionState.FAILED, ExecutionState.COMPLETED_PENDING_VERIFICATION}),
    ExecutionState.CHECKPOINTING: frozenset({ExecutionState.RUNNING, ExecutionState.WAITING, ExecutionState.RETRY_SCHEDULED, ExecutionState.CANCELLING, ExecutionState.FAILED}),
    ExecutionState.WAITING: frozenset({ExecutionState.QUEUED, ExecutionState.LEASED, ExecutionState.RUNNING, ExecutionState.CHECKPOINTING, ExecutionState.RETRY_SCHEDULED, ExecutionState.REPLANNING, ExecutionState.CANCELLING, ExecutionState.FAILED, ExecutionState.COMPLETED_PENDING_VERIFICATION}),
    ExecutionState.RETRY_SCHEDULED: frozenset({ExecutionState.QUEUED, ExecutionState.CANCELLING, ExecutionState.FAILED}),
    ExecutionState.REPLANNING: frozenset({ExecutionState.QUEUED, ExecutionState.WAITING, ExecutionState.CANCELLING, ExecutionState.FAILED}),
    ExecutionState.CANCELLING: frozenset({ExecutionState.CANCELLED, ExecutionState.FAILED}),
    ExecutionState.COMPLETED_PENDING_VERIFICATION: frozenset({ExecutionState.COMPLETED, ExecutionState.RETRY_SCHEDULED, ExecutionState.REPLANNING, ExecutionState.CANCELLING, ExecutionState.FAILED}),
    ExecutionState.CANCELLED: frozenset(),
    ExecutionState.FAILED: frozenset({ExecutionState.RETRY_SCHEDULED, ExecutionState.REPLANNING}),
    ExecutionState.COMPLETED: frozenset(),
}


def validate_transition(source: ExecutionState, target: ExecutionState) -> None:
    if source == target:
        return
    if source in TERMINAL_STATES and target not in TRANSITIONS[source]:
        raise InvalidTransitionError(f"terminal task cannot transition: {source.value} -> {target.value}")
    if target not in TRANSITIONS.get(source, frozenset()):
        raise InvalidTransitionError(f"invalid execution transition: {source.value} -> {target.value}")
