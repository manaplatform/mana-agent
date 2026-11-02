"""Storage-only tests: live delivery and supervision are not implemented here."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from mana_agent.execution.steering import (
    SteeringKind,
    SteeringLedger,
    SteeringScope,
    SteeringSubmission,
)


def submission(request_id: str, **changes: object) -> SteeringSubmission:
    values = dict(
        execution_id="exec-1", session_id="session-1", attempt_id="attempt-1",
        generation=1, request_id=request_id, text="Run tests first",
        kind=SteeringKind.PRIORITY,
    )
    values.update(changes)
    return SteeringSubmission(**values)


def test_persists_order_and_deduplicates(tmp_path):
    path = tmp_path / "steering.sqlite"
    ledger = SteeringLedger(path)
    first, duplicate = ledger.append(submission("one"))
    assert (first.sequence, duplicate) == (1, False)
    assert ledger.append(submission("one")) == (first, True)
    with pytest.raises(ValueError, match="different"):
        ledger.append(submission("one", text="Different instruction"))
    second, _ = ledger.append(submission("two", scope=SteeringScope.TASK, target_id="task-1"))
    assert second.sequence == 2
    assert SteeringLedger(path).list_entries("exec-1") == [first, second]
    assert ledger.list_entries("other") == []


def test_concurrent_submissions_and_retries_are_atomic(tmp_path):
    path = tmp_path / "steering.sqlite"
    SteeringLedger(path)

    def send(index):
        return SteeringLedger(path).append(submission(f"request-{index // 2}"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(send, range(40)))
    entries = SteeringLedger(path).list_entries("exec-1")
    assert [item.sequence for item in entries] == list(range(1, 21))
    assert sum(not duplicate for _, duplicate in results) == 20
    assert len({item.request_id for item in entries}) == 20


@pytest.mark.parametrize("changes", [
    {"text": "   "}, {"attempt_id": " "}, {"generation": 0},
    {"scope": SteeringScope.TASK},
    {"scope": SteeringScope.WORKER, "target_id": " "},
    {"scope": SteeringScope.EXECUTION, "target_id": "task-1"},
])
def test_invalid_inputs_are_rejected(changes):
    with pytest.raises(ValidationError):
        submission("one", **changes)


def test_sequence_is_per_execution_and_ids_are_execution_scoped(tmp_path):
    ledger = SteeringLedger(tmp_path / "steering.sqlite")
    first, _ = ledger.append(submission("same"))
    other, _ = ledger.append(submission("same", execution_id="exec-2"))
    assert first.sequence == other.sequence == 1
    assert ledger.list_entries("exec-1") == [first]
    assert ledger.list_entries("exec-2") == [other]
