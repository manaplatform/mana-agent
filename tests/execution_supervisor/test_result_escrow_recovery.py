from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.errors import (
    EscrowConflictError,
    EscrowCorruptError,
    EscrowIncompatibleVersionError,
)
from mana_agent.execution_supervisor.models import (
    CompletionContract,
    CompletionContractType,
    EscrowLookupStatus,
    EscrowResult,
    EscrowStatus,
    ExecutionState,
    ResultAcknowledgement,
    SideEffectClassification,
    TaskRecord,
    VerificationReport,
    VerificationStatus,
)
from mana_agent.execution_supervisor.store import LocalExecutionStore
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
from mana_agent.gateway.chat_turn_store import ChatTurnRecord, ChatTurnStore


class FakeClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 14, 0, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


@pytest.fixture
def supervisor_runtime(tmp_path):
    clock = FakeClock()
    config = ExecutionSupervisorConfig(
        root=tmp_path / "execution",
        lease_seconds=10,
        heartbeat_seconds=2,
        base_backoff_seconds=1,
        max_backoff_seconds=60,
    )
    supervisor = ExecutionSupervisor(config, clock=clock)
    return supervisor, clock, tmp_path


def _start_task(supervisor, tmp_path, **kwargs):
    opts = {
        "routing_decision_id": "decision_test",
        "side_effect_classification": SideEffectClassification.READ_ONLY,
        "workspace_path": tmp_path,
    }
    opts.update(kwargs)
    task = supervisor.create_task(**opts)
    supervisor.queue(task.task_id)
    leased, token = supervisor.acquire_lease(task.task_id, owner="worker-1")
    supervisor.start(task.task_id, attempt_id=leased.attempt_id, lease_token=token)
    return task, leased.attempt_id, token


def test_scenario_a_durable_recovery_after_caller_restart(supervisor_runtime):
    """Lane completes -> verified result persisted -> process restarts -> caller recovers."""
    supervisor, clock, tmp_path = supervisor_runtime
    task, attempt_id, token = _start_task(supervisor, tmp_path)

    # Submit result (auto-verifies and completes when no custom contract is set)
    completed_task = supervisor.submit_result(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        payload={"chat_result": {"answer": "Execution succeeded.", "mode": "chat"}},
    )
    assert completed_task.state == ExecutionState.COMPLETED

    # Simulate restart by creating a new supervisor instance on the same store root
    restarted = ExecutionSupervisor(supervisor.config, clock=clock)

    # Durable turn recovery lookup by authoritative execution_id
    lookup = restarted.get_verified_execution_result(task.task_id)
    assert lookup.status == EscrowLookupStatus.FOUND
    assert lookup.result is not None
    assert lookup.result.payload["chat_result"]["answer"] == "Execution succeeded."
    assert lookup.is_terminal is True
    assert lookup.is_verified is True


def test_scenario_b_async_completion_and_verification(supervisor_runtime):
    """Lane and caller execute asynchronously: result persisted in escrow before verification."""
    supervisor, clock, tmp_path = supervisor_runtime
    task, attempt_id, token = _start_task(supervisor, tmp_path)

    # Set completion contract requiring an artifact
    supervisor.set_completion_contract(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        contracts=[
            CompletionContract(
                contract_type=CompletionContractType.FILE_EXISTS,
                path="out.txt",
            )
        ],
    )

    # Submit result before artifact is created (verification fails/remains pending)
    supervisor.submit_result(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        payload={"chat_result": {"answer": "Pending check.", "mode": "chat"}},
    )

    # Caller checks status while verification is pending
    lookup_pending = supervisor.get_verified_execution_result(task.task_id)
    assert lookup_pending.status == EscrowLookupStatus.UNVERIFIED
    assert lookup_pending.error_code == "RESULT_NOT_VERIFIED"

    # Async worker creates expected artifact and triggers verification
    (tmp_path / "out.txt").write_text("completed artifact", encoding="utf-8")
    supervisor.verify_completion(task.task_id)

    lookup_done = supervisor.get_verified_execution_result(task.task_id)
    assert lookup_done.status == EscrowLookupStatus.FOUND
    assert lookup_done.is_verified is True
    assert lookup_done.result.payload["chat_result"]["answer"] == "Pending check."


def test_scenario_c_in_memory_result_lost_gateway_recovery(supervisor_runtime):
    """In-memory result is lost; escrow recovers result and emits response."""
    supervisor, clock, tmp_path = supervisor_runtime
    task, attempt_id, token = _start_task(supervisor, tmp_path)

    supervisor.submit_result(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        payload={"chat_result": {"answer": "Durable answer.", "mode": "chat"}},
    )

    # Clean supervisor instance with no in-memory cache
    fresh_supervisor = ExecutionSupervisor(supervisor.config, clock=clock)
    lookup = fresh_supervisor.get_verified_execution_result(task.task_id)
    assert lookup.status == EscrowLookupStatus.FOUND
    assert lookup.result.result_id == task.task_id or lookup.result.execution_id == task.task_id
    assert lookup.result.payload["chat_result"]["answer"] == "Durable answer."


def test_scenario_d_interrupted_delivery_after_verification(supervisor_runtime):
    """Result delivery interrupted after verification: can be re-queried and acknowledged."""
    supervisor, clock, tmp_path = supervisor_runtime
    task, attempt_id, token = _start_task(supervisor, tmp_path)

    supervisor.submit_result(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        payload={"chat_result": {"answer": "Interrupted delivery", "mode": "chat"}},
    )

    # Query without acknowledging
    lookup1 = supervisor.get_verified_execution_result(task.task_id)
    assert lookup1.status == EscrowLookupStatus.FOUND
    assert lookup1.acknowledgement is None

    # Acknowledge result
    ack = supervisor.acknowledge_result(
        lookup1.result.result_id,
        consumer_execution_id=task.task_id,
        consumer_turn_id="turn_123",
    )
    assert ack.consumer_turn_id == "turn_123"

    # Re-query confirms acknowledgement is recorded
    lookup2 = supervisor.get_verified_execution_result(task.task_id)
    assert lookup2.status == EscrowLookupStatus.FOUND
    assert lookup2.acknowledgement is not None
    assert lookup2.acknowledgement.consumer_turn_id == "turn_123"


def test_scenario_e_terminal_failure_durability(supervisor_runtime):
    """Terminal failures (FAILED, CANCELLED, BUDGET_EXHAUSTED) persist to escrow."""
    supervisor, clock, tmp_path = supervisor_runtime

    # 1. Failed task
    task1, _, _ = _start_task(supervisor, tmp_path)
    supervisor.transition(task1.task_id, ExecutionState.FAILED, reason="Compilation error")
    lookup1 = supervisor.get_verified_execution_result(task1.task_id)
    assert lookup1.status == EscrowLookupStatus.FOUND
    assert lookup1.is_terminal is True
    assert lookup1.result.supervisor_state == "failed"
    assert lookup1.result.result_kind == "terminal_failure"
    assert lookup1.result.error_metadata["reason"] == "Compilation error"

    # 2. Cancelled task
    task2, _, _ = _start_task(supervisor, tmp_path)
    supervisor.cancel(task2.task_id, reason="User cancelled")
    lookup2 = supervisor.get_verified_execution_result(task2.task_id)
    assert lookup2.status == EscrowLookupStatus.FOUND
    assert lookup2.is_terminal is True
    assert lookup2.result.supervisor_state == "cancelled"

    # 3. Budget exhausted task
    task3, _, _ = _start_task(supervisor, tmp_path)
    supervisor.transition(task3.task_id, ExecutionState.BUDGET_EXHAUSTED, reason="Cost limit hit")
    lookup3 = supervisor.get_verified_execution_result(task3.task_id)
    assert lookup3.status == EscrowLookupStatus.FOUND
    assert lookup3.is_terminal is True
    assert lookup3.result.supervisor_state == "budget_exhausted"


def test_scenario_f_idempotent_escrow_writes(supervisor_runtime):
    """Identical escrow write succeeds idempotently; conflicting write raises EscrowConflictError."""
    supervisor, clock, tmp_path = supervisor_runtime
    task, attempt_id, token = _start_task(supervisor, tmp_path)

    res1 = EscrowResult(
        result_id="res_idempotent_1",
        execution_id=task.task_id,
        task_id=task.task_id,
        attempt_id=attempt_id,
        attempt_generation=1,
        lease_token_hash="sha256:" + "a" * 64,
        payload={"data": "first"},
        status=EscrowStatus.AVAILABLE,
    )
    supervisor.store.save_result(res1)

    # Identical write is accepted idempotently
    res1_duplicate = EscrowResult(
        result_id="res_idempotent_1",
        execution_id=task.task_id,
        task_id=task.task_id,
        attempt_id=attempt_id,
        attempt_generation=1,
        lease_token_hash="sha256:" + "a" * 64,
        payload={"data": "first"},
        status=EscrowStatus.AVAILABLE,
    )
    supervisor.store.save_result(res1_duplicate)

    # Conflicting payload for the same result_id raises EscrowConflictError
    res1_conflicting = EscrowResult(
        result_id="res_idempotent_1",
        execution_id=task.task_id,
        task_id=task.task_id,
        attempt_id=attempt_id,
        attempt_generation=1,
        lease_token_hash="sha256:" + "a" * 64,
        payload={"data": "conflicting_payload"},
        status=EscrowStatus.AVAILABLE,
    )
    with pytest.raises(EscrowConflictError):
        supervisor.store.save_result(res1_conflicting)


def test_scenario_g_differentiated_lookup_statuses(supervisor_runtime):
    """get_verified_execution_result returns differentiated statuses."""
    supervisor, clock, tmp_path = supervisor_runtime

    # 1. NOT_FOUND for unknown identity
    lookup_not_found = supervisor.get_verified_execution_result("unknown_id")
    assert lookup_not_found.status == EscrowLookupStatus.NOT_FOUND
    assert lookup_not_found.error_code == "RESULT_NOT_FOUND"

    # 2. EXECUTION_STILL_RUNNING for active task
    task_running, _, _ = _start_task(supervisor, tmp_path)
    lookup_running = supervisor.get_verified_execution_result(task_running.task_id)
    assert lookup_running.status == EscrowLookupStatus.EXECUTION_STILL_RUNNING
    assert lookup_running.error_code == "EXECUTION_STILL_RUNNING"

    # 3. UNVERIFIED for task awaiting verification
    task_unverified, att_unver, tok_unver = _start_task(supervisor, tmp_path)
    supervisor.set_completion_contract(
        task_unverified.task_id,
        attempt_id=att_unver,
        lease_token=tok_unver,
        contracts=[
            CompletionContract(
                contract_type=CompletionContractType.FILE_EXISTS,
                path="pending_file.txt",
            )
        ],
    )
    supervisor.submit_result(
        task_unverified.task_id,
        attempt_id=att_unver,
        lease_token=tok_unver,
        payload={"chat_result": {}},
    )
    lookup_unverified = supervisor.get_verified_execution_result(task_unverified.task_id)
    assert lookup_unverified.status == EscrowLookupStatus.UNVERIFIED


def test_scenario_h_corrupt_escrow_file_handling(supervisor_runtime):
    """Corrupted JSON in escrow store raises EscrowCorruptError and returns CORRUPT status."""
    supervisor, clock, tmp_path = supervisor_runtime
    task, attempt_id, token = _start_task(supervisor, tmp_path)
    supervisor.submit_result(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        payload={"chat_result": {"answer": "good"}},
    )

    # Corrupt the escrow file on disk
    res_paths = list((supervisor.store.root / "results").glob("*.json"))
    assert res_paths
    res_paths[0].write_text("{corrupt json ...", encoding="utf-8")

    lookup = supervisor.get_verified_execution_result(task.task_id)
    assert lookup.status == EscrowLookupStatus.CORRUPT
    assert lookup.error_code == "RESULT_CORRUPT"


def test_scenario_i_schema_version_migration_and_rejection(supervisor_runtime):
    """v1 records migrate cleanly with defaults; future version > 2 raises EscrowIncompatibleVersionError."""
    supervisor, clock, tmp_path = supervisor_runtime

    # 1. Legacy v1 record (no execution_id, no schema_version, status="available")
    v1_raw = {
        "result_id": "res_v1_legacy",
        "task_id": "task_v1_legacy",
        "parent_task_id": "task_parent",
        "attempt_id": "attempt_1",
        "attempt_generation": 1,
        "lease_token_hash": "sha256:" + "0" * 64,
        "payload": {"answer": 42},
        "status": "available",
        "created_at": "2026-01-01T00:00:00Z",
    }
    migrated = EscrowResult.model_validate(v1_raw)
    assert migrated.schema_version == 2
    assert migrated.execution_id == "task_v1_legacy"
    assert migrated.root_task_id == "task_v1_legacy"
    assert migrated.result_kind == "chat_result"

    # Save to store and retrieve
    res_path = supervisor.store.root / "results" / "res_v1_legacy.json"
    res_path.write_text(json.dumps(v1_raw), encoding="utf-8")
    loaded = supervisor.store.get_result("res_v1_legacy")
    assert loaded is not None
    assert loaded.execution_id == "task_v1_legacy"

    # 2. Future schema version 3 is rejected
    v3_raw = {
        "result_id": "res_v3_future",
        "task_id": "task_v3",
        "schema_version": 3,
        "payload": {},
        "status": "available",
    }
    res_v3_path = supervisor.store.root / "results" / "res_v3_future.json"
    res_v3_path.write_text(json.dumps(v3_raw), encoding="utf-8")
    with pytest.raises(EscrowIncompatibleVersionError):
        supervisor.store.get_result("res_v3_future")


def test_scenario_j_result_acknowledgement_separation(supervisor_runtime):
    """Acknowledgement is stored in separate record; immutable EscrowResult is unmutated."""
    supervisor, clock, tmp_path = supervisor_runtime
    parent = supervisor.create_task(
        routing_decision_id="decision_parent",
        side_effect_classification=SideEffectClassification.READ_ONLY,
        workspace_path=tmp_path,
    )
    task, attempt_id, token = _start_task(
        supervisor, tmp_path, parent_task_id=parent.task_id
    )
    supervisor.submit_result(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        payload={"chat_result": {"answer": "done"}},
    )

    unack_before = supervisor.store.unacknowledged_results(parent.task_id)
    assert len(unack_before) == 1

    ack = supervisor.acknowledge_result(
        unack_before[0].result_id,
        parent_task_id=parent.task_id,
        consumer_execution_id=parent.task_id,
        consumer_turn_id="turn_42",
    )
    assert ack.consumer_turn_id == "turn_42"

    unack_after = supervisor.store.unacknowledged_results(parent.task_id)
    assert len(unack_after) == 0

    # Ensure acknowledgement file was created separately
    ack_loaded = supervisor.store.get_acknowledgement(unack_before[0].result_id)
    assert ack_loaded is not None
    assert ack_loaded.consumer_turn_id == "turn_42"


def test_scenario_k_cross_turn_exactly_once_response_coordination(tmp_path):
    """ChatTurnStore coordination prevents duplicate response emission on replayed turns."""
    turn_store = ChatTurnStore("test_session_id")
    created, exists = turn_store.create_or_get(
        conversation_id="conv_1",
        user_message_id="msg_1",
        turn_id="turn_abc",
        text="Initial user prompt",
    )
    assert exists is False
    assert created.status == "received"

    # First completion finalizes turn
    created.status = "responded"
    created.response = {"answer": "First response", "payload": {"execution_id": "exec_1"}}
    turn_store.update(created)

    # Subsequent replay checks turn store and gets already-responded outcome
    persisted, exists_again = turn_store.create_or_get(
        conversation_id="conv_1",
        user_message_id="msg_1",
        turn_id="turn_abc",
        text="Initial user prompt",
    )
    assert exists_again is True
    assert persisted.status == "responded"
    assert persisted.response["answer"] == "First response"


def test_scenario_l_identifier_mismatch_regression(supervisor_runtime):
    """task_id != execution_id: authoritative lookup returns FOUND; wrong lookup does not return mismatched result."""
    supervisor, clock, tmp_path = supervisor_runtime
    exec_id = "exec_authoritative_999"
    task_id = "task_internal_888"

    res = EscrowResult(
        result_id="res_mismatch_1",
        execution_id=exec_id,
        task_id=task_id,
        attempt_id="att_1",
        attempt_generation=1,
        lease_token_hash="sha256:" + "1" * 64,
        payload={"chat_result": {"answer": "Found by execution_id"}},
        status=EscrowStatus.AVAILABLE,
        supervisor_state="completed",
        verification_status=VerificationStatus.PASSED,
    )
    supervisor.store.save_result(res)

    # Lookup by authoritative execution_id -> FOUND
    lookup_by_exec = supervisor.get_verified_execution_result(exec_id)
    assert lookup_by_exec.status == EscrowLookupStatus.FOUND
    assert lookup_by_exec.result.payload["chat_result"]["answer"] == "Found by execution_id"

    # Lookup by task_id -> FOUND through legacy/task lookup
    lookup_by_task = supervisor.get_verified_execution_result(task_id)
    assert lookup_by_task.status == EscrowLookupStatus.FOUND

    # Lookup by non-existent ID -> NOT_FOUND (no silent return of wrong result)
    lookup_invalid = supervisor.get_verified_execution_result("wrong_random_id")
    assert lookup_invalid.status == EscrowLookupStatus.NOT_FOUND


def test_scenario_m_resumable_state_recovery(supervisor_runtime):
    """Resumable waiting states (approval_required, auth_required, blocked) are not marked terminal failed."""
    supervisor, clock, tmp_path = supervisor_runtime
    task, attempt_id, token = _start_task(supervisor, tmp_path)

    # Save checkpoint before suspension
    checkpoint = supervisor.checkpoint(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        resume_payload={"step": 1},
    )

    # Task transitions to WAITING for human approval
    supervisor.suspend_for_human_input(
        task.task_id,
        inbox_item_id="inbox_item_approval_1",
        checkpoint_id=checkpoint.checkpoint_id,
        request_type="approval",
    )

    lookup = supervisor.get_verified_execution_result(task.task_id)
    assert lookup.is_resumable is True
    assert lookup.requires_action is True
    assert lookup.is_terminal is False
