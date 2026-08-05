from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.cli import _decision, _operator_retry_decision, tasks_app
from mana_agent.execution_supervisor.errors import (
    BudgetExceededError,
    ConcurrentUpdateError,
    InvalidTransitionError,
    LeaseConflictError,
    RetrySafetyError,
    StaleLeaseError,
    CompletionVerificationError,
)
from mana_agent.execution_supervisor.models import (
    ActionRequestState,
    BudgetOverrunAction,
    BudgetOverrunFinalizationDecision,
    CheckpointRecord,
    CompletionContract,
    CompletionContractType,
    ExecutionState,
    EscrowResult,
    RecoveryAction,
    RecoveryDecision,
    RetryCategory,
    SideEffectClassification,
    TaskRecord,
    VerificationStatus,
    WaitPolicy,
)
from mana_agent.execution_supervisor.store import LocalExecutionStore
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor


class FakeClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


@pytest.fixture
def runtime(tmp_path):
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


def create(supervisor, tmp_path, **changes):
    options = {
        "routing_decision_id": "decision_test",
        "side_effect_classification": SideEffectClassification.READ_ONLY,
        "workspace_path": tmp_path,
    }
    options.update(changes)
    return supervisor.create_task(**options)


def running(supervisor, task):
    supervisor.queue(task.task_id)
    leased, token = supervisor.acquire_lease(task.task_id, owner="worker-a")
    supervisor.start(task.task_id, attempt_id=leased.attempt_id, lease_token=token)
    return leased.attempt_id, token


def decision(task_id, **changes):
    options = {
        "decision_id": "decision_recovery",
        "task_id": task_id,
        "action": RecoveryAction.RETRY,
        "retry_category": RetryCategory.TOOL,
        "reason": "explicit test recovery",
        "safe_to_continue": True,
    }
    options.update(changes)
    return RecoveryDecision(**options)


def test_valid_and_invalid_state_transitions_are_recorded(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    assert supervisor.queue(task.task_id).state == ExecutionState.QUEUED
    with pytest.raises(InvalidTransitionError):
        supervisor.transition(task.task_id, ExecutionState.COMPLETED)
    assert supervisor.store.events_for_task(task.task_id)[-1]["event_type"] == "invalid_transition"


def test_configuration_rejects_heartbeat_not_shorter_than_lease(tmp_path):
    with pytest.raises(ValidationError, match="heartbeat must be shorter"):
        ExecutionSupervisorConfig(
            root=tmp_path,
            lease_seconds=10,
            heartbeat_seconds=10,
        )


def test_atomic_compare_and_set_rejects_stale_writer(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    first = supervisor.store.get_task(task.task_id)
    second = supervisor.store.get_task(task.task_id)
    first.assigned_agent = "one"
    supervisor.store.compare_and_set(first, 0)
    second.assigned_agent = "two"
    with pytest.raises(ConcurrentUpdateError):
        supervisor.store.compare_and_set(second, 0)


def test_concurrent_lease_acquisition_has_one_winner(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    supervisor.queue(task.task_id)

    def claim(owner):
        try:
            return supervisor.acquire_lease(task.task_id, owner=owner)[0].lease_owner
        except LeaseConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(claim, ["one", "two"]))
    assert rows.count("conflict") == 1


def test_heartbeat_renews_lease_and_stale_token_is_rejected(runtime):
    supervisor, clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    attempt_id, token = running(supervisor, task)
    original = supervisor.store.get_task(task.task_id).lease_expires_at
    clock.advance(3)
    renewed = supervisor.heartbeat(task.task_id, attempt_id=attempt_id, lease_token=token)
    assert renewed.lease_expires_at > original
    with pytest.raises(StaleLeaseError):
        supervisor.heartbeat(task.task_id, attempt_id=attempt_id, lease_token="stale")


def test_unstarted_lease_can_be_released_without_reusing_attempt(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    supervisor.queue(task.task_id)
    leased, token = supervisor.acquire_lease(task.task_id, owner="worker-a")
    released = supervisor.release_lease(
        task.task_id,
        attempt_id=leased.attempt_id,
        lease_token=token,
        reason="worker declined before execution",
    )
    assert released.state == ExecutionState.QUEUED
    replacement, _replacement_token = supervisor.acquire_lease(task.task_id, owner="worker-b")
    assert replacement.attempt_id != leased.attempt_id


def test_expired_read_only_lease_recovers_once_and_rejects_old_result(runtime):
    supervisor, clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    attempt_id, token = running(supervisor, task)
    clock.advance(11)
    first = supervisor.recover()
    second = supervisor.recover()
    assert task.task_id in first.retry_scheduled
    assert task.task_id in second.unchanged
    clock.advance(2)
    supervisor.release_retry(task.task_id)
    with pytest.raises(StaleLeaseError):
        supervisor.submit_result(task.task_id, attempt_id=attempt_id, lease_token=token, payload={})


def test_restart_runs_startup_recovery_for_lost_worker(runtime):
    supervisor, clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    running(supervisor, task)
    clock.advance(11)
    restarted = ExecutionSupervisor(supervisor.config, clock=clock)
    recovered = restarted.store.get_task(task.task_id)
    assert recovered.state == ExecutionState.RETRY_SCHEDULED
    assert task.task_id in restarted.startup_recovery_summary.retry_scheduled


def test_restart_after_creation_preserves_unstarted_task(runtime):
    supervisor, clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    restarted = ExecutionSupervisor(supervisor.config, clock=clock)
    assert restarted.store.get_task(task.task_id).state == ExecutionState.CREATED


def test_restart_during_cancellation_finishes_cooperatively(runtime):
    supervisor, clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    running(supervisor, task)
    supervisor.transition(
        task.task_id,
        ExecutionState.CANCELLING,
        reason="operator cancellation interrupted by process exit",
    )
    restarted = ExecutionSupervisor(supervisor.config, clock=clock)
    recovered = restarted.store.get_task(task.task_id)
    assert recovered.state == ExecutionState.CANCELLED
    assert recovered.cancellation_status.value == "completed"


def test_unknown_lease_loss_requires_intervention(runtime):
    supervisor, clock, tmp_path = runtime
    task = create(
        supervisor,
        tmp_path,
        side_effect_classification=SideEffectClassification.UNKNOWN,
    )
    running(supervisor, task)
    clock.advance(11)
    summary = supervisor.recover()
    failed = supervisor.store.get_task(task.task_id)
    assert task.task_id in summary.intervention_required
    assert failed.state == ExecutionState.FAILED
    assert "may already have occurred" in failed.failure_reason


def test_checkpoint_and_durable_result_escrow_survive_restart(runtime):
    supervisor, _clock, tmp_path = runtime
    parent = create(supervisor, tmp_path)
    child = create(supervisor, tmp_path, parent_task_id=parent.task_id)
    attempt_id, token = running(supervisor, child)
    checkpoint = supervisor.checkpoint(
        child.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        resume_payload={"cursor": 4},
        completed_steps=["inspect"],
    )
    supervisor.set_completion_contract(
        child.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        contracts=[CompletionContract(
            contract_type=CompletionContractType.STRUCTURED_RESULT_VALID,
            metadata={"required_keys": ["answer"]},
        )],
    )
    completed = supervisor.submit_result(
        child.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        payload={"answer": 42},
    )
    restarted = ExecutionSupervisor(supervisor.config, clock=supervisor.clock)
    escrow = restarted.store.unacknowledged_results(parent.task_id)
    assert restarted.store.get_checkpoint(checkpoint.checkpoint_id).resume_payload == {"cursor": 4}
    assert completed.state == ExecutionState.COMPLETED
    assert len(escrow) == 1
    restarted.acknowledge_result(escrow[0].result_id, parent_task_id=parent.task_id)
    assert restarted.store.unacknowledged_results(parent.task_id) == []


def test_recovery_relinks_checkpoint_written_before_task_update(runtime):
    supervisor, clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    attempt_id, _token = running(supervisor, task)
    interrupted = CheckpointRecord(
        task_id=task.task_id,
        attempt_id=attempt_id,
        state_version=supervisor.store.get_task(task.task_id).state_version,
        resume_payload={"cursor": 9},
    )
    supervisor.store.save_checkpoint(interrupted)
    clock.advance(11)
    supervisor.recover()
    recovered = supervisor.store.get_task(task.task_id)
    assert recovered.checkpoint_id == interrupted.checkpoint_id
    assert recovered.state == ExecutionState.RETRY_SCHEDULED


def test_recovery_relinks_result_written_before_task_update(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    attempt_id, _token = running(supervisor, task)
    supervisor.set_completion_contract(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=_token,
        contracts=[CompletionContract(
            contract_type=CompletionContractType.STRUCTURED_RESULT_VALID,
            metadata={"required_keys": ["answer"]},
        )],
    )
    active = supervisor.store.get_task(task.task_id)
    interrupted = EscrowResult(
        task_id=task.task_id,
        attempt_id=attempt_id,
        lease_token_hash=active.lease_token,
        payload={"answer": 42},
    )
    supervisor.store.save_result(interrupted)
    summary = supervisor.recover()
    recovered = supervisor.store.get_task(task.task_id)
    assert recovered.result_id == interrupted.result_id
    assert recovered.state == ExecutionState.COMPLETED
    assert task.task_id in summary.recovered


def test_corrupt_checkpoint_refuses_resume(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    attempt_id, token = running(supervisor, task)
    checkpoint = supervisor.checkpoint(
        task.task_id, attempt_id=attempt_id, lease_token=token, resume_payload={"step": 1}
    )
    supervisor.store.root.joinpath("checkpoints", f"{checkpoint.checkpoint_id}.json").write_text("{bad", encoding="utf-8")
    with pytest.raises(RetrySafetyError, match="missing or corrupt"):
        supervisor.retry(task.task_id, decision(
            task.task_id,
            action=RecoveryAction.RESUME_CHECKPOINT,
            resume_checkpoint_id=checkpoint.checkpoint_id,
        ))


def test_corrupt_checkpoint_recovery_fails_closed(runtime):
    supervisor, clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    attempt_id, token = running(supervisor, task)
    checkpoint = supervisor.checkpoint(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        resume_payload={"step": 1},
    )
    supervisor.store.root.joinpath(
        "checkpoints", f"{checkpoint.checkpoint_id}.json"
    ).write_text("{bad", encoding="utf-8")
    clock.advance(11)
    summary = supervisor.recover()
    recovered = supervisor.store.get_task(task.task_id)
    assert recovered.state == ExecutionState.FAILED
    assert task.task_id in summary.intervention_required
    assert "no fallback action" in recovered.failure_reason


def test_retry_budget_backoff_and_idempotency_safety(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    attempt_id, token = running(supervisor, task)
    supervisor.transition(task.task_id, ExecutionState.FAILED, reason="tool failed")
    first = supervisor.retry_policy.backoff_seconds(task, RetryCategory.TOOL)
    supervisor.retry(task.task_id, decision(task.task_id))
    updated = supervisor.store.get_task(task.task_id)
    second = supervisor.retry_policy.backoff_seconds(updated, RetryCategory.TOOL)
    assert second > first

    unsafe = create(
        supervisor,
        tmp_path,
        side_effect_classification=SideEffectClassification.IDEMPOTENT,
    )
    running(supervisor, unsafe)
    supervisor.transition(unsafe.task_id, ExecutionState.FAILED, reason="ambiguous")
    with pytest.raises(RetrySafetyError, match="idempotency key"):
        supervisor.retry(unsafe.task_id, decision(unsafe.task_id))


def test_task_creation_records_provenance_and_ambiguous_actions_block_retry(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    assert task.schema_version == 7
    assert task.completion_contract
    assert task.field_provenance["actual_cost"] == "pending_runtime_accounting"

    attempt_id, token = running(supervisor, task)
    action = supervisor.prepare_action(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        tool_name="server_command",
        action_fingerprint="action-fingerprint",
        classification=SideEffectClassification.UNKNOWN,
    )
    supervisor.update_action(action.action_id, request_state=ActionRequestState.OUTCOME_UNKNOWN)
    supervisor.transition(task.task_id, ExecutionState.FAILED, reason="connection interrupted")

    with pytest.raises(RetrySafetyError, match="outcome is ambiguous"):
        supervisor.retry(task.task_id, decision(task.task_id))


def test_legacy_task_records_upgrade_to_metadata_provenance_schema() -> None:
    legacy = TaskRecord.model_validate(
        {
            "schema_version": 6,
            "task_id": "task_legacy",
            "routing_decision_id": "decision_legacy",
        }
    )

    assert legacy.schema_version == 7
    assert legacy.field_provenance["actual_cost"] == "pending_runtime_accounting"


def test_replan_limit_and_child_limits(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    for index in range(supervisor.config.max_replans):
        if index == 0:
            running(supervisor, task)
        supervisor.transition(task.task_id, ExecutionState.FAILED, reason="plan failed")
        replanning = supervisor.retry(task.task_id, decision(
            task.task_id,
            action=RecoveryAction.REPLAN,
            retry_category=RetryCategory.REPLAN,
        ))
        assert replanning.state == ExecutionState.REPLANNING
        supervisor.release_retry(task.task_id)
        leased, token = supervisor.acquire_lease(task.task_id, owner=f"worker-{index}")
        supervisor.start(task.task_id, attempt_id=leased.attempt_id, lease_token=token)
    supervisor.transition(task.task_id, ExecutionState.FAILED, reason="plan failed again")
    with pytest.raises(RetrySafetyError, match="budget is exhausted"):
        supervisor.retry(task.task_id, decision(
            task.task_id,
            action=RecoveryAction.REPLAN,
            retry_category=RetryCategory.REPLAN,
        ))

    constrained = ExecutionSupervisor(
        supervisor.config.model_copy(update={"max_children_per_task": 1}),
        clock=supervisor.clock,
    )
    parent = create(constrained, tmp_path)
    create(constrained, tmp_path, parent_task_id=parent.task_id)
    with pytest.raises(ValueError, match="maximum supervised children"):
        create(constrained, tmp_path, parent_task_id=parent.task_id)


def test_non_idempotent_manual_retry_is_refused(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(
        supervisor,
        tmp_path,
        side_effect_classification=SideEffectClassification.NON_IDEMPOTENT,
    )
    running(supervisor, task)
    supervisor.transition(task.task_id, ExecutionState.FAILED, reason="ambiguous external failure")
    with pytest.raises(RetrySafetyError, match="may already have produced"):
        supervisor.retry(task.task_id, decision(task.task_id))


def test_unknown_task_may_resume_exact_checkpoint_without_irreversible_side_effect(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(
        supervisor,
        tmp_path,
        side_effect_classification=SideEffectClassification.UNKNOWN,
    )
    attempt_id, token = running(supervisor, task)
    checkpoint = supervisor.checkpoint(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        resume_payload={"cursor": 2},
    )
    supervisor.transition(task.task_id, ExecutionState.FAILED, reason="worker stopped")

    scheduled = supervisor.retry(
        task.task_id,
        decision(
            task.task_id,
            action=RecoveryAction.RESUME_CHECKPOINT,
            resume_checkpoint_id=checkpoint.checkpoint_id,
        ),
    )

    assert scheduled.state == ExecutionState.RETRY_SCHEDULED
    assert scheduled.checkpoint_id == checkpoint.checkpoint_id


def test_unknown_task_may_retry_when_model_authorizes_same_stable_work(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(
        supervisor,
        tmp_path,
        side_effect_classification=SideEffectClassification.UNKNOWN,
    )
    running(supervisor, task)
    supervisor.transition(task.task_id, ExecutionState.FAILED, reason="worker stopped")

    scheduled = supervisor.retry(
        task.task_id,
        decision(
            task.task_id,
            same_task_retry_authorized=True,
        ),
    )

    assert scheduled.state == ExecutionState.RETRY_SCHEDULED


def test_unknown_task_retry_without_model_authorization_still_fails_closed(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(
        supervisor,
        tmp_path,
        side_effect_classification=SideEffectClassification.UNKNOWN,
    )
    running(supervisor, task)
    supervisor.transition(task.task_id, ExecutionState.FAILED, reason="worker stopped")

    with pytest.raises(RetrySafetyError, match="no retry was scheduled"):
        supervisor.retry(task.task_id, decision(task.task_id))


def test_legacy_unknown_retry_setting_does_not_bypass_model_authorization(runtime):
    supervisor, _clock, tmp_path = runtime
    supervisor.config.allow_unknown_side_effect_retry = True
    task = create(
        supervisor,
        tmp_path,
        side_effect_classification=SideEffectClassification.UNKNOWN,
    )
    running(supervisor, task)
    supervisor.transition(task.task_id, ExecutionState.FAILED, reason="worker stopped")

    with pytest.raises(RetrySafetyError, match="no retry was scheduled"):
        supervisor.retry(task.task_id, decision(task.task_id))


def test_operator_retry_decision_is_attached_from_task_id(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    running(supervisor, task)
    failed = supervisor.transition(task.task_id, ExecutionState.FAILED, reason="model failed")

    selected = _operator_retry_decision(
        supervisor,
        task.task_id,
        category=RetryCategory.MODEL,
    )

    assert selected.task_id == task.task_id
    assert selected.decision_id == f"operator-cli:{task.task_id}:{failed.state_version}:retry"
    assert selected.action == RecoveryAction.RETRY
    assert selected.retry_category == RetryCategory.MODEL
    assert selected.safe_to_continue is True
    assert supervisor.retry(task.task_id, selected).state == ExecutionState.RETRY_SCHEDULED


def test_retry_cli_does_not_require_decision_json(runtime, monkeypatch):
    supervisor, _clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    running(supervisor, task)
    supervisor.transition(task.task_id, ExecutionState.FAILED, reason="model failed")
    monkeypatch.setattr(
        "mana_agent.execution_supervisor.cli._supervisor",
        lambda: supervisor,
    )

    result = CliRunner().invoke(tasks_app, ["retry", task.task_id])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["state"] == ExecutionState.RETRY_SCHEDULED.value


def test_routing_decision_registry_has_actionable_retry_error(tmp_path):
    registry = tmp_path / "decisions.json"
    registry.write_text(
        json.dumps(
            {
                "decision_000001": {
                    "decision_id": "decision_000001",
                    "task_id": "task_000001",
                    "selected_route": "simple",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="routing-decision registry"):
        _decision(str(registry))


def test_cancellation_propagates_and_preserves_irreversible_child(runtime):
    supervisor, _clock, tmp_path = runtime
    parent = create(supervisor, tmp_path)
    child = create(supervisor, tmp_path, parent_task_id=parent.task_id)
    attempt_id, token = running(supervisor, child)
    supervisor.mark_irreversible_side_effect(child.task_id, attempt_id=attempt_id, lease_token=token)
    changed = supervisor.cancel(parent.task_id, reason="operator requested")
    assert parent.task_id in changed
    assert child.task_id not in changed
    assert supervisor.store.get_task(child.task_id).cancellation_status.value == "blocked_by_side_effect"
    assert supervisor.store.get_task(parent.task_id).cancellation_status.value == "partially_completed"


def test_parent_wait_timeout_and_minimum_success(runtime):
    supervisor, clock, tmp_path = runtime
    parent = create(
        supervisor,
        tmp_path,
        wait_policy=WaitPolicy.MINIMUM_SUCCESS_COUNT,
        minimum_success_count=1,
        deadline_at=clock() + timedelta(seconds=5),
    )
    create(supervisor, tmp_path, parent_task_id=parent.task_id)
    assert not supervisor.parent_progress(parent.task_id).satisfied
    clock.advance(6)
    assert supervisor.parent_progress(parent.task_id).timed_out


def test_token_and_cost_overrun_requires_a_fresh_model_finalization_decision(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(
        supervisor,
        tmp_path,
        token_budget=1,
        estimated_cost=0.05,
        monetary_budget=0.1,
    )
    attempt_id, token = running(supervisor, task)
    pending = supervisor.submit_result(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        payload={"ok": True},
        token_usage=2,
    )
    assert pending.state == ExecutionState.PENDING_BUDGET_DECISION
    assert pending.result_id
    assert pending.budget_overrun["status"] == "pending_model_decision"

    finalized = supervisor.finalize_budget_overrun(BudgetOverrunFinalizationDecision(
        decision_id="decision_budget_review",
        task_id=task.task_id,
        attempt_id=attempt_id,
        result_id=pending.result_id,
        result_evidence_hash=pending.budget_overrun["evidence_hash"],
        action=BudgetOverrunAction.REQUIRE_REVIEW,
        reason="model requires operator review",
        safe_to_continue=True,
    ))
    assert finalized.state == ExecutionState.PENDING_BUDGET_DECISION
    assert finalized.budget_overrun["status"] == "requires_human_review"


def test_budget_overrun_rejects_stale_model_decision_evidence(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(supervisor, tmp_path, token_budget=1)
    attempt_id, token = running(supervisor, task)
    pending = supervisor.submit_result(
        task.task_id, attempt_id=attempt_id, lease_token=token,
        payload={"ok": True}, token_usage=2,
    )

    with pytest.raises(BudgetExceededError, match="does not match durable result evidence"):
        supervisor.finalize_budget_overrun(BudgetOverrunFinalizationDecision(
            decision_id="decision_stale_budget_evidence",
            task_id=task.task_id,
            attempt_id=attempt_id,
            result_id=pending.result_id,
            result_evidence_hash="sha256:stale",
            action=BudgetOverrunAction.REQUIRE_REVIEW,
            reason="stale evidence",
            safe_to_continue=True,
        ))
    assert supervisor.store.get_task(task.task_id).state == ExecutionState.PENDING_BUDGET_DECISION


def test_duplicate_create_does_not_duplicate_event(runtime):
    supervisor, _clock, tmp_path = runtime
    task = create(supervisor, tmp_path, task_id="task_stable")
    create(supervisor, tmp_path, task_id="task_stable")
    created = [
        row for row in supervisor.store.events_for_task(task.task_id)
        if row["event_type"] == "task_created"
    ]
    assert len(created) == 1


def test_store_redacts_secrets_and_rotates_logs(tmp_path):
    store = LocalExecutionStore(tmp_path / "execution", max_log_bytes=4096)
    supervisor = ExecutionSupervisor(
        ExecutionSupervisorConfig(root=store.root, lease_seconds=10, heartbeat_seconds=2),
        store=store,
    )
    task = supervisor.create_task(
        task_id="task_logs",
        routing_decision_id="decision_logs",
        side_effect_classification=SideEffectClassification.READ_ONLY,
    )
    for index in range(80):
        supervisor._emit("log_event", task, api_key="secret-value", content="x" * 100)
    log_text = (store.root / "logs" / "execution.jsonl").read_text(encoding="utf-8")
    assert "secret-value" not in log_text
    assert (store.root / "logs" / "execution.jsonl.1").exists()


def test_file_completion_and_missing_or_changed_checksums(runtime):
    supervisor, _clock, tmp_path = runtime
    output = tmp_path / "output.txt"
    output.write_text("durable", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    task = create(supervisor, tmp_path)
    attempt_id, token = running(supervisor, task)
    supervisor.set_completion_contract(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        contracts=[CompletionContract(
            contract_type=CompletionContractType.FILE_EXISTS,
            path="output.txt",
            expected_sha256=digest,
            minimum_size=1,
        )],
    )
    assert supervisor.submit_result(
        task.task_id, attempt_id=attempt_id, lease_token=token, payload={}
    ).state == ExecutionState.COMPLETED

    for path, checksum in (("missing.txt", ""), ("output.txt", "0" * 64)):
        pending = create(supervisor, tmp_path)
        attempt_id, token = running(supervisor, pending)
        supervisor.set_completion_contract(
            pending.task_id,
            attempt_id=attempt_id,
            lease_token=token,
            contracts=[CompletionContract(
                contract_type=CompletionContractType.FILE_EXISTS,
                path=path,
                expected_sha256=checksum,
            )],
        )
        result = supervisor.submit_result(
            pending.task_id, attempt_id=attempt_id, lease_token=token, payload={}
        )
        assert result.state == ExecutionState.COMPLETED_PENDING_VERIFICATION
        assert result.verification_status == VerificationStatus.FAILED


def test_completion_paths_with_platform_safe_spaces(tmp_path):
    workspace = tmp_path / "workspace with spaces"
    workspace.mkdir()
    output = workspace / "result file.txt"
    output.write_text("verified", encoding="utf-8")
    config = ExecutionSupervisorConfig(
        root=tmp_path / "execution state",
        lease_seconds=10,
        heartbeat_seconds=2,
    )
    supervisor = ExecutionSupervisor(config)
    task = supervisor.create_task(
        routing_decision_id="decision_paths",
        side_effect_classification=SideEffectClassification.READ_ONLY,
        workspace_path=workspace,
        completion_contract=[CompletionContract(
            contract_type=CompletionContractType.FILE_EXISTS,
            path="result file.txt",
            minimum_size=1,
        )],
    )
    attempt_id, token = running(supervisor, task)
    completed = supervisor.submit_result(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        payload={},
    )
    assert completed.state == ExecutionState.COMPLETED


def test_atomic_writes_leave_no_partial_task_files(runtime):
    supervisor, _clock, tmp_path = runtime
    create(supervisor, tmp_path)
    assert not list((supervisor.store.root / "tasks").glob("*.tmp"))
    for path in (supervisor.store.root / "tasks").glob("*.json"):
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)


def test_atomic_store_retries_transient_replace_denial(runtime, monkeypatch):
    supervisor, _clock, tmp_path = runtime
    real_replace = os.replace
    replace_calls = 0

    def transiently_denied(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            raise PermissionError(13, "Access is denied")
        real_replace(source, destination)

    monkeypatch.setattr(
        "mana_agent.execution_supervisor.store._atomic_replace",
        transiently_denied,
    )
    task = create(supervisor, tmp_path)

    assert supervisor.store.get_task(task.task_id).task_id == task.task_id
    assert replace_calls == 2


def test_consequential_action_duplicate_and_stale_generation_are_fenced(runtime):
    supervisor, clock, tmp_path = runtime
    task = create(supervisor, tmp_path)
    attempt_id, token = running(supervisor, task)
    action = supervisor.prepare_action(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        tool_name="email_send",
        action_fingerprint="recipient:subject:body-hash",
        classification=SideEffectClassification.NON_IDEMPOTENT,
    )
    supervisor.update_action(action.action_id, request_state=ActionRequestState.STARTED)
    with pytest.raises(RetrySafetyError, match="reconcile"):
        supervisor.prepare_action(
            task.task_id,
            attempt_id=attempt_id,
            lease_token=token,
            tool_name="email_send",
            action_fingerprint="recipient:subject:body-hash",
            classification=SideEffectClassification.NON_IDEMPOTENT,
        )

    with pytest.raises(RetrySafetyError, match="outcome is ambiguous"):
        supervisor.retry(task.task_id, decision(task.task_id))
    supervisor.update_action(action.action_id, request_state=ActionRequestState.FAILED)
    supervisor.retry(task.task_id, decision(task.task_id))
    clock.advance(120)
    supervisor.release_retry(task.task_id)
    next_attempt, _next_token = running(supervisor, supervisor.store.get_task(task.task_id))
    assert next_attempt != attempt_id
    with pytest.raises(StaleLeaseError, match="generation"):
        supervisor.update_action(
            action.action_id,
            request_state=ActionRequestState.SUCCEEDED,
            external_receipt="provider-receipt",
        )


def test_completed_file_result_is_reverified_against_recorded_hash(runtime):
    supervisor, _clock, tmp_path = runtime
    output = tmp_path / "reverify.txt"
    output.write_text("first", encoding="utf-8")
    task = create(supervisor, tmp_path)
    attempt_id, token = running(supervisor, task)
    supervisor.set_completion_contract(
        task.task_id,
        attempt_id=attempt_id,
        lease_token=token,
        contracts=[CompletionContract(
            contract_type=CompletionContractType.FILE_EXISTS,
            path="reverify.txt",
            minimum_size=1,
        )],
    )
    assert supervisor.submit_result(
        task.task_id, attempt_id=attempt_id, lease_token=token, payload={}
    ).state == ExecutionState.COMPLETED
    output.write_text("changed", encoding="utf-8")

    with pytest.raises(CompletionVerificationError, match="no longer satisfies"):
        supervisor.reverify_completed(task.task_id)
    assert supervisor.store.get_task(task.task_id).verification_status == VerificationStatus.FAILED
