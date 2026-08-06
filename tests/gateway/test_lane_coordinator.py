from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import threading

import pytest

from mana_agent.execution_supervisor import (
    BudgetOverrunAction,
    BudgetOverrunFinalizationDecision,
    ExecutionState as SupervisorState,
    RecoveryDecision,
    SideEffectClassification,
)
from mana_agent.execution_supervisor.models import RecoveryAction, RetryCategory
from mana_agent.multi_agent.core.types import TaskStatus
from mana_agent.gateway.lane_coordinator import (
    LaneBudget,
    LaneBudgetError,
    LaneCoordinator,
    LaneCoordinatorError,
    LaneHandoff,
    LaneHandoffError,
    LaneReservation,
)
from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.gateway.lanes import (
    LockMode,
    LaneId,
    LanePermissionError,
    LanePriority,
    LaneTaskState,
    configured_lane_contracts,
    default_lane_contracts,
    select_lane,
    validate_tool_permission,
)


@pytest.fixture
def coordinator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LaneCoordinator:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    return LaneCoordinator(root)


def _reserve(
    coordinator: LaneCoordinator,
    lane: LaneId,
    *,
    intent: str = "task",
    files: tuple[str, ...] = (),
    session: str = "session-1",
) -> LaneReservation:
    return coordinator.reserve(
        normalized_intent=intent,
        lane_id=lane,
        session_id=session,
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
        target_files=files,
        requested_input_tokens=100,
        requested_output_tokens=200,
    )


def test_default_contracts_define_all_specialist_lanes() -> None:
    contracts = default_lane_contracts()

    assert set(contracts) == set(LaneId)
    assert contracts[LaneId.CODING].lock_policy == LockMode.FILE_WRITE
    assert contracts[LaneId.RESEARCH].max_concurrent_jobs == 4
    assert contracts[LaneId.REVIEW].can_create_subagents is False
    assert contracts[LaneId.RELEASE].lock_policy == LockMode.REPOSITORY_WRITE


def test_semantic_fingerprint_deduplicates_active_work_across_sessions(
    coordinator: LaneCoordinator,
) -> None:
    first = _reserve(
        coordinator,
        LaneId.CODING,
        intent="create the deployment manifest",
        files=("deploy.yaml",),
        session="session-1",
    )
    coordinator.start(first)

    duplicate = _reserve(
        coordinator,
        LaneId.CODING,
        intent="  create   the deployment manifest ",
        files=("deploy.yaml",),
        session="session-2",
    )

    assert duplicate.duplicate is True
    assert duplicate.execution.task_id == first.execution.task_id
    durable = coordinator.execution_supervisor.store.get_task(first.execution.task_id)
    assert durable.completion_contract
    assert durable.field_provenance["completion_contract"] == "model_selected_lane_contract"


def test_recovery_candidates_include_failed_task_from_another_session(
    coordinator: LaneCoordinator,
) -> None:
    reservation = _reserve(
        coordinator,
        LaneId.CODING,
        intent="create the deployment manifest",
        session="session-before-restart",
    )
    coordinator.start(reservation)
    coordinator.finish(
        reservation.execution.task_id,
        state=LaneTaskState.FAILED,
        error="worker disconnected",
    )
    gateway = object.__new__(AgentChatGateway)
    gateway._lane_coordinator = coordinator

    candidates = gateway._recovery_candidates(
        lane_id=None,
        session_id="session-new",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
    )

    assert [item["task_id"] for item in candidates] == [reservation.execution.task_id]
    assert candidates[0]["session_id"] == "session-before-restart"
    assert candidates[0]["completion_contract"]


def test_exposes_supervisor_store_for_human_inbox_branch_controller(
    coordinator: LaneCoordinator,
) -> None:
    assert coordinator.store is coordinator.execution_supervisor.store


def test_human_inbox_wait_delegates_to_the_active_supervisor_branch(
    coordinator: LaneCoordinator,
) -> None:
    dispatched = threading.Event()
    observed: dict[str, object] = {}

    def dispatch(task_id: str, inbox_item_id: str, resume_claim_id: str, response: dict) -> None:
        observed.update({
            "task_id": task_id,
            "inbox_item_id": inbox_item_id,
            "resume_claim_id": resume_claim_id,
            "response": response,
            "lane_state": coordinator.inspect_task(task_id).state,
        })
        dispatched.set()

    coordinator.set_human_resume_dispatcher(dispatch)
    reservation = _reserve(coordinator, LaneId.CODING, intent="approval gated action")
    coordinator.start(reservation)
    task_id = reservation.execution.task_id
    checkpoint_id = coordinator.checkpoint(task_id, boundary="await-approval")

    waiting = coordinator.suspend_for_human_input(
        task_id,
        inbox_item_id="inbox-approval-1",
        checkpoint_id=checkpoint_id,
        request_type="approval",
    )

    assert waiting.waiting_inbox_item_id == "inbox-approval-1"
    assert coordinator.inspect_task(task_id).state is LaneTaskState.WAITING
    resumed = coordinator.resume_from_human_input(
        task_id,
        inbox_item_id="inbox-approval-1",
        checkpoint_id=checkpoint_id,
        resume_claim_id="resume-claim-1",
        structured_response={"operation": "approve"},
    )
    assert resumed.human_resume_claim_ids == ["resume-claim-1"]
    assert coordinator.inspect_task(task_id).state is LaneTaskState.QUEUED
    assert coordinator.inspect_task(task_id).supervisor_lease_token == ""
    assert coordinator.taskboard.get_task(task_id).status is TaskStatus.QUEUED
    assert dispatched.wait(timeout=1)
    assert observed == {
        "task_id": task_id,
        "inbox_item_id": "inbox-approval-1",
        "resume_claim_id": "resume-claim-1",
        "response": {"operation": "approve"},
        "lane_state": LaneTaskState.QUEUED,
    }


def test_recovered_human_resume_dispatches_only_the_matching_queued_branch(
    coordinator: LaneCoordinator,
) -> None:
    dispatched = threading.Event()
    coordinator.set_human_resume_dispatcher(
        lambda *_args: dispatched.set()
    )
    reservation = _reserve(coordinator, LaneId.CODING, intent="recover approval")
    coordinator.start(reservation)
    task_id = reservation.execution.task_id
    checkpoint_id = coordinator.checkpoint(task_id, boundary="await-approval")
    coordinator.suspend_for_human_input(
        task_id,
        inbox_item_id="inbox-recovery-1",
        checkpoint_id=checkpoint_id,
        request_type="approval",
    )
    coordinator.resume_from_human_input(
        task_id,
        inbox_item_id="inbox-recovery-1",
        checkpoint_id=checkpoint_id,
        resume_claim_id="resume-claim-recovery-1",
        structured_response={"operation": "approve"},
    )
    assert dispatched.wait(timeout=1)
    dispatched.clear()

    assert coordinator.dispatch_queued_human_resume(
        task_id,
        inbox_item_id="inbox-recovery-1",
        resume_claim_id="resume-claim-recovery-1",
        structured_response={"operation": "approve"},
    )
    assert dispatched.wait(timeout=1)
    coordinator.start(LaneReservation(coordinator.inspect_task(task_id)))
    assert not coordinator.dispatch_queued_human_resume(
        task_id,
        inbox_item_id="inbox-recovery-1",
        resume_claim_id="resume-claim-recovery-1",
        structured_response={"operation": "approve"},
    )
    coordinator.finish(task_id)


def test_lane_selection_uses_decision_intent_and_invalid_model_lane_uses_valid_route() -> None:
    assert select_lane(entry_route="coding") == LaneId.CODING
    assert select_lane(intent="verify") == LaneId.VERIFY
    assert select_lane(entry_route="search", model_lane="not-a-lane") == LaneId.RESEARCH
    assert select_lane(entry_route="remote_execution") == LaneId.OPERATIONS
    assert select_lane(entry_route="canvas") == LaneId.CANVAS
    with pytest.raises(ValueError, match="No valid specialist lane decision"):
        select_lane(entry_route="missing", model_lane="not-a-lane")


def test_invalid_lane_configuration_fails_clearly() -> None:
    with pytest.raises(ValueError, match="unknown specialist lane"):
        configured_lane_contracts({"unknown": {"enabled": True}})
    with pytest.raises(ValueError, match="max_concurrent_jobs"):
        configured_lane_contracts({"coding": {"max_concurrent_jobs": 0}})


def test_tool_permissions_are_enforced_by_capability() -> None:
    contracts = default_lane_contracts()

    assert validate_tool_permission(contracts[LaneId.CODING], "edit_file") == {"repository_write"}
    assert validate_tool_permission(contracts[LaneId.RESEARCH], "web_search") == {"web_search"}
    assert validate_tool_permission(
        contracts[LaneId.OPERATIONS], "remote_ssh_execute", task_capabilities=("remote_ssh_execute",)
    ) == {"remote_ssh_execute"}
    assert validate_tool_permission(
        contracts[LaneId.OPERATIONS], "automation_create", task_capabilities=("automation",)
    ) == {"automation"}
    with pytest.raises(LanePermissionError):
        validate_tool_permission(contracts[LaneId.REVIEW], "edit_file")


def test_duplicate_active_task_reuses_existing_reference(coordinator: LaneCoordinator) -> None:
    first = _reserve(coordinator, LaneId.RESEARCH, intent="inspect dependency")
    second = _reserve(coordinator, LaneId.RESEARCH, intent=" inspect   dependency ")

    assert second.duplicate is True
    assert second.execution.task_id == first.execution.task_id


def test_task_id_allocation_skips_supervisor_only_record(
    coordinator: LaneCoordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing_task_id = "task_20260803_000002"
    coordinator.execution_supervisor.create_task(
        task_id=existing_task_id,
        routing_decision_id="existing-decision",
        side_effect_classification=SideEffectClassification.UNKNOWN,
    )
    generated_ids = iter([existing_task_id, "task_20260803_000003"])
    monkeypatch.setattr(
        "mana_agent.multi_agent.taskboard.taskboard.new_task_id",
        lambda: next(generated_ids),
    )

    reservation = _reserve(coordinator, LaneId.RESEARCH, intent="new task")

    assert reservation.execution.task_id == "task_20260803_000003"
    assert (
        coordinator.execution_supervisor.store.get_task(
            existing_task_id
        ).routing_decision_id
        == "existing-decision"
    )


def test_checkpoint_resume_uses_validated_recovery_decision_and_supervisor_retry(
    coordinator: LaneCoordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    reservation = _reserve(coordinator, LaneId.CODING, intent="resume gateway task")
    coordinator.start(reservation)
    task_id = reservation.execution.task_id
    checkpoint_id = coordinator.checkpoint(
        task_id,
        boundary="before-retry",
        completed_steps=("inspect",),
        pending_steps=("patch",),
    )
    coordinator.finish(task_id, state=LaneTaskState.FAILED, error="worker interrupted")
    retry_calls: list[tuple[str, RecoveryDecision]] = []
    original_retry = coordinator.execution_supervisor.retry

    def record_retry(task_id: str, decision: RecoveryDecision):
        retry_calls.append((task_id, decision))
        return original_retry(task_id, decision)

    monkeypatch.setattr(coordinator.execution_supervisor, "retry", record_retry)
    decision = RecoveryDecision(
        decision_id="checkpoint-resume-decision",
        task_id=task_id,
        action=RecoveryAction.RESUME_CHECKPOINT,
        retry_category=RetryCategory.MODEL,
        reason="The validated checkpoint is safe to resume.",
        resume_checkpoint_id=checkpoint_id,
        safe_to_continue=True,
    )

    resumed = coordinator.resume_checkpoint(
        task_id,
        decision=decision,
        session_id="session-1",
    )

    assert resumed.execution.task_id == task_id
    assert coordinator.inspect_task(task_id).state is LaneTaskState.QUEUED
    assert retry_calls == [(task_id, decision)]


def test_same_task_retry_uses_validated_recovery_decision_and_supervisor_retry(
    coordinator: LaneCoordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    reservation = _reserve(coordinator, LaneId.CODING, intent="retry gateway task")
    coordinator.start(reservation)
    task_id = reservation.execution.task_id
    coordinator.finish(task_id, state=LaneTaskState.FAILED, error="model response failed")
    retry_calls: list[tuple[str, RecoveryDecision]] = []
    original_retry = coordinator.execution_supervisor.retry

    def record_retry(task_id: str, decision: RecoveryDecision):
        retry_calls.append((task_id, decision))
        return original_retry(task_id, decision)

    monkeypatch.setattr(coordinator.execution_supervisor, "retry", record_retry)
    decision = RecoveryDecision(
        decision_id="same-task-retry-decision",
        task_id=task_id,
        action=RecoveryAction.RETRY,
        retry_category=RetryCategory.MODEL,
        reason="The model authorized a safe same-task retry.",
        same_task_retry_authorized=True,
        safe_to_continue=True,
    )

    retried = coordinator.retry_task(
        task_id,
        decision=decision,
        session_id="session-1",
    )

    assert retried.execution.task_id == task_id
    assert coordinator.inspect_task(task_id).state is LaneTaskState.QUEUED
    assert retry_calls == [(task_id, decision)]


def test_explicit_taskboard_root_and_child_keep_their_persisted_lineage(
    coordinator: LaneCoordinator,
) -> None:
    board = coordinator.taskboard
    root_task = board.create_task(title="Compound", user_request="Research and create PDF")
    child_task = board.create_child_task(
        root_task.task_id,
        title="Research",
        user_request="Research Hermes Agent",
        decomposition_local_id="research_hermes_agent",
        acceptance_criteria=["Research is sourced"],
    )
    root = coordinator.reserve(
        normalized_intent="Research Hermes Agent and create a PDF",
        lane_id=LaneId.RESEARCH,
        session_id="session-1",
        workspace_id=board.store.workspace_id,
        repository_id=board.store.repository_id,
        requested_input_tokens=100,
        requested_output_tokens=2_000,
        task_type="multi_task_root",
        taskboard_task_id=root_task.task_id,
    )
    coordinator.start(root)

    child = coordinator.reserve(
        normalized_intent="Research Hermes Agent",
        lane_id=LaneId.RESEARCH,
        session_id="session-1",
        workspace_id=board.store.workspace_id,
        repository_id=board.store.repository_id,
        parent_task_id=root.execution.task_id,
        root_task_id=root.execution.root_task_id,
        requested_input_tokens=100,
        requested_output_tokens=200,
        task_type="multi_task_child",
        taskboard_task_id=child_task.task_id,
    )

    assert root.execution.taskboard_task_id == root_task.task_id
    assert child.execution.taskboard_task_id == child_task.task_id
    assert board.get_task(child_task.task_id).parent_task_id == root_task.task_id


def test_non_overlapping_file_locks_can_coexist(coordinator: LaneCoordinator) -> None:
    first = _reserve(coordinator, LaneId.CODING, intent="edit a", files=("a.py",))
    second = _reserve(coordinator, LaneId.CODING, intent="edit b", files=("b.py",), session="session-2")

    coordinator.start(first)
    coordinator.start(second)

    assert first.execution.state == LaneTaskState.RUNNING
    assert second.execution.state == LaneTaskState.RUNNING
    coordinator.finish(first.execution.task_id)
    coordinator.finish(second.execution.task_id)


def test_lane_capacity_waits_in_queue_until_capacity_is_released(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    waiting = threading.Event()

    def sink(event_type: str, title: str, **kwargs) -> None:
        _ = title
        if event_type == "lane.queued" and (kwargs.get("metadata") or {}).get("reason") == "capacity":
            waiting.set()

    coordinator = LaneCoordinator(
        root,
        contracts={"research": {"max_concurrent_jobs": 1, "timeout_seconds": 5}},
        event_sink=sink,
    )
    first = _reserve(coordinator, LaneId.RESEARCH, intent="first")
    coordinator.start(first)
    result: list[LaneReservation] = []
    worker = threading.Thread(
        target=lambda: result.append(
            _reserve(coordinator, LaneId.RESEARCH, intent="second", session="session-2")
        )
    )
    worker.start()
    assert waiting.wait(timeout=2)

    coordinator.finish(first.execution.task_id)
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert result and result[0].execution.state == LaneTaskState.QUEUED


def test_unleased_queued_record_does_not_consume_lane_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    coordinator = LaneCoordinator(
        root,
        contracts={"research": {"max_concurrent_jobs": 1}},
    )

    unleased = _reserve(coordinator, LaneId.RESEARCH, intent="interrupted before start")
    next_reservation = _reserve(
        coordinator,
        LaneId.RESEARCH,
        intent="fresh task",
        session="session-fresh",
    )

    assert unleased.execution.state is LaneTaskState.QUEUED
    assert next_reservation.execution.state is LaneTaskState.QUEUED
    assert not next_reservation.duplicate


def test_provider_limit_waits_until_model_capacity_is_released(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    waiting = threading.Event()
    coordinator = LaneCoordinator(
        root,
        provider_limits={"model-a": 1},
        event_sink=lambda event_type, title, **kwargs: waiting.set()
        if event_type == "lane.queued" and (kwargs.get("metadata") or {}).get("reason") == "capacity"
        else None,
    )
    common = {
        "workspace_id": coordinator.taskboard.store.workspace_id,
        "repository_id": coordinator.taskboard.store.repository_id,
        "requested_input_tokens": 10,
        "model": "model-a",
    }
    first = coordinator.reserve(
        normalized_intent="first model task", lane_id=LaneId.RESEARCH,
        session_id="s1", **common,
    )
    coordinator.start(first)
    result: list[LaneReservation] = []
    worker = threading.Thread(
        target=lambda: result.append(
            coordinator.reserve(
                normalized_intent="second model task", lane_id=LaneId.RESEARCH,
                session_id="s2", **common,
            )
        )
    )
    worker.start()
    assert waiting.wait(timeout=2)
    coordinator.finish(first.execution.task_id)
    worker.join(timeout=2)
    assert result and not worker.is_alive()


def test_interactive_waiter_runs_before_background_without_dropping_background(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    queued = threading.Condition()
    queued_ids: set[str] = set()

    def sink(event_type: str, title: str, **kwargs) -> None:
        _ = title
        if event_type == "lane.queued" and (kwargs.get("metadata") or {}).get("reason") == "capacity":
            with queued:
                queued_ids.add(str((kwargs.get("metadata") or {}).get("task_id")))
                queued.notify_all()

    coordinator = LaneCoordinator(
        root,
        contracts={"research": {"max_concurrent_jobs": 1, "timeout_seconds": 5}},
        event_sink=sink,
    )
    first = _reserve(coordinator, LaneId.RESEARCH, intent="occupy lane")
    coordinator.start(first)
    order: list[tuple[str, LaneReservation]] = []

    def worker(name: str, priority: LanePriority) -> None:
        reservation = coordinator.reserve(
            normalized_intent=name,
            lane_id=LaneId.RESEARCH,
            session_id=name,
            workspace_id=coordinator.taskboard.store.workspace_id,
            repository_id=coordinator.taskboard.store.repository_id,
            priority=priority,
            requested_input_tokens=10,
        )
        coordinator.start(reservation)
        order.append((name, reservation))

    background = threading.Thread(target=worker, args=("background", LanePriority.BACKGROUND))
    interactive = threading.Thread(target=worker, args=("interactive", LanePriority.INTERACTIVE))
    background.start()
    interactive.start()
    with queued:
        assert queued.wait_for(lambda: len(queued_ids) >= 2, timeout=2)

    coordinator.finish(first.execution.task_id)
    interactive.join(timeout=2)
    assert order and order[0][0] == "interactive"
    coordinator.finish(order[0][1].execution.task_id)
    background.join(timeout=2)
    assert [name for name, _ in order] == ["interactive", "background"]
    coordinator.finish(order[1][1].execution.task_id)


def test_overlapping_file_mutations_are_serialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    lock_waiting = threading.Event()
    coordinator = LaneCoordinator(
        root,
        event_sink=lambda event_type, title, **kwargs: lock_waiting.set()
        if event_type == "lock.waiting"
        else None,
    )
    first = _reserve(coordinator, LaneId.CODING, intent="first edit", files=("same.py",))
    second = _reserve(coordinator, LaneId.CODING, intent="second edit", files=("same.py",), session="s2")
    coordinator.start(first)
    worker = threading.Thread(target=lambda: coordinator.start(second))
    worker.start()
    assert lock_waiting.wait(timeout=2)

    coordinator.finish(first.execution.task_id)
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert second.execution.state == LaneTaskState.RUNNING
    coordinator.finish(second.execution.task_id)


def test_repository_write_lock_blocks_read_and_read_locks_coexist(coordinator: LaneCoordinator) -> None:
    repo = coordinator.taskboard.store.repository_id
    workspace = coordinator.taskboard.store.workspace_id
    first = coordinator.lock_manager.acquire(
        task_id="read-1", mode=LockMode.REPOSITORY_READ, workspace_id=workspace,
        repository_id=repo, paths=(), timeout_seconds=0, lease_seconds=60,
    )
    second = coordinator.lock_manager.acquire(
        task_id="read-2", mode=LockMode.REPOSITORY_READ, workspace_id=workspace,
        repository_id=repo, paths=(), timeout_seconds=0, lease_seconds=60,
    )
    assert first and second
    with pytest.raises(Exception, match="Timed out"):
        coordinator.lock_manager.acquire(
            task_id="write", mode=LockMode.REPOSITORY_WRITE, workspace_id=workspace,
            repository_id=repo, paths=(), timeout_seconds=0, lease_seconds=60,
        )
    coordinator.lock_manager.release_task("read-1")
    coordinator.lock_manager.release_task("read-2")


def test_locks_release_after_success_and_failure(coordinator: LaneCoordinator) -> None:
    success = _reserve(coordinator, LaneId.CODING, intent="success", files=("same.py",))
    coordinator.start(success)
    coordinator.finish(success.execution.task_id)

    failure = _reserve(coordinator, LaneId.CODING, intent="failure", files=("same.py",), session="session-2")
    coordinator.start(failure)
    coordinator.finish(failure.execution.task_id, state=LaneTaskState.FAILED, error="boom")

    assert not coordinator._locks


def test_completion_accepts_attempt_created_empty_files_and_directories(
    coordinator: LaneCoordinator,
) -> None:
    reservation = _reserve(coordinator, LaneId.CODING, intent="create package")
    coordinator.start(reservation)
    package = coordinator.root / "package"
    package.mkdir()
    (coordinator.root / "empty.py").write_text("", encoding="utf-8")

    finished = coordinator.finish(
        reservation.execution.task_id,
        changed_files=["empty.py", "package"],
        verification_state={"status": "passed"},
    )

    assert finished.state == LaneTaskState.COMPLETED
    manifest = coordinator.execution_supervisor.store.artifact_manifest(
        reservation.execution.task_id
    )
    assert manifest is not None
    assert [
        check["contract_type"]
        for check in manifest["verification"]["checks"]
    ] == ["file_exists", "directory_exists"]


def test_completion_failure_surfaces_the_persisted_contract_reason(
    coordinator: LaneCoordinator,
) -> None:
    pre_existing = coordinator.root / "pre-existing.txt"
    pre_existing.write_text("unchanged\n", encoding="utf-8")
    reservation = _reserve(coordinator, LaneId.CODING, intent="claim stale file")
    coordinator.start(reservation)

    finished = coordinator.finish(
        reservation.execution.task_id,
        changed_files=["pre-existing.txt"],
        verification_state={"status": "passed"},
    )

    assert finished.state == LaneTaskState.VERIFYING
    assert "artifact was not produced or modified by this attempt" in finished.error


def test_synchronize_usage_is_cumulative_and_does_not_double_count(
    coordinator: LaneCoordinator,
) -> None:
    reservation = _reserve(coordinator, LaneId.CODING, intent="account model usage")
    coordinator.start(reservation)

    coordinator.synchronize_usage(
        reservation.execution.task_id,
        consumed_input_tokens=20,
        consumed_output_tokens=10,
        actual_cost=0.03,
    )
    synchronized = coordinator.synchronize_usage(
        reservation.execution.task_id,
        consumed_input_tokens=20,
        consumed_output_tokens=10,
        actual_cost=0.03,
    )

    assert synchronized.budget.consumed_input_tokens == 20
    assert synchronized.budget.consumed_output_tokens == 10
    assert synchronized.budget.actual_cost == pytest.approx(0.03)

    coordinator.finish(
        reservation.execution.task_id,
        verification_state={"status": "accounted"},
    )
    supervised = coordinator.execution_supervisor.store.get_task(
        reservation.execution.task_id
    )
    assert supervised.token_usage == 30
    assert supervised.actual_cost == pytest.approx(0.03)


def test_stale_lock_recovery(coordinator: LaneCoordinator) -> None:
    lease = coordinator.lock_manager.acquire(
        task_id="stale", mode=LockMode.REPOSITORY_WRITE,
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
        paths=(), timeout_seconds=0, lease_seconds=60,
    )
    assert lease is not None
    lease.expires_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with coordinator._process_state_lock():
        coordinator._persist_locks_file_locked()

    coordinator.recover()

    assert lease.lease_id not in coordinator._locks


def test_lock_leases_are_shared_across_gateway_process_state(coordinator: LaneCoordinator) -> None:
    repo = coordinator.taskboard.store.repository_id
    workspace = coordinator.taskboard.store.workspace_id
    coordinator.lock_manager.acquire(
        task_id="writer-one", mode=LockMode.REPOSITORY_WRITE,
        workspace_id=workspace, repository_id=repo, paths=(),
        timeout_seconds=0, lease_seconds=60,
    )
    second_worker = LaneCoordinator(coordinator.root)

    with pytest.raises(Exception, match="Timed out"):
        second_worker.lock_manager.acquire(
            task_id="reader-two", mode=LockMode.REPOSITORY_READ,
            workspace_id=workspace, repository_id=repo, paths=(),
            timeout_seconds=0, lease_seconds=60,
        )

    coordinator.lock_manager.release_task("writer-one")


def test_token_and_cost_budget_exhaustion(coordinator: LaneCoordinator) -> None:
    coordinator.contracts = configured_lane_contracts({
        "coding": {"token_budget": 100, "cost_budget": 0.10},
    })
    coding = coordinator.contracts[LaneId.CODING]
    assert coding.token_budget is not None
    assert coding.cost_budget is not None
    with pytest.raises(LaneBudgetError):
        coordinator.reserve(
            normalized_intent="too many tokens", lane_id=LaneId.CODING, session_id="s",
            workspace_id=coordinator.taskboard.store.workspace_id,
            repository_id=coordinator.taskboard.store.repository_id,
            requested_input_tokens=coding.token_budget + 1,
        )
    with pytest.raises(LaneBudgetError):
        coordinator.reserve(
            normalized_intent="too much cost", lane_id=LaneId.CODING, session_id="s",
            workspace_id=coordinator.taskboard.store.workspace_id,
            repository_id=coordinator.taskboard.store.repository_id,
            estimated_cost=coding.cost_budget + 0.01,
        )


def test_canvas_task_does_not_wait_for_repository_lock(
    coordinator: LaneCoordinator,
) -> None:
    workspace_id = coordinator.taskboard.store.workspace_id
    repository_id = coordinator.taskboard.store.repository_id
    operations = coordinator.reserve(
        normalized_intent="run an infrastructure operation",
        lane_id=LaneId.OPERATIONS,
        session_id="session-operations",
        workspace_id=workspace_id,
        repository_id=repository_id,
        requested_input_tokens=2,
        requested_output_tokens=2,
        capabilities=("automation",),
    )
    coordinator.start(operations)
    reservation = coordinator.reserve(
        normalized_intent="create a live canvas",
        lane_id=LaneId.CANVAS,
        session_id="session-canvas",
        workspace_id=workspace_id,
        repository_id=repository_id,
        requested_input_tokens=2,
        requested_output_tokens=2,
        capabilities=("canvas",),
    )

    started = coordinator.start(reservation)

    assert started.state is LaneTaskState.RUNNING
    assert not [lock for lock in coordinator._locks.values() if lock.task_id == started.task_id]
    coordinator.finish(started.task_id, state=LaneTaskState.CANCELLED)
    coordinator.finish(operations.execution.task_id, state=LaneTaskState.CANCELLED)


def test_finish_preserves_a_pending_model_budget_overrun_decision(coordinator: LaneCoordinator) -> None:
    events: list[dict[str, object]] = []
    coordinator.event_sink = lambda event_type, title, **kwargs: events.append({
        "event_type": event_type,
        "title": title,
        **kwargs,
    })
    reservation = coordinator.reserve(
        normalized_intent="complete within the reserved budget",
        lane_id=LaneId.OPERATIONS,
        session_id="session-budget-exhaustion",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
        requested_input_tokens=2,
        requested_output_tokens=2,
    )
    coordinator.start(reservation)

    finished = coordinator.finish(
        reservation.execution.task_id,
        consumed_input_tokens=3,
        consumed_output_tokens=2,
        verification_state={"result": "present"},
    )

    assert finished.state is LaneTaskState.PENDING_BUDGET_DECISION
    assert "budget-overrun decision" in finished.error
    supervised = coordinator.execution_supervisor.store.get_task(finished.task_id)
    assert supervised.state is SupervisorState.PENDING_BUDGET_DECISION
    assert events[-2]["event_type"] == "budget.overrun.decision.required"
    assert events[-2]["status"] == "waiting"


def test_accepted_budget_overrun_projects_authoritative_supervisor_completion(
    coordinator: LaneCoordinator,
) -> None:
    reservation = coordinator.reserve(
        normalized_intent="complete with a model-authorized budget overrun",
        lane_id=LaneId.OPERATIONS,
        session_id="session-budget-finalization",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
        requested_input_tokens=2,
        requested_output_tokens=2,
    )
    coordinator.start(reservation)
    pending = coordinator.finish(
        reservation.execution.task_id,
        consumed_input_tokens=3,
        consumed_output_tokens=2,
        verification_state={"result": "present"},
    )
    supervised = coordinator.execution_supervisor.verify_completion(pending.task_id)

    finalized = coordinator.finalize_budget_overrun(BudgetOverrunFinalizationDecision(
        decision_id="decision_accept_budget_overrun",
        task_id=pending.task_id,
        attempt_id=supervised.attempt_id,
        result_id=supervised.result_id,
        result_evidence_hash=supervised.budget_overrun["evidence_hash"],
        action=BudgetOverrunAction.ACCEPT_WITH_OVERRUN,
        reason="verified result is authorized despite immutable-cap overrun",
        safe_to_continue=True,
    ))

    assert finalized.state is LaneTaskState.COMPLETED
    taskboard_task = coordinator.taskboard.get_task(finalized.taskboard_task_id)
    assert taskboard_task.status is TaskStatus.DONE
    assert taskboard_task.supervisor_execution_id == pending.task_id
    assert taskboard_task.verification_status == "passed"


def test_reconciles_completed_overrun_when_the_prior_taskboard_projection_failed(
    coordinator: LaneCoordinator,
) -> None:
    reservation = coordinator.reserve(
        normalized_intent="repair a completed model-authorized budget overrun",
        lane_id=LaneId.OPERATIONS,
        session_id="session-budget-projection-repair",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
        requested_input_tokens=2,
        requested_output_tokens=2,
    )
    coordinator.start(reservation)
    pending = coordinator.finish(
        reservation.execution.task_id,
        consumed_input_tokens=3,
        consumed_output_tokens=2,
        verification_state={"result": "present"},
    )
    supervised = coordinator.execution_supervisor.verify_completion(pending.task_id)
    coordinator.execution_supervisor.finalize_budget_overrun(BudgetOverrunFinalizationDecision(
        decision_id="decision_repair_budget_overrun_projection",
        task_id=pending.task_id,
        attempt_id=supervised.attempt_id,
        result_id=supervised.result_id,
        result_evidence_hash=supervised.budget_overrun["evidence_hash"],
        action=BudgetOverrunAction.ACCEPT_WITH_OVERRUN,
        reason="verified result was already accepted before projection retry",
        safe_to_continue=True,
    ))

    reconciled = coordinator.reconcile_authoritative_completion(pending.task_id)

    assert reconciled.state is LaneTaskState.COMPLETED
    taskboard_task = coordinator.taskboard.get_task(reconciled.taskboard_task_id)
    assert taskboard_task.status is TaskStatus.DONE
    assert taskboard_task.supervisor_execution_id == pending.task_id


def test_recalculate_budget_expands_a_live_reservation_within_lane_policy(
    coordinator: LaneCoordinator,
) -> None:
    coordinator.contracts = configured_lane_contracts({"operations": {"token_budget": 1000}})
    reservation = coordinator.reserve(
        normalized_intent="forecasted provider call", lane_id=LaneId.OPERATIONS,
        session_id="session-recalculate",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
        requested_input_tokens=10, requested_output_tokens=10,
    )
    coordinator.start(reservation)

    revised = coordinator.recalculate_budget(
        reservation.execution.task_id,
        forecast_input_tokens=50,
        forecast_output_tokens=75,
        forecast_cost=0.01,
        accounting_reservation_id="reservation_forecast",
    )

    assert revised.budget.reserved_tokens == 125
    assert revised.budget.revisions[-1]["accounting_reservation_id"] == "reservation_forecast"


def test_recalculate_budget_expands_parent_envelope_for_child_growth(
    coordinator: LaneCoordinator,
) -> None:
    """Child forecast growth must add into the active parent reservation."""
    parent = coordinator.reserve(
        normalized_intent="parent envelope",
        lane_id=LaneId.RESEARCH,
        session_id="session-parent-envelope",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
        requested_input_tokens=40,
        requested_output_tokens=40,
    )
    coordinator.start(parent)
    child = coordinator.reserve(
        normalized_intent="child coding",
        lane_id=LaneId.CODING,
        session_id=parent.execution.session_id,
        workspace_id=parent.execution.workspace_id,
        repository_id=parent.execution.repository_id,
        parent_task_id=parent.execution.task_id,
        root_task_id=parent.execution.root_task_id,
        requested_input_tokens=30,
        requested_output_tokens=30,
        task_type="single",
    )
    coordinator.start(child)

    revised_child = coordinator.recalculate_budget(
        child.execution.task_id,
        forecast_input_tokens=500,
        forecast_output_tokens=400,
        forecast_cost=0.02,
        reason="provider-call forecast",
    )
    revised_parent = coordinator.inspect_task(parent.execution.task_id)

    assert revised_child.budget.reserved_tokens >= 900
    assert revised_parent.budget.reserved_tokens >= revised_child.budget.reserved_tokens
    assert any(
        item.get("reason") == "parent envelope for child recalculation"
        for item in revised_parent.budget.revisions
    )


def test_reserve_expands_parent_when_child_needs_more_than_remaining(
    coordinator: LaneCoordinator,
) -> None:
    """Child reservation grows the parent envelope instead of hard-failing."""
    parent = coordinator.reserve(
        normalized_intent="small parent",
        lane_id=LaneId.RESEARCH,
        session_id="session-reserve-envelope",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
        requested_input_tokens=20,
        requested_output_tokens=20,
    )
    coordinator.start(parent)

    child = coordinator.reserve(
        normalized_intent="large child",
        lane_id=LaneId.CODING,
        session_id=parent.execution.session_id,
        workspace_id=parent.execution.workspace_id,
        repository_id=parent.execution.repository_id,
        parent_task_id=parent.execution.task_id,
        root_task_id=parent.execution.root_task_id,
        requested_input_tokens=200,
        requested_output_tokens=200,
        task_type="single",
    )

    revised_parent = coordinator.inspect_task(parent.execution.task_id)
    assert child.execution.budget.reserved_tokens == 400
    assert revised_parent.budget.reserved_tokens >= 400
    assert any(
        item.get("reason") == "parent envelope for child reservation"
        for item in revised_parent.budget.revisions
    )


def test_recalculate_budget_under_terminal_parent_does_not_block(
    coordinator: LaneCoordinator,
) -> None:
    """Follow-ups under a failed parent must not fail parent-remaining checks."""
    parent = coordinator.reserve(
        normalized_intent="failed parent",
        lane_id=LaneId.RESEARCH,
        session_id="session-terminal-parent",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
        requested_input_tokens=30,
        requested_output_tokens=30,
    )
    coordinator.start(parent)
    coordinator.finish(
        parent.execution.task_id,
        state=LaneTaskState.FAILED,
        error="prior multi-task child failed",
    )
    # Terminal parents skip remaining checks at reserve time.
    child = coordinator.reserve(
        normalized_intent="follow-up under failed parent",
        lane_id=LaneId.CODING,
        session_id=parent.execution.session_id,
        workspace_id=parent.execution.workspace_id,
        repository_id=parent.execution.repository_id,
        parent_task_id=parent.execution.task_id,
        root_task_id=parent.execution.root_task_id,
        requested_input_tokens=100,
        requested_output_tokens=100,
        task_type="single",
    )
    coordinator.start(child)

    revised = coordinator.recalculate_budget(
        child.execution.task_id,
        forecast_input_tokens=800,
        forecast_output_tokens=600,
        forecast_cost=None,
        reason="provider-call forecast",
    )

    assert revised.budget.reserved_tokens >= 1400


def test_recalculate_budget_expands_nested_ancestors(
    coordinator: LaneCoordinator,
) -> None:
    """Grandchild growth expands parent and multi-task-style root ancestors."""
    root = coordinator.reserve(
        normalized_intent="compound root",
        lane_id=LaneId.RESEARCH,
        session_id="session-nested-ancestors",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
        requested_input_tokens=50,
        requested_output_tokens=50,
        task_type="multi_task_root",
    )
    coordinator.start(root)
    mid = coordinator.reserve(
        normalized_intent="mid child",
        lane_id=LaneId.CODING,
        session_id=root.execution.session_id,
        workspace_id=root.execution.workspace_id,
        repository_id=root.execution.repository_id,
        parent_task_id=root.execution.task_id,
        root_task_id=root.execution.root_task_id,
        requested_input_tokens=40,
        requested_output_tokens=40,
        task_type="multi_task_child",
    )
    coordinator.start(mid)
    leaf = coordinator.reserve(
        normalized_intent="nested leaf",
        lane_id=LaneId.CODING,
        session_id=root.execution.session_id,
        workspace_id=root.execution.workspace_id,
        repository_id=root.execution.repository_id,
        parent_task_id=mid.execution.task_id,
        root_task_id=root.execution.root_task_id,
        requested_input_tokens=30,
        requested_output_tokens=30,
        task_type="single",
    )
    coordinator.start(leaf)

    revised_leaf = coordinator.recalculate_budget(
        leaf.execution.task_id,
        forecast_input_tokens=700,
        forecast_output_tokens=500,
        forecast_cost=0.03,
        reason="provider-call forecast",
    )
    revised_mid = coordinator.inspect_task(mid.execution.task_id)
    revised_root = coordinator.inspect_task(root.execution.task_id)

    assert revised_leaf.budget.reserved_tokens >= 1200
    assert revised_mid.budget.reserved_tokens >= revised_leaf.budget.reserved_tokens
    assert revised_root.budget.reserved_tokens >= revised_mid.budget.reserved_tokens


def test_child_agent_reserves_and_consumes_parent_budget(coordinator: LaneCoordinator) -> None:
    parent = _reserve(coordinator, LaneId.RESEARCH, intent="parent research")
    child = coordinator.reserve(
        normalized_intent="child implementation", lane_id=LaneId.CODING,
        session_id=parent.execution.session_id,
        workspace_id=parent.execution.workspace_id,
        repository_id=parent.execution.repository_id,
        parent_task_id=parent.execution.task_id,
        target_files=("child.py",),
        requested_input_tokens=50,
        requested_output_tokens=50,
    )
    coordinator.start(child)
    coordinator.finish(
        child.execution.task_id,
        consumed_input_tokens=20,
        consumed_output_tokens=10,
    )

    assert parent.execution.budget.consumed_tokens == 30
    task = coordinator.taskboard.get_task(child.execution.taskboard_task_id)
    assert task.parent_task_id == parent.execution.taskboard_task_id


def test_forbidden_subagent_creation(coordinator: LaneCoordinator) -> None:
    review = _reserve(coordinator, LaneId.REVIEW, intent="review")
    with pytest.raises(LaneCoordinatorError, match="cannot create subagents"):
        coordinator.can_create_subagent(review.execution.task_id, child_lane=LaneId.CODING)


def test_handoff_preserves_task_and_scope_identity(coordinator: LaneCoordinator) -> None:
    coding = _reserve(coordinator, LaneId.CODING, intent="implement", files=("a.py",))
    coordinator.start(coding)
    before = coding.execution
    handoff = LaneHandoff(
        source_lane=LaneId.CODING, target_lane=LaneId.VERIFY, task_id=before.task_id,
        reason="implementation ready", changed_files=["a.py"], remaining_work=["run tests"],
        verification_state={"status": "pending"}, budget_consumed=LaneBudget(consumed_input_tokens=20),
    )

    after = coordinator.handoff(handoff)

    assert after.task_id == before.task_id
    assert after.session_id == before.session_id
    assert after.workspace_id == before.workspace_id
    assert after.repository_id == before.repository_id
    assert after.owning_lane == LaneId.VERIFY
    assert len(after.lane_history) == 4
    coordinator.finish(after.task_id)


def test_invalid_handoff_stops_without_transition(coordinator: LaneCoordinator) -> None:
    research = _reserve(coordinator, LaneId.RESEARCH, intent="research")
    with pytest.raises(LaneHandoffError):
        coordinator.handoff(
            LaneHandoff(
                source_lane=LaneId.RESEARCH, target_lane=LaneId.RELEASE,
                task_id=research.execution.task_id, reason="invalid",
            )
        )
    assert research.execution.owning_lane == LaneId.RESEARCH


def test_restart_restores_execution_without_creating_new_identity(coordinator: LaneCoordinator) -> None:
    reservation = _reserve(coordinator, LaneId.RESEARCH, intent="persistent")
    coordinator.start(reservation)

    restarted = LaneCoordinator(coordinator.root)

    restored = {item.task_id: item for item in restarted.executions}[reservation.execution.task_id]
    assert restored.session_id == reservation.execution.session_id
    assert restored.repository_id == reservation.execution.repository_id
    assert len(restarted.taskboard.tasks) == 1
    coordinator.finish(reservation.execution.task_id)


def test_state_persistence_retries_transient_windows_replace_denial(
    coordinator: LaneCoordinator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_replace = os.replace
    taskboard_state_path = coordinator.taskboard.store.state_path
    tracked_paths = {taskboard_state_path, coordinator.state_path}
    replace_calls = {path: 0 for path in tracked_paths}
    denied_paths: set[Path] = set()

    def transiently_denied(source: str | Path, destination: str | Path) -> None:
        destination_path = Path(destination)
        if destination_path in replace_calls:
            replace_calls[destination_path] += 1
        if destination_path in replace_calls and replace_calls[destination_path] == 1:
            denied_paths.add(destination_path)
            raise PermissionError(13, "Access is denied")
        real_replace(source, destination)

    monkeypatch.setattr("mana_agent.gateway.lane_coordinator.os.replace", transiently_denied)

    reservation = _reserve(coordinator, LaneId.RESEARCH, intent="retry persistence")

    assert denied_paths == tracked_paths
    assert all(replace_calls[path] >= 2 for path in tracked_paths)
    assert taskboard_state_path.is_file()
    assert coordinator.state_path.is_file()
    assert not list(coordinator.state_path.parent.glob(f".{coordinator.state_path.name}.*.tmp"))
    coordinator.start(reservation)
    coordinator.finish(reservation.execution.task_id)
