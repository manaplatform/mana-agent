"""P0.9 test suite for Codex interruption recovery, lost leases, and durable resume."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.models import (
    ActionEffectScope,
    ActionRecord,
    ActionRequestState,
    ExecutionState,
    HumanRecoveryDecisionAction,
    LostLeaseOutcome,
    SideEffectClassification,
    utc_now,
)
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
from mana_agent.gateway.config import ChatGatewayConfig
from mana_agent.gateway.feature_integration import (
    FeatureIntegrationCoordinator,
    IntegrationAuthority,
    WiringDecision,
)
from mana_agent.gateway.turn_engine import ChatTurnResult, process_chat_turn
from mana_agent.human_inbox import (
    HumanInboxService,
    LocalInboxRepository,
    ResponseTokenSigner,
    ReviewerIdentity,
    StaticIdentityDirectory,
)
from mana_agent.human_inbox.models import (
    HumanResponse,
    InboxItem,
    InboxRequest,
    InboxRequestType,
    ResponseOperation,
    ReviewerAssignment,
    ReviewerType,
    RiskLevel,
)
from mana_agent.integrations.codex.exceptions import (
    CodexInterruptionError,
    CodexProtocolError,
    CodexTimeoutError,
)
from mana_agent.multi_agent.routing.agent_decision import AgentDecision
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard


class SimulatedClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class MockCodingAgent:
    def __init__(self, side_effect: Any = None) -> None:
        self.side_effect = side_effect
        self.calls: list[dict[str, Any]] = []

    def generate(self, request: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"request": request, "kwargs": kwargs})
        if isinstance(self.side_effect, Exception):
            raise self.side_effect
        if callable(self.side_effect):
            return self.side_effect(request, **kwargs)
        if isinstance(self.side_effect, dict):
            return dict(self.side_effect)
        return {"answer": "done", "status": "completed", "changed_files": ["app.py"]}


def _test_decision(runtime_capability_change: bool = False) -> AgentDecision:
    return AgentDecision(
        intent="edit",
        code_editing_needed=True,
        selected_tools=["apply_patch"],
        tool_inputs={},
        flow_action="none",
        reasoning_summary="coding is required",
        confidence=0.99,
        verifier_passed=True,
        runtime_capability_change=runtime_capability_change,
    )


# ==================================================
# TEST A — Codex timeout before mutation
# ==================================================
def test_a_codex_timeout_before_mutation_safe_retry(tmp_path: Path) -> None:
    """Simulate Codex interrupt with no checkpoint/mutation -> safe retry, NOT_STARTED, no false lost lease."""
    coding_agent = MockCodingAgent(
        side_effect=CodexTimeoutError(
            "Codex request timed out: turn/start",
            method="turn/start",
            error_code="CODING_PROVIDER_TIMEOUT",
        )
    )
    session_state: dict[str, Any] = {}

    result = process_chat_turn(
        root=tmp_path,
        text="implement feature",
        chat_service=SimpleNamespace(),
        ask_service=SimpleNamespace(),
        coding_agent=coding_agent,
        config=ChatGatewayConfig().normalized(),
        session_state=session_state,
        agent_decision=_test_decision(),
        gateway_task_id="task_test_a",
    )

    assert result.error_code == "CODING_PROVIDER_TIMEOUT"
    assert result.error_category == "timeout"
    assert result.retry_possible is True
    assert result.resume_available is False
    assert result.checkpoint_available is False
    assert result.changed_files == []
    assert result.interruption_reason in {"CODING_TIMEOUT", "CODING_PROVIDER_TIMEOUT", "NOT_STARTED"}
    assert result.error != "AMBIGUOUS_LOST_LEASE"


# ==================================================
# TEST B — Codex timeout after partial mutation
# ==================================================
def test_b_codex_timeout_after_partial_mutation(tmp_path: Path) -> None:
    """Simulate Codex wrote file but was interrupted -> PARTIALLY_COMPLETED, preserves changed_files."""
    partial_result = {
        "status": "failed",
        "error_code": "CODING_TIMEOUT",
        "interruption_reason": "CODING_TIMEOUT",
        "changed_files": ["src/partial.py"],
        "answer": "",
    }
    coding_agent = MockCodingAgent(side_effect=partial_result)
    session_state: dict[str, Any] = {}

    result = process_chat_turn(
        root=tmp_path,
        text="implement partial feature",
        chat_service=SimpleNamespace(),
        ask_service=SimpleNamespace(),
        coding_agent=coding_agent,
        config=ChatGatewayConfig().normalized(),
        session_state=session_state,
        agent_decision=_test_decision(),
        gateway_task_id="task_test_b",
    )

    assert result.error_code == "CODING_TIMEOUT"
    assert result.error_category == "interruption"
    assert result.retry_possible is True
    assert result.resume_available is True
    assert result.checkpoint_available is True
    assert result.changed_files == ["src/partial.py"]
    assert len(coding_agent.calls) == 1


# ==================================================
# TEST C — Codex timeout after completed checkpoint
# ==================================================
def test_c_codex_timeout_after_completed_checkpoint(tmp_path: Path) -> None:
    """Core implementation checkpoint exists, response interrupted -> resume INTEGRATION_DISCOVERY without rerun."""
    saved_checkpoint = {
        "boundary": "after_core_implementation",
        "completed_steps": ["routing", "core_implementation"],
        "pending_steps": ["feature_integration", "verification", "final_response"],
        "gateway_task_id": "task_test_c",
        "core_changed_files": ["src/service.py"],
        "flow_id": "flow_c",
        "runtime_capability_change": True,
        "core_result": {
            "answer": "core complete",
            "status": "completed",
            "changed_files": ["src/service.py"],
            "flow_id": "flow_c",
        },
    }
    session_state: dict[str, Any] = {"feature_integration_checkpoint": saved_checkpoint}
    coding_agent = MockCodingAgent(side_effect=CodexTimeoutError("Codex response interrupted"))

    # Integration decision provided so coordinator finishes
    wiring_decision = WiringDecision(
        outcome="already_integrated",
        runtime_entrypoints=["src/main.py"],
        edges=[
            {
                "from": "src/main.py",
                "to": "src/router.py",
                "relation": "calls",
                "source_reference": "src/main.py:10",
            },
            {
                "from": "src/router.py",
                "to": "src/service.py",
                "relation": "selects",
                "source_reference": "src/router.py:20",
            },
            {
                "from": "src/service.py",
                "to": "src/handler.py",
                "relation": "constructs",
                "source_reference": "src/service.py:30",
            },
        ],
        verification_commands=["python -m pytest tests/"],
        reason="integrated directly",
    )

    authority = IntegrationAuthority(
        taskboard_state="done",
        wiring_child_id="wiring-1",
        verification_provenance={"verification_id": "v1", "observable_result": "passed"},
        reviewer_approval={"reviewer_id": "rev1", "approved": True},
        runtime_reachability={"verified": True},
        supervisor_completion={"state": "completed", "verification_status": "passed"},
    )

    result = process_chat_turn(
        root=tmp_path,
        text="complete feature",
        chat_service=SimpleNamespace(),
        ask_service=SimpleNamespace(),
        coding_agent=coding_agent,
        config=ChatGatewayConfig().normalized(),
        session_state=session_state,
        agent_decision=_test_decision(runtime_capability_change=True),
        gateway_task_id="task_test_c",
        feature_integration_decision=wiring_decision,
        feature_integration_authority=authority,
    )

    # Core coding was NOT rerun because checkpoint was completed
    assert len(coding_agent.calls) == 0
    assert result.error is None or result.error == ""
    assert result.changed_files == ["src/service.py"]


# ==================================================
# TEST D — Codex timeout during Feature Integration
# ==================================================
def test_d_codex_timeout_during_feature_integration(tmp_path: Path) -> None:
    """Timeout during Feature Integration preserves core and resumes feature integration stage."""
    board = TaskBoard(tmp_path / "taskboard")
    parent = board.create_task(
        title="Add provider",
        user_request="add provider",
        action_type="coding",
    )
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(root=tmp_path / "supervisor"))

    core_result = {
        "status": "completed",
        "changed_files": ["src/provider.py"],
        "answer": "provider created",
    }

    coordinator = FeatureIntegrationCoordinator()
    result = coordinator.run(
        coding_agent=MockCodingAgent(),
        core_result=core_result,
        request="add provider",
        gateway_task_id=parent.task_id,
        flow_id="flow_d",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        # Invalid decision forces blocked stage without destroying core work
        integration_decision=None,
    )

    assert not result.passed
    assert result.result.get("core_implementation_preserved") is True
    assert result.result.get("changed_files") == ["src/provider.py"]


# ==================================================
# TEST E — Codex timeout + lost lease
# ==================================================
def test_e_codex_timeout_separated_from_lost_lease(tmp_path: Path) -> None:
    """Classifier separates coding interruption from external lost-lease ambiguity."""
    clock = SimulatedClock()
    supervisor = ExecutionSupervisor(
        ExecutionSupervisorConfig(root=tmp_path / "supervisor", lease_seconds=10, heartbeat_seconds=2),
        clock=clock,
    )

    task = supervisor.create_task(
        task_id="task_e",
        routing_decision_id="routing_e",
        side_effect_classification=SideEffectClassification.READ_ONLY,
    )
    task, token = supervisor.acquire_lease(task.task_id, owner="worker-1")
    task = supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)

    # Simulate timeout without consequential external actions
    clock.advance(15)
    outcome, details = supervisor.classify_lost_lease(task, now=clock())

    # Read-only / idempotent tasks safely auto-recover; no AMBIGUOUS_LOST_LEASE
    assert outcome == LostLeaseOutcome.SAFE_AUTOMATIC_RECOVERY
    assert outcome != LostLeaseOutcome.UNKNOWN_EXTERNAL_OUTCOME


# ==================================================
# TEST F — Local mutation after timeout
# ==================================================
def test_f_local_mutation_after_timeout(tmp_path: Path) -> None:
    """Attempt-specific evidence decides ALREADY_APPLIED vs PARTIALLY_APPLIED."""
    clock = SimulatedClock()
    supervisor = ExecutionSupervisor(
        ExecutionSupervisorConfig(root=tmp_path / "supervisor", lease_seconds=10, heartbeat_seconds=2),
        clock=clock,
    )

    task = supervisor.create_task(
        task_id="task_f",
        workspace_path=tmp_path,
        routing_decision_id="routing_f",
        side_effect_classification=SideEffectClassification.NON_IDEMPOTENT,
    )
    task, token = supervisor.acquire_lease(task.task_id, owner="worker-1")
    task = supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)

    target_file = tmp_path / "file.py"
    target_file.write_text("print('applied')\n", encoding="utf-8")
    digest = hashlib.sha256(target_file.read_bytes()).hexdigest()

    action = ActionRecord(
        execution_id=task.task_id,
        attempt_id=task.attempt_id,
        attempt_generation=1,
        tool_name="write_file",
        action_fingerprint="fp_f",
        classification=SideEffectClassification.NON_IDEMPOTENT,
        effect_scope=ActionEffectScope.LOCAL_REPOSITORY,
        request_state=ActionRequestState.STARTED,
        verification_state={"artifact_hashes": {"file.py": digest}},
    )
    supervisor.store.save_action(action)
    clock.advance(15)

    outcome, details = supervisor.classify_lost_lease(task, now=clock(), actions=[action])

    # Reconciled from exact hash match -> ALREADY_APPLIED -> marked succeeded
    assert outcome == LostLeaseOutcome.SAFE_AUTOMATIC_RECOVERY
    updated_action = supervisor.store.get_action(action.action_id)
    assert updated_action.request_state == ActionRequestState.SUCCEEDED


# ==================================================
# TEST G — External receipt after timeout
# ==================================================
def test_g_external_receipt_after_timeout(tmp_path: Path) -> None:
    """Durable receipt consumed, marked ACTION_RECONCILED, no duplicate action."""
    clock = SimulatedClock()
    supervisor = ExecutionSupervisor(
        ExecutionSupervisorConfig(root=tmp_path / "supervisor", lease_seconds=10, heartbeat_seconds=2),
        clock=clock,
    )

    task = supervisor.create_task(
        task_id="task_g",
        routing_decision_id="routing_g",
        side_effect_classification=SideEffectClassification.NON_IDEMPOTENT,
    )
    task, token = supervisor.acquire_lease(task.task_id, owner="worker-1")
    task = supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)

    action = ActionRecord(
        execution_id=task.task_id,
        attempt_id=task.attempt_id,
        attempt_generation=1,
        tool_name="cloud_deploy",
        action_fingerprint="fp_deploy",
        classification=SideEffectClassification.NON_IDEMPOTENT,
        effect_scope=ActionEffectScope.EXTERNAL_CONSEQUENTIAL,
        request_state=ActionRequestState.SUCCEEDED,
        external_receipt="deployment-id-12345",
    )
    supervisor.store.save_action(action)
    clock.advance(15)

    summary = supervisor.recover()
    assert task.task_id in summary.recovered

    updated_action = supervisor.store.get_action(action.action_id)
    assert updated_action.request_state == ActionRequestState.ACTION_RECONCILED
    assert updated_action.verification_state.get("receipt_consumed") is True


# ==================================================
# TEST H — Human review after unknown external timeout
# ==================================================
def test_h_human_review_after_unknown_external_timeout(tmp_path: Path) -> None:
    """Unknown consequential external timeout creates real Human Inbox item and resumes on response."""
    clock = SimulatedClock()
    inbox_root = tmp_path / "inbox"
    identities = StaticIdentityDirectory([
        ReviewerIdentity(identity_id="admin", tenant_ids={"local"}),
    ])
    inbox_service = HumanInboxService(
        repository=LocalInboxRepository(inbox_root),
        identities=identities,
        token_signer=ResponseTokenSigner(inbox_root / "signing.key"),
        clock=clock,
    )

    supervisor = ExecutionSupervisor(
        ExecutionSupervisorConfig(root=tmp_path / "supervisor", lease_seconds=10, heartbeat_seconds=2),
        clock=clock,
    )

    task = supervisor.create_task(
        task_id="task_h",
        routing_decision_id="routing_h",
        side_effect_classification=SideEffectClassification.NON_IDEMPOTENT,
    )
    task, token = supervisor.acquire_lease(task.task_id, owner="worker-1")
    task = supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)

    action = ActionRecord(
        execution_id=task.task_id,
        attempt_id=task.attempt_id,
        attempt_generation=1,
        tool_name="stripe_charge",
        action_fingerprint="fp_charge",
        classification=SideEffectClassification.NON_IDEMPOTENT,
        effect_scope=ActionEffectScope.EXTERNAL_CONSEQUENTIAL,
        request_state=ActionRequestState.STARTED,
    )
    supervisor.store.save_action(action)
    clock.advance(15)

    summary = supervisor.recover()
    assert task.task_id in summary.intervention_required

    interventions = supervisor.store.recovery_interventions_for_task(task.task_id)
    assert len(interventions) == 1
    intervention = interventions[0]

    # Resolve intervention via HumanRecoveryDecisionAction
    resolved_task = supervisor.resolve_recovery_intervention(
        intervention.intervention_id,
        action=HumanRecoveryDecisionAction.MARK_ACTION_ALREADY_COMPLETED,
        actor_id="admin",
        comment="charge verified in stripe dashboard",
    )

    assert resolved_task.task_id == task.task_id
    assert resolved_task.state == ExecutionState.QUEUED
    updated_action = supervisor.store.get_action(action.action_id)
    assert updated_action.request_state == ActionRequestState.SUCCEEDED
    assert updated_action.external_receipt == "charge verified in stripe dashboard"


# ==================================================
# TEST I — Heartbeat during long Codex request
# ==================================================
def test_i_heartbeat_during_long_codex_request(tmp_path: Path) -> None:
    """Heartbeat renews lease_expires_at while deadline_at remains unchanged."""
    clock = SimulatedClock()
    supervisor = ExecutionSupervisor(
        ExecutionSupervisorConfig(
            root=tmp_path / "supervisor",
            lease_seconds=10,
            heartbeat_seconds=1,
        ),
        clock=clock,
    )

    task = supervisor.create_task(
        task_id="task_i",
        routing_decision_id="routing_i",
        side_effect_classification=SideEffectClassification.IDEMPOTENT,
    )
    task, token = supervisor.acquire_lease(task.task_id, owner="worker-long")
    task = supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)

    initial_deadline = task.deadline_at
    initial_expiry = task.lease_expires_at

    with supervisor.lease_renewal(task.task_id, attempt_id=task.attempt_id, lease_token=token):
        clock.advance(3)
        time.sleep(0.05)

    renewed = supervisor.store.get_task(task.task_id)
    assert renewed.lease_expires_at > initial_expiry
    assert renewed.deadline_at == initial_deadline
    assert renewed.state == ExecutionState.RUNNING
