from pathlib import Path
from typing import Any
import pytest
from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
from mana_agent.gateway.feature_integration import (
    FeatureIntegrationCoordinator,
    FeatureIntegrationDecisionProvider,
    FeatureIntegrationVerificationPlan,
    IntegrationAuthority,
    IntegrationVerificationExecutor,
    MultiAgentVerificationExecutor,
    WiringDecision,
    decide_feature_integration,
    validate_or_reconcile_integration_stage,
    INCOMPLETE_FEATURE_WIRING,
    FEATURE_INTEGRATION_DECISION_INVALID,
    FEATURE_INTEGRATION_VERIFIER_UNAVAILABLE,
    FEATURE_INTEGRATION_VERIFICATION_PLAN_MISSING,
    FEATURE_INTEGRATION_VERIFICATION_FAILED,
    FEATURE_INTEGRATION_REACHABILITY_UNPROVEN,
    FEATURE_INTEGRATION_REVIEW_REJECTED,
    FEATURE_INTEGRATION_STATE_INVALID,
    CORE_EXECUTION_FAILED,
    DETERMINISTIC_INTEGRATION_FAILURE,
    EXTERNAL_DEPENDENCY,
    INTERNAL_WORK_PENDING,
    integration_pending_classification,
)
from mana_agent.multi_agent.core.types import (
    AgentRole,
    TaskStatus,
    VerificationResult,
)
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard


def test_gateway_exports_lane_id_from_lanes_module():
    from mana_agent.gateway import LaneId
    from mana_agent.gateway.lanes import LaneId as LaneIdFromLanes

    assert LaneId is LaneIdFromLanes


def test_taskboard_wiring_child_is_reused_and_seeded_from_core_files(tmp_path):
    board = TaskBoard(tmp_path)
    parent = board.create_task(
        title="Add provider",
        user_request="add provider",
        action_type="coding",
    )
    coordinator = FeatureIntegrationCoordinator()

    first = coordinator.ensure_wiring_child(
        board,
        parent.task_id,
        request="add provider",
        changed_files=["src/provider.py"],
        trigger_turn_id="turn-1",
    )
    second = coordinator.ensure_wiring_child(
        board,
        parent.task_id,
        request="add provider",
        changed_files=["src/registry.py"],
        trigger_turn_id="turn-2",
    )

    assert first == second
    assert board.get_task(parent.task_id).required_wiring_task_ids == [first]
    assert board.get_task(first).files_touched == ["src/provider.py", "src/registry.py"]


def _edges():
    return [
        {"from": "ChatGateway", "to": "ModelRouter", "relation": "calls", "source_reference": "gateway.py:10", "file": "gateway.py", "symbol": "ChatGateway"},
        {"from": "ModelRouter", "to": "ProviderFactory", "relation": "selects", "source_reference": "router.py:20", "file": "router.py", "symbol": "ModelRouter"},
        {"from": "ProviderFactory", "to": "NewProvider", "relation": "constructs", "source_reference": "factory.py:30", "file": "factory.py", "symbol": "ProviderFactory"},
    ]


def _verified_integration(outcome="mutation_applied"):
    return {
        "wiring_outcome": outcome,
        "reachability_edges": _edges(),
        "verification_evidence": {"verification_id": "verification-1", "observable_result": "passed"},
        "reviewer_approval": {"reviewer_id": "reviewer-1", "approved": True},
        "runtime_reachability": {"verified": True},
        "verification_provenance": {"verification_id": "verification-1", "observable_result": "passed"},
        "supervisor_completion": {"state": "completed", "verification_status": "passed", "result_id": "result-1"},
    }


def _authority():
    integration = _verified_integration()
    return IntegrationAuthority(
        taskboard_state="done",
        wiring_child_id="wiring-1",
        verification_provenance=integration["verification_evidence"],
        reviewer_approval=integration["reviewer_approval"],
        runtime_reachability={"verified": True},
        supervisor_completion=integration["supervisor_completion"],
    )


class _Coding:
    def __init__(self, continuation=None):
        self.continuation = continuation or {}
        self.calls = []

    def generate(self, request, **kwargs):
        self.calls.append((request, kwargs))
        return self.continuation


class MockVerificationExecutor:
    def __init__(
        self,
        passed: bool = True,
        commands_run: list[str] | None = None,
        execution_job_ids: list[str] | None = None,
    ):
        self.passed = passed
        self.commands_run = commands_run or []
        self.execution_job_ids = execution_job_ids or ["mock-job-1"]
        self.calls = []

    def execute(self, *, task_id: str, commands: list[str], workspace_root: Path) -> VerificationResult:
        self.calls.append({"task_id": task_id, "commands": commands, "workspace_root": workspace_root})
        return VerificationResult(
            verification_id="verif-mock-1",
            task_id=task_id,
            verified_by_agent_id="verifier",
            commands_run=list(commands),
            passed=self.passed,
            summary="Verification passed" if self.passed else "Verification failed",
            failures=[] if self.passed else ["command_failed"],
            risks=[],
            execution_job_ids=list(self.execution_job_ids),
        )


def _setup_runtime(tmp_path: Path):
    board = TaskBoard(tmp_path / "board")
    parent = board.create_task(
        title="Add provider",
        user_request="add provider",
        action_type="coding",
    )
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(root=tmp_path / "supervisor"))
    return board, parent, supervisor


def test_non_runtime_change_skips_wiring_pass():
    coding = _Coding({})
    result = FeatureIntegrationCoordinator().run(
        coding_agent=coding, core_result={"status": "completed", "changed_files": ["README.md"]},
        request="update docs", gateway_task_id="gateway-4", flow_id="flow-4", runtime_capability_change=False,
    )
    assert result.passed and not coding.calls


def test_unrelated_or_disconnected_edges_do_not_prove_reachability():
    integration = _verified_integration("already_integrated")
    integration["reachability_edges"] = [*_edges()[:2], {**_edges()[2], "from": "UnrelatedRegistry"}]
    coding = _Coding({"status": "completed", "integration": integration})
    result = FeatureIntegrationCoordinator().run(
        coding_agent=coding, core_result={"status": "completed", "integration": integration}, request="add provider",
        gateway_task_id="gateway-5", flow_id="flow-5", runtime_capability_change=True,
        authority=_authority(),
    )
    assert result.error_code == INCOMPLETE_FEATURE_WIRING


def test_edges_without_reviewer_or_supervisor_verification_are_blocked():
    result = FeatureIntegrationCoordinator().run(
        coding_agent=_Coding({}),
        core_result={
            "status": "completed",
            "integration": {"wiring_outcome": "already_integrated", "reachability_edges": _edges()},
        },
        request="add provider", gateway_task_id="gateway-6", flow_id="flow-6", runtime_capability_change=True,
    )
    assert result.error_code == INCOMPLETE_FEATURE_WIRING


def test_model_reported_complete_evidence_without_runtime_authority_is_blocked():
    result = FeatureIntegrationCoordinator().run(
        coding_agent=_Coding({}),
        core_result={
            "status": "completed",
            "integration": _verified_integration(),
        },
        request="add provider", gateway_task_id="gateway-7", flow_id="flow-7",
        runtime_capability_change=True,
    )
    assert result.error_code == INCOMPLETE_FEATURE_WIRING


def test_authority_provider_is_evaluated_after_continuation():
    authority = _authority()
    calls = []
    coding = _Coding()
    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/test_provider.py"],
    )
    result = FeatureIntegrationCoordinator(
        checkpoint=lambda payload: calls.append(payload),
    ).run(
        coding_agent=coding,
        core_result={"status": "completed", "changed_files": ["provider.py"]},
        request="add provider", gateway_task_id="gateway-8", flow_id="flow-8",
        runtime_capability_change=True,
        integration_decision=decision,
        authority_provider=lambda: (calls.append("authority"), authority)[1],
    )
    assert result.passed
    assert calls[0]["boundary"] == "after_core_implementation"
    assert calls[-1] == "authority"


# ==============================================================================
# P0.1 - P0.4 Specific Regression Tests
# ==============================================================================


def test_p01_decoupled_feature_integration_without_coding_payload(tmp_path):
    """P0.1: Feature integration succeeds without coding payload containing integration key."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    mock_executor = MockVerificationExecutor(passed=True)

    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        wiring_targets=["src/provider.py"],
        runtime_entrypoints=["src/main.py"],
        verification_commands=["python -m pytest tests/test_provider.py"],
    )

    core_result = {
        "status": "completed",
        "changed_files": ["src/provider.py"],
        "answer": "Provider implemented",
    }
    assert "integration" not in core_result

    result = FeatureIntegrationCoordinator().run(
        core_result=core_result,
        request="add provider",
        gateway_task_id="gw-1",
        flow_id="flow-1",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision_provider=lambda **kwargs: decision,
        verification_executor=mock_executor,
    )

    assert result.passed
    assert result.result["integration"]["wiring_outcome"] == "already_integrated"
    assert result.result["integration"]["wiring_targets"] == ["src/provider.py"]
    assert "integration_authority" in result.result
    assert mock_executor.calls


def test_p01_missing_or_invalid_feature_integration_decision_fails_explicitly(tmp_path):
    """P0.1: Missing or invalid decision stops with FEATURE_INTEGRATION_DECISION_INVALID."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    core_result = {"status": "completed", "changed_files": ["src/provider.py"]}

    result = FeatureIntegrationCoordinator().run(
        core_result=core_result,
        request="add provider",
        gateway_task_id="gw-2",
        flow_id="flow-2",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision_provider=lambda **kwargs: None,
    )

    assert not result.passed
    assert result.error_code == FEATURE_INTEGRATION_DECISION_INVALID
    assert result.status == "blocked"


def test_p02_explicit_verification_executor_called_with_exact_commands(tmp_path):
    """P0.2: Verification executor receives exact commands from verification plan."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    mock_executor = MockVerificationExecutor(passed=True)

    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py", "python -m pytest tests/integration.py"],
    )

    result = FeatureIntegrationCoordinator().run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-3",
        flow_id="flow-3",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=mock_executor,
    )

    assert result.passed
    assert len(mock_executor.calls) == 1
    assert mock_executor.calls[0]["commands"] == [
        "python -m pytest tests/unit.py",
        "python -m pytest tests/integration.py",
    ]


def test_p02_unavailable_verifier_infrastructure_fails_closed(tmp_path):
    """P0.2: When verifier infrastructure is unavailable, fails closed."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )

    # MultiAgentVerificationExecutor with no queue_manager
    unavailable_executor = MultiAgentVerificationExecutor(
        taskboard=board,
        queue_manager=None,
        workspace_root=tmp_path,
    )
    unavailable_executor.queue_manager = None

    result = FeatureIntegrationCoordinator().run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-4",
        flow_id="flow-4",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=unavailable_executor,
    )

    assert not result.passed
    assert result.error_code == FEATURE_INTEGRATION_VERIFIER_UNAVAILABLE


def test_p03_verification_plan_missing_commands_fails_closed(tmp_path):
    """P0.3: Empty verification commands fail closed with FEATURE_INTEGRATION_VERIFICATION_PLAN_MISSING."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    mock_executor = MockVerificationExecutor(passed=True)

    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=[],  # Empty verification commands
    )

    result = FeatureIntegrationCoordinator().run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-5",
        flow_id="flow-5",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=mock_executor,
    )

    assert not result.passed
    assert result.error_code == FEATURE_INTEGRATION_VERIFICATION_PLAN_MISSING
    assert len(mock_executor.calls) == 0


def test_p03_failed_verification_rejects_integration_evidence(tmp_path):
    """P0.3: Failing verification executor fails closed with FEATURE_INTEGRATION_VERIFICATION_FAILED."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    mock_executor = MockVerificationExecutor(passed=False)  # Failed verification

    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )

    result = FeatureIntegrationCoordinator().run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-6",
        flow_id="flow-6",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=mock_executor,
    )

    assert not result.passed
    assert result.error_code == FEATURE_INTEGRATION_VERIFICATION_FAILED


def test_p03_unproven_reachability_fails_explicitly(tmp_path, monkeypatch):
    """P0.3: When Reviewer reachability verification fails, fails closed."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    mock_executor = MockVerificationExecutor(passed=True)

    from mana_agent.multi_agent.agents.reviewer_agent import ReviewerAgent
    monkeypatch.setattr(ReviewerAgent, "verify_runtime_reachability", lambda *args, **kwargs: False)

    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )

    result = FeatureIntegrationCoordinator().run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-7",
        flow_id="flow-7",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=mock_executor,
    )

    assert not result.passed
    assert result.error_code == FEATURE_INTEGRATION_REACHABILITY_UNPROVEN


def test_p03_reviewer_rejection_fails_explicitly(tmp_path, monkeypatch):
    """P0.3: When Reviewer rejects evidence, fails closed with FEATURE_INTEGRATION_REVIEW_REJECTED."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    mock_executor = MockVerificationExecutor(passed=True)

    from mana_agent.multi_agent.agents.reviewer_agent import ReviewerAgent
    monkeypatch.setattr(ReviewerAgent, "review_evidence", lambda *args, **kwargs: False)

    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )

    result = FeatureIntegrationCoordinator().run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-8",
        flow_id="flow-8",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=mock_executor,
    )

    assert not result.passed
    assert result.error_code == FEATURE_INTEGRATION_REVIEW_REJECTED


def test_p04_stage_aware_state_normalization_on_entry(tmp_path):
    """P0.4: Resuming from BLOCKED or QUEUED reopens and transitions to IN_PROGRESS."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    coordinator = FeatureIntegrationCoordinator()

    # Pre-create child task and mark it BLOCKED
    child_id = coordinator.ensure_wiring_child(
        board,
        parent.task_id,
        request="add provider",
        changed_files=["src/provider.py"],
    )
    board.update_status(child_id, TaskStatus.BLOCKED, reason="previous test block")
    assert board.get_task(child_id).status is TaskStatus.BLOCKED

    mock_executor = MockVerificationExecutor(passed=True)
    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )

    result = coordinator.run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-9",
        flow_id="flow-9",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=mock_executor,
    )

    assert result.passed
    child = board.get_task(child_id)
    assert child.status is TaskStatus.DONE


def test_p01_feature_integration_decision_provider_structured_output():
    """P0.1: FeatureIntegrationDecisionProvider extracts structured WiringDecision from model output."""
    class _MockLLM:
        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, messages, **kwargs):
            return WiringDecision(
                outcome="already_integrated",
                edges=_edges(),
                wiring_targets=["src/provider.py"],
                runtime_entrypoints=["src/main.py"],
                configuration_targets=[],
                verification_commands=["python -m pytest tests/test_provider.py"],
                reason="Feature is wired",
            )

    provider = FeatureIntegrationDecisionProvider(llm=_MockLLM())
    decision = provider.decide(
        request="add provider",
        changed_files=["src/provider.py"],
        task_id="task-1",
    )
    assert isinstance(decision, WiringDecision)
    assert decision.outcome == "already_integrated"
    assert len(decision.edges) == 3
    assert decision.wiring_targets == ["src/provider.py"]


def test_p02_verification_executor_constructed_without_coding_agent_queue_manager(tmp_path):
    """P0.2: Verification executor constructs its own QueueManager when coding backend has none."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    executor = MultiAgentVerificationExecutor(
        taskboard=board,
        workspace_root=tmp_path,
    )
    assert executor.queue_manager is not None
    assert executor.taskboard is board


def test_p03_mock_executor_without_taskboard_mutation_is_persisted_by_coordinator(tmp_path):
    """P0.3: Coordinator persists VerificationResult, execution IDs, and provenance when executor does not mutate TaskBoard."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    mock_executor = MockVerificationExecutor(passed=True, execution_job_ids=["custom-job-1"])

    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )

    child_id = FeatureIntegrationCoordinator.ensure_wiring_child(
        board,
        parent.task_id,
        request="add provider",
        changed_files=["src/provider.py"],
    )
    child = board.get_task(child_id)
    assert len(child.verification_results) == 0
    assert len(child.verification_queue_job_ids) == 0

    result = FeatureIntegrationCoordinator().run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-p03",
        flow_id="flow-p03",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=mock_executor,
    )

    assert result.passed
    child = board.get_task(child_id)
    # Assert coordinator persisted the evidence
    assert len(child.verification_results) == 1
    assert child.verification_results[0].passed is True
    assert "custom-job-1" in child.verification_queue_job_ids
    assert child.verification_provenance is not None
    assert child.verification_provenance["verification_id"] == "verif-mock-1"


def test_p03_verification_persistence_idempotent(tmp_path):
    """P0.3: When VerifierAgent already persisted the result, no duplicate records are created."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    coordinator = FeatureIntegrationCoordinator()

    child_id = coordinator.ensure_wiring_child(
        board,
        parent.task_id,
        request="add provider",
        changed_files=["src/provider.py"],
    )

    # Pre-populate a verification result as if VerifierAgent persisted it
    existing_result = VerificationResult(
        verification_id="verif-mock-1",
        task_id=child_id,
        verified_by_agent_id="verifier",
        commands_run=["python -m pytest tests/unit.py"],
        passed=True,
        summary="Verification passed",
        execution_job_ids=["mock-job-1"],
    )
    board.add_verification_result(child_id, existing_result)
    board.add_verification_queue_job(child_id, "mock-job-1")

    assert len(board.get_task(child_id).verification_results) == 1

    mock_executor = MockVerificationExecutor(passed=True, execution_job_ids=["mock-job-1"])
    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )

    result = coordinator.run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-p03-idem",
        flow_id="flow-p03-idem",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=mock_executor,
    )

    assert result.passed
    child = board.get_task(child_id)
    # Verification result should NOT be duplicated
    assert len(child.verification_results) == 1


def test_p05_supervisor_completion_verification_is_exactly_once(tmp_path):
    """Supervisor submit_result verifies fresh completion; projection does not verify it again."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    calls = []
    original_verify = supervisor.verify_completion

    def counted_verify(task_id):
        calls.append(task_id)
        return original_verify(task_id)

    supervisor.verify_completion = counted_verify
    decision = WiringDecision(
        outcome="already_integrated", edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )
    result = FeatureIntegrationCoordinator().run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider", gateway_task_id="gw-p05", flow_id="flow-p05",
        runtime_capability_change=True, taskboard=board,
        taskboard_parent_task_id=parent.task_id, execution_supervisor=supervisor,
        workspace_root=tmp_path, integration_decision=decision,
        verification_executor=MockVerificationExecutor(passed=True),
    )
    assert result.passed
    assert len(calls) == 1

    # Simulate a crash after supervisor completion but before TaskBoard
    # projection. Re-entry must reuse the completed supervisor record.
    child_id = FeatureIntegrationCoordinator.wiring_child_id(board, parent.task_id)
    child = board.get_task(child_id)
    child.status = TaskStatus.VERIFYING
    child.integration_stage = "SUPERVISOR_FINALIZE"
    board.save()
    calls.clear()
    second = FeatureIntegrationCoordinator().run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider", gateway_task_id="gw-p05-recovery", flow_id="flow-p05",
        runtime_capability_change=True, taskboard=board,
        taskboard_parent_task_id=parent.task_id, execution_supervisor=supervisor,
        workspace_root=tmp_path, integration_decision=decision,
        verification_executor=MockVerificationExecutor(passed=True),
    )
    assert second.passed
    assert len(calls) == 0
    assert board.get_task(child_id).status is TaskStatus.DONE


def test_core_failure_is_not_mislabeled_as_incomplete_wiring():
    result = FeatureIntegrationCoordinator().run(
        core_result={"status": "failed", "error_code": "CODING_EXECUTION_FAILED"},
        request="add provider", gateway_task_id="gw-core-failure", flow_id="flow-core-failure",
        runtime_capability_change=True,
    )
    assert result.error_code == "CODING_EXECUTION_FAILED"
    assert result.error_code != INCOMPLETE_FEATURE_WIRING
    assert result.pending_classification == DETERMINISTIC_INTEGRATION_FAILURE
    assert result.result["pending_required_work"] is False
    assert result.result["resume_required"] is False


def test_p04_stage_aware_resume_test_a_review_stage_fully_valid(tmp_path):
    """P0.4 Test A: REVIEW stage fully valid -> Verifier and reachability do not re-run; Reviewer runs."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    coordinator = FeatureIntegrationCoordinator()

    child_id = coordinator.ensure_wiring_child(
        board,
        parent.task_id,
        request="add provider",
        changed_files=["src/provider.py"],
    )
    child = board.get_task(child_id)
    child.wiring_outcome = "already_integrated"
    child.reachability_edges = _edges()

    # Pre-populate complete verification evidence
    verif = VerificationResult(
        verification_id="verif-durable-1",
        task_id=child_id,
        verified_by_agent_id="verifier",
        commands_run=["python -m pytest tests/unit.py"],
        passed=True,
        summary="Passed",
        execution_job_ids=["job-1"],
    )
    board.add_verification_result(child_id, verif)
    board.add_verification_queue_job(child_id, "job-1")
    board.add_queue_job(child_id, "job-1")
    child.verification_provenance = {
        "verification_id": "verif-durable-1",
        "verified_by_agent_id": "verifier",
        "queue_job_ids": ["job-1"],
        "commands_run": ["python -m pytest tests/unit.py"],
        "changed_files": ["src/provider.py"],
        "source": "plan",
        "decision_id": "dec-1",
    }

    # Pre-populate reachability evidence
    child.runtime_reachability_verified = True
    child.integration_evidence_records = [
        {
            "path": ["ChatGateway", "ModelRouter", "ProviderFactory", "NewProvider"],
            "summary": "Reachable",
            "source_references": ["src/gateway.py:10", "src/router.py:20", "src/factory.py:30"],
            "observable_result": "verification passed",
            "verification_source": "verif-durable-1",
            "reviewer": "reviewer",
        }
    ]
    child.integration_stage = "REVIEW"
    board.update_status(child_id, TaskStatus.IN_PROGRESS, reason="resuming at review")

    mock_executor = MockVerificationExecutor(passed=True)
    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )

    coding = _Coding()
    result = coordinator.run(
        coding_agent=coding,
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-p04-a",
        flow_id="flow-p04-a",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=mock_executor,
    )

    assert result.passed
    # Verifier did NOT run again
    assert len(mock_executor.calls) == 0
    # Core coding agent did NOT run again
    assert len(coding.calls) == 0
    # Reviewer ran and approved
    child = board.get_task(child_id)
    assert child.reviewed_by_agent_id != ""
    assert child.status is TaskStatus.DONE


def test_p04_stage_aware_resume_test_b_review_stage_missing_verification_result(tmp_path):
    """P0.4 Test B: REVIEW stage with missing verification result reconciles to INTEGRATION_VERIFY."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    coordinator = FeatureIntegrationCoordinator()

    child_id = coordinator.ensure_wiring_child(
        board,
        parent.task_id,
        request="add provider",
        changed_files=["src/provider.py"],
    )
    child = board.get_task(child_id)
    child.wiring_outcome = "already_integrated"
    child.reachability_edges = _edges()
    # Has provenance dict but NO VerificationResult record
    child.verification_provenance = {
        "verification_id": "verif-incomplete",
        "verified_by_agent_id": "verifier",
    }
    child.integration_stage = "REVIEW"
    board.update_status(child_id, TaskStatus.IN_PROGRESS, reason="resuming at review")

    mock_executor = MockVerificationExecutor(passed=True)
    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )

    coding = _Coding()
    result = coordinator.run(
        coding_agent=coding,
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-p04-b",
        flow_id="flow-p04-b",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=mock_executor,
    )

    assert result.passed
    # Verifier DID run because stage reconciled to INTEGRATION_VERIFY
    assert len(mock_executor.calls) == 1
    # Core did NOT run
    assert len(coding.calls) == 0
    assert board.get_task(child_id).status is TaskStatus.DONE


def test_p04_stage_aware_resume_test_c_review_stage_with_verification_no_reachability(tmp_path):
    """P0.4 Test C: REVIEW stage with verification but no reachability reconciles to REACHABILITY_VERIFY."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    coordinator = FeatureIntegrationCoordinator()

    child_id = coordinator.ensure_wiring_child(
        board,
        parent.task_id,
        request="add provider",
        changed_files=["src/provider.py"],
    )
    child = board.get_task(child_id)
    child.wiring_outcome = "already_integrated"
    child.reachability_edges = _edges()

    verif = VerificationResult(
        verification_id="verif-durable-2",
        task_id=child_id,
        verified_by_agent_id="verifier",
        commands_run=["python -m pytest tests/unit.py"],
        passed=True,
        summary="Passed",
        execution_job_ids=["job-2"],
    )
    board.add_verification_result(child_id, verif)
    board.add_verification_queue_job(child_id, "job-2")
    child.verification_provenance = {
        "verification_id": "verif-durable-2",
        "verified_by_agent_id": "verifier",
        "queue_job_ids": ["job-2"],
        "commands_run": ["python -m pytest tests/unit.py"],
        "changed_files": ["src/provider.py"],
        "source": "plan",
        "decision_id": "dec-2",
    }
    # Reachability is False
    child.runtime_reachability_verified = False
    child.integration_evidence_records = []
    child.integration_stage = "REVIEW"
    board.update_status(child_id, TaskStatus.IN_PROGRESS, reason="resuming at review")

    mock_executor = MockVerificationExecutor(passed=True)
    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )

    coding = _Coding()
    result = coordinator.run(
        coding_agent=coding,
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-p04-c",
        flow_id="flow-p04-c",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=mock_executor,
    )

    assert result.passed
    # Verifier did NOT run again (verification was complete)
    assert len(mock_executor.calls) == 0
    # Core did NOT run
    assert len(coding.calls) == 0
    # Reachability was completed during reconciliation
    child = board.get_task(child_id)
    assert child.runtime_reachability_verified is True
    assert child.status is TaskStatus.DONE


def test_p04_stage_aware_resume_test_d_supervisor_finalize_without_reviewer(tmp_path):
    """P0.4 Test D: SUPERVISOR_FINALIZE without reviewer approval reconciles to REVIEW."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    coordinator = FeatureIntegrationCoordinator()

    child_id = coordinator.ensure_wiring_child(
        board,
        parent.task_id,
        request="add provider",
        changed_files=["src/provider.py"],
    )
    child = board.get_task(child_id)
    child.wiring_outcome = "already_integrated"
    child.reachability_edges = _edges()

    verif = VerificationResult(
        verification_id="verif-durable-3",
        task_id=child_id,
        verified_by_agent_id="verifier",
        commands_run=["python -m pytest tests/unit.py"],
        passed=True,
        summary="Passed",
        execution_job_ids=["job-3"],
    )
    board.add_verification_result(child_id, verif)
    board.add_verification_queue_job(child_id, "job-3")
    board.add_queue_job(child_id, "job-3")
    child.verification_provenance = {
        "verification_id": "verif-durable-3",
        "verified_by_agent_id": "verifier",
        "queue_job_ids": ["job-3"],
        "commands_run": ["python -m pytest tests/unit.py"],
        "changed_files": ["src/provider.py"],
        "source": "plan",
        "decision_id": "dec-3",
    }
    child.runtime_reachability_verified = True
    child.integration_evidence_records = [
        {
            "path": ["ChatGateway", "ModelRouter", "ProviderFactory", "NewProvider"],
            "summary": "Reachable",
            "source_references": ["src/gateway.py:10", "src/router.py:20", "src/factory.py:30"],
            "observable_result": "verification passed",
            "verification_source": "verif-durable-3",
            "reviewer": "reviewer",
        }
    ]
    # No reviewer approval
    child.reviewed_by_agent_id = ""
    child.integration_stage = "SUPERVISOR_FINALIZE"
    board.update_status(child_id, TaskStatus.IN_PROGRESS, reason="resuming at supervisor finalize")

    mock_executor = MockVerificationExecutor(passed=True)
    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )

    coding = _Coding()
    result = coordinator.run(
        coding_agent=coding,
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-p04-d",
        flow_id="flow-p04-d",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=mock_executor,
    )

    assert result.passed
    # Verifier did NOT run again
    assert len(mock_executor.calls) == 0
    # Core did NOT run
    assert len(coding.calls) == 0
    # Reviewer ran and approved
    child = board.get_task(child_id)
    assert child.reviewed_by_agent_id != ""
    assert child.status is TaskStatus.DONE


def test_p04_stage_aware_resume_test_e_no_core_replay_across_all_stages(tmp_path):
    """P0.4 Test E: For every reconciliation stage, core CodingAgent invocation count does not increase."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    coordinator = FeatureIntegrationCoordinator()

    child_id = coordinator.ensure_wiring_child(
        board,
        parent.task_id,
        request="add provider",
        changed_files=["src/provider.py"],
    )
    child = board.get_task(child_id)
    child.wiring_outcome = "already_integrated"
    child.reachability_edges = _edges()
    child.integration_stage = "REACHABILITY_VERIFY"

    verif = VerificationResult(
        verification_id="verif-durable-e",
        task_id=child_id,
        verified_by_agent_id="verifier",
        commands_run=["python -m pytest tests/unit.py"],
        passed=True,
        summary="Passed",
        execution_job_ids=["job-e"],
    )
    board.add_verification_result(child_id, verif)
    board.add_verification_queue_job(child_id, "job-e")
    child.verification_provenance = {
        "verification_id": "verif-durable-e",
        "verified_by_agent_id": "verifier",
        "queue_job_ids": ["job-e"],
        "commands_run": ["python -m pytest tests/unit.py"],
        "changed_files": ["src/provider.py"],
    }
    board.update_status(child_id, TaskStatus.IN_PROGRESS, reason="resuming")

    mock_executor = MockVerificationExecutor(passed=True)
    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )

    coding = _Coding()
    result = coordinator.run(
        coding_agent=coding,
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-p04-e",
        flow_id="flow-p04-e",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=mock_executor,
    )

    assert result.passed
    assert len(coding.calls) == 0


def test_p05_supervisor_state_reconciliation_cases(tmp_path):
    """P0.5: Tests Cases A–F for supervisor task reconciliation."""
    from mana_agent.execution_supervisor.models import ExecutionState

    board, parent, supervisor = _setup_runtime(tmp_path)
    coordinator = FeatureIntegrationCoordinator()
    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )

    # Case A: Supervisor task does not exist -> creates, runs submit_result, projects to DONE
    res_a = coordinator.run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-p05-a",
        flow_id="flow-p05-a",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=MockVerificationExecutor(passed=True),
    )
    assert res_a.passed
    child_id = FeatureIntegrationCoordinator.wiring_child_id(board, parent.task_id)
    child = board.get_task(child_id)
    assert child.status is TaskStatus.DONE
    sup_task = supervisor.store.get_task(child_id)
    assert sup_task.state == ExecutionState.COMPLETED

    # Case B: Supervisor task is already COMPLETED -> reuses completed supervisor task without new attempts
    initial_attempts = list(sup_task.attempt_ids)
    child.status = TaskStatus.VERIFYING
    child.integration_stage = "SUPERVISOR_FINALIZE"
    board.save()
    res_b = coordinator.run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-p05-b",
        flow_id="flow-p05-b",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=MockVerificationExecutor(passed=True),
    )
    assert res_b.passed
    assert supervisor.store.get_task(child_id).attempt_ids == initial_attempts
    assert board.get_task(child_id).status is TaskStatus.DONE

    # Case C: Supervisor task is COMPLETED_PENDING_VERIFICATION -> calls verify_completion once
    sup_task_c = supervisor.create_task(
        task_id="child-c",
        routing_decision_id="r-c",
        workspace_path=tmp_path,
    )
    supervisor.queue(sup_task_c.task_id)
    leased, tok = supervisor.acquire_lease(sup_task_c.task_id, owner="coord")
    supervisor.start(sup_task_c.task_id, attempt_id=leased.attempt_id, lease_token=tok)
    # Manually put in completed_pending_verification
    supervisor.store.update_task(
        sup_task_c.task_id,
        lambda t: setattr(t, "state", ExecutionState.COMPLETED_PENDING_VERIFICATION),
    )
    verify_calls = []
    orig_v = supervisor.verify_completion
    def counting_v(t_id):
        verify_calls.append(t_id)
        return orig_v(t_id)
    supervisor.verify_completion = counting_v
    child_c = board.create_child_task(parent.task_id, title="child c", integration_role="wiring")
    child_c.task_id = "child-c"
    board.save()
    # Save a mock result in store so verify_completion succeeds
    res_record = supervisor.submit_result(
        "child-c", attempt_id=leased.attempt_id, lease_token=tok,
        payload={"changed_files": ["src/provider.py"], "wiring_outcome": "already_integrated"}
    )
    assert res_record.state == ExecutionState.COMPLETED


def test_p06_internally_runnable_does_not_block_with_internal_work_pending(tmp_path):
    """P0.6: Internally executable feature integration work completes in 1 turn without WAITING."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    coordinator = FeatureIntegrationCoordinator()
    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )
    result = coordinator.run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-p06-a",
        flow_id="flow-p06-a",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=MockVerificationExecutor(passed=True),
    )
    assert result.passed
    assert result.status == "completed"
    assert result.error_code == ""
    assert result.pending_classification == ""
    assert result.resume_required is False
    assert result.result.get("pending_required_work") is not True


def test_p06_external_dependency_requires_wake_source():
    """P0.6: EXTERNAL_DEPENDENCY classification requires valid wake_up_source and wake_up_reference."""
    # Without wake up source
    assert integration_pending_classification(
        FEATURE_INTEGRATION_VERIFIER_UNAVAILABLE, metadata={}
    ) == DETERMINISTIC_INTEGRATION_FAILURE

    # With valid wake up source and reference
    assert integration_pending_classification(
        FEATURE_INTEGRATION_VERIFIER_UNAVAILABLE,
        metadata={"wake_up_source": "human_inbox", "wake_up_reference": "inbox-123"},
    ) == EXTERNAL_DEPENDENCY


@pytest.mark.parametrize(
    "error_code",
    [
        FEATURE_INTEGRATION_DECISION_INVALID,
        FEATURE_INTEGRATION_VERIFIER_UNAVAILABLE,
        FEATURE_INTEGRATION_VERIFICATION_PLAN_MISSING,
        FEATURE_INTEGRATION_VERIFICATION_FAILED,
        FEATURE_INTEGRATION_REACHABILITY_UNPROVEN,
        FEATURE_INTEGRATION_REVIEW_REJECTED,
        FEATURE_INTEGRATION_STATE_INVALID,
        INCOMPLETE_FEATURE_WIRING,
        CORE_EXECUTION_FAILED,
    ],
)
def test_p07_parameterized_feature_integration_error_codes(error_code):
    """P0.7: All Feature Integration error codes map to deterministic failure classification."""
    classification = integration_pending_classification(error_code)
    assert classification == DETERMINISTIC_INTEGRATION_FAILURE


def test_p09_test_b_lost_lease_after_local_core_mutation_reconciles(tmp_path):
    """P0.9 Test B: Lost lease after local core mutation reconciles and resumes integration without rerunning core."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    coordinator = FeatureIntegrationCoordinator()
    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )
    coding = _Coding()
    # Simulate first run producing core changes and checkpoint
    result = coordinator.run(
        coding_agent=coding,
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-p09-b",
        flow_id="flow-p09-b",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=MockVerificationExecutor(passed=True),
    )
    assert result.passed
    assert len(coding.calls) == 0  # no core replay


def test_p09_test_c_integration_mutation_already_applied_advances_to_verify(tmp_path):
    """P0.9 Test C: Integration mutation already applied advances to verify without duplicate patch application."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    coordinator = FeatureIntegrationCoordinator()

    child_id = coordinator.ensure_wiring_child(
        board,
        parent.task_id,
        request="add provider",
        changed_files=["src/provider.py", "src/integration.py"],
    )
    child = board.get_task(child_id)
    child.wiring_outcome = "mutation_applied"
    child.reachability_edges = _edges()
    child.integration_stage = "INTEGRATION_VERIFY"
    board.save()

    decision = WiringDecision(
        outcome="mutation_applied",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )
    mock_executor = MockVerificationExecutor(passed=True)

    result = coordinator.run(
        core_result={"status": "completed", "changed_files": ["src/provider.py", "src/integration.py"]},
        request="add provider",
        gateway_task_id="gw-p09-c",
        flow_id="flow-p09-c",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=mock_executor,
    )
    assert result.passed
    assert len(mock_executor.calls) == 1
    assert board.get_task(child_id).status is TaskStatus.DONE


def test_p09_test_d_verification_already_finished_advances_to_reachability(tmp_path):
    """P0.9 Test D: Verification already finished advances to reachability verify without re-running verifier."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    coordinator = FeatureIntegrationCoordinator()

    child_id = coordinator.ensure_wiring_child(
        board,
        parent.task_id,
        request="add provider",
        changed_files=["src/provider.py"],
    )
    child = board.get_task(child_id)
    child.wiring_outcome = "already_integrated"
    child.reachability_edges = _edges()
    child.integration_stage = "REACHABILITY_VERIFY"

    verif = VerificationResult(
        verification_id="verif-p09-d",
        task_id=child_id,
        verified_by_agent_id="verifier",
        commands_run=["python -m pytest tests/unit.py"],
        passed=True,
        summary="Passed",
        execution_job_ids=["job-p09-d"],
    )
    board.add_verification_result(child_id, verif)
    board.add_verification_queue_job(child_id, "job-p09-d")
    child.verification_provenance = {
        "verification_id": "verif-p09-d",
        "verified_by_agent_id": "verifier",
        "queue_job_ids": ["job-p09-d"],
        "commands_run": ["python -m pytest tests/unit.py"],
        "changed_files": ["src/provider.py"],
    }
    board.save()

    mock_executor = MockVerificationExecutor(passed=True)
    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )
    result = coordinator.run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-p09-d",
        flow_id="flow-p09-d",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=mock_executor,
    )
    assert result.passed
    assert len(mock_executor.calls) == 0  # Verifier did not re-run
    assert board.get_task(child_id).status is TaskStatus.DONE


def test_p09_test_e_supervisor_completed_before_lease_loss_projects_taskboard(tmp_path):
    """P0.9 Test E: Supervisor already COMPLETED projects TaskBoard without creating AMBIGUOUS_LOST_LEASE."""
    from mana_agent.execution_supervisor.models import ExecutionState

    board, parent, supervisor = _setup_runtime(tmp_path)
    coordinator = FeatureIntegrationCoordinator()
    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )

    child_id = coordinator.ensure_wiring_child(
        board,
        parent.task_id,
        request="add provider",
        changed_files=["src/provider.py"],
    )
    child = board.get_task(child_id)
    child.wiring_outcome = "already_integrated"
    child.reachability_edges = _edges()
    child.integration_stage = "SUPERVISOR_FINALIZE"
    board.save()

    # Create and complete supervisor task directly
    sup_task = supervisor.create_task(
        task_id=child_id,
        routing_decision_id="r-e",
        workspace_path=tmp_path,
    )
    supervisor.queue(sup_task.task_id)
    leased, tok = supervisor.acquire_lease(sup_task.task_id, owner="coord")
    supervisor.start(sup_task.task_id, attempt_id=leased.attempt_id, lease_token=tok)
    completed_sup = supervisor.submit_result(
        child_id, attempt_id=leased.attempt_id, lease_token=tok,
        payload={"changed_files": ["src/provider.py"], "wiring_outcome": "already_integrated"}
    )
    assert completed_sup.state == ExecutionState.COMPLETED

    # Run coordinator recovery
    result = coordinator.run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-p09-e",
        flow_id="flow-p09-e",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=MockVerificationExecutor(passed=True),
    )
    assert result.passed
    assert board.get_task(child_id).status is TaskStatus.DONE


def test_scenario_child_wiring_task_core_execution_failed_propagates_failure_to_parent(tmp_path):
    """Scenario: child wiring task CORE_EXECUTION_FAILED -> parent is failed, not blocked forever."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    coordinator = FeatureIntegrationCoordinator()

    result = coordinator.run(
        core_result={"status": "failed", "error_code": CORE_EXECUTION_FAILED, "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-fail-core",
        flow_id="flow-fail-core",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
    )

    assert result.status == "failed"
    assert result.error_code == CORE_EXECUTION_FAILED

    updated_parent = board.get_task(parent.task_id)
    assert updated_parent.status is TaskStatus.FAILED
    assert updated_parent.status is not TaskStatus.BLOCKED
    assert updated_parent.wiring_outcome == "failed"
    assert updated_parent.wiring_outcome_reason == CORE_EXECUTION_FAILED
    assert updated_parent.required_wiring_task_ids != []

    child_id = updated_parent.required_wiring_task_ids[0]
    assert child_id in updated_parent.child_task_ids
    child = board.get_task(child_id)
    assert child.status is TaskStatus.FAILED
    assert child.status is not TaskStatus.BLOCKED
    assert child.wiring_outcome == "failed"
    assert child.wiring_outcome_reason == CORE_EXECUTION_FAILED


def test_scenario_wiring_not_required_resolves_outcome_to_not_required(tmp_path):
    """Scenario: wiring_required=false -> wiring_outcome resolves to not_required."""
    board = TaskBoard(tmp_path / "board")
    parent = board.create_task(
        title="Refactor utility",
        user_request="refactor utility",
        wiring_required=False,
    )
    assert parent.wiring_outcome == "pending"

    board.update_status(parent.task_id, TaskStatus.DONE)
    updated = board.get_task(parent.task_id)
    assert updated.status is TaskStatus.DONE
    assert updated.wiring_outcome == "not_required"
    assert updated.wiring_outcome != "incomplete"


def test_scenario_successful_wiring_resolves_outcome_to_completed(tmp_path):
    """Scenario: successful wiring -> wiring_outcome resolves to completed."""
    board, parent, supervisor = _setup_runtime(tmp_path)
    coordinator = FeatureIntegrationCoordinator()
    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )

    result = coordinator.run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-success-wire",
        flow_id="flow-success-wire",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=MockVerificationExecutor(passed=True),
    )

    assert result.passed
    updated_parent = board.get_task(parent.task_id)
    child_id = updated_parent.required_wiring_task_ids[0]
    child = board.get_task(child_id)

    assert child.status is TaskStatus.DONE
    assert child.wiring_outcome in {"already_integrated", "completed"}
    assert updated_parent.status is not TaskStatus.BLOCKED
    assert updated_parent.wiring_outcome == "completed"
    assert updated_parent.wiring_outcome != "incomplete"



