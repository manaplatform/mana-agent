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


