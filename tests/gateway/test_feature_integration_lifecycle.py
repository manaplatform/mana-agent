from pathlib import Path
from typing import Any
import pytest
from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
from mana_agent.gateway.feature_integration import (
    FeatureIntegrationCoordinator,
    FeatureIntegrationVerificationPlan,
    IntegrationAuthority,
    IntegrationVerificationExecutor,
    MultiAgentVerificationExecutor,
    WiringDecision,
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
    def __init__(self, passed: bool = True, commands_run: list[str] | None = None):
        self.passed = passed
        self.commands_run = commands_run or []
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


def test_p04_stage_aware_resume_preserves_completed_stages(tmp_path):
    """P0.4: Resuming at REVIEW stage does not re-run verifier execution or mutation."""
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
    child.verification_provenance = {
        "verification_id": "existing-verif-id",
        "verified_by_agent_id": "verifier",
        "queue_job_ids": [],
        "commands_run": ["python -m pytest tests/unit.py"],
        "changed_files": ["src/provider.py"],
    }
    child.integration_stage = "REVIEW"
    board.update_status(child_id, TaskStatus.IN_PROGRESS, reason="resuming at review")

    mock_executor = MockVerificationExecutor(passed=True)
    decision = WiringDecision(
        outcome="already_integrated",
        edges=_edges(),
        verification_commands=["python -m pytest tests/unit.py"],
    )

    result = coordinator.run(
        core_result={"status": "completed", "changed_files": ["src/provider.py"]},
        request="add provider",
        gateway_task_id="gw-10",
        flow_id="flow-10",
        runtime_capability_change=True,
        taskboard=board,
        taskboard_parent_task_id=parent.task_id,
        execution_supervisor=supervisor,
        workspace_root=tmp_path,
        integration_decision=decision,
        verification_executor=mock_executor,
    )

    assert result.passed
    # Verifier was NOT re-executed because the task was already at stage REVIEW
    assert len(mock_executor.calls) == 0
    assert board.get_task(child_id).status is TaskStatus.DONE

