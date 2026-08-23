from mana_agent.gateway.feature_integration import (
    FeatureIntegrationCoordinator,
    IntegrationAuthority,
    INCOMPLETE_FEATURE_WIRING,
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
    def __init__(self, continuation):
        self.continuation = continuation
        self.calls = []

    def generate(self, request, **kwargs):
        self.calls.append((request, kwargs))
        return self.continuation


def test_runtime_change_requires_integration_before_success():
    coding = _Coding({"status": "completed", "integration": _verified_integration()})
    result = FeatureIntegrationCoordinator().run(
        coding_agent=coding, core_result={"status": "completed", "changed_files": ["provider.py"]},
        request="add provider", gateway_task_id="gateway-1", flow_id="flow-1", runtime_capability_change=True,
        authority=_authority(),
    )
    assert result.passed
    assert coding.calls and coding.calls[0][1]["gateway_task_id"] == "gateway-1"


def test_already_integrated_does_not_create_patch():
    coding = _Coding({})
    result = FeatureIntegrationCoordinator().run(
        coding_agent=coding,
        core_result={"status": "completed", "integration": _verified_integration("already_integrated")},
        request="add provider", gateway_task_id="gateway-2", flow_id="flow-2", runtime_capability_change=True,
        authority=_authority(),
    )
    assert result.passed and not coding.calls


def test_missing_provenance_is_blocked_and_checkpointed():
    checkpoints = []
    result = FeatureIntegrationCoordinator(checkpoint=checkpoints.append).run(
        coding_agent=_Coding({"status": "completed"}),
        core_result={"status": "completed", "changed_files": ["provider.py"]},
        request="add provider", gateway_task_id="gateway-3", flow_id="flow-3", runtime_capability_change=True,
    )
    assert result.error_code == INCOMPLETE_FEATURE_WIRING
    assert result.result["core_implementation_preserved"] is True
    assert checkpoints[0]["boundary"] == "after_core_implementation"


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
        coding_agent=coding, core_result={"status": "completed"}, request="add provider",
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
    result = FeatureIntegrationCoordinator(
        checkpoint=lambda payload: calls.append(payload),
    ).run(
        coding_agent=_Coding({"status": "completed", "integration": _verified_integration()}),
        core_result={"status": "completed", "changed_files": ["provider.py"]},
        request="add provider", gateway_task_id="gateway-8", flow_id="flow-8",
        runtime_capability_change=True,
        authority_provider=lambda: authority,
    )
    assert result.passed
    assert calls[0]["boundary"] == "after_core_implementation"
