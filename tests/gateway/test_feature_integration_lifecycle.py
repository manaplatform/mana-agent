from mana_agent.gateway.feature_integration import (
    FeatureIntegrationCoordinator,
    INCOMPLETE_FEATURE_WIRING,
)


def _edges():
    return [
        {"from": "ChatGateway", "to": "ModelRouter", "relation": "calls", "source_reference": "gateway.py:10", "file": "gateway.py", "symbol": "ChatGateway"},
        {"from": "ModelRouter", "to": "ProviderFactory", "relation": "selects", "source_reference": "router.py:20", "file": "router.py", "symbol": "ModelRouter"},
        {"from": "ProviderFactory", "to": "NewProvider", "relation": "constructs", "source_reference": "factory.py:30", "file": "factory.py", "symbol": "ProviderFactory"},
    ]


class _Coding:
    def __init__(self, continuation):
        self.continuation = continuation
        self.calls = []

    def generate(self, request, **kwargs):
        self.calls.append((request, kwargs))
        return self.continuation


def test_runtime_change_requires_integration_before_success():
    coding = _Coding({"status": "completed", "integration": {"wiring_outcome": "mutation_applied", "reachability_edges": _edges()}})
    result = FeatureIntegrationCoordinator().run(
        coding_agent=coding, core_result={"status": "completed", "changed_files": ["provider.py"]},
        request="add provider", gateway_task_id="gateway-1", flow_id="flow-1", runtime_capability_change=True,
    )
    assert result.passed
    assert coding.calls and coding.calls[0][1]["gateway_task_id"] == "gateway-1"


def test_already_integrated_does_not_create_patch():
    coding = _Coding({})
    result = FeatureIntegrationCoordinator().run(
        coding_agent=coding,
        core_result={"status": "completed", "integration": {"wiring_outcome": "already_integrated", "reachability_edges": _edges()}},
        request="add provider", gateway_task_id="gateway-2", flow_id="flow-2", runtime_capability_change=True,
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
    coding = _Coding({"status": "completed", "integration": {"wiring_outcome": "already_integrated", "reachability_edges": [*_edges()[:2], {**_edges()[2], "from": "UnrelatedRegistry"}]}})
    result = FeatureIntegrationCoordinator().run(
        coding_agent=coding, core_result={"status": "completed"}, request="add provider",
        gateway_task_id="gateway-5", flow_id="flow-5", runtime_capability_change=True,
    )
    assert result.error_code == INCOMPLETE_FEATURE_WIRING
