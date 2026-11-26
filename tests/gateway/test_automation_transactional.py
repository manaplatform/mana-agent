from pathlib import Path

import pytest

from mana_agent.automations.runtime_tools import build_automation_tools
from mana_agent.automations.service import AutomationService
from mana_agent.automations.transactional import AutomationCreateAdapter
from mana_agent.transactional_actions.enforcement import TransactionalGatewayRequired, assert_model_tool_routed
from mana_agent.transactional_actions.models import ActionState, PolicyOutcome
from mana_agent.transactional_actions.policy import ActionPolicy
from mana_agent.transactional_actions.runtime import default_action_gateway


def test_create_is_gated_verified_and_deduplicated(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    monkeypatch.setattr("mana_agent.automations.service.reconcile_deployment", lambda root, automation_id: AutomationService(root).get(automation_id))
    tool = {item.name: item for item in build_automation_tools(tmp_path)}["automation_create"]
    args = {"source_decision_id": "decision", "name": "Check", "description": "",
            "trigger": {"type": "interval", "every_seconds": 18000, "anchor_at": "2030-01-01T00:00:00Z"},
            "job": {"type": "agent_prompt", "prompt": "Check the workspace."},
            "timezone": "UTC", "target_runtime": "local", "permission_references": [],
            "retry_policy": {}, "misfire_policy": {}, "idempotency_key": "check"}
    assert_model_tool_routed("automation_create", tool.metadata)
    adapter = AutomationCreateAdapter(root=tmp_path, arguments=args, invoke=lambda: tool.invoke(args), parent_task_id="task")
    gateway = default_action_gateway(tmp_path, enable_human_inbox=False)
    first = gateway.execute(adapter)
    assert first.action.state is ActionState.COMMITTED
    assert first.action.verification.complete
    assert gateway.execute(adapter).duplicate
    assert len(AutomationService(tmp_path).list()) == 1
    assert not adapter.verify(first.action, {"automation_id": "missing", "snapshot_sha256": "invalid"}).complete


def test_public_tool_result_without_prompt_uses_store_for_verification(tmp_path: Path, monkeypatch) -> None:
    import json

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    monkeypatch.setattr("mana_agent.automations.service.reconcile_deployment", lambda root, automation_id: AutomationService(root).get(automation_id))
    tool = {item.name: item for item in build_automation_tools(tmp_path)}["automation_create"]
    args = {"source_decision_id": "decision", "name": "Check", "description": "",
            "trigger": {"type": "interval", "every_seconds": 18000, "anchor_at": "2030-01-01T00:00:00Z"},
            "job": {"type": "agent_prompt", "prompt": "Check the workspace."},
            "timezone": "UTC", "target_runtime": "local", "permission_references": [],
            "retry_policy": {}, "misfire_policy": {}, "idempotency_key": "check"}

    def invoke_public_result():
        result = json.loads(tool.invoke(args))
        del result["persisted"]["job"]["prompt"]
        return result

    adapter = AutomationCreateAdapter(root=tmp_path, arguments=args, invoke=invoke_public_result, parent_task_id="task")
    outcome = default_action_gateway(tmp_path, enable_human_inbox=False).execute(adapter)
    assert outcome.action.state is ActionState.COMMITTED
    assert outcome.action.verification.complete
    mismatched = AutomationCreateAdapter(root=tmp_path, arguments={**args, "job": {"type": "agent_prompt", "prompt": "Different job"}}, invoke=invoke_public_result, parent_task_id="task")
    assert not mismatched.verify(outcome.action, outcome.result).complete


def test_automation_policy_and_other_mutations_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(TransactionalGatewayRequired):
        assert_model_tool_routed("automation_create", {})
    with pytest.raises(TransactionalGatewayRequired):
        assert_model_tool_routed("automation_delete", {"transactional_adapter": "automation_create"})
    with pytest.raises(ValueError):
        AutomationCreateAdapter(root=tmp_path, arguments={}, invoke=lambda: None, parent_task_id="task")
    adapter = AutomationCreateAdapter(root=tmp_path, arguments={"source_decision_id": "decision", "target_runtime": "github"}, invoke=lambda: None, parent_task_id="task")
    outcome, _, _, _ = ActionPolicy()._automation(adapter.build_intent())
    assert outcome is PolicyOutcome.REQUIRE_APPROVAL
