from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import subprocess

import pytest

from mana_agent.transactional_actions.adapters import FileActionAdapter, HttpActionAdapter, ShellActionAdapter
from mana_agent.transactional_actions.approvals import ApprovalRegistry
from mana_agent.transactional_actions.gateway import ActionGateway, ApprovalRequired
from mana_agent.transactional_actions.enforcement import (
    TransactionalGatewayRequired,
    assert_model_tool_routed,
)
from mana_agent.transactional_actions.models import (
    ActionIntent,
    ActionState,
    ApprovalScope,
    BlastRadius,
    DataDisclosure,
    PolicyOutcome,
    Reversibility,
    TransactionIntent,
    TransactionStrategy,
    VALID_TRANSITIONS,
    VerificationEvidence,
    utc_now,
)
from mana_agent.transactional_actions.policy import ActionPolicy, PolicyConfig
from mana_agent.transactional_actions.store import ActionStore
from mana_agent.transactional_actions.transactions import TransactionCoordinator


def gateway(tmp_path: Path) -> ActionGateway:
    state = tmp_path / "state"
    return ActionGateway(
        store=ActionStore(state),
        policy=ActionPolicy(PolicyConfig(workspace_roots=(tmp_path,), allowed_http_hosts=("example.test",))),
        approvals=ApprovalRegistry(state / "approvals"),
    )


def file_adapter(tmp_path: Path, *, operation: str = "create", content: str = "new\n", key: str = "file-key-0001") -> FileActionAdapter:
    return FileActionAdapter(
        workspace_root=tmp_path,
        operation=operation,
        path="sample.txt",
        content=content,
        parent_task_id="task-1",
        actor="user",
        originating_agent="agent-1",
        idempotency_key=key,
        snapshot_root=tmp_path / "snapshots",
    )


def test_file_action_commits_only_after_hash_verification(tmp_path: Path) -> None:
    action_gateway = gateway(tmp_path)
    outcome = action_gateway.execute(file_adapter(tmp_path))
    assert outcome.action.state is ActionState.COMMITTED
    assert outcome.action.verification and outcome.action.verification.complete
    assert (tmp_path / "sample.txt").read_text(encoding="utf-8") == "new\n"


def test_committed_idempotent_retry_returns_verified_prior_result(tmp_path: Path) -> None:
    action_gateway = gateway(tmp_path)
    first = action_gateway.execute(file_adapter(tmp_path))
    retry = action_gateway.execute(file_adapter(tmp_path))
    assert retry.duplicate is True
    assert retry.action.action_id == first.action.action_id
    assert retry.result == first.result


def test_conflicting_idempotency_key_is_rejected(tmp_path: Path) -> None:
    action_gateway = gateway(tmp_path)
    action_gateway.execute(file_adapter(tmp_path))
    with pytest.raises(ValueError, match="materially different"):
        action_gateway.execute(file_adapter(tmp_path, operation="edit", content="changed\n"))


def test_delete_requires_exact_single_use_approval(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("recoverable\n", encoding="utf-8")
    action_gateway = gateway(tmp_path)
    adapter = file_adapter(tmp_path, operation="delete", content="", key="delete-key-0001")
    with pytest.raises(ApprovalRequired) as pending:
        action_gateway.execute(adapter)
    action = pending.value.action
    grant = action_gateway.approvals.issue(action, approved_by="local-user")
    outcome = action_gateway.execute(adapter, approval_id=grant.approval_id)
    assert outcome.action.state is ActionState.COMMITTED
    assert not target.exists()
    with pytest.raises(PermissionError):
        action_gateway.approvals.consume(grant.approval_id, action)


def test_approval_is_invalid_after_preview_binding_changes(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    action_gateway = gateway(tmp_path)
    adapter = file_adapter(tmp_path, operation="delete", content="", key="delete-key-0002")
    action = action_gateway.propose(adapter)
    grant = action_gateway.approvals.issue(action, approved_by="local-user")
    mutated = action.model_copy(update={"preview_digest": "0" * 64})
    assert not grant.valid_for(mutated)


def test_material_action_change_invalidates_pending_approval_and_reevaluates(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    action_gateway = gateway(tmp_path)
    pending_adapter = file_adapter(tmp_path, operation="delete", content="", key="mutable-key-0001")
    pending = action_gateway.propose(pending_adapter)
    grant = action_gateway.approvals.issue(pending, approved_by="local-user")
    changed_adapter = file_adapter(tmp_path, operation="edit", content="after\n", key="mutable-key-0001")
    outcome = action_gateway.execute(changed_adapter)
    invalidated = action_gateway.store.get_action(pending.action_id)
    assert invalidated and invalidated.state is ActionState.CANCELLED
    assert not grant.valid_for(invalidated)
    assert outcome.action.action_id != pending.action_id
    assert outcome.action.state is ActionState.COMMITTED


def test_action_transition_table_rejects_skipping_authorization() -> None:
    action = ActionIntent(
        parent_task_id="task", actor="user", originating_agent="agent", tool_name="file",
        operation_name="create", target_resources=["/workspace/a"], normalized_arguments={"path": "/workspace/a"},
        requested_capabilities=["file.create"], expected_side_effects=["create file"], idempotency_key="transition-key",
        verification_plan=["verify hash"], reversibility=Reversibility.FULLY_REVERSIBLE,
    )
    with pytest.raises(ValueError, match="invalid action transition"):
        action.transition(ActionState.EXECUTING)


@pytest.mark.parametrize(
    ("source", "target"),
    [(source, target) for source, targets in VALID_TRANSITIONS.items() for target in targets],
)
def test_every_declared_action_transition_is_accepted(source: ActionState, target: ActionState) -> None:
    action = ActionIntent(
        parent_task_id="task", actor="user", originating_agent="agent", tool_name="file",
        operation_name="create", target_resources=["/workspace/a"], normalized_arguments={"path": "/workspace/a"},
        requested_capabilities=["file.create"], expected_side_effects=["create file"], idempotency_key="valid-transition-key",
        verification_plan=["verify hash"], reversibility=Reversibility.FULLY_REVERSIBLE,
    ).model_copy(update={"state": source})
    action.transition(target)
    assert action.state is target


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (source, target)
        for source in ActionState
        for target in ActionState
        if target not in VALID_TRANSITIONS[source]
    ],
)
def test_every_undeclared_action_transition_is_rejected(source: ActionState, target: ActionState) -> None:
    action = ActionIntent(
        parent_task_id="task", actor="user", originating_agent="agent", tool_name="file",
        operation_name="create", target_resources=["/workspace/a"], normalized_arguments={"path": "/workspace/a"},
        requested_capabilities=["file.create"], expected_side_effects=["create file"], idempotency_key="invalid-transition-key",
        verification_plan=["verify hash"], reversibility=Reversibility.FULLY_REVERSIBLE,
    ).model_copy(update={"state": source})
    with pytest.raises(ValueError, match="invalid action transition"):
        action.transition(target)


def test_secret_arguments_are_rejected_before_persistence() -> None:
    with pytest.raises(ValueError, match="must be redacted"):
        ActionIntent(
            parent_task_id="task", actor="user", originating_agent="agent", tool_name="http",
            operation_name="POST", target_resources=["https://example.test"],
            normalized_arguments={"authorization": "Bearer secret-value"}, requested_capabilities=["network.write"],
            expected_side_effects=["remote mutation"], idempotency_key="secret-key-0001", verification_plan=["query state"],
        )


def test_unclassified_action_policy_denies_without_fallback() -> None:
    action = ActionIntent(
        parent_task_id="task",
        actor="automation",
        originating_agent="cron",
        tool_name="provider_mutation",
        operation_name="write",
        target_resources=["provider://resource"],
        normalized_arguments={"resource": "provider://resource"},
        requested_capabilities=["provider.write"],
        expected_side_effects=["mutate provider resource"],
        idempotency_key="unclassified-policy-key",
        verification_plan=["query provider state"],
    )
    decision = ActionPolicy().evaluate(action)
    assert decision.outcome is PolicyOutcome.DENY
    assert decision.reason_codes == ["unclassified_tool"]


def test_file_tool_success_without_complete_verification_does_not_commit(tmp_path: Path) -> None:
    class IncompleteFileAdapter(FileActionAdapter):
        def verify(self, action: ActionIntent, result: dict) -> VerificationEvidence:
            return VerificationEvidence(
                complete=False,
                summary="independent verification unavailable",
                checks=[{"check": "forced_incomplete"}],
            )

    adapter = IncompleteFileAdapter(
        workspace_root=tmp_path,
        operation="create",
        path="sample.txt",
        content="new\n",
        parent_task_id="task-1",
        actor="user",
        originating_agent="agent-1",
        idempotency_key="incomplete-verification-key",
        snapshot_root=tmp_path / "snapshots",
    )
    outcome = gateway(tmp_path).execute(adapter)
    assert outcome.action.state is ActionState.FAILED
    assert outcome.action.verification and not outcome.action.verification.complete


def test_shell_preview_uses_exact_argv_and_policy_requires_approval(tmp_path: Path) -> None:
    adapter = ShellActionAdapter(
        argv=["python", "-c", "print('ok')"], cwd=tmp_path, environment={"token": "sensitive"},
        expected_outputs=[], parent_task_id="task", actor="user", originating_agent="agent", idempotency_key="shell-key-0001",
    )
    action = adapter.build_intent()
    preview = adapter.preview(action)
    decision = ActionPolicy(PolicyConfig(workspace_roots=(tmp_path,))).evaluate(action)
    assert preview.exact_invocation["argv"] == ["python", "-c", "print('ok')"]
    assert preview.exact_invocation["environment"]["token"] == "***REDACTED***"
    assert decision.outcome is PolicyOutcome.REQUIRE_APPROVAL


def test_shell_commits_only_when_declared_output_is_observed(tmp_path: Path) -> None:
    def runner(argv, **kwargs):  # noqa: ANN001
        (Path(kwargs["cwd"]) / "declared.txt").write_text("observed", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    action_gateway = gateway(tmp_path)
    adapter = ShellActionAdapter(
        argv=["python", "-c", "write declared output"],
        cwd=tmp_path,
        environment={},
        expected_outputs=["declared.txt"],
        parent_task_id="task",
        actor="user",
        originating_agent="agent",
        idempotency_key="shell-output-key",
        runner=runner,
    )
    pending = action_gateway.propose(adapter)
    grant = action_gateway.approvals.issue(pending, approved_by="local-user")
    outcome = action_gateway.execute(adapter, approval_id=grant.approval_id)
    assert outcome.action.state is ActionState.COMMITTED
    assert outcome.action.verification and outcome.action.verification.complete


def test_http_mutation_verifies_response_semantics_and_records_missing_state_query(tmp_path: Path) -> None:
    action_gateway = gateway(tmp_path)
    adapter = HttpActionAdapter(
        method="POST", url="https://example.test/items", headers={"Authorization": "Bearer hidden"}, body={"name": "x"},
        parent_task_id="task", actor="user", originating_agent="agent", idempotency_key="http-key-0001",
        transport=lambda _request: {"status_code": 201, "headers": {}, "body_preview": "{}"},
    )
    action = action_gateway.propose(adapter)
    grant = action_gateway.approvals.issue(action, approved_by="local-user")
    outcome = action_gateway.execute(adapter, approval_id=grant.approval_id)
    assert outcome.action.state is ActionState.COMMITTED
    assert outcome.action.verification and outcome.action.verification.complete
    assert outcome.action.verification.checks[-1]["observed"] == "not_available"


def test_stale_approval_expires(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    action_gateway = gateway(tmp_path)
    action = action_gateway.propose(file_adapter(tmp_path, operation="delete", content="", key="delete-key-0003"))
    grant = action_gateway.approvals.issue(action, approved_by="local-user")
    grant.expires_at = utc_now() - timedelta(seconds=1)
    action_gateway.approvals._save(grant)
    assert not grant.valid_for(action)


def test_pending_action_and_idempotency_survive_store_restart(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    first_gateway = gateway(tmp_path)
    action = first_gateway.propose(file_adapter(tmp_path, operation="delete", content="", key="restart-key-0001"))
    restarted_store = ActionStore(tmp_path / "state")
    recovered = restarted_store.get_action(action.action_id)
    duplicate = restarted_store.action_for_idempotency_key("restart-key-0001")
    assert recovered and recovered.state is ActionState.AWAITING_APPROVAL
    assert duplicate and duplicate.action_id == action.action_id


def test_unclassified_provider_tool_is_blocked_without_invocation() -> None:
    assert_model_tool_routed("media_play")
    with pytest.raises(TransactionalGatewayRequired, match="default-deny"):
        assert_model_tool_routed("third_party_mutation")
    with pytest.raises(TransactionalGatewayRequired, match="no registered transactional"):
        assert_model_tool_routed("mcp__provider__write")
    with pytest.raises(TransactionalGatewayRequired, match="no registered transactional"):
        assert_model_tool_routed("verify_project")
    assert_model_tool_routed("third_party_read", {"read_only": True})


def test_file_compensation_recovers_from_durable_snapshot_after_restart(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    first_gateway = gateway(tmp_path)
    adapter = file_adapter(
        tmp_path, operation="edit", content="after\n", key="restart-compensation-key"
    )
    committed = first_gateway.execute(adapter)
    restarted_gateway = gateway(tmp_path)
    rebuilt = file_adapter(
        tmp_path, operation="edit", content="after\n", key="restart-compensation-key"
    )
    compensation = restarted_gateway.compensate(committed.action.action_id, rebuilt)
    assert compensation.state is ActionState.COMMITTED
    recovered = restarted_gateway.store.get_action(committed.action.action_id)
    assert recovered and recovered.state is ActionState.COMPENSATED
    assert target.read_text(encoding="utf-8") == "before\n"


def test_file_compensation_refuses_changed_post_action_resource(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("before\n", encoding="utf-8")
    action_gateway = gateway(tmp_path)
    adapter = file_adapter(
        tmp_path, operation="edit", content="after\n", key="unsafe-compensation-key"
    )
    committed = action_gateway.execute(adapter)
    target.write_text("changed again\n", encoding="utf-8")
    compensation = action_gateway.compensate(committed.action.action_id, adapter)
    assert compensation.state is ActionState.FAILED
    assert target.read_text(encoding="utf-8") == "changed again\n"


def test_transaction_partial_failure_compensates_completed_file_action(tmp_path: Path) -> None:
    action_gateway = gateway(tmp_path)
    transaction_id = "txn_compensation_test"
    (tmp_path / "shared.txt").write_text("original", encoding="utf-8")
    first = FileActionAdapter(
        workspace_root=tmp_path, operation="edit", path="shared.txt", content="first",
        parent_task_id="task", actor="user", originating_agent="agent", idempotency_key="transaction-file-1",
        transaction_id=transaction_id, snapshot_root=tmp_path / "snapshots",
    )
    second = FileActionAdapter(
        workspace_root=tmp_path, operation="create", path="shared.txt", content="second",
        parent_task_id="task", actor="user", originating_agent="agent", idempotency_key="transaction-file-2",
        transaction_id=transaction_id, snapshot_root=tmp_path / "snapshots",
    )
    first_action = action_gateway.propose(first)
    second_action = action_gateway.propose(second)
    transaction = TransactionIntent(
        transaction_id=transaction_id,
        parent_task_id="task",
        action_ids=[first_action.action_id, second_action.action_id],
        strategy=TransactionStrategy.COMPENSATE_COMPLETED_ACTIONS,
        commit_conditions=["all actions verify"],
        compensation_plan=["remove the newly created first file"],
    )
    coordinator = TransactionCoordinator(action_gateway, action_gateway.store)
    coordinator.create(transaction)
    outcome = coordinator.execute(
        transaction_id,
        {first_action.action_id: first, second_action.action_id: second},
    )
    assert outcome.failed_action_ids == [second_action.action_id]
    assert outcome.compensated_action_ids == [first_action.action_id]
    assert (tmp_path / "shared.txt").read_text(encoding="utf-8") == "original"


def test_transaction_approval_is_bound_to_exact_plan_and_each_action(tmp_path: Path) -> None:
    action_gateway = gateway(tmp_path)
    transaction_id = "txn_exact_approval"
    adapters: list[FileActionAdapter] = []
    actions = []
    for index in range(2):
        path = tmp_path / f"delete-{index}.txt"
        path.write_text("recoverable", encoding="utf-8")
        adapter = FileActionAdapter(
            workspace_root=tmp_path,
            operation="delete",
            path=path.name,
            parent_task_id="task",
            actor="user",
            originating_agent="agent",
            idempotency_key=f"transaction-delete-{index}",
            transaction_id=transaction_id,
            snapshot_root=tmp_path / "snapshots",
        )
        adapters.append(adapter)
        actions.append(action_gateway.propose(adapter))
    transaction = TransactionIntent(
        transaction_id=transaction_id,
        parent_task_id="task",
        action_ids=[action.action_id for action in actions],
        strategy=TransactionStrategy.STOP_ON_FAILURE,
        commit_conditions=["all deletes verify"],
    )
    coordinator = TransactionCoordinator(action_gateway, action_gateway.store)
    coordinator.create(transaction)
    grants = action_gateway.approve_transaction(
        transaction_id, approved_by="local-user"
    )
    assert set(grants) == set(transaction.action_ids)
    assert all(
        action_gateway.approvals.get(approval_id).scope is ApprovalScope.TRANSACTION
        for approval_id in grants.values()
    )
    outcome = coordinator.execute(
        transaction_id,
        dict(zip(transaction.action_ids, adapters, strict=True)),
        approvals=grants,
    )
    assert not outcome.failed_action_ids
    assert all(action.action.state is ActionState.COMMITTED for action in outcome.actions)


def test_action_events_cover_preview_policy_execution_verification_and_commit(tmp_path: Path) -> None:
    state = tmp_path / "state"
    events: list[dict] = []
    action_gateway = ActionGateway(
        store=ActionStore(state),
        policy=ActionPolicy(PolicyConfig(workspace_roots=(tmp_path,))),
        approvals=ApprovalRegistry(state / "approvals"),
        event_sink=events.append,
    )
    outcome = action_gateway.execute(file_adapter(tmp_path, key="event-key-0001"))
    assert outcome.action.state is ActionState.COMMITTED
    assert [event["event_type"] for event in events] == [
        "action.proposed",
        "action.preview.ready",
        "action.policy.allowed",
        "action.execution.started",
        "action.verification.started",
        "action.verification.completed",
        "action.committed",
    ]
