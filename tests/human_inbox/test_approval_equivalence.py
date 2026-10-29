from __future__ import annotations

import getpass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import anyio
from mana_agent.chat.history import ChatHistory
from mana_agent.human_inbox import default_human_inbox_service
from mana_agent.human_inbox.equivalence import same_approval_request
from mana_agent.human_inbox.models import InboxRequest, InboxRequestType, InboxStatus, ReviewerAssignment, ReviewerType
from mana_agent.transactional_actions.adapters import ShellActionAdapter
from mana_agent.transactional_actions.models import ActionState, PolicyOutcome
from mana_agent.transactional_actions.runtime import default_action_gateway
from mana_agent.tui.app import ManaChatApp
from mana_agent.tui.computer_permission import ComputerPermissionChoice


def make_request(service, **changes):
    values = dict(
        request_type=InboxRequestType.APPROVAL, task_id="task", branch_id="branch",
        requested_by_agent_id="agent", reviewer=ReviewerAssignment(
            reviewer_type=ReviewerType.PERSON, reviewer_id=getpass.getuser()),
        title="Approve operation", summary="Execute operation",
        minimal_context={"operation": "build"},
    )
    values.update(changes)
    return service.create(InboxRequest(**values))


def test_approve_all_only_matching_snapshot(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    service = default_human_inbox_service()
    first, matching = make_request(service), make_request(service)
    different = make_request(service, minimal_context={"operation": "deploy"})
    not_queued = make_request(service)
    app = ManaChatApp(history=ChatHistory(), repo_root=tmp_path)
    app._transactional_modal_queue[:] = [matching.inbox_item_id, different.inbox_item_id]
    choice = ComputerPermissionChoice(first.inbox_item_id, True, transactional=True, approve_all=True)
    anyio.run(app._complete_computer_permission, choice)
    assert service.repository.get(first.inbox_item_id).status == InboxStatus.APPROVED
    assert service.repository.get(matching.inbox_item_id).status == InboxStatus.APPROVED
    assert service.repository.get(different.inbox_item_id).status in {InboxStatus.PENDING, InboxStatus.DELIVERED}
    assert service.repository.get(not_queued.inbox_item_id).status in {InboxStatus.PENDING, InboxStatus.DELIVERED}
    assert app._transactional_modal_queue == [different.inbox_item_id]


def test_unauthorized_user_cannot_approve_or_batch(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    service = default_human_inbox_service()
    reviewer = ReviewerAssignment(reviewer_type=ReviewerType.PERSON, reviewer_id="another-reviewer")
    first, matching = make_request(service, reviewer=reviewer), make_request(service, reviewer=reviewer)
    app = ManaChatApp(history=ChatHistory(), repo_root=tmp_path)
    app._transactional_modal_queue[:] = [matching.inbox_item_id]
    anyio.run(app._complete_computer_permission, ComputerPermissionChoice(first.inbox_item_id, True, transactional=True, approve_all=True))
    assert service.repository.get(first.inbox_item_id).status != InboxStatus.APPROVED
    assert service.repository.get(matching.inbox_item_id).status != InboxStatus.APPROVED
    assert app._transactional_modal_queue == [matching.inbox_item_id]


def test_stale_batch_version_fails_closed(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    service = default_human_inbox_service()
    item = make_request(service)
    app = ManaChatApp(history=ChatHistory(), repo_root=tmp_path)
    choice = ComputerPermissionChoice(item.inbox_item_id, True, transactional=True,
                                      expected_version=item.version + 1)
    anyio.run(app._complete_computer_permission, choice)
    assert service.repository.get(item.inbox_item_id).status != InboxStatus.APPROVED


def test_deny_does_not_batch(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    service = default_human_inbox_service()
    first, matching = make_request(service), make_request(service)
    app = ManaChatApp(history=ChatHistory(), repo_root=tmp_path)
    app._transactional_modal_queue[:] = [matching.inbox_item_id]
    anyio.run(app._complete_computer_permission,
              ComputerPermissionChoice(first.inbox_item_id, None, transactional=True, approve_all=True))
    assert service.repository.get(first.inbox_item_id).status == InboxStatus.DENIED
    assert service.repository.get(matching.inbox_item_id).status != InboxStatus.APPROVED
    assert app._transactional_modal_queue == [matching.inbox_item_id]


def test_expired_match_remains_unapproved(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    service = default_human_inbox_service()
    now = datetime.now(timezone.utc)
    first = make_request(service)
    expired = make_request(service, expires_at=now + timedelta(hours=1))
    monkeypatch.setattr(service, "clock", lambda: now + timedelta(hours=2))
    monkeypatch.setattr("mana_agent.human_inbox.default_human_inbox_service", lambda: service)
    app = ManaChatApp(history=ChatHistory(), repo_root=tmp_path)
    app._transactional_modal_queue[:] = [expired.inbox_item_id]
    anyio.run(app._complete_computer_permission,
              ComputerPermissionChoice(first.inbox_item_id, True, transactional=True, approve_all=True))
    assert service.repository.get(first.inbox_item_id).status == InboxStatus.APPROVED
    assert service.repository.get(expired.inbox_item_id).status == InboxStatus.EXPIRED
    assert app._transactional_modal_queue == [expired.inbox_item_id]


def test_judgment_rejects_scope_and_incomplete_data(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    service = default_human_inbox_service()
    a, b = make_request(service), make_request(service)
    assert same_approval_request(a, b)
    for changes in ({"tenant_id": "else"}, {"project_id": "else"}, {"task_id": "else"},
                    {"execution_id": "different"}, {"checkpoint_id": "different"},
                    {"risk_level": "critical"}, {"minimal_context": {}},
                    {"protected_context_ref": "protected:unknown"}, {"action_id": "other"}):
        assert not same_approval_request(a, b.model_copy(update=changes))
    assert not same_approval_request(a, a)


def test_transactional_judgment_checks_material_and_policy(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway = default_action_gateway(tmp_path)
    adapter = ShellActionAdapter(argv=["python", "-c", "print(1)"], cwd=tmp_path, environment={},
                                 expected_outputs=[], parent_task_id="task", actor="user",
                                 originating_agent="agent", idempotency_key="key-00001")
    action = gateway.propose(adapter)
    assert action.state is ActionState.AWAITING_APPROVAL
    assert action.policy_decision.outcome is PolicyOutcome.REQUIRE_APPROVAL
    duplicate = action.model_copy(deep=True, update={"action_id": "act_duplicate"})
    service = default_human_inbox_service()
    a = make_request(service, action_intent_id=action.action_id, action_digest=action.approval_digest())
    b = make_request(service, action_intent_id=duplicate.action_id, action_digest=duplicate.approval_digest())
    actions = {action.action_id: action, duplicate.action_id: duplicate}
    store = SimpleNamespace(get_action=actions.get)
    assert same_approval_request(a, b, action_store=store)
    duplicate.normalized_arguments = {"different": "argument"}
    assert not same_approval_request(a, b, action_store=store)
    duplicate.normalized_arguments = action.normalized_arguments.copy()
    duplicate.policy_decision.policy_fingerprint = "different"
    assert not same_approval_request(a, b, action_store=store)
    assert not same_approval_request(a, b, action_store=None)


def test_matching_shell_actions_receive_separate_grants(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway = default_action_gateway(tmp_path)

    def propose(key):
        return gateway.propose(ShellActionAdapter(
            argv=["python", "-c", "print(1)"], cwd=tmp_path, environment={},
            expected_outputs=[], parent_task_id="task", actor="user",
            originating_agent="agent", idempotency_key=key,
        ))

    first, second = propose("batch-test-001"), propose("batch-test-002")
    assert first.inbox_item_id and second.inbox_item_id
    service = default_human_inbox_service()
    app = ManaChatApp(history=ChatHistory(), repo_root=tmp_path)
    app._transactional_modal_queue[:] = [second.inbox_item_id]
    anyio.run(app._complete_computer_permission,
              ComputerPermissionChoice(first.inbox_item_id, True, transactional=True, approve_all=True))
    assert service.repository.get(first.inbox_item_id).status == InboxStatus.APPROVED
    assert service.repository.get(second.inbox_item_id).status == InboxStatus.APPROVED
    grant1 = gateway.approvals.find_valid(gateway.store.get_action(first.action_id))
    grant2 = gateway.approvals.find_valid(gateway.store.get_action(second.action_id))
    assert grant1 and grant2 and grant1.approval_id != grant2.approval_id
