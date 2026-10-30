from __future__ import annotations

import getpass
from pathlib import Path
from types import SimpleNamespace

import pytest

from mana_agent.coding.models import WorkspaceContext
from mana_agent.coding.programmatic import ToolExecutionDispatcher
from mana_agent.human_inbox.approval_tools import approval_request
from mana_agent.transactional_actions.runtime import default_action_gateway


def dispatcher(root: Path, task: str = "task-one") -> ToolExecutionDispatcher:
    return ToolExecutionDispatcher(
        workspace=WorkspaceContext(repository_path=root, worktree_path=root, working_directory=root,
                                   sandbox="workspaceWrite", allow_in_place_write=True),
        task_id=task,
    )


def test_human_approval_resumes_exact_pending_action(tmp_path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    agent = dispatcher(tmp_path)
    args = {"command": "python -c 'print(123)'"}
    pending = agent.dispatch("run_command", args, call_id="first")
    assert pending["permission_required"] and pending["action_id"]
    gateway = default_action_gateway(tmp_path)
    grant = gateway.approve(pending["action_id"], approved_by=getpass.getuser())
    result = agent.dispatch("run_command", {**args, "action_approval_id": grant}, call_id="second")
    assert result["ok"] and "123" in result["stdout"]
    assert result["action_id"] == pending["action_id"]


def test_model_cannot_approve_or_deny_pending_request(tmp_path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    pending = dispatcher(tmp_path).dispatch("run_command", {"command": "python -c 'print(1)'"}, call_id="x")
    llm = SimpleNamespace(invoke=lambda _: pytest.fail("model must not decide human approval"))
    result = approval_request(action_id=pending["action_id"], command="python -c 'print(1)'",
                              workspace_root=tmp_path, timeout_seconds=0.01,
                              enable_model_judgment=True, llm=llm)
    assert result["status"] == "pending" and not result["approved"]


def test_model_assessment_is_explicitly_advisory():
    from mana_agent.multi_agent.agents.approval_agent import judge_approval_request

    prompts = []
    def invoke(messages):
        prompts.extend(messages)
        return SimpleNamespace(content='{"judgment":"accept", "reason":"advisory"}')

    assert judge_approval_request(command="example", llm=SimpleNamespace(invoke=invoke)).judgment == "accept"
    assert "cannot authorize execution" in prompts[0].content
    assert "must not run automatically" in prompts[0].content


def test_foreign_changed_and_unknown_approvals_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    agent = dispatcher(tmp_path)
    args = {"command": "python -c 'print(1)'"}
    pending = agent.dispatch("run_command", args, call_id="first")
    gateway = default_action_gateway(tmp_path)
    grant = gateway.approve(pending["action_id"], approved_by=getpass.getuser())
    for other, request in [
        (agent, {"command": "python -c 'print(2)'", "action_approval_id": grant}),
        (dispatcher(tmp_path, "task-two"), {**args, "action_approval_id": grant}),
        (agent, {**args, "action_approval_id": "unrecognized"}),
    ]:
        result = other.dispatch("run_command", request, call_id="retry")
        assert not result["ok"] and result["error_code"] == "INVALID_APPROVAL"
    assert gateway.approvals.get(grant).consumed_at is None


def test_programmatic_and_direct_schemas_expose_exact_approval_flow(tmp_path):
    from mana_agent.coding.internal_engine import InternalCodingEngine
    from mana_agent.coding.programmatic import build_programmatic_tool_definitions

    direct = {tool["function"]["name"]: tool["function"]["parameters"]
              for tool in InternalCodingEngine(model="test")._build_tool_definitions(dispatcher(tmp_path))}
    programmatic = {tool["name"]: tool for tool in build_programmatic_tool_definitions()
                    if tool.get("type") == "function"}
    for name in ("shell", "run_command", "run_script_once", "git_commit", "git_generic"):
        assert "action_approval_id" in direct[name]["properties"]
        assert "action_approval_id" in programmatic[name]["parameters"]["properties"]
        assert {"action_id", "inbox_item_id", "permission_required"} <= set(
            programmatic[name]["output_schema"]["properties"]
        )
    assert "action_id" in programmatic["approval_request"]["output_schema"]["properties"]


@pytest.mark.parametrize("nested", [True, False])
def test_shell_exact_human_approval_and_wrapper_binding(tmp_path, monkeypatch, nested):
    import subprocess

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    agent = dispatcher(tmp_path)
    command = "python -c 'print(123)'"
    args = {"action": {"commands": [command], "timeout_ms": 5000}} if nested else {
        "commands": [command], "timeout_ms": 5000,
    }
    pending = agent.dispatch("shell", args)
    assert pending["permission_required"] and pending["action_id"] and pending["inbox_item_id"]
    gateway = default_action_gateway(tmp_path)
    action = gateway.store.get_action(pending["action_id"])
    assert action.normalized_arguments["argv"][-1] == command
    assert action.normalized_arguments["timeout_seconds"] == 5
    grant = gateway.approve(action.action_id, approved_by=getpass.getuser())
    changed = agent.dispatch("shell", {**args, "action_approval_id": grant,
                                       "commands": ["python -V"]})
    assert not changed["ok"]
    cross_wrapper = agent.dispatch("run_command", {"cmd": command, "timeout": 5, "action_approval_id": grant})
    assert cross_wrapper["error_code"] == "INVALID_APPROVAL"
    assert gateway.approvals.get(grant).consumed_at is None
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "123\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = agent.dispatch("shell", {**args, "action_approval_id": grant})
    assert result["ok"] and result["action_id"] == pending["action_id"]
    assert len(calls) == 1 and calls[0][0][-1] == command and calls[0][1]["timeout"] == 5
    assert gateway.approvals.get(grant).consumed_at is not None


@pytest.mark.parametrize("arguments", [
    {"action": {"commands": []}},
    {"commands": ["python -V", "git status"]},
    {"action": {"commands": ["python -V"]}, "commands": ["git status"]},
    {"action": {"commands": [""]}},
])
def test_shell_rejects_ambiguous_or_empty_commands(tmp_path, monkeypatch, arguments):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    result = dispatcher(tmp_path).dispatch("shell", arguments)
    assert not result["ok"] and result["error_code"] == "TOOL_EXECUTION_ERROR"


@pytest.mark.parametrize("command", [
    "git status; git commit -m example",
    "git show --output=created.txt",
    "git status $(python -c 'print(1)')",
])
def test_shell_read_only_prefix_does_not_bypass_approval(tmp_path, monkeypatch, command):
    import subprocess

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("command must not execute"))
    result = dispatcher(tmp_path).dispatch("shell", {"commands": [command]})
    assert not result["ok"] and result["permission_required"]
    assert result["action_id"] and result["inbox_item_id"]


def test_policy_fingerprint_invalidates_pre_fix_pending_actions():
    import hashlib
    import json

    from mana_agent.transactional_actions.policy import PolicyConfig

    config = PolicyConfig(version="custom-version")
    previous = hashlib.sha256(json.dumps(config.model_dump(mode="json"), sort_keys=True,
                                         default=str).encode("utf-8")).hexdigest()
    assert config.fingerprint() != previous


def test_failed_approved_command_does_not_run_model_suggested_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    agent = dispatcher(tmp_path)
    agent.llm = SimpleNamespace(invoke=lambda _: pytest.fail("failure must not invoke model judgment"))
    args = {"command": "python -c 'import sys; sys.exit(7)'"}
    pending = agent.dispatch("run_command", args)
    grant = default_action_gateway(tmp_path).approve(pending["action_id"], approved_by=getpass.getuser())
    result = agent.dispatch("run_command", {**args, "action_approval_id": grant})
    assert not result["ok"]
    assert result["returncode"] == 7
    assert "model_judgment" not in result
    assert result["action_id"] == pending["action_id"]


def test_unknown_action_does_not_create_unrelated_approval(tmp_path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    with pytest.raises(LookupError):
        approval_request(action_id="act_unknown", command="python -V", workspace_root=tmp_path,
                         timeout_seconds=0.01)


def test_approval_lookup_checks_owner_and_workspace_before_disclosing_grant(tmp_path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    pending = dispatcher(repo, "task-one").dispatch("run_command", {"command": "python -V"})
    gateway = default_action_gateway(repo)
    action = gateway.store.get_action(pending["action_id"])
    decision = action.policy_decision.decision_id
    gateway.approve(action.action_id, approved_by=getpass.getuser())

    for selector in ({"action_id": action.action_id}, {"inbox_item_id": pending["inbox_item_id"]}):
        for provenance in ({"task_id": "task-other"}, {"agent_id": "other-agent"},
                           {"source_decision_id": "other-decision"}):
            with pytest.raises(PermissionError, match="different"):
                approval_request(**selector, **provenance, workspace_root=repo, timeout_seconds=0.01)
        with pytest.raises(PermissionError, match="different workspace"):
            approval_request(**selector, workspace_root=other, timeout_seconds=0.01)

    result = approval_request(inbox_item_id=pending["inbox_item_id"], workspace_root=repo,
                              task_id="task-one", agent_id="internal_coding",
                              source_decision_id=decision, timeout_seconds=0.01)
    assert result["approved"] and result["approval_id"] and result["action_id"] == action.action_id
    assert gateway.approvals.get(result["approval_id"]).consumed_at is None


def test_internal_approval_request_rejects_model_workspace_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    agent = dispatcher(repo)
    for key in ("workspace_root", "repo_path", "cwd"):
        result = agent.dispatch("approval_request", {"command": "python -V", key: str(other)})
        assert not result["ok"] and "assigned workspace" in result["error"]


def test_approved_inbox_with_missing_action_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    pending = dispatcher(tmp_path).dispatch("run_command", {"command": "python -c 'print(1)'"})
    gateway = default_action_gateway(tmp_path)
    gateway.approve(pending["action_id"], approved_by=getpass.getuser())
    monkeypatch.setattr("mana_agent.transactional_actions.runtime.default_action_gateway", lambda root: gateway)
    monkeypatch.setattr(gateway.store, "get_action", lambda action_id: None)
    with pytest.raises(LookupError, match="unknown transactional action"):
        approval_request(inbox_item_id=pending["inbox_item_id"], workspace_root=tmp_path, timeout_seconds=0.01)


def test_legacy_model_review_cannot_issue_grant(tmp_path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    pending = dispatcher(tmp_path).dispatch("run_command", {"command": "python -c 'print(1)'"}, call_id="x")
    gateway = default_action_gateway(tmp_path)
    action = gateway.store.get_action(pending["action_id"])
    item = gateway.inbox_service.repository.get(pending["inbox_item_id"])
    from mana_agent.human_inbox.models import InboxStatus
    forged = item.model_copy(update={"status": InboxStatus.APPROVED, "response_channel": "model_judgment"})
    with pytest.raises(PermissionError, match="model judgment"):
        gateway.grant_for_approved_inbox(action, forged)


def test_orphan_grant_does_not_bypass_pending_inbox(tmp_path, monkeypatch):
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    agent = dispatcher(tmp_path)
    args = {"command": "python -c 'print(1)'"}
    pending = agent.dispatch("run_command", args, call_id="same-call")
    gateway = default_action_gateway(tmp_path)
    action = gateway.store.get_action(pending["action_id"])
    grant = gateway.approvals.issue(action, approved_by="model_judgment")
    retry = agent.dispatch("run_command", args, call_id="same-call")
    assert not retry["ok"]
    assert "unresolved durable inbox" in retry["error"]
    assert gateway.approvals.get(grant.approval_id).consumed_at is None


def test_task_grant_requires_human_approval_of_source_action(tmp_path, monkeypatch):
    from datetime import timedelta

    from mana_agent.transactional_actions.adapters import ActionAdapter
    from mana_agent.transactional_actions.gateway import ApprovalRequired
    from mana_agent.transactional_actions.models import (
        ActionIntent, ActionPreview, ApprovalScope, BlastRadius, DataDisclosure,
        Reversibility, VerificationEvidence, utc_now,
    )

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    gateway = default_action_gateway(tmp_path)
    executed = []

    class ComputerAdapter(ActionAdapter):
        def __init__(self, name):
            self.name = name

        def build_intent(self):
            return ActionIntent(
                parent_task_id="task-scope", actor="computer_control", originating_agent="model_tool",
                tool_name="computer", operation_name="filesystem.mkdir", target_resources=[self.name],
                normalized_arguments={"execution_context": {"root_task_id": "task-scope"}},
                requested_capabilities=["computer.files.write"], expected_side_effects=["mkdir"],
                data_disclosure=DataDisclosure.NONE, blast_radius=BlastRadius.SINGLE_RESOURCE,
                reversibility=Reversibility.UNKNOWN, idempotency_key=f"computer-test-{self.name}",
                verification_plan=["observe receipt"],
            )

        def preview(self, action):
            return ActionPreview(summary="mkdir", resources=[{"target": self.name}],
                                 expected_side_effects=action.expected_side_effects)

        def execute(self, action):
            executed.append(self.name)
            return {"ok": True}

        def verify(self, action, result):
            return VerificationEvidence(complete=True, summary="receipt")

    first, second = ComputerAdapter("first"), ComputerAdapter("second")
    with pytest.raises(ApprovalRequired) as pending:
        gateway.execute(first)
    source = pending.value.action
    grant = gateway.approvals.issue(source, approved_by="model_judgment", scope=ApprovalScope.TASK)
    with pytest.raises(PermissionError, match="originating action"):
        gateway.execute(second)
    assert not executed and gateway.approvals.get(grant.approval_id).use_count == 0
    gateway.approve(source.action_id, approved_by=getpass.getuser())
    # A valid multi-use grant and an approved inbox item do not extend the
    # originating policy decision's lifetime.
    source = gateway.store.get_action(source.action_id)
    policy_expiry = source.policy_decision.expires_at
    source.policy_decision.expires_at = utc_now() - timedelta(seconds=1)
    gateway.store.save_action(source)
    with pytest.raises(PermissionError, match="originating action"):
        gateway.execute(second)
    assert not executed and gateway.approvals.get(grant.approval_id).use_count == 0
    source.policy_decision.expires_at = policy_expiry
    gateway.store.save_action(source)
    outcome = gateway.execute(second)
    assert outcome.action.state.value == "committed" and executed == ["second"]


def test_resumed_approved_action_rechecks_human_expiry(tmp_path, monkeypatch):
    from datetime import timedelta

    from mana_agent.transactional_actions.models import ActionState, utc_now

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    pending = dispatcher(tmp_path).dispatch("run_command", {"command": "python -c 'print(1)'"})
    gateway = default_action_gateway(tmp_path)
    gateway.approve(pending["action_id"], approved_by=getpass.getuser())
    action = gateway.store.get_action(pending["action_id"])
    action.transition(ActionState.APPROVED)
    gateway.store.save_action(action)
    item = gateway.inbox_service.repository.get(pending["inbox_item_id"])
    expired = item.model_copy(update={"expires_at": utc_now() - timedelta(seconds=1)})
    monkeypatch.setattr(gateway.inbox_service.repository, "find_for_action", lambda _: [expired])

    class Adapter:
        def build_intent(self):
            return action

        def execute(self, _action):
            pytest.fail("expired human approval must not execute")

    with pytest.raises(PermissionError, match="human approval has expired"):
        gateway.execute(Adapter())
    assert gateway.store.get_action(action.action_id).state is ActionState.APPROVED


def test_resumed_approved_action_requires_durable_human_decision(tmp_path, monkeypatch):
    from mana_agent.transactional_actions.models import ActionState

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    pending = dispatcher(tmp_path).dispatch("run_command", {"command": "python -c 'print(1)'"})
    gateway = default_action_gateway(tmp_path)
    gateway.approve(pending["action_id"], approved_by=getpass.getuser())
    action = gateway.store.get_action(pending["action_id"])
    action.transition(ActionState.APPROVED)
    gateway.store.save_action(action)
    monkeypatch.setattr(gateway.inbox_service.repository, "find_for_action", lambda _: [])

    class Adapter:
        def build_intent(self):
            return action

        def execute(self, _action):
            pytest.fail("an action without its human decision must not execute")

    with pytest.raises(PermissionError, match="lacks its durable human approval"):
        gateway.execute(Adapter())
    assert gateway.store.get_action(action.action_id).execution_attempts == 0


def test_git_commit_approval_invalidated_when_staged_content_changes(tmp_path, monkeypatch):
    import subprocess
    from mana_agent.multi_agent.tools import git_tools
    from mana_agent.transactional_actions.adapters import ActionInvalidatedError
    from mana_agent.transactional_actions.approvals import ApprovalRegistry
    from mana_agent.transactional_actions.gateway import ActionGateway
    from mana_agent.transactional_actions.policy import ActionPolicy, PolicyConfig
    from mana_agent.transactional_actions.store import ActionStore

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    target = tmp_path / "sample.txt"
    target.write_text("first\n")
    subprocess.run(["git", "add", "sample.txt"], cwd=tmp_path, check=True)
    state = tmp_path.parent / (tmp_path.name + "-action-state")
    gateway = ActionGateway(store=ActionStore(state), approvals=ApprovalRegistry(state / "approvals"),
                            policy=ActionPolicy(PolicyConfig(workspace_roots=(tmp_path,))))
    monkeypatch.setattr("mana_agent.transactional_actions.runtime.default_action_gateway", lambda root: gateway)
    original_run = git_tools._run_git_direct

    def no_commit(args, **kwargs):
        if args[0] == "commit":
            pytest.fail("commit must not execute")
        return original_run(args, **kwargs)

    monkeypatch.setattr(git_tools, "_run_git_direct", no_commit)
    pending = git_tools.commit(message="test approval", repo_path=tmp_path)
    assert pending["permission_required"]
    action = gateway.store.get_action(pending["action_id"])
    gateway.approvals.issue(action, approved_by="test-reviewer")
    target.write_text("second\n")
    subprocess.run(["git", "add", "sample.txt"], cwd=tmp_path, check=True)
    with pytest.raises(ActionInvalidatedError):
        git_tools.commit(message="test approval", repo_path=tmp_path)


def test_supervisor_cancellation_fences_approved_action(tmp_path, monkeypatch):
    from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
    from mana_agent.execution_supervisor.models import CheckpointRecord, ExecutionState, TaskRecord
    from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
    from mana_agent.human_inbox import default_human_inbox_service
    from mana_agent.human_inbox.models import ResponseOperation, ResponseSubmission
    from mana_agent.transactional_actions.gateway import ActionGateway
    from mana_agent.transactional_actions.runtime import default_action_gateway

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(root=tmp_path / "supervisor"))
    task = supervisor.store.create_task(
        TaskRecord(task_id="task-sup-1", attempt_id="att-1", checkpoint_id="cp-1", state=ExecutionState.RUNNING)
    )
    supervisor.store.save_checkpoint(
        CheckpointRecord(checkpoint_id="cp-1", task_id="task-sup-1", attempt_id="att-1", state_version=0)
    )
    inbox = default_human_inbox_service(branch_controller=supervisor)
    gateway = default_action_gateway(tmp_path)
    gateway.inbox_service = inbox
    monkeypatch.setattr("mana_agent.transactional_actions.runtime.default_action_gateway", lambda root: gateway)

    agent = dispatcher(tmp_path, task="task-sup-1")
    pending = agent.dispatch("run_command", {"command": "python -c 'print(456)'"})
    assert pending["permission_required"]

    item = inbox.repository.get(pending["inbox_item_id"])
    inbox.respond(
        ResponseSubmission(
            inbox_item_id=pending["inbox_item_id"],
            operation=ResponseOperation.APPROVE,
            actor_id=getpass.getuser(),
            channel="test",
            idempotency_key=f"app_{pending['inbox_item_id']}",
            expected_version=item.version,
            current_action_digest=gateway.store.get_action(pending["action_id"]).approval_digest(),
        )
    )

    grant = gateway.approve(pending["action_id"], approved_by=getpass.getuser())

    # Cancel the supervisor task
    supervisor.cancel("task-sup-1", reason="user cancelled")
    assert supervisor.store.get_task("task-sup-1").state is ExecutionState.CANCELLED

    # Attempt to execute through dispatcher
    result = agent.dispatch("run_command", {"command": "python -c 'print(456)'", "action_approval_id": grant})
    assert not result["ok"]
    assert "cancelled" in result["error"].lower()


def test_supervisor_cancellation_fences_ancestor_task(tmp_path, monkeypatch):
    from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
    from mana_agent.execution_supervisor.models import CheckpointRecord, ExecutionState, TaskRecord
    from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
    from mana_agent.human_inbox import default_human_inbox_service
    from mana_agent.human_inbox.models import ResponseOperation, ResponseSubmission
    from mana_agent.transactional_actions.runtime import default_action_gateway

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(root=tmp_path / "supervisor"))
    parent = supervisor.store.create_task(
        TaskRecord(task_id="task-parent", attempt_id="att-p", checkpoint_id="cp-p", state=ExecutionState.RUNNING)
    )
    supervisor.store.save_checkpoint(
        CheckpointRecord(checkpoint_id="cp-p", task_id="task-parent", attempt_id="att-p", state_version=0)
    )
    child = supervisor.store.create_task(
        TaskRecord(
            task_id="task-child",
            parent_task_id="task-parent",
            attempt_id="att-c",
            checkpoint_id="cp-c",
            state=ExecutionState.RUNNING,
        )
    )
    supervisor.store.save_checkpoint(
        CheckpointRecord(checkpoint_id="cp-c", task_id="task-child", attempt_id="att-c", state_version=0)
    )
    inbox = default_human_inbox_service(branch_controller=supervisor)
    gateway = default_action_gateway(tmp_path)
    gateway.inbox_service = inbox
    monkeypatch.setattr("mana_agent.transactional_actions.runtime.default_action_gateway", lambda root: gateway)

    agent = dispatcher(tmp_path, task="task-child")
    pending = agent.dispatch("run_command", {"command": "python -c 'print(789)'"})
    assert pending["permission_required"]

    item = inbox.repository.get(pending["inbox_item_id"])
    inbox.respond(
        ResponseSubmission(
            inbox_item_id=pending["inbox_item_id"],
            operation=ResponseOperation.APPROVE,
            actor_id=getpass.getuser(),
            channel="test",
            idempotency_key=f"app_{pending['inbox_item_id']}",
            expected_version=item.version,
            current_action_digest=gateway.store.get_action(pending["action_id"]).approval_digest(),
        )
    )

    grant = gateway.approve(pending["action_id"], approved_by=getpass.getuser())

    # Cancel the parent task
    supervisor.cancel("task-parent", reason="parent aborted", propagate=False)
    assert supervisor.store.get_task("task-parent").state is ExecutionState.CANCELLED

    # Child execution must be fenced
    result = agent.dispatch("run_command", {"command": "python -c 'print(789)'", "action_approval_id": grant})
    assert not result["ok"]
    assert "ancestor" in result["error"].lower() or "cancelled" in result["error"].lower()


def test_supervisor_atomic_execution_claim_blocks_cancellation(tmp_path, monkeypatch):
    from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
    from mana_agent.execution_supervisor.models import CancellationStatus, CheckpointRecord, ExecutionState, TaskRecord
    from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
    from mana_agent.human_inbox import default_human_inbox_service
    from mana_agent.human_inbox.models import ResponseOperation, ResponseSubmission
    from mana_agent.transactional_actions.runtime import default_action_gateway

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(root=tmp_path / "supervisor"))
    task = supervisor.store.create_task(
        TaskRecord(task_id="task-side-effect", attempt_id="att-se", checkpoint_id="cp-se", state=ExecutionState.RUNNING)
    )
    supervisor.store.save_checkpoint(
        CheckpointRecord(checkpoint_id="cp-se", task_id="task-side-effect", attempt_id="att-se", state_version=0)
    )
    inbox = default_human_inbox_service(branch_controller=supervisor)
    gateway = default_action_gateway(tmp_path)
    gateway.inbox_service = inbox

    agent = dispatcher(tmp_path, task="task-side-effect")
    pending = agent.dispatch("run_command", {"command": "python -c 'print(101)'"})
    assert pending["permission_required"]

    item = inbox.repository.get(pending["inbox_item_id"])
    inbox.respond(
        ResponseSubmission(
            inbox_item_id=pending["inbox_item_id"],
            operation=ResponseOperation.APPROVE,
            actor_id=getpass.getuser(),
            channel="test",
            idempotency_key=f"app_{pending['inbox_item_id']}",
            expected_version=item.version,
            current_action_digest=gateway.store.get_action(pending["action_id"]).approval_digest(),
        )
    )

    # Claim branch execution in supervisor atomically
    gateway._claim_branch_execution(item)
    updated_task = supervisor.store.get_task("task-side-effect")
    assert updated_task.irreversible_side_effect_started is True

    # Cancellation during side effect must be blocked
    supervisor.cancel("task-side-effect", reason="late cancellation")
    cancelled_task = supervisor.store.get_task("task-side-effect")
    assert cancelled_task.cancellation_status is CancellationStatus.BLOCKED_BY_SIDE_EFFECT
    assert cancelled_task.state is not ExecutionState.CANCELLED


def test_git_commit_fenced_index_isolates_concurrent_git_add(tmp_path, monkeypatch):
    import subprocess
    from mana_agent.multi_agent.tools import git_tools
    from mana_agent.transactional_actions.runtime import default_action_gateway

    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "original.txt").write_text("first\n")
    subprocess.run(["git", "add", "original.txt"], cwd=repo, check=True)

    agent = dispatcher(repo, task="task-git-1")
    pending = agent.dispatch("git_commit", {"message": "approved commit"})
    assert pending["permission_required"]

    gateway = default_action_gateway(repo)
    grant = gateway.approve(pending["action_id"], approved_by=getpass.getuser())

    original_run_direct = git_tools._run_git_direct

    def concurrent_git_direct(args, **kwargs):
        if args[0] == "commit":
            # Simulate another process/thread modifying the main index concurrently
            (repo / "concurrent.txt").write_text("concurrent content\n")
            subprocess.run(["git", "add", "concurrent.txt"], cwd=repo, check=True)
        return original_run_direct(args, **kwargs)

    monkeypatch.setattr(git_tools, "_run_git_direct", concurrent_git_direct)

    result = agent.dispatch("git_commit", {"message": "approved commit", "action_approval_id": grant})
    assert result["ok"]

    # Verify that the commit contains ONLY original.txt, NOT concurrent.txt
    show = subprocess.run(["git", "show", "--name-only", "--oneline", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
    committed_files = [line.strip() for line in show.stdout.strip().splitlines()[1:] if line.strip()]
    assert committed_files == ["original.txt"]

    # Verify that concurrent.txt remains staged in the working index
    st = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True)
    assert "A  concurrent.txt" in st.stdout

