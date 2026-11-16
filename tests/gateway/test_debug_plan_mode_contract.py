"""Mode contracts across the real TaskBoard, supervisor, and human inbox."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mana_agent.agent.mode import ModePhase, PlanRevision, PlanStep
from mana_agent.agent.plan_service import EvidenceRequest, PlanService
from mana_agent.config.autonomous_config import AutonomousExecutionConfig
from mana_agent.gateway.lane_coordinator import LaneCoordinator
from mana_agent.gateway.mode_workflow import ModeWorkflow, validate_step_patch
from mana_agent.human_inbox import default_human_inbox_service
from mana_agent.human_inbox.models import ResponseOperation


def draft(path="app.py"):
    return {
        "action": "draft", "reason": "Scope identified",
        "draft": {
            "objectives": ["Handle the failure"], "findings": "The handler returns the wrong value.",
            "scope": ["Change the handler"], "exclusions": ["New behavior elsewhere"],
            "steps": [
                {"step_id": "fix", "instruction": "Fix the handler", "changes": {path: ["update"]},
                 "acceptance_criteria": ["The handler returns a value"]},
                {"step_id": "verify", "instruction": "Run the focused check", "depends_on": ["fix"],
                 "verification": [{"argv": ["python", "app.py"]}],
                 "acceptance_criteria": ["The focused check passes"]},
            ],
        },
    }


class Decisions:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)
        self.inputs = []

    def decide_sync(self, messages, schema):
        self.inputs.append(json.loads(messages[-1].content))
        return next(self.decisions)


@pytest.fixture
def mode_gateway(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text("before\n")
    lane = LaneCoordinator(root)
    repository = lane.taskboard.store
    session = SimpleNamespace(workspace_id=repository.workspace_id, primary_repository_id=repository.repository_id)
    gateway = SimpleNamespace(
        root=root, _lane_coordinator=lane,
        _workspaces=SimpleNamespace(store=SimpleNamespace(get_session=lambda _: session)),
        _autonomous_controller=SimpleNamespace(config=AutonomousExecutionConfig()),
        human_inbox_service=default_human_inbox_service(branch_controller=lane),
        _entry_router=SimpleNamespace(llm=None), _coding_agent=None,
        _event_sink=None, _append_session_message=lambda *a, **kw: None,
    )
    workflow = ModeWorkflow(gateway)

    def resume(task_id, inbox_id, claim_id, _response):
        inbox = gateway.human_inbox_service
        item = inbox.repository.get(inbox_id)
        protected = inbox.repository.read_protected_context(item.protected_context_ref)
        workflow.resume(task_id, item, protected, claim_id)

    lane.set_human_resume_dispatcher(resume)
    return gateway, workflow


def test_draft_waits_for_exact_approval_and_executes_ordered_steps(mode_gateway, monkeypatch):
    gateway, workflow = mode_gateway
    events = []
    (gateway.root / "untouched.txt").write_text("user work")
    decisions = Decisions(draft(), {"disposition": "in_scope", "reason": "Only the handler changed"})
    gateway._entry_router.llm = decisions

    class Coding:
        def generate(self, _prompt, **kwargs):
            assert kwargs["execution_mode"] == "read_only"
            assert kwargs["coding_mode_files"] == ["app.py"]
            assert kwargs["coding_mode_proposal"]
            events.append("fix")
            patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-before\n+after\n*** End Patch"
            return {"status": "completed", "output": json.dumps({"patch": patch})}

    gateway._coding_agent = Coding()

    async def verify(root, command, **kwargs):
        assert (root / "app.py").read_text() == "after\n"
        assert command.argv == ("python", "app.py")
        events.append("verify")
        return {"passed": True, "exit_code": 0, "output": "passed"}

    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_isolated_verification", verify)
    from mana_agent.approval.policy import set_approval_mode
    from mana_agent.approval.models import ApprovalMode

    set_approval_mode(ApprovalMode.FULL_ACCESS)
    result = workflow.start("session", "Fix the handler", save=False)
    assert result.status == "input_required", result.message
    assert result.data["phase"] == "awaiting_approval"
    assert not events and (gateway.root / "app.py").read_text() == "before\n"
    key = result.data["inbox_id"]
    with pytest.raises(PermissionError):
        workflow.respond("other-session", key, ResponseOperation.APPROVE, frontend="cli")
    workflow.respond("session", key, ResponseOperation.APPROVE, frontend="cli")
    final = workflow.wait_for_response(result.data["task_id"], key, timeout=8)
    assert final.data.get("phase") == "completed", final.message
    assert events == ["fix", "verify"]
    assert (gateway.root / "untouched.txt").read_text() == "user work"
    task = workflow.board.get_task(result.data["task_id"])
    assert set(task.mode_workflow["done_steps"]) == {"fix", "verify"}
    assert task.mode_workflow["approved_digest"] == result.data["digest"]


def test_clarification_is_structured_and_kept_in_revision(mode_gateway):
    gateway, workflow = mode_gateway
    decisions = Decisions({
        "action": "clarify", "reason": "Two behaviors are possible",
        "questions": [{"field_id": "behavior", "prompt": "What should an empty value do?",
                       "reason": "Changes the contract", "choices": {"keep": "Keep it", "reject": "Reject it"},
                       "recommended_choice": "reject"}],
    }, draft())
    gateway._entry_router.llm = decisions
    waiting = workflow.start("session", "Handle an empty value", save=False)
    assert waiting.next_prompt["kind"] == "mode.clarification"
    result = workflow.receive("session", "reject", task=workflow.active("session"))
    assert result.status == "success", result.message
    proposed = workflow.wait_for_response(waiting.data["task_id"], waiting.data["inbox_id"], timeout=8)
    assert proposed.next_prompt["kind"] == "mode.approval"
    assert decisions.inputs[-1]["constraints"] == {"behavior": "reject"}
    plan = PlanRevision.model_validate(workflow.active("session").mode_workflow["plan"])
    assert plan.constraints == {"behavior": "reject"}


def test_debug_search_stays_with_reported_file_and_rejects_unjustified_expansion(mode_gateway, monkeypatch):
    gateway, workflow = mode_gateway
    (gateway.root / "other.py").write_text("before\n")
    opened = []
    original_open = Path.open

    def track(path, *args, **kwargs):
        if path.parent == gateway.root:
            opened.append(path.name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", track)
    model = Decisions(
        {"action": "inspect", "reason": "Reported stack location", "evidence": {
            "tool": "read_file", "level": 1, "paths": ["app.py"], "reason": "Stack points to app.py:1"}},
        {"action": "inspect", "reason": "Broaden", "evidence": {
            "tool": "repo_search", "level": 4, "query": "before", "reason": "Look everywhere"}},
    )
    result = workflow.start("session", "Traceback points at app.py:1", mode="debug", llm=model)
    assert result.status == "error"
    assert "narrower scope" in result.message
    assert opened == ["app.py"]


def test_exact_patch_operations_and_protected_paths(tmp_path):
    (tmp_path / "app.py").write_text("before\n")
    step = PlanStep(step_id="fix", instruction="Fix the handler", changes={"app.py": ["update"]},
                    acceptance_criteria=["Handler succeeds"])
    approval_patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-before\n+after\n*** End Patch"
    assert validate_step_patch(tmp_path, step, approval_patch) == [("app.py", "update")]
    for material in (approval_patch.replace("app.py", "other.py"),
                     "*** Begin Patch\n*** Delete File: app.py\n*** End Patch"):
        with pytest.raises(PermissionError):
            validate_step_patch(tmp_path, step, material)
    (tmp_path / "app.py").unlink()
    (tmp_path / "app.py").symlink_to(tmp_path / "another.py")
    with pytest.raises(ValueError, match="Symbolic links"):
        validate_step_patch(tmp_path, step, approval_patch)


def test_planning_without_a_model_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="native decision model"):
        PlanService(tmp_path, llm=None).next_decision("plan", {"request": "change a thing"})


def test_scoped_search_excludes_unrelated_file(tmp_path):
    (tmp_path / "related.py").write_text("target\n")
    (tmp_path / "unrelated.py").write_text("target\n")
    service = PlanService(tmp_path, llm=None)
    result = service.inspect(EvidenceRequest(tool="repo_search", level=1, paths=["related.py"],
                                             query="target", reason="Reported file"))
    assert [match["file"] for match in result["matches"]] == ["related.py"]


def test_revision_supersedes_the_previous_approval(mode_gateway):
    gateway, workflow = mode_gateway
    gateway._entry_router.llm = Decisions(draft(), draft())
    initial = workflow.start("session", "Fix the handler", save=False)
    old_id = initial.data["inbox_id"]
    workflow.respond("session", old_id, ResponseOperation.DENY,
                     frontend="cli", comment="Revise plan:\nRetain the existing public method")
    revised = workflow.wait_for_response(initial.data["task_id"], old_id, timeout=8)
    assert revised.data["revision"] == 2
    assert revised.data["digest"] != initial.data["digest"]
    with pytest.raises(PermissionError):
        workflow.respond("session", old_id, ResponseOperation.APPROVE, frontend="cli")
    assert (gateway.root / "app.py").read_text() == "before\n"


@pytest.mark.parametrize("backend", ["internal", "external"])
def test_backend_patch_cannot_write_an_unapproved_file(mode_gateway, backend):
    gateway, workflow = mode_gateway
    gateway._entry_router.llm = Decisions(draft())
    (gateway.root / "other.py").write_text("before\n")
    patch = "*** Begin Patch\n*** Update File: other.py\n@@\n-before\n+after\n*** End Patch"
    gateway._coding_agent = SimpleNamespace(generate=lambda *a, **kw: {
        "status": "completed", "backend": backend, "output": json.dumps({"patch": patch}),
    })
    result = workflow.start("session", "Fix the handler", save=False)
    key = result.data["inbox_id"]
    workflow.respond("session", key, ResponseOperation.APPROVE, frontend="cli")
    final = workflow.wait_for_response(result.data["task_id"], key, timeout=8)
    assert final.data["phase"] == "blocked"
    assert "outside step" in final.message
    assert (gateway.root / "other.py").read_text() == "before\n"


def test_cancelled_task_discards_a_late_coding_proposal(mode_gateway):
    gateway, workflow = mode_gateway
    gateway._entry_router.llm = Decisions(draft())
    result = workflow.start("session", "Fix the handler", save=False)
    task_id = result.data["task_id"]

    def propose(*args, **kwargs):
        workflow.lane.cancel_tree(task_id)
        workflow.note_cancellation(task_id)
        patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-before\n+after\n*** End Patch"
        return {"status": "completed", "output": json.dumps({"patch": patch})}

    gateway._coding_agent = SimpleNamespace(generate=propose)
    key = result.data["inbox_id"]
    workflow.respond("session", key, ResponseOperation.APPROVE, frontend="cli")
    final = workflow.wait_for_response(task_id, key, timeout=8)
    assert final.data["phase"] == "cancelled"
    assert (gateway.root / "app.py").read_text() == "before\n"


def test_steering_requests_a_new_revision_before_a_write(mode_gateway):
    from mana_agent.execution.steering import SteeringCoordinator, SteeringKind, SteeringScope, SteeringSubmission

    gateway, workflow = mode_gateway
    gateway._entry_router.llm = Decisions(draft(), draft())
    result = workflow.start("session", "Fix the handler", save=False)
    task_id = result.data["task_id"]
    key = result.data["inbox_id"]
    original = workflow._steering

    def add_steering(task, state, plan, step, service):
        durable = workflow.lane.execution_supervisor.get_task(task)
        SteeringCoordinator(workflow.lane.execution_supervisor).submit(SteeringSubmission(
            execution_id=task, session_id="session", attempt_id=durable.attempt_id,
            generation=durable.attempt_generation, request_id="scope-change",
            text="Change the plan to keep the old handler as well",
            kind=SteeringKind.SCOPE_CHANGE, scope=SteeringScope.EXECUTION, requires_approval=True,
        ), authenticated_session_id="session")
        return original(task, state, plan, step, service)

    workflow._steering = add_steering
    workflow.respond("session", key, ResponseOperation.APPROVE, frontend="cli")
    final = workflow.wait_for_response(task_id, key, timeout=8)
    assert final.data["phase"] == "awaiting_approval", final.message
    assert final.data["revision"] == 2
    assert final.data["inbox_id"] != key
    assert (gateway.root / "app.py").read_text() == "before\n"


def test_debug_repair_uses_causal_evidence_and_runs_the_focused_check(mode_gateway, monkeypatch):
    from mana_agent.approval.policy import set_approval_mode

    set_approval_mode("full_access")
    gateway, workflow = mode_gateway
    correction = draft()
    correction.update(action="repair", root_cause="The handler emits the invalid value.",
                      cause_evidence=["evidence-1"])
    gateway._entry_router.llm = Decisions(
        {"action": "inspect", "reason": "Read the reported location", "evidence": {
            "tool": "read_file", "level": 1, "paths": ["app.py"], "reason": "Stack points to app.py:1"}},
        correction,
        {"disposition": "in_scope", "reason": "The patch changes the invalid value"},
    )
    patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-before\n+after\n*** End Patch"
    gateway._coding_agent = SimpleNamespace(generate=lambda *a, **kw: {
        "status": "completed", "output": json.dumps({"patch": patch}),
    })
    checks = []

    async def verify(root, command, **kwargs):
        checks.append(command.argv)
        return {"passed": (root / "app.py").read_text() == "after\n", "exit_code": 0, "output": "after"}

    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_isolated_verification", verify)
    final = workflow.start("session", "Fix the handler at app.py:1", mode="debug", repair=True)
    assert final.status == "success", final.message
    assert "invalid value" in final.message
    assert checks == [("python", "app.py")]


def test_resuming_failed_verification_does_not_replay_a_completed_patch(mode_gateway, monkeypatch):
    from mana_agent.approval.policy import set_approval_mode

    set_approval_mode("full_access")
    gateway, workflow = mode_gateway
    gateway._entry_router.llm = Decisions(draft(), {"disposition": "in_scope", "reason": "Matches the step"})
    patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-before\n+after\n*** End Patch"
    proposals = []

    def propose(*_args, **_kwargs):
        proposals.append(patch)
        return {"status": "completed", "output": json.dumps({"patch": patch})}

    gateway._coding_agent = SimpleNamespace(generate=propose)
    outcomes = iter([False, True])

    async def verify(*_args, **_kwargs):
        passed = next(outcomes)
        return {"passed": passed, "exit_code": 0 if passed else 1, "output": "fixture result"}

    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_isolated_verification", verify)
    result = workflow.start("session", "Fix the handler", save=False)
    task_id, key = result.data["task_id"], result.data["inbox_id"]
    workflow.respond("session", key, ResponseOperation.APPROVE, frontend="cli")
    final = workflow.wait_for_response(task_id, key, timeout=8)
    assert final.data["phase"] == "blocked", final.message
    assert "Verification failed" in final.message
    assert workflow.retry(task_id).status == "success"
    assert proposals == [patch]


def test_batch_proposal_wakes_only_the_bound_step(mode_gateway, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from mana_agent.approval.policy import set_approval_mode

    set_approval_mode("full_access")
    gateway, workflow = mode_gateway
    gateway._entry_router.llm = Decisions(draft(), {"disposition": "in_scope", "reason": "Only the handler changed"})
    gateway._bind_runtime_session = lambda _session_id: None
    gateway.background_processes = None
    gateway._autonomous_controller.batch_manager = SimpleNamespace(
        store=SimpleNamespace(get_job=lambda _: SimpleNamespace(
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).timestamp(),
            completed_at=datetime.now(timezone.utc).timestamp(),
        )),
    )
    driver = {}
    monkeypatch.setattr("mana_agent.batch.watcher.ensure_batch_watcher", lambda *_a: None)
    monkeypatch.setattr("mana_agent.batch.operations.BatchOperationStore",
                        lambda _store: SimpleNamespace(load_run=lambda task_id: driver[task_id]))

    def propose(_prompt, *, gateway_task_id, batch_resume, **kwargs):
        if gateway_task_id not in driver:
            workflow.lane.checkpoint(gateway_task_id, boundary="before_batch")
            operation = {"task_id": gateway_task_id, "operation_id": "operation", "batch_id": "provider-job"}
            driver[gateway_task_id] = {"resume": batch_resume, "pending_operation": operation}
            return {"status": "waiting", "pending_operation": operation}
        patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-before\n+after\n*** End Patch"
        return {"status": "completed", "output": json.dumps({"patch": patch})}

    gateway._coding_agent = SimpleNamespace(_internal_engine=SimpleNamespace(model="fixture", provider="fixture"),
                                           generate=propose)

    async def verify(*_a, **_kw):
        return {"passed": True, "exit_code": 0, "output": "verified"}

    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_isolated_verification", verify)
    plan = workflow.start("session", "Fix the handler with Batch", save=False,
                          execution_policy={"mode": "batch", "required": True})
    task_id, item = plan.data["task_id"], plan.data["inbox_id"]
    workflow.respond("session", item, ResponseOperation.APPROVE, frontend="cli")
    import time

    for _ in range(100):
        children = gateway._lane_coordinator.execution_supervisor.store.list_tasks()
        waiting = next((child for child in children if child.parent_task_id == task_id and child.waiting_reason == "waiting_for_batch"), None)
        if waiting:
            break
        time.sleep(0.03)
    assert waiting is not None
    assert (gateway.root / "app.py").read_text() == "before\n"
    child_id = waiting.task_id
    with pytest.raises(PermissionError):
        workflow.resume_batch(task_id, child_id, "another operation", driver[child_id])
    workflow.resume_batch(task_id, child_id, "operation", driver[child_id])
    assert workflow.board.get_task(task_id).mode_workflow["phase"] == "completed"
    assert (gateway.root / "app.py").read_text() == "after\n"


def test_checkout_change_blocks_execution_even_when_target_content_matches(mode_gateway):
    import subprocess

    gateway, workflow = mode_gateway
    subprocess.run(["git", "init", "--quiet", "--initial-branch=first"], cwd=gateway.root, check=True)
    gateway._entry_router.llm = Decisions(draft())
    result = workflow.start("session", "Fix the handler", save=False)
    subprocess.run(["git", "symbolic-ref", "HEAD", "refs/heads/second"], cwd=gateway.root, check=True)
    key = result.data["inbox_id"]
    workflow.respond("session", key, ResponseOperation.APPROVE, frontend="cli")
    final = workflow.wait_for_response(result.data["task_id"], key, timeout=8)
    assert final.data["phase"] == "blocked"
    assert "branch or base commit changed" in final.message
    assert (gateway.root / "app.py").read_text() == "before\n"


def test_destructive_patch_resumes_the_exact_action_after_its_inbox_approval(mode_gateway, monkeypatch):
    import getpass
    import subprocess

    from mana_agent.human_inbox.models import ResponseSubmission

    gateway, workflow = mode_gateway
    subprocess.run(["git", "init", "--quiet"], cwd=gateway.root, check=True)
    subprocess.run(["git", "add", "app.py"], cwd=gateway.root, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
                    "commit", "--quiet", "--message=fixture"], cwd=gateway.root, check=True)
    request = draft()
    request["draft"]["steps"][0]["changes"] = {"app.py": ["delete"]}
    check = {"disposition": "in_scope", "reason": "The approved change removes this file"}
    gateway._entry_router.llm = Decisions(request, check, check)
    proposals = []

    def propose(*args, **kwargs):
        proposals.append(True)
        patch = "*** Begin Patch\n*** Delete File: app.py\n*** End Patch"
        return {"status": "completed", "output": json.dumps({"patch": patch})}

    gateway._coding_agent = SimpleNamespace(generate=propose)

    async def verify(*_a, **_kw):
        return {"passed": not (gateway.root / "app.py").exists(), "exit_code": 0, "output": "checked"}

    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_isolated_verification", verify)
    result = workflow.start("session", "Remove the old handler", save=False)
    task_id, key = result.data["task_id"], result.data["inbox_id"]
    workflow.respond("session", key, ResponseOperation.APPROVE, frontend="cli")
    blocked = workflow.wait_for_response(task_id, key, timeout=8)
    assert blocked.data["phase"] == "blocked", blocked.message
    assert (gateway.root / "app.py").exists()
    pending = workflow.board.get_task(task_id).mode_workflow["pending_patch_approval"]
    item = workflow.inbox.repository.find_for_action(pending["action_id"])[0]
    workflow.inbox.respond(ResponseSubmission(
        inbox_item_id=item.inbox_item_id, operation=ResponseOperation.APPROVE,
        actor_id=getpass.getuser(), current_action_digest=item.action_digest,
    ))
    final = workflow.retry(task_id)
    assert final.status == "success", final.message
    assert not (gateway.root / "app.py").exists()
    assert len(proposals) == 1


def test_invalid_mode_decision_can_resume_to_a_draft_without_execution(mode_gateway):
    gateway, workflow = mode_gateway
    gateway._entry_router.llm = Decisions({"action": "draft", "reason": "Missing structured plan"}, draft())
    result = workflow.start("session", "Fix the handler", save=False)
    assert result.data["phase"] == "blocked"
    assert "draft and repair need a draft" in result.message
    task_id = result.data["task_id"]
    assert f"/task resume {task_id}" in workflow.describe(workflow.active("session")).message

    resumed = workflow.retry(task_id)
    assert resumed.data["task_id"] == task_id
    assert resumed.data["phase"] == "awaiting_approval", resumed.message
    assert (gateway.root / "app.py").read_text() == "before\n"
