"""Mode contracts across the real TaskBoard, supervisor, and human inbox."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from mana_agent.agent.mode import CodingWorkflowDecision, ModePhase, PlanRevision, PlanStep
from mana_agent.agent.plan_service import EvidenceRequest, PlanService
from mana_agent.config.autonomous_config import AutonomousExecutionConfig
from mana_agent.execution.autonomous_controller import AutonomousExecutionController
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
        payload = next(self.decisions)
        return {"decision": payload} if schema["title"] == "ModeDecision" else payload


def file_evidence(path, **fields):
    return {"tool": "read_file", "level": 1, "paths": [path], "reason": "Read the named file", **fields}


def inspection_batch(*evidence):
    return {"action": "inspect_batch", "reason": "Inspect independent locations", "evidence": list(evidence)}


def workflow_choice(*, mode="auto", effect="write", clarification=""):
    return CodingWorkflowDecision(
        mode=mode, requested_effect=effect, objective="Add the handler behavior",
        reason="A repository behavior is missing", investigation_scope="app.py and its tests",
        expected_result="The handler returns a value", risk_and_uncertainty="Potential caller regression",
        verification_requirements=("Run the focused handler check",), clarification=clarification,
    )


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
        _autonomous_controller=AutonomousExecutionController(
            config=AutonomousExecutionConfig(), supervisor=lane.execution_supervisor,
            taskboard=lane.taskboard,
        ),
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


@pytest.mark.parametrize("backend", ["internal", "external"])
def test_auto_records_internal_plan_before_applying_and_verifying(mode_gateway, monkeypatch, backend):
    gateway, workflow = mode_gateway
    operation = {"action": "inspect", "reason": "Inspect the handler", "evidence": file_evidence("app.py")}
    gateway._entry_router.llm = Decisions(operation, draft(), {"disposition": "in_scope", "reason": "Only app.py differs"})

    def propose(_prompt, **kwargs):
        task = next(row for row in workflow.board.list_tasks() if row.mode_workflow.get("mode") == "auto")
        assert task.mode_workflow["plan"]["draft"]["steps"][0]["step_id"] == "fix"
        assert task.mode_workflow["phase"] == "executing"
        assert kwargs["execution_mode"] == "read_only"
        assert (gateway.root / "app.py").read_text() == "before\n"
        patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-before\n+after\n*** End Patch"
        return {"status": "completed", "output": json.dumps({"patch": patch})}

    async def verify(root, command, **kwargs):
        assert (root / "app.py").read_text() == "after\n"
        return {"passed": True, "exit_code": 0, "output": "focused check passed"}

    gateway._coding_agent = SimpleNamespace(generate=propose, active_engine_mode=backend)
    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_isolated_verification", verify)
    from mana_agent.approval.models import ApprovalMode
    from mana_agent.approval.policy import set_approval_mode

    set_approval_mode(ApprovalMode.FULL_ACCESS)
    result = workflow.start("session", "Add the handler behavior", mode="auto", repair=True,
                            workflow=workflow_choice(), source_message_ids=("user-1",))
    assert result.status == "success", result.message
    assert result.data["verification"][0]["passed"]
    task = workflow.board.get_task(result.data["task_id"])
    assert task.mode_workflow["evidence"][0]["id"] == "evidence-1"
    assert task.mode_workflow["approved_digest"] == ""
    assert not task.mode_workflow.get("inbox_id")


def test_auto_switches_to_debug_on_the_same_task_with_observed_evidence(mode_gateway):
    gateway, workflow = mode_gateway

    class Reassessment(Decisions):
        def decide_sync(self, messages, schema):
            payload = super().decide_sync(messages, schema)
            if payload["decision"]["action"] == "switch_workflow":
                payload["decision"]["task_id"] = self.inputs[-1]["task_id"]
            return payload

    gateway._entry_router.llm = Reassessment(
        {"action": "inspect", "reason": "Inspect the handler", "evidence": file_evidence("app.py")},
        {"action": "switch_workflow", "reason": "The handler exists but is broken", "mode": "debug",
         "evidence_ids": ["evidence-1"], "task_id": "filled from request"},
        {"action": "block", "reason": "The repair needs an unavailable dependency"},
    )
    result = workflow.start("session", "Add the handler behavior", mode="auto", repair=True,
                            workflow=workflow_choice(), source_message_ids=("user-1",))
    assert result.status == "error"
    assert result.data["mode"] == "debug"
    inputs = gateway._entry_router.llm.inputs
    assert {item["task_id"] for item in inputs} == {result.data["task_id"]}
    assert inputs[-1]["evidence"] == inputs[-2]["evidence"]
    assert (gateway.root / "app.py").read_text() == "before\n"


def test_auto_uses_the_required_worktree_and_preserves_dirty_primary_checkout(mode_gateway, monkeypatch):
    gateway, workflow = mode_gateway
    from mana_agent.approval.models import ApprovalMode
    from mana_agent.approval.policy import set_approval_mode

    subprocess.run(["git", "init", "-q"], cwd=gateway.root, check=True)
    subprocess.run(["git", "add", "app.py"], cwd=gateway.root, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "Fixture"], cwd=gateway.root, check=True)
    source = gateway.root / "app.py"
    source.write_text("uncommitted user work\n")
    gateway._entry_router.llm = Decisions(
        {"action": "inspect", "reason": "Inspect app.py", "evidence": file_evidence("app.py")},
        draft(), {"disposition": "in_scope", "reason": "Only app.py differs"},
    )

    def propose(_prompt, **kwargs):
        worktree = Path(kwargs["coding_mode_root"])
        assert worktree != gateway.root
        assert (worktree / "app.py").read_text() == "before\n"
        patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-before\n+after\n*** End Patch"
        return {"status": "completed", "output": json.dumps({"patch": patch})}

    async def verify(root, command, **kwargs):
        assert root != gateway.root
        assert (root / "app.py").read_text() == "after\n"
        return {"passed": True, "exit_code": 0, "output": "verified in worktree"}

    gateway._coding_agent = SimpleNamespace(
        generate=propose, active_engine_mode="external",
        codex_settings=SimpleNamespace(worktree_isolation=True),
    )
    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_isolated_verification", verify)
    set_approval_mode(ApprovalMode.FULL_ACCESS)
    result = workflow.start("session", "Add the handler behavior", mode="auto", repair=True,
                            workflow=workflow_choice(), source_message_ids=("user-1",))
    assert result.status == "success", result.message
    assert "Verified worktree:" in result.message
    assert source.read_text() == "uncommitted user work\n"


def test_auto_repairs_failed_verification_without_repeating_completed_patch(mode_gateway, monkeypatch):
    gateway, workflow = mode_gateway
    from mana_agent.approval.models import ApprovalMode
    from mana_agent.approval.policy import set_approval_mode

    second = draft()
    second["draft"]["steps"][0]["step_id"] = "repair"
    second["draft"]["steps"][1]["depends_on"] = ["repair"]
    review = {"disposition": "in_scope", "reason": "Only app.py changes"}
    model = Decisions(
        {"action": "inspect", "reason": "Read app.py", "evidence": file_evidence("app.py")},
        draft(), review, second, review,
    )
    gateway._entry_router.llm = model
    writes = []

    def propose(_prompt, **kwargs):
        prior = (gateway.root / "app.py").read_text().strip()
        new = {"before": "after", "after": "repaired"}[prior]
        writes.append((prior, new))
        patch = f"*** Begin Patch\n*** Update File: app.py\n@@\n-{prior}\n+{new}\n*** End Patch"
        return {"status": "completed", "output": json.dumps({"patch": patch})}

    async def verify(root, command, **kwargs):
        passed = (root / "app.py").read_text() == "repaired\n"
        return {"passed": passed, "exit_code": 0 if passed else 1, "output": "observed handler behavior"}

    gateway._coding_agent = SimpleNamespace(generate=propose, active_engine_mode="internal")
    set_approval_mode(ApprovalMode.FULL_ACCESS)
    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_isolated_verification", verify)
    result = workflow.start("session", "Add the handler behavior", mode="auto", repair=True,
                            workflow=workflow_choice(), source_message_ids=("user-1",))
    assert result.status == "success", result.message
    assert writes == [("before", "after"), ("after", "repaired")]
    assert [result["passed"] for result in result.data["verification"]] == [False, True]
    assert len({row["task_id"] for row in model.inputs if "task_id" in row}) == 1


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
    assert "The next permitted level is 2." in result.message
    assert model.inputs[0]["investigation_scope"]["max_next_level"] == 1
    assert model.inputs[1]["investigation_scope"] == {
        "max_next_level": 2, "highest_observed_level": 1, "evidence_levels": {"evidence-1": 1},
    }
    assert opened == ["app.py"]


def test_debug_mode_activity_keeps_the_frontend_turn(mode_gateway):
    from mana_agent.services.activity_context import ActivityContext, activity_scope, emit_activity
    from mana_agent.services.execution_event_hub import get_execution_event_hub

    gateway, workflow = mode_gateway
    repository = gateway._workspaces.store.get_session("session").primary_repository_id

    class ObservedDecisions(Decisions):
        def decide_sync(self, messages, schema):
            if not self.inputs:
                emit_activity("model.started", title="Waiting for model", status="running",
                              summary="fixture-protected-text", operation_id="fixture-decision")
            return super().decide_sync(messages, schema)

    model = ObservedDecisions(
        {"action": "inspect", "reason": "Read the reported file", "evidence": {
            "tool": "read_file", "level": 1, "paths": ["app.py"], "reason": "Reported file"}},
        {"action": "diagnose", "reason": "The file contains the old value", "root_cause": "Old handler value",
         "cause_evidence": ["evidence-1"]},
    )
    context = ActivityContext(conversation_id="session", execution_id="turn-mode-activity",
                              repository_id=repository, protected_text=("fixture-protected-text",))
    with activity_scope(context):
        result = workflow.start("session", "Inspect the old value", mode="debug", llm=model)

    assert result.status == "success", result.message
    task_id = result.data["task_id"]
    assert workflow.lane.execution_supervisor.get_task(task_id).trigger_turn_id == context.execution_id
    events = get_execution_event_hub().history(conversation_id="session", repository_id=repository,
                                               execution_id=context.execution_id)
    by_type = {event["type"]: event for event in events if event.get("task_id") == task_id}
    for kind in ("task.started", "model.started", "tool.started", "tool.completed", "task.completed"):
        assert by_type[kind]["execution_id"] == context.execution_id
    assert by_type["tool.started"]["tool_name"] == "read_file"
    assert "fixture-protected-text" not in by_type["model.started"]["summary"]


def test_debug_reads_a_known_file_after_a_repository_audit(mode_gateway):
    _, workflow = mode_gateway
    model = Decisions(
        {"action": "inspect", "reason": "Explicit repository audit", "evidence": {
            "tool": "repo_search", "level": 4, "query": "before", "reason": "Find uses of the unexpected value",
            "initial_repository_scope": "explicit_audit"}},
        {"action": "inspect", "reason": "Read the matching file", "evidence": {
            "tool": "read_file", "level": 1, "paths": ["app.py"], "reason": "Inspect the search match",
            "prior_evidence": ["evidence-1"]}},
        {"action": "diagnose", "reason": "The file contains the unexpected value",
         "root_cause": "The handler returns the old value.", "cause_evidence": ["evidence-2"]},
    )

    result = workflow.start("session", "Audit the repository for the old value", mode="debug", llm=model)

    assert result.status == "success", result.message
    search, read = model.inputs[-1]["evidence"]
    assert search["level"] == 4
    assert search["result"]["matches"][0]["file"] == "app.py"
    assert read["level"] == 1
    assert read["result"]["content"] == "before\n"


@pytest.mark.parametrize("mode", ["plan", "debug"])
def test_investigation_batch_reads_multiple_files_in_one_model_decision(mode_gateway, mode):
    gateway, workflow = mode_gateway
    (gateway.root / "test_app.py").write_text("assert handler()\n")
    final = draft() if mode == "plan" else {
        "action": "diagnose", "reason": "Both files explain the issue", "root_cause": "Old handler value",
        "cause_evidence": ["evidence-1", "evidence-2"],
    }
    gateway._autonomous_controller.config = AutonomousExecutionConfig(max_tool_calls=2, max_model_turns=2)
    model = Decisions(inspection_batch(file_evidence("app.py"), file_evidence("test_app.py")), final)

    result = workflow.start("session", "Inspect independently known code and test files",
                            mode=mode, llm=model, no_code=True, save=False)

    assert result.status == "success", result.message
    assert [call["remaining_tool_calls"] for call in model.inputs] == [2, 0]
    observations = model.inputs[1]["evidence"]
    assert [(item["id"], item["result"]["content"]) for item in observations] == [
        ("evidence-1", "before\n"), ("evidence-2", "assert handler()\n"),
    ]
    persisted = workflow.board.get_task(result.data["task_id"]).mode_workflow["evidence"]
    assert len(persisted) == 2
    assert all("execution" in item["result"] for item in persisted)
    assert all("execution" not in item["result"] and "fingerprint" not in item for item in observations)


@pytest.mark.parametrize(("second", "tool_budget", "detail"), [
    (file_evidence("test_app.py"), 1, "remaining tool budget (1)"),
    (file_evidence("app.py", reason="A different reason"), 2, "Choose distinct inspection operations"),
    (file_evidence("test_app.py", level=2, prior_evidence=["evidence-1"]), 2, "The next permitted level is 1"),
])
def test_investigation_batch_rejects_all_reads_if_any_selection_is_invalid(
    mode_gateway, monkeypatch, second, tool_budget, detail,
):
    gateway, workflow = mode_gateway
    gateway._autonomous_controller.config = AutonomousExecutionConfig(max_tool_calls=tool_budget)
    calls = []
    monkeypatch.setattr(PlanService, "inspect", lambda _, request: calls.append(request))
    model = Decisions(inspection_batch(file_evidence("app.py"), second))

    result = workflow.start("session", "Inspect two files", mode="debug", llm=model)

    assert result.data["phase"] == "blocked"
    assert detail in result.message
    assert not calls


def test_investigation_batch_resume_uses_completed_observations(mode_gateway, monkeypatch):
    from mana_agent.multi_agent.runtime import tool_call_gate

    gateway, workflow = mode_gateway
    (gateway.root / "test_app.py").write_text("assert handler()\n")
    calls = []
    real_dispatch = tool_call_gate.execute_host_operation

    def dispatch(**kwargs):
        path = kwargs["arguments"]["path"]
        calls.append(path)
        if len(calls) == 2:
            raise PermissionError("Tool dispatch was interrupted")
        return real_dispatch(**kwargs)

    monkeypatch.setattr(tool_call_gate, "execute_host_operation", dispatch)
    model = Decisions(
        inspection_batch(file_evidence("app.py"), file_evidence("test_app.py")),
        {"action": "inspect", "reason": "Read the outstanding file", "evidence": file_evidence("test_app.py")},
        {"action": "diagnose", "reason": "Both files explain the issue", "root_cause": "Old handler value",
         "cause_evidence": ["evidence-1", "evidence-2"]},
    )
    gateway._entry_router.llm = model

    blocked = workflow.start("session", "Inspect two files", mode="debug")
    assert blocked.data["phase"] == "blocked"
    task_id = blocked.data["task_id"]
    saved = workflow.board.get_task(task_id).mode_workflow["evidence"]
    assert [item["id"] for item in saved] == ["evidence-1"]

    resumed = workflow.retry(task_id)

    assert resumed.status == "success", resumed.message
    assert calls == ["app.py", "test_app.py", "test_app.py"]
    assert model.inputs[1]["investigation_scope"]["evidence_levels"] == {"evidence-1": 1}
    assert [item["id"] for item in model.inputs[2]["evidence"]] == ["evidence-1", "evidence-2"]


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
    assert "draft: missing" in result.message
    task_id = result.data["task_id"]
    assert f"/task resume {task_id}" in workflow.describe(workflow.active("session")).message

    resumed = workflow.retry(task_id)
    assert resumed.data["task_id"] == task_id
    assert resumed.data["phase"] == "awaiting_approval", resumed.message
    assert (gateway.root / "app.py").read_text() == "before\n"
