"""Mode contracts across the real TaskBoard, supervisor, and human inbox."""

from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
from threading import Event
from types import SimpleNamespace

import pytest

from repository_investigation_fixture import FileInvestigator, evidence_call, inspection, investigation, report_call

from mana_agent.agent.mode import CodingWorkflowDecision, ModePhase, PlanRevision, PlanStep
from mana_agent.agent.plan_service import EvidenceRequest, PlanService
from mana_agent.config.autonomous_config import AutonomousExecutionConfig
from mana_agent.execution.autonomous_controller import AutonomousExecutionController
from mana_agent.execution_supervisor.models import ExecutionState
from mana_agent.gateway.lane_coordinator import LaneCoordinator
from mana_agent.gateway.lanes import LaneId
from mana_agent.gateway.mode_workflow import ModeWorkflow, validate_step_patch
from mana_agent.human_inbox import default_human_inbox_service
from mana_agent.human_inbox.models import InboxStatus, ResponseOperation
from mana_agent.llm.types import ModelResponse


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


class Decisions(FileInvestigator):
    def __init__(self, *decisions, investigations=()):
        self.decisions = iter(decisions)
        self.inputs = []
        self.investigations = iter(investigations)
        self.native_inputs = []

    def decide_sync(self, messages, schema):
        self.inputs.append(json.loads(messages[-1].content))
        payload = next(self.decisions)
        return {"decision": payload} if schema["title"] == "ModeDecision" else payload

    def with_tools(self, tools):
        scripted = next(self.investigations, None)
        if scripted is None:
            return super().with_tools(tools)
        responses = iter(scripted)

        def generate(messages, *, continuation):
            self.native_inputs.append(list(messages))
            return ModelResponse(tool_calls=next(responses))

        return SimpleNamespace(generate_sync=generate)


def workflow_choice(*, mode="auto", effect="write"):
    return CodingWorkflowDecision(
        mode=mode, requested_effect=effect, objective="Add the handler behavior",
        reason="A repository behavior is missing", investigation_scope="app.py and its tests",
        expected_result="The handler returns a value", risk_and_uncertainty="Potential caller regression",
        verification_requirements=("Run the focused handler check",),
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
    operation = inspection("app.py")
    gateway._entry_router.llm = Decisions(operation, draft(), {"disposition": "in_scope", "reason": "Only app.py differs"})

    def propose(_prompt, **kwargs):
        task = next(row for row in workflow.board.list_tasks() if row.mode_workflow.get("mode") == "auto")
        assert task.mode_workflow["plan"]["draft"]["steps"][0]["step_id"] == "fix"
        assert task.mode_workflow["phase"] == "executing"
        assert task.owning_lane == "coding"
        assert len(task.mode_workflow["evidence"]) == 1
        assert "investigation" not in task.mode_workflow
        assert kwargs["execution_mode"] == "read_only"
        assert (Path(kwargs["coding_mode_root"]) / "app.py").read_text() == "before\n"
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
        inspection("app.py"),
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
        inspection("app.py"),
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
        inspection("app.py"),
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


@pytest.mark.parametrize("invalid_field", ["prompt", "reason", "field_id"])
def test_invalid_clarification_does_not_open_another_wait(mode_gateway, invalid_field):
    gateway, workflow = mode_gateway
    question = {"field_id": "missing_policy", "prompt": "What policy should this handler enforce?",
                "reason": "There is no existing policy and the user must choose the behavior"}
    next_question = {**question, "field_id": "limit", "prompt": "What limit applies?"}
    next_question[invalid_field] = question["field_id"] if invalid_field == "field_id" else "  "
    gateway._entry_router.llm = model = Decisions(
        {"action": "clarify", "reason": "A user choice blocks the task", "questions": [question]},
        {"action": "clarify", "reason": "Another question is needed", "questions": [next_question]},
    )
    request = "Investigate how the handler should enforce the new policy"
    waiting = workflow.start("session", request, mode="debug")
    task_id, inbox_id = waiting.data["task_id"], waiting.data["inbox_id"]
    assert waiting.next_prompt["questions"][0]["field_id"] == question["field_id"]
    assert not model.inputs[0]["evidence"]

    workflow.respond("session", inbox_id, ResponseOperation.ANSWER, frontend="cli",
                     answer={question["field_id"]: "reject"})
    blocked = workflow.wait_for_response(task_id, inbox_id, timeout=8)

    assert blocked.data["phase"] == "blocked", blocked.message
    assert blocked.next_prompt is None
    assert model.inputs[1]["request"] == request
    state = workflow.board.get_task(task_id).mode_workflow
    assert state["questions_asked"] == 1
    assert state["constraints"] == {question["field_id"]: "reject"}
    assert not state.get("inbox_id")


@pytest.mark.parametrize("mode", ["auto", "debug"])
def test_repository_clarification_resumes_exact_wait_and_preserves_evidence(mode_gateway, monkeypatch, mode):
    from mana_agent.chat_commands import CommandContext, CommandDispatcher, build_default_registry
    from mana_agent.services.execution_event_hub import get_execution_event_hub
    from mana_agent.tui.app import ManaChatApp

    gateway, workflow = mode_gateway
    reviewer = "user_fixture_mode_reviewer"
    monkeypatch.setattr("mana_agent.config.user_config.resolve_local_user_id", lambda _settings=None: reviewer)
    published = Event()
    gateway._event_sink = lambda *args, **kwargs: published.set()
    gateway._entry_router.llm = model = Decisions(
        inspection("app.py"),
        {"action": "clarify", "reason": "Separate trigger from behavior", "questions": [
            {"field_id": "trigger", "prompt": "Which input triggers it?", "reason": "Identify the trigger",
             "choices": {"blank": "Blank input", "missing": "Missing input"}, "allow_free_form": False},
            {"field_id": "behavior", "prompt": "What should happen?", "reason": "Identify the contract",
             "choices": {"reject": "Reject it", "keep": "Keep it"}, "allow_free_form": True},
        ]},
        {"action": "clarify", "reason": "The custom behavior needs a limit", "questions": [
            {"field_id": "limit", "prompt": "What is the limit?", "reason": "Completes the contract"},
        ]},
        {"action": "diagnose", "reason": "The file and answers explain the failure",
         "root_cause": "The handler still returns the old value.", "cause_evidence": ["evidence-1"]}
        if mode == "debug" else draft(),
    )
    waiting = workflow.start("session", "Investigate the handler", mode=mode, save=False, read_only=True,
                             workflow=workflow_choice(mode=mode, effect="write" if mode == "auto" else "read"),
                             reviewer_id=reviewer)
    assert waiting.status == "input_required", waiting.message
    task_id, inbox_id = waiting.data["task_id"], waiting.data["inbox_id"]
    supervisor = workflow.lane.execution_supervisor
    suspended = supervisor.get_task(task_id)
    assert suspended.state == ExecutionState.WAITING
    assert suspended.waiting_inbox_item_id == inbox_id
    assert workflow.pending_prompt("session", task_id, inbox_id) == waiting.next_prompt
    assert workflow.pending_prompt("other-session", task_id, inbox_id) is None
    screens = []
    tui = SimpleNamespace(gateway=SimpleNamespace(mode_workflow=workflow), _gateway_session_id="session",
                          screen=object(), push_screen=lambda screen, _finish: screens.append(screen))
    ManaChatApp._show_mode_prompt(tui, waiting.next_prompt)
    assert [screen.prompt for screen in screens] == [waiting.next_prompt]
    answers = {"trigger": "blank", "behavior": "Use the caller's default"}
    with pytest.raises(PermissionError):
        workflow.respond("session", inbox_id, ResponseOperation.ANSWER, frontend="tui",
                         actor_id="another-user", answer=answers)
    with pytest.raises(ValueError, match="outside its choices"):
        workflow.respond("session", inbox_id, ResponseOperation.ANSWER, frontend="tui",
                         answer={**answers, "trigger": "unlisted"})
    assert len(model.inputs) == 2
    workflow.respond("session", inbox_id, ResponseOperation.ANSWER, frontend="tui", answer=answers)

    assert published.wait(8)
    second_wait = workflow.wait_for_response(task_id, inbox_id, timeout=0.5)
    second_id = second_wait.data["inbox_id"]
    assert second_id != inbox_id
    assert second_wait.next_prompt["questions"][0]["field_id"] == "limit"
    assert workflow.pending_prompt("session", task_id, inbox_id) is None
    assert workflow.pending_prompt("session", task_id, second_id) == second_wait.next_prompt
    assert workflow.inbox.repository.get(inbox_id).response_actor_id == reviewer
    events = get_execution_event_hub().history(
        conversation_id="session",
        repository_id=workflow.board.get_task(task_id).primary_repository_id,
    )
    notice = next(event for event in events if event["type"] == "mode.input_ready" and event["task_id"] == task_id)
    assert notice["status"] == "waiting"
    assert notice["metadata"]["inbox_id"] == second_id

    dispatcher = CommandDispatcher(build_default_registry())
    context = CommandContext(frontend="cli", session_id="session", capabilities={"gateway"}, gateway=gateway)
    command = dispatcher.dispatch(f"""/inbox answer {second_id} '{{"limit":"one"}}'""", context)
    assert command.status == "success", command.message
    final = workflow.wait_for_response(task_id, second_id, timeout=8)
    assert final.data["phase"] == "completed", final.message
    assert {decision["task_id"] for decision in model.inputs} == {task_id}
    assert model.inputs[2]["constraints"] == answers
    assert model.inputs[-1]["constraints"] == {**answers, "limit": "one"}
    assert model.inputs[-1]["evidence"] == model.inputs[1]["evidence"]
    assert workflow.inbox.repository.get(second_id).status == InboxStatus.ANSWERED
    assert (gateway.root / "app.py").read_text() == "before\n"


def test_debug_research_suggestion_requires_a_workflow_clarify_decision(mode_gateway, monkeypatch):
    from mana_agent.execution.continuation_decision import ContinuationAction

    gateway, workflow = mode_gateway
    gateway.settings = SimpleNamespace(mana_coding_research_enabled=True)
    gateway._continuation_evaluator = None
    suggestion = SimpleNamespace(action=ContinuationAction.WAIT_FOR_USER,
                                 user_message="Which release should be supported?",
                                 reason="Determine the supported release")
    monkeypatch.setattr(gateway._autonomous_controller, "coding_research_runtime", lambda **_: SimpleNamespace(
        decide=lambda **_: suggestion, context=lambda _: {"findings": []},
    ))
    gateway._entry_router.llm = model = Decisions(
        inspection("app.py"),
        {"action": "research", "reason": "Check an unfamiliar dependency", "debugging_report": {
            "unresolved_error": "Unexpected return", "local_evidence": ["evidence-1: old return value"],
            "missing_external_information": "Return value for the supported release",
        }},
        {"action": "diagnose", "reason": "The request already specifies the release",
         "root_cause": "The handler returned the old value.", "cause_evidence": ["evidence-1"]},
    )

    result = workflow.start("session", "Diagnose the handler for Widget 2", mode="debug")

    assert result.status == "success", result.message
    assert model.inputs[-1]["request"] == "Diagnose the handler for Widget 2"
    suggestion_context = model.inputs[-1]["evidence"][-1]["result"]["clarification_suggestion"]
    assert suggestion_context["question"] == suggestion.user_message
    state = workflow.board.get_task(result.data["task_id"]).mode_workflow
    assert state["questions_asked"] == 0
    assert not state.get("inbox_id")


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
        inspection("app.py"),
        inspection(None, evidence_call("search", "repo_search", level=4, query="before", reason="Look everywhere")),
    )
    result = workflow.start("session", "Traceback points at app.py:1", mode="debug", llm=model)
    assert result.status == "error"
    assert "The next permitted level is 2." in result.message
    assert len(model.inputs) == 2
    assert model.inputs[0]["max_next_level"] == 1
    assert model.inputs[1]["evidence"][0]["id"] == "evidence-1"
    assert opened == ["app.py"]


def test_debug_can_find_an_error_with_no_reported_location(mode_gateway, monkeypatch):
    from mana_agent.tools import repository

    gateway, workflow = mode_gateway
    root = gateway.root
    for directory in ("runtime", "docs", ".private", ".venv"):
        (root / directory).mkdir(exist_ok=True)
    (root / "runtime" / "handler.py").write_text("raise ValueError('Unknown handler value')\n")
    (root / "docs" / "example.py").write_text("# Unknown handler value\n")
    visited = []
    iter_files = repository._iter_files

    def track(directory):
        visited.append(directory)
        yield from iter_files(directory)

    monkeypatch.setattr(repository, "_iter_files", track)
    model = Decisions(
        inspection(None, evidence_call("layout", "ls", reason="Locate the owning directory")),
        inspection(None, evidence_call("locate", "repo_search", glob="runtime/**/*.py", query="Unknown handler value")),
        {"action": "diagnose", "reason": "The error is raised in the handler",
         "root_cause": "The handler raises a ValueError.", "cause_evidence": ["evidence-2"]},
    )

    result = workflow.start("session", "Why do I get Unknown handler value?", mode="debug", llm=model)

    assert result.status == "success", result.message
    assert "Unknown handler value" in model.inputs[0]["request"]
    layout, search = model.inputs[-1]["evidence"]
    assert layout["level"] == 1
    assert layout["result"]["directories"] == ["docs", "runtime"]
    assert model.inputs[1]["evidence"][0]["result"] == layout["result"]
    assert [match["file"] for match in search["result"]["matches"]] == ["runtime/handler.py"]
    assert visited == [root.resolve() / "runtime"]


@pytest.mark.parametrize(("root_args", "expected"), [
    ({"paths": ["./"]}, ["app.py", "notes.txt"]),
    ({}, ["app.py", "notes.txt"]),
    ({"glob": "**/*.py"}, ["app.py"]),
])
def test_investigation_list_files_keeps_root_shallow(mode_gateway, monkeypatch, root_args, expected):
    from mana_agent.tools import repository

    gateway, workflow = mode_gateway
    root = gateway.root
    (root / "runtime").mkdir()
    (root / "runtime" / "handler.py").write_text("pass\n")
    (root / "notes.txt").write_text("fixture\n")
    (root / ".env").write_text("EXAMPLE=fixture\n")
    (root / "root-alias.py").symlink_to(root / "app.py")
    visited = []
    iter_files = repository._iter_files

    def track(directory):
        visited.append(directory)
        yield from iter_files(directory)

    monkeypatch.setattr(repository, "_iter_files", track)
    model = Decisions(
        inspection(None, evidence_call("root", "list_files", **root_args),
                   evidence_call("runtime", "list_files", paths=["./runtime//"])),
        {"action": "diagnose", "reason": "Compared the selected directories",
         "uncertainty": "Source was not inspected"},
    )

    result = workflow.start("session", "List files in the root and runtime", mode="debug", llm=model)

    assert result.status == "success", result.message
    listings = [item["result"]["files"] for item in model.inputs[-1]["evidence"]]
    assert listings == [expected, ["runtime/handler.py"]]
    assert visited == [root.resolve() / "runtime"]


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
        inspection("app.py"),
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


@pytest.mark.parametrize(("tool", "arguments"), [
    ("repo_search", {"query": "before"}),
    ("list_files", {"glob": "**/*.py"}),
])
def test_debug_reads_a_known_file_after_a_repository_audit(mode_gateway, tool, arguments):
    gateway, workflow = mode_gateway
    (gateway.root / "runtime").mkdir()
    (gateway.root / "runtime" / "handler.py").write_text("pass\n")
    model = Decisions(
        inspection(None, evidence_call("find", tool, level=4, initial_repository_scope="explicit_audit", **arguments)),
        inspection(None, evidence_call("match", paths=["app.py"], prior_evidence=["evidence-1"])),
        {"action": "diagnose", "reason": "The file contains the unexpected value",
         "root_cause": "The handler returns the old value.", "cause_evidence": ["evidence-2"]},
    )

    result = workflow.start("session", "Audit the repository for the old value", mode="debug", llm=model)

    assert result.status == "success", result.message
    assert all(entry["inspection_available"] for entry in model.inputs)
    assert not model.native_inputs
    search, read = model.inputs[-1]["evidence"]
    assert search["level"] == 4
    assert model.inputs[1]["evidence"][0]["result"] == search["result"]
    if tool == "list_files":
        assert search["result"]["files"] == ["app.py", "runtime/handler.py"]
    else:
        assert search["result"]["matches"][0]["file"] == "app.py"
    assert read["level"] == 1
    assert read["result"]["content"] == "before\n"


@pytest.mark.parametrize("mode", ["plan", "debug"])
def test_repository_read_batch_saves_observations_for_the_next_decision(mode_gateway, mode):
    gateway, workflow = mode_gateway
    (gateway.root / "test_app.py").write_text("assert handler()\n")
    final = draft() if mode == "plan" else {
        "action": "diagnose", "reason": "Both files explain the issue", "root_cause": "Old handler value",
        "cause_evidence": ["evidence-1", "evidence-2"],
    }
    gateway._autonomous_controller.config = AutonomousExecutionConfig(max_tool_calls=2, max_model_turns=4)
    batch = [evidence_call("first", paths=["app.py"]), evidence_call("second", paths=["test_app.py"])]
    if mode == "plan":
        model = Decisions(investigation("app.py and its test"), final, investigations=[[batch]])
    else:
        model = Decisions(inspection(None, *batch), final)

    result = workflow.start("session", "Inspect independently known code and test files",
                            mode=mode, llm=model, no_code=True, save=False)

    assert result.status == "success", result.message
    availability = "investigation_available" if mode == "plan" else "inspection_available"
    assert [call[availability] for call in model.inputs] == [True, False]
    observations = model.inputs[1]["evidence"]
    assert [(item["id"], item["result"]["content"]) for item in observations] == [
        ("evidence-1", "before\n"), ("evidence-2", "assert handler()\n"),
    ]
    state = workflow.board.get_task(result.data["task_id"]).mode_workflow
    assert (state["decision_count"], state["tool_operations"]) == (3 if mode == "plan" else 2, 2)
    if mode == "plan":
        assert state["investigation"]["status"] == "completed"
        assert state["investigation"]["report"]["evidence_ids"] == ["evidence-1", "evidence-2"]
    else:
        assert "investigation" not in state
    persisted = state["evidence"]
    assert len(persisted) == 2
    assert all("execution" in item["result"] for item in persisted)
    assert all("execution" not in item["result"] and "fingerprint" not in item for item in observations)


def test_diagnosis_from_reported_evidence_skips_investigation(mode_gateway):
    _, workflow = mode_gateway
    model = Decisions({"action": "diagnose", "reason": "The supplied traceback identifies the exception",
                       "root_cause": "The handler received a missing key", "uncertainty": "Source was not inspected"})
    model.with_tools = lambda _: pytest.fail("No coding handoff was selected")

    result = workflow.start("session", "Explain the reported KeyError", mode="debug", llm=model)

    assert result.status == "success", result.message
    state = workflow.board.get_task(result.data["task_id"]).mode_workflow
    assert "investigation" not in state
    assert state["decision_count"] == 1


def test_coding_uses_its_last_shared_turn_to_report_saved_observations(mode_gateway):
    gateway, workflow = mode_gateway
    gateway._autonomous_controller.config = AutonomousExecutionConfig(max_model_turns=3)
    model = Decisions(
        inspection("app.py"),
        inspection(None, evidence_call("second", paths=["app.py"], start_line=2)),
        {"action": "diagnose", "reason": "Use the available source", "root_cause": "Old handler value",
         "cause_evidence": ["evidence-1"], "uncertainty": "The test behavior remains unverified."},
    )

    result = workflow.start("session", "Investigate app.py", mode="debug", llm=model)

    assert result.data["phase"] == "completed"
    assert "The test behavior remains unverified." in result.message
    state = workflow.board.get_task(result.data["task_id"]).mode_workflow
    assert [turn["inspection_available"] for turn in model.inputs] == [True, True, False]
    assert (state["decision_count"], state["tool_operations"]) == (3, 2)
    assert [row["id"] for row in state["evidence"]] == ["evidence-1", "evidence-2"]


def test_coding_inspection_and_diagnosis_record_provider_usage_for_one_task(mode_gateway, monkeypatch):
    from mana_agent.llm.capabilities import ModelProfile, observation
    from mana_agent.llm.engine import NativeLLMEngine

    gateway, workflow = mode_gateway
    admitted, recorded = [], []

    def admit(_segments, *, task_id, **kwargs):
        admitted.append(task_id)
        return f"reservation-{len(admitted)}", None

    def record(reservation, *, task_id, usage, **kwargs):
        recorded.append((reservation, task_id, usage["input_tokens"], usage["output_tokens"]))

    gateway._stack = SimpleNamespace(context_cost_governor=SimpleNamespace(
        enabled=True, scoped_execution_identity=lambda **_: nullcontext(),
        before_model_call=admit, record_model_call=record,
    ))
    decisions = iter([
        inspection("app.py"),
        {"action": "diagnose", "reason": "The source explains the behavior",
         "root_cause": "Old handler value", "cause_evidence": ["evidence-1"]},
    ])

    class Adapter:
        async def generate(self, request):
            assert request.response_schema
            return ModelResponse(content=json.dumps({"decision": next(decisions)}),
                                 usage={"input_tokens": 70, "output_tokens": 10})

    facts = {name: observation(value, source="provider", confidence=1) for name, value in {
        "input_text": True, "output_text": True, "native_structured_output": True,
        "context_window": 65536, "max_output_tokens": 1024,
    }.items()}
    profile = ModelProfile(provider="openai", model_id="test-model", facts=facts)
    model = NativeLLMEngine(provider="openai", model="test-model", api_key="fixture",
                            base_url="https://example.invalid", endpoint="responses", adapter=Adapter())
    monkeypatch.setattr(model, "profile", lambda *_: profile)

    result = workflow.start("session", "Explain app.py", mode="debug", llm=model)

    assert result.status == "success", result.message
    task_id = result.data["task_id"]
    assert admitted == [task_id, task_id]
    assert recorded == [("reservation-1", task_id, 70, 10), ("reservation-2", task_id, 70, 10)]
    assert result.data["repository_observations"]["workflow_model_decisions"] == 2
    assert result.data["repository_observations"]["workflow_tool_operations"] == 1


def test_coding_inspection_can_exceed_the_former_investigator_caps(mode_gateway):
    gateway, workflow = mode_gateway
    decisions = []
    for batch in range(9):
        calls = []
        for index in range(3):
            path = f"dependency_{batch}_{index}.py"
            (gateway.root / path).write_text(f"value = {batch * 3 + index}\n")
            calls.append(evidence_call(path, paths=[path]))
        decisions.append(inspection(None, *calls))
    decisions.append({"action": "diagnose", "reason": "The last dependency is the cause",
                      "root_cause": "The dependency returns the old value", "cause_evidence": ["evidence-27"]})
    model = Decisions(*decisions)
    model.with_tools = lambda *_: pytest.fail("Coding must not start a separate investigator")

    result = workflow.start("session", "Trace the dependency chain", mode="debug", llm=model)

    assert result.status == "success", result.message
    state = workflow.board.get_task(result.data["task_id"]).mode_workflow
    assert (state["decision_count"], state["tool_operations"]) == (10, 27)
    assert len(state["evidence"]) == 27


def test_repeated_repository_results_stop_inspection_and_leave_a_final_decision(mode_gateway):
    gateway, workflow = mode_gateway
    gateway._autonomous_controller.config = AutonomousExecutionConfig(max_no_progress_cycles=2)
    queries = [
        inspection(None, evidence_call(query, "repo_search", paths=["app.py"], query=query))
        for query in ("before", "bef", "befo")
    ]
    model = Decisions(*queries, {"action": "block", "reason": "The caller is still unknown"})

    result = workflow.start("session", "Find the caller of the handler", mode="debug", llm=model)

    assert result.data["phase"] == "blocked"
    assert "Repository reads stopped yielding new evidence" in result.message
    assert result.data["repository_observations"]["evidence_ids"] == ["evidence-1", "evidence-2", "evidence-3"]
    assert [turn["inspection_available"] for turn in model.inputs] == [True, True, True, False]


def test_cancel_during_native_response_discards_late_tools(mode_gateway):
    _, workflow = mode_gateway
    model = Decisions(investigation("app.py"))

    def with_tools(_):
        def generate(messages, *, continuation):
            task_id = model.inputs[0]["task_id"]
            workflow.lane.cancel_tree(task_id)
            workflow.note_cancellation(task_id)
            return ModelResponse(tool_calls=[evidence_call("late", paths=["app.py"])])
        return SimpleNamespace(generate_sync=generate)

    model.with_tools = with_tools
    result = workflow.start("session", "Investigate app.py", mode="plan", llm=model)

    assert result.data["phase"] == "cancelled"
    state = workflow.board.get_task(result.data["task_id"]).mode_workflow
    assert state["investigation"]["status"] == "cancelled"
    assert state["evidence"] == []


def test_cancel_during_coding_decision_discards_the_read_batch(mode_gateway, monkeypatch):
    _, workflow = mode_gateway
    monkeypatch.setattr(PlanService, "inspect", lambda *_: pytest.fail("A retired attempt cannot read files"))

    class CancelledDecision(Decisions):
        def decide_sync(self, messages, schema):
            decision = super().decide_sync(messages, schema)
            task_id = self.inputs[-1]["task_id"]
            workflow.lane.cancel_tree(task_id)
            workflow.note_cancellation(task_id)
            return decision

    result = workflow.start("session", "Inspect app.py", mode="debug", llm=CancelledDecision(inspection("app.py")))

    assert result.data["phase"] == "cancelled"
    state = workflow.board.get_task(result.data["task_id"]).mode_workflow
    assert state["decision_count"] == 1
    assert state.get("tool_operations", 0) == 0
    assert not state["evidence"]


@pytest.mark.parametrize(("second", "tool_budget", "detail"), [
    (evidence_call("second", paths=["app.py"], reason="A different reason"), 2, "Choose distinct inspection operations"),
    (evidence_call("second", paths=["test_app.py"], level=2, prior_evidence=["evidence-1"]), 2,
     "The next permitted level is 1"),
    (evidence_call("second", "list_files", paths=["tests/**/*.py"]), 2,
     "Discovery paths must name files or directories"),
    (evidence_call("second", "list_files", paths=["/"]), 2,
     "paths.0: Discovery paths must name files or directories"),
    (evidence_call("second", "list_files", paths=["runtime/../docs"]), 2,
     "paths.0: Discovery paths must name files or directories"),
    (evidence_call("second", "list_files", paths=["./.env/"]), 2,
     "paths.0: use an unprotected, canonical repository-relative path"),
    (evidence_call("second", "list_files", level=4), 2, "The next permitted level is 1"),
    (evidence_call("second", "list_files", level=4, reason=" "), 2, "Provide a reason for the requested level-4 scope"),
    (evidence_call("second", "repo_search", paths=["."], query="before"), 2,
     "Use list_files to list top-level files or select a narrower path."),
    (evidence_call("second", "repo_search", query="before"), 2, "Supply literal paths or a targeted glob"),
    (evidence_call("second", "repo_search", query="before", glob="**/*.py"), 2,
     "Repository-wide discovery requires an authorized level 4"),
    (evidence_call("second", "ls", paths=["docs"]), 2, "arguments did not match the registered schema"),
    (evidence_call("second", "repo_search", level=4, query="before", initial_repository_scope="explicit_audit"), 2,
     "arguments did not match the registered schema"),
    (evidence_call("second", paths=[".env"]), 2, "unprotected, canonical repository-relative path"),
    (evidence_call("second", "apply_patch", patch="unauthorized"), 2, "outside the registered read-only set"),
])
def test_investigation_batch_rejects_all_reads_if_any_selection_is_invalid(
    mode_gateway, monkeypatch, second, tool_budget, detail,
):
    gateway, workflow = mode_gateway
    gateway._autonomous_controller.config = AutonomousExecutionConfig(max_tool_calls=tool_budget)
    calls = []
    monkeypatch.setattr(PlanService, "inspect", lambda _, request: calls.append(request))
    model = Decisions(investigation("app.py and its test"),
                      investigations=[[[evidence_call("first", paths=["app.py"]), second]]])

    result = workflow.start("session", "Inspect two files", mode="plan", llm=model)

    assert result.data["phase"] == "blocked"
    assert detail in result.message
    assert not calls


@pytest.mark.parametrize("mode", ["plan", "debug"])
def test_over_budget_read_batch_returns_control_without_a_partial_dispatch(mode_gateway, monkeypatch, mode):
    gateway, workflow = mode_gateway
    gateway._autonomous_controller.config = AutonomousExecutionConfig(max_tool_calls=1)
    monkeypatch.setattr(PlanService, "inspect", lambda *_: pytest.fail("The batch cannot be partially dispatched"))
    calls = [evidence_call("first", paths=["app.py"]), evidence_call("second", paths=["test_app.py"])]
    conclusion = {"action": "block", "reason": "Need both files to compare the behavior"}
    if mode == "plan":
        model = Decisions(investigation("Compare two files"), conclusion, investigations=[[calls]])
    else:
        model = Decisions(inspection(None, *calls), conclusion)

    result = workflow.start("session", "Compare app.py with test_app.py", mode=mode, llm=model)

    assert len(model.inputs) == 2
    assert "exceeds the remaining shared tool budget" in result.message
    assert not result.data["repository_observations"]["workflow_tool_operations"]


@pytest.mark.parametrize(("first", "second"), [
    ({"paths": ["runtime"]}, {"paths": ["./runtime/"]}),
    ({}, {"paths": ["."]}),
])
def test_investigation_list_files_deduplicates_aliases_before_dispatch(mode_gateway, monkeypatch, first, second):
    _, workflow = mode_gateway
    monkeypatch.setattr(PlanService, "inspect", lambda *_: pytest.fail("Duplicate scope must stop before dispatch"))
    model = Decisions(inspection(None, evidence_call("first", "list_files", **first),
                                 evidence_call("second", "list_files", **second)))

    result = workflow.start("session", "List the files", mode="debug", llm=model)

    assert result.data["phase"] == "blocked"
    assert "Choose distinct inspection operations" in result.message


def test_investigation_list_files_distinguishes_audit_from_shallow_scope():
    from mana_agent.agent.plan_service import FileListEvidence

    shallow = FileListEvidence(tool="list_files", level=1, reason="List top-level files")
    audit = FileListEvidence(tool="list_files", level=4, reason="Audit repository files")

    assert shallow.fingerprint != audit.fingerprint


def test_investigation_batch_resume_uses_completed_observations(mode_gateway, monkeypatch):
    from mana_agent.multi_agent.agents.tool_runner_agent import ToolRunnerAgent

    gateway, workflow = mode_gateway
    (gateway.root / "test_app.py").write_text("assert handler()\n")
    calls = []
    real_dispatch = ToolRunnerAgent.execute

    def dispatch(runner, request, implementation):
        path = request.arguments["paths"][0]
        calls.append(path)
        if len(calls) == 2:
            raise PermissionError("Tool dispatch was interrupted")
        return real_dispatch(runner, request, implementation)

    monkeypatch.setattr(ToolRunnerAgent, "execute", dispatch)
    model = Decisions(
        inspection("app.py", evidence_call("second", paths=["test_app.py"])),
        inspection("test_app.py"),
        {"action": "diagnose", "reason": "Both files explain the issue", "root_cause": "Old handler value",
         "cause_evidence": ["evidence-1", "evidence-2"]},
    )
    gateway._entry_router.llm = model

    request = "Inspect the old handler in app.py and compare its behavior with test_app.py"
    session = gateway._workspaces.store.get_session("session")
    reservation = workflow.lane.reserve(normalized_intent=request, lane_id=LaneId.RESEARCH,
                                        session_id="session", workspace_id=session.workspace_id,
                                        repository_id=session.primary_repository_id)
    workflow.lane.start(reservation)
    blocked = workflow.start("session", request, task_id=reservation.execution.task_id, mode="debug",
                             workflow=workflow_choice(mode="debug", effect="read"))
    assert blocked.data["phase"] == "blocked"
    task_id = blocked.data["task_id"]
    saved = workflow.board.get_task(task_id).mode_workflow
    assert [item["id"] for item in saved["evidence"]] == ["evidence-1"]
    # Reproduce a checkpoint saved before the original request was handed to the mode model.
    for key in ("request", "resolved_request", "conversation_context"):
        saved.pop(key)
    saved["goal"] = saved["coding_decision"]["objective"] = "H"
    saved["coding_decision"]["clarification"] = "What should I investigate?"
    workflow.board.update_task(task_id, mode_workflow=saved)

    resumed = workflow.retry(task_id)

    assert resumed.status == "success", resumed.message
    assert calls == ["app.py", "test_app.py", "test_app.py"]
    assert model.inputs[1]["evidence_levels"] == {"evidence-1": 1}
    assert [item["id"] for item in model.inputs[2]["evidence"]] == ["evidence-1", "evidence-2"]
    assert model.inputs[1]["request"] == request
    assert "clarification" not in model.inputs[1]["coding_workflow"]
    assert resumed.data["task_id"] == task_id
    state = workflow.board.get_task(task_id).mode_workflow
    assert (state["decision_count"], state["tool_operations"]) == (3, 3)


def test_programmatic_investigation_preserves_provider_continuation_and_callers(mode_gateway, monkeypatch):
    from mana_agent.llm.capabilities import ModelProfile, observation
    from mana_agent.llm.engine import NativeLLMEngine
    from mana_agent.llm.providers.openai import OpenAIProvider

    _, workflow = mode_gateway
    provider = OpenAIProvider(provider="openai", api_key="fixture", base_url="https://example.invalid",
                              endpoint="responses")
    program = {"type": "program", "id": "program-1", "code": "/* fixture only */"}
    caller = {"type": "program", "call_id": "program-1"}
    requests = []

    class Adapter:
        async def generate(self, request):
            requests.append(request)
            if request.response_schema:
                final = draft()
                return ModelResponse(content=json.dumps({"decision": investigation("app.py") if len(requests) == 1 else final}))
            assert request.tools[0]["type"] == "programmatic_tool_calling"
            assert next(tool for tool in request.tools if tool.get("name") == "report_investigation")["allowed_callers"] == ["direct"]
            if len(requests) == 2:
                output = [program, {"type": "function_call", "call_id": "read-1", "name": "read_file", "caller": caller,
                                    "arguments": json.dumps(evidence_call("read-1", paths=["app.py"]).args)}]
            else:
                sent = provider.encode(request)["input"]
                result = sent[-1]
                assert request.continuation.response_id == "response-2"
                assert any(item.get("id") == program["id"] for item in sent)
                assert (result["type"], result["call_id"], result["caller"]) == ("function_call_output", "read-1", caller)
                assert json.loads(result["output"])["result"]["content"] == "before\n"
                report = report_call("evidence-1")
                output = [{"type": "function_call", "call_id": report.id, "name": report.name,
                           "arguments": json.dumps(report.args)}]
            return provider.normalize({"id": f"response-{len(requests)}", "status": "completed", "output": output}, request)

    facts = {name: observation(value, source="provider", confidence=1) for name, value in {
        "input_text": True, "output_text": True, "tool_calling": True, "native_structured_output": True,
        "programmatic_tool_calling": True, "context_window": 65536, "max_output_tokens": 1024,
    }.items()}
    profile = ModelProfile(provider="openai", model_id="test-model", facts=facts)
    model = NativeLLMEngine(provider="openai", model="test-model", api_key="fixture",
                            base_url="https://example.invalid", endpoint="responses", adapter=Adapter())
    monkeypatch.setattr(model, "profile", lambda *_: profile)
    result = workflow.start("session", "Investigate app.py", mode="plan", llm=model, no_code=True, save=False)

    assert result.status == "success", result.message
    assert len(requests) == 4
    assert [bool(req.response_schema) for req in requests] == [True, False, False, True]


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
        inspection("app.py"),
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
