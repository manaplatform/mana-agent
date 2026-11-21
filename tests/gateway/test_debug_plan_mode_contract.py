"""Mode contracts across the real TaskBoard, supervisor, and human inbox."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
from threading import Event
from types import SimpleNamespace

import pytest

from repository_investigation_fixture import evidence_call, inspection, investigation
from mana_agent.agent.mode import CodingWorkflowDecision, ModePhase, PlanRevision, PlanStep
from mana_agent.agent.plan_service import EvidenceRequest, PlanService
from mana_agent.config.autonomous_config import AutonomousExecutionConfig
from mana_agent.execution.autonomous_controller import AutonomousExecutionController
from mana_agent.execution_supervisor.models import ActionRequestState, ExecutionState, utc_now
from mana_agent.gateway.lane_coordinator import LaneCoordinator
from mana_agent.gateway.lanes import LaneId, LaneTaskState
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


class Decisions:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)
        self.inputs = []

    def decide_sync(self, messages, schema):
        self.inputs.append(json.loads(messages[-1].content))
        payload = next(self.decisions)
        return {"decision": payload} if schema["title"] == "ModeDecision" else payload

    def with_tools(self, _tools):
        pytest.fail("Repository inspection must not start a second model loop")


def workflow_choice(*, mode="auto", effect="write"):
    return CodingWorkflowDecision(
        mode=mode, requested_effect=effect, objective="Add the handler behavior",
        reason="A repository behavior is missing", investigation_scope="app.py and its tests",
        expected_result="The handler returns a value", risk_and_uncertainty="Potential caller regression",
        verification_requirements=("Run the focused handler check",),
    )


@pytest.mark.parametrize(("mode", "first_check"), [
    pytest.param("auto", "failure", id="auto"),
    pytest.param("debug", "failure", id="debug"),
    pytest.param("debug", "setup_failure", id="debug-setup-fails"),
    pytest.param("debug", "missing_exit", id="debug-check-missing-exit"),
    pytest.param("debug", "mutates_source", id="debug-check-mutates-source"),
])
def test_adaptive_coding_edits_and_repairs_on_one_assignment(mode_gateway, monkeypatch, mode, first_check):
    from mana_agent.coding.facade import UnifiedCodingAgentFacade
    from mana_agent.coding.internal_engine import InternalCodingEngine
    from mana_agent.coding.models import CODING_RESULT_TOOL, ISOLATED_CHECK_TOOL

    gateway, workflow = mode_gateway
    filename = "src/handler.py" if mode == "auto" else "app.py"
    if mode == "auto":
        (gateway.root / "src").mkdir()
        (gateway.root / "app.py").rename(gateway.root / filename)
    (gateway.root / "other.py").write_text("other = 0\n")
    gateway._entry_router.llm = Decisions()  # No ModeDecision draft or investigator.
    verify_calls = []
    check_changes_source = first_check == "mutates_source"
    check_has_setup = first_check in {"setup_failure", "missing_exit"}
    failed_exit_code = None if check_has_setup else 1

    async def check(root, command, *, task_id):
        verify_calls.append((task_id, command.argv))
        passed = (root / filename).read_text() == "fixed\n"
        if not passed and check_changes_source:
            (root / "other.py").write_text("other = 2\n")
        outcome = {"passed": passed, "exit_code": 0 if passed else failed_exit_code,
                   "output": "passed" if passed else "check failed"}
        if check_has_setup:
            outcome["setup_receipts"] = [{"exit_code": 1 if not passed and first_check == "setup_failure" else 0}]
        return outcome

    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_checkout_verification", check)
    verify = (ISOLATED_CHECK_TOOL, {"argv": ["python", "check.py"],
                                  **({"setup_argv": [["python", "setup.py"]]} if check_has_setup else {})})
    selections = []
    if mode == "auto":
        selections.append(("repo_search", {"query": "before", "paths": ["src"],
            "evidence_need": "Locate the handler to edit."}))
    selections.extend([
        ("read_file", {"path": filename, "evidence_need": "Confirm the handler body before editing it."}),
        ("write_file", {"path": filename, "content": "first attempt\n"}),
        verify,
        ("read_file", {"path": filename, "evidence_need": "Confirm the emitted value before repairing the failed check."}),
        ("write_file", {"path": filename, "content": "fixed\n"}),
    ])
    if mode == "auto":
        selections.append(("write_file", {"path": "other.py", "content": "other = 1\n"}))
    selections.extend([verify, (CODING_RESULT_TOOL, {"disposition": "complete", "summary": "Fixed and checked."})])

    class Model:
        index = 0

        def with_tools(self, tools):
            self.tools = {tool["function"]["name"] for tool in tools}
            return self

        async def generate(self, messages, *, continuation):
            if self.index:
                observation = json.loads(next(message.content for message in reversed(messages) if message.role == "tool"))
                previous = selections[self.index - 1][0]
                if previous == "repo_search":
                    assert any(match["file"] == filename for match in observation["matches"])
                if previous == ISOLATED_CHECK_TOOL:
                    is_first_check = self.index == selections.index(verify) + 1
                    assert observation.get("exit_code") == (failed_exit_code if is_first_check else 0), observation
                if previous == "read_file" and self.index > selections.index(verify):
                    assert observation["content"] == "first attempt\n"
            name, args = selections[self.index]
            self.index += 1
            assert name in self.tools
            return SimpleNamespace(tool_calls=[{"id": f"call-{self.index}", "name": name, "args": args}])

    gateway._coding_agent = UnifiedCodingAgentFacade(repo_root=gateway.root, default_mode="internal",
        internal_engine=InternalCodingEngine(llm_client=Model(), programmatic_tool_calling_enabled=False))
    request = "Integrate the replacement handler for the old behavior" if mode == "auto" else "Fix app.py"
    result = workflow.start("session", request, mode=mode, repair=True, workflow=workflow_choice(mode=mode))

    blocked = first_check in {"missing_exit", "mutates_source"}
    assert result.status == ("error" if blocked else "success"), result.message
    assert (gateway.root / filename).read_text() == "fixed\n"
    expected_files = [filename, "other.py"] if mode == "auto" or check_changes_source else [filename]
    assert sorted(result.data["changed_files"]) == sorted(expected_files)
    parent = workflow.board.get_task(result.data["task_id"])
    children = workflow.board.list_tasks(parent_task_id=parent.task_id)
    assert len(children) == 1
    expected_exits = [failed_exit_code] if check_changes_source else [failed_exit_code, 0]
    assert [task_id for task_id, _ in verify_calls] == [children[0].task_id] * len(expected_exits)
    assert not parent.mode_workflow.get("plan")
    assert not gateway._entry_router.llm.inputs
    checks = result.data["verification"]
    assert [check["exit_code"] for check in checks] == expected_exits
    assert len({check["attempt_id"] for check in checks}) == 1
    assert children[0].evidence_refs
    actions = workflow.lane.execution_supervisor.store.actions_for_task(children[0].task_id)
    failed_check, = [action for action in actions if action.tool_name == ISOLATED_CHECK_TOOL
                     and action.verification_state["status"] == "failed"]
    if check_changes_source:
        assert failed_check.request_state == ActionRequestState.OUTCOME_UNKNOWN
        assert "reconcile its durable receipt before retry" in result.message
    elif first_check == "missing_exit":
        assert failed_check.request_state == ActionRequestState.FAILED
        assert "supervisor_completion_gate" in result.message
    else:
        assert failed_check.request_state == ActionRequestState.RECONCILED
        assert failed_check.verification_state["exit_code"] == 1


def test_adaptive_coding_reconciles_legacy_mutation_paths(mode_gateway, monkeypatch):
    from mana_agent.repository_evidence.tool_queue import ToolQueueStore

    _, workflow = mode_gateway
    receipts = [
        {"mutations": [{"status": "applied", "path": os.path.join("pkg", "handler.py"),
                        "old_path": os.path.join("src", "original.py"),
                        "before_sha256": "original", "after_sha256": "original"}]},
        {"mutations": [{"status": "applied", "path": "pkg/handler.py",
                        "before_sha256": "original", "after_sha256": "updated"}]},
    ]
    monkeypatch.setattr(ToolQueueStore, "operation_receipts", lambda _store, _task_id: receipts)

    assert workflow._adaptive_mutations("child") == {"pkg/handler.py": "updated", "src/original.py": "missing"}


@pytest.mark.parametrize("rejected_kind", [
    "oversized_group", "missing_evidence_need", "unoffered_tool", "unscoped_search", "shell_schema",
])
def test_adaptive_coding_preserves_patch_budget_after_evidence_rejection(mode_gateway, monkeypatch, rejected_kind):
    from mana_agent.coding.facade import UnifiedCodingAgentFacade
    from mana_agent.coding.internal_engine import InternalCodingEngine
    from mana_agent.coding.models import CODING_RESULT_TOOL, ISOLATED_CHECK_TOOL
    from mana_agent.llm.errors import InvalidResponseError

    gateway, workflow = mode_gateway
    gateway._entry_router.llm = Decisions()
    (gateway.root / "caller.py").write_text("from app import handler\n")
    patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-before\n+after\n*** End Patch"

    def call(identity, name, args):
        return {"id": identity, "name": name, "args": args}

    if rejected_kind == "oversized_group":
        rejected = [call("rejected-search", "batch_search_files", {"patterns": []}),
                    call("rejected-list", "batch_list_files", {"queries": []})]
        expected_error = "BUDGET_EXCEEDED"
    elif rejected_kind == "missing_evidence_need":
        rejected = [call("rejected-read", "read_file", {"path": "caller.py"})]
        expected_error = "EVIDENCE_NEED_REQUIRED"
    elif rejected_kind == "unscoped_search":
        rejected = [call("rejected-search", "repo_search", {"query": "caller",
            "evidence_need": "Find the caller before editing app.py."})]
        expected_error = "SCOPED_EVIDENCE_REQUIRED"
    else:
        rejected = []
        expected_error = "TOOL_NOT_OFFERED"

    selections = [
        [call("initial-read", "read_file", {"path": "app.py", "evidence_need": "Identify the value to patch."})],
        rejected,
        [call("focused-read", "read_file", {"path": "caller.py", "evidence_need": "Confirm the caller contract before changing app.py."})],
        [call("patch", "apply_patch", {"patch": patch})],
        [call("check", ISOLATED_CHECK_TOOL, {"argv": ["python", "check.py"]})],
        [call("done", CODING_RESULT_TOOL, {"disposition": "complete", "summary": "Applied and verified the fix."})],
    ]
    checks = []

    async def check(root, command, *, task_id):
        assert (root / "app.py").read_text() == "after\n"
        checks.append(task_id)
        return {"passed": True, "exit_code": 0, "output": "pass"}

    class Model:
        turn = 0

        def with_tools(self, tools):
            self.offered = {tool["function"]["name"] for tool in tools}
            return self

        async def generate(self, messages, *, continuation):
            if self.turn == 1:
                assert {"read_file", "repo_search", "apply_patch", ISOLATED_CHECK_TOOL} <= self.offered
                assert "investigate_repository" not in self.offered
                assert "3 tool calls" in messages[-1].content
                if rejected_kind == "unoffered_tool":
                    self.turn += 1
                    raise InvalidResponseError("Unregistered tool", reason_code="tool_not_offered")
                if rejected_kind == "shell_schema":
                    self.turn += 1
                    raise InvalidResponseError("Invalid shell arguments", public_detail="Shell arguments failed validation.",
                                               reason_code="tool_arguments_invalid")
            elif self.turn == 2:
                outputs = {message.tool_call_id: json.loads(message.content) for message in messages if message.role == "tool"}
                assert outputs["initial-read"]["content"] == "before\n"
                if rejected_kind == "unoffered_tool":
                    assert "outside this turn's allowance" in messages[-2].content
                if rejected_kind == "shell_schema":
                    assert "Shell arguments failed validation." in messages[-2].content
                    assert "arguments declared in each offered tool schema" in messages[-2].content
                for rejected_call in rejected:
                    error = outputs[rejected_call["id"]]
                    assert error["error_code"] == expected_error
                    assert error["executed"] == 0 and error["optional_remaining"] == 1
            elif self.turn == 3:
                output = json.loads(messages[-2].content)
                assert output["content"] == "from app import handler\n"
                assert "read_file" not in self.offered
            elif self.turn == 5:
                assert self.offered == {CODING_RESULT_TOOL}
            selected = selections[self.turn]
            if self.turn != 1:
                assert {item["name"] for item in selected} <= self.offered
            self.turn += 1
            return SimpleNamespace(tool_calls=selected)

    model = Model()
    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_checkout_verification", check)
    gateway._coding_agent = UnifiedCodingAgentFacade(repo_root=gateway.root, default_mode="internal",
        internal_engine=InternalCodingEngine(llm_client=model, max_tool_calls=4, programmatic_tool_calling_enabled=False))
    result = workflow.start("session", "Fix app.py", mode="auto", repair=True, workflow=workflow_choice())

    assert result.status == "success", result.message
    assert result.data["changed_files"] == ["app.py"]
    parent = workflow.board.get_task(result.data["task_id"])
    child, = workflow.board.list_tasks(parent_task_id=parent.task_id)
    assert checks == [child.task_id]
    assert model.turn == len(selections)
    assert parent.mode_workflow["tool_operations"] == 4


@pytest.mark.parametrize(("search_scope", "scope_error"), [({}, "SCOPED_EVIDENCE_REQUIRED"),
    ({"paths": ["src/*.py"]}, "INVALID_SEARCH_SCOPE")])
def test_adaptive_coding_corrects_shell_and_search_scope_separately(mode_gateway, monkeypatch, search_scope, scope_error):
    from mana_agent.coding.facade import UnifiedCodingAgentFacade
    from mana_agent.coding.internal_engine import InternalCodingEngine
    from mana_agent.coding.models import CODING_RESULT_TOOL, ISOLATED_CHECK_TOOL

    gateway, workflow = mode_gateway
    gateway._entry_router.llm = Decisions()
    (gateway.root / "src").mkdir()
    (gateway.root / "src/handler.py").write_text("before\n")
    search = {"query": "before", "evidence_need": "Find the value to change in the handler."}
    scoped_search = {**search, "paths": ["src"]}
    selections = [
        [("repo_search", scoped_search), ("run_script_once", {"script": "ls"})],
        [("repo_search", {**search, **search_scope})],
        [("ls", {"evidence_need": "Locate a top-level directory for the handler search."})],
        [("repo_search", scoped_search)],
        [("write_file", {"path": "src/handler.py", "content": "after\n"})],
        [(ISOLATED_CHECK_TOOL, {"argv": ["python", "src/handler.py"]})],
        [(CODING_RESULT_TOOL, {"disposition": "complete", "summary": "Changed and checked the handler."})],
    ]

    async def check(root, command, *, task_id):
        assert (root / "src/handler.py").read_text() == "after\n"
        return {"passed": True, "exit_code": 0, "output": "passed"}

    class Model:
        turn = 0

        def with_tools(self, tools):
            self.offered = {tool["function"]["name"] for tool in tools}
            return self

        async def generate(self, messages, *, continuation):
            outputs = {message.tool_call_id: json.loads(message.content) for message in messages if message.role == "tool"}
            if self.turn == 1:
                assert outputs["0-0"]["rejected_call_id"] == "0-1"
                assert outputs["0-0"]["executed"] == 0 and "resubmit" in outputs["0-0"]["error"]
                assert outputs["0-1"]["error_code"] == "USE_REPOSITORY_TOOL"
            if self.turn == 2:
                assert outputs["1-0"]["error_code"] == scope_error
                assert outputs["1-0"]["executed"] == 0 and "Use ls" in outputs["1-0"]["error"]
                if search_scope:
                    assert "must name literal files or directories" in outputs["1-0"]["error"]
            if self.turn == 3:
                assert "src" in outputs["2-0"]["directories"]
            if self.turn == 4:
                assert outputs["3-0"]["matches"][0]["file"] == "src/handler.py"
            calls = [{"id": f"{self.turn}-{index}", "name": name, "args": args}
                     for index, (name, args) in enumerate(selections[self.turn])]
            assert all(call["name"] in self.offered for call in calls)
            self.turn += 1
            return SimpleNamespace(tool_calls=calls)

    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_checkout_verification", check)
    gateway._coding_agent = UnifiedCodingAgentFacade(repo_root=gateway.root, default_mode="internal",
        internal_engine=InternalCodingEngine(llm_client=Model(), max_tool_calls=4, programmatic_tool_calling_enabled=False))
    result = workflow.start("session", "Fix the handler", mode="auto", repair=True, workflow=workflow_choice())

    assert result.status == "success", result.message
    assert result.data["changed_files"] == ["src/handler.py"]
    parent = workflow.board.get_task(result.data["task_id"])
    assert parent.mode_workflow["tool_operations"] == 4


def test_adaptive_coding_recovers_from_shell_inspection_and_checks_build(mode_gateway, monkeypatch):
    from mana_agent.coding.adaptive import SHELL_TOOL_NAMES
    from mana_agent.coding.facade import UnifiedCodingAgentFacade
    from mana_agent.coding.internal_engine import InternalCodingEngine
    from mana_agent.coding.models import CODING_RESULT_TOOL, ISOLATED_CHECK_TOOL
    from mana_agent.llm.errors import InvalidResponseError

    gateway, workflow = mode_gateway
    subprocess.run(["git", "init", "--quiet"], cwd=gateway.root, check=True)
    gateway._entry_router.llm = Decisions()
    ls = ("ls", {"evidence_need": "Locate the source and build files."})
    verify = (ISOLATED_CHECK_TOOL, {"argv": ["python", "app.py"]})
    edit = ("write_file", {"path": "app.py", "content": "after\n"})
    selections = [
        [("run_command", {"cmd": "ls -la && git status --short"}), ls],
        None,
        [ls, ("git_status", {})],
        [("read_file", {"path": "app.py", "evidence_need": "Confirm the initial behavior."})],
        [("shell", {"action": {"commands": ["ls -la && python app.py"]}, "cwd": "."}), edit],
        [edit, verify],
        [(CODING_RESULT_TOOL, {"disposition": "complete", "summary": "Repaired and checked app.py."})],
    ]
    checks = []

    async def check(root, command, *, task_id):
        assert command.argv == ("python", "app.py")
        contents = (root / "app.py").read_text()
        checks.append(contents)
        return {"passed": contents == "after\n", "exit_code": int(contents != "after\n"), "output": contents}

    class Model:
        turn = 0

        def with_tools(self, tools):
            self.offered = {tool["function"]["name"]: tool["function"] for tool in tools}
            return self

        async def generate(self, messages, *, continuation):
            turn = self.turn
            self.turn += 1
            outputs = {message.tool_call_id: json.loads(message.content) for message in messages if message.role == "tool"}
            if turn in {0, 3, 4}:
                assert SHELL_TOOL_NAMES <= self.offered.keys()
            if turn in {1, 2, 5}:
                assert SHELL_TOOL_NAMES.isdisjoint(self.offered)
                assert {"ls", "git_status", "read_file", ISOLATED_CHECK_TOOL} <= self.offered.keys()
                assert set(self.offered["ls"]["parameters"]["properties"]) == {"evidence_need"}
            if turn == 1:
                assert outputs["0-1"]["executed"] == 0 and outputs["0-1"]["rejected_call_id"] == "0-0"
                assert "call ls with only evidence_need" in outputs["0-0"]["error"]
                raise InvalidResponseError("Unexpected ls argument", reason_code="tool_arguments_invalid",
                    public_detail="Provider tool arguments violate the registered schema for ls")
            if turn == 2:
                assert "registered schema for ls" in messages[-2].content
                assert "shell" not in messages[-2].content
            if turn == 3:
                assert outputs["2-0"]["ok"] and outputs["2-1"]["ok"]
            if turn == 5:
                assert outputs["3-0"]["content"] == "before\n"
                assert outputs["4-0"]["error_code"] == "USE_REPOSITORY_TOOL"
                assert f"Select {ISOLATED_CHECK_TOOL} separately" in outputs["4-0"]["error"]
                assert outputs["4-1"]["executed"] == 0
            selected = selections[turn]
            assert selected and all(name in self.offered for name, _ in selected)
            return SimpleNamespace(tool_calls=[{"id": f"{turn}-{index}", "name": name, "args": args}
                                              for index, (name, args) in enumerate(selected)])

    model = Model()
    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_checkout_verification", check)
    gateway._coding_agent = UnifiedCodingAgentFacade(repo_root=gateway.root, default_mode="internal",
        internal_engine=InternalCodingEngine(llm_client=model, max_tool_calls=6, programmatic_tool_calling_enabled=False))
    result = workflow.start("session", "Fix the failing build", mode="debug", repair=True,
                            workflow=workflow_choice(mode="debug"))

    assert result.status == "success", result.message
    assert result.data["changed_files"] == ["app.py"]
    assert model.turn == len(selections)
    assert checks == ["after\n"]
    parent = workflow.board.get_task(result.data["task_id"])
    assert parent.mode_workflow["tool_operations"] == 5


@pytest.mark.parametrize("authorized", [True, False])
def test_adaptive_coding_renews_through_accounting_on_the_same_assignment(mode_gateway, monkeypatch, authorized):
    from mana_agent.coding.adaptive import BUDGET_RENEWAL_TOOL
    from mana_agent.coding.facade import UnifiedCodingAgentFacade
    from mana_agent.coding.internal_engine import InternalCodingEngine
    from mana_agent.coding.models import CODING_RESULT_TOOL, ISOLATED_CHECK_TOOL
    from mana_agent.config.autonomous_config import AutoRenewConfig

    gateway, workflow = mode_gateway
    gateway._entry_router.llm = Decisions()
    controller = gateway._autonomous_controller
    controller.config = AutonomousExecutionConfig(max_tool_calls=2, max_model_turns=6,
        auto_renew=AutoRenewConfig(renew_tool_calls=2, renew_model_turns=2, max_auto_renewals=1))
    accounting_requests = []

    def account(**values):
        accounting_requests.append(values)
        return authorized, "Accounting denied the extension." if not authorized else "Authorized."

    controller.accounting_service = SimpleNamespace(validate_autonomous_renewal=account)
    (gateway.root / "caller.py").write_text("before\n")
    seen_checks = []

    async def check(root, command, *, task_id):
        assert (root / "app.py").read_text() == "first change\n"
        assert command.setup_argv == (("make",),)
        seen_checks.append(task_id)
        passed = (root / "caller.py").read_text() == "after\n"
        return {"passed": passed, "exit_code": 0 if passed else 7, "output": "passed" if passed else "caller needs repair"}

    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_checkout_verification", check)
    correct = {"argv": ["ctest", "--test-dir", "build", "--output-on-failure"], "setup_argv": [["make"]]}
    patch = "*** Begin Patch\n*** Update File: caller.py\n@@\n-before\n+after\n*** End Patch"
    choices = [("write_file", {"path": "app.py", "content": "first change\n"}),
               (ISOLATED_CHECK_TOOL, correct),
               (BUDGET_RENEWAL_TOOL, {"reason": "The first file changed. Finish caller.py and verify the integrated result."})]
    if authorized:
        choices.extend([
            ("apply_patch_batch", {"patches": [{"patch": "--- a/caller.py\n+++ b/caller.py\n"}]}),
            ("apply_patch_batch", {"patches": [{"patch": patch}]}),
            (ISOLATED_CHECK_TOOL, {**correct, "cwd": str(gateway.root)}),
            (ISOLATED_CHECK_TOOL, correct),
            (CODING_RESULT_TOOL, {"disposition": "complete", "summary": "Changed both files and verified."}),
        ])
    else:
        choices.append((CODING_RESULT_TOOL, {"disposition": "blocked", "summary": "Accounting denied the extension."}))

    class Model:
        turn = 0

        def with_tools(self, tools):
            self.offered = {tool["function"]["name"] for tool in tools}
            return self

        async def generate(self, messages, *, continuation):
            if self.turn == 2:
                assert "0 tool calls remain" in messages[-1].content
                assert self.offered == {BUDGET_RENEWAL_TOOL, CODING_RESULT_TOOL}
            elif self.turn == 3:
                renewal = json.loads(messages[-2].content)
                assert renewal["ok"] is authorized
                if authorized:
                    assert renewal["tool_calls"] == 2 and "2 tool calls" in messages[-1].content
                else:
                    assert renewal["error_code"] == "BUDGET_RENEWAL_DENIED" and BUDGET_RENEWAL_TOOL not in self.offered
            elif self.turn in {4, 6}:
                rejection = json.loads(messages[-2].content)
                assert rejection["executed"] == 0
                assert rejection["error_code"] == ("INVALID_PATCH_FORMAT" if self.turn == 4 else "INVALID_VERIFICATION_COMMAND")
            tool, args = choices[self.turn]
            assert tool in self.offered
            self.turn += 1
            return SimpleNamespace(tool_calls=[{"id": f"renewal-{self.turn}", "name": tool, "args": args}])

    model = Model()
    gateway._coding_agent = UnifiedCodingAgentFacade(repo_root=gateway.root, default_mode="internal",
        internal_engine=InternalCodingEngine(llm_client=model, max_tool_calls=2, programmatic_tool_calling_enabled=False))
    result = workflow.start("session", "Fix the two files", mode="auto", repair=True, workflow=workflow_choice())

    assert result.status == ("success" if authorized else "error"), result.message
    parent = workflow.board.get_task(result.data["task_id"])
    child, = workflow.board.list_tasks(parent_task_id=parent.task_id)
    request, = accounting_requests
    assert request["respect_budget"] and request["task_budget_remaining"] > 0
    assert request["renew_turns"] == request["renew_tool_calls"] == 2
    assert parent.mode_workflow["tool_operations"] == (4 if authorized else 2)
    assert len(parent.mode_workflow.get("adaptive_budget_renewals", [])) == int(authorized)
    assert model.turn == len(choices)
    assert seen_checks == [child.task_id] * (2 if authorized else 1)


@pytest.mark.parametrize("repairs_until_success", [4, None], ids=["success", "repeat"])
def test_adaptive_checkout_build_renews_or_alarms_on_repeat(mode_gateway, monkeypatch, repairs_until_success):
    from mana_agent.coding.adaptive import BUDGET_RENEWAL_TOOL
    from mana_agent.coding.facade import UnifiedCodingAgentFacade
    from mana_agent.coding.internal_engine import InternalCodingEngine
    from mana_agent.coding.models import CODING_RESULT_TOOL, ISOLATED_CHECK_TOOL
    from mana_agent.config.autonomous_config import AutoRenewConfig

    gateway, workflow = mode_gateway
    subprocess.run(["git", "init", "--quiet"], cwd=gateway.root, check=True)
    (gateway.root / ".gitignore").write_text("build/\n")
    (gateway.root / "build").mkdir()
    gateway._entry_router.llm = Decisions()
    controller = gateway._autonomous_controller
    controller.config = AutonomousExecutionConfig(max_tool_calls=2 if repairs_until_success else 8,
        max_model_turns=6, max_no_progress_cycles=1,
        auto_renew=AutoRenewConfig(renew_tool_calls=2, renew_model_turns=2))
    renewals, builds, alarms = [], [], []

    def authorize_renewal(**request):
        renewals.append(request)
        return True, "Authorized"

    def run(_gateway, adapter):
        assert adapter.cwd == gateway.root.resolve()
        if adapter.argv == ["cmake", "-S", ".", "-B", "build"]:
            (adapter.cwd / "build" / "configured").touch()
            return {"ok": True, "returncode": 0}
        assert adapter.argv == ["cmake", "--build", "build"]
        assert (adapter.cwd / "build" / "configured").is_file()
        builds.append((adapter.parent_task_id, adapter.idempotency_key))
        (adapter.cwd / "build" / "output").write_text(str(len(builds)))
        passed = repairs_until_success is not None and len(builds) > repairs_until_success
        return {"ok": passed, "returncode": 0 if passed else 2, "stderr": f"Build output {len(builds)}"}

    def activity(_event, *, title, **fields):
        if fields.get("metadata", {}).get("reason_code") == "CODING_NO_PROGRESS":
            alarms.append(title)

    controller.accounting_service = SimpleNamespace(validate_autonomous_renewal=authorize_renewal)
    monkeypatch.setattr("mana_agent.coding.programmatic.execute_coding_shell_action", run)
    monkeypatch.setattr("mana_agent.services.activity_context.emit_activity", activity)
    verify = (ISOLATED_CHECK_TOOL, {"argv": ["cmake", "--build", "build"],
                                  "setup_argv": [["cmake", "-S", ".", "-B", "build"]]})
    if repairs_until_success is None:
        groups = [[verify], [verify], [verify]]
    else:
        groups = []
        for index in range(repairs_until_success + 1):
            if index:
                groups.append([(BUDGET_RENEWAL_TOOL, {"reason": "Another source repair and build are needed."})])
            groups.append([("write_file", {"path": "app.py", "content": f"repair = {index}\n"}), verify])
        groups.append([(CODING_RESULT_TOOL, {"disposition": "complete", "summary": "Build passed."})])

    class Model:
        turn = 0

        def with_tools(self, tools):
            self.offered = {tool["function"]["name"] for tool in tools}
            return self

        async def generate(self, messages, *, continuation):
            group = groups[self.turn]
            self.turn += 1
            assert all(name in self.offered for name, _ in group)
            return SimpleNamespace(tool_calls=[{"id": f"build-{self.turn}-{index}", "name": name, "args": args}
                                              for index, (name, args) in enumerate(group)])

    gateway._coding_agent = UnifiedCodingAgentFacade(repo_root=gateway.root, default_mode="internal",
        internal_engine=InternalCodingEngine(llm_client=Model(), max_tool_calls=controller.config.max_tool_calls,
                                            programmatic_tool_calling_enabled=False))
    result = workflow.start("session", "Repair and build", mode="auto", repair=True, workflow=workflow_choice())

    assert len({task_id for task_id, _ in builds}) == 1
    assert len({key for _, key in builds}) == len(builds)
    if repairs_until_success is None:
        assert result.status == "error"
        assert len(builds) == 3
        assert len(alarms) == 1
        assert not renewals
    else:
        assert result.status == "success", result.message
        assert len(renewals) == repairs_until_success
        assert result.data["changed_files"] == ["app.py"]
        assert [row["exit_code"] for row in result.data["verification"]] == [2] * repairs_until_success + [0]
        assert not alarms


def test_adaptive_coding_resumes_after_a_model_wait_failure(mode_gateway, monkeypatch):
    from mana_agent.coding.facade import UnifiedCodingAgentFacade
    from mana_agent.coding.internal_engine import InternalCodingEngine
    from mana_agent.coding.models import CODING_RESULT_TOOL, ISOLATED_CHECK_TOOL

    gateway, workflow = mode_gateway
    gateway._entry_router.llm = Decisions()
    actions = iter([
        ("read_file", {"path": "app.py", "evidence_need": "Confirm the body before editing."}),
        ("write_file", {"path": "app.py", "content": "fixed\n"}),
        None,
        (ISOLATED_CHECK_TOOL, {"argv": ["python", "app.py"]}),
        (CODING_RESULT_TOOL, {"disposition": "complete", "summary": "Fixed and checked."}),
    ])
    calls = []

    class Model:
        def with_tools(self, tools):
            self.offered = {tool["function"]["name"] for tool in tools}
            return self

        async def generate(self, messages, *, continuation):
            selection = next(actions)
            calls.append(selection)
            if selection is None:
                raise TimeoutError("Provider attempt interrupted")
            tool, args = selection
            if tool == ISOLATED_CHECK_TOOL:
                assert any("same assignment" in message.content for message in messages)
                assert any('"stale": true' in message.content for message in messages)
                assert "read_file" in self.offered
                assert "investigate_repository" not in self.offered
            return SimpleNamespace(tool_calls=[{"id": f"resume-{len(calls)}", "name": tool, "args": args}])

    async def check(root, command, *, task_id):
        assert (root / "app.py").read_text() == "fixed\n"
        return {"passed": True, "exit_code": 0, "output": "passed"}

    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_checkout_verification", check)
    gateway._coding_agent = UnifiedCodingAgentFacade(repo_root=gateway.root, default_mode="internal",
        internal_engine=InternalCodingEngine(llm_client=Model(), programmatic_tool_calling_enabled=False))
    first = workflow.start("session", "Fix app.py", mode="debug", repair=True, workflow=workflow_choice(mode="debug"))
    task_id = first.data["task_id"]
    child, = workflow.board.list_tasks(parent_task_id=task_id)
    initial_attempt = workflow.lane.execution_supervisor.get_task(child.task_id).attempt_id
    assert first.status == "error"
    assert first.data["changed_files"] == ["app.py"]
    assert child.evidence_refs

    resumed = workflow.retry(task_id)
    assert resumed.status == "success", resumed.message
    assert resumed.data["changed_files"] == ["app.py"]
    assert len(workflow.board.list_tasks(parent_task_id=task_id)) == 1
    assert resumed.data["verification"][0]["attempt_id"] != initial_attempt


def test_adaptive_coding_deadline_stops_checkpointing_without_retry(mode_gateway, monkeypatch):
    from mana_agent.coding.facade import UnifiedCodingAgentFacade
    from mana_agent.coding.internal_engine import InternalCodingEngine
    from mana_agent.gateway.chat_gateway import AgentChatGateway

    gateway, workflow = mode_gateway
    gateway._entry_router.llm = Decisions()
    supervisor = workflow.lane.execution_supervisor
    now = utc_now()
    monkeypatch.setattr(supervisor, "clock", lambda: now)
    checkpoint = workflow.lane.checkpoint
    saved_checkpoints = []

    def checkpoint_at_deadline(task_id, *, boundary, **kwargs):
        nonlocal now
        if boundary != "adaptive_coding_tools":
            return checkpoint(task_id, boundary=boundary, **kwargs)
        if saved_checkpoints:
            now = supervisor.get_task(task_id).deadline_at
        saved = checkpoint(task_id, boundary=boundary, **kwargs)
        saved_checkpoints.append(saved)
        return saved

    monkeypatch.setattr(workflow.lane, "checkpoint", checkpoint_at_deadline)
    actions = iter([
        ("write_file", {"path": "app.py", "content": "fixed\n"}),
        ("read_file", {"path": "app.py", "evidence_need": "Confirm the updated contents to choose a focused check."}),
    ])

    class Model:
        calls = 0

        async def generate(self, messages, *, continuation):
            self.calls += 1
            tool, args = next(actions)
            return SimpleNamespace(tool_calls=[{"id": f"deadline-{self.calls}", "name": tool, "args": args}])

    model = Model()
    gateway._coding_agent = UnifiedCodingAgentFacade(repo_root=gateway.root, default_mode="internal",
        internal_engine=InternalCodingEngine(llm_client=model, programmatic_tool_calling_enabled=False))
    result = workflow.start("session", "Fix app.py", mode="debug", repair=True, workflow=workflow_choice(mode="debug"))

    assert result.status == "error"
    assert result.data["error_code"] == "DEADLINE_EXPIRED"
    assert "new task" in result.message
    assert result.data["changed_files"] == ["app.py"]
    turn = AgentChatGateway._mode_turn_result(result)
    assert turn.payload["status"] == "timed_out"
    assert turn.error_code == "DEADLINE_EXPIRED"
    assert turn.error_category == "deadline"
    assert (gateway.root / "app.py").read_text() == "fixed\n"
    task_id = result.data["task_id"]
    child, = workflow.board.list_tasks(parent_task_id=task_id)
    for expired_id in (task_id, child.task_id):
        durable = supervisor.get_task(expired_id)
        assert durable.state == ExecutionState.FAILED
        assert not durable.lease_token
        assert workflow.lane.inspect_task(expired_id).state == LaneTaskState.TIMED_OUT
    assert saved_checkpoints == [supervisor.get_task(child.task_id).checkpoint_id]
    assert saved_checkpoints[0]
    assert "/task resume" not in workflow.describe(workflow.board.get_task(task_id)).message
    retry = workflow.retry(task_id)
    assert retry.data["error_code"] == "DEADLINE_EXPIRED"
    assert "new task" in retry.message
    assert model.calls == 2


def test_adaptive_coding_rejects_unsupported_completion_and_read_only_write(mode_gateway):
    from mana_agent.coding.facade import UnifiedCodingAgentFacade
    from mana_agent.coding.internal_engine import InternalCodingEngine
    from mana_agent.coding.models import CODING_RESULT_TOOL

    gateway, workflow = mode_gateway
    gateway._entry_router.llm = Decisions()
    finish = {"disposition": "complete", "summary": "Claimed completion."}
    selections = iter([(CODING_RESULT_TOOL, finish), ("write_file", {"path": "app.py", "content": "forbidden"}),
                       (CODING_RESULT_TOOL, finish)])

    class Model:
        calls = 0

        def with_tools(self, tools):
            assert "write_file" not in {tool["function"]["name"] for tool in tools}
            return self

        async def generate(self, messages, *, continuation):
            if self.calls:
                result = json.loads(messages[-1].content)
                assert result["error_code"] == ("CODING_RESULT_REJECTED" if self.calls == 1 else "TOOL_NOT_AUTHORIZED")
            tool, args = next(selections)
            self.calls += 1
            return SimpleNamespace(tool_calls=[{"id": f"denied-{self.calls}", "name": tool, "args": args}])

    gateway._coding_agent = UnifiedCodingAgentFacade(repo_root=gateway.root, default_mode="internal",
        internal_engine=InternalCodingEngine(llm_client=Model(), programmatic_tool_calling_enabled=False))
    result = workflow.start("session", "Diagnose app.py", mode="debug", workflow=workflow_choice(mode="debug", effect="read"))
    assert result.status == "error"
    assert "after correction" in result.message
    assert not result.data["goal_satisfied"]
    assert (gateway.root / "app.py").read_text() == "before\n"


@pytest.mark.parametrize("corrected", [True, False], ids=["repaired-chain", "still-disconnected"])
def test_adaptive_coding_corrects_runtime_wiring(mode_gateway, monkeypatch, corrected):
    from mana_agent.coding.facade import UnifiedCodingAgentFacade
    from mana_agent.coding.internal_engine import InternalCodingEngine
    from mana_agent.coding.models import CODING_RESULT_TOOL, ISOLATED_CHECK_TOOL

    gateway, workflow = mode_gateway
    gateway._entry_router.llm = Decisions()
    checks, accepted = [], []

    async def check(root, command, *, task_id):
        assert (root / "app.py").read_text() == "fixed\n"
        checks.append(task_id)
        return {"passed": True, "exit_code": 0, "output": "passed"}

    def accept_wiring(_task_id, state, plan):
        assert plan is None
        assert len(state["verification"]) == 1
        assert state["verification"][0]["source_versions"] == state["adaptive_file_versions"]
        accepted.append(state["runtime_wiring"])

    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_checkout_verification", check)
    monkeypatch.setattr(workflow, "_require_runtime_evidence", accept_wiring)

    def edge(source, target):
        return {"source": source, "target": target, "path": "app.py", "relation": "calls", "line": 1}

    finish = {"disposition": "complete", "summary": "Verified the change."}
    connected = {"reason": "Main calls the new handler.", "edges": [
        edge("main", "dispatch"), edge("dispatch", "handler"), edge("handler", "render"),
    ]}
    disconnected = {"reason": "The submitted report skipped intermediate links.", "edges": [
        edge("main", "dispatch"), edge("rejected_source_label", "other_path"), edge("handler", "render"),
    ]}
    selections = [
        ("write_file", {"path": "app.py", "content": "fixed\n"}),
        (ISOLATED_CHECK_TOOL, {"argv": ["python", "check.py"]}),
        (CODING_RESULT_TOOL, {**finish, "wiring": disconnected}),
        (CODING_RESULT_TOOL, {**finish, "wiring": connected if corrected else disconnected}),
    ]

    class Model:
        calls = 0

        def with_tools(self, tools):
            schema = next(tool["function"]["parameters"] for tool in tools
                          if tool["function"]["name"] == CODING_RESULT_TOOL)
            assert "Required for complete" in schema["properties"]["wiring"]["description"]
            edges = schema["$defs"]["RuntimeWiringEvidence"]["properties"]["edges"]
            assert "exactly match the source of the next edge" in edges["description"]
            return self

        async def generate(self, messages, *, continuation):
            if self.calls == 3:
                feedback = json.loads(next(message.content for message in reversed(messages) if message.role == "tool"))
                assert feedback["error_code"] == "CODING_RESULT_REJECTED"
                assert "wiring.edges" in feedback["error"]
                assert "1 → 2, 2 → 3" in feedback["error"]
                assert "input_value" not in feedback["error"] and "rejected_source_label" not in feedback["error"]
            name, args = selections[self.calls]
            self.calls += 1
            return SimpleNamespace(tool_calls=[{"id": f"wiring-{self.calls}", "name": name, "args": args}])

    model = Model()
    gateway._coding_agent = UnifiedCodingAgentFacade(repo_root=gateway.root, default_mode="internal",
        internal_engine=InternalCodingEngine(llm_client=model, programmatic_tool_calling_enabled=False))
    result = workflow.start("session", "Add the default handler", mode="auto", repair=True,
                            runtime_capability_change=True, workflow=workflow_choice())

    assert model.calls == 4
    assert len(checks) == 1
    assert result.data["changed_files"] == ["app.py"]
    if corrected:
        assert result.status == "success", result.message
        assert accepted == [connected]
    else:
        assert result.status == "error"
        assert "after correction" in result.message and "1 → 2, 2 → 3" in result.message
        assert "input_value" not in result.message and "rejected_source_label" not in result.message
        assert not accepted and not result.data["goal_satisfied"]


def test_adaptive_coding_corrects_evidence_reference_and_reuses_the_read(mode_gateway):
    from mana_agent.coding.facade import UnifiedCodingAgentFacade
    from mana_agent.coding.internal_engine import InternalCodingEngine
    from mana_agent.coding.models import CODING_RESULT_TOOL

    gateway, workflow = mode_gateway
    gateway._entry_router.llm = Decisions()

    class Model:
        step = 0

        async def generate(self, messages, *, continuation):
            self.step += 1
            if self.step == 1:
                name, args = "read_file", {"path": "app.py"}
            elif self.step == 2:
                first = json.loads(messages[-1].content)
                self.ref, = first["repository_evidence_refs"]
                name, args = "repository_evidence_get", {**self.ref, "revision": 999}
            elif self.step == 3:
                error = json.loads(messages[-1].content)
                assert error["error_code"] == "UNKNOWN_EVIDENCE_REF" and ":999" in error["error"]
                name, args = "repository_evidence_get", self.ref
            elif self.step == 4:
                assert json.loads(messages[-1].content)["evidence_id"] == self.ref["evidence_id"]
                name, args = "read_file", {"path": "app.py"}
            else:
                duplicate = json.loads(messages[-1].content)
                assert duplicate["reused"] and "content" not in duplicate
                assert duplicate["repository_evidence_refs"] == [self.ref]
                name, args = CODING_RESULT_TOOL, {"disposition": "complete", "summary": "Observed the current value."}
            return SimpleNamespace(tool_calls=[{"id": f"evidence-{self.step}", "name": name, "args": args}])

    gateway._coding_agent = UnifiedCodingAgentFacade(repo_root=gateway.root, default_mode="internal",
        internal_engine=InternalCodingEngine(llm_client=Model(), programmatic_tool_calling_enabled=False))
    result = workflow.start("session", "Diagnose app.py", mode="debug",
                            workflow=workflow_choice(mode="debug", effect="read"))
    assert result.status == "success", result.message
    assert result.data["repository_observations"]["evidence_ids"]


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
@pytest.mark.parametrize("initial_evidence", ["none", "inspect", "investigate"])
def test_auto_records_internal_plan_before_applying_and_verifying(mode_gateway, monkeypatch, backend, initial_evidence):
    gateway, workflow = mode_gateway
    investigate = initial_evidence == "investigate"
    inspect = initial_evidence == "inspect"
    if not investigate:
        budget = 4 if inspect else 3
        gateway._autonomous_controller.config = AutonomousExecutionConfig(max_model_turns=budget, max_tool_calls=budget)
    initial = {"none": [], "inspect": [inspection("app.py")],
               "investigate": [investigation("app.py", paths=["app.py"])]}
    gateway._entry_router.llm = Decisions(*initial[initial_evidence],
                                        draft(), {"disposition": "in_scope", "reason": "Only app.py differs"})

    def propose(_prompt, **kwargs):
        task = next(row for row in workflow.board.list_tasks() if row.mode_workflow.get("mode") == "auto")
        assert task.mode_workflow["plan"]["draft"]["steps"][0]["step_id"] == "fix"
        assert task.mode_workflow["phase"] == "executing"
        assert task.owning_lane == "coding"
        if not inspect:
            assert bool(task.evidence_refs) is investigate
        assert not any(child.task_kind == "investigation"
                       for child in workflow.board.list_tasks(parent_task_id=task.task_id))
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
    assert result.message == "Changed `app.py`. All required verification checks passed."
    assert result.data["repository_observations"]["findings"] == "The handler returns the wrong value."
    task = workflow.board.get_task(result.data["task_id"])
    if investigate:
        assert task.mode_workflow["evidence"][0]["board_ref"] in task.evidence_refs
    else:
        assert len(task.mode_workflow["evidence"]) == int(inspect)
        assert result.data["repository_observations"]["workflow_model_decisions"] == budget
        draft_input = gateway._entry_router.llm.inputs[int(inspect)]
        assert "draft" in draft_input["allowed_actions"]
        assert draft_input["remaining_inspections"] == 0
        assert "reserved for patching and verification" in draft_input["inspection_stop"]
    assert task.mode_workflow["approved_digest"] == ""
    assert not task.mode_workflow.get("inbox_id")


@pytest.mark.parametrize("case", ["corrected", "malformed", "out_of_scope", "model_budget", "tool_budget"])
def test_realtime_patch_format_correction_is_bounded(mode_gateway, monkeypatch, case):
    import mana_agent.gateway.mode_workflow as mode_module
    from mana_agent.approval.policy import set_approval_mode

    gateway, workflow = mode_gateway
    set_approval_mode("full_access")
    gateway._autonomous_controller.config = AutonomousExecutionConfig(
        max_model_turns=3 if case == "model_budget" else 4,
        max_tool_calls=3 if case == "tool_budget" else 4,
    )
    decisions = Decisions(draft(), {"disposition": "in_scope", "reason": "Only app.py differs"})
    gateway._entry_router.llm = decisions
    valid = "*** Begin Patch\n*** Update File: app.py\n@@\n-before\n+after\n*** End Patch"
    malformed = valid + "\n@@\n+unexpected\n*** End Patch"
    replacement = {
        "corrected": valid,
        "malformed": malformed,
        "out_of_scope": valid.replace("app.py", "other.py"),
    }.get(case)
    proposals = []
    original_apply = mode_module.safe_apply_patch
    writes = []
    checks = []

    def propose(prompt, **kwargs):
        assert (gateway.root / "app.py").read_text() == "before\n"
        assert kwargs["execution_mode"] == "read_only"
        assert kwargs["coding_mode_files"] == ["app.py"]
        if proposals:
            assert len(proposals) == 1 and replacement is not None
            assert kwargs["gateway_task_id"] == proposals[0]["gateway_task_id"]
            assert kwargs["turn_id"] != proposals[0]["turn_id"]
            assert kwargs["execution_runtime"]["remaining_model_turns"] == 1
            assert kwargs["execution_runtime"]["remaining_tool_calls"] == 0
            assert "premature *** End Patch at patch line 6" in prompt
        patch = replacement if proposals else malformed
        proposals.append(kwargs)
        return {"status": "completed", "output": json.dumps({"patch": patch})}

    def observe_apply(**kwargs):
        if not kwargs.get("check_only"):
            writes.append(kwargs["patch"])
        return original_apply(**kwargs)

    async def verify(root, command, **kwargs):
        assert (root / "app.py").read_text() == "after\n"
        checks.append(command.argv)
        return {"passed": True, "exit_code": 0, "output": "focused check passed"}

    gateway._coding_agent = SimpleNamespace(generate=propose, active_engine_mode="internal")
    monkeypatch.setattr(mode_module, "safe_apply_patch", observe_apply)
    monkeypatch.setattr(mode_module, "run_isolated_verification", verify)
    result = workflow.start("session", "Add the handler behavior", mode="auto", repair=True,
                            workflow=workflow_choice())

    reviewed = [entry["patch"] for entry in decisions.inputs if "patch" in entry]
    assert len(proposals) == (1 if replacement is None else 2)
    if case == "corrected":
        assert result.status == "success", result.message
        assert reviewed == writes == [valid]
        assert checks == [("python", "app.py")]
        assert result.data["repository_observations"]["workflow_model_decisions"] == 4
        assert result.data["repository_observations"]["workflow_tool_operations"] == 4
    else:
        assert result.data["phase"] == "blocked", result.message
        assert not reviewed and not writes and not checks
        assert (gateway.root / "app.py").read_text() == "before\n"
        assert not (gateway.root / "other.py").exists()
        if case == "out_of_scope":
            assert "outside step fix" in result.message
        else:
            assert "premature *** End Patch at patch line 6" in result.message
            assert "patch_context_not_found" not in result.message
            assert ("workflow budget" if replacement is None else "Patch correction failed") in result.message


def test_verification_history_is_reopened_with_checkout_versions(mode_gateway, monkeypatch):
    from mana_agent.approval.policy import set_approval_mode
    from mana_agent.human_inbox.models import canonical_digest
    from mana_agent.services.coding_memory_service import CodingMemoryService

    gateway, workflow = mode_gateway
    gateway.config = SimpleNamespace(coding_memory=True)
    set_approval_mode("full_access")
    gateway._entry_router.llm = Decisions(draft(), {"disposition": "in_scope", "reason": "Only app.py changes"})
    patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-before\n+after\n*** End Patch"
    gateway._coding_agent = SimpleNamespace(
        generate=lambda *_args, **_kwargs: {"status": "completed", "output": json.dumps({"patch": patch})},
    )

    async def verify(*_args, **_kwargs):
        return {"passed": True, "exit_code": 0, "output": "observed check"}

    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_isolated_verification", verify)
    result = workflow.start("session", "Verify the named entrypoint", mode="auto", repair=True,
                            workflow=workflow_choice())
    assert result.status == "success", result.message
    task = workflow.board.get_task(result.data["task_id"])
    history = CodingMemoryService(project_root=gateway.root, session_id="session")
    flow_id = canonical_digest([task.workspace_id, task.primary_repository_id, str(gateway.root), task.task_id])
    with history._connect() as connection:
        rows = connection.execute("SELECT results_json FROM coding_flow_verification_results WHERE flow_id=?",
                                  (flow_id,)).fetchall()
    receipt, = [json.loads(row["results_json"]) for row in rows]
    assert receipt["checkout"] == task.mode_workflow["plan"]["base_checkout"]
    assert receipt["operation_id"] == result.data["verification"][0]["operation_id"]
    assert receipt["source_versions"] == {"app.py": workflow._file_digest(gateway.root, "app.py")}
    assert history.get_active_flow_id() != flow_id


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
    assert result.data["mode"] == "auto"
    assert result.data["strategy"] == "debug"
    inputs = gateway._entry_router.llm.inputs
    assert {item["task_id"] for item in inputs} == {result.data["task_id"]}
    assert inputs[-1]["evidence"] == inputs[-2]["evidence"]
    assert inputs[-1]["user_mode"] == "auto"
    assert inputs[-1]["mode"] == "debug"
    assert (gateway.root / "app.py").read_text() == "before\n"


@pytest.mark.parametrize("case,code", [
    ("wrong_task", "mode_task_identity"),
    ("unknown_switch_citation", "mode_unknown_evidence_reference"),
    ("unoffered_action", "mode_action_unavailable"),
    ("unauthorized_repair", "mode_repair_unauthorized"),
    ("missing_repair_check", "mode_passing_check"),
    ("missing_recheck", "mode_required_recheck"),
    ("old_evidence_revision", "mode_unknown_evidence_reference"),
    ("wrong_checkpoint", "mode_resume_checkpoint"),
    ("unsatisfied_checkpoint", "mode_resume_evidence"),
])
def test_mode_decision_rejects_invalid_handoff_in_schema(case, code):
    from dataclasses import replace
    from pydantic import ValidationError
    from mana_agent.agent.plan_service import InvestigationScope, ModeDecision, ModeDecisionContext
    from mana_agent.repository_evidence.models import EvidenceRef

    board_id = "repo_ev_" + "a" * 32
    workflow = ModeDecisionContext(
        task_id="active", strategy="debug", repair_authorized=True,
        allowed_actions=frozenset({"switch_workflow", "repair", "use_evidence", "resume_implementation"}),
        current_evidence_refs=(EvidenceRef(evidence_id=board_id, revision=2),),
        waiting_task_id="waiting", checkpoint_id="checkpoint-current", resume_ready=True,
    )
    if case in {"wrong_task", "unknown_switch_citation"}:
        decision = {"action": "switch_workflow", "reason": "New source evidence", "mode": "auto",
                    "task_id": "unrelated" if case == "wrong_task" else "active",
                    "evidence_ids": ["evidence-9" if case == "unknown_switch_citation" else "evidence-1"]}
    elif case == "unoffered_action":
        decision = draft()
    elif case in {"unauthorized_repair", "missing_repair_check", "missing_recheck"}:
        decision = draft()
        decision.update(action="repair", root_cause="Old handler value", cause_evidence=["evidence-1"])
        if case == "unauthorized_repair":
            workflow = replace(workflow, repair_authorized=False)
        elif case == "missing_repair_check":
            decision["draft"]["steps"].pop()
        else:
            workflow = replace(workflow, repair_limits={"app.py": ["update"]}, required_recheck={"argv": ["check"]})
    elif case == "old_evidence_revision":
        decision = {"action": "use_evidence", "reason": "Use the source", "evidence": [
            {"evidence_id": board_id, "revision": 1},
        ]}
    else:
        decision = {"action": "resume_implementation", "reason": "Continue the assigned step", "recovery": {
            "decision_id": "resume", "task_id": "waiting", "action": "resume_checkpoint", "reason": "Continue",
            "resume_checkpoint_id": "checkpoint-old" if case == "wrong_checkpoint" else "checkpoint-current",
            "safe_to_continue": True,
        }}
        if case == "unsatisfied_checkpoint":
            workflow = replace(workflow, resume_ready=False)

    with pytest.raises(ValidationError) as rejected:
        ModeDecision.model_validate({"decision": decision}, context={
            "mode_decision_context": workflow, "investigation_scope": InvestigationScope({"evidence-1": 1}),
        })
    assert rejected.value.errors(include_input=False)[0]["type"] == code


def test_auto_corrects_investigation_recovery_after_direct_inspection(mode_gateway, monkeypatch):
    from mana_agent.llm.capabilities import ModelProfile, observation
    from mana_agent.llm.engine import NativeLLMEngine
    from mana_agent.llm.messages import ToolCall

    gateway, workflow = mode_gateway
    proposed = draft()
    proposed["draft"]["steps"][0]["changes"] = [{"path": "app.py", "operations": ["update"]}]
    invalid = investigation("app.py", paths=["app.py"], recovery={
        "decision_id": "recover", "task_id": "evidence-1", "action": "resume_checkpoint",
        "reason": "Use the observation", "resume_checkpoint_id": "evidence-1", "safe_to_continue": True,
    })
    actions = iter([inspection("app.py"), invalid, proposed])
    requests = []

    class Adapter:
        async def generate(self, request):
            requests.append(request)
            return ModelResponse(tool_calls=[ToolCall(
                id=f"decision-{len(requests)}", name="record_decision", args={"decision": next(actions)},
            )])

    profile = ModelProfile(provider="openrouter", model_id="anthropic/claude-haiku-5.5", facts={
        name: observation(value, source="provider", confidence=1) for name, value in {
            "tool_calling": True, "input_text": True, "output_text": True, "native_structured_output": True,
            "context_window": 65536, "max_output_tokens": 1024,
        }.items()
    })
    model = NativeLLMEngine(provider=profile.provider, model=profile.model_id, api_key="fixture",
                            base_url="https://example.invalid", endpoint="anthropic_messages", adapter=Adapter())
    monkeypatch.setattr(model, "profile", lambda *_a, **_kw: profile)
    gateway._entry_router.llm = model
    gateway._coding_agent = SimpleNamespace(for_assignment=lambda **_: pytest.fail("Recovery must not reach the worker"))
    result = workflow.start("session", "Plan a focused change to the handler", mode="auto", repair=True,
                            no_code=True, workflow=workflow_choice())

    assert result.status == "success", result.message
    assert len(requests) == 3
    assert requests[1].tools == requests[2].tools
    context = json.loads(requests[1].messages[1].content)
    assert context["saved_investigation"] is None
    assert context["evidence_levels"] == {"evidence-1": 1}
    assert "No saved investigation is available to recover" in requests[2].messages[-1].content
    task_id = result.data["task_id"]
    assert workflow.board.get_task(task_id).mode_workflow["plan"]["draft"]["steps"][0]["step_id"] == "fix"
    assert workflow.board.list_tasks(parent_task_id=task_id) == []


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
    assert "Verified in worktree" in result.message
    assert source.read_text() == "uncommitted user work\n"


def test_auto_repairs_failed_verification_without_repeating_completed_patch(mode_gateway, monkeypatch):
    gateway, workflow = mode_gateway
    from mana_agent.approval.models import ApprovalMode
    from mana_agent.approval.policy import set_approval_mode

    second = draft()
    second["draft"]["steps"][0]["step_id"] = "repair"
    second["draft"]["steps"][1]["depends_on"] = ["repair"]
    review = {"disposition": "in_scope", "reason": "Only app.py changes"}
    third = draft()
    third["draft"]["steps"][0]["step_id"] = "repair_again"
    third["draft"]["steps"][1]["depends_on"] = ["repair_again"]
    third.update(action="repair", root_cause="The revised handler still returns a wrong value",
                 cause_evidence=["evidence-2"])
    model = Decisions(draft(), review, second, review, third, review)
    gateway._entry_router.llm = model
    writes = []

    def propose(_prompt, **kwargs):
        prior = (gateway.root / "app.py").read_text().strip()
        new = {"before": "after", "after": "partially repaired", "partially repaired": "repaired"}[prior]
        writes.append((prior, new))
        patch = f"*** Begin Patch\n*** Update File: app.py\n@@\n-{prior}\n+{new}\n*** End Patch"
        return {"status": "completed", "output": json.dumps({"patch": patch})}

    async def verify(root, command, **kwargs):
        state = (root / "app.py").read_text().strip()
        passed = state == "repaired"
        return {"passed": passed, "exit_code": 0 if passed else 1, "output": f"handler behavior: {state}"}

    gateway._coding_agent = SimpleNamespace(generate=propose, active_engine_mode="internal")
    set_approval_mode(ApprovalMode.FULL_ACCESS)
    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_isolated_verification", verify)
    result = workflow.start("session", "Add the handler behavior", mode="auto", repair=True,
                            workflow=workflow_choice(), source_message_ids=("user-1",))
    assert result.status == "success", result.message
    assert writes == [("before", "after"), ("after", "partially repaired"), ("partially repaired", "repaired")]
    assert [row["passed"] for row in result.data["verification"]] == [False, False, True]
    assert len({row["operation_id"] for row in result.data["verification"]}) == 3
    assert all(row["argv"] == ["python", "app.py"] and row["attempt_id"] for row in result.data["verification"])
    assert len({row["task_id"] for row in model.inputs if "task_id" in row}) == 1
    assert [item["verification_failure"]["result"]["exit_code"] for item in model.inputs
            if item.get("verification_failure")] == [1, 1]


def test_equivalent_evidence_queries_stop_without_using_the_tool_budget(mode_gateway):
    gateway, workflow = mode_gateway
    gateway._autonomous_controller.config = AutonomousExecutionConfig(max_no_progress_cycles=2)
    query = {"action": "query_evidence", "reason": "Find saved handler behavior", "query": {"paths": ["app.py"]}}
    model = Decisions(query, {**query, "reason": "Look for the handler result"}, query)
    gateway._entry_router.llm = model
    result = workflow.start("session", "Add the handler behavior", mode="auto", repair=True,
                            workflow=workflow_choice())
    assert result.status == "error"
    assert "Repeated evidence queries" in result.message
    assert result.data["repository_observations"]["workflow_model_decisions"] == 3
    assert result.data["repository_observations"]["workflow_tool_operations"] == 0
    assert "no new current knowledge" in model.inputs[-1]["evidence_discovery_stop"]


def test_repeated_verification_failure_retains_patches_and_blocks_completion(mode_gateway, monkeypatch):
    from mana_agent.approval.policy import set_approval_mode

    gateway, workflow = mode_gateway
    set_approval_mode("full_access")
    gateway._autonomous_controller.config = AutonomousExecutionConfig(max_no_progress_cycles=2)
    review = {"disposition": "in_scope", "reason": "Only app.py changes"}
    repair = draft()
    repair["draft"]["steps"][0]["step_id"] = "retry_handler"
    repair["draft"]["steps"][1]["depends_on"] = ["retry_handler"]
    gateway._entry_router.llm = Decisions(draft(), review, repair, review)
    patches = iter([("before", "after"), ("after", "repaired")])

    def propose(*_args, **_kwargs):
        before, after = next(patches)
        patch = f"*** Begin Patch\n*** Update File: app.py\n@@\n-{before}\n+{after}\n*** End Patch"
        return {"status": "completed", "output": json.dumps({"patch": patch})}

    async def verify(*_args, **_kwargs):
        return {"passed": False, "exit_code": 1, "output": "same failing assertion"}

    gateway._coding_agent = SimpleNamespace(generate=propose, active_engine_mode="internal")
    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_isolated_verification", verify)
    result = workflow.start("session", "Fix the handler", mode="auto", repair=True, workflow=workflow_choice())
    assert result.status == "error"
    assert "same verification failure recurred" in result.message
    assert len(result.data["verification"]) == 2
    assert not result.data["goal_satisfied"]
    assert (gateway.root / "app.py").read_text() == "repaired\n"
    assert workflow.board.get_task(result.data["task_id"]).mode_workflow["plan"]


def test_implementation_resumes_exact_step_after_reusing_evidence_from_another_session(mode_gateway, monkeypatch):
    from mana_agent.approval.policy import set_approval_mode
    from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
    from mana_agent.repository_evidence.models import EvidenceQuery

    gateway, workflow = mode_gateway
    (gateway.root / "config.py").write_text("ENABLED = True\n")
    (gateway.root / "other.py").write_text("before\n")
    set_approval_mode("full_access")
    gateway._entry_router.llm = Decisions(inspection("config.py"), {
        "action": "diagnose", "reason": "Read the named option", "root_cause": "The option is enabled",
        "cause_evidence": ["evidence-1"],
    })
    seed = workflow.start("old-session", "Read the named option", mode="debug",
                          workflow=workflow_choice(mode="debug", effect="read"))
    assert seed.status == "success", seed.message
    reopened = TaskBoard(gateway.root)
    saved, = reopened.evidence_board().query(EvidenceQuery(paths=("config.py",)))["records"]
    ref = {key: saved[key] for key in ("evidence_id", "revision")}

    proposed_steps = draft()
    proposed_steps["draft"]["steps"].insert(1, {
        "step_id": "consumer", "instruction": "Update consumer", "depends_on": ["fix"],
        "changes": {"other.py": ["update"]}, "acceptance_criteria": ["Consumer handles the option"],
    })
    proposed_steps["draft"]["steps"][-1]["depends_on"] = ["consumer"]
    review = {"disposition": "in_scope", "reason": "Matches the scoped change"}
    proposed = []
    gap_task_id = ""
    continuation = {"executed_operations": [{"tool": "read_file", "call_id": "before-wait"}]}

    class AdaptiveDecisions(Decisions):
        def decide_sync(self, messages, schema):
            response = super().decide_sync(messages, schema)
            selected = response.get("decision", {})
            if selected.get("action") == "resume_implementation":
                assert self.inputs[-1]["completed_step_ids"] == ["fix"]
                assert self.inputs[-1]["pending_repository_evidence"]["task_id"] == gap_task_id
                saved_task = workflow.lane.execution_supervisor.get_task(gap_task_id)
                assert saved_task.checkpoint_id == self.inputs[-1]["pending_coding_checkpoint_id"]
                checkpoint = workflow.lane.execution_supervisor.store.get_checkpoint(saved_task.checkpoint_id)
                assert checkpoint.resume_payload["continuation"] == continuation
                selected["recovery"] = {
                    "decision_id": "resume-consumer", "task_id": gap_task_id, "action": "resume_checkpoint",
                    "retry_category": "model", "reason": "The exact requirement is satisfied",
                    "resume_checkpoint_id": saved_task.checkpoint_id,
                    "same_task_retry_authorized": True, "safe_to_continue": True,
                }
            return response

    gateway._entry_router.llm = AdaptiveDecisions(
        proposed_steps, review,
        {"action": "query_evidence", "reason": "Find the observed option", "query": {"paths": ["config.py"]}},
        {"action": "use_evidence", "reason": "Reuse the observed option", "evidence": [ref], "requirement_id": "option"},
        {"action": "resume_implementation", "reason": "The exact requirement is satisfied"}, review,
    )

    def propose(_prompt, *, gateway_task_id, coding_mode_files, coding_resume_context, **kwargs):
        nonlocal gap_task_id
        path, = coding_mode_files
        proposed.append((gateway_task_id, path))
        if path == "other.py" and not gap_task_id:
            assert (gateway.root / "app.py").read_text() == "after\n"
            gap_task_id = gateway_task_id
            return {"status": "waiting", "continuation_context": continuation, "missing_evidence": [{
                "task_id": gap_task_id, "reason": "Need the configuration option",
                "requirement": {"requirement_id": "option", "objective": "Read the option",
                                "paths": ["config.py"], "acceptance_criteria": ["Current config is observed"]},
            }]}
        if path == "other.py":
            assert coding_resume_context == continuation
            assert gateway_task_id == gap_task_id
            assert ref in workflow.board.get_task(gateway_task_id).evidence_refs
        patch = f"*** Begin Patch\n*** Update File: {path}\n@@\n-before\n+after\n*** End Patch"
        return {"status": "completed", "output": json.dumps({"patch": patch})}

    async def verify(*_args, **_kwargs):
        assert (gateway.root / "other.py").read_text() == "after\n"
        return {"passed": True, "exit_code": 0, "output": "Both changes verified"}

    monkeypatch.setattr("mana_agent.agent.plan_service.PlanService.inspect",
                        lambda *_args: pytest.fail("Current evidence should not require a new inspection"))
    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_isolated_verification", verify)
    gateway._coding_agent = SimpleNamespace(generate=propose, active_engine_mode="internal")
    result = workflow.start("new-session", "Implement handler and consumer", mode="auto", repair=True,
                            workflow=workflow_choice())
    assert result.status == "success", result.message
    assert [path for _, path in proposed] == ["app.py", "other.py", "other.py"]
    children = workflow.board.list_tasks(parent_task_id=result.data["task_id"])
    assert [child.task_kind for child in children] == ["implementation", "implementation"]
    assert result.data["repository_observations"]["evidence_accounting"]["reused_evidence"] == 1


def test_optional_scoped_investigation_does_not_block_implementation(mode_gateway, monkeypatch):
    from mana_agent.approval.policy import set_approval_mode

    gateway, workflow = mode_gateway
    set_approval_mode("full_access")
    gateway._entry_router.llm = Decisions(
        investigation("optional.py", paths=["optional.py"]), draft(),
        {"disposition": "in_scope", "reason": "Only app.py changes"},
    )

    def propose(_prompt, **kwargs):
        assert not kwargs.get("repository_investigation")
        patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-before\n+after\n*** End Patch"
        return {"status": "completed", "output": json.dumps({"patch": patch})}

    async def verify(*_args, **_kwargs):
        return {"passed": True, "exit_code": 0, "output": "Independent check passed"}

    gateway._coding_agent = SimpleNamespace(active_engine_mode="internal", generate=propose)
    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_isolated_verification", verify)
    result = workflow.start("session", "Add the handler behavior", mode="auto", repair=True,
                            workflow=workflow_choice())
    assert result.status == "success", result.message
    assert result.data["verification"][-1]["passed"]
    assert (gateway.root / "app.py").read_text() == "after\n"
    assert all(child.task_kind != "investigation" for child in workflow.board.list_tasks(
        parent_task_id=result.data["task_id"],
    ))


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
    assert set(opened) == {"app.py"}


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
        model = Decisions(investigation("app.py and its test", *batch, paths=["app.py", "test_app.py"]), final)
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
    assert (state["decision_count"], state["tool_operations"]) == (2, 2)
    if mode == "plan":
        assert state["investigation"]["status"] == "returned"
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


@pytest.mark.parametrize("corrected", [True, False])
def test_coding_inspection_path_correction_preserves_saved_evidence(mode_gateway, monkeypatch, corrected):
    from mana_agent.llm.capabilities import ModelProfile, observation
    from mana_agent.llm.engine import NativeLLMEngine
    from mana_agent.llm.messages import ToolCall

    gateway, workflow = mode_gateway
    (gateway.root / "test_app.py").write_text("assert handler()\n")
    first = inspection("app.py", evidence_call("test", paths=["test_app.py"]),
                       evidence_call("layout", "list_files", glob="*"))
    protected_path = ".git/fixture-private-target"
    invalid = inspection(None, evidence_call("extension", paths=["app.py"], start_line=2),
                         evidence_call("protected", "repo_search", paths=[protected_path], query="handler"))
    conclusion = {"action": "diagnose", "reason": "Use the saved source and test observations",
                  "root_cause": "Old handler value", "cause_evidence": ["evidence-1", "evidence-2"]}
    decisions = iter([first, invalid, conclusion] if corrected else [first, invalid, invalid, conclusion])
    requests = []

    class Adapter:
        async def generate(self, request):
            requests.append(request)
            assert [tool["name"] for tool in request.tools] == ["record_decision"]
            assert request.response_schema is None
            return ModelResponse(tool_calls=[ToolCall(
                id=f"decision-{len(requests)}", name="record_decision", args={"decision": next(decisions)},
            )])

    facts = {name: observation(value, source="provider", confidence=1) for name, value in {
        "input_text": True, "output_text": True, "tool_calling": True,
        "context_window": 65536, "max_output_tokens": 1024,
    }.items()}
    model = NativeLLMEngine(provider="openrouter", model="anthropic/claude-sonnet-5.5", api_key="fixture",
                            base_url="https://example.invalid", endpoint="anthropic_messages", adapter=Adapter())
    profile = ModelProfile(provider=model.provider, model_id=model.model, facts=facts)
    monkeypatch.setattr(model, "profile", lambda *_: profile)
    gateway._entry_router.llm = model

    result = workflow.start("session", "Explain the handler behavior", mode="debug", llm=model)
    task_id = result.data["task_id"]
    evidence_ids = ["evidence-1", "evidence-2", "evidence-3"]
    state = workflow.board.get_task(task_id).mode_workflow
    assert [row["id"] for row in state["evidence"]] == evidence_ids
    assert state["tool_operations"] == 3
    assert len(requests) == 3
    assert requests[1].tools == requests[2].tools
    correction = requests[2].messages[-1].content
    detail = ("decision.inspect.requests.1.repo_search.paths.0: Protected repository paths are unavailable. "
              "Choose an accessible path or report the gap.")
    assert detail in correction
    assert "literal error" not in correction
    assert "too short" not in correction
    assert protected_path not in correction

    if not corrected:
        assert result.data["phase"] == "blocked"
        assert detail in result.message
        assert protected_path not in result.message
        result = workflow.retry(task_id)
        assert result.data["task_id"] == task_id
        assert list(json.loads(requests[-1].messages[1].content)["evidence_levels"]) == evidence_ids

    assert result.status == "success", result.message
    assert result.data["repository_observations"]["evidence_ids"] == evidence_ids
    assert result.data["repository_observations"]["workflow_tool_operations"] == 3


@pytest.mark.parametrize("invalid_selection", ["search_paths", "listing_options", "search_query"])
def test_inspection_correction_rebuilds_invalid_reads_as_scoped_discovery(mode_gateway, monkeypatch, invalid_selection):
    from mana_agent.llm.capabilities import ModelProfile, observation
    from mana_agent.llm.engine import NativeLLMEngine
    from mana_agent.llm.messages import ToolCall

    gateway, workflow = mode_gateway
    runtime = gateway.root / "runtime"
    runtime.mkdir()
    (runtime / "handler.py").write_text("value = 'before'\n")
    rejected_globs = [f"fixture-private-{index}/**/*.py" for index in range(4)]
    if invalid_selection == "search_paths":
        invalid = inspection(None, *(
            evidence_call(str(index), "repo_search", paths=[path], query="before")
            for index, path in enumerate(rejected_globs)
        ))
    elif invalid_selection == "listing_options":
        invalid = inspection(None, evidence_call("layout", "ls", glob=rejected_globs[0]))
    else:
        invalid = inspection(None, evidence_call("layout", "ls"),
                             evidence_call("search", "repo_search", paths=["runtime"], query=" \t"))
    layout = inspection(None, evidence_call("layout", "ls"))
    scoped = inspection(None, evidence_call("search", "repo_search", paths=[], glob="runtime/**/*.py", query="before"))
    conclusion = {"action": "diagnose", "reason": "Read the value in runtime/handler.py",
                  "root_cause": "The handler returns the old value", "cause_evidence": ["evidence-2"]}
    decisions = iter([invalid, layout, scoped, conclusion])
    requests, reads = [], []

    class Adapter:
        async def generate(self, request):
            requests.append(request)
            if len(requests) == 2:
                assert not reads
            return ModelResponse(tool_calls=[ToolCall(
                id=f"decision-{len(requests)}", name="record_decision", args={"decision": next(decisions)},
            )])

    real_inspect = PlanService.inspect

    def inspect(service, request):
        reads.append((request.tool, request.paths))
        return real_inspect(service, request)

    monkeypatch.setattr(PlanService, "inspect", inspect)
    facts = {name: observation(value, source="provider", confidence=1) for name, value in {
        "input_text": True, "output_text": True, "tool_calling": True,
        "context_window": 65536, "max_output_tokens": 1024,
    }.items()}
    model = NativeLLMEngine(provider="openrouter", model="deepseek/deepseek-v4.1-flash", api_key="fixture",
                            base_url="https://example.invalid", endpoint="chat_completions", adapter=Adapter())
    profile = ModelProfile(provider=model.provider, model_id=model.model, facts=facts)
    monkeypatch.setattr(model, "profile", lambda *_: profile)

    result = workflow.start("session", "Find where the old value comes from", mode="debug", llm=model)

    assert result.status == "success", result.message
    assert reads == [("ls", ()), ("repo_search", ())]
    assert len(requests) == 4
    assert requests[0].tools == requests[1].tools
    correction = requests[1].messages[-1].content
    assert "Provider tool arguments violate the registered schema" in correction
    assert "none of its reads ran" in correction
    assert 'paths [] with glob "src/**/*.py"' in correction
    assert '{"tool":"ls","level":1,"reason":"Locate the owning directory"}' in correction
    assert "put the literal text to find in query" in correction
    assert all(glob not in correction for glob in rejected_globs)
    assert "correction_hint" not in requests[1].parameters
    assert result.data["repository_observations"]["evidence_ids"] == ["evidence-1", "evidence-2"]


@pytest.mark.parametrize("corrected", [True, False])
def test_unscoped_inspection_search_correction_uses_saved_layout(mode_gateway, monkeypatch, corrected):
    from mana_agent.llm.capabilities import ModelProfile, observation
    from mana_agent.llm.engine import NativeLLMEngine
    from mana_agent.llm.messages import ToolCall

    gateway, workflow = mode_gateway
    component = gateway.root / "handlers"
    component.mkdir()
    (component / "handler.py").write_text("value = 'before'\n")
    layout = inspection(None, evidence_call("layout", "ls"))
    invalid = inspection(None, evidence_call("search", "repo_search", paths=[], glob="**/*", query="before"))
    scoped = inspection(None, evidence_call("search", "repo_search", paths=["handlers"], query="before"))
    diagnosis = {"action": "diagnose", "reason": "Use the matched line", "root_cause": "The handler uses the old value",
                 "cause_evidence": ["evidence-2"]}
    decisions = iter([layout, invalid, scoped, diagnosis] if corrected else [layout, invalid, invalid])
    requests, reads = [], []

    class Adapter:
        async def generate(self, request):
            requests.append(request)
            if len(requests) == 3:
                assert reads == [("ls", ())]
            return ModelResponse(tool_calls=[ToolCall(
                id=f"decision-{len(requests)}", name="record_decision", args={"decision": next(decisions)},
            )])

    profile = ModelProfile(provider="openrouter", model_id="deepseek/deepseek-v4.1-flash", facts={
        name: observation(value, source="provider", confidence=1) for name, value in {
            "input_text": True, "output_text": True, "tool_calling": True,
            "context_window": 65536, "max_output_tokens": 1024,
        }.items()
    })
    model = NativeLLMEngine(provider=profile.provider, model=profile.model_id, api_key="fixture",
                            base_url="https://example.invalid", endpoint="chat_completions", adapter=Adapter())
    monkeypatch.setattr(model, "profile", lambda *_: profile)
    original = PlanService.inspect

    def inspect(service, request):
        reads.append((request.tool, request.paths))
        return original(service, request)

    monkeypatch.setattr(PlanService, "inspect", inspect)
    result = workflow.start("session", "Find where the old value comes from", mode="debug", llm=model)

    assert requests[1].tools == requests[2].tools
    correction = requests[2].messages[-1].content
    assert "A query with an empty path list and a broad glob has no search scope" in correction
    assert "Repository-wide discovery requires an authorized level 4 and a reason" in correction
    assert "too short" not in correction
    assert "choose the next permitted action from allowed_actions" in correction
    example = next(json.loads(line) for line in correction.splitlines() if line.startswith('{"decision":'))
    assert example["decision"]["requests"][0]["paths"] == ["handlers"]
    if corrected:
        assert result.status == "success", result.message
        assert len(requests) == 4
        assert reads == [("ls", ()), ("repo_search", ("handlers",))]
    else:
        assert result.data["phase"] == "blocked", result.message
        assert len(requests) == 3
        assert reads == [("ls", ())]
        assert "has no search scope" in result.message
        assert "too short" not in result.message
        assert result.data["repository_observations"]["evidence_ids"] == ["evidence-1"]


def test_empty_inspection_requests_report_the_required_batch():
    from pydantic import ValidationError
    from mana_agent.agent.plan_service import ModeDecision
    from mana_agent.llm.errors import describe_decision_validation

    with pytest.raises(ValidationError) as rejected:
        ModeDecision.model_validate({"decision": inspection()})
    detail = describe_decision_validation(rejected.value, ModeDecision.model_json_schema())
    assert "decision.inspect.requests: Include at least one inspection request or select another allowed action" in detail


@pytest.mark.parametrize(("corrected", "older"), [(True, False), (True, True), (False, True)])
def test_failed_saved_inspection_uses_bounded_correction(mode_gateway, monkeypatch, corrected, older):
    from mana_agent.llm.capabilities import ModelProfile, observation
    from mana_agent.llm.engine import NativeLLMEngine
    from mana_agent.llm.messages import ToolCall

    gateway, workflow = mode_gateway
    (gateway.root / "test_app.py").write_text("assert handler()\n")
    repeated_path = "missing.py"
    read_app = inspection("app.py")
    invalid = inspection(repeated_path, evidence_call("new", paths=["test_app.py"]))
    prior_decisions = [inspection(repeated_path)]
    prior_reads = [repeated_path]
    evidence_id = "evidence-1"
    if older:
        (gateway.root / "runtime").mkdir()
        prior_decisions.insert(0, inspection(None, evidence_call("layout", "ls")))
        prior_reads.insert(0, "ls")
        evidence_id = "evidence-2"
        for start in (0, 4):
            paths = [f"other_{index}.py" for index in range(start, start + 4)]
            for path in paths:
                (gateway.root / path).write_text(f"# observation in {path}\n")
            prior_decisions.append(inspection(None, *(evidence_call(path, paths=[path]) for path in paths)))
            prior_reads.extend(paths)
    found_evidence = f"evidence-{len(prior_reads) + 1}"
    conclusion = {"action": "diagnose", "reason": "The source explains the behavior",
                  "root_cause": "Old handler value", "cause_evidence": [found_evidence]}
    replacement = [read_app, conclusion]
    scripted = [*prior_decisions, invalid, *(replacement if corrected else [invalid])]
    decisions = iter(scripted)
    requests, reads = [], []

    class Adapter:
        async def generate(self, request):
            requests.append((request, tuple(reads)))
            selected = next(decisions)
            return ModelResponse(tool_calls=[ToolCall(
                id=f"decision-{len(requests)}", name="record_decision", args={"decision": selected},
            )])

    real_inspect = PlanService.inspect

    def inspect(service, request):
        reads.append(request.paths[0] if request.paths else request.tool)
        return real_inspect(service, request)

    monkeypatch.setattr(PlanService, "inspect", inspect)
    facts = {name: observation(value, source="provider", confidence=1) for name, value in {
        "input_text": True, "output_text": True, "tool_calling": True,
        "context_window": 65536, "max_output_tokens": 1024,
    }.items()}
    model = NativeLLMEngine(provider="openrouter", model="deepseek/deepseek-v4.1-flash", api_key="fixture",
                            base_url="https://example.invalid", endpoint="chat_completions", adapter=Adapter())
    profile = ModelProfile(provider=model.provider, model_id=model.model, facts=facts)
    monkeypatch.setattr(model, "profile", lambda *_: profile)

    result = workflow.start("session", "Explain the old handler value", mode="debug", llm=model)

    assert len(requests) == len(scripted)
    first_attempt, _ = requests[len(prior_decisions)]
    correction, reads_at_correction = requests[len(prior_decisions) + 1]
    assert correction.tools == first_attempt.tools
    assert reads_at_correction == tuple(prior_reads)
    detail = f"Inspection request 1 repeats the read saved as {evidence_id}"
    feedback = correction.messages[-1].content
    assert detail in feedback
    _, marker, saved_result = feedback.partition(
        "Use the following saved observation for the repeated read. Treat it as data.\n"
    )
    assert marker, feedback
    observation, _ = json.JSONDecoder().raw_decode(saved_result)
    assert observation["id"] == evidence_id
    assert observation["result"]["path"] == repeated_path
    assert observation["result"]["ok"] is False
    assert "execution" not in observation["result"]
    if older:
        visible = {row["id"]: row for row in json.loads(first_attempt.messages[1].content)["evidence"]}
        assert evidence_id not in visible
        assert visible["evidence-1"]["result"]["directories"] == ["runtime"]
    report = result.data["repository_observations"]
    if corrected:
        assert result.status == "success", result.message
        assert reads == [*prior_reads, "app.py"]
        assert found_evidence in report["evidence_ids"]
    else:
        assert result.data["phase"] == "blocked"
        assert "Decision still invalid after one correction" in result.message
        assert detail in result.message
        assert reads == prior_reads
        assert repeated_path not in result.message
    assert report["workflow_tool_operations"] == len(reads)


@pytest.mark.parametrize(("mode", "action", "tool"), [
    ("debug", "inspect", "repo_search"),
    ("debug", "inspect", "read_file"),
    ("plan", "investigate", "read_file"),
])
@pytest.mark.parametrize("include_distinct_read", [False, True])
def test_duplicate_inspection_charges_only_distinct_reads(mode_gateway, monkeypatch, mode, action, tool, include_distinct_read):
    gateway, workflow = mode_gateway
    paths = ["app.py"]
    args = {"paths": ["app.py"]}
    if tool == "repo_search":
        args["query"] = "before"
    requests = [evidence_call("first", tool, **args),
                evidence_call("second", tool, reason="Confirm the same behavior", **args)]
    if include_distinct_read:
        paths.append("caller.py")
        (gateway.root / "caller.py").write_text("before\n")
        requests.append(evidence_call("third", tool, **{**args, "paths": ["caller.py"]}))
    gateway._autonomous_controller.config = AutonomousExecutionConfig(max_tool_calls=len(paths))
    if action == "investigate":
        decision = investigation("Read the handler", *requests, paths=paths)
    else:
        decision = inspection(None, *requests)
    conclusion = draft() if mode == "plan" else {
        "action": "diagnose", "reason": "The observation explains the behavior",
        "root_cause": "Old handler value", "cause_evidence": ["evidence-1"],
    }
    model = Decisions(decision, conclusion)
    seen = []
    inspect = PlanService.inspect

    def record(service, request):
        seen.append((request.tool, request.paths, request.query))
        return inspect(service, request)

    monkeypatch.setattr(PlanService, "inspect", record)
    result = workflow.start("session", "Explain the old handler value", mode=mode, llm=model, no_code=True, save=False)

    assert result.status == "success", result.message
    assert len(model.inputs) == 2
    assert seen == [(tool, (path,), args.get("query", "")) for path in paths]
    assert model.inputs[1]["deduplicated_inspection_requests"] == 1
    assert result.data["repository_observations"]["workflow_tool_operations"] == len(paths)


@pytest.mark.parametrize("case", ["bounded", "cited_expansion", "uncited_expansion"])
def test_duplicate_inspection_correction_preserves_evidence_levels(mode_gateway, monkeypatch, case):
    from mana_agent.llm.capabilities import ModelProfile, observation
    from mana_agent.llm.engine import NativeLLMEngine
    from mana_agent.llm.messages import ToolCall

    gateway, workflow = mode_gateway
    (gateway.root / "test_app.py").write_text("assert handler()\n")
    level = 1 if case == "bounded" else 2
    refs = ["evidence-1"] if case == "cited_expansion" else []
    duplicate = inspection(None,
        evidence_call("first", paths=["test_app.py"], level=level, prior_evidence=refs),
        evidence_call("duplicate", paths=["test_app.py"], level=1))
    if case == "uncited_expansion":
        # The first entry is authorized. The second selects the same read but has no justification to widen scope.
        duplicate["requests"].reverse()
    diagnosis = {"action": "diagnose", "reason": "Use the saved source and test observations", "root_cause": "Old handler value",
                 "cause_evidence": ["evidence-1", "evidence-2"]}
    script = [inspection("app.py"), duplicate, duplicate if case == "uncited_expansion" else diagnosis]
    decisions = iter(script)
    requests, reads, rejections = [], [], []

    class Adapter:
        async def generate(self, request):
            requests.append(request)
            if len(requests) == 3 and case == "uncited_expansion":
                assert reads == [("app.py", 1, ())]
            return ModelResponse(tool_calls=[ToolCall(
                id=f"decision-{len(requests)}", name="record_decision", args={"decision": next(decisions)},
            )])

    profile = ModelProfile(provider="openrouter", model_id="deepseek/deepseek-v4.1-flash", facts={
        name: observation(value, source="provider", confidence=1) for name, value in {
            "input_text": True, "output_text": True, "tool_calling": True,
            "context_window": 65536, "max_output_tokens": 1024,
        }.items()
    })
    model = NativeLLMEngine(provider=profile.provider, model=profile.model_id, api_key="fixture",
                            base_url="https://example.invalid", endpoint="chat_completions", adapter=Adapter())
    monkeypatch.setattr(model, "profile", lambda *_: profile)
    original = PlanService.inspect

    def inspect(service, request):
        reads.append((request.paths[0], request.level, request.prior_evidence))
        return original(service, request)

    def capture_rejection(kind, **fields):
        assert kind == "model.decision.rejected"
        rejections.append(fields["metadata"]["decision_validation"])

    monkeypatch.setattr("mana_agent.agent.plan_service.emit_activity", capture_rejection)
    monkeypatch.setattr(PlanService, "inspect", inspect)
    result = workflow.start("session", "Explain the old handler value", mode="debug", llm=model)

    assert len(requests) == len(script)
    if case == "uncited_expansion":
        assert result.data["phase"] == "blocked"
        assert reads == [("app.py", 1, ())]
        assert result.data["repository_observations"]["evidence_ids"] == ["evidence-1"]
        assert requests[1].tools == requests[2].tools
        assert len(rejections) == 2
        rejected = rejections[-1]
        assert rejected["action"] == "inspect" and rejected["final"]
        assert rejected["code"] == "mode_scope_missing_evidence"
        assert (rejected["field"], rejected["requested_level"], rejected["available_refs"]) == ("prior_evidence", 2, ["evidence-1"])
    else:
        assert result.status == "success", result.message
        assert reads == [("app.py", 1, ()), ("test_app.py", level, tuple(refs))]
        assert not rejections


@pytest.mark.parametrize("invalid_audit", [False, True])
def test_duplicate_inspection_after_saved_layout(mode_gateway, monkeypatch, invalid_audit):
    from mana_agent.llm.capabilities import ModelProfile, observation
    from mana_agent.llm.engine import NativeLLMEngine
    from mana_agent.llm.messages import ToolCall

    gateway, workflow = mode_gateway
    (gateway.root / "runtime").mkdir()
    (gateway.root / "runtime" / "handler.py").write_text("return_value = 'before'\n")
    layout = evidence_call("layout", "ls")
    duplicate = inspection(None, layout, evidence_call("another-directory", "ls", reason="Find the tests"))
    diagnosis = {"action": "diagnose", "reason": "The search found the source", "root_cause": "Old handler value",
                 "cause_evidence": ["evidence-2"]}
    requests, reads = [], []

    search = evidence_call("narrow", "repo_search", paths=["runtime"], query="return_value")
    if invalid_audit:
        search.args["initial_repository_scope"] = "explicit_audit"
    selected = inspection(None, search)
    decisions = iter([duplicate, duplicate, selected, selected if invalid_audit else diagnosis])

    class Adapter:
        async def generate(self, request):
            requests.append(request)
            if len(requests) == 3:
                context = json.loads(request.messages[1].content)
                assert context["reused_evidence_ids"] == ["evidence-1"]
                assert context["deduplicated_inspection_requests"] == 1
                assert context["evidence"][0]["result"]["directories"] == ["runtime"]
                assert reads == [("ls", ())]
            return ModelResponse(tool_calls=[ToolCall(
                id=f"decision-{len(requests)}", name="record_decision", args={"decision": next(decisions)},
            )])

    profile = ModelProfile(provider="openrouter", model_id="deepseek/deepseek-v4.1-flash", facts={
        name: observation(value, source="provider", confidence=1) for name, value in {
            "input_text": True, "output_text": True, "tool_calling": True,
            "context_window": 65536, "max_output_tokens": 1024,
        }.items()
    })
    model = NativeLLMEngine(provider=profile.provider, model=profile.model_id, api_key="fixture",
                            base_url="https://example.invalid", endpoint="chat_completions", adapter=Adapter())
    monkeypatch.setattr(model, "profile", lambda *_: profile)
    original = PlanService.inspect

    def inspect(service, request):
        reads.append((request.tool, request.paths))
        return original(service, request)

    monkeypatch.setattr(PlanService, "inspect", inspect)
    result = workflow.start("session", "Find where the old handler value comes from", mode="debug", llm=model)

    assert len(requests) == 4, result.message
    if invalid_audit:
        assert result.data["phase"] == "blocked"
        assert "Decision still invalid after one correction" in result.message
        assert "initial_repository_scope requires level 4" in result.message
        assert reads == [("ls", ())]
    else:
        assert result.status == "success", result.message
        assert reads == [("ls", ()), ("repo_search", ("runtime",))]
    assert result.data["repository_observations"]["workflow_tool_operations"] == len(reads)


def test_compact_observation_bounds_saved_layout():
    from mana_agent.agent.repository_investigation import compact_observation

    directories = [f"dir_{index}" for index in range(51)]
    row = {"id": "evidence-1", "level": 1, "result": {"ok": True, "directories": directories}}

    preview = compact_observation(row)

    assert preview["result"]["directories"] == directories[:50]
    assert preview["result"]["directories_truncated"] is True
    assert row["result"]["directories"] == directories


def test_coding_inspection_has_no_investigator_round_trips_at_child_limit(mode_gateway):
    gateway, workflow = mode_gateway
    supervisor = workflow.lane.execution_supervisor
    supervisor.config = supervisor.config.model_copy(update={"max_children_per_task": 0})
    gateway._coding_agent = SimpleNamespace(
        active_engine_mode="internal",
        for_assignment=lambda **_: pytest.fail("An inspect batch must execute on the parent"),
    )
    decisions = []
    for batch in range(21):
        calls = []
        for index in range(3):
            path = f"dependency_{batch}_{index}.py"
            (gateway.root / path).write_text(f"value = {batch * 3 + index}\n")
            calls.append(evidence_call(path, paths=[path]))
        decisions.append(inspection(None, *calls))
    decisions.append({"action": "diagnose", "reason": "The last dependency is the cause",
                      "root_cause": "The dependency returns the old value", "cause_evidence": ["evidence-63"]})
    model = Decisions(*decisions)
    model.with_tools = lambda *_: pytest.fail("Coding must not start a separate investigator")

    result = workflow.start("session", "Trace the dependency chain", mode="debug", llm=model)

    assert result.status == "success", result.message
    task = workflow.board.get_task(result.data["task_id"])
    state = task.mode_workflow
    assert (state["decision_count"], state["tool_operations"]) == (22, 63)
    assert len(state["evidence"]) == len(task.evidence_refs) == 63
    assert not workflow.board.list_tasks(parent_task_id=task.task_id)
    assert not supervisor.get_task(task.task_id).child_task_ids


def test_scoped_investigation_uses_no_child_slots(mode_gateway):
    gateway, workflow = mode_gateway
    supervisor = workflow.lane.execution_supervisor
    supervisor.config = supervisor.config.model_copy(update={"max_children_per_task": 0})
    gateway._coding_agent = SimpleNamespace(
        active_engine_mode="external",
        for_assignment=lambda **_: pytest.fail("Scoped reads must stay on the current task"),
    )
    model = Decisions(investigation("app.py", paths=["app.py"]),
                      {"action": "diagnose", "reason": "The code explains the result",
                       "root_cause": "Old handler value", "cause_evidence": ["evidence-1"]})
    result = workflow.start("session", "Investigate a missing source fact", mode="debug", llm=model)

    assert result.status == "success", result.message
    task_id = result.data["task_id"]
    assert not workflow.board.list_tasks(parent_task_id=task_id)
    assert not supervisor.get_task(task_id).child_task_ids


def test_coding_inspection_rereads_a_stale_source_changed_during_the_model_decision(mode_gateway):
    gateway, workflow = mode_gateway

    class SourceChanges(Decisions):
        def decide_sync(self, messages, schema):
            payload = super().decide_sync(messages, schema)
            if len(self.inputs) == 2:
                (gateway.root / "app.py").write_text("new value\n")
            return payload

    model = SourceChanges(
        inspection("app.py"), inspection("app.py"), inspection("app.py"),
        {"action": "diagnose", "reason": "The current source explains the result",
         "root_cause": "The source returns the new value", "cause_evidence": ["evidence-2"]},
    )
    result = workflow.start("session", "Read the current handler", mode="debug", llm=model)

    assert result.status == "success", result.message
    task = workflow.board.get_task(result.data["task_id"])
    previous, current = task.mode_workflow["evidence"]
    assert previous["id"] == "evidence-1" and previous["stale"]
    assert "content" not in previous["result"]
    assert current["id"] == "evidence-2" and current["result"]["content"] == "new value\n"
    assert current["board_ref"] in task.evidence_refs
    assert model.inputs[-1]["reused_evidence_ids"] == ["evidence-2"]
    assert result.data["repository_observations"]["workflow_tool_operations"] == 2


def test_coding_inspection_reads_only_the_selected_follow_up_range(mode_gateway):
    gateway, workflow = mode_gateway
    content = [f"line {i}\n" for i in range(1, 132)]
    (gateway.root / "app.py").write_text("".join(content))
    model = Decisions(
        inspection("app.py"),
        inspection(None, evidence_call("tail", paths=["app.py"], start_line=121, end_line=129)),
        {"action": "diagnose", "reason": "The next range contains the result",
         "root_cause": "The ending establishes the result", "cause_evidence": ["evidence-2"]},
    )
    result = workflow.start("session", "Read the handler ending", mode="debug", llm=model)

    assert result.status == "success", result.message
    first, second = [row["result"] for row in model.inputs[-1]["evidence"]]
    assert (first["start_line"], first["end_line"], first["total_lines"]) == (1, 131, 131)
    assert first["complete"]
    assert (second["start_line"], second["end_line"], second["total_lines"]) == (121, 129, 131)
    assert second["content"] == "".join(content[120:129])


def test_coding_inspection_does_not_stamp_a_character_truncated_line(mode_gateway):
    gateway, _workflow = mode_gateway
    (gateway.root / "app.py").write_bytes(b"first\n" + b"x" * 16_000 + b"\nlast\n")
    request = EvidenceRequest.model_validate(evidence_call("source", paths=["app.py"]).args | {"tool": "read_file"})

    result = PlanService(gateway.root, llm=None).inspect(request)

    assert result["content_truncated"]
    assert (result["end_line"], result["total_lines"]) == (1, 3)
    assert result["source_evidence"]["excerpt"] == "first\n"


def test_coding_inspection_reuses_saved_layout_in_mixed_batch(mode_gateway, monkeypatch):
    gateway, workflow = mode_gateway
    gateway._autonomous_controller.config = AutonomousExecutionConfig(max_tool_calls=2)
    (gateway.root / "runtime").mkdir()
    layout = evidence_call("layout", "ls", reason="Locate the source directory")
    model = Decisions(
        inspection(None, layout),
        inspection(None, layout, evidence_call("source", paths=["app.py"])),
        {"action": "diagnose", "reason": "The layout and source are available",
         "root_cause": "The handler returns the old value", "cause_evidence": ["evidence-1", "evidence-2"]},
    )
    reads = []
    real_inspect = PlanService.inspect

    def inspect(service, request):
        reads.append(request.tool)
        return real_inspect(service, request)

    monkeypatch.setattr(PlanService, "inspect", inspect)

    result = workflow.start("session", "Explain where the old value comes from", mode="debug", llm=model)

    assert result.status == "success", result.message
    assert reads == ["ls", "read_file"]
    assert len(model.inputs) == 3
    assert model.inputs[-1]["reused_evidence_ids"] == ["evidence-1"]
    assert [item["id"] for item in model.inputs[-1]["evidence"]] == ["evidence-1", "evidence-2"]
    assert result.data["repository_observations"]["workflow_tool_operations"] == 2


@pytest.mark.parametrize("reuse", [False, True])
def test_repeated_repository_results_stop_inspection_and_leave_a_final_decision(mode_gateway, reuse):
    gateway, workflow = mode_gateway
    gateway._autonomous_controller.config = AutonomousExecutionConfig(max_no_progress_cycles=2)
    queries = [
        inspection(None, evidence_call(query, "repo_search", paths=["app.py"], query=query))
        for query in (("before",) * 3 if reuse else ("before", "bef", "befo"))
    ]
    model = Decisions(*queries, {"action": "block", "reason": "The caller is still unknown"})

    result = workflow.start("session", "Find the caller of the handler", mode="debug", llm=model)

    assert result.data["phase"] == "blocked"
    assert "Repository reads stopped yielding new evidence" in result.message
    report = result.data["repository_observations"]
    assert report["evidence_ids"] == (["evidence-1"] if reuse else ["evidence-1", "evidence-2", "evidence-3"])
    assert report["workflow_tool_operations"] == (1 if reuse else 3)
    assert model.inputs[-1]["reused_evidence_ids"] == (["evidence-1"] if reuse else [])
    assert [turn["inspection_available"] for turn in model.inputs] == [True, True, True, False]


def test_cancel_during_scoped_investigation_selection_discards_late_reads(mode_gateway, monkeypatch):
    _, workflow = mode_gateway
    monkeypatch.setattr(PlanService, "inspect", lambda *_: pytest.fail("Cancelled task cannot read source"))

    class CancelledInvestigation(Decisions):
        def decide_sync(self, messages, schema):
            decision = super().decide_sync(messages, schema)
            task_id = self.inputs[-1]["task_id"]
            workflow.lane.cancel_tree(task_id)
            workflow.note_cancellation(task_id)
            return decision

    result = workflow.start("session", "Investigate app.py", mode="plan",
                            llm=CancelledInvestigation(investigation("app.py")))

    assert result.data["phase"] == "cancelled"
    assert workflow.board.get_task(result.data["task_id"]).mode_workflow["evidence"] == []


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
    (evidence_call("second", paths=["test_app.py"], level=2, prior_evidence=["evidence-1"]), 2,
     "The next permitted level is 1"),
    (evidence_call("second", "list_files", paths=["tests/**/*.py"]), 2,
     "Discovery paths must name literal files or directories"),
    (evidence_call("second", "list_files", paths=["/"]), 2,
     "Discovery paths must name literal files or directories"),
    (evidence_call("second", "list_files", paths=["runtime/../docs"]), 2,
     "Discovery paths must name literal files or directories"),
    (evidence_call("second", "list_files", paths=["./.env/"]), 2,
     "Protected repository paths are unavailable"),
    (evidence_call("second", "list_files", level=4), 2, "The next permitted level is 1"),
    (evidence_call("second", "list_files", level=4, reason=" "), 2, "Provide a reason for the requested level-4 scope"),
    (evidence_call("second", "repo_search", paths=["."], query="before"), 2,
     "Use list_files to list top-level files or select a narrower path."),
    (evidence_call("second", "repo_search", query="before"), 2, "Supply literal paths or a targeted glob"),
    (evidence_call("second", "repo_search", query="before", glob="**/*.py"), 2,
     "Repository-wide discovery requires an authorized level 4"),
    (evidence_call("second", "ls", paths=["docs"]), 2, "Send only tool ls"),
    (evidence_call("second", "repo_search", level=4, query="before", initial_repository_scope="explicit_audit"), 2,
     "Investigation reads must stay inside the selected paths"),
    (evidence_call("second", paths=[".env"]), 2, "Protected repository paths are unavailable"),
    (evidence_call("second", "apply_patch", patch="unauthorized"), 2, "Invalid ModeDecision"),
])
def test_investigation_batch_rejects_all_reads_if_any_selection_is_invalid(
    mode_gateway, monkeypatch, second, tool_budget, detail,
):
    gateway, workflow = mode_gateway
    gateway._autonomous_controller.config = AutonomousExecutionConfig(max_tool_calls=tool_budget)
    calls = []
    monkeypatch.setattr(PlanService, "inspect", lambda _, request: calls.append(request))
    selected = (investigation("Inspect app.py", evidence_call("first", paths=["app.py"]), second, paths=["app.py"])
                if second.args.get("initial_repository_scope") else inspection("app.py", second))
    model = Decisions(selected)

    result = workflow.start("session", "Inspect two files", mode="plan", llm=model)

    assert result.data["phase"] == "blocked"
    assert detail in result.message
    assert not calls


@pytest.mark.parametrize("mode", ["plan", "debug"])
def test_over_budget_read_batch_returns_control_without_a_partial_dispatch(mode_gateway, monkeypatch, mode):
    gateway, workflow = mode_gateway
    gateway._autonomous_controller.config = AutonomousExecutionConfig(max_tool_calls=1)
    seen = []
    inspect = PlanService.inspect

    def record(service, request):
        seen.append(request.paths)
        return inspect(service, request)

    monkeypatch.setattr(PlanService, "inspect", record)
    calls = [evidence_call("first", paths=["app.py"]), evidence_call("second", paths=["test_app.py"])]
    final = draft() if mode == "plan" else {
        "action": "diagnose", "reason": "Observed the cause", "root_cause": "Old handler value",
        "cause_evidence": ["evidence-1"],
    }
    oversized = (investigation("Compare two files", *calls, paths=["app.py", "test_app.py"])
                 if mode == "plan" else inspection(None, *calls))
    model = Decisions(oversized, inspection("app.py"), final)

    result = workflow.start("session", "Compare app.py with test_app.py", mode=mode, llm=model, no_code=True, save=False)

    assert result.status == "success", result.message
    assert model.inputs[1]["rejected_inspection_batch"] == {"required": 2, "remaining": 1}
    assert seen == [("app.py",)]
    assert result.data["repository_observations"]["workflow_tool_operations"] == 1


@pytest.mark.parametrize(("first", "second"), [
    ({"paths": ["runtime"]}, {"paths": ["./runtime/"]}),
    ({}, {"paths": ["."]}),
])
def test_investigation_list_files_deduplicates_aliases_before_dispatch(mode_gateway, first, second):
    gateway, workflow = mode_gateway
    gateway._autonomous_controller.config = AutonomousExecutionConfig(max_tool_calls=1)
    (gateway.root / "runtime").mkdir()
    (gateway.root / "runtime" / "handler.py").write_text("pass\n")
    model = Decisions(
        inspection(None, evidence_call("first", "list_files", **first), evidence_call("second", "list_files", **second)),
        {"action": "diagnose", "reason": "Inspected the layout", "uncertainty": "File contents are unknown"},
    )

    result = workflow.start("session", "List the files", mode="debug", llm=model)

    assert result.status == "success", result.message
    assert model.inputs[1]["deduplicated_inspection_requests"] == 1
    assert result.data["repository_observations"]["workflow_tool_operations"] == 1


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


def test_responses_provider_investigation_uses_only_coding_decisions(mode_gateway, monkeypatch):
    from mana_agent.llm.capabilities import ModelProfile, observation
    from mana_agent.llm.engine import NativeLLMEngine

    _, workflow = mode_gateway
    requests = []
    decisions = iter([investigation("app.py"), draft()])

    class Adapter:
        async def generate(self, request):
            requests.append(request)
            assert request.response_schema
            return ModelResponse(content=json.dumps({"decision": next(decisions)}))

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
    assert len(requests) == 2


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


@pytest.mark.parametrize("case", [
    "unknown_diagnosis", "unknown_repair", "diagnosis_only", "repeated_unknown", "provider_timeout", "stale_after_correction",
])
def test_debug_corrects_evidence_after_inspection_stops(mode_gateway, monkeypatch, case):
    import asyncio
    from mana_agent.approval.policy import set_approval_mode
    from mana_agent.llm.capabilities import ModelProfile, observation
    from mana_agent.llm.engine import NativeLLMEngine
    from mana_agent.llm.messages import ToolCall
    from mana_agent.services.execution_event_hub import conversation_events_path

    set_approval_mode("full_access")
    gateway, workflow = mode_gateway
    gateway._autonomous_controller.config = AutonomousExecutionConfig(max_no_progress_cycles=2)
    repair = draft()
    repair.update(action="repair", root_cause="The handler emits the old value.", cause_evidence=["evidence-1"])
    repair["draft"]["steps"][0]["changes"] = [{"path": "app.py", "operations": ["update"]}]
    bad_citation = "api_key=unrecognized-private-reference" if case == "provider_timeout" else "evidence-999"
    if case in {"unknown_diagnosis", "diagnosis_only"}:
        rejected = {
            "action": "diagnose", "reason": "The read shows the old value.", "root_cause": repair["root_cause"],
            "cause_evidence": ["evidence-1" if case == "diagnosis_only" else bad_citation],
        }
    else:
        rejected = {**repair, "cause_evidence": [bad_citation]}
    read = inspection("app.py")
    corrected = rejected if case == "repeated_unknown" else repair
    actions = iter([read, read, read, rejected, corrected])
    requests = []

    class Adapter:
        async def generate(self, request):
            requests.append(request)
            if not request.tools:
                return ModelResponse(content=json.dumps({"disposition": "in_scope", "reason": "Only app.py changed"}))
            if case == "provider_timeout" and len(requests) == 5:
                raise asyncio.TimeoutError()
            if case == "stale_after_correction" and len(requests) == 5:
                (gateway.root / "app.py").write_text("changed during the model request\n")
            return ModelResponse(tool_calls=[ToolCall(
                id=f"decision-{len(requests)}", name="record_decision", args={"decision": next(actions)},
            )])

    profile = ModelProfile(provider="openrouter", model_id="anthropic/claude-haiku-5.5", facts={
        name: observation(value, source="provider", confidence=1) for name, value in {
            "tool_calling": True, "input_text": True, "output_text": True, "native_structured_output": True,
            "context_window": 65536, "max_output_tokens": 1024,
        }.items()
    })
    model = NativeLLMEngine(provider=profile.provider, model=profile.model_id, api_key="fixture",
                            base_url="https://example.invalid", endpoint="anthropic_messages", adapter=Adapter())
    monkeypatch.setattr(model, "profile", lambda *_a, **_kw: profile)
    gateway._entry_router.llm = model
    proposals, checks = [], []

    def propose(*args, **kwargs):
        proposals.append(kwargs["execution_mode"])
        patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-before\n+after\n*** End Patch"
        return {"status": "completed", "output": json.dumps({"patch": patch})}

    async def verify(root, command, **kwargs):
        checks.append(command.argv)
        return {"passed": (root / "app.py").read_text() == "after\n", "exit_code": 0, "output": "after"}

    gateway._coding_agent = SimpleNamespace(generate=propose)
    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_isolated_verification", verify)
    result = workflow.start("session", "Fix the handler at app.py:1", mode="debug", repair=True,
                            workflow=workflow_choice(mode="debug"), source_message_ids=("user-1",))

    mode_requests = [request for request in requests if request.tools]
    assert len(mode_requests) == 5
    assert mode_requests[3].tools == mode_requests[4].tools
    stopped = json.loads(mode_requests[3].messages[1].content)
    assert stopped["inspection_stop"] == "Repository reads stopped yielding new evidence."
    assert stopped["inspection_available"] is False
    assert stopped["evidence_levels"] == {"evidence-1": 1}
    correction = mode_requests[4].messages[-1].content
    expected_error = (
        "The user requested a repair." if case == "diagnosis_only"
        else "cause_evidence must contain only exact observation IDs"
    )
    assert expected_error in correction
    assert bad_citation not in correction
    if case != "diagnosis_only":
        assert 'Valid references for this field: ["evidence-1"]' in correction
    task = workflow.board.get_task(result.data["task_id"])
    events = conversation_events_path(task.primary_repository_id, task.session_id).read_text()
    diagnostics = [event["metadata"]["decision_validation"] for line in events.splitlines()
                   if (event := json.loads(line))["type"] == "model.decision.rejected" and event["task_id"] == task.task_id]
    assert diagnostics[0]["correction_attempt"] == 0
    assert not diagnostics[0]["final"]
    if case != "diagnosis_only":
        assert diagnostics[0]["code"] == "mode_unknown_evidence_reference"
        assert diagnostics[0]["field"] == "cause_evidence"
        assert diagnostics[0]["available_refs"] == ["evidence-1"]
        assert diagnostics[0]["expected_format"] == "observation ID from evidence_levels"
        if case == "provider_timeout":
            assert diagnostics[0]["rejected_id"].startswith("<malformed reference ")
            assert "unrecognized-private-reference" not in events
        else:
            assert diagnostics[0]["rejected_id"] == bad_citation
    report = result.data["repository_observations"]
    if case == "stale_after_correction":
        assert result.data["phase"] == "blocked", result.message
        assert expected_error in result.message
        assert not proposals and not checks
        assert (gateway.root / "app.py").read_text() == "changed during the model request\n"
        return

    assert report["evidence_ids"] == ["evidence-1"]
    if case in {"repeated_unknown", "provider_timeout"}:
        assert result.data["phase"] == "blocked", result.message
        assert report["workflow_tool_operations"] == 1
        assert ("timed out" if case == "provider_timeout" else expected_error) in result.message
        assert bad_citation not in result.message
        assert not proposals and not checks
        assert len(diagnostics) == 2
        assert diagnostics[1]["correction_attempt"] == 1
        assert diagnostics[1]["final"]
        assert diagnostics[1]["final_reason"]
    else:
        assert len(diagnostics) == 1
        assert result.status == "success", result.message
        assert result.data["changed_files"] == ["app.py"]
        assert proposals == ["read_only"]
        assert checks == [("python", "app.py")]


def test_scoped_investigation_accepts_nested_request_arrays_in_strict_json():
    from mana_agent.agent.plan_service import InvestigationScope, ModeDecision

    decision = investigation("Explain handler behavior",
        evidence_call("source", paths=["src/app.py"]),
        evidence_call("callers", "repo_search", paths=["src"], query="handler"), paths=["src"])
    result = ModeDecision.model_validate_json(json.dumps({"decision": decision}), strict=True,
        context={"investigation_scope": InvestigationScope({})})

    assert result.decision.paths == ("src",)
    assert [item.paths for item in result.decision.requests] == [("src/app.py",), ("src",)]


def test_checkout_git_verification_is_invalid_before_docker(mode_gateway, monkeypatch):
    import asyncio

    import mana_agent.gateway.mode_workflow as mode_module
    from mana_agent.agent.mode import VerificationCommand

    gateway, workflow = mode_gateway
    decision = draft()
    decision["draft"]["steps"].insert(0, {"step_id": "branch", "instruction": "Check the current branch",
        "verification": [{"argv": ["git", "rev-parse", "--abbrev-ref", "HEAD"]}],
        "acceptance_criteria": ["The expected branch is active"]})
    real_which = mode_module.shutil.which
    monkeypatch.setattr(mode_module.shutil, "which",
        lambda name, *a, **kw: pytest.fail("Checkout checks must fail before Docker lookup") if name == "docker"
        else real_which(name, *a, **kw))

    result = workflow.start("session", "Fix the handler", mode="auto", repair=True,
                            workflow=workflow_choice(), llm=Decisions(decision))
    assert result.data["phase"] == "blocked"
    assert "source snapshot without .git" in result.message
    assert result.data["changed_files"] == []
    # A saved revision can still be decoded. Its stale checkout command gets
    # the targeted runtime error instead of a missing-image diagnosis.
    with pytest.raises(mode_module.VerificationUnavailable, match=r"no \.git metadata"):
        asyncio.run(mode_module.run_isolated_verification(gateway.root,
            VerificationCommand(argv=("git", "rev-parse", "--abbrev-ref", "HEAD")), task_id="legacy"))


@pytest.mark.parametrize("setup_exit", [0, 7])
def test_adaptive_coding_verification_runs_setup_in_the_same_snapshot(tmp_path, monkeypatch, setup_exit):
    import asyncio

    import mana_agent.execution.providers.local_docker as docker
    import mana_agent.gateway.mode_workflow as mode_module
    from mana_agent.coding.adaptive import AdaptiveVerificationCommand
    from mana_agent.execution.models import ExecutionResult, NetworkMode

    root = tmp_path.resolve()
    (root / "app.py").write_text("source\n")
    calls = []
    cleanups = []

    def run(argv, **kwargs):
        return SimpleNamespace(stdout=b"app.py\0" if argv[0] == "git" else "fixture-image", returncode=0)

    class Provider:
        docker_binary = "docker"
        default_image = "fixture"

        async def provision(self, spec):
            assert spec.network.mode == NetworkMode.DENY_ALL and spec.read_only_root
            assert spec.repository_source != root
            return spec.repository_source

        async def start(self, handle):
            return handle

        async def execute(self, handle, request):
            calls.append(request.argv)
            assert (handle / request.environment["TMPDIR"].removeprefix("/workspace/")).is_dir()
            if request.argv == ["make"]:
                (handle / "build-marker").touch()
                exit_code = setup_exit
            else:
                assert (handle / "build-marker").is_file()
                exit_code = 0
            return ExecutionResult(exit_code=exit_code, started_at=utc_now(), completed_at=utc_now(),
                                   provider="fixture", sandbox_id="fixture")

        async def cleanup(self, handle):
            cleanups.append(handle)

    monkeypatch.setattr(mode_module.shutil, "which", lambda name: "/bin/docker")
    monkeypatch.setattr(mode_module.subprocess, "run", run)
    monkeypatch.setattr(docker, "LocalDockerProvider", Provider)
    result = asyncio.run(mode_module.run_isolated_verification(root,
        AdaptiveVerificationCommand(argv=("ctest",), setup_argv=(("make",),)), task_id="fixture"))

    assert result["passed"] is (setup_exit == 0)
    assert result["exit_code"] == (None if setup_exit else 0)
    assert result["setup_receipts"][0]["exit_code"] == setup_exit
    assert calls == ([["make"]] if setup_exit else [["make"], ["ctest"]])
    assert len(cleanups) == 1
    assert not (root / "build-marker").exists()


@pytest.mark.parametrize("revised_outcome", [
    "patch_pending", "complete", "lost_required_check", "requires_pre_patch", "repeats_pre_patch",
])
def test_unavailable_pre_patch_verification_allows_model_directed_repair(mode_gateway, monkeypatch, revised_outcome):
    import mana_agent.gateway.mode_workflow as mode_module
    from mana_agent.approval.policy import set_approval_mode

    set_approval_mode("full_access")
    gateway, workflow = mode_gateway
    precheck = {"step_id": "baseline", "instruction": "Confirm the diagnostic",
        "verification": [{"argv": ["python", "baseline.py"], "expected": "fail", "output_marker": "old value"}],
        "acceptance_criteria": ["The baseline diagnostic is reproduced"]}
    first_plan = draft()
    first_plan.update(action="repair", root_cause="The handler emits the old value", cause_evidence=["evidence-1"])
    first_plan["draft"]["steps"].insert(0, precheck)
    first_plan["draft"]["steps"][1]["depends_on"] = ["baseline"]
    revised = draft()
    revised.update(action="repair", root_cause=first_plan["root_cause"], cause_evidence=["evidence-1"],
                   uncertainty="The baseline command could not run")
    if revised_outcome == "lost_required_check":
        revised["draft"]["steps"][-1]["verification"] = [{"argv": ["python", "unrelated.py"]}]
    if revised_outcome == "requires_pre_patch":
        revised = {"action": "block", "reason": "The requested baseline must run before editing."}
    if revised_outcome == "repeats_pre_patch":
        revised = first_plan
    model = Decisions(inspection("app.py"), first_plan, revised,
                      {"disposition": "in_scope", "reason": "Only the handler changes"})
    gateway._entry_router.llm = model
    operations = []
    patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-before\n+after\n*** End Patch"

    def propose(*args, **kwargs):
        operations.append("patch")
        return {"status": "completed", "output": json.dumps({"patch": patch})}

    async def verify(root, command, **kwargs):
        operations.append(command.argv[-1])
        if command.argv[-1] == "baseline.py":
            assert (root / "app.py").read_text() == "before\n"
        else:
            assert (root / "app.py").read_text() == "after\n"
        if command.argv[-1] == "baseline.py" or revised_outcome == "patch_pending":
            raise mode_module.VerificationUnavailable("Docker image missing")
        return {"passed": True, "exit_code": 0, "output": "checked"}

    gateway._coding_agent = SimpleNamespace(generate=propose)
    monkeypatch.setattr(mode_module, "run_isolated_verification", verify)
    result = workflow.start("session", "Fix the handler", mode="debug", repair=True,
                            workflow=workflow_choice(mode="debug"))

    review = model.inputs[2]
    assert review["verification_unavailable"]["command"]["argv"] == ["python", "baseline.py"]
    assert review["verification_failure"] is None
    assert review["allowed_actions"] == ["block", "clarify", "repair"]
    assert [item["id"] for item in review["evidence"]] == ["evidence-1"]
    if revised_outcome in {"lost_required_check", "requires_pre_patch", "repeats_pre_patch"}:
        assert result.data["phase"] == "blocked"
        assert operations == ["baseline.py"]
        assert result.data["changed_files"] == []
        assert result.data["repository_observations"]["evidence_ids"] == ["evidence-1"]
        expected = {"lost_required_check": "required regression checks", "requires_pre_patch": "before editing",
                    "repeats_pre_patch": "Put the required isolated checks after the changes"}
        assert expected[revised_outcome] in result.message
    else:
        assert operations == ["baseline.py", "patch", "app.py"]
        assert result.data["changed_files"] == ["app.py"]
        assert result.data["goal_satisfied"] is (revised_outcome == "complete")
        if revised_outcome == "patch_pending":
            assert "**Verification pending.**" in result.message
            assert result.data["verification"] == []


def test_missing_verification_image_reports_applied_changes_and_can_resume(mode_gateway, monkeypatch):
    import mana_agent.gateway.mode_workflow as mode_module
    from mana_agent.approval.policy import set_approval_mode

    set_approval_mode("full_access")
    gateway, workflow = mode_gateway
    subprocess.run(["git", "init", "-q"], cwd=gateway.root, check=True)
    plan = draft()
    apply_step, verify_step = plan["draft"]["steps"]
    apply_step["verification"] = verify_step["verification"]
    plan["draft"]["steps"] = [apply_step]
    gateway._entry_router.llm = Decisions(
        inspection("app.py"), plan, {"disposition": "in_scope", "reason": "Matches the step"},
    )
    patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-before\n+after\n*** End Patch"
    proposals, activity = [], []

    def propose(*args, **kwargs):
        proposals.append(patch)
        return {"status": "completed", "output": json.dumps({"patch": patch})}

    gateway._coding_agent = SimpleNamespace(generate=propose)
    real_run = subprocess.run
    real_which = mode_module.shutil.which

    def run(argv, **kwargs):
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="No such image")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(mode_module.subprocess, "run", run)
    monkeypatch.setattr(mode_module.shutil, "which",
                        lambda name, *args, **kwargs: "/fixture/docker" if name == "docker"
                        else real_which(name, *args, **kwargs))
    monkeypatch.setattr(mode_module, "emit_activity", lambda event, **fields: activity.append(fields))

    blocked = workflow.start("session", "Fix the handler", mode="auto", repair=True, workflow=workflow_choice())
    task_id = blocked.data["task_id"]
    assert blocked.status == "error"
    assert blocked.data["phase"] == "blocked"
    assert blocked.data["goal_satisfied"] is False
    assert blocked.data["changed_files"] == ["app.py"]
    assert blocked.data["verification"] == []
    assert blocked.data["repository_observations"]["evidence_ids"] == ["evidence-1"]
    assert (gateway.root / "app.py").read_text() == "after\n"
    assert blocked.message.startswith("Changed `app.py`.\n\n**Verification pending.**")
    assert "local image `python:3.12`" in blocked.message
    assert f"`/task resume {task_id}`" in blocked.message
    assert "The handler returns the wrong value" not in blocked.message
    assert "evidence-1" not in blocked.message
    assert "('python', 'app.py')" not in blocked.message
    described = workflow.describe(workflow.board.get_task(task_id))
    assert described.status == blocked.status
    assert described.data == blocked.data
    assert described.message == blocked.message
    pending = next(record for record in activity if record["title"] == "Verification pending")
    assert pending["status"] == "unverified"
    assert pending["metadata"]["verification_status"] == "not_run"
    assert "('python', 'app.py')" in pending["detail"]
    assert pending["operation_id"] == next(record["operation_id"] for record in activity
                                           if record["title"] == "Running verification command")

    async def verify(root, command, **kwargs):
        assert command.argv == ("python", "app.py")
        assert (root / "app.py").read_text() == "after\n"
        return {"passed": True, "exit_code": 0, "output": "checked"}

    monkeypatch.setattr(mode_module, "run_isolated_verification", verify)
    resumed = workflow.retry(task_id)
    assert resumed.status == "success", resumed.message
    assert resumed.message == "Changed `app.py`. All required verification checks passed."
    assert proposals == [patch]


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
    assert final.status == "error"
    assert final.data["phase"] == "blocked", final.message
    assert final.data["verification"][-1]["exit_code"] == 1
    assert final.message.startswith("Changed `app.py`.\n\n**Verification failed.**")
    assert "Coding Activity" in final.message
    assert f"`/task resume {task_id}`" in final.message
    assert "fixture result" not in final.message
    assert "# Implementation Plan" not in final.message
    assert workflow.retry(task_id).status == "success"
    assert proposals == [patch]


@pytest.mark.parametrize("malformed", [False, True])
def test_batch_proposal_wakes_only_the_bound_step(mode_gateway, monkeypatch, malformed):
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
    proposals = []
    monkeypatch.setattr("mana_agent.batch.watcher.ensure_batch_watcher", lambda *_a: None)
    monkeypatch.setattr("mana_agent.batch.operations.BatchOperationStore",
                        lambda _store: SimpleNamespace(load_run=lambda task_id: driver[task_id]))

    def propose(_prompt, *, gateway_task_id, batch_resume, **kwargs):
        proposals.append(gateway_task_id)
        if gateway_task_id not in driver:
            workflow.lane.checkpoint(gateway_task_id, boundary="before_batch")
            operation = {"task_id": gateway_task_id, "operation_id": "operation", "batch_id": "provider-job"}
            driver[gateway_task_id] = {"resume": batch_resume, "pending_operation": operation}
            return {"status": "waiting", "pending_operation": operation}
        patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-before\n+after\n*** End Patch"
        if malformed:
            patch += "\n@@\n+unexpected\n*** End Patch"
        return {"status": "completed", "output": json.dumps({"patch": patch})}

    gateway._coding_agent = SimpleNamespace(_internal_engine=SimpleNamespace(model="fixture", provider="fixture"),
                                           generate=propose)

    async def verify(*_a, **_kw):
        return {"passed": True, "exit_code": 0, "output": "verified"}

    monkeypatch.setattr("mana_agent.gateway.mode_workflow.run_isolated_verification", verify)
    plan = workflow.start("session", "Fix the handler with Batch", save=False,
                          execution_policy={"mode": "batch", "required": True})
    task_id, item = plan.data["task_id"], plan.data["inbox_id"]
    approval_finished = Event()
    resume = workflow.resume

    def resume_approval(*args):
        try:
            return resume(*args)
        finally:
            approval_finished.set()

    monkeypatch.setattr(workflow, "resume", resume_approval)
    workflow.respond("session", item, ResponseOperation.APPROVE, frontend="cli")

    assert approval_finished.wait(15), "Approval did not reach the Batch wait"
    child_id = workflow.board.get_task(task_id).mode_workflow.get("batch_child")
    assert child_id
    waiting = gateway._lane_coordinator.execution_supervisor.get_task(child_id)
    assert waiting.parent_task_id == task_id
    assert waiting.state is ExecutionState.WAITING
    assert waiting.waiting_reason == "waiting_for_batch"
    assert (gateway.root / "app.py").read_text() == "before\n"
    with pytest.raises(PermissionError):
        workflow.resume_batch(task_id, child_id, "another operation", driver[child_id])
    workflow.resume_batch(task_id, child_id, "operation", driver[child_id])
    assert proposals == [child_id, child_id]
    state = workflow.board.get_task(task_id).mode_workflow
    if malformed:
        assert state["phase"] == "blocked"
        assert "premature *** End Patch at patch line 6" in state["result"]
        assert (gateway.root / "app.py").read_text() == "before\n"
    else:
        assert state["phase"] == "completed"
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
