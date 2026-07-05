from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mana_agent.commands.cli import app
from mana_agent.commands.cli_internal import _record_multi_agent_request
from mana_agent.commands import cli_internal
from mana_agent.multi_agent import MainAgent
from mana_agent.multi_agent.agents.coding_agent import CodingAgent
from mana_agent.multi_agent.agents.verifier_agent import VerifierAgent
from mana_agent.multi_agent.communication.decision_room import DecisionRoom
from mana_agent.multi_agent.communication.message_bus import MessageBus
from mana_agent.multi_agent.core.errors import AgentRegistryError, InvalidTaskTransition, ToolPermissionError
from mana_agent.multi_agent.core.ids import (
    new_agent_id,
    new_decision_id,
    new_discussion_id,
    new_message_id,
    new_queue_job_id,
    new_subagent_id,
    new_task_id,
    new_trace_id,
)
from mana_agent.multi_agent.core.types import (
    AgentState,
    AgentRole,
    MessageType,
    QueueJobStatus,
    QueueJobType,
    TaskStatus,
)
from mana_agent.multi_agent.runtime.model_levels import MODEL_LEVEL_2_CODING, model_level_for_role
from mana_agent.multi_agent.queue.queue_manager import QueueManager
from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
from mana_agent.multi_agent.routing.router import Router
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.multi_agent.tools.permissions import assert_shell_allowed


def test_id_generation_is_readable_and_unique():
    values = [
        new_agent_id("main"),
        new_subagent_id("coding tests"),
        new_task_id(),
        new_queue_job_id(),
        new_message_id(),
        new_discussion_id(),
        new_decision_id(),
        new_trace_id(),
    ]
    assert len(values) == len(set(values))
    assert values[0].startswith("agent_main_")
    assert values[1].startswith("subagent_coding_tests_")
    assert values[2].startswith("task_")


def test_taskboard_create_update_save_load_and_invalid_transition(tmp_path):
    board = TaskBoard(tmp_path)
    task = board.create_task(title="Test", user_request="plan test")
    board.update_status(task.task_id, TaskStatus.PLANNING)
    board.add_assumption(task.task_id, "Assume compatibility.")
    board.add_evidence(task.task_id, "Evidence row.")
    board.add_files_to_inspect(task.task_id, ["src/example.py"])
    reloaded = TaskBoard(tmp_path)
    loaded = reloaded.get_task(task.task_id)
    assert loaded.assumptions == ["Assume compatibility."]
    assert loaded.evidence == ["Evidence row."]
    with pytest.raises(InvalidTaskTransition):
        reloaded.update_status(task.task_id, TaskStatus.DONE)


def test_message_bus_send_broadcast_and_thread(tmp_path):
    bus = MessageBus(tmp_path)
    sent = bus.send(
        task_id="task_1",
        from_agent_id="agent_a",
        to_agent_id="agent_b",
        message_type=MessageType.QUESTION,
        content="Need evidence.",
        discussion_id="discussion_1",
    )
    bus.broadcast("task_1", "agent_a", MessageType.EVIDENCE, "Broadcast evidence.")
    assert sent in bus.inbox("agent_b")
    assert len(bus.thread("discussion_1")) == 1


def test_decision_room_open_close_records_taskboard_decision(tmp_path):
    board = TaskBoard(tmp_path)
    task = board.create_task(title="Decision", user_request="edit file")
    bus = MessageBus(tmp_path)
    room = DecisionRoom(tmp_path, board, bus)
    discussion = room.open_discussion(task.task_id, "Mutation decision", ["agent_head", "agent_planner"], created_by_agent_id="agent_head")
    decision = room.close_with_decision(
        task_id=task.task_id,
        discussion_id=discussion.discussion_id,
        made_by_agent_id="agent_head",
        summary="Use coding route.",
        rationale_summary="Mutation requires verification.",
        selected_route="coding",
        assigned_agent_ids=["agent_coding"],
        required_verification=["pytest"],
    )
    assert decision.decision_id in board.get_task(task.task_id).decision_ids
    assert room.discussions.discussions[discussion.discussion_id].final_decision_id == decision.decision_id


def test_agent_registry_hierarchy_and_cycle_guard():
    registry = AgentRegistry()
    assert registry.find_by_role(AgentRole.MAIN).agent_id.startswith("agent_main_")
    coding = registry.find_by_role(AgentRole.CODING)
    subagent = registry.create_subagent(AgentRole.CODING, coding.agent_id, ["coding"])
    assert subagent.parent_agent_id == coding.agent_id
    with pytest.raises(AgentRegistryError):
        registry.set_parent(coding.agent_id, subagent.agent_id)


def test_router_selects_required_routes():
    router = Router()
    assert router.route(task_id="task_1", user_request="/plan update docs").route_name == "planning"
    assert router.route(task_id="task_1", user_request="/analyze repo").route_name == "analyze"
    assert router.route(task_id="task_1", user_request="edit README.md").route_name == "coding"
    assert router.route(task_id="task_1", user_request="run pytest").route_name == "tool"
    readme_route = router.route(task_id="task_1", user_request="project architecture changed, update README.md")
    assert readme_route.task_size == "large"
    assert readme_route.required_subagents == ["repo_inventory", "docs"]


def test_queue_manager_serializes_write_jobs_and_tracks_status(tmp_path):
    board = TaskBoard(tmp_path)
    task = board.create_task(title="Queue", user_request="run status")
    queue = QueueManager(tmp_path, taskboard=board)
    job = queue.enqueue(
        task_id=task.task_id,
        requested_by_agent_id="agent_tool",
        approved_by_agent_id="agent_head_decision_0001",
        job_type=QueueJobType.GIT_STATUS,
        payload={},
        purpose="Inspect repository status.",
        priority=1,
        lock_key="repo",
        requires_write_lock=True,
    )
    assert job.queue_job_id == job.job_id
    assert job.tool_name == "git_status"
    assert job.tool_args == {}
    assert job.status == QueueJobStatus.QUEUED
    assert job.approved_by_agent_id == "agent_head_decision_0001"
    ran = queue.run_until_idle()
    assert ran == [job]
    assert queue.get_job(job.job_id).status.value in {"done", "failed"}
    assert job.job_id in board.get_task(task.task_id).queue_job_ids
    assert job.result_summary


def test_queue_manager_runs_batch_read_through_tools_manager(tmp_path):
    (tmp_path / "README.md").write_text("# Test\n", encoding="utf-8")
    board = TaskBoard(tmp_path)
    task = board.create_task(title="Batch read", user_request="read docs")
    queue = QueueManager(tmp_path, taskboard=board)
    job = queue.enqueue(
        task_id=task.task_id,
        requested_by_agent_id="agent_coding_0001",
        approved_by_agent_id="agent_head_decision_0001",
        job_type=QueueJobType.REPO_BATCH_READ,
        payload={"files": ["README.md"]},
        purpose="Batch read selected repository files.",
    )

    queue.run_until_idle()

    assert job.status == QueueJobStatus.DONE
    assert job.result and job.result["ok"] is True
    assert job.result["files"][0]["path"] == "README.md"


def test_patch_context_failure_requires_fresh_read(tmp_path):
    (tmp_path / "README.md").write_text("current\n", encoding="utf-8")
    board = TaskBoard(tmp_path)
    task = board.create_task(title="Patch", user_request="edit README.md")
    queue = QueueManager(tmp_path, taskboard=board)
    job = queue.enqueue(
        task_id=task.task_id,
        requested_by_agent_id="agent_coding_0001",
        approved_by_agent_id="agent_head_decision_0001",
        job_type=QueueJobType.APPLY_PATCH,
        payload={"patch": "*** Begin Patch\n*** Update File: README.md\n@@\n-stale\n+fresh\n*** End Patch\n"},
        purpose="Apply README update.",
    )

    queue.run_until_idle()

    assert job.status == QueueJobStatus.FAILED
    assert "reread target file" in str(job.error)


def test_tools_manager_blocks_dangerous_shell_commands():
    with pytest.raises(ToolPermissionError):
        assert_shell_allowed("git reset --hard")
    with pytest.raises(ToolPermissionError):
        assert_shell_allowed("cat .env")
    assert_shell_allowed("python -m compileall src")


def test_coding_agent_cannot_directly_execute_tools(tmp_path):
    main = MainAgent(tmp_path)
    agent = main.agents[AgentRole.CODING]
    assert isinstance(agent, CodingAgent)
    with pytest.raises(PermissionError):
        agent.execute_tool_directly("git status")


def test_verifier_agent_records_failed_verification(tmp_path):
    main = MainAgent(tmp_path)
    task = main.taskboard.create_task(title="Verify", user_request="verify")
    verifier = main.agents[AgentRole.VERIFIER]
    assert isinstance(verifier, VerifierAgent)
    result = verifier.record_failed_verification(task.task_id, "pytest failed")
    assert result in main.taskboard.get_task(task.task_id).verification_results


def test_verifier_does_not_mark_planned_commands_as_passed(tmp_path):
    main = MainAgent(tmp_path)
    task = main.taskboard.create_task(title="Verify", user_request="verify")
    verifier = main.agents[AgentRole.VERIFIER]
    assert isinstance(verifier, VerifierAgent)
    result = verifier.verify_no_mutation(task.task_id, ["python -m compileall src"])
    assert result.passed is False
    assert result.risks == ["planned_verification_not_executed"]


def test_main_agent_routes_chat_analyze_and_plan(tmp_path):
    assert MainAgent(tmp_path).run_user_request("hello", entrypoint="chat").route_name == "simple"
    assert MainAgent(tmp_path).run_user_request("scan repo", entrypoint="analyze").route_name == "analyze"
    assert MainAgent(tmp_path).run_user_request("update docs", entrypoint="plan").route_name == "planning"


def test_main_agent_records_large_docs_subagents_and_deactivates_them(tmp_path):
    main = MainAgent(tmp_path)
    result = main.run_user_request("project architecture changed, update README.md, cannot use diff", entrypoint="chat")
    task = main.taskboard.get_task(result.task_id)
    assert result.route_name == "coding"
    assert result.task_size == "large"
    assert result.required_subagents == ["repo_inventory", "docs"]
    assert len(task.assigned_subagent_ids) == 2
    assert all(main.registry.agents[subagent_id].state == AgentState.DONE for subagent_id in task.assigned_subagent_ids)
    assert any("deactivated" in item for item in task.evidence)


def test_model_levels_are_configurable_by_tier(monkeypatch):
    monkeypatch.setenv("MANA_MODEL_CODING", "MODEL_LEVEL_2_CUSTOM")
    assert model_level_for_role(AgentRole.CODING).model_level == "MODEL_LEVEL_2_CUSTOM"
    monkeypatch.delenv("MANA_MODEL_CODING")
    assert model_level_for_role(AgentRole.CODING).model_level == MODEL_LEVEL_2_CODING


def test_cli_commands_exist_and_record_multi_agent_route(tmp_path):
    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "chat" in help_result.output
    assert "analyze" in help_result.output
    assert "plan" in help_result.output
    task_id = _record_multi_agent_request(tmp_path, "plan a change", entrypoint="plan")
    assert task_id.startswith("task_")


def test_public_command_routes_once_when_root_dispatches_plan(monkeypatch, tmp_path):
    calls: list[tuple[str, str]] = []

    class _FakeMainAgent:
        def __init__(self, root):
            self.root = root

        def run_user_request(self, request: str, *, entrypoint: str = "chat"):
            calls.append((entrypoint, request))
            return SimpleNamespace(task_id=f"task_{len(calls):06d}", route_name="planning", answer="", required_agents=[])

    monkeypatch.setattr(cli_internal, "MainAgent", _FakeMainAgent)
    result = CliRunner().invoke(app, ["--plan", "--repo", str(tmp_path)], input="Add CLI banner\n")

    assert result.exit_code == 0
    assert calls == [("root", "root --plan")]


def test_public_command_callbacks_route_through_main_agent(monkeypatch, tmp_path):
    calls: list[tuple[str, str]] = []

    class _FakeMainAgent:
        def __init__(self, root):
            self.root = root

        def run_user_request(self, request: str, *, entrypoint: str = "chat"):
            calls.append((entrypoint, request))
            return SimpleNamespace(task_id=f"task_{len(calls):06d}", route_name="simple", answer="", required_agents=[])

    class _FakeAnalyzeResult:
        artifacts: dict[str, str] = {}
        errors: list[str] = []
        report: dict = {
            "inventory": {},
            "architecture": {},
            "risks": {},
            "recommendations": {},
            "dependencies": {},
            "symbols": {},
        }

    class _FakeProjectAnalyzeService:
        def run(self, *args, **kwargs):
            return _FakeAnalyzeResult()

    monkeypatch.setattr(cli_internal, "MainAgent", _FakeMainAgent)
    monkeypatch.setattr(cli_internal, "ProjectAnalyzeService", _FakeProjectAnalyzeService)

    runner = CliRunner()
    assert runner.invoke(app, ["plan", "--repo", str(tmp_path), "--no-code", "Add CLI smoke"]).exit_code == 0
    assert runner.invoke(app, ["skills", "list", "--repo", str(tmp_path)]).exit_code == 0
    assert runner.invoke(app, ["skills", "show", "cli", "--repo", str(tmp_path)]).exit_code == 0
    assert runner.invoke(app, ["analyze", "--repo", str(tmp_path), "--max-files", "1"]).exit_code == 0
    continue_result = runner.invoke(app, ["continue", "--root-dir", str(tmp_path), "--run-id", "missing"])
    assert continue_result.exit_code != 0

    assert ("plan", "Add CLI smoke") in calls
    assert ("skills", "skills list") in calls
    assert ("skills", "skills show cli") in calls
    assert ("analyze", ".") in calls
    assert ("continue", "continue run missing") in calls


def test_no_stale_mana_agent_llm_imports_remain():
    roots = [
        "src/mana_agent",
        "tests",
        "docs",
        "README.md",
        "pyproject.toml",
    ]
    offenders: list[str] = []
    for root in roots:
        path = __import__("pathlib").Path(root)
        files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
        for item in files:
            item_path = item.as_posix()
            if "docs/analyze" in item_path or "__pycache__" in item_path or item.suffix == ".pyc":
                continue
            text = item.read_text(encoding="utf-8", errors="ignore")
            legacy_module = "mana_agent" + ".llm"
            legacy_module_exec = "python -m " + legacy_module
            legacy_path = "src/mana_agent" + "/llm"
            if legacy_module in text or legacy_module_exec in text or legacy_path in text:
                offenders.append(item_path)
    assert offenders == []


def test_no_multi_agent_disable_flag_or_env_bypass():
    runner = CliRunner()
    help_result = runner.invoke(app, ["chat", "--help"])
    assert help_result.exit_code == 0
    assert "--no-multi-agent" not in help_result.output
    bypass = "MANA_" + "MULTI_AGENT=0"
    assert bypass not in help_result.output
