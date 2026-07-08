from __future__ import annotations

import json
import subprocess
from pathlib import Path
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
    ExecutionContext,
    MessageType,
    QueueJobStatus,
    QueueJobType,
    TaskStatus,
)
from mana_agent.multi_agent.observability.trace import TraceWriter
from mana_agent.multi_agent.runtime.model_levels import (
    MODEL_LEVEL_1_FAST_TOOL,
    MODEL_LEVEL_2_CODING,
    MODEL_LEVEL_3_HIGH_REASONING,
    model_level_for_role,
    resolve_model_for_role,
)
from mana_agent.multi_agent.queue.queue_manager import QueueManager
from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
from mana_agent.multi_agent.routing.hierarchy import HierarchyPolicy, HierarchyViolation
from mana_agent.multi_agent.routing.router import Router
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
from mana_agent.multi_agent.tools.permissions import assert_shell_allowed
from mana_agent.services.memory_service import MultiAgentMemoryService


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)


def _init_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    init = _git(path, "init", "-b", "main")
    if init.returncode != 0:
        assert _git(path, "init").returncode == 0
        assert _git(path, "branch", "-M", "main").returncode == 0
    assert _git(path, "config", "user.name", "Mana Agent Test").returncode == 0
    assert _git(path, "config", "user.email", "test@example.com").returncode == 0
    (path / "README.md").write_text("hello\n", encoding="utf-8")
    assert _git(path, "add", "README.md").returncode == 0
    assert _git(path, "commit", "-m", "test: initial commit").returncode == 0
    return path


def _git_job_commands(main: MainAgent, task_id: str) -> list[list[str]]:
    commands: list[list[str]] = []
    for job in main.queue_manager.jobs_for_task(task_id):
        if job.job_type != QueueJobType.GIT:
            continue
        nested = job.payload.get("args") if isinstance(job.payload.get("args"), dict) else {}
        raw = nested.get("args") if isinstance(nested, dict) else None
        if isinstance(raw, list):
            commands.append([str(item) for item in raw])
    return commands


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


def test_duplicate_task_not_created_twice(tmp_path):
    memory = MultiAgentMemoryService(root=tmp_path)
    board = TaskBoard(tmp_path, memory_service=memory)

    first = board.create_task(title="Docs", user_request="Update README.md", related_files=["README.md"])
    second = board.create_task(title="Docs again", user_request="Update README.md", related_files=["./README.md"])

    assert first.memory_status["duplicate_checked"] is True
    assert second.memory_status["duplicate_of"] == first.task_id
    assert second.status == TaskStatus.SKIPPED
    assert memory.task_records[second.task_id].duplicate_of == first.task_id


def test_duplicate_task_merged_into_existing_task(tmp_path):
    memory = MultiAgentMemoryService(root=tmp_path)
    board = TaskBoard(tmp_path, memory_service=memory)
    first = board.create_task(title="Docs", user_request="Update README.md", related_files=["README.md"])
    duplicate = board.create_task(title="Docs duplicate", user_request="Update README.md", related_files=["README.md"])

    assert duplicate.status == TaskStatus.SKIPPED
    assert board.get_task(duplicate.task_id).blockers == [f"duplicate_of:{first.task_id}"]


def test_queue_rejects_duplicate_fingerprint(tmp_path):
    memory = MultiAgentMemoryService(root=tmp_path)
    board = TaskBoard(tmp_path, memory_service=memory)
    task = board.create_task(title="Queue", user_request="run status")
    queue = QueueManager(tmp_path, taskboard=board, memory_service=memory)

    first = queue.enqueue(
        task_id=task.task_id,
        requested_by_agent_id="agent_tool_0001",
        job_type=QueueJobType.GIT_STATUS,
        payload={},
        purpose="Inspect status",
    )
    second = queue.enqueue(
        task_id=task.task_id,
        requested_by_agent_id="agent_tool_0001",
        job_type=QueueJobType.GIT_STATUS,
        payload={},
        purpose="Inspect status again",
    )

    assert first.status == QueueJobStatus.QUEUED
    assert second.status == QueueJobStatus.CANCELLED
    assert second.duplicate_of == first.job_id


def test_multi_agent_file_read_is_direct_without_duplicate_cache(tmp_path):
    (tmp_path / "README.md").write_text("# One\n", encoding="utf-8")
    memory = MultiAgentMemoryService(root=tmp_path)

    first_content, first_record, first_hit = memory.read_file_with_memory(
        file_path="README.md",
        task_id="task_1",
        agent_id="agent_coding_0001",
    )
    second_content, second_record, second_hit = memory.read_file_with_memory(
        file_path="./README.md",
        task_id="task_1",
        agent_id="agent_coding_0001",
    )

    assert first_content == second_content == "# One\n"
    assert first_hit is False
    assert second_hit is False
    assert first_record.content_hash == second_record.content_hash


def test_multi_agent_file_read_reflects_hash_changed_without_storing_cache(tmp_path):
    target = tmp_path / "README.md"
    target.write_text("# One\n", encoding="utf-8")
    memory = MultiAgentMemoryService(root=tmp_path)
    memory.read_file_with_memory(file_path="README.md", task_id="task_1", agent_id="agent_coding_0001")

    target.write_text("# Two\n", encoding="utf-8")
    content, record, cache_hit = memory.read_file_with_memory(
        file_path="README.md",
        task_id="task_1",
        agent_id="agent_coding_0001",
    )

    assert content == "# Two\n"
    assert cache_hit is False
    assert record.changed_since_last_read is False


def test_agent_receives_scoped_memory_bundle(tmp_path):
    memory = MultiAgentMemoryService(root=tmp_path)
    normalized, fingerprint = memory.normalize_task(goal="Update docs", target_files=["README.md"])
    memory.register_task(task_id="task_1", normalized_goal=normalized, fingerprint=fingerprint, related_files=["README.md"])

    bundle = memory.build_bundle(
        agent_id="agent_coding_0001",
        agent_role=AgentRole.CODING,
        task_id="task_1",
        target_files=["README.md"],
    )

    assert bundle.agent_id == "agent_coding_0001"
    assert bundle.privilege_level == "coding"
    assert bundle.relevant_task_memory[0]["task_id"] == "task_1"


def test_lower_agent_cannot_access_upper_memory(tmp_path):
    (tmp_path / "secret.md").write_text("upper only\n", encoding="utf-8")
    memory = MultiAgentMemoryService(root=tmp_path)
    memory.project_memory.append({"fact": "full project architecture"})
    memory.read_file_with_memory(file_path="secret.md", task_id="task_upper", agent_id="agent_main_0001")

    lower_bundle = memory.build_bundle(
        agent_id="agent_coding_0001",
        agent_role=AgentRole.CODING,
        task_id="task_lower",
        target_files=["README.md"],
    )
    upper_bundle = memory.build_bundle(
        agent_id="agent_main_0001",
        agent_role=AgentRole.MAIN,
        task_id="task_lower",
    )

    assert lower_bundle.relevant_project_memory == []
    assert lower_bundle.relevant_file_cache == []
    assert upper_bundle.relevant_project_memory == [{"fact": "full project architecture"}]
    assert upper_bundle.relevant_file_cache == []


def test_tool_result_reused_when_args_same(tmp_path):
    (tmp_path / "README.md").write_text("needle\n", encoding="utf-8")
    memory = MultiAgentMemoryService(root=tmp_path)
    board = TaskBoard(tmp_path, memory_service=memory)
    task = board.create_task(title="Search", user_request="search needle")
    queue = QueueManager(tmp_path, taskboard=board, memory_service=memory)

    first = queue.enqueue(
        task_id=task.task_id,
        requested_by_agent_id="agent_tool_0001",
        job_type=QueueJobType.REPO_SEARCH,
        payload={"query": "needle"},
    )
    queue.run_next()
    memory.queue_fingerprints.clear()
    second = queue.enqueue(
        task_id=task.task_id,
        requested_by_agent_id="agent_tool_0001",
        job_type=QueueJobType.REPO_SEARCH,
        payload={"query": "needle"},
    )
    queue.run_next()

    assert first.result is not None
    assert second.result is not None
    assert second.result["cache_hit"] is True
    assert second.result["source"] == "memory"


def test_write_tool_not_reused_as_execution(tmp_path):
    (tmp_path / "README.md").write_text("old\n", encoding="utf-8")
    memory = MultiAgentMemoryService(root=tmp_path)
    memory.record_tool_execution(
        tool_name="apply_patch",
        args={"patch": "*** Begin Patch\n*** End Patch\n"},
        task_id="task_1",
        agent_id="agent_coding_0001",
        status="ok",
        result_summary="patched",
        result={"ok": True},
    )

    cached = memory.get_reusable_tool_result(
        tool_name="apply_patch",
        args={"patch": "*** Begin Patch\n*** End Patch\n"},
    )

    assert cached is None


def test_reusable_tool_memory_adds_cache_metadata(tmp_path):
    memory = MultiAgentMemoryService(root=tmp_path)
    memory.record_tool_execution(
        tool_name="repo_search",
        args={"query": "needle"},
        task_id="task_1",
        agent_id="agent_tool_0001",
        status="ok",
        result_summary="found",
        result={"stdout": "needle\n"},
    )

    cached = memory.get_reusable_tool_result(
        tool_name="repo_search",
        args={"query": "needle"},
    )

    assert cached is not None
    assert cached.result["cache_hit"] is True
    assert cached.result["source"] == "memory"


def test_verifier_uses_previous_memory(tmp_path):
    memory = MultiAgentMemoryService(root=tmp_path)
    memory.record_verification(
        task_id="task_1",
        verifier_agent_id="agent_verifier_0001",
        checked_files=["README.md"],
        tests_run=["python -m compileall src"],
        result="failed",
        findings=["planned_verification_not_executed"],
    )

    bundle = memory.build_bundle(
        agent_id="agent_verifier_0001",
        agent_role=AgentRole.VERIFIER,
        task_id="task_1",
        target_files=["README.md"],
    )

    assert bundle.verification_history[0]["tests_run"] == ["python -m compileall src"]


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


def test_only_main_agent_can_create_subagents(tmp_path):
    main = MainAgent(tmp_path)
    coding = main.agents[AgentRole.CODING]
    request = coding.create_subagent(AgentRole.CODING, ["repo_read"])
    assert request["type"] == "capacity_request"
    assert request["requested_by_agent_id"] == coding.agent_id

    task = main.taskboard.create_task(title="Capacity", user_request="need worker")
    node = main.agent_factory.create_subagent(
        main.registry.find_by_role(AgentRole.CODING).agent_id,
        AgentRole.TOOL_WORKER,
        task.task_id,
        ["tool_execution"],
        budget=500,
    )
    assert node.agent_id in main.taskboard.get_task(task.task_id).assigned_subagent_ids


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
    task_after = board.get_task(task.task_id)
    assert task_after.actual_tool_events
    assert all(event["agent_id"] != "main" for event in task_after.actual_tool_events)
    assert task_after.budget_records
    assert task_after.cost_by_queue_job_id[job.job_id] >= 1


def test_hierarchy_policy_rejects_main_agent_tool_execution(tmp_path):
    board = TaskBoard(tmp_path)
    task = board.create_task(title="Policy", user_request="read")
    policy = HierarchyPolicy(taskboard=board)
    with pytest.raises(HierarchyViolation):
        policy.assert_can_execute_tool(
            "agent_main_0001",
            "read_file",
            task_id=task.task_id,
            queue_job_id="queue_job_1",
        )
    assert board.get_task(task.task_id).hierarchy_violations


def test_tool_event_with_main_agent_fails_hierarchy_policy(tmp_path):
    board = TaskBoard(tmp_path)
    task = board.create_task(title="Policy", user_request="read")
    policy = HierarchyPolicy(taskboard=board)
    with pytest.raises(HierarchyViolation):
        policy.approve_tool_event(
            {
                "type": "tool.started",
                "agent_id": "main",
                "tool_name": "read_file",
                "task_id": task.task_id,
                "queue_job_id": "queue_job_1",
            }
        )


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
    task_after = board.get_task(task.task_id)
    assert task_after.actual_tool_events[0]["agent_role"] == "tool_worker"
    assert task_after.actual_tool_events[0]["queue_job_id"] == job.job_id


def test_queue_job_execution_records_subagent_identity(tmp_path):
    (tmp_path / "README.md").write_text("# Test\n", encoding="utf-8")
    board = TaskBoard(tmp_path)
    task = board.create_task(title="Batch read", user_request="read docs", owner_agent_id="agent_main_0002")
    worker_id = "subagent_tool_worker_0001"
    board.assign_subagent(task.task_id, worker_id)
    queue = QueueManager(tmp_path, taskboard=board, default_worker_agent_id=worker_id)
    job = queue.enqueue(
        task_id=task.task_id,
        requested_by_agent_id="agent_coding_0001",
        job_type=QueueJobType.REPO_BATCH_READ,
        payload={"files": ["README.md"]},
    )

    queue.run_until_idle()

    task_after = board.get_task(task.task_id)
    assert worker_id in task_after.assigned_subagent_ids
    assert task_after.actual_tool_events
    assert all(event["subagent_id"] == worker_id for event in task_after.actual_tool_events)
    assert all(event["queue_job_id"] == job.job_id for event in task_after.actual_tool_events)
    assert all(event["root_task_id"] == task.task_id for event in task_after.actual_tool_events)


def test_queue_job_default_approver_uses_current_task_owner(tmp_path):
    board = TaskBoard(tmp_path)
    first = board.create_task(title="First", user_request="first", owner_agent_id="agent_main_0001")
    second = board.create_task(title="Second", user_request="second", owner_agent_id="agent_main_0002")
    queue = QueueManager(tmp_path, taskboard=board)

    first_job = queue.enqueue(
        task_id=first.task_id,
        requested_by_agent_id="agent_coding_0001",
        job_type=QueueJobType.GIT_STATUS,
        payload={},
    )
    second_job = queue.enqueue(
        task_id=second.task_id,
        requested_by_agent_id="agent_coding_0001",
        job_type=QueueJobType.GIT_STATUS,
        payload={"scope": "second"},
    )

    assert first_job.approved_by_agent_id == "agent_main_0001"
    assert second_job.approved_by_agent_id == "agent_main_0002"
    assert second_job.root_task_id == second.task_id


def test_trace_writer_persists_execution_identity(tmp_path):
    writer = TraceWriter(tmp_path)
    event = writer.emit(
        "tool.finished",
        context=ExecutionContext(
            agent_id="subagent_tool_worker_0001",
            agent_role="tool_worker",
            parent_agent_id="agent_coding_0001",
            requested_by_agent_id="agent_coding_0001",
            queue_job_id="queue_job_1",
            task_id="task_1",
            root_task_id="task_1",
            delegation_path=["agent_coding_0001", "subagent_tool_worker_0001"],
        ),
        payload={"tool_name": "read_file"},
    )

    assert event.subagent_id == "subagent_tool_worker_0001"
    saved = (tmp_path / ".mana" / "traces" / f"{event.trace_id}.json").read_text(encoding="utf-8")
    assert '"subagent_id": "subagent_tool_worker_0001"' in saved
    assert '"queue_job_id": "queue_job_1"' in saved


def test_queue_manager_requires_budget_before_every_job(tmp_path):
    board = TaskBoard(tmp_path)
    task = board.create_task(title="Budget", user_request="search")
    queue = QueueManager(tmp_path, taskboard=board)
    job = queue.enqueue(
        task_id=task.task_id,
        requested_by_agent_id="agent_coding_0001",
        approved_by_agent_id="agent_main_0001",
        job_type=QueueJobType.REPO_SEARCH,
        payload={"query": "needle"},
    )
    assert job.budget_reserved > 0
    assert any(record.get("queue_job_id") == job.job_id and record.get("action") == "queue_job_reserved" for record in board.get_task(task.task_id).budget_records)


def test_queue_manager_rejects_main_agent_queue_job_creation(tmp_path):
    board = TaskBoard(tmp_path)
    task = board.create_task(title="Policy", user_request="search")
    queue = QueueManager(tmp_path, taskboard=board)
    with pytest.raises(HierarchyViolation):
        queue.enqueue(
            task_id=task.task_id,
            requested_by_agent_id="agent_main_0001",
            approved_by_agent_id="agent_main_0001",
            job_type=QueueJobType.REPO_SEARCH,
            payload={"query": "needle"},
        )


def test_batch_read_result_reused_when_args_same(tmp_path):
    (tmp_path / "README.md").write_text("# Test\n", encoding="utf-8")
    board = TaskBoard(tmp_path)
    task = board.create_task(title="Cached read", user_request="read docs twice")
    queue = QueueManager(tmp_path, taskboard=board)

    first = queue.enqueue(
        task_id=task.task_id,
        requested_by_agent_id="agent_coding_0001",
        approved_by_agent_id="agent_head_decision_0001",
        job_type=QueueJobType.REPO_BATCH_READ,
        payload={"files": ["README.md"]},
        purpose="Batch read selected repository files.",
    )
    second = queue.enqueue(
        task_id=task.task_id,
        requested_by_agent_id="agent_coding_0001",
        approved_by_agent_id="agent_head_decision_0001",
        job_type=QueueJobType.REPO_BATCH_READ,
        payload={"files": ["README.md"]},
        purpose="Batch read selected repository files again.",
    )

    queue.run_until_idle()

    assert first.result and first.result["cache_hit"] is False
    assert second.result and second.result["cache_hit"] is True
    assert second.result["files"][0]["path"] == "README.md"


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


def test_verifier_executes_real_verification_queue_job(tmp_path):
    main = MainAgent(tmp_path)
    task = main.taskboard.create_task(title="Verify", user_request="verify")
    worker = main.agent_factory.create_subagent(
        main.registry.find_by_role(AgentRole.CODING).agent_id,
        AgentRole.TOOL_WORKER,
        task.task_id,
        ["tool_execution"],
        budget=500,
    )
    main.queue_manager.default_worker_agent_id = worker.agent_id
    verifier = main.agents[AgentRole.VERIFIER]
    assert isinstance(verifier, VerifierAgent)
    result = verifier.execute_verification(task.task_id, ["python -m compileall ."])
    task_after = main.taskboard.get_task(task.task_id)
    assert result.passed is True
    assert task_after.verification_queue_job_ids
    assert task_after.actual_tool_events[-1]["queue_job_id"] in task_after.verification_queue_job_ids


def test_reviewer_rejects_planned_verification_without_queue_job(tmp_path):
    main = MainAgent(tmp_path)
    task = main.taskboard.create_task(title="Review", user_request="verify")
    verifier = main.agents[AgentRole.VERIFIER]
    reviewer = main.agents[AgentRole.REVIEWER]
    assert isinstance(verifier, VerifierAgent)
    verifier.verify_no_mutation(task.task_id, ["python -m compileall src"])
    assert reviewer.review_evidence(task.task_id, route_name="simple", requires_verification=True) is False
    assert any("verification lacks executed queue job evidence" in item for item in main.taskboard.get_task(task.task_id).blockers)


def test_main_agent_routes_chat_analyze_and_plan(tmp_path):
    assert MainAgent(tmp_path).run_user_request("hello", entrypoint="chat").route_name == "simple"
    assert MainAgent(tmp_path).run_user_request("scan repo", entrypoint="analyze").route_name == "analyze"
    assert MainAgent(tmp_path).run_user_request("update docs", entrypoint="plan").route_name == "planning"


def test_main_agent_uses_routing_llm_for_head_decision(tmp_path):
    class ModelRouter:
        def invoke(self, _messages):  # noqa: ANN001
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "intent": "web_research",
                        "confidence": 0.91,
                        "selected_tools": ["web_search"],
                        "tool_inputs": {"web_search": {"query": "openclaw description"}},
                        "repo_context_needed": False,
                        "web_search_needed": True,
                        "code_editing_needed": False,
                        "reasoning_summary": "Model selected public web research.",
                    }
                )
            )

    main = MainAgent(tmp_path, routing_llm=ModelRouter())
    result = main.run_user_request("search internet and give me description about openclaw", entrypoint="chat")
    assert result.route_name == "research"
    decisions = list(main.decision_room.decisions.values())
    assert decisions[-1].selected_route == "research"
    assert decisions[-1].rationale_summary == "Model selected public web research."


def test_main_agent_records_large_docs_subagents_and_deactivates_them(tmp_path):
    main = MainAgent(tmp_path)
    result = main.run_user_request("project architecture changed, update README.md, cannot use diff", entrypoint="chat")
    task = main.taskboard.get_task(result.task_id)
    assert result.route_name == "coding"
    assert result.task_size == "large"
    assert result.required_subagents == ["repo_inventory", "docs"]
    coding_helpers = [item for item in task.assigned_subagent_ids if main.registry.agents[item].role == AgentRole.CODING]
    assert len(coding_helpers) == 2
    assert all(main.registry.agents[subagent_id].state == AgentState.DONE for subagent_id in task.assigned_subagent_ids)
    assert any("deactivated" in item for item in task.evidence)


def test_main_agent_coding_route_has_queue_worker_verifier_and_review_evidence(tmp_path):
    (tmp_path / "README.md").write_text("# Test\n", encoding="utf-8")
    main = MainAgent(tmp_path)
    result = main.run_user_request("fix README.md heading", entrypoint="chat")
    task = main.taskboard.get_task(result.task_id)
    assert result.route_name == "coding"
    assert task.queue_job_ids
    assert task.verification_queue_job_ids
    assert task.actual_tool_events
    assert all(not str(event.get("agent_id", "")).startswith("agent_main_") for event in task.actual_tool_events)
    assert task.reviewed_by_agent_id == main.registry.find_by_role(AgentRole.REVIEWER).agent_id
    assert any(main.registry.agents[item].role == AgentRole.TOOL_WORKER for item in task.assigned_subagent_ids)


def test_git_commit_push_request_queues_git_inspection_and_does_not_repo_search(tmp_path):
    repo = _init_git_repo(tmp_path / "repo")
    (repo / "README.md").write_text("hello\nchanged\n", encoding="utf-8")
    main = MainAgent(repo)

    result = main.run_user_request("commit changes and push to main", entrypoint="chat")

    task = main.taskboard.get_task(result.task_id)
    commands = _git_job_commands(main, result.task_id)
    assert result.route_name == "high_risk_tool"
    assert "git_status" in task.required_capabilities
    assert "git_diff" in task.required_capabilities
    assert "git_commit" in task.required_capabilities
    assert "git_push" in task.required_capabilities
    assert ["status", "--short", "--branch"] in commands
    assert ["diff", "--stat"] in commands
    assert not any(job.job_type == QueueJobType.REPO_SEARCH for job in main.queue_manager.jobs_for_task(result.task_id))
    assert any(command and command[0] == "commit" for command in commands)
    assert any("Git push blocked: no remote exists" in item for item in task.blockers)
    assert task.status == TaskStatus.BLOCKED


def test_git_push_to_main_inspects_remote_and_blocks_when_behind(tmp_path):
    remote = tmp_path / "remote.git"
    assert subprocess.run(["git", "init", "--bare", str(remote)], text=True, capture_output=True, check=False).returncode == 0
    repo = _init_git_repo(tmp_path / "repo")
    assert _git(repo, "remote", "add", "origin", str(remote)).returncode == 0
    assert _git(repo, "push", "-u", "origin", "main").returncode == 0
    other = tmp_path / "other"
    assert subprocess.run(["git", "clone", "-b", "main", str(remote), str(other)], text=True, capture_output=True, check=False).returncode == 0
    assert _git(other, "config", "user.name", "Mana Agent Test").returncode == 0
    assert _git(other, "config", "user.email", "test@example.com").returncode == 0
    (other / "README.md").write_text("hello\nremote change\n", encoding="utf-8")
    assert _git(other, "commit", "-am", "test: remote change").returncode == 0
    assert _git(other, "push", "origin", "main").returncode == 0
    assert _git(repo, "fetch", "origin").returncode == 0
    main = MainAgent(repo)

    result = main.run_user_request("push to main", entrypoint="chat")

    task = main.taskboard.get_task(result.task_id)
    commands = _git_job_commands(main, result.task_id)
    assert ["branch", "--show-current"] in commands
    assert ["remote", "-v"] in commands
    assert any(command[:1] == ["rev-list"] for command in commands)
    assert not any(command and command[0] == "push" for command in commands)
    assert any("branch is behind remote" in item for item in task.blockers)
    assert task.status == TaskStatus.BLOCKED


def test_git_commit_inspects_diff_and_uses_diff_derived_message(tmp_path):
    repo = _init_git_repo(tmp_path / "repo")
    (repo / "README.md").write_text("hello\ncommit only\n", encoding="utf-8")
    main = MainAgent(repo)

    result = main.run_user_request("commit", entrypoint="chat")

    task = main.taskboard.get_task(result.task_id)
    commands = _git_job_commands(main, result.task_id)
    commit_commands = [command for command in commands if command and command[0] == "commit"]
    assert ["diff", "--stat"] in commands
    assert ["diff"] in commands
    assert commit_commands
    assert "README" in " ".join(commit_commands[0]) or "readme" in " ".join(commit_commands[0]).lower()
    assert "hardcoded" not in " ".join(commit_commands[0]).lower()
    assert _git(repo, "log", "-1", "--pretty=%s").stdout.strip().lower().startswith("docs: update readme")
    assert task.status == TaskStatus.DONE


def test_git_create_new_branch_inspects_status_before_branch_creation(tmp_path):
    repo = _init_git_repo(tmp_path / "repo")
    main = MainAgent(repo)

    result = main.run_user_request("create new branch feature/git-workflow", entrypoint="chat")

    task = main.taskboard.get_task(result.task_id)
    commands = _git_job_commands(main, result.task_id)
    assert ["status", "--short", "--branch"] in commands
    assert ["switch", "-c", "feature/git-workflow"] in commands
    assert _git(repo, "branch", "--show-current").stdout.strip() == "feature/git-workflow"
    assert task.status == TaskStatus.DONE


def test_model_levels_are_configurable_by_tier(monkeypatch):
    monkeypatch.setenv("MANA_MODEL_CODING", "MODEL_LEVEL_2_CUSTOM")
    assert model_level_for_role(AgentRole.CODING).model_level == "MODEL_LEVEL_2_CUSTOM"
    monkeypatch.delenv("MANA_MODEL_CODING")
    assert model_level_for_role(AgentRole.CODING).model_level == MODEL_LEVEL_2_CODING


def test_role_model_resolution_uses_distinct_level_envs(monkeypatch):
    for name in (
        "MANA_MODEL_MAIN",
        "MANA_MODEL_CODING",
        "MANA_MODEL_TOOL_WORKER",
        MODEL_LEVEL_3_HIGH_REASONING,
        MODEL_LEVEL_2_CODING,
        MODEL_LEVEL_1_FAST_TOOL,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(MODEL_LEVEL_3_HIGH_REASONING, "high-model")
    monkeypatch.setenv(MODEL_LEVEL_2_CODING, "coding-model")
    monkeypatch.setenv(MODEL_LEVEL_1_FAST_TOOL, "fast-model")

    assert resolve_model_for_role(AgentRole.MAIN, global_model="fallback").resolved_model == "high-model"
    assert resolve_model_for_role(AgentRole.CODING, global_model="fallback").resolved_model == "coding-model"
    assert resolve_model_for_role(AgentRole.TOOL_WORKER, global_model="fallback").resolved_model == "fast-model"


def test_role_specific_model_env_overrides_level_env(monkeypatch):
    monkeypatch.setenv(MODEL_LEVEL_1_FAST_TOOL, "fast-model")
    monkeypatch.setenv("MANA_MODEL_TOOL_WORKER", "tool-worker-override")

    assignment = resolve_model_for_role(AgentRole.TOOL_WORKER, global_model="fallback")

    assert assignment.model_level == MODEL_LEVEL_1_FAST_TOOL
    assert assignment.resolved_model == "tool-worker-override"


def test_missing_symbolic_model_level_falls_back_to_global(monkeypatch):
    monkeypatch.delenv("MANA_MODEL_CODING", raising=False)
    monkeypatch.delenv(MODEL_LEVEL_2_CODING, raising=False)

    assignment = resolve_model_for_role(AgentRole.CODING, global_model="global-model")

    assert assignment.model_level == MODEL_LEVEL_2_CODING
    assert assignment.resolved_model == "global-model"


def test_execution_context_preserves_model_metadata():
    ctx = ExecutionContext.from_mapping(
        {
            "agent_id": "subagent_tool_worker_0001",
            "agent_role": "tool_worker",
            "model_level": MODEL_LEVEL_1_FAST_TOOL,
            "resolved_model": "fast-model",
        }
    )

    assert ctx.agent_role == "tool_worker"
    assert ctx.subagent_id == "subagent_tool_worker_0001"
    assert ctx.as_dict()["model_level"] == MODEL_LEVEL_1_FAST_TOOL
    assert ctx.as_dict()["resolved_model"] == "fast-model"


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
        def __init__(self, root, **_kwargs):
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
        def __init__(self, root, **_kwargs):
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
