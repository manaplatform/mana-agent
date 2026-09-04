"""Concurrency, crash recovery, and large-workspace performance tests for SQLite persistence."""

import concurrent.futures
import time
from pathlib import Path

import pytest

from mana_agent.gateway.lanes import LaneId, LanePriority, LaneTaskState
from mana_agent.gateway.lane_models import LaneBudget, LaneExecution
from mana_agent.multi_agent.core.types import RiskLevel, TaskBoardItem, TaskStatus, utc_now
from mana_agent.persistence.workspace_db import WorkspaceDatabase
from mana_agent.persistence.workspace_repository import WorkspaceRepository


@pytest.fixture
def perf_workspace(tmp_path: Path):
    db_path = tmp_path / "perf_state.db"
    db = WorkspaceDatabase("perf-workspace", db_path=db_path)
    repo = WorkspaceRepository("perf-workspace", db=db)
    yield db, repo
    db.close()


def test_concurrent_task_writes(perf_workspace):
    _, repo = perf_workspace

    def write_task(idx: int):
        task = TaskBoardItem(
            task_id=f"concurrent-task-{idx}",
            parent_task_id=None,
            root_task_id=f"concurrent-task-{idx}",
            title=f"Concurrent Task {idx}",
            user_request="Request",
            normalized_goal="Goal",
            status=TaskStatus.NEW,
            priority=100,
            risk_level=RiskLevel.LOW,
            workspace_id="perf-workspace",
        )
        repo.save_task(task)
        repo.append_task_event(task.task_id, "task.created", {"idx": idx})
        return idx

    num_threads = 20
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(write_task, i) for i in range(num_threads)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == num_threads
    assert repo.count_tasks() == num_threads

    events = repo.list_task_events(limit=100)
    assert len(events) == num_threads


def test_large_workspace_performance(perf_workspace):
    _, repo = perf_workspace

    num_tasks = 500
    # Batch creation
    start_time = time.perf_counter()
    for i in range(num_tasks):
        task = TaskBoardItem(
            task_id=f"perf-task-{i}",
            parent_task_id=None,
            root_task_id=f"perf-task-{i}",
            title=f"Performance Task {i}",
            user_request=f"User request {i}",
            normalized_goal=f"Normalized goal {i}",
            status=TaskStatus.IN_PROGRESS if i % 2 == 0 else TaskStatus.DONE,
            priority=i,
            risk_level=RiskLevel.LOW,
            workspace_id="perf-workspace",
            session_id=f"session-{i % 5}",
            depends_on=[f"perf-task-{max(0, i-1)}"],
            files_touched=[f"file_{i}.py"],
        )
        repo.save_task(task)
    write_duration = time.perf_counter() - start_time

    # Performance expectation: 500 tasks saved efficiently (with CI tolerance)
    assert write_duration < 45.0
    assert repo.count_tasks() == num_tasks

    # Targeted query by task_id: should be fast
    q_start = time.perf_counter()
    single_task = repo.get_task("perf-task-250")
    q_duration = time.perf_counter() - q_start
    assert single_task.title == "Performance Task 250"
    assert q_duration < 1.0

    # Filtered query by status
    in_progress = repo.list_tasks(status=TaskStatus.IN_PROGRESS)
    assert len(in_progress) == 250

    # Filtered query by session_id
    session_0_tasks = repo.list_tasks(session_id="session-0")
    assert len(session_0_tasks) == 100


def test_concurrent_execution_updates(perf_workspace):
    _, repo = perf_workspace

    # Initialize execution
    execution = LaneExecution(
        task_id="shared-lane-task",
        root_task_id="shared-lane-task",
        parent_task_id=None,
        owning_lane=LaneId.CODING,
        state=LaneTaskState.RUNNING,
        normalized_intent="Shared execution intent",
        repository_id="repo-1",
        workspace_id="perf-workspace",
        session_id="session-1",
        priority=LanePriority.HIGH,
        budget=LaneBudget(reserved_input_tokens=10000),
    )
    repo.save_execution(execution)

    def update_consumed_tokens(tokens: int):
        ex = repo.get_execution("shared-lane-task")
        if ex:
            ex.budget.consumed_input_tokens += tokens
            repo.save_execution(ex)
        return tokens

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(update_consumed_tokens, 10) for _ in range(15)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 15
    final_ex = repo.get_execution("shared-lane-task")
    assert final_ex is not None
    assert final_ex.budget.consumed_input_tokens > 0
