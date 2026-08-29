"""Regression tests for Coding Agent lifecycle events and scheduling observability."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from mana_agent.coding.event_visibility import (
    EventSemanticKind,
    EventVisibility,
    classify_coding_event,
    semantic_kind_for_event_type,
)
from mana_agent.coding.models import (
    AgentEvent,
    CodingTask,
    CodingTaskResult,
    WorkspaceContext,
    compute_duration_breakdown,
)
from mana_agent.execution_supervisor import (
    ExecutionSupervisor,
    ExecutionSupervisorConfig,
)
from mana_agent.gateway.lane_coordinator import LaneCoordinator
from mana_agent.gateway.lanes import LaneId, LanePriority, LaneTaskState
from mana_agent.integrations.codex.event_adapter import adapt_codex_event
from mana_agent.integrations.codex.result_parser import parse_codex_result
from mana_agent.multi_agent.core.types import TaskBoardItem, TaskStatus
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard


def test_compute_duration_breakdown_separates_queue_and_provider_latency() -> None:
    """Verify that queue delay and provider execution time are separated accurately."""
    t0 = datetime(2026, 8, 29, 10, 0, 0, tzinfo=timezone.utc)
    # Task sits in queue for 15 minutes and 57 seconds
    t1 = t0 + timedelta(minutes=15, seconds=57)
    # Worker claimed 500ms after scheduled
    t2 = t1 + timedelta(milliseconds=500)
    # Provider starts 500ms later
    t3 = t2 + timedelta(milliseconds=500)
    # Provider fails or runs for 2.5 seconds
    t4 = t3 + timedelta(seconds=2, milliseconds=500)
    # Task finalization takes 500ms
    t5 = t4 + timedelta(milliseconds=500)

    breakdown = compute_duration_breakdown(
        task_created_at=t0,
        scheduled_at=t1,
        worker_claimed_at=t2,
        provider_started_at=t3,
        provider_completed_at=t4,
        task_completed_at=t5,
    )

    assert breakdown["queue_delay_ms"] == 957000
    assert breakdown["worker_acquisition_delay_ms"] == 500
    assert breakdown["provider_startup_delay_ms"] == 500
    assert breakdown["provider_execution_time_ms"] == 2500
    assert breakdown["finalization_time_ms"] == 500
    assert breakdown["total_task_duration_ms"] == 961000

    # Ensure provider execution time is 2.5 seconds, NOT 16 minutes
    assert breakdown["provider_execution_time_ms"] < 3000
    assert breakdown["queue_delay_ms"] > 900000


def test_user_message_event_normalization_is_lifecycle_never_assistant_generation() -> None:
    """Verify userMessage is classified as LIFECYCLE, not ASSISTANT_GENERATION or assistant.completed."""
    # Test semantic_kind_for_event_type directly
    assert semantic_kind_for_event_type("user.message") == EventSemanticKind.LIFECYCLE
    assert semantic_kind_for_event_type("turn.input") == EventSemanticKind.LIFECYCLE
    assert semantic_kind_for_event_type("tool.call.started", tool_name="userMessage") == EventSemanticKind.LIFECYCLE
    assert semantic_kind_for_event_type("tool.call.completed", tool_name="usermessage") == EventSemanticKind.LIFECYCLE

    # Test adapt_codex_event for userMessage items
    user_item_started = {
        "method": "item/started",
        "params": {
            "threadId": "thread-123",
            "item": {"type": "userMessage", "text": "Fix the bug"},
        },
    }
    event_started = adapt_codex_event("task-1", user_item_started)
    assert event_started.event_type == "user.message.started"
    assert event_started.semantic_kind == EventSemanticKind.LIFECYCLE.value
    assert event_started.event_type != "assistant.started"

    user_item_completed = {
        "method": "item/completed",
        "params": {
            "threadId": "thread-123",
            "item": {"type": "userMessage", "text": "Fix the bug"},
        },
    }
    event_completed = adapt_codex_event("task-1", user_item_completed)
    assert event_completed.event_type == "user.message.completed"
    assert event_completed.semantic_kind == EventSemanticKind.LIFECYCLE.value
    assert event_completed.event_type != "assistant.completed"


def test_failed_provider_turns_cannot_emit_successful_assistant_generation() -> None:
    """Verify failed provider turns emit error events, not successful assistant completion."""
    failed_item = {
        "method": "item/completed",
        "params": {
            "threadId": "thread-123",
            "item": {
                "type": "agentMessage",
                "status": "failed",
                "message": "Upstream 400 Bad Request",
            },
        },
    }
    event = adapt_codex_event("task-1", failed_item)
    assert event.status == "failed"
    assert event.event_type == "error"
    assert event.semantic_kind == EventSemanticKind.ERROR.value
    assert event.event_type != "assistant.completed"

    # Turn failed notification
    turn_failed = {
        "method": "turn/failed",
        "params": {
            "threadId": "thread-123",
            "message": "Connection to upstream provider timed out after 2.5s",
            "error_code": "CODING_PROVIDER_TIMEOUT",
        },
    }
    turn_event = adapt_codex_event("task-1", turn_failed)
    assert turn_event.status == "failed"
    assert turn_event.event_type == "error"
    assert turn_event.semantic_kind == EventSemanticKind.ERROR.value


def test_system_error_notification_and_error_preservation(tmp_path: Path) -> None:
    """Verify systemError notifications are parsed and original errors preserved."""
    repo = tmp_path / "repo"
    repo.mkdir()
    ws = WorkspaceContext(
        repository_path=repo,
        worktree_path=repo,
        sandbox="workspaceWrite",
        allow_in_place_write=True,
    )
    task = CodingTask(task_id="task-sys-1", goal="Test systemError handling")

    notifications = [
        {"method": "turn/started", "params": {"threadId": "t1"}},
        {
            "method": "systemError",
            "params": {
                "threadId": "t1",
                "message": "Upstream provider returned 401 Unauthorized",
                "http_status": 401,
                "error_code": "CODING_PROVIDER_AUTH_ERROR",
            },
        },
    ]

    t_start = datetime(2026, 8, 29, 10, 0, 0, tzinfo=timezone.utc)
    t_end = t_start + timedelta(seconds=2, milliseconds=500)

    result = parse_codex_result(
        task=task,
        workspace=ws,
        worker_id="w1",
        thread_id="t1",
        turn_id="turn-1",
        notifications=notifications,
        changed_files=[],
        task_created_at=t_start,
        scheduled_at=t_start,
        worker_claimed_at=t_start,
        provider_started_at=t_start,
        provider_completed_at=t_end,
        task_completed_at=t_end,
    )

    assert result.status == "failed"
    assert any("401" in err or "Unauthorized" in err for err in result.errors)
    assert result.codex_metadata["http_status"] == 401
    assert result.codex_metadata["original_error"] != ""
    assert result.duration_breakdown["provider_execution_time_ms"] == 2500


def test_lane_coordinator_scheduling_diagnostics_exposes_timestamps_and_durations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify lane coordinator tracks timestamps and computes duration breakdown in scheduling_diagnostics."""
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "home"))
    root = tmp_path / "repo"
    root.mkdir()
    coordinator = LaneCoordinator(root)

    res = coordinator.reserve(
        normalized_intent="observability test",
        lane_id=LaneId.CODING,
        session_id="session-1",
        workspace_id=coordinator.taskboard.store.workspace_id,
        repository_id=coordinator.taskboard.store.repository_id,
    )
    task_id = res.execution.task_id

    # Initially in queued state
    assert res.execution.task_created_at != ""
    assert res.execution.scheduled_at != ""
    assert res.execution.state == LaneTaskState.QUEUED

    # Start the task
    exec_running = coordinator.start(res)
    assert exec_running.worker_claimed_at != ""
    assert exec_running.state == LaneTaskState.RUNNING

    # Finish the task with provider timestamps
    p_start = (datetime.now(timezone.utc) - timedelta(seconds=3)).isoformat()
    p_end = datetime.now(timezone.utc).isoformat()
    finished = coordinator.finish(
        task_id,
        state=LaneTaskState.COMPLETED,
        verification_state={
            "provider_started_at": p_start,
            "provider_completed_at": p_end,
            "lane_state": "completed",
            "verification_evidence_present": True,
        },
    )

    assert finished.provider_started_at == p_start
    assert finished.provider_completed_at == p_end
    assert finished.task_completed_at != ""
    assert isinstance(finished.duration_breakdown, dict)

    # Check scheduling diagnostics
    diag = coordinator.scheduling_diagnostics(task_id)
    assert "timestamps" in diag
    assert diag["timestamps"]["task_created_at"] is not None
    assert diag["timestamps"]["scheduled_at"] is not None
    assert diag["timestamps"]["worker_claimed_at"] is not None
    assert diag["timestamps"]["provider_started_at"] is not None
    assert diag["timestamps"]["provider_completed_at"] is not None
    assert diag["timestamps"]["task_completed_at"] is not None

    assert "durations_ms" in diag
    assert "queue_delay_ms" in diag["durations_ms"]
    assert "worker_acquisition_delay_ms" in diag["durations_ms"]
    assert "provider_execution_time_ms" in diag["durations_ms"]
    assert "total_task_duration_ms" in diag["durations_ms"]
