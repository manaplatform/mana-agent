"""Tests for WorkspaceDatabase and WorkspaceRepository."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mana_agent.gateway.lanes import LaneId, LanePriority, LaneTaskState, LockMode
from mana_agent.gateway.lane_models import LaneBudget, LaneExecution, LaneHandoff, LockLease
from mana_agent.execution_supervisor.models import CheckpointRecord, EscrowResult
from mana_agent.multi_agent.core.types import (
    AgentMessage,
    DecisionRecord,
    DecisionStatus,
    DiscussionStatus,
    DiscussionThread,
    HandoffRecord,
    MessageType,
    RiskLevel,
    TaskBoardItem,
    TaskStatus,
    VerificationResult,
    utc_now,
)
from mana_agent.persistence.workspace_db import WorkspaceDatabase
from mana_agent.persistence.workspace_repository import WorkspaceRepository


@pytest.fixture
def temp_workspace(tmp_path: Path):
    db_path = tmp_path / "state.db"
    db = WorkspaceDatabase("test-workspace", db_path=db_path)
    repo = WorkspaceRepository("test-workspace", db=db)
    yield db, repo, tmp_path
    db.close()


def test_workspace_database_initialization(temp_workspace):
    db, repo, _ = temp_workspace
    with db.connect() as conn:
        # Check pragmas
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.lower() == "wal"
        synchronous = conn.execute("PRAGMA synchronous;").fetchone()[0]
        assert synchronous == 1  # NORMAL
        foreign_keys = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert foreign_keys == 1

        # Check schema migration recorded
        row = conn.execute("SELECT version, migration_name FROM schema_migrations WHERE version = 1;").fetchone()
        assert row is not None
        assert row["migration_name"] == "initial_normalized_workspace_schema"


def test_task_crud_and_collections(temp_workspace):
    _, repo, _ = temp_workspace

    task = TaskBoardItem(
        task_id="task-100",
        parent_task_id=None,
        root_task_id="task-100",
        title="Refactor persistence",
        user_request="Refactor to SQLite",
        normalized_goal="Use versioned SQLite",
        status=TaskStatus.IN_PROGRESS,
        priority=50,
        risk_level=RiskLevel.HIGH,
        workspace_id="test-workspace",
        session_id="session-1",
        trigger_turn_id="turn-1",
        primary_repository_id="repo-main",
        repository_ids=["repo-main", "repo-sub"],
        depends_on=["task-099"],
        required_capabilities=["code_edit", "sqlite"],
        files_to_inspect=["src/main.py"],
        files_touched=["src/main.py", "src/db.py"],
        wiring_required=True,
        wiring_reason="Need SQLite wiring",
        wiring_outcome="completed",
        implementation_verified=True,
        integration_verified=True,
        runtime_reachability_verified=True,
        verification_commands=["pytest tests/"],
        verification_results=[
            VerificationResult(
                verification_id="v-1",
                task_id="task-100",
                verified_by_agent_id="agent-verifier",
                commands_run=["pytest tests/"],
                passed=True,
                summary="All tests passed",
            )
        ],
        handoff_records=[
            HandoffRecord(
                from_agent_id="agent-planner",
                to_agent_id="agent-coder",
                task_id="task-100",
                reason="Plan complete",
            )
        ],
        budget_records=[
            {"agent_id": "agent-coder", "budget_reserved_tokens": 5000, "budget_used_tokens": 1200}
        ],
        integration_evidence_records=[
            {
                "entrypoint": "main()",
                "evidence_path": ["src/main.py:42"],
                "path": ["src/main.py:42"],
                "summary": "Verified connection initialization",
                "source_references": ["src/db.py"],
                "observable_result": "Connected successfully",
                "verification_source": "unit_test",
                "reviewer": "agent-reviewer",
                "recorded_at": utc_now().isoformat(),
            }
        ],
    )

    repo.save_task(task)

    assert repo.task_exists("task-100") is True
    assert repo.count_tasks() == 1

    fetched = repo.get_task("task-100")
    assert fetched.task_id == "task-100"
    assert fetched.title == "Refactor persistence"
    assert fetched.status == TaskStatus.IN_PROGRESS
    assert fetched.risk_level == RiskLevel.HIGH
    assert fetched.repository_ids == ["repo-main", "repo-sub"]
    assert fetched.depends_on == ["task-099"]
    assert fetched.required_capabilities == ["code_edit", "sqlite"]
    assert fetched.wiring_required is True
    assert fetched.implementation_verified is True
    assert len(fetched.verification_results) == 1
    assert fetched.verification_results[0].passed is True
    assert len(fetched.handoff_records) == 1
    assert fetched.handoff_records[0].reason == "Plan complete"
    assert len(fetched.budget_records) == 1
    assert len(fetched.integration_evidence_records) == 1
    assert fetched.integration_evidence_records[0]["entrypoint"] == "main()"

    # Test list_tasks filter
    active_tasks = repo.list_tasks(status=TaskStatus.IN_PROGRESS)
    assert len(active_tasks) == 1
    assert active_tasks[0].task_id == "task-100"

    done_tasks = repo.list_tasks(status=TaskStatus.DONE)
    assert len(done_tasks) == 0

    session_tasks = repo.list_tasks(session_id="session-1")
    assert len(session_tasks) == 1

    # Delete task
    repo.delete_task("task-100")
    assert repo.task_exists("task-100") is False
    assert repo.get_task_or_none("task-100") is None
    assert repo.count_tasks() == 0


def test_task_events_ledger(temp_workspace):
    _, repo, _ = temp_workspace

    repo.append_task_event("task-1", "task.created", {"title": "Task 1"})
    repo.append_task_event("task-1", "task.assigned", {"agent": "agent-worker"})
    repo.append_task_event("task-2", "task.created", {"title": "Task 2"})

    all_events = repo.list_task_events()
    assert len(all_events) == 3

    task_1_events = repo.list_task_events(task_id="task-1")
    assert len(task_1_events) == 2
    assert task_1_events[0]["event_type"] == "task.created"
    assert task_1_events[1]["event_type"] == "task.assigned"


def test_discussions_and_decisions(temp_workspace):
    _, repo, _ = temp_workspace

    disc = DiscussionThread(
        discussion_id="disc-1",
        task_id="task-1",
        title="Architecture review",
        status=DiscussionStatus.OPEN,
        participant_agent_ids=["agent-1", "agent-2"],
        message_ids=["msg-1"],
        created_by_agent_id="agent-1",
    )
    repo.save_discussion(disc)

    dec = DecisionRecord(
        decision_id="dec-1",
        task_id="task-1",
        discussion_id="disc-1",
        made_by_agent_id="agent-1",
        decision_status=DecisionStatus.APPROVED,
        summary="Use SQLite WAL mode",
        rationale_summary="Improves concurrent access",
        selected_route="sqlite_wal",
        assigned_agent_ids=["agent-2"],
        required_verification=["pytest"],
    )
    repo.save_decision(dec)

    fetched_disc = repo.get_discussion("disc-1")
    assert fetched_disc is not None
    assert fetched_disc.title == "Architecture review"
    assert fetched_disc.participant_agent_ids == ["agent-1", "agent-2"]

    fetched_dec = repo.get_decision("dec-1")
    assert fetched_dec is not None
    assert fetched_dec.summary == "Use SQLite WAL mode"
    assert fetched_dec.decision_status == DecisionStatus.APPROVED

    discs = repo.list_discussions(task_id="task-1")
    assert len(discs) == 1
    decs = repo.list_decisions(task_id="task-1")
    assert len(decs) == 1


def test_messages(temp_workspace):
    _, repo, _ = temp_workspace

    msg1 = AgentMessage(
        message_id="msg-1",
        discussion_id="disc-1",
        from_agent_id="agent-1",
        to_agent_id="agent-2",
        task_id="task-1",
        message_type=MessageType.PROPOSAL,
        content="Proposed plan for migration",
        root_task_id="task-1",
    )
    msg2 = AgentMessage(
        message_id="msg-2",
        discussion_id="disc-1",
        from_agent_id="agent-2",
        to_agent_id="agent-1",
        task_id="task-1",
        message_type=MessageType.APPROVAL,
        content="Plan approved",
        root_task_id="task-1",
    )
    repo.save_message(msg1)
    repo.save_message(msg2)

    assert repo.get_message("msg-1") is not None
    assert len(repo.list_messages_for_task("task-1")) == 2
    assert len(repo.list_messages_for_thread("disc-1")) == 2
    assert len(repo.list_messages_for_inbox("agent-2")) == 1


def test_gateway_executions_and_locks(temp_workspace):
    _, repo, _ = temp_workspace

    budget = LaneBudget(
        reserved_input_tokens=1000,
        reserved_output_tokens=500,
        consumed_input_tokens=200,
        consumed_output_tokens=100,
        estimated_cost=0.05,
        actual_cost=0.01,
        estimated_cost_known=True,
        actual_cost_known=True,
    )
    handoff = LaneHandoff(
        source_lane=LaneId.RESEARCH,
        target_lane=LaneId.CODING,
        task_id="lane-task-1",
        reason="Plan finished",
    )
    execution = LaneExecution(
        task_id="lane-task-1",
        root_task_id="lane-task-1",
        parent_task_id=None,
        owning_lane=LaneId.CODING,
        state=LaneTaskState.RUNNING,
        normalized_intent="Code task",
        repository_id="repo-1",
        workspace_id="test-workspace",
        session_id="session-1",
        priority=LanePriority.HIGH,
        budget=budget,
        handoffs=[handoff],
    )
    repo.save_execution(execution)

    fetched_ex = repo.get_execution("lane-task-1")
    assert fetched_ex is not None
    assert fetched_ex.owning_lane == LaneId.CODING
    assert fetched_ex.state == LaneTaskState.RUNNING
    assert fetched_ex.budget.reserved_input_tokens == 1000
    assert len(fetched_ex.handoffs) == 1
    assert fetched_ex.handoffs[0].source_lane == LaneId.RESEARCH

    lock = LockLease(
        lease_id="lock-1",
        task_id="lane-task-1",
        mode=LockMode.WORKSPACE_WRITE,
        workspace_id="test-workspace",
        repository_id="repo-1",
        paths=["src/"],
        owner_pid=1234,
        acquired_at=utc_now().isoformat(),
        expires_at=utc_now().isoformat(),
    )
    repo.save_lock(lock)

    locks = repo.list_locks("test-workspace")
    assert len(locks) == 1
    assert locks[0].lease_id == "lock-1"
    assert locks[0].mode == LockMode.WORKSPACE_WRITE

    repo.delete_lock("lock-1")
    assert len(repo.list_locks("test-workspace")) == 0


def test_gateway_turns_idempotency(temp_workspace):
    _, repo, _ = temp_workspace

    turn, existing = repo.create_or_get_turn(
        conversation_id="conv-1",
        user_message_id="msg-user-1",
        turn_id="turn-1",
        text="Hello agent",
    )
    assert existing is False
    assert turn.turn_id == "turn-1"

    # Same conversation and message ID with exact same text -> returns existing
    turn2, existing2 = repo.create_or_get_turn(
        conversation_id="conv-1",
        user_message_id="msg-user-1",
        turn_id="turn-different",
        text="Hello agent",
    )
    assert existing2 is True
    assert turn2.turn_id == "turn-1"

    # Same user_message_id with different text -> raises ValueError
    with pytest.raises(ValueError, match="user_message_id already belongs to a different message"):
        repo.create_or_get_turn(
            conversation_id="conv-1",
            user_message_id="msg-user-1",
            turn_id="turn-3",
            text="Different text",
        )
