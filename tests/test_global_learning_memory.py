"""Comprehensive tests for the Global Self-Learning Memory System."""

from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import pytest

from mana_agent.memory.learning import (
    GlobalMemory,
    LearningCategory,
    LearningMemory,
    LearningScope,
    LearningStatus,
    MemoryLearningEngine,
    get_global_memory,
    reset_global_memory,
)
from mana_agent.memory.learning.store import LearningMemoryStore
from mana_agent.coding.models import CodingTask, WorkspaceContext
from mana_agent.coding.internal_engine import InternalCodingEngine
from mana_agent.coding.external_engine import ExternalCodingEngine
from mana_agent.execution.autonomous_controller import AutonomousExecutionController
from mana_agent.execution.continuation_decision import ContinuationAction, ContinuationDecision
from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.models import ExecutionState
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
from mana_agent.multi_agent.taskboard.taskboard import TaskBoard


@pytest.fixture(autouse=True)
def clean_memory_singleton():
    reset_global_memory()
    yield
    reset_global_memory()


@pytest.fixture
def memory_db_path(tmp_path: Path) -> Path:
    return tmp_path / "memory" / "test_learning.sqlite3"


@pytest.fixture
def global_memory(memory_db_path: Path) -> GlobalMemory:
    store = LearningMemoryStore(db_path=memory_db_path)
    return GlobalMemory(store=store, default_project_id="test_repo")


# 1. A failed coding command can produce a learning record
def test_1_failed_command_produces_learning_candidate(global_memory: GlobalMemory):
    mem = global_memory.record_failure(
        attempted_action="npm run bulid",
        error="npm ERR! Missing script: 'bulid'. Did you mean 'build'?",
        category=LearningCategory.COMMAND,
        context="Building web frontend",
        execution_id="exec_1",
    )
    assert mem is not None
    assert mem.category == LearningCategory.COMMAND
    assert mem.attempted_action == "npm run bulid"
    assert mem.correction == "npm run build"
    assert "build" in (mem.prevention_rule or "")
    assert mem.status in {LearningStatus.CANDIDATE, LearningStatus.VERIFIED}
    assert mem.failure_count == 1


# 2. The corrected command is retrieved in a later execution
def test_2_corrected_command_verified_and_retrieved(global_memory: GlobalMemory):
    mem = global_memory.record_failure(
        attempted_action="npm run bulid",
        error="Missing script: 'bulid'",
        category=LearningCategory.COMMAND,
        execution_id="exec_1",
    )
    assert mem is not None

    # Verify the learning with positive evidence
    verified = global_memory.verify_learning(mem.id, success=True, evidence="Exit code 0; bundle generated.")
    assert verified is not None
    assert verified.status == LearningStatus.VERIFIED
    assert verified.success_count == 1
    assert verified.confidence > 0.5

    # Retrieve before next execution
    relevant = global_memory.retrieve_relevant("npm run bulid")
    assert len(relevant) >= 1
    assert relevant[0].id == mem.id
    assert relevant[0].correction == "npm run build"

    formatted = global_memory.format_retrieved_rules(relevant)
    assert "npm run build" in formatted


# 3. An invalid API call produces reusable knowledge after its correction is verified
def test_3_invalid_api_call_produces_reusable_knowledge(global_memory: GlobalMemory):
    # Failed API call
    mem = global_memory.record_failure(
        attempted_action="POST https://api.example.com/v1/users",
        error="HTTP 400: Missing required header 'X-Organization-Id'",
        category=LearningCategory.API,
        context="Creating organization member",
        execution_id="api_exec_1",
        metadata={"status_code": 400},
    )
    assert mem is not None
    assert mem.category == LearningCategory.API

    # Record correction
    corrected = global_memory.record_correction(
        mem.id,
        correction="POST https://api.example.com/v1/users with header X-Organization-Id",
        prevention_rule="Always include header 'X-Organization-Id' when calling /v1/users.",
        evidence="HTTP 201 Created",
    )
    assert corrected is not None
    assert corrected.prevention_rule is not None
    assert "X-Organization-Id" in corrected.prevention_rule

    global_memory.verify_learning(mem.id, success=True, evidence="201 Created")

    # In a later execution, retrieve knowledge for this endpoint
    retrieved = global_memory.retrieve_relevant("POST https://api.example.com/v1/users", category="API")
    assert len(retrieved) >= 1
    assert "X-Organization-Id" in retrieved[0].prevention_rule


# 4. Different engines can access the same authorized memory
def test_4_different_engines_share_same_authorized_memory(global_memory: GlobalMemory, tmp_path: Path):
    # Record verified rule in GlobalMemory
    mem = global_memory.record_failure(
        attempted_action="python -m pyetst",
        error="No module named pyetst",
        category=LearningCategory.COMMAND,
        execution_id="shared_1",
    )
    assert mem is not None
    global_memory.verify_learning(mem.id, success=True, evidence="pytest passed")

    # Internal engine access
    internal_engine = InternalCodingEngine(
        model="fake",
        global_memory=global_memory,
    )
    assert internal_engine.global_memory is global_memory
    internal_rules = internal_engine.global_memory.retrieve_relevant("python -m pyetst")
    assert len(internal_rules) >= 1
    assert internal_rules[0].correction == "python -m pytest"

    # External engine access (Codex - Rule 32 compliant)
    class DummyCodexAdapter:
        async def start(self): pass
        async def execute(self, task, ws):
            return SimpleNamespace(status="success", errors=[], changed_files=[], duration_breakdown={})

    external_engine = ExternalCodingEngine(
        provider_adapter=DummyCodexAdapter(),
        global_memory=global_memory,
    )
    assert external_engine.global_memory is global_memory
    external_rules = external_engine.global_memory.retrieve_relevant("python -m pyetst")
    assert len(external_rules) >= 1
    assert external_rules[0].correction == "python -m pytest"


# 5. Learning persists after process and application restarts
def test_5_learning_persists_across_restarts(memory_db_path: Path):
    # Session 1: Create and verify memory
    gm1 = GlobalMemory(store=LearningMemoryStore(db_path=memory_db_path), default_project_id="repo1")
    mem = gm1.record_failure(
        attempted_action="git comit -m 'fix'",
        error="git: 'comit' is not a git command. See 'git --help'.",
        category=LearningCategory.COMMAND,
    )
    assert mem is not None
    gm1.verify_learning(mem.id, success=True, evidence="commit created")
    mem_id = mem.id

    # Simulate restart: reset singleton and re-open from same SQLite database
    reset_global_memory()
    gm2 = GlobalMemory(store=LearningMemoryStore(db_path=memory_db_path), default_project_id="repo1")

    persisted = gm2.store.get(mem_id)
    assert persisted is not None
    assert persisted.attempted_action == "git comit -m 'fix'"
    assert persisted.correction == "git commit -m 'fix'"
    assert persisted.status == LearningStatus.VERIFIED
    assert persisted.success_count == 1

    # Check history is also persisted
    history = gm2.get_learning_history(mem_id)
    assert len(history) >= 2
    actions = [h["action"] for h in history]
    assert "created" in actions
    assert "verified" in actions


# 6. Duplicate failures do not produce unlimited duplicate records
def test_6_duplicate_failures_merge_and_bound(global_memory: GlobalMemory):
    for i in range(5):
        global_memory.record_failure(
            attempted_action="cargo biuld --release",
            error="error: no such subcommand: `biuld`",
            category=LearningCategory.COMMAND,
            execution_id=f"run_{i}",
        )

    all_matches = global_memory.store.find_by_action_exact("cargo biuld --release")
    assert len(all_matches) == 1
    record = all_matches[0]
    assert record.failure_count == 5
    assert record.correction == "cargo build --release"


# 7. Contradictory memories are identified and resolved safely
def test_7_contradictory_memories_handled_safely(global_memory: GlobalMemory):
    mem1 = global_memory.record_failure(
        attempted_action="use_sync_client",
        error="Blocked async event loop",
        category=LearningCategory.CODING,
    )
    assert mem1 is not None
    global_memory.record_correction(
        mem1.id,
        correction="use_async_client",
        prevention_rule="Always use async client in async contexts.",
    )
    global_memory.verify_learning(mem1.id, success=True)

    # Later discovery: a newer or different convention supersedes
    mem2 = global_memory.record_failure(
        attempted_action="use_legacy_async",
        error="Deprecated in v2",
        category=LearningCategory.CODING,
    )
    assert mem2 is not None
    global_memory.record_correction(
        mem2.id,
        correction="use_httpx_async",
        prevention_rule="Use httpx.AsyncClient for HTTP calls.",
    )

    # Supersede mem1 with mem2
    old_m, new_m = global_memory.supersede(mem1.id, mem2.id, reason="Migrated to httpx standard")
    assert old_m is not None and old_m.status == LearningStatus.SUPERSEDED
    assert old_m.superseded_by == mem2.id
    assert new_m is not None and new_m.replaces_memory_id == mem1.id

    # Active retrieval does not return superseded memory
    active_rules = global_memory.retrieve_relevant("use_sync_client")
    assert mem1.id not in [m.id for m in active_rules]


# 8. Project-specific rules do not leak into unrelated projects
def test_8_project_specific_rules_do_not_leak(global_memory: GlobalMemory):
    # Record project-specific rule in repo_alpha
    alpha_mem = global_memory.record_failure(
        attempted_action="pytest tests/unit",
        error="Target directory tests/unit not found; use tests/core instead",
        category=LearningCategory.COMMAND,
        scope=LearningScope.PROJECT,
        project_id="repo_alpha",
    )
    assert alpha_mem is not None
    global_memory.record_correction(
        alpha_mem.id,
        correction="pytest tests/core",
        prevention_rule="In repo_alpha, unit tests are in tests/core.",
    )
    global_memory.verify_learning(alpha_mem.id, success=True)

    # Search from repo_beta: must NOT retrieve repo_alpha's rule
    beta_results = global_memory.retrieve_relevant("pytest tests/unit", project_id="repo_beta")
    assert len(beta_results) == 0

    # Search from repo_alpha: must retrieve it
    alpha_results = global_memory.retrieve_relevant("pytest tests/unit", project_id="repo_alpha")
    assert len(alpha_results) == 1
    assert alpha_results[0].id == alpha_mem.id


# 9. Sensitive information is not persisted
def test_9_sensitive_information_is_redacted(global_memory: GlobalMemory):
    secret_token = "Bearer gh_secret_super_private_token_12345678"
    secret_key = "api_key=sk-live-abcdef1234567890abcdef"
    error_msg = f"Failed calling https://user:p@ssw0rd123@api.secret.com with {secret_token} and {secret_key}"

    mem = global_memory.record_failure(
        attempted_action=f"curl -H '{secret_token}' https://api.secret.com?token=secret_987654321",
        error=error_msg,
        category=LearningCategory.API,
    )
    assert mem is not None
    # Verify no raw secrets in persisted record
    assert "gh_secret_super_private_token" not in mem.attempted_action
    assert "sk-live-abcdef" not in mem.error
    assert "p@ssw0rd123" not in mem.error
    assert "[REDACTED]" in mem.attempted_action
    assert "[REDACTED]" in mem.error


# 10. Unverified corrections are never presented as established facts
def test_10_unverified_corrections_remain_hypotheses(global_memory: GlobalMemory):
    mem = global_memory.record_failure(
        attempted_action="configure_widget(mode='fast')",
        error="WidgetCrash: unknown configuration mode",
        category=LearningCategory.REASONING,
    )
    assert mem is not None
    assert mem.status == LearningStatus.CANDIDATE
    assert mem.is_hypothesis is True
    assert mem.confidence <= 0.5


# 11. Memory retrieval does not cause unbounded latency or context growth
def test_11_memory_retrieval_bounded(global_memory: GlobalMemory):
    # Insert 25 memories
    for i in range(25):
        global_memory.record_failure(
            attempted_action=f"action_number_{i}",
            error=f"Error code {i} happened during test",
            category=LearningCategory.GENERAL,
        )

    # Retrieval should be strictly bounded by limit
    results = global_memory.retrieve_relevant("action_number", limit=5)
    assert len(results) == 5

    formatted = global_memory.format_retrieved_rules(results)
    # Ensure compact representation
    assert len(formatted.splitlines()) <= 10


# 12. Parallel actions cannot corrupt memory state
def test_12_concurrent_writes_thread_safety(global_memory: GlobalMemory):
    errors: list[Exception] = []

    def worker(worker_id: int):
        try:
            for j in range(10):
                mem = global_memory.record_failure(
                    attempted_action=f"worker_cmd_{worker_id}_{j}",
                    error=f"Threaded error {worker_id}-{j}",
                    category=LearningCategory.COMMAND,
                )
                if mem:
                    global_memory.verify_learning(mem.id, success=True, evidence="thread ok")
        except Exception as exc:
            errors.append(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(8)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0


# 13. Memory unavailability has a safe fallback
def test_13_memory_unavailability_safe_fallback(tmp_path: Path):
    # Point store to a directory that cannot be written or closed store
    bad_store = LearningMemoryStore(db_path=tmp_path / "valid.sqlite3")
    gm = GlobalMemory(store=bad_store)

    # Simulate store closing or database lock failure
    bad_store.db_path = Path("/dev/null/impossible_dir/file.sqlite3")

    # None of these operations should crash or raise uncaught exceptions
    res1 = gm.record_failure(attempted_action="test", error="err")
    assert res1 is None

    res2 = gm.retrieve_relevant("test")
    assert res2 == []

    res3 = gm.verify_learning("nonexistent_id", success=True)
    assert res3 is None


# 14. Existing Loop, Supervisor, TaskBoard, and coding-engine behavior remains compatible
def test_14_autonomous_loop_supervisor_taskboard_compatibility(tmp_path: Path):
    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(root=tmp_path / "supervisor", startup_recovery=False))
    taskboard = TaskBoard(root=tmp_path / "workspace")
    gm = GlobalMemory(store=LearningMemoryStore(db_path=tmp_path / "global.sqlite3"), default_project_id="test_repo")

    # Create controller with GlobalMemory
    class FakeEvaluator:
        def evaluate(self, **kwargs):
            return ContinuationDecision(
                task_id=kwargs["task_id"],
                action=ContinuationAction.FINALIZE,
                goal_satisfied=True,
                progress_detected=True,
                evidence_summary="Goal verified successfully",
                safe_to_continue=True,
                reason="goal satisfied with verification",
            )

    controller = AutonomousExecutionController(
        supervisor=supervisor,
        taskboard=taskboard,
        continuation_evaluator=FakeEvaluator(),
        global_memory=gm,
    )

    # Run single-step goal that succeeds
    def runner(step_index, ctx):
        assert "learning_memories" in ctx or "taskboard" in ctx
        return {
            "output": "All components completed seamlessly.",
            "verification_passed": True,
        }

    result = controller.execute_goal(
        task_id="auto_compat_task",
        user_goal="Validate self-learning memory architectural quad compatibility",
        step_runner=runner,
    )

    assert result.status == "completed"
    assert result.goal_satisfied is True
    assert result.verification_passed is True

    # TaskBoard recorded task and supervisor recorded state
    tb_task = taskboard.get_task("auto_compat_task")
    assert tb_task.crossed_off is True
    sup_task = supervisor.get_task("auto_compat_task")
    assert sup_task.state == ExecutionState.COMPLETED


# 15. AskAgent records tool mistakes and injects learned prevention rules
def test_15_ask_agent_records_tool_mistake_and_injects_prevention_rule(tmp_path: Path):
    from langchain_core.messages import SystemMessage
    from mana_agent.context_cost import ContextCostGovernor
    from mana_agent.multi_agent.runtime.ask_agent import AskAgent
    from mana_agent.search.config import SearchConfig

    gm = GlobalMemory(store=LearningMemoryStore(db_path=tmp_path / "ask_learning.sqlite3"), default_project_id=str(tmp_path))

    class FakeAIMessage:
        def __init__(self, content: str, tool_calls: list | None = None) -> None:
            self.content = content
            self.tool_calls = tool_calls or []

    captured_messages: list[list[Any]] = []

    class FakeLLM:
        def __init__(self, responses: list[FakeAIMessage]) -> None:
            self._responses = list(responses)

        def bind_tools(self, tools: list[Any], tool_choice: str | None = None) -> "FakeLLM":
            return self

        def invoke(self, messages: list[Any], config: Any = None, **kwargs: Any) -> FakeAIMessage:
            captured_messages.append(list(messages))
            return self._responses.pop(0)

    governor = ContextCostGovernor(
        session_id="test-ask-agent-learning",
        settings=SimpleNamespace(
            mana_context_governor_enabled=False,
            mana_context_cost_log_enabled=False,
        ),
    )

    class FakeSearchService:
        pass

    agent = AskAgent(
        api_key="test",
        model="fake-agent",
        search_service=FakeSearchService(),
        project_root=tmp_path,
        context_cost_governor=governor,
        global_memory=gm,
    )
    agent._resolved_index = tmp_path / ".mana/index"
    agent.search_config = SearchConfig(enable_ask_agent=False)

    # Turn 1: model calls a tool with an error
    agent.llm = FakeLLM(
        [
            FakeAIMessage(
                "",
                tool_calls=[{"id": "c1", "name": "unknown_tool", "args": {"foo": "bar"}}],
            ),
            FakeAIMessage(
                "I see unknown_tool failed. Let me answer with what I have.",
                tool_calls=[],
            ),
        ]
    )

    res1 = agent.run("Why did unknown_tool fail?", tmp_path / ".mana/index", 2, max_steps=3)
    assert res1.answer != ""

    # Verify that GlobalMemory recorded the failure
    failures = gm.store.search("unknown_tool", active_only=False)
    assert len(failures) >= 1
    failed_mem = failures[0]
    assert failed_mem.category == LearningCategory.TOOL
    assert "unknown tool" in failed_mem.error.lower()

    # Add a verified correction / prevention rule
    gm.record_correction(
        failed_mem.id,
        correction="list_files",
        prevention_rule="Do not call unknown_tool; use list_files instead.",
    )
    gm.verify_learning(failed_mem.id, success=True, evidence="list_files works")

    # Turn 2: AskAgent is asked a similar question; verified prevention rule MUST be injected in SystemMessage
    captured_messages.clear()
    agent.llm = FakeLLM(
        [
            FakeAIMessage("Answer using injected rule.", tool_calls=[]),
        ]
    )

    res2 = agent.run("Why did unknown_tool fail?", tmp_path / ".mana/index", 2, max_steps=1)
    assert len(captured_messages) > 0
    first_messages = captured_messages[0]
    sys_msgs = [m for m in first_messages if isinstance(m, SystemMessage)]
    assert len(sys_msgs) >= 1
    system_text = str(sys_msgs[0].content)
    assert "Past Learnings & Prevention Rules" in system_text
    assert "Do not call unknown_tool; use list_files instead." in system_text


# 16. Subtask failure in AutonomousExecutionController records to GlobalMemory and updates TaskBoard
def test_16_controller_subtask_failure_records_to_global_memory_and_taskboard(tmp_path: Path):
    from mana_agent.execution.continuation_decision import ContinuationAction, ContinuationDecision
    from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
    from mana_agent.multi_agent.taskboard.taskboard import TaskBoard
    from mana_agent.multi_agent.core.types import TaskStatus

    supervisor = ExecutionSupervisor(ExecutionSupervisorConfig(root=tmp_path / "supervisor", startup_recovery=False))
    taskboard = TaskBoard(root=tmp_path / "workspace")
    gm = GlobalMemory(store=LearningMemoryStore(db_path=tmp_path / "subtask_learning.sqlite3"), default_project_id=str(tmp_path))

    class FakeEvaluator:
        def evaluate(self, **kwargs):
            return ContinuationDecision(
                task_id=kwargs["task_id"],
                action=ContinuationAction.FINALIZE,
                goal_satisfied=True,
                progress_detected=True,
                evidence_summary="Subtask executed and verified",
                safe_to_continue=True,
                reason="subtask executed and verified",
            )

    controller = AutonomousExecutionController(
        supervisor=supervisor,
        taskboard=taskboard,
        continuation_evaluator=FakeEvaluator(),
        global_memory=gm,
    )

    # Subtask that encounters an error
    def failing_subtask_executor(task_item, ctx):
        return {
            "output": "",
            "error": "command failed: missing required config file config.json",
            "success": False,
        }

    ctx = {
        "subtasks": [
            {
                "local_id": "sub_1",
                "title": "Run initial configuration",
                "description": "Configure system using config.json",
                "tool": "config_tool",
                "tool_args": {"path": "config.json"},
            }
        ],
        "task_executors": {
            "sub_1": failing_subtask_executor,
            "config_tool": failing_subtask_executor,
        },
    }

    controller.execute_goal(
        task_id="subtask_fail_task",
        user_goal="Configure system using config.json",
        context=ctx,
        max_turns=1,
    )

    # Verify GlobalMemory captured the subtask failure
    tool_failures = gm.store.search("config_tool", active_only=False)
    assert len(tool_failures) >= 1
    assert "missing required config file" in tool_failures[0].error

    # Verify TaskBoard has learning reference
    tb_task = taskboard.get_task("subtask_fail_task")
    assert len(tb_task.learning_references) >= 1

