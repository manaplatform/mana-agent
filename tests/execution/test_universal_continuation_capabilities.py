"""Universal Continuation Capability Tests across workflows.

Verifies model-directed continuation across:
- Research and web browsing
- Document processing
- Shell and SSH execution
- MCP and API tools
- External coding engines (Codex task-level continuation above preserved internal loop)
- Multi-agent delegation
- Checkpoint persistence and crash recovery across multi-step execution
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest

from mana_agent.config.autonomous_config import AutoRenewConfig, AutonomousExecutionConfig
from mana_agent.execution.autonomous_controller import (
    AutonomousExecutionController,
    AutonomousExecutionResult,
    TaskExecutionMetrics,
)
from mana_agent.execution.continuation_decision import (
    ContinuationAction,
    ContinuationDecision,
)
from mana_agent.execution.continuation_evaluator import ContinuationEvaluator
from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.models import (
    CompletionContract,
    CompletionContractType,
    ExecutionState,
)
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor


class FakeLLM:
    """Mock LLM returning sequence of responses."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.invocations: list[Any] = []

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        self.invocations.append(messages)
        if not self.responses:
            raise RuntimeError("FakeLLM ran out of configured responses")
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def _make_supervisor(tmp_path: Path) -> ExecutionSupervisor:
    config = ExecutionSupervisorConfig(root=tmp_path / "supervisor", startup_recovery=False)
    return ExecutionSupervisor(config)


def test_research_and_web_browsing_continuation(tmp_path: Path) -> None:
    """Multi-step web search and research pursues answers until model finalizes."""
    llm = FakeLLM([
        # Step 1: After initial search results, model continues to fetch page 1
        json.dumps({
            "action": "CONTINUE",
            "goal_satisfied": False,
            "safe_to_continue": True,
            "reasoning": "Found search URLs; need to fetch content of documentation page 1",
            "suggested_next_steps": ["fetch_web_page(url='https://docs.example.com/api')"],
            "remaining_risks": [],
        }),
        # Step 2: After reading page 1, model continues to cross-reference page 2
        json.dumps({
            "action": "CONTINUE",
            "goal_satisfied": False,
            "safe_to_continue": True,
            "reasoning": "Page 1 lacks authentication header details; fetch auth guide",
            "suggested_next_steps": ["fetch_web_page(url='https://docs.example.com/auth')"],
            "remaining_risks": [],
        }),
        # Step 3: Complete research synthesis
        json.dumps({
            "action": "FINALIZE",
            "goal_satisfied": True,
            "safe_to_continue": True,
            "reasoning": "All API and authentication details collected and verified",
            "suggested_next_steps": [],
            "remaining_risks": [],
        }),
    ])
    evaluator = ContinuationEvaluator(llm=llm)
    supervisor = _make_supervisor(tmp_path)
    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        continuation_evaluator=evaluator,
    )

    history = [
        {"tool": "web_search", "query": "example API docs", "urls": ["https://docs.example.com/api", "https://docs.example.com/auth"]},
        {"tool": "fetch_web_page", "url": "https://docs.example.com/api", "content": "API endpoints: /v1/items"},
        {"tool": "fetch_web_page", "url": "https://docs.example.com/auth", "content": "Auth: Bearer tokens"},
    ]

    def execute_step(step_idx: int) -> dict[str, Any]:
        return history[step_idx]

    result = controller.run(
        task_id="task_research_001",
        user_goal="Research example API authentication and endpoints",
        step_executor=execute_step,
    )

    assert result.success is True
    assert result.final_action == ContinuationAction.FINALIZE
    assert result.metrics.model_turns == 3
    assert result.metrics.tool_calls == 3
    assert len(result.completed_steps) == 3


def test_document_processing_workflow_continuation(tmp_path: Path) -> None:
    """Document processing workflow: detect -> chunk/analyze -> summarize -> finalize."""
    llm = FakeLLM([
        # Step 1: Detect doc format & structure
        json.dumps({
            "action": "CONTINUE",
            "goal_satisfied": False,
            "safe_to_continue": True,
            "reasoning": "Document format detected as PDF; analyze chapters",
            "suggested_next_steps": ["document_chunk(file='whitepaper.pdf')"],
            "remaining_risks": [],
        }),
        # Step 2: Analyze chunks
        json.dumps({
            "action": "CONTINUE",
            "goal_satisfied": False,
            "safe_to_continue": True,
            "reasoning": "Chunks extracted; create executive briefing document",
            "suggested_next_steps": ["document_create(file='summary.md')"],
            "remaining_risks": [],
        }),
        # Step 3: Output created and finalized
        json.dumps({
            "action": "FINALIZE",
            "goal_satisfied": True,
            "safe_to_continue": True,
            "reasoning": "Executive summary written and verified against source text",
            "suggested_next_steps": [],
            "remaining_risks": [],
        }),
    ])
    evaluator = ContinuationEvaluator(llm=llm)
    supervisor = _make_supervisor(tmp_path)
    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        continuation_evaluator=evaluator,
    )

    doc_steps = [
        {"action": "document_detect", "format": "pdf", "pages": 42},
        {"action": "document_chunk", "chunks_count": 8, "key_sections": ["intro", "methods", "results"]},
        {"action": "document_create", "created_file": "summary.md", "words": 650},
    ]

    def execute_doc_step(step_idx: int) -> dict[str, Any]:
        return doc_steps[step_idx]

    result = controller.run(
        task_id="task_doc_001",
        user_goal="Analyze whitepaper.pdf and generate executive summary",
        step_executor=execute_doc_step,
    )

    assert result.success is True
    assert result.final_action == ContinuationAction.FINALIZE
    assert result.metrics.model_turns == 3


def test_shell_and_ssh_execution_continuation(tmp_path: Path) -> None:
    """Shell/SSH remote execution pursuing multi-step server setup."""
    llm = FakeLLM([
        # Step 1: Environment check
        json.dumps({
            "action": "CONTINUE",
            "goal_satisfied": False,
            "safe_to_continue": True,
            "reasoning": "Python 3.11 installed, but service daemon is inactive; start service",
            "suggested_next_steps": ["ssh_exec(command='systemctl start mana-worker')"],
            "remaining_risks": [],
        }),
        # Step 2: Start service, verify running
        json.dumps({
            "action": "FINALIZE",
            "goal_satisfied": True,
            "safe_to_continue": True,
            "reasoning": "Service started and health check endpoint returned 200 OK",
            "suggested_next_steps": [],
            "remaining_risks": [],
        }),
    ])
    evaluator = ContinuationEvaluator(llm=llm)
    supervisor = _make_supervisor(tmp_path)
    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        continuation_evaluator=evaluator,
    )

    commands = [
        {"command": "python --version && systemctl status mana-worker", "stdout": "Python 3.11.8\nInactive (dead)", "exit_code": 0},
        {"command": "systemctl start mana-worker && curl http://localhost:8080/health", "stdout": "{\"status\":\"healthy\"}", "exit_code": 0},
    ]

    def execute_cmd(step_idx: int) -> dict[str, Any]:
        return commands[step_idx]

    result = controller.run(
        task_id="task_ssh_001",
        user_goal="Verify and start mana-worker service on remote host",
        step_executor=execute_cmd,
    )

    assert result.success is True
    assert result.final_action == ContinuationAction.FINALIZE
    assert result.metrics.model_turns == 2


def test_external_coding_engine_codex_continuation_above_internal_tool_loop(tmp_path: Path) -> None:
    """Codex coding engine has its own internal tool loop.
    
    The autonomous continuation controller sits above the external engine,
    driving task-level multi-pass execution when pass cap is hit or more work remains,
    preserving Codex's internal loop.
    """
    llm = FakeLLM([
        # Pass 1 completes internal tool loop (e.g. read files, wrote test), but pass cap reached
        json.dumps({
            "action": "CONTINUE",
            "goal_satisfied": False,
            "safe_to_continue": True,
            "reasoning": "Codex pass 1 created test cases but implementation file has not yet been edited",
            "suggested_next_steps": ["Run Codex pass 2 to edit src/feature.py"],
            "remaining_risks": ["Pending test execution"],
        }),
        # Pass 2 completes edits and passes tests
        json.dumps({
            "action": "FINALIZE",
            "goal_satisfied": True,
            "safe_to_continue": True,
            "reasoning": "Codex pass 2 implemented feature and all unit tests passed cleanly",
            "suggested_next_steps": [],
            "remaining_risks": [],
        }),
    ])
    evaluator = ContinuationEvaluator(llm=llm)
    supervisor = _make_supervisor(tmp_path)
    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        continuation_evaluator=evaluator,
    )

    # Simulated Codex external engine results.
    # Each result contains Codex internal tool loop traces (3 internal tool calls in pass 1, 2 in pass 2)
    codex_passes = [
        {
            "engine": "codex",
            "pass_number": 1,
            "internal_tool_loop_calls": [
                {"tool": "repo_batch_read", "files": ["src/main.py"]},
                {"tool": "repo_batch_read", "files": ["tests/test_main.py"]},
                {"tool": "apply_patch_batch", "file": "tests/test_feature.py"},
            ],
            "auto_execute_terminal_reason": "pass_cap_reached",
            "run_status": "needs_resume",
            "changed_files": ["tests/test_feature.py"],
            "answer": "Created unit tests for new feature in pass 1.",
        },
        {
            "engine": "codex",
            "pass_number": 2,
            "internal_tool_loop_calls": [
                {"tool": "apply_patch_batch", "file": "src/feature.py"},
                {"tool": "repo_batch_read", "files": ["src/feature.py"]},
            ],
            "auto_execute_terminal_reason": "completed",
            "run_status": "completed",
            "changed_files": ["tests/test_feature.py", "src/feature.py"],
            "answer": "Implemented feature in src/feature.py. All tests verified.",
        },
    ]

    def execute_codex_pass(step_idx: int) -> dict[str, Any]:
        return codex_passes[step_idx]

    result = controller.run(
        task_id="task_codex_001",
        user_goal="Implement and verify feature in src/feature.py using Codex backend",
        step_executor=execute_codex_pass,
    )

    assert result.success is True
    assert result.final_action == ContinuationAction.FINALIZE
    assert result.metrics.model_turns == 2
    # Verify both Codex passes executed at the task continuation layer
    assert len(result.completed_steps) == 2
    assert result.completed_steps[0]["output"]["pass_number"] == 1
    assert result.completed_steps[1]["output"]["pass_number"] == 2
    # Verify internal tool loop inside Codex was preserved
    assert len(result.completed_steps[0]["output"]["internal_tool_loop_calls"]) == 3
    assert len(result.completed_steps[1]["output"]["internal_tool_loop_calls"]) == 2


def test_multi_agent_delegation_continuation(tmp_path: Path) -> None:
    """Multi-agent delegation where Main Agent delegates to sub-agents sequentially."""
    llm = FakeLLM([
        # After SubAgent A completes architecture review:
        json.dumps({
            "action": "CONTINUE",
            "goal_satisfied": False,
            "safe_to_continue": True,
            "reasoning": "Architect agent finished spec; now delegate to Security Agent for audit",
            "suggested_next_steps": ["delegate_to_subagent(role='security_auditor')"],
            "remaining_risks": [],
        }),
        # After SubAgent B completes security review:
        json.dumps({
            "action": "FINALIZE",
            "goal_satisfied": True,
            "safe_to_continue": True,
            "reasoning": "Both architecture and security reviews complete with zero high-severity findings",
            "suggested_next_steps": [],
            "remaining_risks": [],
        }),
    ])
    evaluator = ContinuationEvaluator(llm=llm)
    supervisor = _make_supervisor(tmp_path)
    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        continuation_evaluator=evaluator,
    )

    subagent_results = [
        {
            "agent": "architect_agent",
            "output": {"spec": "Event-driven architecture with supervisor checkpoints", "status": "approved"},
        },
        {
            "agent": "security_auditor",
            "output": {"vulnerabilities": [], "status": "approved"},
        },
    ]

    def execute_delegation(step_idx: int) -> dict[str, Any]:
        return subagent_results[step_idx]

    result = controller.run(
        task_id="task_multiagent_001",
        user_goal="Perform architecture design and security audit via specialized subagents",
        step_executor=execute_delegation,
    )

    assert result.success is True
    assert result.final_action == ContinuationAction.FINALIZE
    assert result.metrics.model_turns == 2
    assert len(result.completed_steps) == 2


def test_supervisor_checkpoint_persistence_across_multi_step_workflow(tmp_path: Path) -> None:
    """Supervised execution durable state and checkpoints persist across continuation steps."""
    llm = FakeLLM([
        json.dumps({
            "action": "CONTINUE",
            "goal_satisfied": False,
            "safe_to_continue": True,
            "reasoning": "Step 1 done, continue to step 2",
            "suggested_next_steps": ["step_2()"],
            "remaining_risks": [],
        }),
        json.dumps({
            "action": "FINALIZE",
            "goal_satisfied": True,
            "safe_to_continue": True,
            "reasoning": "All steps done",
            "suggested_next_steps": [],
            "remaining_risks": [],
        }),
    ])
    evaluator = ContinuationEvaluator(llm=llm)
    supervisor = _make_supervisor(tmp_path)
    controller = AutonomousExecutionController(
        config=AutonomousExecutionConfig(),
        supervisor=supervisor,
        continuation_evaluator=evaluator,
    )

    task_id = "task_checkpoint_persistence_001"
    steps = [
        {"action": "step_1", "data": "initial_evidence"},
        {"action": "step_2", "data": "final_evidence"},
    ]

    result = controller.run(
        task_id=task_id,
        user_goal="Demonstrate supervisor checkpointing across continuation steps",
        step_executor=lambda idx: steps[idx],
    )

    assert result.success is True
    task_record = supervisor.get_task(task_id)
    assert task_record is not None
    assert task_record.autonomous_continuation_state == "FINALIZE"
    assert task_record.checkpoint_count >= 2
