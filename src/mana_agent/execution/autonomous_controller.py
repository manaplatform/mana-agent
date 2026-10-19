"""Universal Model-Directed Autonomous Execution Controller.

Integrates with Agent Orchestrator, Execution Supervisor, accounting infrastructure,
task state machine, context compactor, and checkpoint system across every Mana-Agent capability.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import contextvars
from enum import Enum
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
from pydantic import Field

from mana_agent.config.autonomous_config import AutonomousExecutionConfig
from mana_agent.context_cost.accounting import ModelTokenAccountingService
from mana_agent.context_cost.governor import ContextCostGovernor
from mana_agent.execution.continuation_decision import (
    ContinuationAction,
    ContinuationDecision,
    ContinuationDecisionError,
)
from mana_agent.execution.continuation_evaluator import ContinuationEvaluator
from mana_agent.execution_supervisor.models import (
    ExecutionState,
    StrictModel,
    TaskRecord,
    utc_now,
)
from mana_agent.execution_supervisor.supervisor import ExecutionSupervisor
from mana_agent.multi_agent.core.types import TaskStatus

if TYPE_CHECKING:
    from mana_agent.gateway.context_compactor import ContextCompactor
    from mana_agent.multi_agent.taskboard.taskboard import TaskBoard

logger = logging.getLogger(__name__)


class ExecutionStrategy(str, Enum):
    """Execution strategy classifications integrated directly into the Autonomous Loop."""

    SIMPLE = "simple"
    CODING = "coding"
    MULTI_TASK = "multi_task"
    RESEARCH = "research"
    TOOL = "tool"
    BATCH = "batch"
    GITOPS = "gitops"
    CONVERSATION = "conversation"


def normalize_strategy(strategy: str | None) -> str:
    """Normalize routing classifications and hints into standard ExecutionStrategy values."""
    s = str(strategy or "").strip().lower()
    if s in {"multi_task", "compound", "dag"}:
        return ExecutionStrategy.MULTI_TASK.value
    if s in {"coding", "edit", "coding_task", "code"}:
        return ExecutionStrategy.CODING.value
    if s in {
        "repo_search",
        "semantic_qa",
        "research",
        "web_search",
        "github_search",
        "analyze",
        "search",
    }:
        return ExecutionStrategy.RESEARCH.value
    if s in {
        "tool",
        "tool_execution",
        "high_risk_tool",
        "shell",
        "mcp",
        "browser",
        "browser_task",
        "command",
        "artifact",
        "gmail",
    }:
        return ExecutionStrategy.TOOL.value
    if s in {"batch", "batch_file"}:
        return ExecutionStrategy.BATCH.value
    if s in {"gitops", "git"}:
        return ExecutionStrategy.GITOPS.value
    if s in {"conversation", "chat", "qa", "simple"}:
        return ExecutionStrategy.SIMPLE.value
    return s or ExecutionStrategy.SIMPLE.value


def _publish_hub_event(event_data: dict[str, Any]) -> None:
    """Publish an execution event to the central event hub safely."""
    try:
        from mana_agent.services.execution_event_hub import get_execution_event_hub

        get_execution_event_hub().publish(event_data)
    except Exception:
        pass


def _extract_target_paths_safe(tool_name: str, args: dict[str, Any]) -> set[str]:
    """Extract affected target file paths for conflict detection."""
    try:
        from mana_agent.coding.scheduler import extract_target_paths

        return extract_target_paths(tool_name, args)
    except Exception:
        paths = set()
        for k in ("path", "file_path", "filePath", "target", "source", "destination"):
            v = args.get(k)
            if isinstance(v, str) and v.strip():
                paths.add(v.strip().replace("\\", "/").lstrip("./"))
        return paths


class TaskExecutionMetrics(StrictModel):
    """Centralized execution metrics tracked across the entire task tree."""

    task_id: str
    root_task_id: str
    model_turns: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    tokens_consumed: int = Field(default=0, ge=0)
    monetary_cost: float = Field(default=0.0, ge=0)
    start_time: datetime = Field(default_factory=utc_now)
    duration_seconds: float = Field(default=0.0, ge=0)
    renewals_count: int = Field(default=0, ge=0)
    consecutive_no_progress_cycles: int = Field(default=0, ge=0)
    verification_results: list[dict[str, Any]] = Field(default_factory=list)
    progress_history: list[dict[str, Any]] = Field(default_factory=list)
    child_task_metrics: dict[str, Any] = Field(default_factory=dict)

    def record_step(
        self,
        *,
        model_turns: int = 1,
        tool_calls: int = 0,
        tokens: int = 0,
        cost: float = 0.0,
        progress_summary: str = "",
        evidence_fingerprint: str = "",
    ) -> None:
        """Update metrics monotonically after each meaningful execution turn."""
        self.model_turns += max(0, model_turns)
        self.tool_calls += max(0, tool_calls)
        self.tokens_consumed += max(0, tokens)
        self.monetary_cost += max(0.0, cost)
        now = datetime.now(timezone.utc)
        self.duration_seconds = max(0.0, (now - self.start_time).total_seconds())
        self.progress_history.append(
            {
                "timestamp": now.isoformat(),
                "model_turns": self.model_turns,
                "tool_calls": self.tool_calls,
                "tokens": self.tokens_consumed,
                "cost": self.monetary_cost,
                "summary": progress_summary,
                "fingerprint": evidence_fingerprint,
            }
        )


class AutonomousExecutionResult(StrictModel):
    """Normalized terminal outcome of an autonomous execution pursuit."""

    task_id: str
    status: str
    goal_satisfied: bool
    answer: str = ""
    metrics: TaskExecutionMetrics
    decision: ContinuationDecision | None = None
    verification_passed: bool = False
    error: str = ""
    trace: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    completed_steps: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.goal_satisfied and self.status == "completed"

    @property
    def final_action(self) -> ContinuationAction | None:
        return self.decision.action if self.decision else None

    @property
    def final_response(self) -> str:
        return self.answer


def _compute_evidence_fingerprint(data: Any) -> str:
    """Compute a deterministic SHA-256 fingerprint of step evidence."""
    try:
        serialized = json.dumps(data, sort_keys=True, default=str)
    except Exception:
        serialized = str(data)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class AutonomousExecutionController:
    """Unified, model-directed execution controller across all Mana-Agent workflows.
    
    The model controls semantic continuation decisions until the goal is satisfied,
    blocked, or requires user input. The runtime enforces permissions, budgets,
    verification, cancellation, crash recovery, and context compaction.
    """

    def __init__(
        self,
        *,
        config: AutonomousExecutionConfig | None = None,
        supervisor: ExecutionSupervisor,
        taskboard: Any | None = None,
        accounting_service: ModelTokenAccountingService | None = None,
        context_governor: ContextCostGovernor | None = None,
        compactor: ContextCompactor | None = None,
        evaluator: ContinuationEvaluator | None = None,
        continuation_evaluator: ContinuationEvaluator | None = None,
        global_memory: Any | None = None,
        batch_manager: Any | None = None,
        router: Any | None = None,
        coding_engine: Any | None = None,
        tools: Any | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        maximum_concurrency: int = 4,
    ) -> None:
        self.config = config or AutonomousExecutionConfig()
        self.supervisor = supervisor
        self.taskboard = taskboard
        self.taskboard_operations = None
        if taskboard is not None:
            from mana_agent.multi_agent.taskboard.operations import TaskBoardOperationsHandler
            self.taskboard_operations = TaskBoardOperationsHandler(taskboard)
        self.accounting_service = accounting_service
        self.context_governor = context_governor
        self.compactor = compactor
        self.evaluator = evaluator or continuation_evaluator
        if global_memory is not None:
            self.global_memory = global_memory
        else:
            try:
                from mana_agent.memory.learning import get_global_memory
                self.global_memory = get_global_memory()
            except Exception:
                self.global_memory = None
        if batch_manager is not None:
            self.batch_manager = batch_manager
        else:
            try:
                from mana_agent.batch.manager import BatchLifecycleManager
                self.batch_manager = BatchLifecycleManager(accounting_service=accounting_service)
            except Exception:
                self.batch_manager = None
        self.router = router
        self.coding_engine = coding_engine
        self.tools = dict(tools) if isinstance(tools, dict) else (tools or {})
        self.event_sink = event_sink
        self.maximum_concurrency = max(1, int(maximum_concurrency))

    def _emit(
        self,
        event_type: str,
        data: dict[str, Any],
        sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Emit an execution event to the hub and active sinks."""
        event = {
            "event_type": event_type,
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        _publish_hub_event(event)
        if sink is not None:
            try:
                sink(event)
            except Exception:
                pass
        if self.event_sink is not None and self.event_sink is not sink:
            try:
                self.event_sink(event)
            except Exception:
                pass

    def run(
        self,
        *,
        task_id: str,
        user_goal: str,
        step_executor: Callable[..., Any] | None = None,
        step_runner: Callable[..., Any] | None = None,
        context: dict[str, Any] | None = None,
        initial_metrics: TaskExecutionMetrics | None = None,
        strategy_hint: str | None = None,
        session_id: str | None = None,
        workspace_root: str | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> AutonomousExecutionResult:
        """Convenience method running the universal continuation loop."""
        return self.execute_goal(
            task_id=task_id,
            user_goal=user_goal,
            step_runner=step_runner,
            step_executor=step_executor,
            context=context,
            initial_metrics=initial_metrics,
            strategy_hint=strategy_hint,
            session_id=session_id,
            workspace_root=workspace_root,
            event_sink=event_sink,
        )

    def execute_goal(
        self,
        *,
        task_id: str,
        user_goal: str,
        step_runner: Callable[..., Any] | None = None,
        step_executor: Callable[..., Any] | None = None,
        max_turns: int | None = None,
        context: dict[str, Any] | None = None,
        initial_metrics: TaskExecutionMetrics | None = None,
        strategy_hint: str | None = None,
        session_id: str | None = None,
        workspace_root: str | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> AutonomousExecutionResult:
        """Run the universal model-directed autonomous continuation loop."""
        runner = step_runner or step_executor
        if runner is None:
            runner = self._build_internal_loop_runner(
                task_id=task_id,
                user_goal=user_goal,
                strategy_hint=strategy_hint,
                event_sink=event_sink,
            )

        ctx = dict(context or {})
        if strategy_hint and "strategy_hint" not in ctx:
            ctx["strategy_hint"] = strategy_hint

        # Ensure task exists and is in RUNNING state with lease
        try:
            task = self.supervisor.get_task(task_id)
        except Exception:
            task = self.supervisor.create_task(
                task_id=task_id,
                routing_decision_id="autonomous_decision",
                requested_operation="autonomous_execution",
                normalized_intent=user_goal,
            )

        if task.state == ExecutionState.CREATED:
            task = self.supervisor.queue(task.task_id)
        if task.state == ExecutionState.QUEUED:
            task, token = self.supervisor.acquire_lease(task.task_id, owner="autonomous_controller")
            self.supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)
            task = self.supervisor.get_task(task.task_id)
        elif task.state == ExecutionState.LEASED:
            token = task.lease_token
            if not token:
                task, token = self.supervisor.acquire_lease(task.task_id, owner="autonomous_controller")
            self.supervisor.start(task.task_id, attempt_id=task.attempt_id, lease_token=token)
            task = self.supervisor.get_task(task.task_id)

        root_task_id = task.root_task_id or task_id

        # Initialize or synchronize TaskBoard
        if self.taskboard is not None:
            try:
                tb_task = self.taskboard.get_task(task_id)
            except Exception:
                tb_task = self.taskboard.create_task(
                    title=user_goal[:80] or "Autonomous Task",
                    user_request=user_goal,
                    normalized_goal=user_goal,
                )
                if tb_task.task_id != task_id and task_id not in self.taskboard.tasks:
                    self.taskboard.tasks[task_id] = tb_task
                    tb_task.task_id = task_id
                    tb_task.root_task_id = root_task_id
                    self.taskboard.save()
            tb_task.supervisor_execution_id = task_id
            tb_task.supervisor_state = task.state.value
            tb_task.supervisor_state_version = task.state_version
            self.taskboard.save()

        metrics = initial_metrics or TaskExecutionMetrics(
            task_id=task_id,
            root_task_id=root_task_id,
        )

        effective_max_turns = max_turns if max_turns is not None else self.config.max_model_turns
        effective_max_tools = self.config.max_tool_calls

        step_index = 0
        completed_steps: list[dict[str, Any]] = []
        overall_trace: list[dict[str, Any]] = []
        last_evidence_fingerprint = ""
        last_answer = ""
        warnings: list[str] = []
        last_failed_memory_id: str | None = None

        # Pre-execution retrieval from GlobalMemory
        if self.global_memory is not None:
            try:
                past_learnings = self.global_memory.retrieve_relevant(
                    action_or_query=user_goal,
                    limit=5,
                )
                if past_learnings:
                    ctx["learning_memories"] = past_learnings
                    rules_text = self.global_memory.format_retrieved_rules(past_learnings)
                    ctx["learned_prevention_rules"] = rules_text
                    if self.taskboard is not None:
                        for mem in past_learnings:
                            try:
                                self.taskboard.add_learning_reference(task_id, mem.id)
                            except Exception:
                                pass
            except Exception as mem_exc:
                logger.debug("Pre-execution memory retrieval failed safely: %s", mem_exc)

        # Pre-execution resumption of pending OpenAI Batch API jobs
        if self.batch_manager is not None and getattr(self.batch_manager.config, "auto_resume", True):
            try:
                self.batch_manager.resume_pending_batches()
            except Exception as resume_exc:
                logger.debug("Pre-execution batch auto-resume failed safely: %s", resume_exc)

        self._emit(
            "loop.turn_received",
            {"task_id": task_id, "user_goal": user_goal, "title": "User turn received", "status": "running"},
            sink=event_sink,
        )
        self._emit(
            "loop.initialized",
            {"task_id": task_id, "root_task_id": root_task_id, "title": "Autonomous Loop initialized", "status": "success"},
            sink=event_sink,
        )

        def _finish(res: AutonomousExecutionResult) -> AutonomousExecutionResult:
            st = "success" if res.goal_satisfied and res.status == "completed" else res.status
            self._emit(
                "loop.completed",
                {
                    "task_id": task_id,
                    "status": res.status,
                    "goal_satisfied": res.goal_satisfied,
                    "title": "Execution completed" if res.status == "completed" else f"Execution {res.status}",
                    "status": st,
                },
                sink=event_sink,
            )
            return res

        while True:
            # 0. Provide live TaskBoard progress, operations, and batch manager in execution context
            ctx.setdefault("evidence", [])
            ctx.setdefault("touched_files", [])
            ctx.setdefault("tool_events", [])
            ctx["batch_manager"] = self.batch_manager
            if self.taskboard is not None:
                ctx["taskboard_progress"] = self.taskboard.get_progress(root_task_id=root_task_id)
                ctx["taskboard"] = self.taskboard
                ctx["taskboard_operations"] = self.taskboard_operations

            # 1. Hard Gate: Cancellation Check
            if self.supervisor.cancellation_requested(task_id):
                self.supervisor.cancel(task_id, reason="user_cancelled")
                if self.taskboard is not None:
                    try:
                        self.taskboard.mark_cancelled(task_id, reason="user_cancelled")
                    except Exception:
                        pass
                return _finish(AutonomousExecutionResult(
                    task_id=task_id,
                    status="cancelled",
                    goal_satisfied=False,
                    answer="Task was cancelled by user or operator.",
                    metrics=metrics,
                    error="cancelled",
                    trace=overall_trace,
                    warnings=warnings,
                    completed_steps=completed_steps,
                ))

            # 2. Checkpoint before execution step
            current_task = self.supervisor.get_task(task_id)
            if current_task.state == ExecutionState.RUNNING and current_task.attempt_id and current_task.lease_token:
                try:
                    batch_job_ids: list[str] = []
                    if self.batch_manager is not None:
                        try:
                            batch_job_ids = [b.batch_id for b in self.batch_manager.store.list_jobs(task_id=task_id)]
                        except Exception:
                            pass
                    self.supervisor.checkpoint(
                        task_id,
                        attempt_id=current_task.attempt_id,
                        lease_token=current_task.lease_token,
                        resume_payload={
                            "step_index": step_index,
                            "metrics": metrics.model_dump(mode="json"),
                            "batch_job_ids": batch_job_ids,
                        },
                        resume_cursor=f"step_{step_index}",
                        completed_steps=[str(s.get("summary", "")) for s in completed_steps],
                        pending_steps=["execute_step", "evaluate_continuation"],
                    )
                except Exception as cp_exc:
                    logger.warning("Checkpoint before step %d failed: %s", step_index, cp_exc)

            # 4. Execute Step via Capability Runner
            step_t0 = time.perf_counter()
            try:
                try:
                    raw_step_result = runner(step_index, ctx)
                except TypeError:
                    raw_step_result = runner(step_index)
                step_result = raw_step_result if isinstance(raw_step_result, dict) else {"output": raw_step_result}
            except Exception as exc:
                logger.exception("Step runner execution failed on task %s step %d", task_id, step_index)
                try:
                    self.supervisor.transition(task_id, ExecutionState.FAILED)
                except Exception:
                    pass
                return _finish(AutonomousExecutionResult(
                    task_id=task_id,
                    status="failed",
                    goal_satisfied=False,
                    answer="",
                    metrics=metrics,
                    error=f"Step execution error: {exc}",
                    trace=overall_trace,
                    warnings=warnings,
                    completed_steps=completed_steps,
                ))

            step_duration_ms = (time.perf_counter() - step_t0) * 1000.0

            # Immediate cancellation check if cancelled during step execution
            if self.supervisor.cancellation_requested(task_id):
                self.supervisor.cancel(task_id, reason="user_cancelled")
                if self.taskboard is not None:
                    try:
                        self.taskboard.mark_cancelled(task_id, reason="user_cancelled")
                    except Exception:
                        pass
                return _finish(AutonomousExecutionResult(
                    task_id=task_id,
                    status="cancelled",
                    goal_satisfied=False,
                    answer="Task was cancelled by user or operator.",
                    metrics=metrics,
                    error="cancelled",
                    trace=overall_trace,
                    warnings=warnings,
                    completed_steps=completed_steps,
                ))

            # 5. Extract Step Evidence & Update Central Metrics
            step_answer = str(step_result.get("answer") or step_result.get("output") or "").strip()
            if step_answer:
                last_answer = step_answer
            tool_calls = list(step_result.get("tool_calls") or [])
            if not tool_calls:
                if step_result.get("tool"):
                    tool_calls = [{"name": str(step_result.get("tool")), "args": step_result}]
                elif step_result.get("action"):
                    tool_calls = [{"name": str(step_result.get("action")), "args": step_result}]
                elif step_result.get("command"):
                    tool_calls = [{"name": "shell_command", "args": step_result}]
                elif step_result.get("agent"):
                    tool_calls = [{"name": "subagent_delegation", "args": step_result}]

            tool_results = list(step_result.get("tool_results") or [])
            tokens_used = int(step_result.get("tokens_used") or 0)
            cost_incurred = float(step_result.get("cost") or 0.0)
            step_warnings = [str(w) for w in (step_result.get("warnings") or []) if str(w).strip()]
            warnings.extend(step_warnings)

            step_trace = step_result.get("trace") or []
            if isinstance(step_trace, list):
                overall_trace.extend(step_trace)

            evidence_summary = str(step_result.get("summary") or step_answer or "step completed")
            evidence_fingerprint = _compute_evidence_fingerprint(
                {
                    "answer": step_answer,
                    "tool_calls": tool_calls,
                    "tool_results": tool_results,
                    "changed_files": step_result.get("changed_files") or [],
                }
            )

            metrics.record_step(
                model_turns=1,
                tool_calls=len(tool_calls),
                tokens=tokens_used,
                cost=cost_incurred,
                progress_summary=evidence_summary,
                evidence_fingerprint=evidence_fingerprint,
            )

            # Synchronize TaskBoard with step results
            if self.taskboard is not None:
                try:
                    if step_result.get("changed_files"):
                        self.taskboard.add_files_touched(task_id, step_result.get("changed_files"))
                    for tc in tool_calls:
                        self.taskboard.record_tool_event(task_id, tc)
                    if evidence_summary:
                        self.taskboard.add_evidence(task_id, evidence_summary)
                except Exception as tb_exc:
                    logger.debug("TaskBoard update after step %d failed: %s", step_index, tb_exc)

            # 5b. Observe Step Mistakes & Corrections with GlobalMemory
            step_error = step_result.get("error")
            if step_error:
                if self.global_memory is not None:
                    try:
                        failed_mem = self.global_memory.record_failure(
                            attempted_action=tool_calls[0].get("name", "step_execution") if tool_calls else "step_execution",
                            error=str(step_error),
                            category="EXECUTION",
                            context=f"Goal: {user_goal} | Step {step_index}",
                            execution_id=task_id,
                        )
                        if failed_mem:
                            last_failed_memory_id = failed_mem.id
                            if self.taskboard is not None:
                                try:
                                    self.taskboard.add_learning_reference(task_id, failed_mem.id)
                                except Exception:
                                    pass
                    except Exception as mem_exc:
                        logger.debug("GlobalMemory record_failure failed safely: %s", mem_exc)
            elif last_failed_memory_id and not step_error:
                if self.global_memory is not None:
                    try:
                        self.global_memory.record_correction(
                            memory_id=last_failed_memory_id,
                            correction=evidence_summary,
                            execution_id=task_id,
                            evidence=step_answer,
                        )
                    except Exception as mem_exc:
                        logger.debug("GlobalMemory record_correction failed safely: %s", mem_exc)
                last_failed_memory_id = None

            # 6. Check for Pending Human Approval Gate
            if step_result.get("is_approval_pending"):
                current_task = self.supervisor.get_task(task_id)
                if current_task.state == ExecutionState.RUNNING:
                    try:
                        self.supervisor.transition(task_id, ExecutionState.WAITING, reason="waiting_for_approval")
                    except Exception:
                        pass
                if self.taskboard is not None:
                    try:
                        self.taskboard.update_status(task_id, TaskStatus.WAITING_FOR_TOOLS, reason="waiting_for_approval")
                    except Exception:
                        pass
                return _finish(AutonomousExecutionResult(
                    task_id=task_id,
                    status="waiting",
                    goal_satisfied=False,
                    answer=step_answer or "Action requires human approval before continuing.",
                    metrics=metrics,
                    error="waiting_for_approval",
                    trace=overall_trace,
                    warnings=warnings,
                    completed_steps=completed_steps,
                ))

            # 7. Progress Detection
            if evidence_fingerprint == last_evidence_fingerprint and not step_result.get("changed_files"):
                metrics.consecutive_no_progress_cycles += 1
            else:
                metrics.consecutive_no_progress_cycles = 0
            last_evidence_fingerprint = evidence_fingerprint

            if (
                self.config.auto_renew.stop_on_no_progress
                and metrics.consecutive_no_progress_cycles >= self.config.max_no_progress_cycles
            ):
                blocker_msg = (
                    f"Execution halted: {metrics.consecutive_no_progress_cycles} consecutive steps "
                    "produced no observable progress toward the goal."
                )
                current_task = self.supervisor.get_task(task_id)
                if current_task.state == ExecutionState.RUNNING:
                    try:
                        self.supervisor.transition(task_id, ExecutionState.FAILED, reason="no_progress_stopped")
                    except Exception:
                        pass
                if self.taskboard is not None:
                    try:
                        self.taskboard.mark_blocked(task_id, reason=blocker_msg)
                    except Exception:
                        pass
                return _finish(AutonomousExecutionResult(
                    task_id=task_id,
                    status="blocked",
                    goal_satisfied=False,
                    answer=blocker_msg,
                    metrics=metrics,
                    error="no_progress_stopped",
                    trace=overall_trace,
                    warnings=warnings,
                    completed_steps=completed_steps,
                ))

            # Record step in completed steps
            completed_steps.append(
                {
                    "step_index": step_index,
                    "summary": evidence_summary,
                    "tool_name": tool_calls[0].get("name") if tool_calls else "llm_turn",
                    "status": "ok" if not step_result.get("error") else "error",
                    "duration_ms": step_duration_ms,
                    "output": step_result,
                }
            )

            # Check Soft Limits & Auto-Renewal
            soft_turns_reached = (metrics.model_turns + self.config.soft_limit_turns_headroom) >= effective_max_turns
            soft_tools_reached = (metrics.tool_calls + self.config.soft_limit_tools_headroom) >= effective_max_tools

            if soft_turns_reached or soft_tools_reached:
                if self.config.auto_renew.enabled:
                    renewal_ok, renewal_reason = self._try_auto_renew(
                        task=task,
                        metrics=metrics,
                        effective_max_turns=effective_max_turns,
                        effective_max_tools=effective_max_tools,
                    )
                    if renewal_ok:
                        effective_max_turns += self.config.auto_renew.renew_model_turns
                        effective_max_tools += self.config.auto_renew.renew_tool_calls
                        metrics.renewals_count += 1
                        logger.info(
                            "Autonomous execution budget renewed: task=%s renewal=%d turns=%d tools=%d reason=%s",
                            task_id,
                            metrics.renewals_count,
                            effective_max_turns,
                            effective_max_tools,
                            renewal_reason,
                        )
                    else:
                        # Cannot renew due to budget, max renewals, or no progress
                        if self.config.auto_renew.require_approval_above_budget and "budget" in renewal_reason:
                            current_task = self.supervisor.get_task(task_id)
                            if current_task.state == ExecutionState.RUNNING:
                                try:
                                    self.supervisor.transition(task_id, ExecutionState.WAITING, reason="waiting_for_approval")
                                except Exception:
                                    pass
                            return _finish(AutonomousExecutionResult(
                                task_id=task_id,
                                status="waiting",
                                goal_satisfied=False,
                                answer=f"Task paused: Budget extension requires user approval ({renewal_reason}).",
                                metrics=metrics,
                                error="waiting_for_approval",
                                trace=overall_trace,
                                warnings=warnings,
                                completed_steps=completed_steps,
                            ))
                        # Mark budget exhausted
                        current_task = self.supervisor.get_task(task_id)
                        if current_task.state == ExecutionState.RUNNING:
                            try:
                                self.supervisor.transition(task_id, ExecutionState.BUDGET_EXHAUSTED)
                            except Exception:
                                pass
                        return _finish(AutonomousExecutionResult(
                            task_id=task_id,
                            status="budget_exhausted",
                            goal_satisfied=False,
                            answer=f"Autonomous execution budget exhausted: {renewal_reason}.",
                            metrics=metrics,
                            error="budget_exhausted",
                            trace=overall_trace,
                            warnings=warnings,
                            completed_steps=completed_steps,
                        ))
                else:
                    current_task = self.supervisor.get_task(task_id)
                    if current_task.state == ExecutionState.RUNNING:
                        try:
                            self.supervisor.transition(task_id, ExecutionState.BUDGET_EXHAUSTED)
                        except Exception:
                            pass
                    return _finish(AutonomousExecutionResult(
                        task_id=task_id,
                        status="budget_exhausted",
                        goal_satisfied=False,
                        answer="Autonomous execution soft limits reached and auto_renew is disabled.",
                        metrics=metrics,
                        error="budget_exhausted",
                        trace=overall_trace,
                        warnings=warnings,
                        completed_steps=completed_steps,
                    ))

            # 8. Checkpoint state after step
            current_task = self.supervisor.get_task(task_id)
            if current_task.state == ExecutionState.RUNNING and current_task.attempt_id and current_task.lease_token:
                try:
                    post_batch_job_ids: list[str] = []
                    if self.batch_manager is not None:
                        try:
                            post_batch_job_ids = [b.batch_id for b in self.batch_manager.store.list_jobs(task_id=task_id)]
                        except Exception:
                            pass
                    self.supervisor.checkpoint(
                        task_id,
                        attempt_id=current_task.attempt_id,
                        lease_token=current_task.lease_token,
                        resume_payload={
                            "step_index": step_index,
                            "metrics": metrics.model_dump(mode="json"),
                            "last_answer": last_answer,
                            "batch_job_ids": post_batch_job_ids,
                        },
                        resume_cursor=f"after_step_{step_index}",
                        completed_steps=[str(s.get("summary", "")) for s in completed_steps],
                        pending_steps=["evaluate_continuation"],
                    )
                except Exception as cp_exc:
                    logger.warning("Checkpoint after step %d failed: %s", step_index, cp_exc)

            # 9. Evaluate Semantic Continuation Decision
            if self.evaluator is None:
                tb_task = self.taskboard.get_task(task_id) if self.taskboard is not None else None
                tb_status = getattr(tb_task, "status", None)
                if tb_status in {TaskStatus.BLOCKED, "blocked"}:
                    decision = ContinuationDecision(
                        task_id=task_id,
                        action=ContinuationAction.BLOCKED,
                        goal_satisfied=False,
                        safe_to_continue=False,
                        reason=getattr(tb_task, "result_summary", "") or "Task marked blocked on TaskBoard",
                        evidence_summary="Task is blocked",
                    )
                elif tb_status in {TaskStatus.DONE, TaskStatus.VERIFYING, "done", "verifying"} or step_result.get("goal_satisfied"):
                    decision = ContinuationDecision(
                        task_id=task_id,
                        action=ContinuationAction.FINALIZE,
                        goal_satisfied=True,
                        safe_to_continue=True,
                        reason="Task execution completed and verified",
                        evidence_summary=evidence_summary,
                        user_message=step_answer,
                    )
                else:
                    raise ContinuationDecisionError(
                        "Model decision failed: continuation_evaluator is required for autonomous execution."
                    )
            else:
                tb_progress = self.taskboard.get_progress(root_task_id=root_task_id) if self.taskboard is not None else None
                try:
                    decision = self.evaluator.evaluate(
                        task_id=task_id,
                        user_goal=user_goal,
                        step_output={
                            "answer": step_answer,
                            "tool_calls": tool_calls,
                            "tool_results_count": len(tool_results),
                            "changed_files": step_result.get("changed_files") or [],
                            "error": step_result.get("error") or "",
                        },
                        completed_steps=completed_steps,
                        metrics=metrics.model_dump(mode="json"),
                        soft_limit_reached=(metrics.model_turns >= effective_max_turns),
                        consecutive_no_progress_cycles=metrics.consecutive_no_progress_cycles,
                        independent_verification_status=task.verification_status.value,
                        taskboard_progress=tb_progress,
                    )
                except TypeError:
                    decision = self.evaluator.evaluate(
                        task_id=task_id,
                        user_goal=user_goal,
                        step_output={
                            "answer": step_answer,
                            "tool_calls": tool_calls,
                            "tool_results_count": len(tool_results),
                            "changed_files": step_result.get("changed_files") or [],
                            "error": step_result.get("error") or "",
                        },
                        completed_steps=completed_steps,
                        metrics=metrics.model_dump(mode="json"),
                        soft_limit_reached=(metrics.model_turns >= effective_max_turns),
                        consecutive_no_progress_cycles=metrics.consecutive_no_progress_cycles,
                        independent_verification_status=task.verification_status.value,
                    )

            overall_trace.append(
                {
                    "layer": "autonomous_controller",
                    "step_index": step_index,
                    "action": decision.action.value,
                    "goal_satisfied": decision.goal_satisfied,
                    "reason": decision.reason,
                    "evidence_summary": decision.evidence_summary,
                }
            )

            # Update metrics and state on TaskRecord
            def update_task_metrics(curr: TaskRecord) -> None:
                curr.autonomous_continuation_state = decision.action.value
                curr.autonomous_model_turns = metrics.model_turns
                curr.autonomous_tool_calls = metrics.tool_calls
                curr.autonomous_renewals_count = metrics.renewals_count
                curr.autonomous_no_progress_cycles = metrics.consecutive_no_progress_cycles
                curr.updated_at = self.supervisor.clock()

            try:
                self.supervisor.store.update_task(task_id, update_task_metrics)
            except Exception:
                pass

            # 10. Handle Model Decision
            if decision.action == ContinuationAction.FINALIZE:
                current_task = self.supervisor.get_task(task_id)
                final_answer = decision.user_message or last_answer or "Task completed successfully."

                # If task is RUNNING, submit result to escrow (which internally triggers completion verification)
                if current_task.state == ExecutionState.RUNNING and current_task.attempt_id and current_task.lease_token:
                    try:
                        verified_task = self.supervisor.submit_result(
                            task_id,
                            attempt_id=current_task.attempt_id,
                            lease_token=current_task.lease_token,
                            payload={
                                "answer": final_answer,
                                "goal_satisfied": True,
                                "metrics": metrics.model_dump(mode="json"),
                                "completed_steps": completed_steps,
                            },
                        )
                        current_task = verified_task
                    except Exception as submit_exc:
                        logger.warning("submit_result error: %s", submit_exc)
                        current_task = self.supervisor.get_task(task_id)

                verification_passed = True
                if step_result.get("verification_passed") is False or ctx.get("verification_passed") is False:
                    verification_passed = False

                if self.config.require_verification and current_task.completion_contract:
                    if current_task.state in {
                        ExecutionState.COMPLETED_PENDING_VERIFICATION,
                        ExecutionState.PENDING_BUDGET_DECISION,
                    }:
                        try:
                            verified_task = self.supervisor.verify_completion(task_id)
                            current_task = verified_task
                        except Exception as verif_exc:
                            logger.warning("Completion verification error for task %s: %s", task_id, verif_exc)

                    contract_passed = bool(
                        current_task.state == ExecutionState.COMPLETED
                        or getattr(current_task.verification_status, "value", "") == "passed"
                    )
                    if not contract_passed:
                        verification_passed = False
                    metrics.verification_results.append(
                        {
                            "status": getattr(current_task.verification_status, "value", "unknown"),
                            "artifacts": [a.model_dump(mode="json") for a in current_task.completion_artefacts],
                        }
                    )

                if not verification_passed:
                    # Cannot finalize when verification fails
                    logger.warning("Completion verification failed for task %s; rejecting FINALIZE", task_id)
                    if self.global_memory is not None:
                        try:
                            fail_mem = self.global_memory.record_failure(
                                attempted_action=f"Verification for goal: {user_goal[:80]}",
                                error="independent completion verification failed",
                                category="EXECUTION",
                                context=f"Completed {len(completed_steps)} steps; independent verification failed.",
                                execution_id=task_id,
                            )
                            if fail_mem and self.taskboard is not None:
                                self.taskboard.add_learning_reference(task_id, fail_mem.id)
                        except Exception as gm_exc:
                            logger.debug("GlobalMemory record failure for verification failed safely: %s", gm_exc)

                    if self.taskboard is not None:
                        try:
                            for child in self.taskboard.list_tasks(root_task_id=root_task_id):
                                if child.status in {TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.DONE} and not child.implementation_verified:
                                    self.taskboard.reopen(child.task_id, reason="autonomous replanning after verification failure")
                        except Exception:
                            pass
                    decision = ContinuationDecision(
                        task_id=task_id,
                        action=ContinuationAction.REPLAN,
                        goal_satisfied=False,
                        progress_detected=False,
                        evidence_summary="independent verification failed",
                        remaining_work="resolve verification failures and satisfy completion contracts",
                        safe_to_continue=True,
                        reason="independent completion verification failed",
                    )
                else:
                    current_task = self.supervisor.get_task(task_id)
                    if current_task.state == ExecutionState.COMPLETED_PENDING_VERIFICATION:
                        try:
                            self.supervisor.transition(task_id, ExecutionState.COMPLETED)
                        except Exception:
                            pass
                    if self.taskboard is not None:
                        try:
                            tb_task = self.taskboard.get_task(task_id)
                            if tb_task.status != TaskStatus.VERIFYING:
                                self.taskboard.mark_completed(task_id, evidence=final_answer, cross_off=True)
                            tb_task.verification_status = "passed"
                            tb_task.implementation_verified = True
                            self.taskboard.save()
                        except Exception:
                            pass

                    # Reinforce verified learning memories
                    if self.global_memory is not None:
                        try:
                            if self.taskboard is not None:
                                tb_task = self.taskboard.get_task(task_id)
                                for ref_id in getattr(tb_task, "learning_references", []):
                                    self.global_memory.verify_learning(
                                        ref_id,
                                        success=True,
                                        evidence=final_answer,
                                        execution_id=task_id,
                                    )
                        except Exception as gm_exc:
                            logger.debug("GlobalMemory verify_learning reinforcement failed safely: %s", gm_exc)
                    return _finish(AutonomousExecutionResult(
                        task_id=task_id,
                        status="completed",
                        goal_satisfied=True,
                        answer=final_answer,
                        metrics=metrics,
                        decision=decision,
                        verification_passed=True,
                        trace=overall_trace,
                        warnings=warnings,
                        completed_steps=completed_steps,
                    ))

            if decision.action == ContinuationAction.WAIT_FOR_USER:
                current_task = self.supervisor.get_task(task_id)
                if current_task.state == ExecutionState.RUNNING:
                    try:
                        self.supervisor.transition(task_id, ExecutionState.WAITING, reason="waiting_for_clarification")
                    except Exception:
                        pass
                if self.taskboard is not None:
                    try:
                        self.taskboard.update_status(task_id, TaskStatus.WAITING_FOR_TOOLS, reason=decision.reason or "waiting_for_clarification")
                    except Exception:
                        pass
                return _finish(AutonomousExecutionResult(
                    task_id=task_id,
                    status="waiting",
                    goal_satisfied=False,
                    answer=decision.user_message or "Execution is paused waiting for user input.",
                    metrics=metrics,
                    decision=decision,
                    trace=overall_trace,
                    warnings=warnings,
                    completed_steps=completed_steps,
                ))

            if decision.action == ContinuationAction.BLOCKED:
                current_task = self.supervisor.get_task(task_id)
                if current_task.state == ExecutionState.RUNNING:
                    try:
                        self.supervisor.transition(task_id, ExecutionState.FAILED, reason=decision.reason)
                    except Exception:
                        pass
                if self.taskboard is not None:
                    try:
                        self.taskboard.mark_blocked(task_id, reason=decision.reason)
                    except Exception:
                        pass
                return _finish(AutonomousExecutionResult(
                    task_id=task_id,
                    status="blocked",
                    goal_satisfied=False,
                    answer=decision.user_message or f"Task is blocked: {decision.reason}",
                    metrics=metrics,
                    decision=decision,
                    error=decision.reason,
                    trace=overall_trace,
                    warnings=warnings,
                    completed_steps=completed_steps,
                ))

            # For CONTINUE and REPLAN: proceed with next step
            ctx["continuation_remaining_work"] = decision.remaining_work
            ctx["continuation_decision"] = decision.model_dump(mode="json")
            if self.taskboard is not None:
                ctx["taskboard_progress"] = self.taskboard.get_progress(root_task_id=root_task_id)
            step_index += 1

    def _try_auto_renew(
        self,
        *,
        task: TaskRecord,
        metrics: TaskExecutionMetrics,
        effective_max_turns: int,
        effective_max_tools: int,
    ) -> tuple[bool, str]:
        """Validate and apply accounting-based auto-renewal."""
        if metrics.renewals_count >= self.config.auto_renew.max_auto_renewals:
            return False, f"maximum auto-renewals ({self.config.auto_renew.max_auto_renewals}) reached"

        if self.config.auto_renew.require_progress and metrics.consecutive_no_progress_cycles > 0:
            return False, "auto-renewal rejected: no progress detected in recent steps"

        if self.accounting_service is not None and self.config.auto_renew.respect_account_budget:
            # Independent validation via accounting service
            valid, reason = self.accounting_service.validate_autonomous_renewal(
                task_id=task.task_id,
                session_id=task.session_id,
                renew_turns=self.config.auto_renew.renew_model_turns,
                renew_tool_calls=self.config.auto_renew.renew_tool_calls,
                token_estimate=metrics.tokens_consumed,
                estimated_cost=None,
                task_budget_remaining=(
                    None if task.token_budget is None else max(0, task.token_budget - metrics.tokens_consumed)
                ),
                session_budget_remaining=None,
                respect_budget=self.config.auto_renew.respect_account_budget,
            )
            if not valid:
                return False, reason

        # Persist revision in supervisor
        revised_token_budget = (
            task.token_budget + (self.config.auto_renew.renew_model_turns * 2000)
            if task.token_budget is not None
            else None
        )
        try:
            self.supervisor.revise_budget(
                task.task_id,
                token_budget=revised_token_budget or (metrics.tokens_consumed + 50000),
                estimated_cost=task.estimated_cost,
                reason="autonomous_execution_auto_renew",
                evidence={
                    "renewals_count": metrics.renewals_count + 1,
                    "previous_turns_limit": effective_max_turns,
                    "new_turns_limit": effective_max_turns + self.config.auto_renew.renew_model_turns,
                },
            )
        except Exception:
            pass
        return True, "progress verified and budget capacity available"

    def _decide_or_revise_strategy(
        self,
        user_goal: str,
        current_strategy: str | None,
        step_index: int,
        ctx: dict[str, Any],
        task_id: str,
        sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[str, bool, str]:
        """Determine or dynamically revise the execution strategy without restarting the loop."""
        initial = current_strategy or ctx.get("strategy_hint") or ("multi_task" if ctx.get("subtasks") else None)
        revised = ctx.get("revised_strategy")
        if revised and normalize_strategy(revised) != normalize_strategy(current_strategy):
            new_strat = normalize_strategy(revised)
            reason = f"Strategy dynamically updated to {new_strat} during iteration {step_index + 1}"
            self._emit(
                "loop.strategy.selected",
                {
                    "task_id": task_id,
                    "strategy": new_strat,
                    "previous_strategy": current_strategy,
                    "changed": True,
                    "reason": reason,
                    "title": f"Strategy: {new_strat}",
                    "status": "success",
                },
                sink=sink,
            )
            return new_strat, True, reason

        if current_strategy is not None:
            return current_strategy, False, "strategy_retained"

        # Step 0: determine initial strategy
        if initial:
            strat = normalize_strategy(initial)
            reason = f"Initial strategy selected from hint: {strat}"
        elif self.router is not None:
            try:
                if hasattr(self.router, "route"):
                    route_dec = self.router.route(task_id=task_id, user_request=user_goal)
                    strat = normalize_strategy(getattr(route_dec, "route_name", None) or getattr(route_dec, "kind", None))
                    reason = f"Router classified request as {strat}"
                elif hasattr(self.router, "classify"):
                    strat = normalize_strategy(self.router.classify(user_goal))
                    reason = f"Router classified request as {strat}"
                else:
                    strat = ExecutionStrategy.SIMPLE.value
                    reason = "Default simple strategy"
            except Exception as router_exc:
                logger.debug("Router classification failed: %s; using simple strategy", router_exc)
                strat = ExecutionStrategy.SIMPLE.value
                reason = "Router unavailable, simple strategy selected"
        else:
            strat = ExecutionStrategy.SIMPLE.value
            reason = "Simple single-task strategy"

        self._emit(
            "loop.strategy.selected",
            {
                "task_id": task_id,
                "strategy": strat,
                "changed": False,
                "reason": reason,
                "title": f"Strategy: {strat}",
                "status": "success",
            },
            sink=sink,
        )
        return strat, False, reason

    def _decompose_work(
        self,
        root_task_id: str,
        user_goal: str,
        ctx: dict[str, Any],
        sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[str]:
        """Decompose compound or multi-task requests into TaskBoard subtasks."""
        if self.taskboard is None:
            return []

        # Check if child tasks already exist on taskboard for root_task_id (resume/restart)
        try:
            existing = self.taskboard.list_tasks(parent_task_id=root_task_id)
            if not existing:
                existing = [
                    t for t in self.taskboard.list_tasks(root_task_id=root_task_id)
                    if t.task_id != root_task_id
                ]
            if existing:
                return [t.task_id for t in existing]
        except Exception:
            pass

        created_ids: list[str] = []
        provided_subtasks = ctx.get("subtasks") or ctx.get("decomposition_plan") or []
        if isinstance(provided_subtasks, list) and provided_subtasks:
            local_to_task_id: dict[str, str] = {}
            for idx, item in enumerate(provided_subtasks, start=1):
                local_id = item.get("local_id") if isinstance(item, dict) else f"step_{idx}"
                local_id = str(local_id or f"step_{idx}")
                if isinstance(item, dict):
                    title = item.get("title") or f"Subtask {idx}"
                    req = item.get("request") or item.get("description") or title
                    crit = list(item.get("acceptance_criteria") or [title])
                    target_paths = list(item.get("target_paths") or [])
                    tool = item.get("tool")
                    tool_args = item.get("tool_args") or {}
                else:
                    title = f"Subtask {idx}: {str(item)[:50]}"
                    req = str(item)
                    crit = [title]
                    target_paths = []
                    tool = None
                    tool_args = {}
                child = self.taskboard.create_child_task(
                    root_task_id,
                    title=title,
                    user_request=req,
                    acceptance_criteria=crit,
                    plan=[req],
                    decomposition_local_id=local_id,
                )
                if target_paths:
                    child.implementation_targets = list(target_paths)
                if tool:
                    child.allowed_tools = [tool]
                    child.routing_evidence["tool"] = tool
                if tool_args:
                    child.routing_evidence["tool_args"] = tool_args
                created_ids.append(child.task_id)
                local_to_task_id[local_id] = child.task_id

            for child_id, item in zip(created_ids, provided_subtasks):
                raw_deps = list(item.get("depends_on") or []) if isinstance(item, dict) else []
                child = self.taskboard.get_task(child_id)
                child.depends_on = [local_to_task_id.get(d, d) for d in raw_deps]
        elif self.router is not None and hasattr(self.router, "decompose"):
            try:
                plan = self.router.decompose(user_prompt=user_goal, context=ctx)
                if hasattr(plan, "tasks"):
                    from mana_agent.multi_agent.runtime.multi_task_orchestrator import MultiTaskOrchestrator
                    mto = MultiTaskOrchestrator(llm=getattr(self.router, "llm", None), taskboard=self.taskboard)
                    mapping = mto.create_children(root_task_id=root_task_id, plan=plan)
                    created_ids = list(mapping.values())
            except Exception as dec_exc:
                logger.debug("Router decomposition failed: %s", dec_exc)

        if not created_ids:
            return []

        self.taskboard.save()
        ready = self._get_ready_tasks(root_task_id)
        self._emit(
            "loop.taskboard.updated",
            {
                "task_id": root_task_id,
                "summary": f"{len(created_ids)} subtasks registered",
                "ready_count": len(ready),
                "pending_count": len(created_ids),
                "title": "TaskBoard updated",
                "status": "success",
            },
            sink=sink,
        )
        return created_ids

    def _get_ready_tasks(self, root_task_id: str) -> list[Any]:
        """Find subtasks on TaskBoard whose dependencies are all completed."""
        if self.taskboard is None:
            return []
        try:
            tasks = self.taskboard.list_tasks(parent_task_id=root_task_id)
            if not tasks:
                tasks = [
                    t for t in self.taskboard.list_tasks(root_task_id=root_task_id)
                    if t.task_id != root_task_id
                ]
            done_statuses = {TaskStatus.DONE, TaskStatus.SKIPPED, "completed", "skipped"}
            evaluated_statuses = done_statuses | {TaskStatus.FAILED, TaskStatus.BLOCKED, "failed", "blocked"}
            evaluated_ids = {t.task_id for t in tasks if t.status in evaluated_statuses}
            ready = []
            for t in tasks:
                if t.status in done_statuses or t.status in {TaskStatus.CANCELLED, "cancelled"}:
                    continue
                deps = getattr(t, "depends_on", []) or []
                if all(dep in evaluated_ids for dep in deps):
                    ready.append(t)
            return ready
        except Exception as exc:
            logger.debug("Failed querying ready tasks: %s", exc)
            return []

    def _partition_non_conflicting_tasks(self, ready_tasks: list[Any]) -> tuple[list[Any], list[Any]]:
        """Partition ready tasks into concurrent batch vs. deferred to avoid shared-file conflicts."""
        batch: list[Any] = []
        deferred: list[Any] = []
        allocated_paths: set[str] = set()

        for t in ready_tasks:
            if len(batch) >= self.maximum_concurrency:
                deferred.append(t)
                continue
            target_paths: set[str] = set()
            for p in getattr(t, "implementation_targets", []) or []:
                target_paths.add(str(p).strip().replace("\\", "/").lstrip("./"))
            for p in getattr(t, "files_touched", []) or []:
                target_paths.add(str(p).strip().replace("\\", "/").lstrip("./"))
            for p in getattr(t, "files_to_inspect", []) or []:
                target_paths.add(str(p).strip().replace("\\", "/").lstrip("./"))
            if hasattr(t, "metadata") and isinstance(t.metadata, dict):
                for p in t.metadata.get("target_paths", []) or []:
                    target_paths.add(str(p).strip().replace("\\", "/").lstrip("./"))

            tool_meta = (
                getattr(t, "routing_evidence", {}).get("tool_args")
                or (t.metadata.get("tool_args") if hasattr(t, "metadata") and isinstance(t.metadata, dict) else {})
            )
            tool_name = str(
                getattr(t, "routing_evidence", {}).get("tool")
                or (t.metadata.get("tool") if hasattr(t, "metadata") and isinstance(t.metadata, dict) else "")
            )
            if tool_name and tool_meta:
                target_paths.update(_extract_target_paths_safe(tool_name, tool_meta))

            if target_paths and (target_paths & allocated_paths):
                # Shared-file conflict: defer to next iteration
                deferred.append(t)
            else:
                batch.append(t)
                allocated_paths.update(target_paths)

        return batch, deferred

    def _execute_single_subtask(
        self,
        root_task_id: str,
        task_item: Any,
        ctx: dict[str, Any],
        sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Execute a single subtask through its capability, coding engine, or tool."""
        child_id = getattr(task_item, "task_id", str(task_item))
        title = getattr(task_item, "title", f"Subtask {child_id}")
        t0 = time.perf_counter()

        if self.taskboard is not None:
            try:
                self.taskboard.update_status(child_id, TaskStatus.IN_PROGRESS, reason="Autonomous loop execution")
            except Exception:
                pass

        tool_name = (
            getattr(task_item, "routing_evidence", {}).get("tool")
            or (task_item.metadata.get("tool") if hasattr(task_item, "metadata") and isinstance(task_item.metadata, dict) else None)
            or (getattr(task_item, "allowed_tools", [None])[0] if getattr(task_item, "allowed_tools", None) else None)
            or getattr(task_item, "preferred_tool", None)
            or "subtask_executor"
        )
        tool_args = (
            getattr(task_item, "routing_evidence", {}).get("tool_args")
            or (task_item.metadata.get("tool_args") if hasattr(task_item, "metadata") and isinstance(task_item.metadata, dict) else None)
            or {"task_id": child_id, "title": title}
        )

        self._emit(
            "loop.tool.running",
            {
                "task_id": root_task_id,
                "subtask_id": child_id,
                "tool_name": tool_name,
                "args": tool_args,
                "title": f"↳ Tool {tool_name} — Running",
                "sub_operation": True,
                "status": "running",
            },
            sink=sink,
        )

        subtask_executors = ctx.get("task_executors") or {}
        local_id = getattr(task_item, "decomposition_local_id", "")
        executor_fn = (
            subtask_executors.get(child_id)
            or (subtask_executors.get(local_id) if local_id else None)
            or subtask_executors.get(tool_name)
        )
        result_output = ""
        error_msg = ""
        changed_files: list[str] = []

        try:
            if callable(executor_fn):
                exec_res = executor_fn(task_item, ctx)
                if isinstance(exec_res, dict):
                    result_output = str(exec_res.get("output") or exec_res.get("answer") or "completed")
                    changed_files = list(exec_res.get("changed_files") or [])
                    error_msg = str(exec_res.get("error") or "")
                else:
                    result_output = str(exec_res)
            elif tool_name in self.tools and callable(self.tools[tool_name]):
                tool_res = self.tools[tool_name](**tool_args)
                result_output = str(tool_res)
            elif self.coding_engine is not None and getattr(ctx.get("strategy"), "value", "") == ExecutionStrategy.CODING.value:
                coding_res = self.coding_engine.execute(task_item, ctx)
                result_output = str(coding_res)
            else:
                result_output = f"Completed subtask: {title}"
        except Exception as exc:
            error_msg = str(exc)
            logger.exception("Error executing subtask %s: %s", child_id, exc)

        duration_ms = int((time.perf_counter() - t0) * 1000)
        success = not bool(error_msg)

        if self.taskboard is not None:
            try:
                if success:
                    self.taskboard.mark_completed(child_id, result_summary=result_output)
                    if changed_files:
                        self.taskboard.add_files_touched(child_id, changed_files)
                else:
                    self.taskboard.mark_blocked(child_id, reason=error_msg)
                self.taskboard.save()
            except Exception:
                pass

        if not success or error_msg:
            if self.global_memory is not None:
                try:
                    fail_mem = self.global_memory.record_failure(
                        attempted_action=f"{tool_name}({json.dumps(tool_args, default=str)[:100]})",
                        error=error_msg,
                        category="TOOL",
                        context=f"Subtask: {title} ({child_id})",
                        execution_id=root_task_id,
                    )
                    if fail_mem and self.taskboard is not None:
                        self.taskboard.add_learning_reference(root_task_id, fail_mem.id)
                        self.taskboard.add_learning_reference(child_id, fail_mem.id)
                except Exception as gm_exc:
                    logger.debug("Failed recording subtask failure in global memory: %s", gm_exc)
        else:
            if self.global_memory is not None:
                try:
                    if self.taskboard is not None:
                        tb_task = self.taskboard.get_task(child_id)
                        for ref_id in getattr(tb_task, "learning_references", []):
                            self.global_memory.verify_learning(
                                ref_id,
                                success=True,
                                evidence=result_output[:200],
                                execution_id=root_task_id,
                            )
                except Exception as gm_exc:
                    logger.debug("Failed verifying subtask learning in global memory: %s", gm_exc)

        if result_output or error_msg:
            ctx.setdefault("evidence", []).append({
                "subtask_id": child_id,
                "title": title,
                "tool_name": tool_name,
                "output": result_output,
                "error": error_msg,
                "success": success,
            })
            if self.taskboard is not None:
                try:
                    summary_text = f"[{title}] {result_output or error_msg}"[:300]
                    self.taskboard.add_evidence(root_task_id, summary_text)
                except Exception:
                    pass

        self._emit(
            "loop.tool.completed",
            {
                "task_id": root_task_id,
                "subtask_id": child_id,
                "tool_name": tool_name,
                "duration_ms": duration_ms,
                "status": "success" if success else "failed",
                "title": f"↳ Tool {tool_name} — {'Completed' if success else 'Failed'}",
                "sub_operation": True,
                "error": error_msg,
            },
            sink=sink,
        )

        return {
            "subtask_id": child_id,
            "title": title,
            "tool_name": tool_name,
            "output": result_output,
            "error": error_msg,
            "success": success,
            "changed_files": changed_files,
            "duration_ms": duration_ms,
        }

    def _verify_goal(
        self,
        root_task_id: str,
        user_goal: str,
        ctx: dict[str, Any],
        step_index: int,
        executed_results: list[dict[str, Any]],
        sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[bool, str]:
        """Verify results and goal satisfaction independently."""
        if self.taskboard is not None:
            tasks = self.taskboard.list_tasks(parent_task_id=root_task_id)
            if not tasks:
                tasks = [
                    t for t in self.taskboard.list_tasks(root_task_id=root_task_id)
                    if t.task_id != root_task_id
                ]
            if tasks:
                pending = [
                    t for t in tasks
                    if t.status not in {TaskStatus.DONE, TaskStatus.SKIPPED, "completed", "skipped"}
                ]
                failed = [
                    t for t in tasks
                    if t.status in {TaskStatus.FAILED, TaskStatus.BLOCKED, "failed", "blocked"}
                ]
                if failed:
                    summary = f"Verification failed: {len(failed)} subtasks failed or blocked."
                    self._emit(
                        "loop.verification.completed",
                        {"task_id": root_task_id, "passed": False, "summary": summary, "title": "Verification failed", "status": "failed"},
                        sink=sink,
                    )
                    return False, summary
                if pending:
                    summary = f"Verification pending: {len(pending)} subtasks still in progress."
                    return False, summary

        verifier = ctx.get("verifier")
        if callable(verifier):
            try:
                verif_res = verifier(root_task_id, ctx)
                passed = bool(getattr(verif_res, "passed", verif_res))
                summary = str(getattr(verif_res, "summary", "Verification executed"))
                self._emit(
                    "loop.verification.completed",
                    {"task_id": root_task_id, "passed": passed, "summary": summary, "title": "Goal satisfied" if passed else "Verification failed", "status": "success" if passed else "failed"},
                    sink=sink,
                )
                return passed, summary
            except Exception as verif_exc:
                summary = f"Verification error: {verif_exc}"
                return False, summary

        summary = "Goal satisfied"
        self._emit(
            "loop.verification.completed",
            {"task_id": root_task_id, "passed": True, "summary": summary, "title": "Goal satisfied", "status": "success"},
            sink=sink,
        )
        return True, summary

    def _build_internal_loop_runner(
        self,
        *,
        task_id: str,
        user_goal: str,
        strategy_hint: str | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> Callable[[int, dict[str, Any]], dict[str, Any]]:
        """Construct the unified internal execution runner for the autonomous continuation loop."""
        def runner(step_index: int, ctx: dict[str, Any]) -> dict[str, Any]:
            iteration = step_index + 1
            current_strat = ctx.get("current_strategy")
            strat, changed, reason = self._decide_or_revise_strategy(
                user_goal=user_goal,
                current_strategy=current_strat,
                step_index=step_index,
                ctx=ctx,
                task_id=task_id,
                sink=event_sink,
            )
            ctx["current_strategy"] = strat

            if step_index == 0:
                phase_label = "Analyze request"
            elif ctx.get("goal_verified"):
                phase_label = "Finalize execution"
            else:
                phase_label = "Execute ready tasks"

            self._emit(
                "loop.iteration.started",
                {
                    "task_id": task_id,
                    "iteration": iteration,
                    "step_index": step_index,
                    "phase": phase_label,
                    "title": f"Iteration {iteration} — {phase_label}",
                    "status": "running",
                },
                sink=event_sink,
            )

            if (ctx.get("subtasks") or strat == ExecutionStrategy.MULTI_TASK.value) and not ctx.get("decomposed"):
                self._decompose_work(task_id, user_goal, ctx, sink=event_sink)
                ctx["decomposed"] = True

            ready_tasks = self._get_ready_tasks(task_id)
            executed_results: list[dict[str, Any]] = []
            changed_files: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            # If no ready tasks on TaskBoard, query memory to unblock/recover any failed or blocked subtasks
            if not ready_tasks and self.taskboard is not None:
                all_children = [t for t in self.taskboard.list_tasks(root_task_id=task_id) if t.task_id != task_id]
                blocked_or_failed = [t for t in all_children if t.status in {TaskStatus.FAILED, TaskStatus.BLOCKED, "failed", "blocked"}]
                if blocked_or_failed and not ctx.get("_retried_with_memory"):
                    ctx["_retried_with_memory"] = True
                    for bf_task in blocked_or_failed:
                        if self.global_memory is not None:
                            try:
                                past_learnings = self.global_memory.retrieve_relevant(
                                    action_or_query=f"{bf_task.title} {getattr(bf_task, 'result_summary', '')}",
                                    limit=3,
                                )
                                if past_learnings:
                                    rules_text = self.global_memory.format_retrieved_rules(past_learnings)
                                    ctx["learned_prevention_rules"] = rules_text
                                    for mem in past_learnings:
                                        self.taskboard.add_learning_reference(bf_task.task_id, mem.id)
                            except Exception as mem_ret_exc:
                                logger.debug("Memory retrieval during taskboard recovery failed: %s", mem_ret_exc)
                        try:
                            self.taskboard.reopen(bf_task.task_id, reason="Reopening with learned prevention rules from memory")
                        except Exception:
                            pass
                    self.taskboard.save()
                    ready_tasks = self._get_ready_tasks(task_id)

            if ready_tasks:
                concurrent_batch, deferred = self._partition_non_conflicting_tasks(ready_tasks)
                if len(concurrent_batch) == 1:
                    res = self._execute_single_subtask(task_id, concurrent_batch[0], ctx, sink=event_sink)
                    executed_results.append(res)
                    if res.get("changed_files"):
                        changed_files.extend(res["changed_files"])
                    tool_calls.append({"name": res.get("tool_name", "tool"), "args": {"subtask_id": res.get("subtask_id")}})
                    self._emit(
                        "loop.operations.completed",
                        {
                            "task_id": task_id,
                            "count": 1,
                            "type": "sequential",
                            "title": "Sequential operations completed",
                            "status": "success",
                        },
                        sink=event_sink,
                    )
                elif len(concurrent_batch) > 1:
                    with ThreadPoolExecutor(max_workers=min(len(concurrent_batch), self.maximum_concurrency)) as pool:
                        futures = {
                            pool.submit(self._execute_single_subtask, task_id, item, ctx, event_sink): item
                            for item in concurrent_batch
                        }
                        for fut in as_completed(futures):
                            res = fut.result()
                            executed_results.append(res)
                            if res.get("changed_files"):
                                changed_files.extend(res["changed_files"])
                            tool_calls.append({"name": res.get("tool_name", "tool"), "args": {"subtask_id": res.get("subtask_id")}})
                    self._emit(
                        "loop.operations.completed",
                        {
                            "task_id": task_id,
                            "count": len(executed_results),
                            "type": "parallel",
                            "title": "Parallel operations completed",
                            "status": "success",
                        },
                        sink=event_sink,
                    )
            else:
                direct_fn = ctx.get("direct_executor")
                if callable(direct_fn):
                    direct_res = direct_fn(user_goal, ctx)
                    if isinstance(direct_res, dict):
                        ans = str(direct_res.get("answer") or direct_res.get("output") or "")
                        cf = list(direct_res.get("changed_files") or [])
                        err = str(direct_res.get("error") or "")
                        ok = bool(direct_res.get("success", not bool(err)))
                        changed_files.extend(cf)
                        executed_results.append({"output": ans, "changed_files": cf, "success": ok, "error": err})
                        if not ok and err and self.global_memory is not None:
                            try:
                                fail_mem = self.global_memory.record_failure(
                                    attempted_action=f"direct_executor({user_goal[:80]})",
                                    error=err,
                                    category="EXECUTION",
                                    context=f"Direct execution for goal: {user_goal[:200]}",
                                    execution_id=task_id,
                                )
                                if fail_mem and self.taskboard is not None:
                                    self.taskboard.add_learning_reference(task_id, fail_mem.id)
                            except Exception as gm_exc:
                                logger.debug("Failed recording direct executor error: %s", gm_exc)
                    else:
                        ans = str(direct_res)
                        executed_results.append({"output": ans, "success": True})
                elif not executed_results:
                    ans = f"Processed request: {user_goal}"
                    executed_results.append({"output": ans, "success": True})

            # Check verification
            self._emit(
                "loop.iteration.started",
                {
                    "task_id": task_id,
                    "iteration": iteration + 1,
                    "step_index": step_index,
                    "phase": "Verify results",
                    "title": f"Iteration {iteration + 1} — Verify results",
                    "status": "running",
                },
                sink=event_sink,
            )
            goal_satisfied, verif_summary = self._verify_goal(
                root_task_id=task_id,
                user_goal=user_goal,
                ctx=ctx,
                step_index=step_index,
                executed_results=executed_results,
                sink=event_sink,
            )

            output_summary = "\n".join(
                f"- {r.get('title', 'Operation')}: {r.get('output', 'done')}" for r in executed_results
            ) if executed_results else verif_summary

            return {
                "answer": output_summary or verif_summary,
                "summary": output_summary or verif_summary,
                "tool_calls": tool_calls,
                "changed_files": changed_files,
                "verification_passed": goal_satisfied,
                "goal_satisfied": goal_satisfied,
                "tokens_used": 50,
                "cost": 0.0002,
            }

        return runner
