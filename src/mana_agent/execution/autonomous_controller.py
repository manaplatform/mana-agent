"""Universal Model-Directed Autonomous Execution Controller.

Integrates with Agent Orchestrator, Execution Supervisor, accounting infrastructure,
task state machine, context compactor, and checkpoint system across every Mana-Agent capability.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
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

if TYPE_CHECKING:
    from mana_agent.gateway.context_compactor import ContextCompactor

logger = logging.getLogger(__name__)


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
        accounting_service: ModelTokenAccountingService | None = None,
        context_governor: ContextCostGovernor | None = None,
        compactor: ContextCompactor | None = None,
        evaluator: ContinuationEvaluator | None = None,
        continuation_evaluator: ContinuationEvaluator | None = None,
    ) -> None:
        self.config = config or AutonomousExecutionConfig()
        self.supervisor = supervisor
        self.accounting_service = accounting_service
        self.context_governor = context_governor
        self.compactor = compactor
        self.evaluator = evaluator or continuation_evaluator

    def run(
        self,
        *,
        task_id: str,
        user_goal: str,
        step_executor: Callable[..., Any] | None = None,
        step_runner: Callable[..., Any] | None = None,
        context: dict[str, Any] | None = None,
        initial_metrics: TaskExecutionMetrics | None = None,
    ) -> AutonomousExecutionResult:
        """Convenience method running the universal continuation loop."""
        return self.execute_goal(
            task_id=task_id,
            user_goal=user_goal,
            step_runner=step_runner,
            step_executor=step_executor,
            context=context,
            initial_metrics=initial_metrics,
        )

    def execute_goal(
        self,
        *,
        task_id: str,
        user_goal: str,
        step_runner: Callable[..., Any] | None = None,
        step_executor: Callable[..., Any] | None = None,
        context: dict[str, Any] | None = None,
        initial_metrics: TaskExecutionMetrics | None = None,
    ) -> AutonomousExecutionResult:
        """Run the universal model-directed autonomous continuation loop."""
        runner = step_runner or step_executor
        if runner is None:
            raise ValueError("step_runner or step_executor is required")

        ctx = dict(context or {})

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

        metrics = initial_metrics or TaskExecutionMetrics(
            task_id=task_id,
            root_task_id=root_task_id,
        )

        effective_max_turns = self.config.max_model_turns
        effective_max_tools = self.config.max_tool_calls

        step_index = 0
        completed_steps: list[dict[str, Any]] = []
        overall_trace: list[dict[str, Any]] = []
        last_evidence_fingerprint = ""
        last_answer = ""
        warnings: list[str] = []

        while True:
            # 1. Hard Gate: Cancellation Check
            if self.supervisor.cancellation_requested(task_id):
                self.supervisor.cancel(task_id, reason="user_cancelled")
                return AutonomousExecutionResult(
                    task_id=task_id,
                    status="cancelled",
                    goal_satisfied=False,
                    answer="Task was cancelled by user or operator.",
                    metrics=metrics,
                    error="cancelled",
                    trace=overall_trace,
                    warnings=warnings,
                    completed_steps=completed_steps,
                )

            # 2. Checkpoint before execution step
            current_task = self.supervisor.get_task(task_id)
            if current_task.state == ExecutionState.RUNNING and current_task.attempt_id and current_task.lease_token:
                try:
                    self.supervisor.checkpoint(
                        task_id,
                        attempt_id=current_task.attempt_id,
                        lease_token=current_task.lease_token,
                        resume_payload={"step_index": step_index, "metrics": metrics.model_dump(mode="json")},
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
                return AutonomousExecutionResult(
                    task_id=task_id,
                    status="failed",
                    goal_satisfied=False,
                    answer="",
                    metrics=metrics,
                    error=f"Step execution error: {exc}",
                    trace=overall_trace,
                    warnings=warnings,
                    completed_steps=completed_steps,
                )

            step_duration_ms = (time.perf_counter() - step_t0) * 1000.0

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

            # 6. Check for Pending Human Approval Gate
            if step_result.get("is_approval_pending"):
                current_task = self.supervisor.get_task(task_id)
                if current_task.state == ExecutionState.RUNNING:
                    try:
                        self.supervisor.transition(task_id, ExecutionState.WAITING, reason="waiting_for_approval")
                    except Exception:
                        pass
                return AutonomousExecutionResult(
                    task_id=task_id,
                    status="waiting",
                    goal_satisfied=False,
                    answer=step_answer or "Action requires human approval before continuing.",
                    metrics=metrics,
                    error="waiting_for_approval",
                    trace=overall_trace,
                    warnings=warnings,
                    completed_steps=completed_steps,
                )

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
                return AutonomousExecutionResult(
                    task_id=task_id,
                    status="blocked",
                    goal_satisfied=False,
                    answer=blocker_msg,
                    metrics=metrics,
                    error="no_progress_stopped",
                    trace=overall_trace,
                    warnings=warnings,
                    completed_steps=completed_steps,
                )

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
                            return AutonomousExecutionResult(
                                task_id=task_id,
                                status="waiting",
                                goal_satisfied=False,
                                answer=f"Task paused: Budget extension requires user approval ({renewal_reason}).",
                                metrics=metrics,
                                error="waiting_for_approval",
                                trace=overall_trace,
                                warnings=warnings,
                                completed_steps=completed_steps,
                            )
                        # Mark budget exhausted
                        current_task = self.supervisor.get_task(task_id)
                        if current_task.state == ExecutionState.RUNNING:
                            try:
                                self.supervisor.transition(task_id, ExecutionState.BUDGET_EXHAUSTED)
                            except Exception:
                                pass
                        return AutonomousExecutionResult(
                            task_id=task_id,
                            status="budget_exhausted",
                            goal_satisfied=False,
                            answer=f"Autonomous execution budget exhausted: {renewal_reason}.",
                            metrics=metrics,
                            error="budget_exhausted",
                            trace=overall_trace,
                            warnings=warnings,
                            completed_steps=completed_steps,
                        )
                else:
                    current_task = self.supervisor.get_task(task_id)
                    if current_task.state == ExecutionState.RUNNING:
                        try:
                            self.supervisor.transition(task_id, ExecutionState.BUDGET_EXHAUSTED)
                        except Exception:
                            pass
                    return AutonomousExecutionResult(
                        task_id=task_id,
                        status="budget_exhausted",
                        goal_satisfied=False,
                        answer="Autonomous execution soft limits reached and auto_renew is disabled.",
                        metrics=metrics,
                        error="budget_exhausted",
                        trace=overall_trace,
                        warnings=warnings,
                        completed_steps=completed_steps,
                    )

            # 8. Checkpoint state after step
            current_task = self.supervisor.get_task(task_id)
            if current_task.state == ExecutionState.RUNNING and current_task.attempt_id and current_task.lease_token:
                try:
                    self.supervisor.checkpoint(
                        task_id,
                        attempt_id=current_task.attempt_id,
                        lease_token=current_task.lease_token,
                        resume_payload={
                            "step_index": step_index,
                            "metrics": metrics.model_dump(mode="json"),
                            "last_answer": last_answer,
                        },
                        resume_cursor=f"after_step_{step_index}",
                        completed_steps=[str(s.get("summary", "")) for s in completed_steps],
                        pending_steps=["evaluate_continuation"],
                    )
                except Exception as cp_exc:
                    logger.warning("Checkpoint after step %d failed: %s", step_index, cp_exc)

            # 9. Evaluate Semantic Continuation Decision
            if self.evaluator is None:
                raise ContinuationDecisionError(
                    "Model decision failed: continuation_evaluator is required for autonomous execution."
                )

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

                    verification_passed = bool(
                        current_task.state == ExecutionState.COMPLETED
                        or getattr(current_task.verification_status, "value", "") == "passed"
                    )
                    metrics.verification_results.append(
                        {
                            "status": getattr(current_task.verification_status, "value", "unknown"),
                            "artifacts": [a.model_dump(mode="json") for a in current_task.completion_artefacts],
                        }
                    )

                if not verification_passed:
                    # Cannot finalize when verification fails
                    logger.warning("Completion verification failed for task %s; rejecting FINALIZE", task_id)
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
                    return AutonomousExecutionResult(
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
                    )

            if decision.action == ContinuationAction.WAIT_FOR_USER:
                current_task = self.supervisor.get_task(task_id)
                if current_task.state == ExecutionState.RUNNING:
                    try:
                        self.supervisor.transition(task_id, ExecutionState.WAITING, reason="waiting_for_clarification")
                    except Exception:
                        pass
                return AutonomousExecutionResult(
                    task_id=task_id,
                    status="waiting",
                    goal_satisfied=False,
                    answer=decision.user_message or "Execution is paused waiting for user input.",
                    metrics=metrics,
                    decision=decision,
                    trace=overall_trace,
                    warnings=warnings,
                    completed_steps=completed_steps,
                )

            if decision.action == ContinuationAction.BLOCKED:
                current_task = self.supervisor.get_task(task_id)
                if current_task.state == ExecutionState.RUNNING:
                    try:
                        self.supervisor.transition(task_id, ExecutionState.FAILED, reason=decision.reason)
                    except Exception:
                        pass
                return AutonomousExecutionResult(
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
                )

            # For CONTINUE and REPLAN: proceed with next step
            ctx["continuation_remaining_work"] = decision.remaining_work
            ctx["continuation_decision"] = decision.model_dump(mode="json")
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
        except Exception as exc:
            logger.warning("Supervisor budget revision failed during auto_renew: %s", exc)

        return True, "progress verified and budget capacity available"
