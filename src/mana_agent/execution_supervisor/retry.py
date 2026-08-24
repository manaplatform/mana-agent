"""Retry validation and deterministic backoff."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Iterable

from mana_agent.execution_supervisor.config import ExecutionSupervisorConfig
from mana_agent.execution_supervisor.errors import RetrySafetyError
from mana_agent.execution_supervisor.models import (
    ActionEffectScope,
    ActionRequestState,
    ActionRecord,
    RecoveryAction,
    RecoveryDecision,
    RetryCategory,
    SideEffectClassification,
    TaskRecord,
)


SAFE_AUTOMATIC_RETRY = frozenset(
    {
        SideEffectClassification.READ_ONLY,
        SideEffectClassification.IDEMPOTENT,
        SideEffectClassification.DEDUPLICATED,
        SideEffectClassification.COMPENSATABLE,
    }
)


class RetryPolicy:
    def __init__(self, config: ExecutionSupervisorConfig) -> None:
        self.config = config

    def validate(
        self,
        task: TaskRecord,
        decision: RecoveryDecision,
        *,
        actions: Iterable[ActionRecord] = (),
        now: datetime | None = None,
    ) -> None:
        if decision.task_id != task.task_id:
            raise RetrySafetyError("recovery decision task does not match the execution task")
        if not decision.safe_to_continue:
            raise RetrySafetyError("recovery decision did not authorize continued execution")
        if task.wall_clock_deadline_exceeded(now):
            raise RetrySafetyError(
                "task wall-clock deadline exceeded; create a new task instead of retrying"
            )
        if decision.action not in {
            RecoveryAction.RETRY,
            RecoveryAction.RESUME_CHECKPOINT,
            RecoveryAction.REASSIGN,
            RecoveryAction.REPLAN,
        }:
            raise RetrySafetyError(f"recovery action cannot execute work: {decision.action.value}")
        if (decision.action == RecoveryAction.REPLAN) != (
            decision.retry_category == RetryCategory.REPLAN
        ):
            raise RetrySafetyError(
                "replan actions must use the replan budget, and the replan budget cannot fund "
                "a different recovery action"
            )
        if decision.action == RecoveryAction.REASSIGN and not any(
            (decision.selected_agent, decision.selected_worker, decision.selected_model)
        ):
            raise RetrySafetyError(
                "reassignment requires an explicitly selected agent, worker, or model"
            )
        classification = task.side_effect_classification
        ambiguous_actions = [
            action.action_id
            for action in actions
            if action.request_state in {
                ActionRequestState.STARTED,
                ActionRequestState.OUTCOME_UNKNOWN,
            }
            and action.classification not in {
                SideEffectClassification.READ_ONLY,
                SideEffectClassification.IDEMPOTENT,
            }
            and getattr(action, "effect_scope", ActionEffectScope.LOCAL_REPOSITORY) == ActionEffectScope.EXTERNAL_CONSEQUENTIAL
            and not action.external_receipt
        ]
        if ambiguous_actions:
            raise RetrySafetyError(
                "consequential action outcome is ambiguous; reconcile durable action receipts "
                "before retrying: " + ", ".join(ambiguous_actions)
            )
        exact_checkpoint_resume = (
            decision.action == RecoveryAction.RESUME_CHECKPOINT
            and bool(decision.resume_checkpoint_id or task.checkpoint_id)
            and not task.irreversible_side_effect_started
        )
        if classification in {
            SideEffectClassification.NON_IDEMPOTENT,
            SideEffectClassification.UNKNOWN,
        }:
            model_authorized_same_task_retry = (
                classification == SideEffectClassification.UNKNOWN
                and decision.action == RecoveryAction.RETRY
                and decision.same_task_retry_authorized
                and not task.irreversible_side_effect_started
            )
            lease_loss_no_side_effects = (
                decision.retry_category == RetryCategory.LEASE_LOSS
                and not task.irreversible_side_effect_started
                and not ambiguous_actions
            )
            if not exact_checkpoint_resume and not model_authorized_same_task_retry and not lease_loss_no_side_effects:
                raise RetrySafetyError(
                    f"{classification.value} task may already have produced an external side effect; "
                    "no retry was scheduled"
                )
        if classification in {
            SideEffectClassification.DEDUPLICATED,
            SideEffectClassification.IDEMPOTENT,
            SideEffectClassification.CONDITIONALLY_IDEMPOTENT,
        }:
            if not task.idempotency_key and decision.retry_category != RetryCategory.LEASE_LOSS and not exact_checkpoint_resume:
                raise RetrySafetyError(
                    f"{classification.value} retry requires a stable idempotency key"
                )
        if classification == SideEffectClassification.COMPENSATABLE and not task.compensation_strategy:
            raise RetrySafetyError(
                "compensatable retry requires a tested compensation strategy"
            )
        category = decision.retry_category
        if task.retry_budget.remaining(category, task.retry_usage) <= 0:
            raise RetrySafetyError(f"{category.value} retry budget is exhausted")
        if decision.action == RecoveryAction.RESUME_CHECKPOINT:
            checkpoint_id = decision.resume_checkpoint_id or task.checkpoint_id
            if not checkpoint_id:
                raise RetrySafetyError("checkpoint resume requires a valid checkpoint ID")

    def backoff_seconds(self, task: TaskRecord, category: RetryCategory) -> float:
        retry_number = int(task.retry_usage.get(category.value, 0)) + 1
        exponential = min(
            self.config.max_backoff_seconds,
            self.config.base_backoff_seconds * (2 ** max(0, retry_number - 1)),
        )
        seed = f"{task.task_id}:{category.value}:{retry_number}".encode("utf-8")
        jitter_ratio = int.from_bytes(hashlib.sha256(seed).digest()[:2], "big") / 65535
        return min(self.config.max_backoff_seconds, exponential * (0.75 + 0.5 * jitter_ratio))

    def automatic_recovery_decision(
        self,
        task: TaskRecord,
        *,
        category: RetryCategory,
        reason: str,
        actions: Iterable[ActionRecord] = (),
    ) -> RecoveryDecision | None:
        if task.wall_clock_deadline_exceeded():
            return None
        if task.retry_budget.remaining(category, task.retry_usage) <= 0:
            return None

        action_list = list(actions)
        ambiguous = [
            action
            for action in action_list
            if action.request_state in {ActionRequestState.STARTED, ActionRequestState.OUTCOME_UNKNOWN}
            and action.classification not in {
                SideEffectClassification.READ_ONLY,
                SideEffectClassification.IDEMPOTENT,
            }
            and getattr(action, "effect_scope", ActionEffectScope.LOCAL_REPOSITORY) == ActionEffectScope.EXTERNAL_CONSEQUENTIAL
            and not action.external_receipt
        ]
        if ambiguous:
            return None
        if task.irreversible_side_effect_started and not task.checkpoint_id:
            return None

        if category == RetryCategory.LEASE_LOSS or not action_list or not task.irreversible_side_effect_started:
            if task.side_effect_classification == SideEffectClassification.COMPENSATABLE and not task.compensation_strategy:
                return None
            checkpoint = task.checkpoint_id
            return RecoveryDecision(
                decision_id=f"policy:{task.task_id}:{task.state_version}:{category.value}",
                task_id=task.task_id,
                action=RecoveryAction.RESUME_CHECKPOINT if checkpoint else RecoveryAction.RETRY,
                retry_category=category,
                reason=reason,
                resume_checkpoint_id=checkpoint,
                safe_to_continue=True,
            )

        if task.side_effect_classification not in SAFE_AUTOMATIC_RETRY:
            return None
        if (
            task.side_effect_classification
            in {SideEffectClassification.IDEMPOTENT, SideEffectClassification.DEDUPLICATED}
            and not task.idempotency_key
        ):
            return None
        if (
            task.side_effect_classification == SideEffectClassification.COMPENSATABLE
            and not task.compensation_strategy
        ):
            return None

        checkpoint = task.checkpoint_id
        return RecoveryDecision(
            decision_id=f"policy:{task.task_id}:{task.state_version}:{category.value}",
            task_id=task.task_id,
            action=RecoveryAction.RESUME_CHECKPOINT if checkpoint else RecoveryAction.RETRY,
            retry_category=category,
            reason=reason,
            resume_checkpoint_id=checkpoint,
            safe_to_continue=True,
        )
