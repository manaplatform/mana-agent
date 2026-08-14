from __future__ import annotations

import pytest

from mana_agent.gateway.checkpoint_resume import (
    CHECKPOINT_RESUME_MAX_OUTPUT_TOKENS,
    CheckpointResumeDecider,
    CheckpointResumeError,
)
from mana_agent.context_cost.models import (
    BudgetSnapshot,
    ContextBreakdown,
    ContextBudget,
    ContextBudgetExceeded,
    GovernorDecision,
)


class StructuredDecisionModel:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.invocation_kwargs: list[dict[str, object]] = []

    def with_structured_output(self, _schema, *, method: str, strict: bool):
        assert method == "json_schema"
        assert strict is True
        return self

    def invoke(self, _messages, **kwargs):
        self.invocation_kwargs.append(kwargs)
        return self.payload


class ContextBlockedDecisionModel:
    def with_structured_output(self, _schema, *, method: str, strict: bool):
        assert method == "json_schema"
        assert strict is True
        return self

    def invoke(self, _messages, **_kwargs):
        snapshot = BudgetSnapshot(
            breakdown=ContextBreakdown(),
            budget=ContextBudget(context_window=1_000),
            used_tokens=1_510,
            remaining_tokens=0,
            utilization_ratio=1.51,
            cumulative_tokens=1_510,
            remaining_task_tokens=None,
            cumulative_cost=0.0,
            remaining_cost=None,
            estimated=True,
            status="blocked",
        )
        raise ContextBudgetExceeded(
            GovernorDecision(
                action="block",
                reason="context_limit_deficit:510",
                allowed=False,
                snapshot=snapshot,
            )
        )


def candidate() -> dict[str, object]:
    return {
        "task_id": "task_existing",
        "checkpoint_id": "checkpoint_existing",
        "normalized_intent": "continue repository refactor",
        "lane": "coding",
        "state": "running",
        "updated_at": "2026-08-01T10:00:00Z",
        "failure_reason": "worker stopped",
        "completed_steps": ["inspect"],
        "pending_steps": ["edit"],
        "resume_payload_fields": ["cursor"],
        "generated_files": [],
        "verification_status": "pending",
        "resume_eligible": True,
    }


def test_model_may_resume_exact_non_stale_checkpoint() -> None:
    model = StructuredDecisionModel(
            {
                "decision_id": "resume-decision-1",
                "action": "resume_checkpoint",
                "task_id": "task_existing",
                "checkpoint_id": "checkpoint_existing",
                "same_work": True,
                "fresh_data_required": False,
                "checkpoint_still_valid": True,
                "side_effects_safe_to_repeat": True,
                "safe_to_continue": True,
                "reason": "same repository work and checkpoint remains applicable",
            }
        )
    decider = CheckpointResumeDecider(model)

    decision = decider.decide(
        current_request="continue the repository refactor",
        route="coding",
        requires_live_data=False,
        candidates=[candidate()],
    )

    assert decision.action == "resume_checkpoint"
    assert decision.task_id == "task_existing"
    assert model.invocation_kwargs == [
        {"max_tokens": CHECKPOINT_RESUME_MAX_OUTPUT_TOKENS}
    ]
    # Headroom for providers whose thinking/reasoning tokens share max_tokens.
    assert CHECKPOINT_RESUME_MAX_OUTPUT_TOKENS >= 4_096


def test_model_may_replan_the_same_stopped_task() -> None:
    decider = CheckpointResumeDecider(
        StructuredDecisionModel(
            {
                "decision_id": "replan-decision-1",
                "action": "replan_task",
                "task_id": "task_existing",
                "checkpoint_id": "",
                "same_work": True,
                "fresh_data_required": False,
                "checkpoint_still_valid": False,
                "side_effects_safe_to_repeat": True,
                "safe_to_continue": True,
                "reason": "the same goal needs a revised plan",
            }
        )
    )

    decision = decider.decide(
        current_request="retry the repository refactor with a corrected plan",
        route="coding",
        requires_live_data=False,
        candidates=[{**candidate(), "checkpoint_id": ""}],
    )

    assert decision.action == "replan_task"
    assert decision.task_id == "task_existing"


def test_empty_candidate_set_still_requires_a_model_decision() -> None:
    decider = CheckpointResumeDecider(
        StructuredDecisionModel(
            {
                "decision_id": "fresh-decision-1",
                "action": "start_fresh",
                "task_id": "",
                "checkpoint_id": "",
                "same_work": False,
                "fresh_data_required": False,
                "checkpoint_still_valid": False,
                "side_effects_safe_to_repeat": False,
                "safe_to_continue": True,
                "reason": "no durable task applies",
            }
        )
    )

    assert decider.decide(
        current_request="a new repository task",
        route="coding",
        requires_live_data=False,
        candidates=[],
    ).decision_id == "fresh-decision-1"


def test_context_budget_block_is_reported_as_a_typed_checkpoint_decision_error() -> None:
    decider = CheckpointResumeDecider(ContextBlockedDecisionModel())

    with pytest.raises(CheckpointResumeError, match="context_limit_deficit:510") as raised:
        decider.decide(
            current_request="a new repository task",
            route="coding",
            requires_live_data=False,
            candidates=[],
        )

    assert raised.value.code == "context_budget_blocked"


class LengthLimitedDecisionModel:
    def with_structured_output(self, _schema, *, method: str, strict: bool):
        assert method == "json_schema"
        assert strict is True
        return self

    def invoke(self, _messages, **_kwargs):
        class LengthFinishReasonError(Exception):
            pass

        raise LengthFinishReasonError(
            "Could not parse response content as the length limit was reached - "
            "CompletionUsage(completion_tokens=512, prompt_tokens=1348, total_tokens=1860)"
        )


def test_output_length_limit_fails_safely_without_fallback_action() -> None:
    """Truncated structured output must stop; no resume/start-fresh fallback."""
    decider = CheckpointResumeDecider(LengthLimitedDecisionModel())

    with pytest.raises(CheckpointResumeError, match="length limit was reached") as raised:
        decider.decide(
            current_request="test mine latest gmail.",
            route="gmail",
            requires_live_data=True,
            candidates=[],
        )

    assert raised.value.code == "checkpoint_resume_invalid"
    assert "max_tokens=" in str(raised.value)
    assert "No task was resumed or started" in str(raised.value)


def test_live_data_route_cannot_reuse_checkpoint_even_if_model_requests_it() -> None:
    decider = CheckpointResumeDecider(
        StructuredDecisionModel(
            {
                "decision_id": "resume-decision-live",
                "action": "resume_checkpoint",
                "task_id": "task_existing",
                "checkpoint_id": "checkpoint_existing",
                "same_work": True,
                "fresh_data_required": False,
                "checkpoint_still_valid": True,
                "side_effects_safe_to_repeat": True,
                "safe_to_continue": True,
                "reason": "reuse old mailbox result",
            }
        )
    )

    with pytest.raises(CheckpointResumeError, match="safety fields are inconsistent"):
        decider.decide(
            current_request="check my mailbox now",
            route="gmail",
            requires_live_data=True,
            candidates=[candidate()],
        )


def test_model_can_require_fresh_execution_for_time_sensitive_work() -> None:
    decider = CheckpointResumeDecider(
        StructuredDecisionModel(
            {
                "decision_id": "resume-decision-fresh",
                "action": "start_fresh",
                "task_id": "",
                "checkpoint_id": "",
                "same_work": True,
                "fresh_data_required": True,
                "checkpoint_still_valid": False,
                "side_effects_safe_to_repeat": False,
                "safe_to_continue": True,
                "reason": "the requested price must be fetched again",
            }
        )
    )

    decision = decider.decide(
        current_request="get the current price",
        route="search",
        requires_live_data=True,
        candidates=[candidate()],
    )

    assert decision.action == "start_fresh"
    assert decision.fresh_data_required is True


def test_mcp_submission_route_starts_fresh_instead_of_resuming_upload_checkpoint() -> None:
    completed_upload = candidate()
    completed_upload["state"] = "completed"
    completed_upload["verification_status"] = "passed"
    decider = CheckpointResumeDecider(
        StructuredDecisionModel(
            {
                "decision_id": "fresh-kaggle-submission",
                "action": "start_fresh",
                "task_id": "",
                "checkpoint_id": "",
                "same_work": True,
                "fresh_data_required": True,
                "checkpoint_still_valid": False,
                "side_effects_safe_to_repeat": False,
                "safe_to_continue": True,
                "reason": "Kaggle submission state must be observed and executed again.",
            }
        )
    )

    decision = decider.decide(
        current_request="Use Kaggle MCP to retry the competition submission.",
        route="mcp",
        requires_live_data=True,
        candidates=[completed_upload],
    )

    assert decision.action == "start_fresh"
    assert decision.fresh_data_required is True


def test_model_may_retry_same_stable_task_without_checkpoint() -> None:
    retry_candidate = candidate()
    retry_candidate["checkpoint_id"] = ""
    retry_candidate["checkpoint_available"] = False
    decider = CheckpointResumeDecider(
        StructuredDecisionModel(
            {
                "decision_id": "retry-task-decision",
                "action": "retry_task",
                "task_id": "task_existing",
                "checkpoint_id": "",
                "same_work": True,
                "fresh_data_required": False,
                "checkpoint_still_valid": False,
                "side_effects_safe_to_repeat": True,
                "safe_to_continue": True,
                "reason": "same stable work can safely restart under its existing identity",
            }
        )
    )

    decision = decider.decide(
        current_request="retry the same repository refactor",
        route="coding",
        requires_live_data=False,
        candidates=[retry_candidate],
    )

    assert decision.action == "retry_task"
    assert decision.task_id == "task_existing"


def test_model_may_retry_same_task_even_when_a_checkpoint_is_listed() -> None:
    """A full same-task restart is valid; checkpoint_id on the decision must stay empty."""
    checkpointed = candidate()
    checkpointed["checkpoint_available"] = True
    decider = CheckpointResumeDecider(
        StructuredDecisionModel(
            {
                "decision_id": "retry-despite-checkpoint",
                "action": "retry_task",
                "task_id": "task_existing",
                "checkpoint_id": "",
                "same_work": True,
                "fresh_data_required": False,
                "checkpoint_still_valid": False,
                "side_effects_safe_to_repeat": True,
                "safe_to_continue": True,
                "reason": "restart the failed compound goal under its existing task identity",
            }
        )
    )

    decision = decider.decide(
        current_request="retry the five-step recovery task",
        route="multi_task",
        requires_live_data=False,
        candidates=[checkpointed],
    )

    assert decision.action == "retry_task"
    assert decision.task_id == "task_existing"
    assert decision.checkpoint_id == ""


def test_model_may_replan_same_task_even_when_a_checkpoint_is_listed() -> None:
    checkpointed = candidate()
    decider = CheckpointResumeDecider(
        StructuredDecisionModel(
            {
                "decision_id": "replan-despite-checkpoint",
                "action": "replan_task",
                "task_id": "task_existing",
                "checkpoint_id": "",
                "same_work": True,
                "fresh_data_required": False,
                "checkpoint_still_valid": False,
                "side_effects_safe_to_repeat": True,
                "safe_to_continue": True,
                "reason": "revise the plan then restart under the same identity",
            }
        )
    )

    decision = decider.decide(
        current_request="retry the recovery steps with a corrected plan",
        route="multi_task",
        requires_live_data=False,
        candidates=[checkpointed],
    )

    assert decision.action == "replan_task"
    assert decision.checkpoint_id == ""


def test_retry_or_replan_must_not_carry_a_checkpoint_id() -> None:
    decider = CheckpointResumeDecider(
        StructuredDecisionModel(
            {
                "decision_id": "retry-with-checkpoint-id",
                "action": "retry_task",
                "task_id": "task_existing",
                "checkpoint_id": "checkpoint_existing",
                "same_work": True,
                "fresh_data_required": False,
                "checkpoint_still_valid": False,
                "side_effects_safe_to_repeat": True,
                "safe_to_continue": True,
                "reason": "retry but also copied the checkpoint",
            }
        )
    )

    with pytest.raises(CheckpointResumeError, match="leave checkpoint_id empty"):
        decider.decide(
            current_request="retry the same repository refactor",
            route="coding",
            requires_live_data=False,
            candidates=[candidate()],
        )


def test_live_data_route_cannot_retry_old_task_without_checkpoint() -> None:
    retry_candidate = candidate()
    retry_candidate["checkpoint_id"] = ""
    decider = CheckpointResumeDecider(
        StructuredDecisionModel(
            {
                "decision_id": "retry-task-live",
                "action": "retry_task",
                "task_id": "task_existing",
                "checkpoint_id": "",
                "same_work": True,
                "fresh_data_required": False,
                "checkpoint_still_valid": False,
                "side_effects_safe_to_repeat": True,
                "safe_to_continue": True,
                "reason": "reuse old task",
            }
        )
    )

    with pytest.raises(CheckpointResumeError, match="safety fields are inconsistent"):
        decider.decide(
            current_request="check current email now",
            route="gmail",
            requires_live_data=True,
            candidates=[retry_candidate],
        )


def test_model_cannot_start_new_task_for_same_stable_work() -> None:
    decider = CheckpointResumeDecider(
        StructuredDecisionModel(
            {
                "decision_id": "improper-fresh-task",
                "action": "start_fresh",
                "task_id": "",
                "checkpoint_id": "",
                "same_work": True,
                "fresh_data_required": False,
                "checkpoint_still_valid": False,
                "side_effects_safe_to_repeat": True,
                "safe_to_continue": True,
                "reason": "start another task",
            }
        )
    )

    with pytest.raises(CheckpointResumeError, match="must reuse its stopped task identity"):
        decider.decide(
            current_request="retry the same repository refactor",
            route="coding",
            requires_live_data=False,
            candidates=[candidate()],
        )


def test_model_may_start_fresh_for_same_work_when_no_recoverable_candidates() -> None:
    """Deadline-dead prior work leaves an empty candidate set; a new task is required."""
    decider = CheckpointResumeDecider(
        StructuredDecisionModel(
            {
                "decision_id": "fresh-after-deadline",
                "action": "start_fresh",
                "task_id": "",
                "checkpoint_id": "",
                "same_work": True,
                "fresh_data_required": False,
                "checkpoint_still_valid": False,
                "side_effects_safe_to_repeat": True,
                "safe_to_continue": True,
                "reason": "prior task is wall-clock dead; create a new task identity",
            }
        )
    )

    decision = decider.decide(
        current_request="retry putting the logo in readme.md",
        route="coding",
        requires_live_data=False,
        candidates=[],
    )

    assert decision.action == "start_fresh"
    assert decision.same_work is True


def test_terminal_failed_task_checkpoint_cannot_be_implicitly_resumed() -> None:
    failed_candidate = candidate()
    failed_candidate["state"] = "failed"
    failed_candidate["resume_eligible"] = False
    failed_candidate["is_terminal"] = True

    decider = CheckpointResumeDecider(
        StructuredDecisionModel(
            {
                "decision_id": "resume-terminal-fail",
                "action": "resume_checkpoint",
                "task_id": "task_existing",
                "checkpoint_id": "checkpoint_existing",
                "same_work": True,
                "fresh_data_required": False,
                "checkpoint_still_valid": True,
                "side_effects_safe_to_repeat": True,
                "safe_to_continue": True,
                "reason": "attempt to resume a terminal failed task",
            }
        )
    )

    with pytest.raises(CheckpointResumeError, match="selected checkpoint is not an offered durable candidate"):
        decider.decide(
            current_request="continue the failed task",
            route="coding",
            requires_live_data=False,
            candidates=[failed_candidate],
        )


def test_terminal_failed_task_can_be_retried_with_new_attempt() -> None:
    failed_candidate = candidate()
    failed_candidate["state"] = "failed"
    failed_candidate["resume_eligible"] = False
    failed_candidate["is_terminal"] = True

    decider = CheckpointResumeDecider(
        StructuredDecisionModel(
            {
                "decision_id": "retry-terminal-fail",
                "action": "retry_task",
                "task_id": "task_existing",
                "checkpoint_id": "",
                "same_work": True,
                "fresh_data_required": False,
                "checkpoint_still_valid": False,
                "side_effects_safe_to_repeat": True,
                "safe_to_continue": True,
                "reason": "retry failed task under existing identity with new attempt",
            }
        )
    )

    decision = decider.decide(
        current_request="retry the failed repository task",
        route="coding",
        requires_live_data=False,
        candidates=[failed_candidate],
    )

    assert decision.action == "retry_task"
    assert decision.task_id == "task_existing"
    assert decision.checkpoint_id == ""

