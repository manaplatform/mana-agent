import json
import pytest

from mana_agent.gateway.followup_classifier import (
    FollowupClassificationError,
    FollowupClassifier,
)
from mana_agent.tools.context_retrieval import (
    TurnRetrievalLedger,
    execute_conversation_context_read,
)


class _StructuredModel:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def with_structured_output(self, _schema, *, method: str, strict: bool):
        assert method == "json_schema"
        assert strict is True
        return self

    def invoke(self, _messages):
        return self.payload


class _MultiTurnStructuredModel:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.invocations: list[list[object]] = []

    def with_structured_output(self, _schema, *, method: str, strict: bool):
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return {"category": "conversation_only", "safe_to_continue": True, "reason": "default"}


def test_expansion_must_select_an_offered_task() -> None:
    classifier = FollowupClassifier(
        _StructuredModel(
            {
                "category": "task_expansion",
                "related_task_id": "task_1",
                "safe_to_continue": True,
                "reason": "the user extends the completed task",
            }
        )
    )
    decision = classifier.decide(
        message="add dashboard support",
        recent_history=[],
        candidates=[{"task_id": "task_1", "normalized_intent": "fix chat"}],
    )
    assert decision.category == "task_expansion"
    assert decision.related_task_id == "task_1"
    assert decision.decision_id.startswith("followup:")


def test_followup_cannot_attach_to_an_unoffered_task() -> None:
    classifier = FollowupClassifier(
        _StructuredModel(
            {
                "category": "followup_task",
                "related_task_id": "task_missing",
                "safe_to_continue": True,
                "reason": "incorrect task",
            }
        )
    )
    with pytest.raises(FollowupClassificationError, match="not offered"):
        classifier.decide(
            message="fix tests too",
            recent_history=[],
            candidates=[{"task_id": "task_1"}],
        )


def test_non_task_category_cannot_select_a_related_task() -> None:
    classifier = FollowupClassifier(
        _StructuredModel(
            {
                "category": "conversation_only",
                "related_task_id": "task_1",
                "safe_to_continue": True,
                "reason": "the user is only acknowledging prior output",
            }
        )
    )

    with pytest.raises(FollowupClassificationError, match="non-task category"):
        classifier.decide(
            message="thanks",
            recent_history=[],
            candidates=[{"task_id": "task_1"}],
        )


def test_completed_task_cannot_be_classified_as_a_resume() -> None:
    classifier = FollowupClassifier(
        _StructuredModel(
            {
                "category": "resume_request",
                "related_task_id": "task_completed",
                "safe_to_continue": True,
                "reason": "continue the earlier work",
            }
        )
    )

    with pytest.raises(FollowupClassificationError, match="completed task cannot be resumed"):
        classifier.decide(
            message="submit the uploaded Kaggle file now",
            recent_history=[],
            candidates=[{"task_id": "task_completed", "state": "completed"}],
        )


def test_independent_category_overrides_unsafe_to_safe() -> None:
    """new_task with no related task is unambiguous by construction.

    Even if the model sets safe_to_continue=false, the structural invariant
    in decide() overrides it to true because the classification is independent.
    """
    classifier = FollowupClassifier(
        _StructuredModel(
            {
                "category": "new_task",
                "related_task_id": "",
                "safe_to_continue": False,
                "reason": "the request is a bare email address with no context",
            }
        )
    )

    decision = classifier.decide(
        message="user@example.com",
        recent_history=[],
        candidates=[{"task_id": "task_1"}],
    )
    assert decision.category == "new_task"
    assert decision.related_task_id == ""
    assert decision.safe_to_continue is True


def test_conversation_only_overrides_unsafe_to_safe() -> None:
    """conversation_only with no related task is also structurally unambiguous."""
    classifier = FollowupClassifier(
        _StructuredModel(
            {
                "category": "conversation_only",
                "related_task_id": "",
                "safe_to_continue": False,
                "reason": "terse input with empty history",
            }
        )
    )

    decision = classifier.decide(
        message="hello",
        recent_history=[],
        candidates=[{"task_id": "task_1"}],
    )
    assert decision.category == "conversation_only"
    assert decision.safe_to_continue is True


def test_task_bound_category_with_unsafe_still_raises() -> None:
    """Categories that require a related task should still respect safe_to_continue=false."""
    classifier = FollowupClassifier(
        _StructuredModel(
            {
                "category": "followup_task",
                "related_task_id": "task_1",
                "safe_to_continue": False,
                "reason": "two conflicting task identities",
            }
        )
    )

    with pytest.raises(
        FollowupClassificationError,
        match="two conflicting task identities",
    ):
        classifier.decide(
            message="continue that task",
            recent_history=[],
            candidates=[{"task_id": "task_1"}, {"task_id": "task_2"}],
        )


def test_why_retrieves_previous_turn_before_followup_classification() -> None:
    """Follow-up 'why?' performs retrieval before final classification."""
    model = _MultiTurnStructuredModel(
        [
            {
                "action": "retrieve_context",
                "retrieval": {"query": "OAuth2 PKCE", "max_turns": 1},
                "reason": "Need previous recommendation context to classify 'why?'",
            },
            {
                "action": "classify",
                "category": "task_expansion",
                "related_task_id": "task_1",
                "safe_to_continue": True,
                "reason": "User asks for justification of the OAuth2 recommendation",
            },
        ]
    )
    tool_calls: list[dict[str, object]] = []

    def mock_conv_tool(query: str = "", max_turns: int = 1) -> str:
        tool_calls.append({"query": query, "max_turns": max_turns})
        return json.dumps(
            {
                "source": "conversation_context",
                "turns_returned": 1,
                "turns": [
                    {"turn_id": "turn_1", "role": "assistant", "content": "I recommend OAuth2 with PKCE."},
                ],
                "tokens": 25,
                "empty": False,
            }
        )

    classifier = FollowupClassifier(model)
    decision = classifier.decide(
        message="why?",
        recent_history=[],
        candidates=[{"task_id": "task_1", "normalized_intent": "auth architecture"}],
        conversation_tool=mock_conv_tool,
    )

    assert decision.category == "task_expansion"
    assert decision.related_task_id == "task_1"
    assert decision.safe_to_continue is True
    assert len(tool_calls) == 1
    assert tool_calls[0]["query"] == "OAuth2 PKCE"
    assert len(model.invocations) == 2

    # Verify initial prompt has recent_history: [] and no raw transcript
    first_payload = json.loads(model.invocations[0][1].content)
    assert first_payload["message"] == "why?"
    assert first_payload["recent_history"] == []
    assert "retrieved_context" not in first_payload

    # Verify second prompt contains retrieved_context
    second_payload = json.loads(model.invocations[1][1].content)
    assert "retrieved_context" in second_payload
    assert second_payload["retrieved_context"][0]["turns"][0]["content"] == "I recommend OAuth2 with PKCE."


def test_context_budget_blocked_followup_classification() -> None:
    from mana_agent.context_cost.models import (
        BudgetSnapshot,
        ContextBreakdown,
        ContextBudget,
        ContextBudgetExceeded,
        GovernorDecision,
    )

    class _BlockedModel:
        def with_structured_output(self, _schema, *, method: str, strict: bool):
            return self

        def invoke(self, _messages):
            snapshot = BudgetSnapshot(
                breakdown=ContextBreakdown(),
                budget=ContextBudget(context_window=8000),
                used_tokens=9000,
                remaining_tokens=0,
                utilization_ratio=1.125,
                cumulative_tokens=9000,
                remaining_task_tokens=0,
                cumulative_cost=0.05,
                remaining_cost=0.0,
                estimated=True,
                status="blocked",
            )
            raise ContextBudgetExceeded(
                GovernorDecision(
                    action="block",
                    reason="context_limit_deficit:1000",
                    allowed=False,
                    snapshot=snapshot,
                )
            )

    classifier = FollowupClassifier(_BlockedModel())
    with pytest.raises(FollowupClassificationError) as exc_info:
        classifier.decide(
            message="continue that task",
            recent_history=[],
            candidates=[{"task_id": "task_1"}],
        )
    assert exc_info.value.code == "context_budget_blocked"
    assert "Context budget blocked" in str(exc_info.value)
