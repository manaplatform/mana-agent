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

    def decide_sync(self, messages, schema, **kwargs):
        raw = self.generate_sync(messages, **kwargs)
        if isinstance(raw, schema):
            return raw
        if isinstance(raw, dict):
            return schema.model_validate(raw)
        return schema.model_validate_json(getattr(raw, "content", raw))

    def generate_sync(self, _messages, **_native_options):
        return self.payload


class _MultiTurnStructuredModel:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.invocations: list[list[object]] = []

    def decide_sync(self, messages, schema, **kwargs):
        raw = self.generate_sync(messages, **kwargs)
        if isinstance(raw, schema):
            return raw
        if isinstance(raw, dict):
            return schema.model_validate(raw)
        return schema.model_validate_json(getattr(raw, "content", raw))

    def generate_sync(self, messages, **_native_options):
        self.invocations.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return {"category": "conversation_only", "safe_to_continue": True, "reason": "default"}


def test_assistant_offer_resolves_without_task_candidates() -> None:
    offer = "Details " * 100 + "I can add the retry control. Would you like it?"
    classifier = FollowupClassifier(_StructuredModel({
        "category": "new_task", "safe_to_continue": True, "reason": "User accepted the visible offer",
        "resolved_request": "Add the retry control", "source_message_ids": ["assistant-offer"],
    }))
    result = classifier.decide(message="yes, I want it", candidates=[], recent_history=[
        {"role": "assistant", "message_id": "assistant-offer", "content": offer},
    ])
    assert result.resolved_request == "Add the retry control"
    assert result.source_message_ids == ("assistant-offer",)


def test_followup_resolves_role_and_turn_references_to_assistant_message() -> None:
    offer = "I found a bug in auth.py. Would you like me to fix it?"
    classifier = FollowupClassifier(_StructuredModel({
        "category": "new_task", "safe_to_continue": True, "reason": "User requested fix to model response",
        "resolved_request": "Fix bug in auth.py", "source_message_ids": ["assistant"],
    }))
    result = classifier.decide(message="fix this.", candidates=[], recent_history=[
        {"role": "assistant", "message_id": "msg-model-1", "turn_id": "turn-1", "content": offer},
    ])
    assert result.resolved_request == "Fix bug in auth.py"
    assert result.source_message_ids == ("msg-model-1",)

    # Also verify turn_id resolution
    classifier_turn = FollowupClassifier(_StructuredModel({
        "category": "new_task", "safe_to_continue": True, "reason": "User requested fix to model turn",
        "resolved_request": "Fix bug in auth.py", "source_message_ids": ["turn-1"],
    }))
    result_turn = classifier_turn.decide(message="fix this.", candidates=[], recent_history=[
        {"role": "assistant", "message_id": "msg-model-1", "turn_id": "turn-1", "content": offer},
    ])
    assert result_turn.source_message_ids == ("msg-model-1",)


def test_followup_rejects_invented_message_provenance() -> None:
    classifier = FollowupClassifier(_StructuredModel({
        "category": "new_task", "safe_to_continue": True, "reason": "Invalid source",
        "resolved_request": "Add a control", "source_message_ids": ["another-session-message"],
    }))
    with pytest.raises(FollowupClassificationError, match="unobserved message"):
        classifier.decide(message="yes", candidates=[], recent_history=[])


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


def test_independent_category_rejects_unsafe_decision() -> None:
    """Independent categories cannot override an unsafe model decision."""
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

    with pytest.raises(FollowupClassificationError, match="did not authorize"):
        classifier.decide(message="user@example.com", recent_history=[], candidates=[{"task_id": "task_1"}])


def test_conversation_only_rejects_unsafe_decision() -> None:
    """Conversation-only classification still requires model authorization."""
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

    with pytest.raises(FollowupClassificationError, match="did not authorize"):
        classifier.decide(message="hello", recent_history=[], candidates=[{"task_id": "task_1"}])


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


def test_classifier_receives_bounded_dialogue_history() -> None:
    model = _MultiTurnStructuredModel([
        {
            "category": "followup_task",
            "related_task_id": "task_1",
            "safe_to_continue": True,
            "reason": "the current turn continues the offered task",
        }
    ])
    classifier = FollowupClassifier(model)
    classifier.decide(
        message="why?",
        recent_history=[
            ("user", "question 1"),
            ("assistant", "answer 1"),
            ("user", "question 2"),
            ("assistant", "answer 2"),
        ],
        candidates=[{"task_id": "task_1", "state": "completed"}],
    )
    payload = json.loads(model.invocations[0][1].content)
    assert payload["recent_history"] == [
        ["user", "question 1"],
        ["assistant", "answer 1"],
        ["user", "question 2"],
        ["assistant", "answer 2"],
    ]


def test_context_budget_blocked_followup_classification() -> None:
    from mana_agent.context_cost.models import (
        BudgetSnapshot,
        ContextBreakdown,
        ContextBudget,
        ContextBudgetExceeded,
        GovernorDecision,
    )

    class _BlockedModel:
        def decide_sync(self, messages, schema, **kwargs):
            raw = self.generate_sync(messages, **kwargs)
            if isinstance(raw, schema):
                return raw
            if isinstance(raw, dict):
                return schema.model_validate(raw)
            return schema.model_validate_json(getattr(raw, "content", raw))

        def generate_sync(self, _messages, **_native_options):
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


def test_followup_structured_request_failure_stops_without_plain_retry() -> None:
    class Model:
        calls = 0
        def decide_sync(self, messages, schema, **kwargs):
            self.calls += 1
            raise RuntimeError("controlled structured request failure")
        def generate_sync(self, *args, **kwargs):
            pytest.fail("A plain-format retry is forbidden")
    model = Model()
    with pytest.raises(FollowupClassificationError):
        FollowupClassifier(model).decide(message="new request", recent_history=[], candidates=[])
    assert model.calls == 1


def test_followup_malformed_structured_output_stops_without_plain_retry() -> None:
    class Model:
        calls = 0
        def decide_sync(self, messages, schema, **kwargs):
            self.calls += 1
            return schema.model_validate_json("<think>private</think>")
        def generate_sync(self, *args, **kwargs):
            pytest.fail("A plain-format retry is forbidden")
    model = Model()
    with pytest.raises(FollowupClassificationError):
        FollowupClassifier(model).decide(message="new request", recent_history=[], candidates=[])
    assert model.calls == 1
