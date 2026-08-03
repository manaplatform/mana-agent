import pytest

from mana_agent.gateway.followup_classifier import (
    FollowupClassificationError,
    FollowupClassifier,
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


def test_expansion_must_select_an_offered_task() -> None:
    classifier = FollowupClassifier(
        _StructuredModel(
            {
                "decision_id": "followup_1",
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


def test_followup_cannot_attach_to_an_unoffered_task() -> None:
    classifier = FollowupClassifier(
        _StructuredModel(
            {
                "decision_id": "followup_2",
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
                "decision_id": "followup_2b",
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
                "decision_id": "followup_completed_resume",
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


def test_unsafe_classification_reports_the_model_reason() -> None:
    classifier = FollowupClassifier(
        _StructuredModel(
            {
                "decision_id": "followup_3",
                "category": "new_task",
                "related_task_id": "",
                "safe_to_continue": False,
                "reason": "the request refers to two conflicting task identities",
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
