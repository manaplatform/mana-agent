"""Tests for the automatic AI engine selection layer and Decisions API integration."""

from __future__ import annotations

import json
from typing import Any
import httpx
import pytest
from pydantic import BaseModel

from mana_agent.engine_selection import (
    DEFAULT_DECISIONS_MODEL,
    DecisionsAnswer,
    DecisionsClient,
    DecisionsChoiceItem,
    DecisionsScoreLevel,
    EngineSelectionResult,
    EngineSelector,
    EngineType,
    ExecutionResult,
    Operation,
    OperationKind,
    SmartDispatcher,
    batch_op,
    choice_op,
    conversation_op,
    deterministic_op,
    executeAuto,
    execute_auto,
    function_call_op,
    generation_op,
    predicate_op,
    reasoning_op,
    score_op,
    structured_op,
)


class MockLLM:
    """Mock LLM for structured and text responses in testing."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[Any] = []

    def invoke(self, messages: list[Any], **kwargs: Any) -> Any:
        self.calls.append(messages)
        content = messages[-1].content if messages else ""
        text_resp = self.responses.get(str(content))
        if text_resp is None:
            for k, v in self.responses.items():
                if k in str(content):
                    text_resp = v
                    break
        if text_resp is None:
            if self.responses:
                text_resp = next(iter(self.responses.values()))
            else:
                text_resp = json.dumps({"result": True, "choice": "billing", "score": 1.0, "level": "Good", "name": "mock"})
        if isinstance(text_resp, dict):
            text_resp = json.dumps(text_resp)
        from types import SimpleNamespace
        return SimpleNamespace(content=str(text_resp))

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        class _Runner:
            def __init__(self, parent: MockLLM) -> None:
                self.parent = parent

            def invoke(self, messages: list[Any]) -> Any:
                self.parent.calls.append(messages)
                if isinstance(schema, type) and issubclass(schema, BaseModel):
                    return schema.model_validate({"key": "extracted_val"})
                return {"key": "extracted_val"}

        return _Runner(self)


def _make_mock_client(response_dict: dict[str, Any], status_code: int = 200) -> tuple[DecisionsClient, list[httpx.Request]]:
    """Create a DecisionsClient backed by a mock HTTP transport."""
    recorded_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        return httpx.Response(status_code, json=response_dict)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = DecisionsClient(api_key="sk-test-key", http_client=http_client)
    return client, recorded_requests


# -----------------------------------------------------------------------------
# 1. Supported classifications automatically use Decisions
# -----------------------------------------------------------------------------

def test_supported_predicate_uses_decisions() -> None:
    """Boolean semantic evaluation automatically uses Decisions predicate."""
    selector = EngineSelector()
    op = predicate_op(
        instructions="Does the code contain security vulnerabilities?",
        evidence="def test(): pass",
        name="sec_check",
    )
    selection = selector.select_engine(op)
    assert selection.selected_engine == EngineType.DECISIONS_PREDICATE
    assert selection.is_decisions_eligible is True

    # Execution via Decisions client
    client, requests = _make_mock_client({
        "answers": [
            {"type": "predicate", "name": "sec_check", "result": False, "probability": 0.05, "confidence": 0.95}
        ]
    })
    dispatcher = SmartDispatcher(selector=selector, decisions_client=client)
    result = dispatcher.execute_auto(op)

    assert result.success is True
    assert result.engine_used == EngineType.DECISIONS_PREDICATE
    assert result.result is False
    assert result.confidence == 0.95
    assert len(requests) == 1
    req_body = json.loads(requests[0].content)
    assert req_body["model"] == DEFAULT_DECISIONS_MODEL
    assert req_body["questions"][0]["type"] == "predicate"


def test_supported_choice_uses_decisions() -> None:
    """Scenario A: Intent classification among fixed options uses Decisions choice."""
    selector = EngineSelector()
    op = choice_op(
        instructions="Select the appropriate route for this user request.",
        choices=["github", "repository", "search", "conversation"],
        evidence="Search GitHub for issues related to authentication.",
        name="intent_route",
    )
    selection = selector.select_engine(op)
    assert selection.selected_engine == EngineType.DECISIONS_CHOICE
    assert selection.is_decisions_eligible is True

    client, requests = _make_mock_client({
        "answers": [
            {"type": "choice", "name": "intent_route", "choice": "github", "confidence": 0.94}
        ]
    })
    dispatcher = SmartDispatcher(selector=selector, decisions_client=client)
    result = dispatcher.execute_auto(op)

    assert result.success is True
    assert result.engine_used == EngineType.DECISIONS_CHOICE
    assert result.result == "github"
    assert result.confidence == 0.94
    assert len(requests) == 1
    req_body = json.loads(requests[0].content)
    assert req_body["questions"][0]["type"] == "choice"
    assert len(req_body["questions"][0]["choices"]) == 4


def test_supported_score_uses_decisions() -> None:
    """Evaluation against ordered levels/rubric uses Decisions score."""
    selector = EngineSelector()
    op = score_op(
        instructions="Evaluate the severity of this issue.",
        levels=[
            {"label": "Cosmetic", "description": "Appearance only", "value": 0},
            {"label": "Workaround", "description": "Alternative available", "value": 1},
            {"label": "Blocker", "description": "Complete blockage", "value": 2},
        ],
        evidence="Export button does nothing in Safari.",
        name="severity_score",
    )
    selection = selector.select_engine(op)
    assert selection.selected_engine == EngineType.DECISIONS_SCORE
    assert selection.is_decisions_eligible is True

    client, requests = _make_mock_client({
        "answers": [
            {"type": "score", "name": "severity_score", "score": 1.2, "confidence": 0.88}
        ]
    })
    dispatcher = SmartDispatcher(selector=selector, decisions_client=client)
    result = dispatcher.execute_auto(op)

    assert result.success is True
    assert result.engine_used == EngineType.DECISIONS_SCORE
    assert result.result == 1.2
    assert len(requests) == 1
    req_body = json.loads(requests[0].content)
    assert req_body["questions"][0]["type"] == "score"
    assert len(req_body["questions"][0]["levels"]) == 3


def test_supported_batch_uses_decisions() -> None:
    """Multiple independent evaluations use batched Decisions."""
    selector = EngineSelector()
    sub1 = predicate_op("Is it broken?", "error log", name="q1")
    sub2 = choice_op("Which component?", ["ui", "api", "db"], "error log", name="q2")
    op = batch_op([sub1, sub2], evidence="error log", name="batch_check")

    selection = selector.select_engine(op)
    assert selection.selected_engine == EngineType.DECISIONS_BATCH
    assert selection.is_decisions_eligible is True

    client, requests = _make_mock_client({
        "answers": [
            {"type": "predicate", "name": "q1", "result": True, "confidence": 0.9},
            {"type": "choice", "name": "q2", "choice": "api", "confidence": 0.85},
        ]
    })
    dispatcher = SmartDispatcher(selector=selector, decisions_client=client)
    result = dispatcher.execute_auto(op)

    assert result.success is True
    assert result.engine_used == EngineType.DECISIONS_BATCH
    assert result.result == [True, "api"]
    assert len(requests) == 1
    req_body = json.loads(requests[0].content)
    assert len(req_body["questions"]) == 2


# -----------------------------------------------------------------------------
# 2. Unsupported operations never call Decisions
# -----------------------------------------------------------------------------

def test_structured_output_uses_responses_never_decisions() -> None:
    """Arbitrary JSON object generation uses Responses Structured Outputs, never Decisions."""
    class TargetSchema(BaseModel):
        key: str

    selector = EngineSelector()
    op = structured_op(
        instructions="Extract data into schema",
        output_schema=TargetSchema,
        evidence={"text": "sample text"},
        name="extract_data",
    )
    selection = selector.select_engine(op)
    assert selection.selected_engine == EngineType.RESPONSES_STRUCTURED
    assert selection.is_decisions_eligible is False

    client, requests = _make_mock_client({})
    mock_llm = MockLLM()
    dispatcher = SmartDispatcher(selector=selector, decisions_client=client)
    result = dispatcher.execute_auto(op, llm=mock_llm)

    assert result.success is True
    assert result.engine_used == EngineType.RESPONSES_STRUCTURED
    assert len(requests) == 0  # Decisions was NEVER called
    assert len(mock_llm.calls) == 1


def test_tool_arguments_uses_function_calling_never_decisions() -> None:
    """Scenario B: Tool arguments generation uses function calling, never Decisions."""
    selector = EngineSelector()
    op = function_call_op(
        tool_name="create_issue",
        tool_schema={"type": "object", "properties": {"title": {"type": "string"}, "body": {"type": "string"}}},
        input_prompt="Create an issue in repository X with title 'Bug' and description 'Broken'",
    )
    selection = selector.select_engine(op)
    assert selection.selected_engine == EngineType.FUNCTION_CALLING
    assert selection.is_decisions_eligible is False

    client, requests = _make_mock_client({})
    mock_llm = MockLLM()
    dispatcher = SmartDispatcher(selector=selector, decisions_client=client)
    result = dispatcher.execute_auto(op, llm=mock_llm)

    assert result.success is True
    assert result.engine_used == EngineType.FUNCTION_CALLING
    assert len(requests) == 0  # Decisions was NEVER called


def test_complex_coding_uses_reasoning_never_decisions() -> None:
    """Scenario C: Complex coding and reasoning uses reasoning engine, never Decisions."""
    selector = EngineSelector()
    op = reasoning_op(
        instructions="Analyze and refactor this application.",
        input_data="class LegacyService: pass",
    )
    selection = selector.select_engine(op)
    assert selection.selected_engine == EngineType.REASONING_ENGINE
    assert selection.is_decisions_eligible is False

    client, requests = _make_mock_client({})
    mock_llm = MockLLM({"LegacyService": "Refactored code"})
    dispatcher = SmartDispatcher(selector=selector, decisions_client=client)
    result = dispatcher.execute_auto(op, llm=mock_llm)

    assert result.success is True
    assert result.engine_used == EngineType.REASONING_ENGINE
    assert len(requests) == 0  # Decisions was NEVER called


def test_open_ended_conversation_uses_llm_never_decisions() -> None:
    """Open-ended conversation uses conversation LLM, never Decisions."""
    selector = EngineSelector()
    op = conversation_op("Hello, how are you today?")
    selection = selector.select_engine(op)
    assert selection.selected_engine == EngineType.CONVERSATION_LLM
    assert selection.is_decisions_eligible is False

    client, requests = _make_mock_client({})
    mock_llm = MockLLM({"Hello": "I'm doing well!"})
    dispatcher = SmartDispatcher(selector=selector, decisions_client=client)
    result = dispatcher.execute_auto(op, llm=mock_llm)

    assert result.success is True
    assert result.engine_used == EngineType.CONVERSATION_LLM
    assert len(requests) == 0  # Decisions was NEVER called


def test_generation_uses_generation_engine_never_decisions() -> None:
    """Text generation and summarization uses generation engine, never Decisions."""
    selector = EngineSelector()
    op = generation_op("Summarize this document", "Long text...")
    selection = selector.select_engine(op)
    assert selection.selected_engine == EngineType.GENERATION_ENGINE
    assert selection.is_decisions_eligible is False


# -----------------------------------------------------------------------------
# 3. Deterministic operations bypass AI
# -----------------------------------------------------------------------------

def test_deterministic_operations_bypass_ai() -> None:
    """Scenario E: An already validated internal command with an exact handler executes directly without AI."""
    handler_called = False

    def local_handler(input_data: Any) -> str:
        nonlocal handler_called
        handler_called = True
        return f"Handled: {input_data}"

    op = deterministic_op(local_handler, "internal_command_arg", name="help_command")
    selector = EngineSelector()
    selection = selector.select_engine(op)

    assert selection.selected_engine == EngineType.DETERMINISTIC
    assert selection.is_decisions_eligible is False

    client, requests = _make_mock_client({})
    mock_llm = MockLLM()
    dispatcher = SmartDispatcher(selector=selector, decisions_client=client)
    result = dispatcher.execute_auto(op, llm=mock_llm)

    assert result.success is True
    assert result.engine_used == EngineType.DETERMINISTIC
    assert result.result == "Handled: internal_command_arg"
    assert handler_called is True
    assert len(requests) == 0  # Zero Decisions requests
    assert len(mock_llm.calls) == 0  # Zero LLM requests


# -----------------------------------------------------------------------------
# 4. Decisions eligibility: Evidence formats & invalid questions
# -----------------------------------------------------------------------------

def test_unsupported_hosted_image_url_skips_decisions() -> None:
    """Decisions does not support hosted external image URLs; must proactively skip Decisions."""
    selector = EngineSelector()
    op = predicate_op(
        instructions="Is this product damaged?",
        evidence={
            "type": "input_image",
            "image_url": "https://example.com/hosted_photo.png",  # Hosted external URL
        },
    )
    selection = selector.select_engine(op)
    assert selection.selected_engine == EngineType.COMPATIBLE_FALLBACK
    assert selection.is_decisions_eligible is False
    assert "hosted external image URLs" in selection.reason


def test_supported_base64_image_url_uses_decisions() -> None:
    """Decisions supports base64 data URLs for images."""
    selector = EngineSelector()
    op = predicate_op(
        instructions="Is this product damaged?",
        evidence={
            "type": "input_image",
            "image_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==",
        },
    )
    selection = selector.select_engine(op)
    assert selection.selected_engine == EngineType.DECISIONS_PREDICATE
    assert selection.is_decisions_eligible is True


def test_insufficient_choices_skips_decisions() -> None:
    """Choice operation with fewer than 2 choices cannot use Decisions API."""
    selector = EngineSelector()
    op = choice_op(
        instructions="Pick an option",
        choices=["only_one_choice"],
        evidence="data",
    )
    selection = selector.select_engine(op)
    assert selection.selected_engine == EngineType.COMPATIBLE_FALLBACK
    assert selection.is_decisions_eligible is False
    assert "at least two fixed choices" in selection.reason


def test_insufficient_score_levels_skips_decisions() -> None:
    """Score operation without explicit rubric levels cannot use Decisions API."""
    selector = EngineSelector()
    op = score_op(
        instructions="Score this item",
        levels=[],  # Empty rubric
        evidence="data",
    )
    selection = selector.select_engine(op)
    assert selection.selected_engine == EngineType.COMPATIBLE_FALLBACK
    assert selection.is_decisions_eligible is False
    assert "explicit ordered rubric/levels" in selection.reason


# -----------------------------------------------------------------------------
# 5. Avoid duplicate AI calls (Shared decisions)
# -----------------------------------------------------------------------------

def test_shared_decisions_are_not_recomputed() -> None:
    """Repeated calls for the same operation and context use cached result without extra AI calls."""
    selector = EngineSelector()
    op = predicate_op("Is user authenticated?", {"user": "alice"}, name="auth_check")

    client, requests = _make_mock_client({
        "answers": [{"type": "predicate", "name": "auth_check", "result": True, "confidence": 1.0}]
    })
    dispatcher = SmartDispatcher(selector=selector, decisions_client=client)

    # First call
    res1 = dispatcher.execute_auto(op, context="session_123")
    assert res1.success is True
    assert res1.cached is False
    assert len(requests) == 1

    # Second call (identical operation & context)
    res2 = dispatcher.execute_auto(op, context="session_123")
    assert res2.success is True
    assert res2.cached is True
    assert res2.result == res1.result
    assert len(requests) == 1  # No second request made!


# -----------------------------------------------------------------------------
# 6. Fallback behavior: API failures trigger safe bounded fallback
# -----------------------------------------------------------------------------

def test_decisions_refusal_triggers_safe_fallback() -> None:
    """Model refusal in Decisions triggers bounded fallback to compatible engine."""
    selector = EngineSelector()
    op = choice_op(
        instructions="Classify this complaint department",
        choices=["billing", "shipping", "technical"],
        evidence="I was charged twice.",
        name="dept_choice",
    )

    # Decisions client returns a refusal
    client, requests = _make_mock_client({
        "answers": [{"type": "refusal", "name": "dept_choice", "refusal": "I cannot answer this question."}]
    })
    mock_llm = MockLLM({"dept_choice": json.dumps({"choice": "billing", "reason": "Mentions charged twice"})})
    dispatcher = SmartDispatcher(selector=selector, decisions_client=client)

    result = dispatcher.execute_auto(op, llm=mock_llm)
    assert result.success is True
    assert result.fallback_occurred is True
    assert "refused" in result.fallback_reason.lower()
    assert result.result == "billing"
    assert len(requests) == 1  # 1 Decisions call made (and failed)
    assert len(mock_llm.calls) == 1  # Safe bounded fallback invoked once


def test_decisions_timeout_triggers_safe_fallback() -> None:
    """Decisions timeout triggers bounded fallback to compatible engine."""
    selector = EngineSelector()
    op = predicate_op("Is file clean?", "clean content", name="clean_check")

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Connection timed out")

    transport = httpx.MockTransport(timeout_handler)
    client = DecisionsClient(api_key="test", http_client=httpx.Client(transport=transport))
    mock_llm = MockLLM({"clean_check": json.dumps({"result": True, "reason": "No issues"})})

    dispatcher = SmartDispatcher(selector=selector, decisions_client=client)
    result = dispatcher.execute_auto(op, llm=mock_llm)

    assert result.success is True
    assert result.fallback_occurred is True
    assert "timed out" in result.fallback_reason.lower()
    assert result.result is True


def test_decisions_http_error_triggers_safe_fallback() -> None:
    """Decisions HTTP 500 error triggers bounded fallback to compatible engine."""
    selector = EngineSelector()
    op = score_op(
        instructions="Rate quality",
        levels=[{"label": "Poor"}, {"label": "Good"}],
        evidence="sample",
        name="quality_score",
    )

    client, _ = _make_mock_client({"error": "Internal server error"}, status_code=500)
    mock_llm = MockLLM({"quality_score": json.dumps({"score": 1.0, "reason": "Good"})})

    dispatcher = SmartDispatcher(selector=selector, decisions_client=client)
    result = dispatcher.execute_auto(op, llm=mock_llm)

    assert result.success is True
    assert result.fallback_occurred is True
    assert "http 500" in result.fallback_reason.lower()


# -----------------------------------------------------------------------------
# 7. No mandatory Decisions calls
# -----------------------------------------------------------------------------

def test_decisions_disabled_bypasses_decisions_completely() -> None:
    """When Decisions is disabled, supported operations skip Decisions proactively."""
    selector = EngineSelector(decisions_enabled=False)
    op = predicate_op("Is this valid?", "data", name="valid_check")

    selection = selector.select_engine(op)
    assert selection.is_decisions_eligible is False
    assert selection.selected_engine == EngineType.COMPATIBLE_FALLBACK

    client, requests = _make_mock_client({})
    mock_llm = MockLLM({"valid_check": json.dumps({"result": True})})
    dispatcher = SmartDispatcher(selector=selector, decisions_client=client)
    result = dispatcher.execute_auto(op, llm=mock_llm)

    assert result.success is True
    assert len(requests) == 0  # Zero Decisions requests attempted


def test_non_openai_provider_bypasses_decisions_completely() -> None:
    """When provider is non-OpenAI (e.g. NVIDIA), Decisions is proactively skipped."""
    selector = EngineSelector(provider="nvidia", base_url="https://integrate.api.nvidia.com/v1")
    op = choice_op("Route question", ["route_a", "route_b"], "data")

    selection = selector.select_engine(op)
    assert selection.is_decisions_eligible is False
    assert selection.selected_engine == EngineType.COMPATIBLE_FALLBACK
    assert "nvidia" in selection.reason.lower()


# -----------------------------------------------------------------------------
# 8. API parity: execute_auto and executeAuto
# -----------------------------------------------------------------------------

def test_execute_auto_and_executeAuto_parity() -> None:
    """execute_auto and executeAuto behave identically."""
    op = deterministic_op(lambda x: x.upper(), "test_input")
    res1 = execute_auto(op)
    res2 = executeAuto(op)

    assert res1.result == "TEST_INPUT"
    assert res2.result == "TEST_INPUT"
    assert res1.engine_used == EngineType.DETERMINISTIC
    assert res2.engine_used == EngineType.DETERMINISTIC
