"""Unit tests for model text extraction and reasoning filter."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from mana_agent.utils.text import extract_model_text


def test_extract_model_text_plain_string() -> None:
    assert extract_model_text("Hello world") == "Hello world"
    assert extract_model_text("") == ""
    assert extract_model_text(None) == ""


def test_extract_model_text_filters_reasoning_blocks() -> None:
    content = [
        {
            "id": "rs_035abc8957930c0b006a7fabd2d08c87d2b028b7f037237cd2",
            "summary": [],
            "type": "reasoning",
            "content": [],
            "encrypted_content": "gAAAAABqf6vTRu0G3QbNBBKTB59L8Uzumn4LPQnbKNuwVbnGE4JfzPh51k...",
        },
        {
            "type": "text",
            "text": "Got it — for the rest of this conversation, **test = 1**.",
        },
    ]
    extracted = extract_model_text(content)
    assert extracted == "Got it — for the rest of this conversation, **test = 1**."
    assert "rs_035abc89" not in extracted
    assert "encrypted_content" not in extracted


def test_extract_model_text_filters_thought_and_thinking_blocks() -> None:
    content = [
        {"type": "thought", "thought": "Internal thoughts here"},
        {"type": "thinking", "thinking": "More reasoning here"},
        {"type": "redacted_thinking", "data": "secret"},
        {"type": "text", "text": "Visible answer."},
    ]
    extracted = extract_model_text(content)
    assert extracted == "Visible answer."
    assert "Internal thoughts" not in extracted


def test_extract_model_text_handles_multiple_text_blocks() -> None:
    content = [
        {"type": "text", "text": "Part 1."},
        {"type": "text", "text": "Part 2."},
    ]
    assert extract_model_text(content) == "Part 1. Part 2."


def test_extract_model_text_from_ai_message() -> None:
    message = AIMessage(
        content=[
            {"type": "reasoning", "encrypted_content": "xyz123"},
            {"type": "text", "text": "The answer is 42."},
        ]
    )
    assert extract_model_text(message) == "The answer is 42."


def test_extract_model_text_handles_plain_strings_in_list() -> None:
    content = ["Hello", "World"]
    assert extract_model_text(content) == "Hello World"
