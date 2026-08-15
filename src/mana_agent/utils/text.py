"""Text extraction and normalization utilities for model outputs."""
from __future__ import annotations

import json
from typing import Any


def extract_model_text(content: Any) -> str:
    """Extract human/assistant-visible text from model output, discarding reasoning/thinking blocks.

    Handles string outputs, list-of-blocks structures (from Anthropic, Gemini, OpenAI thinking,
    LangChain AIMessage, etc.), and nested structures. Discards reasoning and thinking dictionaries
    rather than stringifying raw metadata payloads.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                trimmed = item.strip()
                if trimmed:
                    parts.append(trimmed)
                continue
            if isinstance(item, dict):
                block_type = str(item.get("type", "") or "").strip().lower()
                if block_type in {
                    "reasoning",
                    "thought",
                    "thinking",
                    "redacted_thinking",
                }:
                    continue
                if "text" in item and item["text"] is not None:
                    text_val = str(item["text"]).strip()
                    if text_val:
                        parts.append(text_val)
                    continue
                if "output_text" in item and item["output_text"] is not None:
                    text_val = str(item["output_text"]).strip()
                    if text_val:
                        parts.append(text_val)
                    continue
                if "input_text" in item and item["input_text"] is not None:
                    text_val = str(item["input_text"]).strip()
                    if text_val:
                        parts.append(text_val)
                    continue
                nested_content = item.get("content")
                if isinstance(nested_content, str):
                    nested_val = nested_content.strip()
                    if nested_val:
                        parts.append(nested_val)
                elif isinstance(nested_content, (list, tuple)):
                    nested_extracted = extract_model_text(list(nested_content))
                    if nested_extracted:
                        parts.append(nested_extracted)
                continue
            if hasattr(item, "text") and isinstance(item.text, str):
                text_val = item.text.strip()
                if text_val:
                    parts.append(text_val)
                continue
            if hasattr(item, "content"):
                extracted = extract_model_text(item.content)
                if extracted:
                    parts.append(extracted)
                continue
        return " ".join(parts).strip()
    if hasattr(content, "content"):
        return extract_model_text(content.content)
    if hasattr(content, "text") and isinstance(content.text, str):
        return content.text.strip()
    return str(content).strip()
