from __future__ import annotations
from collections.abc import AsyncIterator
from typing import Protocol
from mana_agent.llm.types import InferenceRequest, ModelResponse, StreamEvent


def validate_parameters(request: InferenceRequest) -> None:
    """SDK extra bodies cannot replace the validated native operation."""
    reserved = {"model", "messages", "input", "tools", "tool_choice", "response_format", "text", "previous_response_id", "conversation"}
    if reserved & request.parameters.keys():
        raise ValueError("Use native identity, history, tools, schema and continuation fields")
    extra = request.parameters.get("extra_body") or {}
    if not isinstance(extra, dict) or (reserved | {"max_tokens", "max_output_tokens", "max_completion_tokens", "stream"}) & extra.keys():
        raise ValueError("Provider extras cannot replace validated native request fields")

class InferenceProvider(Protocol):
    async def generate(self, request: InferenceRequest) -> ModelResponse: ...
    def stream(self, request: InferenceRequest) -> AsyncIterator[StreamEvent]: ...
    async def close(self) -> None: ...
