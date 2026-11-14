"""Provider-independent inference and streaming contracts."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from mana_agent.llm.messages import AssistantMessage, Message, ToolCall


class Usage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)


class ContinuationState(BaseModel):
    """Explicit protocol checkpoint, separate from user-visible message history."""
    schema_version: Literal[1] = 1
    provider: str
    model: str
    endpoint: str
    response_id: str | None = None
    mode: Literal["stateless", "server", "volatile"] = "stateless"
    items: list[dict[str, Any]] = Field(default_factory=list)
    message_count: int = Field(default=0, ge=0)
    configuration_fingerprint: str = ""
    message_fingerprint: str = ""


class InferenceRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    messages: list[Message]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    response_schema: dict[str, Any] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    continuation: ContinuationState | None = None
    request_id: str = ""
    task_id: str = ""
    attempt_id: str = ""
    configuration_fingerprint: str = ""


class ModelResponse(BaseModel):
    content: str | list[dict[str, Any]] = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content_blocks(cls, value: Any) -> Any:
        return Message.normalize_content_blocks(value)

    @property
    def message(self) -> AssistantMessage:
        return AssistantMessage(content=self.content, tool_calls=self.tool_calls)

    id: str = ""
    provider: str = ""
    model: str = ""
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str = "stop"
    continuation: ContinuationState | None = None
    public_summaries: list[str] = Field(default_factory=list)
    protocol_events: list[dict[str, Any]] = Field(default_factory=list)
    protocol_pending: bool = False
    refusal: str | None = None
    configuration_observation: dict[str, Any] = Field(default_factory=dict)


class StreamEvent(BaseModel):
    type: Literal["text.delta", "tool.arguments.delta", "tool.completed", "summary", "completed"]
    request_id: str
    sequence: int
    item_id: str = ""
    call_id: str = ""
    name: str = ""
    delta: str = ""
    response: ModelResponse | None = None
