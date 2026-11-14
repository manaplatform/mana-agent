"""Mana-owned, JSON-serializable inference messages. No SDK objects live here."""
from __future__ import annotations

import json
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContentBlock(BaseModel):
    """A content part, retaining its explicitly supplied modality payload."""
    model_config = ConfigDict(extra="allow")
    type: str = Field(min_length=1)
    text: str | None = None


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    args: dict[str, Any]
    caller: dict[str, Any] | None = None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        if key not in type(self).model_fields:
            raise KeyError(key)
        return getattr(self, key)


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    caller: dict[str, Any] | None = None
    name: str | None = None
    message_id: str | None = None
    turn_id: str | None = None

    def __init__(self, content: Any = "", **values: Any) -> None:
        super().__init__(content=content, **values)

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content_blocks(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [ContentBlock.model_validate(part).model_dump(exclude_unset=True) for part in value]
        return value

    @property
    def message(self) -> Message:
        return self

    @property
    def type(self) -> str:
        return self.role

    @model_validator(mode="after")
    def validate_role(self) -> Message:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("Tool results require the originating tool call ID")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("Only assistant messages may declare tool calls")
        ids = [call.id for call in self.tool_calls]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate tool call IDs in an assistant message")
        return self


class UserMessage(Message):
    role: Literal["user"] = "user"


class SystemMessage(Message):
    role: Literal["system"] = "system"


class DeveloperMessage(Message):
    role: Literal["developer"] = "developer"


class AssistantMessage(Message):
    role: Literal["assistant"] = "assistant"


class ToolResultMessage(Message):
    role: Literal["tool"] = "tool"


def messages_from_records(values: Any) -> list[Message]:
    """Read native messages or persisted role/content records without losing data."""
    if isinstance(values, str):
        return [UserMessage(values)]
    result: list[Message] = []
    for value in values:
        if isinstance(value, Message):
            result.append(value)
            continue
        if not isinstance(value, dict):
            raise TypeError("Inference history must contain native messages or JSON records")
        record = dict(value)
        version = record.pop("schema_version", 1)
        if version != 1:
            raise ValueError("Unsupported inference message record version")
        if record.get("content") is None and record.get("role") == "assistant" and record.get("tool_calls"):
            record["content"] = ""
        # Version-one durable history used type/content for the two conversational roles.
        if "role" not in record and record.get("type") in {"human", "ai", "system", "tool"}:
            record["role"] = {"human": "user", "ai": "assistant"}.get(record["type"], record["type"])
            record.pop("type")
        calls = []
        for call in record.get("tool_calls") or []:
            if isinstance(call, ToolCall):
                calls.append(call)
            elif "function" in call:
                function = call["function"]
                arguments = function["arguments"]
                calls.append(ToolCall(id=call["id"], name=function["name"],
                    args=json.loads(arguments) if isinstance(arguments, str) else arguments, caller=call.get("caller")))
            else:
                calls.append(ToolCall.model_validate(call))
        record["tool_calls"] = calls
        message_type = {"user": UserMessage, "assistant": AssistantMessage, "system": SystemMessage,
            "developer": DeveloperMessage, "tool": ToolResultMessage}.get(record.get("role"), Message)
        result.append(message_type.model_validate(record))
    return result


def validate_conversation(messages: list[Message]) -> None:
    pending: set[str] = set()
    callers: dict[str, Any] = {}
    seen: set[str] = set()
    for message in messages:
        if message.role == "tool":
            if message.tool_call_id not in pending:
                raise ValueError("Orphan or duplicate tool result; history requires reconciliation")
            if message.caller != callers.get(message.tool_call_id):
                raise ValueError("Tool result program caller does not match its originating call")
            pending.remove(message.tool_call_id)
        elif message.tool_calls:
            if pending:
                raise ValueError("Unresolved tool calls precede the next assistant message")
            for call in message.tool_calls:
                if call.id in seen:
                    raise ValueError("Reused tool call identity in conversation")
                seen.add(call.id)
                pending.add(call.id)
                callers[call.id] = call.caller
        elif pending and message.role not in {"system", "developer"}:
            raise ValueError("Tool calls require correlated results before another conversation message")
    if pending:
        raise ValueError("Cannot infer with unresolved tool calls")
