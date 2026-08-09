"""Transform Chat Completions SSE chunks into Responses API SSE events."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator

from mana_agent.integrations.codex.text_cleanup import sanitize_assistant_visible_text


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _sse(event_type: str, data: dict[str, Any], *, sequence: int) -> str:
    payload = dict(data)
    payload.setdefault("type", event_type)
    payload.setdefault("sequence_number", sequence)
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@dataclass
class _ToolCallState:
    index: int
    item_id: str
    call_id: str
    name: str = ""
    arguments: str = ""
    output_index: int = 0
    added: bool = False


@dataclass
class ChatToResponsesStreamAdapter:
    """Stateful Chat Completions → Responses SSE translator."""

    model: str
    response_id: str = field(default_factory=lambda: _new_id("resp"))
    sequence: int = 0
    created_at: int = field(default_factory=lambda: int(time.time()))
    message_item_id: str = field(default_factory=lambda: _new_id("msg"))
    reasoning_item_id: str = field(default_factory=lambda: _new_id("rs"))
    content_index: int = 0
    message_output_index: int = 0
    reasoning_output_index: int = 0
    message_started: bool = False
    content_started: bool = False
    reasoning_started: bool = False
    text_parts: list[str] = field(default_factory=list)
    # DeepSeek / NVIDIA thinking tokens — never mix into assistant content.
    reasoning_parts: list[str] = field(default_factory=list)
    tool_calls: dict[int, _ToolCallState] = field(default_factory=dict)
    next_output_index: int = 1
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    completed: bool = False
    # Progress flags used to decide safe recovery after mid-stream failures.
    received_stream_data: bool = False
    tool_calls_started: bool = False
    open_emitted: bool = False

    def _next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    @property
    def tool_side_effects(self) -> bool:
        """True when tool-call items were emitted (arguments may have been executed)."""
        return self.tool_calls_started or bool(self.tool_calls)

    def progress_snapshot(self) -> dict[str, Any]:
        return {
            "received_stream_data": self.received_stream_data,
            "tool_side_effects": self.tool_side_effects,
            "text_chars": sum(len(part) for part in self.text_parts),
            "reasoning_chars": sum(len(part) for part in self.reasoning_parts),
            "tool_call_count": len(self.tool_calls),
            "open_emitted": self.open_emitted,
            "completed": self.completed,
        }

    def _ensure_reasoning_item(self) -> list[str]:
        """Open a Responses reasoning item for provider chain-of-thought."""
        if self.reasoning_started:
            return []
        self.reasoning_started = True
        # Prefer index 0 when nothing else has started so multi-turn round-trip
        # can reattach reasoning to the following assistant tool/message turn.
        if not self.message_started and not self.tool_calls_started:
            self.reasoning_output_index = 0
            self.message_output_index = 1
            self.next_output_index = max(self.next_output_index, 2)
        else:
            self.reasoning_output_index = self.next_output_index
            self.next_output_index += 1
        return [
            _sse(
                "response.output_item.added",
                {
                    "output_index": self.reasoning_output_index,
                    "item": {
                        "type": "reasoning",
                        "id": self.reasoning_item_id,
                        "status": "in_progress",
                        "summary": [],
                    },
                },
                sequence=self._next_sequence(),
            )
        ]

    def open_events(self) -> list[str]:
        self.open_emitted = True
        seq = self._next_sequence()
        created = {
            "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": self.created_at,
                "status": "in_progress",
                "model": self.model,
                "output": [],
            }
        }
        return [_sse("response.created", created, sequence=seq)]

    def _ensure_message_item(self) -> list[str]:
        events: list[str] = []
        if not self.message_started:
            self.message_started = True
            events.append(
                _sse(
                    "response.output_item.added",
                    {
                        "output_index": self.message_output_index,
                        "item": {
                            "type": "message",
                            "id": self.message_item_id,
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                    sequence=self._next_sequence(),
                )
            )
        if not self.content_started:
            self.content_started = True
            events.append(
                _sse(
                    "response.content_part.added",
                    {
                        "item_id": self.message_item_id,
                        "output_index": self.message_output_index,
                        "content_index": self.content_index,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    },
                    sequence=self._next_sequence(),
                )
            )
        return events

    def _tool_state(self, index: int) -> _ToolCallState:
        state = self.tool_calls.get(index)
        if state is None:
            state = _ToolCallState(
                index=index,
                item_id=_new_id("fc"),
                call_id=_new_id("call"),
                output_index=self.next_output_index,
            )
            self.next_output_index += 1
            self.tool_calls[index] = state
        return state

    def ingest_chat_chunk(self, chunk: dict[str, Any]) -> list[str]:
        events: list[str] = []
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
            self.usage = {
                "input_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
                "total_tokens": int(
                    usage.get("total_tokens")
                    or (
                        int(usage.get("prompt_tokens") or 0)
                        + int(usage.get("completion_tokens") or 0)
                    )
                ),
            }
            self.received_stream_data = True
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        if not choices:
            return events
        choice = choices[0] if isinstance(choices[0], dict) else {}
        if choice.get("finish_reason"):
            self.finish_reason = str(choice.get("finish_reason"))
            self.received_stream_data = True
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        # DeepSeek thinking mode streams CoT on reasoning_content (same level as
        # content). Never fold it into assistant text — multi-turn tool loops
        # require it as a separate field on the next request.
        reasoning_delta = delta.get("reasoning_content")
        if reasoning_delta not in (None, ""):
            self.received_stream_data = True
            events.extend(self._ensure_reasoning_item())
            text = str(reasoning_delta)
            self.reasoning_parts.append(text)
            events.append(
                _sse(
                    "response.reasoning_summary_text.delta",
                    {
                        "item_id": self.reasoning_item_id,
                        "output_index": self.reasoning_output_index,
                        "delta": text,
                    },
                    sequence=self._next_sequence(),
                )
            )
        content = delta.get("content")
        if content:
            self.received_stream_data = True
            events.extend(self._ensure_message_item())
            text = str(content)
            self.text_parts.append(text)
            events.append(
                _sse(
                    "response.output_text.delta",
                    {
                        "item_id": self.message_item_id,
                        "output_index": self.message_output_index,
                        "content_index": self.content_index,
                        "delta": text,
                    },
                    sequence=self._next_sequence(),
                )
            )
        tool_calls = delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []
        for tool_delta in tool_calls:
            if not isinstance(tool_delta, dict):
                continue
            self.received_stream_data = True
            self.tool_calls_started = True
            index = int(tool_delta.get("index") or 0)
            state = self._tool_state(index)
            if tool_delta.get("id"):
                state.call_id = str(tool_delta["id"])
            function = tool_delta.get("function") if isinstance(tool_delta.get("function"), dict) else {}
            if function.get("name"):
                state.name = str(function["name"])
            # Never parse partial JSON arguments; only accumulate fragments.
            if function.get("arguments"):
                state.arguments += str(function["arguments"])
            if not state.added:
                state.added = True
                events.append(
                    _sse(
                        "response.output_item.added",
                        {
                            "output_index": state.output_index,
                            "item": {
                                "type": "function_call",
                                "id": state.item_id,
                                "call_id": state.call_id,
                                "name": state.name,
                                "arguments": "",
                                "status": "in_progress",
                            },
                        },
                        sequence=self._next_sequence(),
                    )
                )
            if function.get("arguments"):
                events.append(
                    _sse(
                        "response.function_call_arguments.delta",
                        {
                            "item_id": state.item_id,
                            "output_index": state.output_index,
                            "delta": str(function["arguments"]),
                        },
                        sequence=self._next_sequence(),
                    )
                )
        return events

    def close_events(self, *, failed: bool = False, error: dict[str, Any] | None = None) -> list[str]:
        if self.completed:
            return []
        self.completed = True
        events: list[str] = []
        if failed:
            events.append(
                _sse(
                    "response.failed",
                    {
                        "response": {
                            "id": self.response_id,
                            "object": "response",
                            "status": "failed",
                            "error": error or {"code": "bridge_error", "message": "stream failed"},
                            "model": self.model,
                            "output": [],
                        }
                    },
                    sequence=self._next_sequence(),
                )
            )
            return events

        full_reasoning = "".join(self.reasoning_parts)
        if self.reasoning_started or full_reasoning:
            if not self.reasoning_started:
                events.extend(self._ensure_reasoning_item())
            events.append(
                _sse(
                    "response.reasoning_summary_text.done",
                    {
                        "item_id": self.reasoning_item_id,
                        "output_index": self.reasoning_output_index,
                        "text": full_reasoning,
                    },
                    sequence=self._next_sequence(),
                )
            )
            events.append(
                _sse(
                    "response.output_item.done",
                    {
                        "output_index": self.reasoning_output_index,
                        "item": {
                            "type": "reasoning",
                            "id": self.reasoning_item_id,
                            "status": "completed",
                            "summary": (
                                [{"type": "summary_text", "text": full_reasoning}]
                                if full_reasoning
                                else []
                            ),
                        },
                    },
                    sequence=self._next_sequence(),
                )
            )

        if self.message_started:
            # Sanitize completed message text. Live deltas may still carry partial
            # markers; final item text must not poison Codex multi-turn history.
            full_text = sanitize_assistant_visible_text("".join(self.text_parts))
            events.append(
                _sse(
                    "response.output_text.done",
                    {
                        "item_id": self.message_item_id,
                        "output_index": self.message_output_index,
                        "content_index": self.content_index,
                        "text": full_text,
                    },
                    sequence=self._next_sequence(),
                )
            )
            events.append(
                _sse(
                    "response.content_part.done",
                    {
                        "item_id": self.message_item_id,
                        "output_index": self.message_output_index,
                        "content_index": self.content_index,
                        "part": {"type": "output_text", "text": full_text, "annotations": []},
                    },
                    sequence=self._next_sequence(),
                )
            )
            events.append(
                _sse(
                    "response.output_item.done",
                    {
                        "output_index": self.message_output_index,
                        "item": {
                            "type": "message",
                            "id": self.message_item_id,
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": full_text, "annotations": []}
                            ],
                        },
                    },
                    sequence=self._next_sequence(),
                )
            )

        for index in sorted(self.tool_calls):
            state = self.tool_calls[index]
            events.append(
                _sse(
                    "response.function_call_arguments.done",
                    {
                        "item_id": state.item_id,
                        "output_index": state.output_index,
                        "arguments": state.arguments,
                    },
                    sequence=self._next_sequence(),
                )
            )
            events.append(
                _sse(
                    "response.output_item.done",
                    {
                        "output_index": state.output_index,
                        "item": {
                            "type": "function_call",
                            "id": state.item_id,
                            "call_id": state.call_id,
                            "name": state.name,
                            "arguments": state.arguments,
                            "status": "completed",
                        },
                    },
                    sequence=self._next_sequence(),
                )
            )

        # Stable order for clients that only read response.completed.output:
        # reasoning → message → function_calls (DeepSeek multi-turn contract).
        output: list[dict[str, Any]] = []
        if self.reasoning_started or full_reasoning:
            output.append(
                {
                    "type": "reasoning",
                    "id": self.reasoning_item_id,
                    "status": "completed",
                    "summary": (
                        [{"type": "summary_text", "text": full_reasoning}]
                        if full_reasoning
                        else []
                    ),
                }
            )
        if self.message_started:
            completed_text = sanitize_assistant_visible_text("".join(self.text_parts))
            output.append(
                {
                    "type": "message",
                    "id": self.message_item_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": completed_text,
                            "annotations": [],
                        }
                    ],
                }
            )
        for index in sorted(self.tool_calls):
            state = self.tool_calls[index]
            output.append(
                {
                    "type": "function_call",
                    "id": state.item_id,
                    "call_id": state.call_id,
                    "name": state.name,
                    "arguments": state.arguments,
                    "status": "completed",
                }
            )
        status = "completed"
        if self.finish_reason == "length":
            status = "incomplete"
        events.append(
            _sse(
                "response.completed",
                {
                    "response": {
                        "id": self.response_id,
                        "object": "response",
                        "created_at": self.created_at,
                        "status": status,
                        "model": self.model,
                        "output": output,
                        "usage": self.usage,
                        "incomplete_details": (
                            {"reason": "max_output_tokens"} if status == "incomplete" else None
                        ),
                    }
                },
                sequence=self._next_sequence(),
            )
        )
        return events


def parse_chat_sse_lines(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Yield Chat Completions JSON objects from an SSE line iterator."""
    data_lines: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if not data_lines:
                continue
            payload = "\n".join(data_lines)
            data_lines = []
            if payload.strip() == "[DONE]":
                return
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                yield parsed
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        payload = "\n".join(data_lines)
        if payload.strip() != "[DONE]":
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                return
            if isinstance(parsed, dict):
                yield parsed


__all__ = [
    "ChatToResponsesStreamAdapter",
    "parse_chat_sse_lines",
]
