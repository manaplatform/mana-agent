"""Native Anthropic Messages protocol, including explicit opaque continuations."""
from __future__ import annotations

import json
import time
import uuid
import copy
from collections.abc import AsyncIterator
from typing import Any
from mana_agent.llm.errors import InvalidResponseError
from mana_agent.llm.messages import ToolCall
from mana_agent.llm.providers.openai import plain
from mana_agent.llm.types import ContinuationState, InferenceRequest, ModelResponse, StreamEvent, Usage
from mana_agent.utils.text import _strip_thinking_tags


class AnthropicProvider:
    def __init__(self, *, provider: str, api_key: str, base_url: str, endpoint: str,
                 headers: dict[str, str] | None = None, query: dict[str, str] | None = None,
                 timeout: float = 600, client: Any = None,
                 volatile_continuations: dict | None = None):
        self.options = dict(api_key=api_key, base_url=base_url, default_headers=headers,
                            default_query=query, timeout=timeout, max_retries=0)
        self.client = client
        self.owns_client = client is None
        self.volatile_continuations = volatile_continuations if volatile_continuations is not None else {}

    def _client(self):
        if self.client is None:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(**self.options)
        return self.client

    def encode(self, request: InferenceRequest) -> dict[str, Any]:
        from mana_agent.llm.providers.base import validate_parameters
        validate_parameters(request)
        body = {"model": request.model, **request.parameters}
        effort = body.pop("reasoning_effort", None)
        if effort is not None:
            body["extra_body"] = {**body.get("extra_body", {}), "output_config": {
                **body.get("extra_body", {}).get("output_config", {}), "effort": effort}}
        for key in ("callbacks", "config", "stream"):
            body.pop(key, None)
        maximum = body.pop("max_output_tokens", body.pop("max_completion_tokens", None))
        if maximum is not None:
            body["max_tokens"] = maximum
        if not body.get("max_tokens"):
            raise ValueError("Anthropic requests require an evidence-backed output limit")
        system: list[dict[str, str]] = []
        messages = []
        if request.continuation is not None:
            stored = self.volatile_continuations.get(request.continuation.response_id)
            if stored is None or stored[0] < time.monotonic():
                raise InvalidResponseError("Anthropic volatile continuation expired or was interrupted. Resume through Supervisor reconciliation; private thinking is never saved in checkpoints.")
            messages = copy.deepcopy(stored[1])
        initial_instructions = True
        for index, message in enumerate(request.messages):
            if message.role in {"system", "developer"}:
                if not initial_instructions:
                    raise ValueError("Anthropic transport cannot move mid-conversation instructions")
                if not isinstance(message.content, str):
                    raise ValueError("Anthropic system instructions must be text")
                system.append({"type": "text", "text": message.content})
                continue
            initial_instructions = False
            if request.continuation and index < request.continuation.message_count:
                continue
            if message.role == "tool":
                role, parts = "user", [{"type": "tool_result", "tool_use_id": message.tool_call_id, "content": message.content}]
            else:
                role = message.role
                parts = [{"type": "text", "text": message.content}] if isinstance(message.content, str) and message.content else list(message.content or [])
                mapped = []
                for part in parts:
                    if part.get("type") == "image_url":
                        value = part["image_url"]
                        url = value.get("url") if isinstance(value, dict) else value
                        if url.startswith("data:"):
                            media, encoded = url.split(",", 1)
                            mapped.append({"type": "image", "source": {"type": "base64", "media_type": media[5:].split(";", 1)[0], "data": encoded}})
                        else:
                            mapped.append({"type": "image", "source": {"type": "url", "url": url}})
                    elif part.get("type") in {"text", "image", "document"}:
                        mapped.append(part)
                    else:
                        raise ValueError(f"Anthropic transport cannot encode part {part.get('type')}; no content was discarded")
                parts = mapped
                parts.extend({"type": "tool_use", "id": call.id, "name": call.name, "input": call.args} for call in message.tool_calls)
            if not parts:
                raise ValueError("Empty Anthropic conversation message")
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"].extend(parts)
            else:
                messages.append({"role": role, "content": parts})
        body.update(system=system, messages=messages)
        if request.tools:
            if any(tool.get("type") for tool in request.tools):
                raise ValueError("Hosted tool specifications require their selected native transport")
            body["tools"] = [{"name": tool["name"], "description": tool.get("description", ""), "input_schema": tool["parameters"]} for tool in request.tools]
        if request.tool_choice is not None:
            choice = request.tool_choice
            if isinstance(choice, str):
                body["tool_choice"] = {"type": {"required": "any"}.get(choice, choice)}
            else:
                body["tool_choice"] = {"type": "tool", "name": choice.get("name") or choice.get("function", {}).get("name")}
        if request.response_schema:
            body["extra_body"] = {**body.get("extra_body", {}), "output_config": {**body.get("extra_body", {}).get("output_config", {}), "format": {"type": "json_schema", "schema": request.response_schema}}}
        return body

    def normalize(self, raw: Any, request: InferenceRequest) -> ModelResponse:
        data = plain(raw)
        if data.get("stop_reason") in {"max_tokens", "pause_turn", "refusal"}:
            raise InvalidResponseError(f"Anthropic response is incomplete/refused: {data['stop_reason']}")
        text, calls, opaque = [], [], []
        has_private_thinking = False
        for block in data.get("content", []):
            if block["type"] == "text":
                text.append(block["text"] if request.response_schema else _strip_thinking_tags(block["text"]))
            elif block["type"] == "tool_use":
                calls.append(ToolCall(id=block["id"], name=block["name"], args=block["input"]))
            elif block["type"] == "redacted_thinking":
                opaque.append(block)
            elif block["type"] == "thinking":
                has_private_thinking = True
            else:
                raise InvalidResponseError(f"Unsupported Anthropic output block: {block['type']}")
        continuation = None
        if (calls and has_private_thinking) or opaque or request.continuation:
            # Signed thinking is necessary for this provider's tool continuation. Keep
            # it in bounded process memory only, never visible messages or checkpoints.
            now = time.monotonic()
            for key, value in list(self.volatile_continuations.items()):
                if value[0] < now:
                    del self.volatile_continuations[key]
            if len(self.volatile_continuations) >= 32:
                del self.volatile_continuations[next(iter(self.volatile_continuations))]
            receipt = uuid.uuid4().hex
            history = self.encode(request)["messages"]
            self.volatile_continuations[receipt] = (now + 600, [*history, {"role": "assistant", "content": data.get("content", [])}])
            continuation = ContinuationState(provider=request.provider, model=request.model, endpoint="anthropic_messages",
                response_id=receipt, mode="volatile", message_count=len(request.messages) + 1)
        counts = data.get("usage") or {}
        cached = counts.get("cache_read_input_tokens", 0) or 0
        incoming = (counts.get("input_tokens", 0) or 0) + cached + (counts.get("cache_creation_input_tokens", 0) or 0)
        outgoing = counts.get("output_tokens", 0) or 0
        usage = Usage(input_tokens=incoming, output_tokens=outgoing, total_tokens=incoming + outgoing,
            cached_input_tokens=cached)
        return ModelResponse(id=data.get("id", ""), provider=request.provider, model=request.model, content="".join(text),
            tool_calls=calls, usage=usage, finish_reason=data.get("stop_reason") or "stop",
            continuation=continuation,
            configuration_observation={"reasoning_effort": (data.get("output_config") or {}).get("effort")})

    async def generate(self, request: InferenceRequest) -> ModelResponse:
        raw = await self._client().messages.create(**self.encode(request))
        try:
            return self.normalize(raw, request)
        except Exception as exc:
            exc.provider_usage = plain(raw).get("usage")
            exc.response_accepted = True
            raise

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamEvent]:
        sequence = 0
        async with self._client().messages.stream(**self.encode(request)) as stream:
            tool_ids: dict[int, tuple[str, str]] = {}
            async for raw in stream:
                event = plain(raw)
                if event["type"] == "content_block_start" and event["content_block"]["type"] == "tool_use":
                    block = event["content_block"]
                    tool_ids[event["index"]] = (block["id"], block["name"])
                elif event["type"] == "content_block_delta":
                    delta = event["delta"]
                    if delta["type"] == "text_delta":
                        sequence += 1
                        yield StreamEvent(type="text.delta", request_id=request.request_id, sequence=sequence, delta=delta["text"])
                    elif delta["type"] == "input_json_delta":
                        identity, name = tool_ids[event["index"]]
                        sequence += 1
                        yield StreamEvent(type="tool.arguments.delta", request_id=request.request_id, sequence=sequence,
                            call_id=identity, name=name, delta=delta["partial_json"])
                elif event["type"] == "error":
                    raise InvalidResponseError("Anthropic stream failed")
            response = self.normalize(await stream.get_final_message(), request)
            for call in response.tool_calls:
                sequence += 1
                yield StreamEvent(type="tool.completed", request_id=request.request_id, sequence=sequence, call_id=call.id, name=call.name)
            yield StreamEvent(type="completed", request_id=request.request_id, sequence=sequence + 1, response=response)

    async def close(self) -> None:
        if self.owns_client and self.client is not None:
            await self.client.close()
            self.client = None
