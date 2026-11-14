"""Direct SDK adapters for Responses and Chat Completions protocols."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from mana_agent.llm.errors import InvalidResponseError, StructuredOutputError
from mana_agent.llm.messages import ToolCall
from mana_agent.llm.types import ContinuationState, InferenceRequest, ModelResponse, StreamEvent, Usage
from mana_agent.utils.text import _strip_thinking_tags


def plain(value: Any) -> Any:
    return value.model_dump(mode="json", exclude_none=True) if hasattr(value, "model_dump") else value


def usage(value: Any) -> Usage:
    raw = plain(value) or {}
    prompt = raw.get("input_tokens", raw.get("prompt_tokens", 0)) or 0
    output = raw.get("output_tokens", raw.get("completion_tokens", 0)) or 0
    return Usage(input_tokens=prompt, output_tokens=output, total_tokens=raw.get("total_tokens") or prompt + output,
        cached_input_tokens=(raw.get("input_tokens_details") or raw.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0,
        reasoning_tokens=(raw.get("output_tokens_details") or raw.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Create the provider's strict schema without changing application validation."""
    if isinstance(schema, list):
        return [strict_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    if not schema or (schema.get("type") == "object" and not schema.get("properties") and schema.get("additionalProperties") is not False):
        # Strict provider schemas require fixed object keys. A free-form JSON
        # value uses a declared string on the wire, then the native schema
        # validates the decoded value. This is one operation, never a retry.
        return {"type": "string", "description": "A JSON-encoded value for this field. " + schema.get("description", "")}
    value = dict(schema)
    value.pop("default", None)
    for key in ("properties", "$defs", "definitions"):
        if isinstance(value.get(key), dict):
            value[key] = {name: strict_schema(item) for name, item in value[key].items()}
    for key in ("items", "anyOf", "oneOf", "allOf"):
        if key in value:
            value[key] = strict_schema(value[key])
    if value.get("type") == "object" or "properties" in value:
        properties = value.get("properties", {})
        value["required"] = list(properties)
        value["additionalProperties"] = False
    return value


def decode_structured(content: str, schema: dict[str, Any] | None) -> str:
    if schema is None:
        return content
    from mana_agent.multi_agent.tools.operation_policy import validate_arguments
    try:
        value = json.loads(content)
    except (ValueError, TypeError) as exc:
        raise StructuredOutputError("The provider response was not valid JSON for the requested schema.") from exc
    wire_schema = strict_schema(schema)
    try:
        validate_arguments(value, wire_schema)
    except (ValueError, TypeError) as exc:
        raise StructuredOutputError("The provider returned JSON that did not match the requested structured-output schema.") from exc
    def restore(item, shape):
        if "$ref" in shape:
            target = schema
            for key in shape["$ref"].removeprefix("#/").split("/"):
                target = target[key.replace("~1", "/").replace("~0", "~")]
            return restore(item, target)
        if not shape or (shape.get("type") == "object" and not shape.get("properties") and shape.get("additionalProperties") is not False):
            return json.loads(item)
        for option in shape.get("anyOf", []):
            try:
                validate_arguments(item, strict_schema(option), _root=wire_schema)
            except (ValueError, TypeError):
                continue
            return restore(item, option)
        if isinstance(item, dict):
            return {key: restore(data, shape.get("properties", {}).get(key, {})) for key, data in item.items()}
        if isinstance(item, list):
            return [restore(data, shape.get("items", {})) for data in item]
        return item
    try:
        native = restore(value, schema)
        validate_arguments(native, schema)
    except (ValueError, TypeError) as exc:
        raise StructuredOutputError("The provider response did not match the application schema after decoding.") from exc
    return json.dumps(native, ensure_ascii=False)


class OpenAIProvider:
    def __init__(self, *, provider: str, api_key: str, base_url: str, endpoint: str,
                 headers: dict[str, str] | None = None, query: dict[str, str] | None = None,
                 timeout: float = 600, client: Any = None):
        self.provider, self.endpoint = provider, endpoint
        self.options = dict(api_key=api_key, base_url=base_url, default_headers=headers,
                            default_query=query, timeout=timeout, max_retries=0)
        self.client = client
        self.owns_client = client is None

    def _client(self):
        if self.client is None:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(**self.options)
        return self.client

    def encode(self, request: InferenceRequest) -> dict[str, Any]:
        from mana_agent.llm.providers.base import validate_parameters
        validate_parameters(request)
        parameters = dict(request.parameters)
        output_token_parameter = parameters.pop("output_token_parameter", None)
        if output_token_parameter is not None and output_token_parameter not in {"max_tokens", "max_completion_tokens", "max_output_tokens"}:
            raise ValueError("Unknown output token field for the native adapter")
        for key in ("callbacks", "config", "stream"):
            parameters.pop(key, None)
        body: dict[str, Any] = {"model": request.model, **parameters}
        if self.endpoint == "responses":
            items: list[dict[str, Any]] = []
            selected_messages = request.messages[request.continuation.message_count:] if request.continuation else request.messages
            for message in selected_messages:
                if message.role == "tool":
                    items.append({"type": "function_call_output", "call_id": message.tool_call_id, "output": message.content, **({"caller": message.caller} if message.caller else {})})
                    continue
                if message.content:
                    content = message.content
                    if isinstance(content, list):
                        content = [self._response_part(part, message.role) for part in content]
                    items.append({"role": message.role, "content": content})
                items.extend({"type": "function_call", "call_id": call.id, "name": call.name,
                              "arguments": json.dumps(call.args, ensure_ascii=False), **({"caller": call.caller} if call.caller else {})} for call in message.tool_calls)
            if request.continuation and request.continuation.mode == "server":
                if not request.continuation.response_id:
                    raise InvalidResponseError("Server continuation has no response receipt")
                body["previous_response_id"] = request.continuation.response_id
            elif request.continuation:
                items = [*request.continuation.items, *items]
            body["input"] = items
            body.setdefault("store", False)
            if body.get("store") is False and (request.tools or body.get("reasoning") or body.get("reasoning_effort")):
                body["include"] = list(dict.fromkeys([*body.get("include", []), "reasoning.encrypted_content"]))
            if request.tools:
                body["tools"] = [{"type": "function", **tool} if "type" not in tool else tool for tool in request.tools]
            if request.response_schema:
                body["text"] = {"format": {"type": "json_schema", "name": "mana_response", "strict": True,
                                            "schema": strict_schema(request.response_schema)}}
            if "reasoning_effort" in body:
                body["reasoning"] = {**body.get("reasoning", {}), "effort": body.pop("reasoning_effort")}
            for key in ("max_tokens", "max_completion_tokens"):
                if key in body:
                    body["max_output_tokens"] = body.pop(key)
        else:
            records = []
            for message in request.messages:
                record = {"role": message.role, "content": message.content}
                if message.name:
                    record["name"] = message.name
                if message.tool_calls:
                    record["tool_calls"] = [{"id": call.id, "type": "function", "function": {
                        "name": call.name, "arguments": json.dumps(call.args, ensure_ascii=False)}} for call in message.tool_calls]
                if message.tool_call_id:
                    record["tool_call_id"] = message.tool_call_id
                records.append(record)
            body["messages"] = records
            if request.tools:
                if any(tool.get("type") not in {None, "function"} for tool in request.tools):
                    raise ValueError("Selected Chat Completions transport cannot encode hosted tools")
                body["tools"] = [{"type": "function", "function": tool} for tool in request.tools]
            if request.response_schema:
                body["response_format"] = {"type": "json_schema", "json_schema": {"name": "mana_response", "strict": True,
                    "schema": strict_schema(request.response_schema)}}
                if self.provider == "openrouter":
                    # OpenRouter can otherwise drop parameters unsupported by the served endpoint.
                    extra = dict(body.get("extra_body") or {})
                    provider = body.pop("provider", {})
                    provider_extra = extra.get("provider", {})
                    if not isinstance(provider, dict) or not isinstance(provider_extra, dict):
                        raise ValueError("OpenRouter provider routing settings must be an object")
                    extra["provider"] = {**provider, **provider_extra, "require_parameters": True}
                    body["extra_body"] = extra
            if "max_output_tokens" in body:
                key = output_token_parameter or ("max_completion_tokens" if self.provider == "openai" else "max_tokens")
                if key == "max_output_tokens":
                    raise ValueError("Chat Completions requires max_tokens or max_completion_tokens")
                body[key] = body.pop("max_output_tokens")
        if request.tool_choice is not None:
            choice = request.tool_choice
            if isinstance(choice, dict) and choice.get("type") == "function":
                name = choice.get("name") or choice.get("function", {}).get("name")
                choice = {"type": "function", "name": name} if self.endpoint == "responses" else {"type": "function", "function": {"name": name}}
            body["tool_choice"] = choice
        return body

    @staticmethod
    def _response_part(part: dict[str, Any], role: str) -> dict[str, Any]:
        kind = part.get("type")
        if kind == "text":
            return {"type": "output_text" if role == "assistant" else "input_text", "text": part["text"]}
        if kind == "image_url":
            image = part["image_url"]
            return {"type": "input_image", "image_url": image["url"] if isinstance(image, dict) else image,
                    **({"detail": image["detail"]} if isinstance(image, dict) and "detail" in image else {})}
        if kind in {"input_text", "input_image", "input_file", "input_audio", "output_text"}:
            return dict(part)
        raise ValueError(f"Responses cannot encode content part {kind}; no attachment was discarded")

    def normalize(self, raw: Any, request: InferenceRequest) -> ModelResponse:
        data = plain(raw)
        if self.endpoint == "responses":
            status = data.get("status")
            if data.get("error") or status not in {None, "completed"}:
                raise InvalidResponseError("Provider rejected the request or returned incomplete output")
            content, calls, receipts, summaries, refusals, events = [], [], [], [], [], []
            has_final_message = False
            for item in data.get("output", []):
                kind = item.get("type")
                if kind == "message":
                    has_final_message = True
                    public_parts = []
                    for part in item.get("content", []):
                        if part["type"] == "output_text":
                            visible = part["text"] if request.response_schema else _strip_thinking_tags(part["text"])
                            content.append(visible)
                            public_parts.append({**part, "text": visible})
                        elif part["type"] == "refusal":
                            refusals.append(part.get("refusal", "refused"))
                        else:
                            raise InvalidResponseError(f"Unrepresentable output part: {part['type']}")
                    receipts.append({**item, "content": public_parts})
                elif kind == "function_call":
                    if item.get("status") not in {None, "completed"}:
                        raise InvalidResponseError("Provider returned an incomplete function call")
                    calls.append(ToolCall(id=item["call_id"], name=item["name"], args=json.loads(item["arguments"]), caller=item.get("caller")))
                    receipts.append(item)
                elif kind == "reasoning":
                    summaries.extend(part["text"] for part in item.get("summary", []) if part.get("type") == "summary_text")
                    if item.get("encrypted_content"):
                        receipts.append({key: item[key] for key in ("type", "id", "encrypted_content") if key in item})
                    elif request.parameters.get("store") is not True and any(part.get("type") == "function_call" for part in data["output"]):
                        raise InvalidResponseError("Stateless reasoning/tool continuation requires the provider's encrypted reasoning receipt")
                elif kind in {"programmatic_tool_call", "programmatic_tool_calling", "program", "program_output"}:
                    receipts.append(item)
                    events.append(item)
                else:
                    raise InvalidResponseError(f"Unsupported Responses output item: {kind}")
            return ModelResponse(id=data.get("id", ""), provider=request.provider, model=request.model,
                content=decode_structured("".join(content), request.response_schema) if not refusals else "", tool_calls=calls, usage=usage(data.get("usage")),
                public_summaries=summaries, protocol_events=events, protocol_pending=not has_final_message and bool(events), configuration_observation={"reasoning_effort": (data.get("reasoning") or {}).get("effort")}, refusal="; ".join(refusals) or None,
                continuation=ContinuationState(provider=request.provider, model=request.model, endpoint=self.endpoint,
                    response_id=data.get("id"), mode="server" if request.parameters.get("store") is True else "stateless",
                    items=[] if request.parameters.get("store") is True else [*self.encode(request)["input"], *receipts], message_count=len(request.messages) + 1) if receipts else None)
        choices = data.get("choices") or []
        if len(choices) != 1:
            raise InvalidResponseError("Exactly one inference choice is required")
        choice, message = choices[0], choices[0].get("message") or {}
        if message.get("role") not in {None, "assistant"}:
            raise InvalidResponseError("Provider choice did not contain an assistant message")
        if choice.get("finish_reason") in {"length", "content_filter"}:
            raise InvalidResponseError(f"Incomplete provider response: {choice['finish_reason']}")
        calls = [ToolCall(id=call["id"], name=call["function"]["name"], args=json.loads(call["function"]["arguments"])) for call in message.get("tool_calls") or []]
        content = message.get("content") or ""
        if isinstance(content, str) and not request.response_schema:
            content = _strip_thinking_tags(content)
        if message.get("audio"):
            content = ([{"type": "text", "text": content}] if content else []) + [{"type": "audio", **message["audio"]}]
        if request.response_schema and not message.get("refusal"):
            content = decode_structured(content, request.response_schema)
        return ModelResponse(id=data.get("id", ""), provider=request.provider, model=request.model,
            content=content, tool_calls=calls, usage=usage(data.get("usage")),
            finish_reason=choice.get("finish_reason") or "stop", refusal=message.get("refusal"),
            configuration_observation={"reasoning_effort": data.get("reasoning_effort")})

    async def generate(self, request: InferenceRequest) -> ModelResponse:
        client = self._client()
        operation = client.responses.create if self.endpoint == "responses" else client.chat.completions.create
        raw = await operation(**self.encode(request))
        try:
            return self.normalize(raw, request)
        except Exception as exc:
            exc.provider_usage = usage(plain(raw).get("usage")).model_dump()
            exc.response_accepted = True
            raise

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamEvent]:
        client = self._client()
        operation = client.responses.create if self.endpoint == "responses" else client.chat.completions.create
        body = {**self.encode(request), "stream": True}
        if self.endpoint != "responses":
            body["stream_options"] = {"include_usage": True}
        stream = await operation(**body)
        texts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        seen: set[int] = set()
        last_sequence = -1
        sequence = 0
        response = None
        final_usage = None
        final_reason = None
        try:
            async for raw in stream:
                data = plain(raw)
                source_sequence = data.get("sequence_number")
                if source_sequence is not None:
                    if source_sequence in seen:
                        continue
                    if source_sequence < last_sequence:
                        raise InvalidResponseError("Provider stream sequence regressed; reconcile the interrupted response")
                    last_sequence = source_sequence
                    seen.add(source_sequence)
                event = None
                if self.endpoint == "responses":
                    kind = data.get("type")
                    if kind == "response.output_text.delta":
                        event = dict(type="text.delta", delta=data["delta"], item_id=data.get("item_id", ""))
                    elif kind == "response.output_item.added" and data.get("item", {}).get("type") == "function_call":
                        calls[data["output_index"]] = dict(data["item"])
                    elif kind == "response.function_call_arguments.delta":
                        item = calls.get(data["output_index"])
                        if item is None:
                            raise InvalidResponseError("Tool argument delta preceded its declaration")
                        item["arguments"] = item.get("arguments", "") + data["delta"]
                        event = dict(type="tool.arguments.delta", item_id=data.get("item_id", ""),
                            call_id=item["call_id"], name=item["name"], delta=data["delta"])
                    elif kind == "response.function_call_arguments.done":
                        item = calls.get(data["output_index"])
                        if item is None or not isinstance(data.get("arguments"), str):
                            raise InvalidResponseError("Tool argument completion lacks its original call")
                        if item.get("arguments") and json.loads(item["arguments"]) != json.loads(data["arguments"]):
                            raise InvalidResponseError("Complete tool arguments differ from the streamed fragments")
                        item["arguments"] = data["arguments"]
                    elif kind == "response.completed":
                        response = self.normalize(data["response"], request)
                        by_id = {call.id: call for call in response.tool_calls}
                        for item in calls.values():
                            call = by_id.get(item["call_id"])
                            if call is None or call.name != item["name"]:
                                raise InvalidResponseError("Completed tool call does not match its stream identity")
                            if item.get("arguments") and json.loads(item["arguments"]) != call.args:
                                raise InvalidResponseError("Completed tool call does not match its stream arguments")
                    elif kind in {"response.failed", "response.incomplete", "error"}:
                        raise InvalidResponseError(f"Provider stream failed: {kind}")
                else:
                    final_usage = data.get("usage") or final_usage
                    for choice in data.get("choices", []):
                        if choice.get("index", 0) != 0:
                            raise InvalidResponseError("More than one streamed response choice is unsupported")
                        final_reason = choice.get("finish_reason") or final_reason
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            texts.append(delta["content"])
                            sequence += 1
                            yield StreamEvent(type="text.delta", request_id=request.request_id, sequence=sequence, delta=delta["content"])
                        if delta.get("refusal"):
                            raise InvalidResponseError("Provider refused the streamed request")
                        for call in delta.get("tool_calls") or []:
                            index = call["index"]
                            item = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                            if call.get("id") and item["id"] and call["id"] != item["id"]:
                                raise InvalidResponseError("Chat stream changed a tool call identity")
                            item["id"] = call.get("id") or item["id"]
                            function = call.get("function") or {}
                            item["name"] += function.get("name") or ""
                            fragment = function.get("arguments") or ""
                            item["arguments"] += fragment
                            if fragment:
                                if not item["id"] or not item["name"]:
                                    raise InvalidResponseError("Chat tool arguments arrived before the call identity")
                                sequence += 1
                                yield StreamEvent(type="tool.arguments.delta", request_id=request.request_id, sequence=sequence,
                                    call_id=item["id"], name=item["name"], delta=fragment)
                if event:
                    sequence += 1
                    yield StreamEvent(request_id=request.request_id, sequence=sequence, **event)
            if self.endpoint != "responses":
                if final_reason not in {"stop", "tool_calls"}:
                    raise InvalidResponseError(f"Incomplete Chat Completions stream: {final_reason}")
                response = ModelResponse(provider=request.provider, model=request.model, content="".join(texts),
                    tool_calls=[ToolCall(id=item["id"], name=item["name"], args=json.loads(item["arguments"])) for _, item in sorted(calls.items())],
                    usage=usage(final_usage), finish_reason=final_reason)
                if request.response_schema:
                    response.content = decode_structured(response.content, request.response_schema)
                else:
                    response.content = _strip_thinking_tags(response.content)
            if response is None:
                raise InvalidResponseError("Provider stream ended without a completed response")
            for call in response.tool_calls:
                sequence += 1
                yield StreamEvent(type="tool.completed", request_id=request.request_id, sequence=sequence, call_id=call.id, name=call.name)
            yield StreamEvent(type="completed", request_id=request.request_id, sequence=sequence + 1, response=response)
        finally:
            await stream.close()

    async def close(self) -> None:
        if self.owns_client and self.client is not None:
            await self.client.close()
            self.client = None
