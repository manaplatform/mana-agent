from __future__ import annotations
import asyncio
from types import SimpleNamespace
import pytest
from mana_agent.llm.messages import UserMessage, AssistantMessage, ToolCall, ToolResultMessage
from mana_agent.llm.providers.openai import OpenAIProvider
from mana_agent.llm.providers.anthropic import AnthropicProvider
from mana_agent.llm.types import InferenceRequest
from mana_agent.llm.errors import InvalidResponseError, StructuredOutputError

def request(**values):
    return InferenceRequest(provider="openai", model="test-model", messages=[UserMessage("hello")], request_id="request-1", **values)

def test_chat_encoding_preserves_exact_ids_and_provider_body():
    adapter = OpenAIProvider(provider="custom", api_key="test", base_url="https://example.org", endpoint="chat_completions")
    selected = request(parameters={"extra_body": {"chat_template_kwargs": {"thinking": False}}})
    selected.messages.extend([AssistantMessage(tool_calls=[ToolCall(id="call-1", name="echo", args={"value": "a"})]), ToolResultMessage(content="exact result", tool_call_id="call-1")])
    body = adapter.encode(selected)
    assert body["messages"][-1] == {"role": "tool", "content": "exact result", "tool_call_id": "call-1"}
    assert body["extra_body"]["chat_template_kwargs"] == {"thinking": False}

def test_responses_preserves_program_caller_and_stateless_replay():
    adapter = OpenAIProvider(provider="openai", api_key="test", base_url="https://example.org", endpoint="responses")
    selected = request()
    response = adapter.normalize({"id": "response-1", "status": "completed", "output": [
        {"type": "program", "call_id": "program-1", "fingerprint": "opaque", "code": "await tools.echo({value:'a'})"},
        {"type": "function_call", "call_id": "call-1", "name": "echo", "arguments": '{"value":"a"}', "caller": {"type": "program", "caller_id": "program-1"}},
    ]}, selected)
    assert response.tool_calls[0].caller["caller_id"] == "program-1"
    selected.messages.extend([response.message, ToolResultMessage(content="a", tool_call_id="call-1", caller=response.tool_calls[0].caller)])
    selected.continuation = response.continuation
    encoded = adapter.encode(selected)
    assert encoded["input"][-1]["caller"]["caller_id"] == "program-1"
    assert sum(item.get("type") == "function_call" for item in encoded["input"]) == 1

def test_native_schema_uses_provider_schema_parameter():
    schema = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}
    selected = request(response_schema=schema)
    responses = OpenAIProvider(provider="openai", api_key="test", base_url="https://example.org", endpoint="responses")
    chat = OpenAIProvider(provider="custom", api_key="test", base_url="https://example.org", endpoint="chat_completions")
    assert responses.encode(selected)["text"]["format"]["type"] == "json_schema"
    assert chat.encode(selected)["response_format"]["type"] == "json_schema"

def test_openrouter_structured_output_requires_parameters_and_preserves_endpoint_preferences():
    schema = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}
    request_settings = {"extra_body": {"provider": {"order": ["fixture"], "require_parameters": False}, "sort": "latency"}}
    selected = request(parameters=request_settings, response_schema=schema)
    adapter = OpenAIProvider(provider="openrouter", api_key="test", base_url="https://example.org", endpoint="chat_completions")
    body = adapter.encode(selected)
    assert body["extra_body"]["provider"] == {"order": ["fixture"], "require_parameters": True}
    assert body["extra_body"]["sort"] == "latency"
    assert selected.parameters["extra_body"]["provider"]["require_parameters"] is False
    assert "extra_body" not in adapter.encode(request())


@pytest.mark.parametrize(("content", "diagnostic"), [
    ("fixture-sensitive-output", "not valid JSON"),
    ('{"value": "ok", "unexpected": "fixture-sensitive-output"}', "did not match the requested structured-output schema"),
])
def test_openrouter_reports_invalid_structured_output_without_exposing_response(content, diagnostic):
    selected = request(response_schema={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]})
    adapter = OpenAIProvider(provider="openrouter", api_key="test", base_url="https://example.org", endpoint="chat_completions")
    raw = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}
    with pytest.raises(StructuredOutputError, match=diagnostic) as error:
        adapter.normalize(raw, selected)
    assert "fixture-sensitive-output" not in str(error.value)


def test_anthropic_maps_native_tools_system_and_schema():
    adapter = AnthropicProvider(provider="anthropic", api_key="test", base_url="https://example.org", endpoint="anthropic_messages")
    selected = request(parameters={"max_output_tokens": 128}, tools=[{"name": "echo", "parameters": {"type": "object"}}], response_schema={"type": "object"})
    body = adapter.encode(selected)
    assert body["max_tokens"] == 128
    assert body["tools"][0]["input_schema"] == {"type": "object"}
    assert body["extra_body"]["output_config"]["format"]["type"] == "json_schema"

@pytest.mark.parametrize("status", ["failed", "incomplete", "cancelled"])
def test_incomplete_responses_are_not_success(status):
    adapter = OpenAIProvider(provider="openai", api_key="test", base_url="https://example.org", endpoint="responses")
    with pytest.raises(InvalidResponseError):
        adapter.normalize({"status": status, "output": []}, request())

def test_chat_stream_reassembles_arguments_and_reports_usage():
    class Stream:
        def __aiter__(self):
            async def events():
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "echo", "arguments": '{"value":'}}]}}]}
                yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"a"}'}}]}, "finish_reason": "tool_calls"}]}
                yield {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}}
            return events()
        async def close(self):
            self.closed = True
    stream = Stream()
    async def create(**kwargs):
        return stream
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    adapter = OpenAIProvider(provider="custom", api_key="test", base_url="https://example.org", endpoint="chat_completions", client=client)
    async def collect():
        return [event async for event in adapter.stream(request())]
    events = asyncio.run(collect())
    assert events[-1].response.tool_calls[0].args == {"value": "a"}
    assert events[-1].response.usage.total_tokens == 5
    assert stream.closed


def test_anthropic_stream_preserves_tool_arguments_and_public_text():
    class Stream:
        closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            self.closed = True

        async def __aiter__(self):
            yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Checking"}}
            yield {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "call-1", "name": "echo"}}
            for fragment in ('{"value":', '"a"}'):
                yield {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": fragment}}

        async def get_final_message(self):
            return {"id": "message-1", "stop_reason": "tool_use", "content": [
                {"type": "text", "text": "Checking"}, {"type": "tool_use", "id": "call-1", "name": "echo", "input": {"value": "a"}},
            ], "usage": {"input_tokens": 2, "output_tokens": 3, "cache_read_input_tokens": 4}}

    stream = Stream()
    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **options: stream))
    adapter = AnthropicProvider(provider="anthropic", api_key="test", base_url="https://example.org",
                                endpoint="anthropic_messages", client=client)

    async def collect():
        return [event async for event in adapter.stream(request(parameters={"max_output_tokens": 128}))]

    events = asyncio.run(collect())
    assert events[0].delta == "Checking"
    assert [event.call_id for event in events if event.type == "tool.arguments.delta"] == ["call-1", "call-1"]
    assert events[-1].response.tool_calls[0].args == {"value": "a"}
    assert events[-1].response.usage.total_tokens == 9
    assert stream.closed

@pytest.mark.parametrize("parameters", [{"model": "another"}, {"tools": []}, {"extra_body": {"messages": []}}, {"extra_body": {"max_output_tokens": 1000000}}])
def test_provider_extras_cannot_replace_validated_operation(parameters):
    for adapter in [OpenAIProvider(provider="openai", api_key="test", base_url="https://example.org", endpoint="responses"),
                    AnthropicProvider(provider="anthropic", api_key="test", base_url="https://example.org", endpoint="anthropic_messages")]:
        with pytest.raises(ValueError):
            adapter.encode(request(parameters=parameters))


def test_schema_wire_encoding_restores_freeform_maps_without_schema_weakening():
    import json
    from mana_agent.llm.providers.openai import strict_schema, decode_structured
    schema = {"type": "object", "properties": {"arguments": {"type": "object", "additionalProperties": True}}, "required": ["arguments"]}
    wire = strict_schema(schema)
    assert wire["properties"]["arguments"]["type"] == "string"
    assert json.loads(decode_structured('{"arguments":"{\\"path\\":\\"README.md\\"}"}', schema)) == {"arguments": {"path": "README.md"}}
    with pytest.raises(ValueError):
        decode_structured('{"arguments":"invalid JSON"}', schema)


def test_anthropic_signed_thinking_stays_in_volatile_receipt():
    adapter = AnthropicProvider(provider="anthropic", api_key="test", base_url="https://example.org", endpoint="anthropic_messages")
    selected = request(parameters={"max_output_tokens": 128})
    selected.provider = "anthropic"
    result = adapter.normalize({"id": "message", "stop_reason": "tool_use", "content": [
        {"type": "thinking", "thinking": "PRIVATE", "signature": "signed"},
        {"type": "tool_use", "id": "call-1", "name": "echo", "input": {"value": "a"}},
    ]}, selected)
    assert result.continuation.mode == "volatile"
    assert "PRIVATE" not in result.model_dump_json()
    selected.messages.extend([result.message, ToolResultMessage(content="a", tool_call_id="call-1")])
    selected.continuation = result.continuation
    body = adapter.encode(selected)
    assert body["messages"][-2]["content"][0]["signature"] == "signed"
    adapter.volatile_continuations.clear()
    with pytest.raises(InvalidResponseError, match="expired or was interrupted"):
        adapter.encode(selected)


def test_responses_stream_preserves_ids_and_checks_completed_arguments():
    class Stream:
        async def __aiter__(self):
            item = {"type": "function_call", "id": "item", "call_id": "call", "name": "echo", "arguments": ""}
            yield {"type": "response.output_item.added", "sequence_number": 0, "output_index": 0, "item": item}
            yield {"type": "response.function_call_arguments.delta", "sequence_number": 1, "output_index": 0, "item_id": "item", "delta": '{"value":'}
            yield {"type": "response.function_call_arguments.delta", "sequence_number": 2, "output_index": 0, "item_id": "item", "delta": '"a"}'}
            yield {"type": "response.function_call_arguments.done", "sequence_number": 3, "output_index": 0, "arguments": '{"value":"a"}'}
            yield {"type": "response.completed", "sequence_number": 4, "response": {
                "status": "completed", "output": [{**item, "arguments": '{"value":"a"}', "status": "completed"}]}}
        async def close(self):
            self.closed = True
    stream = Stream()
    async def create(**kwargs):
        assert kwargs["stream"] is True
        return stream
    adapter = OpenAIProvider(provider="openai", api_key="test", base_url="https://provider.example/v1",
        endpoint="responses", client=SimpleNamespace(responses=SimpleNamespace(create=create)))
    async def collect():
        return [event async for event in adapter.stream(request())]
    events = asyncio.run(collect())
    assert [event.call_id for event in events if event.type == "tool.arguments.delta"] == ["call", "call"]
    assert events[-1].response.tool_calls[0].args == {"value": "a"}
    assert stream.closed


@pytest.mark.parametrize("provider", ["openai", "openrouter", "nvidia", "xai", "custom", "anthropic"])
def test_direct_sdks_preserve_auth_headers_query_and_disable_sdk_retries(monkeypatch, provider):
    options = []
    class Client:
        def __init__(self, **kwargs):
            options.append(kwargs)
    monkeypatch.setattr("openai.AsyncOpenAI", Client)
    monkeypatch.setattr("anthropic.AsyncAnthropic", Client)
    common = dict(provider=provider, api_key="fixture-credential", base_url="https://provider.example",
        headers={"X-Project": "one"}, query={"api-version": "fixture"}, timeout=7)
    if provider == "anthropic":
        adapter = AnthropicProvider(endpoint="anthropic_messages", **common)
    else:
        adapter = OpenAIProvider(endpoint="responses" if provider == "openai" else "chat_completions", **common)
    adapter._client()
    assert options == [dict(api_key="fixture-credential", base_url="https://provider.example",
        default_headers={"X-Project": "one"}, default_query={"api-version": "fixture"}, timeout=7, max_retries=0)]
