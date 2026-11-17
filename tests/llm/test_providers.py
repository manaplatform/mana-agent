from __future__ import annotations
import asyncio
import json
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


def test_openrouter_claude_mode_schema_preserves_local_constraints():
    from mana_agent.agent.plan_service import ModeDecision
    from mana_agent.llm.providers.openai import strict_schema

    adapter = OpenAIProvider(provider="openrouter", api_key="test", base_url="https://example.org",
                              endpoint="chat_completions")
    schema = ModeDecision.model_json_schema()
    selected = InferenceRequest(provider="openrouter", model="anthropic/claude-haiku-5.5",
                                messages=[UserMessage("Explain the handler")], response_schema=schema)
    body = adapter.encode(selected)
    output_format = body["response_format"]["json_schema"]
    wire = output_format["schema"]
    assert output_format["strict"]
    assert body["extra_body"]["provider"]["require_parameters"]
    assert wire["properties"]["decision"]["anyOf"] == schema["properties"]["decision"]["anyOf"]
    investigate = wire["$defs"]["InvestigateDecision"]
    assert investigate["properties"]["action"]["const"] == "investigate"
    assert "objective" in investigate["required"]
    objective = investigate["properties"]["objective"]
    assert "maxLength" not in objective
    assert "maxLength: 2000" in objective["description"]
    assert objective["pattern"] == r"\S"
    report = wire["$defs"]["ModeDebuggingReport"]["properties"]["local_evidence"]
    assert "maxItems" not in report
    assert report["minItems"] == 1
    assert "maxItems: 20" in report["description"]

    def normalize(objective):
        content = json.dumps({"decision": {"action": "investigate", "reason": "Read the caller",
            "objective": objective, "scope": "app.py", "initial_repository_scope": ""}})
        return adapter.normalize({"choices": [{"message": {"content": content}}]}, selected).content

    assert ModeDecision.model_validate_json(normalize("Trace the return value"), strict=True).decision.action == "investigate"
    with pytest.raises(StructuredOutputError):
        normalize("x" * 2001)

    assert selected.response_schema == ModeDecision.model_json_schema()
    other_model = selected.model_copy(update={"model": "openai/fixture"})
    assert adapter.encode(other_model)["response_format"]["json_schema"]["schema"] == strict_schema(schema)


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
    from mana_agent.llm.providers.openai import strict_schema, decode_structured
    schema = {"type": "object", "properties": {"arguments": {"type": "object", "additionalProperties": True}}, "required": ["arguments"]}
    wire = strict_schema(schema)
    assert wire["properties"]["arguments"]["type"] == "string"
    assert json.loads(decode_structured('{"arguments":"{\\"path\\":\\"README.md\\"}"}', schema)) == {"arguments": {"path": "README.md"}}
    with pytest.raises(ValueError):
        decode_structured('{"arguments":"invalid JSON"}', schema)


@pytest.mark.parametrize("action,fields", [
    ("inspect", {"requests"}),
    ("investigate", {"objective", "scope", "initial_repository_scope"}),
    ("clarify", {"questions"}),
    ("draft", {"draft"}),
    ("diagnose", {"root_cause", "cause_evidence", "uncertainty"}),
    ("repair", {"draft", "root_cause", "cause_evidence", "uncertainty"}),
    ("research", {"debugging_report"}),
    ("block", set()),
])
def test_mode_decision_wire_schema_exposes_only_the_selected_action_fields(action, fields):
    from mana_agent.agent.plan_service import ModeDecision
    from mana_agent.llm.providers.openai import strict_schema

    wire = strict_schema(ModeDecision.model_json_schema())
    assert wire["type"] == "object"
    assert wire["required"] == ["decision"]
    branches = [wire["$defs"][ref["$ref"].rsplit("/", 1)[-1]]
                for ref in wire["properties"]["decision"]["anyOf"]]
    branch = next(shape for shape in branches if shape["properties"]["action"]["const"] == action)
    assert set(branch["required"]) == {"action", "reason"} | fields
    assert branch["additionalProperties"] is False


def test_mode_decision_wire_schema_preserves_inspection_tool_alternatives():
    from mana_agent.agent.plan_service import ModeDecision
    from mana_agent.llm.providers.openai import strict_schema

    wire = strict_schema(ModeDecision.model_json_schema())
    items = wire["$defs"]["InspectDecision"]["properties"]["requests"]["items"]
    assert set(items) == {"anyOf"}
    tools = set()
    for ref in items["anyOf"]:
        branch = wire["$defs"][ref["$ref"].rsplit("/", 1)[-1]]
        tag = branch["properties"]["tool"]
        tools.update(tag.get("enum", [tag.get("const")]))
    assert tools == {"read_file", "ls", "list_files", "repo_search", "find_symbols", "call_graph"}


@pytest.mark.parametrize("payload", [
    {"action": "investigate", "reason": "Investigate the file"},
    {"action": "clarify", "reason": "Two choices", "questions": []},
    {"action": "draft", "reason": "Ready", "draft": None},
    {"action": "research", "reason": "Need documentation"},
    {"action": "block", "reason": "Evidence unavailable", "evidence": None},
])
def test_mode_decision_rejects_missing_and_unrelated_payloads(payload):
    from pydantic import ValidationError

    from mana_agent.agent.plan_service import ModeDecision
    from mana_agent.llm.providers.openai import decode_structured

    content = json.dumps({"decision": payload})
    with pytest.raises(StructuredOutputError, match="did not match the requested structured-output schema"):
        decode_structured(content, ModeDecision.model_json_schema())
    with pytest.raises(ValidationError):
        ModeDecision.model_validate_json(content, strict=True)


def test_native_investigation_tool_schema_separates_file_reads_from_discovery():
    from mana_agent.agent.repository_investigation import read_tool_schemas
    from mana_agent.multi_agent.tools.operation_policy import validate_arguments

    tools = read_tool_schemas()
    read = tools["read_file"]
    assert read["properties"]["level"]["enum"] == [1, 2, 3]
    assert read["properties"]["paths"]["minItems"] == read["properties"]["paths"]["maxItems"] == 1
    args = {"level": 1, "paths": ["app.py"], "reason": "Read the known file"}
    validate_arguments(args, read)
    for invalid in ({"level": 4}, {"paths": []}, {"paths": ["app.py", "other.py"]},
                    {"initial_repository_scope": "explicit_audit"}):
        with pytest.raises(ValueError):
            validate_arguments({**args, **invalid}, read)
    discovery = tools["repo_search"]
    validate_arguments({"level": 4, "query": "target", "reason": "Audit the repository"}, discovery)
    assert "default" not in discovery["properties"]["glob"]
    assert "require level 4" in discovery["properties"]["glob"]["description"]
    file_list = tools["list_files"]
    validate_arguments({"level": 1, "reason": "List the top-level files"}, file_list)
    assert "default" not in file_list["properties"]["glob"]
    assert "top-level files only" in file_list["properties"]["glob"]["description"]
    listing = tools["ls"]
    assert set(listing["properties"]) == {"level", "reason"}
    validate_arguments({"level": 1, "reason": "Locate the owning directory"}, listing)
    with pytest.raises(ValueError):
        validate_arguments({"level": 4, "reason": "Audit the repository"}, listing)
    assert all("initial_repository_scope" not in tool["properties"] for tool in tools.values())


@pytest.mark.parametrize("endpoint", ["responses", "chat_completions"])
def test_native_mode_decision_uses_typed_records_and_preserves_plan_digest(endpoint):
    from mana_agent.agent.mode import PlanRevision
    from mana_agent.agent.plan_service import DraftDecision, ModeDecision

    legacy = DraftDecision.model_validate({
        "action": "draft", "reason": "Ready",
        "draft": {"objectives": ["Fix the handler"], "findings": "Wrong return value", "scope": ["Handler"], "steps": [
            {"step_id": "fix", "instruction": "Fix the handler", "changes": {"app.py": ["update"]},
             "acceptance_criteria": ["Return the expected value"]},
            {"step_id": "verify", "instruction": "Run focused test", "depends_on": ["fix"],
             "verification": [{"argv": ["pytest"]}], "acceptance_criteria": ["The test passes"]},
        ]},
    })
    selected = request(response_schema=ModeDecision.model_json_schema())
    adapter = OpenAIProvider(provider="openai", api_key="test", base_url="https://example.org", endpoint=endpoint)
    encoded = adapter.encode(selected)
    shape = (encoded["text"]["format"]["schema"] if endpoint == "responses"
             else encoded["response_format"]["json_schema"]["schema"])
    operations = shape["$defs"]["PlanStep"]["properties"]["changes"]["items"]["properties"]["operations"]
    assert operations["type"] == "array"
    assert operations["minItems"] == 1
    assert operations["items"]["enum"] == ["add", "update", "delete"]

    def normalize(payload):
        content = json.dumps(payload)
        if endpoint == "responses":
            raw = {"status": "completed", "output": [
                {"type": "message", "content": [{"type": "output_text", "text": content}]},
            ]}
        else:
            raw = {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
        return adapter.normalize(raw, selected).content

    wire = {"decision": legacy.model_dump(mode="json")}
    fix, verify = wire["decision"]["draft"]["steps"]
    fix["changes"] = [{"path": "app.py", "operations": ["update"]}]
    verify["changes"] = []
    parsed = ModeDecision.model_validate_json(normalize(wire), strict=True).decision
    assert parsed == legacy
    revision = PlanRevision(plan_id="plan", revision=1, session_id="session", workspace_id="workspace",
                            repository_id="repo", repository_root="/project", goal="Fix the handler",
                            constraints={}, draft=legacy.draft)
    assert revision.model_copy(update={"draft": parsed.draft}).digest == revision.digest
    saved = revision.model_dump_json()
    assert json.loads(saved)["draft"]["steps"][0]["changes"] == {"app.py": ["update"]}
    assert PlanRevision.model_validate_json(saved, strict=True).digest == revision.digest

    fix["changes"] = '{"app.py":"update"}'
    with pytest.raises(StructuredOutputError, match="did not match the requested structured-output schema"):
        normalize(wire)

    fix["changes"] = [{"path": "app.py", "operations": ["update"]}, {"path": "app.py", "operations": ["delete"]}]
    with pytest.raises(ValueError, match="entries must use unique path values"):
        ModeDecision.model_validate_json(normalize(wire), strict=True)


def test_native_debug_report_maps_are_typed_records():
    from mana_agent.agent.plan_service import ModeDecision
    from mana_agent.llm.providers.openai import decode_structured

    legacy = ModeDecision.model_validate({"decision": {
        "action": "research", "reason": "Need dependency information",
        "debugging_report": {"unresolved_error": "Method unavailable", "local_evidence": ["Local method lookup failed"],
                             "dependency_versions": {"example": "1.0"},
                             "missing_external_information": "Method signature for the installed version"},
    }})
    wire = legacy.model_dump(mode="json")
    wire["decision"]["debugging_report"]["dependency_versions"] = [{"name": "example", "value": "1.0"}]
    wire["decision"]["debugging_report"]["runtime_environment"] = []
    content = decode_structured(json.dumps(wire), ModeDecision.model_json_schema())
    assert ModeDecision.model_validate_json(content, strict=True) == legacy


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
