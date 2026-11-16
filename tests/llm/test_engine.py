from __future__ import annotations
import asyncio
import pytest
from pydantic import BaseModel
from mana_agent.llm.capabilities import observation
from mana_agent.llm.engine import NativeLLMEngine
from mana_agent.llm.errors import CapabilityError, InvalidResponseError, InferenceTimeoutError
from mana_agent.llm.messages import UserMessage
from mana_agent.llm.types import ModelResponse

def engine(profile, monkeypatch, adapter):
    value = NativeLLMEngine(provider="openai", model="test-model", api_key="test", base_url="https://example.org", endpoint="responses", adapter=adapter)
    monkeypatch.setattr(value, "profile", lambda *args, **kwargs: profile)
    return value

def test_unknown_capability_blocks_before_provider_call(native_profile, monkeypatch):
    calls = []
    class Adapter:
        async def generate(self, request, **_native_options):
            calls.append(request)
    native_profile.facts.pop("tool_calling")
    value = engine(native_profile, monkeypatch, Adapter()).with_tools([{"name": "echo", "parameters": {"type": "object"}}])
    with pytest.raises(CapabilityError):
        value.generate_sync([UserMessage("hello")])
    assert calls == []

def test_invalid_native_structured_output_never_retries_as_text(native_profile, monkeypatch):
    class Decision(BaseModel):
        approved: bool
    class Adapter:
        calls = 0
        async def generate(self, request, **_native_options):
            self.calls += 1
            assert request.response_schema
            return ModelResponse(content="not JSON")
    adapter = Adapter()
    value = engine(native_profile, monkeypatch, adapter)
    with pytest.raises(ValueError):
        value.decide_sync("hello", Decision)
    assert adapter.calls == 1

def test_timeout_cancels_selected_request_without_other_backend(native_profile, monkeypatch):
    class Adapter:
        cancelled = False
        async def generate(self, request, **_native_options):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
    adapter = Adapter()
    value = engine(native_profile, monkeypatch, adapter)
    value.timeout = .001
    with pytest.raises(InferenceTimeoutError):
        value.generate_sync("hello")
    assert adapter.cancelled

def test_empty_response_does_not_prove_completion(native_profile, monkeypatch):
    class Adapter:
        async def generate(self, request, **_native_options):
            return ModelResponse()
    with pytest.raises(InvalidResponseError):
        engine(native_profile, monkeypatch, Adapter()).generate_sync("hello")


def test_conversation_receipt_cannot_replay_with_changed_history(native_profile, monkeypatch):
    from mana_agent.llm.types import ContinuationState
    class Adapter:
        calls = 0
        async def generate(self, request):
            self.calls += 1
            return ModelResponse(content="answer", continuation=ContinuationState(
                provider="openai", model="test-model", endpoint="responses", message_count=2))
    adapter = Adapter()
    model = engine(native_profile, monkeypatch, adapter)
    response = model.generate_sync("original")
    with pytest.raises(InvalidResponseError, match="history changed"):
        model.generate_sync([UserMessage("changed"), response.message], continuation=response.continuation)
    assert adapter.calls == 1


def test_native_chat_does_not_probe_unknown_features(native_profile, monkeypatch):
    from mana_agent.llm import probes
    async def forbidden(*args, **kwargs):
        pytest.fail("Chat must never probe capabilities")
    monkeypatch.setattr(probes, "run_capability_probe", forbidden)
    class Adapter:
        async def generate(self, request):
            return ModelResponse(content="answer")
    assert engine(native_profile, monkeypatch, Adapter()).generate_sync("hello").content == "answer"


def test_model_response_rejects_unregistered_and_invalid_tool_calls(native_profile, monkeypatch):
    from mana_agent.llm.messages import ToolCall
    class Adapter:
        async def generate(self, request):
            return ModelResponse(tool_calls=[ToolCall(id="call", name="echo", args={"value": "wrong type"})])
    model = engine(native_profile, monkeypatch, Adapter()).with_tools([
        {"name": "echo", "parameters": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}},
    ])
    with pytest.raises(InvalidResponseError, match="registered schema"):
        model.generate_sync("hello")


@pytest.mark.parametrize("strategy", ["native", "tool", "text"])
def test_mode_decision_repairs_semantic_failure_on_selected_strategy(tmp_path, native_profile, monkeypatch, strategy):
    import json

    from mana_agent.agent.plan_service import PlanService
    from mana_agent.llm.messages import ToolCall

    if strategy != "native":
        native_profile.facts.pop("native_structured_output")
    if strategy == "text":
        native_profile.facts.pop("tool_calling")

    request_log = []

    class Adapter:
        async def generate(self, request):
            request_log.append(request)
            # The first payload is structurally valid but recommends a missing choice.
            args = {"decision": {
                "action": "clarify", "reason": "This changes the public contract",
                "questions": [{"field_id": "compat", "prompt": "Which behavior?", "reason": "Required decision",
                               "choices": [{"value": "keep", "description": "Retain current behavior"},
                                           {"value": "change", "description": "Use new behavior"}],
                               "recommended_choice": "missing" if len(request_log) == 1 else "keep"}],
            }}
            if strategy == "tool":
                assert [tool["name"] for tool in request.tools] == ["record_decision"]
                assert request.tool_choice is None  # Named choice was not advertised.
                return ModelResponse(tool_calls=[ToolCall(id=f"call-{len(request_log)}", name="record_decision", args=args)])
            assert bool(request.response_schema) == (strategy == "native")
            assert not request.tools
            return ModelResponse(content=json.dumps(args))

    model = engine(native_profile, monkeypatch, Adapter())
    result = PlanService(tmp_path, llm=model).next_decision("plan", {"request": "Change the handler"})
    assert result.action == "clarify"
    assert result.questions[0].recommended_choice == "keep"
    assert result.questions[0].choices["keep"] == "Retain current behavior"
    assert len(request_log) == 2
    assert "recommended_choice must be an exact key in choices" in request_log[1].messages[-1].content


@pytest.mark.parametrize("strategy", ["native", "tool", "text"])
def test_mode_decision_corrects_a_repeated_investigation_handoff(tmp_path, native_profile, monkeypatch, strategy):
    import json

    from mana_agent.agent.plan_service import PlanService
    from mana_agent.llm.messages import ToolCall

    if strategy != "native":
        native_profile.facts.pop("native_structured_output")
    if strategy == "text":
        native_profile.facts.pop("tool_calling")

    requests = []

    class Adapter:
        async def generate(self, request):
            requests.append(request)
            action = (
                {"action": "investigate", "reason": "Check the source again", "objective": "Explain the old result",
                 "scope": "app.py"}
                if len(requests) == 1 else {"action": "block", "reason": "The existing observations are insufficient"}
            )
            args = {"decision": action}
            if request.tools:
                return ModelResponse(tool_calls=[ToolCall(id="mode-decision", name="record_decision", args=args)])
            return ModelResponse(content=json.dumps(args))

    service = PlanService(tmp_path, llm=engine(native_profile, monkeypatch, Adapter()))
    decision = service.next_decision("debug", {"request": "Inspect app.py"}, investigation_available=False)

    assert len(requests) == 2
    assert requests[0].response_schema == requests[1].response_schema
    assert not json.loads(requests[0].messages[1].content)["investigation_available"]
    assert "Investigation is unavailable for this decision." in requests[1].messages[-1].content
    assert "validation_context" not in requests[0].parameters
    assert decision.action == "block"


def test_mode_decision_reports_only_safe_fields_after_correction_fails(tmp_path, native_profile, monkeypatch):
    import json

    from mana_agent.agent.plan_service import PlanService

    request_log = []

    class Adapter:
        async def generate(self, request):
            request_log.append(request)
            return ModelResponse(content=json.dumps({"decision": {
                "action": "investigate", "reason": "Inspect evidence", "objective": "Explain the behavior",
                "scope": "a.py", "unknown-private-key": "rejected-private-value",
            }}))

    model = engine(native_profile, monkeypatch, Adapter())
    with pytest.raises(RuntimeError, match=r"ModeDecision.*decision\..*extra forbidden") as exc:
        PlanService(tmp_path, llm=model).next_decision("plan", {"request": "Inspect the entry point"})
    assert len(request_log) == 2
    assert "unknown-private-key" not in str(exc.value)
    assert "rejected-private-value" not in str(exc.value)
    assert "rejected-private-value" not in request_log[1].messages[-1].content


@pytest.mark.parametrize(("finish_reason", "detail"), [
    ("length", "The provider reached the output-token limit"),
    ("content_filter", "The provider's content filter stopped the response."),
])
def test_mode_decision_reports_why_the_provider_stopped_correction(
    tmp_path, native_profile, monkeypatch, finish_reason, detail,
):
    from mana_agent.agent.plan_service import PlanService
    from mana_agent.llm.providers.openai import OpenAIProvider

    provider = OpenAIProvider(provider="openrouter", api_key="test", base_url="https://example.org",
                              endpoint="chat_completions")
    requests, activity = [], []

    class Adapter:
        async def generate(self, request):
            requests.append(request)
            if len(requests) == 1:
                return ModelResponse(content='{"decision":{"action":"draft","reason":"Missing draft"}}')
            return provider.normalize({"choices": [{
                "finish_reason": finish_reason, "message": {"content": "fixture-protected-response"},
            }]}, request)

    monkeypatch.setattr("mana_agent.llm.engine._emit_inference_activity",
                        lambda event, **fields: activity.append((event, fields)))
    service = PlanService(tmp_path, llm=engine(native_profile, monkeypatch, Adapter()))
    with pytest.raises(RuntimeError, match="Invalid ModeDecision model decision") as exc:
        service.next_decision("plan", {"request": "Plan a fix for the handler"})

    assert len(requests) == 2
    assert requests[0].response_schema == requests[1].response_schema
    assert detail in str(exc.value)
    failed = next(fields for event, fields in activity if event == "model.failed")
    assert detail in failed["summary"]
    assert failed["operation_id"] == requests[-1].request_id
    assert "fixture-protected-response" not in str(exc.value)
    assert "fixture-protected-response" not in str(activity)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(("status", "reason_code"), [
    (400, "upstream_invalid_request"),
    (429, "upstream_rate_limited"),
])
def test_mode_decision_reports_provider_http_failure_without_replaying(
    tmp_path, native_profile, monkeypatch, stream, status, reason_code,
):
    from types import SimpleNamespace

    import httpx
    from openai import APIStatusError

    from mana_agent.agent.plan_service import PlanService
    from mana_agent.chat.activity import project_activity
    from mana_agent.llm.providers.openai import OpenAIProvider

    requests, activity = [], []
    protected = "fixture-provider-body-containing-private-request-text"

    async def create(**body):
        requests.append(body)
        response = httpx.Response(status, request=httpx.Request("POST", "https://example.org/chat/completions"))
        raise APIStatusError(protected, response=response, body={"error": {"message": protected}})

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    adapter = OpenAIProvider(provider="openrouter", api_key="test", base_url="https://example.org",
                              endpoint="chat_completions", client=client)
    model = NativeLLMEngine(provider="openrouter", model="anthropic/claude-haiku-5.5", api_key="test",
                            base_url="https://example.org", endpoint="chat_completions",
                            parameters={"stream": stream}, adapter=adapter)
    monkeypatch.setattr(model, "profile", lambda *args, **kwargs: native_profile)
    monkeypatch.setattr("mana_agent.llm.engine._emit_inference_activity",
                        lambda event, **fields: activity.append(project_activity({"type": event, **fields})))

    with pytest.raises(RuntimeError, match=rf"ModeDecision model decision failed.*HTTP {status}") as exc:
        PlanService(tmp_path, llm=model).next_decision("debug", {"request": "Explain the handler's return value"})

    assert len(requests) == 1
    assert requests[0]["model"] == model.model
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert not requests[0].get("tools")
    failed = next(event for event in activity if event["type"] == "model.failed")
    started = next(event for event in activity if event["type"] == "model.started")
    assert failed["operation_id"] == started["operation_id"]
    assert f"HTTP {status}" in failed["summary"]
    assert failed["metadata"]["reason_code"] == reason_code
    assert protected not in str(exc.value)
    assert protected not in str(activity)
