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
@pytest.mark.parametrize("invalid_fields", [{"level": 4}, {"paths": []}, {"paths": ["app.py", "other.py"]}])
def test_mode_decision_corrects_invalid_file_read(tmp_path, native_profile, monkeypatch, strategy, invalid_fields):
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
            evidence = {"tool": "read_file", "level": 1, "paths": ["app.py"], "reason": "Reported file"}
            if len(requests) == 1:
                evidence.update(invalid_fields)
            args = {"decision": {"action": "inspect", "reason": "Read the reported location", "evidence": evidence}}
            if request.tools:
                return ModelResponse(tool_calls=[ToolCall(id="read-decision", name="record_decision", args=args)])
            return ModelResponse(content=json.dumps(args))

    (tmp_path / "app.py").write_text("target\n", encoding="utf-8")
    service = PlanService(tmp_path, llm=engine(native_profile, monkeypatch, Adapter()))
    decision = service.next_decision("debug", {"request": "Inspect app.py"})

    assert len(requests) == 2
    assert "put exactly one repository-relative file in paths" in requests[0].messages[0].content
    assert "The decision was invalid." in requests[1].messages[-1].content
    assert decision.evidence.paths == ("app.py",)
    assert decision.evidence.level == 1
    assert service.inspect(decision.evidence)["content"] == "target\n"


@pytest.mark.parametrize("invalid_fields,detail", [
    ({"level": 3}, "The next permitted level is 2."),
    ({"prior_evidence": []}, "Levels above 1 require prior_evidence IDs."),
    ({"prior_evidence": ["fixture-rejected-id"]}, "Use only prior_evidence IDs from evidence_levels."),
    ({"prior_evidence": ["evidence-2"]}, "cite at least one prior_evidence ID from level 1."),
])
def test_mode_decision_corrects_investigation_scope(tmp_path, native_profile, monkeypatch, invalid_fields, detail):
    import json

    from mana_agent.agent.plan_service import InvestigationScope, PlanService

    requests = []

    class Adapter:
        async def generate(self, request):
            requests.append(request)
            evidence = {
                "tool": "repo_search", "level": 2, "paths": ["app.py"], "query": "target",
                "prior_evidence": ["evidence-1"], "reason": "Follow up on the earlier observation",
            }
            if len(requests) == 1:
                evidence.update(invalid_fields)
            return ModelResponse(content=json.dumps({"decision": {
                "action": "inspect", "reason": "Inspect the local handler", "evidence": evidence,
            }}))

    service = PlanService(tmp_path, llm=engine(native_profile, monkeypatch, Adapter()))
    # An older result can fall outside the detailed history window while its ID still applies.
    recent = [{"id": "evidence-2", "level": 0, "result": {}}]
    scope = InvestigationScope({"evidence-1": 1, "evidence-2": 0})
    decision = service.next_decision("debug", {"request": "Inspect app.py", "evidence": recent},
                                     investigation_scope=scope)

    assert len(requests) == 2
    assert requests[0].response_schema == requests[1].response_schema
    assert "validation_context" not in requests[0].parameters
    data = json.loads(requests[0].messages[1].content)
    assert data["evidence"] == recent
    assert data["investigation_scope"] == {
        "highest_observed_level": 1, "max_next_level": 2, "evidence_levels": scope.evidence_levels,
    }
    correction = requests[1].messages[-1].content
    assert detail in correction
    assert "fixture-rejected-id" not in correction
    assert decision.evidence.level == 2
    assert decision.evidence.prior_evidence == ("evidence-1",)


def test_mode_decision_reports_only_safe_fields_after_correction_fails(tmp_path, native_profile, monkeypatch):
    import json

    from mana_agent.agent.plan_service import PlanService

    request_log = []

    class Adapter:
        async def generate(self, request):
            request_log.append(request)
            return ModelResponse(content=json.dumps({"decision": {
                "action": "inspect", "reason": "Inspect evidence",
                "evidence": {"tool": "read_file", "level": 1, "paths": ["a.py"], "reason": "Evidence",
                             "unknown-private-key": "rejected-private-value"},
            }}))

    model = engine(native_profile, monkeypatch, Adapter())
    with pytest.raises(RuntimeError, match=r"ModeDecision.*decision\..*evidence\..*extra forbidden") as exc:
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
