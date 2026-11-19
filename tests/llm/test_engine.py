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


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("corrected", [False, True])
def test_entry_route_reports_schema_rejection_separately_from_request_failure(native_profile, monkeypatch, stream, corrected):
    from mana_agent.gateway.entry_routing import (
        EntryRouteContext,
        EntryRouteRegistry,
        EntryRouter,
        EntryRoutingError,
        RouteAvailability,
        RouteRegistration,
    )
    from mana_agent.llm.messages import ToolCall
    from mana_agent.llm.types import StreamEvent

    native_profile.facts.pop("native_structured_output")
    activity = []
    requests = []

    class Adapter:
        def response(self, request):
            requests.append(request)
            args = {"reason": "Answer the greeting."}
            if len(requests) == 2 and corrected:
                args.update(route="conversation", confidence=0.9, required_sources=["none"])
            return ModelResponse(tool_calls=[ToolCall(id=f"route-{len(requests)}", name="record_decision", args=args)])

        async def generate(self, request):
            return self.response(request)

        async def stream(self, request):
            yield StreamEvent(type="completed", request_id=request.request_id, sequence=1, response=self.response(request))

    monkeypatch.setattr("mana_agent.llm.engine._emit_inference_activity",
                        lambda event, **fields: activity.append((event, fields)))
    model = engine(native_profile, monkeypatch, Adapter())
    model.parameters["stream"] = stream
    registry = EntryRouteRegistry()
    registry.register(RouteRegistration(
        "conversation", "Direct reply", lambda: RouteAvailability(available=True),
    ))
    router = EntryRouter(llm=model, registry=registry)
    context = EntryRouteContext(session_id="s", conversation_id="s", turn_id="t")
    if corrected:
        assert router.route(user_prompt="Hello", context=context).route == "conversation"
    else:
        with pytest.raises(EntryRoutingError, match="Decision still invalid after one correction"):
            router.route(user_prompt="Hello", context=context)

    assert len(requests) == 2
    assert requests[0].tools == requests[1].tools
    assert "_decision_correction_available" not in requests[0].parameters
    assert "Return JSON only" not in requests[0].messages[0].content
    assert "Required top-level fields are route, confidence, reason, required_sources" in requests[0].messages[-1].content
    assert "missing confidence, required_sources, route" in requests[1].messages[-1].content
    rejected = next(fields for event, fields in activity if event == "model.failed")
    assert rejected["operation_id"] == requests[0].request_id
    assert rejected["title"] == "Decision needs correction"
    assert rejected["status"] == "rejected"
    event, last = activity[-1]
    assert last["operation_id"] == requests[1].request_id
    assert (event, last["status"]) == (("model.completed", "success") if corrected else ("model.failed", "failed"))


@pytest.mark.parametrize("strategy", ["native", "tool", "text"])
@pytest.mark.parametrize("corrected", [True, False])
def test_entry_route_reason_correction_stays_on_selected_strategy(
    native_profile, monkeypatch, strategy, corrected,
):
    import json

    from mana_agent.gateway.entry_routing import (
        EntryRouteContext,
        EntryRouteRegistry,
        EntryRouter,
        EntryRoutingError,
        RouteAvailability,
        RouteRegistration,
    )
    from mana_agent.llm.messages import ToolCall

    if strategy != "native":
        native_profile.facts.pop("native_structured_output")
    if strategy == "text":
        native_profile.facts.pop("tool_calling")
    requests = []

    class Adapter:
        async def generate(self, request):
            requests.append(request)
            reason = "  Answer the greeting.  " if corrected and len(requests) > 1 else " \t\n"
            decision = {
                "route": "conversation", "confidence": 0.9,
                "required_sources": ["none"], "reason": reason,
            }
            assert bool(request.response_schema) == (strategy == "native")
            if strategy == "tool":
                assert [tool["name"] for tool in request.tools] == ["record_decision"]
                return ModelResponse(tool_calls=[
                    ToolCall(id="routing-decision", name="record_decision", args=decision),
                ])
            assert not request.tools
            return ModelResponse(content=json.dumps(decision))

    registry = EntryRouteRegistry()
    registry.register(RouteRegistration(
        "conversation", "Direct reply", lambda: RouteAvailability(available=True),
    ))
    router = EntryRouter(llm=engine(native_profile, monkeypatch, Adapter()), registry=registry)
    context = EntryRouteContext(session_id="s", conversation_id="s", turn_id="t")

    if corrected:
        assert router.route(user_prompt="Hello", context=context).reason == "Answer the greeting."
    else:
        with pytest.raises(EntryRoutingError, match=r"Decision still invalid after one correction.*reason"):
            router.route(user_prompt="Hello", context=context)

    assert len(requests) == 2
    assert "reason: string too short" in requests[1].messages[-1].content


@pytest.mark.parametrize("case", ["expired_profile", "expired_limits", "missing_output_after_refresh"])
def test_entry_route_refreshes_capability_evidence(tmp_path, native_profile, monkeypatch, case):
    import hashlib
    from datetime import datetime, timedelta, timezone

    from mana_agent.config.user_config import save_model_cache
    from mana_agent.gateway.entry_routing import _invoke_routing_model
    from mana_agent.llm.capabilities import ModelProfile, metadata_facts
    from mana_agent.llm.discovery import ModelDiscoveryService
    from mana_agent.llm.registry import GLOBAL_CAPABILITY_REGISTRY, metadata_binding

    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    model = NativeLLMEngine(
        provider="openrouter", model="deepseek/deepseek-v4.1-flash", api_key="fixture-key",
        base_url="https://provider.example/api/v1", endpoint="chat_completions",
    )
    empty_revision = hashlib.sha256(b"{}").hexdigest()
    binding = metadata_binding(model.provider, model.base_url, hashlib.sha256(b"fixture-key").hexdigest(),
                               empty_revision, empty_revision)
    record = {
        "id": model.model,
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "context_length": 65536,
        "top_provider": {"max_completion_tokens": 8192},
        "supported_parameters": ["structured_outputs"],
    }
    facts = metadata_facts(record, source="provider", scope=binding)
    expired = datetime.now(timezone.utc) - timedelta(days=1)
    for name, fact in facts.items():
        if case != "expired_limits" or name in {"context_window", "max_output_tokens"}:
            fact.observed_at = expired - timedelta(days=1)
            fact.expires_at = expired
    GLOBAL_CAPABILITY_REGISTRY.merge(ModelProfile(
        provider=model.provider, model_id=model.model, endpoint=model.base_url, configuration_fingerprint=binding,
    ), facts)
    # The model picker can have a recent list without endpoint-bound capability evidence.
    save_model_cache(model.provider, model.base_url, [record])
    assert model.profile().is_stale("max_output_tokens")
    assert model.profile().is_stale("output_text") == (case != "expired_limits")

    calls = []

    async def fetch_records(self, **connection):
        calls.append("metadata")
        assert (connection["provider"], connection["base_url"], connection["api_key"]) == (
            model.provider, model.base_url, model.api_key,
        )
        if case == "missing_output_after_refresh":
            return [{**record, "architecture": {"input_modalities": ["text"]}, "supported_parameters": []}]
        return [record]

    async def public_catalog(self):
        return {}

    class Adapter:
        async def generate(self, request):
            calls.append("inference")
            assert (request.provider, request.model) == (model.provider, model.model)
            assert request.response_schema
            assert not request.tools
            return ModelResponse(content='{"route":"conversation","confidence":0.95,'
                                         '"reason":"Answer directly.","required_sources":["none"]}')

        async def close(self):
            pass

    monkeypatch.setattr(ModelDiscoveryService, "fetch_records", fetch_records)
    monkeypatch.setattr(ModelDiscoveryService, "catalog", public_catalog)
    monkeypatch.setattr(model, "_adapter", Adapter)

    if case == "missing_output_after_refresh":
        with pytest.raises(CapabilityError, match="output_text=UNKNOWN"):
            _invoke_routing_model(model, [UserMessage("Hello")])
        assert calls == ["metadata"]
    else:
        _, decision = _invoke_routing_model(model, [UserMessage("Hello")])
        assert decision["route"] == "conversation"
        assert calls == ["metadata", "inference"]


@pytest.mark.parametrize("case", ["newer_bound_cache", "expired_tools", "tools_missing_after_refresh"])
def test_entry_route_refreshes_required_tool_calling(tmp_path, native_profile, monkeypatch, case):
    import hashlib
    from datetime import datetime, timedelta, timezone

    from mana_agent.config.user_config import save_model_cache
    from mana_agent.gateway.entry_routing import _invoke_routing_model
    from mana_agent.llm.capabilities import ModelProfile, metadata_facts
    from mana_agent.llm.discovery import ModelDiscoveryService
    from mana_agent.llm.messages import ToolCall
    from mana_agent.llm.registry import GLOBAL_CAPABILITY_REGISTRY, metadata_binding

    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    model = NativeLLMEngine(
        provider="openrouter", model="anthropic/claude-haiku-5.5", api_key="fixture-key",
        base_url="https://provider.example/api/v1", endpoint="anthropic_messages",
    )
    empty_revision = hashlib.sha256(b"{}").hexdigest()
    binding = metadata_binding(model.provider, model.base_url, hashlib.sha256(b"fixture-key").hexdigest(),
                               empty_revision, empty_revision)
    record = {
        "id": model.model,
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "context_length": 65536,
        "top_provider": {"max_completion_tokens": 8192},
        "supported_parameters": ["tools", "structured_outputs"],
    }
    facts = metadata_facts(record, source="provider", scope=binding)
    expired = datetime.now(timezone.utc) - timedelta(hours=1)
    for name, fact in facts.items():
        if case == "newer_bound_cache" or name == "tool_calling":
            fact.observed_at = expired - timedelta(days=1)
            fact.expires_at = expired
    GLOBAL_CAPABILITY_REGISTRY.merge(ModelProfile(
        provider=model.provider, model_id=model.model, endpoint=model.base_url, configuration_fingerprint=binding,
    ), facts)
    if case == "newer_bound_cache":
        save_model_cache(model.provider, model.base_url, [{**record, "_mana_catalog_binding": binding}])
    else:
        assert model.profile().is_stale("tool_calling")
        assert model.profile().supports("output_text")
        assert model.profile().supports("max_output_tokens")

    calls = []

    async def fetch_records(self, **connection):
        calls.append("metadata")
        assert connection["provider"] == model.provider
        assert connection["base_url"] == model.base_url
        if case == "tools_missing_after_refresh":
            return [{**record, "supported_parameters": ["structured_outputs"]}]
        return [record]

    async def public_catalog(self):
        return {}

    class Adapter:
        async def generate(self, request):
            calls.append("inference")
            assert (request.provider, request.model) == (model.provider, model.model)
            assert [tool["name"] for tool in request.tools] == ["record_decision"]
            assert request.response_schema is None
            return ModelResponse(tool_calls=[ToolCall(id="route", name="record_decision", args={
                "route": "conversation", "confidence": 0.95, "reason": "Answer directly.",
                "required_sources": ["none"],
            })])

        async def close(self):
            pass

    monkeypatch.setattr(ModelDiscoveryService, "fetch_records", fetch_records)
    monkeypatch.setattr(ModelDiscoveryService, "catalog", public_catalog)
    monkeypatch.setattr(model, "_adapter", Adapter)

    if case == "tools_missing_after_refresh":
        with pytest.raises(CapabilityError, match="tool_calling=UNKNOWN"):
            _invoke_routing_model(model, [UserMessage("Hello")])
        assert calls == ["metadata"]
    else:
        _, decision = _invoke_routing_model(model, [UserMessage("Hello")])
        assert decision["route"] == "conversation"
        assert calls == (["inference"] if case == "newer_bound_cache" else ["metadata", "inference"])


def test_timeout_cancels_selected_request_without_other_backend(native_profile, monkeypatch):
    from mana_agent.coding.internal_engine import InternalCodingEngine

    activity = []
    monkeypatch.setattr("mana_agent.services.activity_context.emit_activity",
                        lambda event, **fields: activity.append((event, fields)))

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
    async def coding_request():
        return await InternalCodingEngine()._await_model_turn(
            "coding-task", 0, value, [UserMessage("hello")], None, asyncio.Event(),
        )
    with pytest.raises(InferenceTimeoutError):
        asyncio.run(coding_request())
    assert adapter.cancelled
    assert [event for event, _ in activity] == ["model.started", "model.failed"]
    started, failed = activity[0][1], activity[1][1]
    assert failed["operation_id"] == started["operation_id"]
    assert started["metadata"]["timeout_seconds"] == value.timeout
    assert failed["metadata"]["reason_code"] == "inference_timeout"
    assert failed["title"] == "Model request timed out"

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


@pytest.mark.parametrize(("args", "issue"), [
    ({"value": "wrong type"}, "arguments.value: expected integer"),
    ({"value": 1, "PRIVATE_FIXTURE": "unexpected property"}, "arguments: unregistered arguments"),
    ({"value": 1, "lookup": {"PRIVATE_FIXTURE": "wrong type"}}, "arguments.lookup.<key>: expected integer"),
])
def test_model_response_rejects_unregistered_and_invalid_tool_calls(native_profile, monkeypatch, args, issue):
    from mana_agent.llm.messages import ToolCall
    class Adapter:
        async def generate(self, request):
            return ModelResponse(tool_calls=[ToolCall(id="call", name="echo", args=args)])
    model = engine(native_profile, monkeypatch, Adapter()).with_tools([
        {"name": "echo", "parameters": {
            "type": "object", "required": ["value"], "additionalProperties": False,
            "properties": {
                "value": {"type": "integer"},
                "lookup": {"type": "object", "additionalProperties": {"type": "integer"}},
            },
        }},
    ])
    with pytest.raises(InvalidResponseError, match="registered schema") as error:
        model.generate_sync("hello")
    assert error.value.__suppress_context__
    assert "PRIVATE_FIXTURE" not in str(error.value)
    assert "for echo" in str(error.value)
    assert issue in error.value.public_detail


@pytest.mark.parametrize("classification", [" Observed ", "PRIVATE_FIXTURE"])
def test_native_investigation_report_uses_typed_validation(native_profile, monkeypatch, tmp_path, classification):
    from types import SimpleNamespace

    from mana_agent.coding.internal_engine import InternalCodingEngine
    from mana_agent.coding.models import CodingTask, WorkspaceContext
    from mana_agent.coding.programmatic import ProgrammaticExecutionSession
    from mana_agent.llm.errors import StructuredOutputError
    from mana_agent.llm.messages import ToolCall
    from mana_agent.repository_evidence.handoff import INVESTIGATION_REPORT_TOOL, parse_investigation_result
    from mana_agent.repository_evidence.models import CodingAssignment

    assignment = CodingAssignment(
        task_id="task", task_version=1, objective="Inspect app.py", task_kind="investigation",
        backend="internal", source_view=str(tmp_path), decision_id="selected", allowed_scope=("app.py",),
    )
    task = CodingTask(
        task_id="task", goal=assignment.objective, requires_repository_write=False,
        context={"repository_investigation": assignment.model_dump(mode="json")},
    )
    workspace = WorkspaceContext(repository_path=tmp_path, worktree_path=tmp_path, sandbox="readOnly")
    requests = []

    class Adapter:
        async def generate(self, request):
            requests.append(request)
            report = next(tool for tool in request.tools if tool["name"] == INVESTIGATION_REPORT_TOOL)
            definitions = report["parameters"]["$defs"]
            assert definitions["InvestigationFinding"]["properties"]["classification"]["enum"] == [
                "inferred", "hypothetical",
            ]
            file_citation, directory_citation = definitions["InvestigationCitation"]["anyOf"]
            assert {"start_line", "end_line"} <= set(file_citation["required"])
            assert "excerpt" not in file_citation["properties"]
            assert "start_line" not in directory_citation["properties"]
            assert report["parameters"]["additionalProperties"] is False
            limitations = report["parameters"]["properties"]["limitations"]
            assert limitations["type"] == "array"
            assert limitations["items"]["type"] == "string"
            return ModelResponse(tool_calls=[ToolCall(id="report", name=INVESTIGATION_REPORT_TOOL, args={
                "findings": [{
                    "category": "layout", "statement": "The app defines a value.", "confidence": 0.9,
                    "classification": classification,
                    "sources": [{"path": "app.py", "excerpt": "value = 1"}],
                    "architecture_relationships": [{"app": "defines value"}],
                }],
                "limitations": "Other files are outside the assigned scope.",
                "architecture_relationships": [{"source": "PRIVATE_FIXTURE", "relation": "calls", "target": "app"}],
                "PRIVATE_FIELD": {"task_id": "invented-task"},
            })])

    native = engine(native_profile, monkeypatch, Adapter())
    coding = InternalCodingEngine(model="test-model", provider="openai", programmatic_tool_calling_enabled=False)
    monkeypatch.setattr(coding, "_workspace_instruction", lambda workspace: "Inspect the assigned source.")
    monkeypatch.setattr(coding, "_build_tool_definitions", lambda dispatcher: [])
    session = ProgrammaticExecutionSession(task_id="task", dispatcher=SimpleNamespace(
        dispatch=lambda *args, **kwargs: pytest.fail("A terminal report must never dispatch a tool"),
    ))

    async def run():
        return await coding._run_model_tool_loop(task, workspace, session, asyncio.Event(), llm=native)

    if classification == "PRIVATE_FIXTURE":
        with pytest.raises(StructuredOutputError, match=r"findings\.0\.classification") as error:
            asyncio.run(run())
        assert "PRIVATE_FIXTURE" not in str(error.value)
        assert error.value.public_detail == str(error.value)
    else:
        report = parse_investigation_result(asyncio.run(run()))
        assert report.findings[0].classification == "inferred"
        assert report.findings[0].sources[0].excerpt == "value = 1"
        assert report.limitations == (
            "Other files are outside the assigned scope.",
            "Ignored 2 unsupported top-level report fields.",
        )
        assert "PRIVATE_" not in report.model_dump_json()
    assert len(requests) == 1
    assert session.state.tool_calls_executed == 0


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
def test_mode_decision_corrects_an_unavailable_coding_inspection(tmp_path, native_profile, monkeypatch, strategy):
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
                {"action": "inspect", "reason": "Check the source again", "requests": [
                    {"tool": "read_file", "level": 1, "paths": ["app.py"], "reason": "Explain the result"},
                ]}
                if len(requests) == 1 else {"action": "block", "reason": "The existing observations are insufficient"}
            )
            args = {"decision": action}
            if request.tools:
                return ModelResponse(tool_calls=[ToolCall(id="mode-decision", name="record_decision", args=args)])
            return ModelResponse(content=json.dumps(args))

    service = PlanService(tmp_path, llm=engine(native_profile, monkeypatch, Adapter()))
    decision = service.next_decision("debug", {"request": "Inspect app.py"}, inspection_available=False)

    assert len(requests) == 2
    assert requests[0].response_schema == requests[1].response_schema
    assert not json.loads(requests[0].messages[1].content)["inspection_available"]
    assert "Use the saved observations for the next decision." in requests[1].messages[-1].content
    assert "validation_context" not in requests[0].parameters
    assert decision.action == "block"


@pytest.mark.parametrize("strategy", ["native", "tool"])
def test_mode_decision_reports_only_safe_fields_after_correction_fails(tmp_path, native_profile, monkeypatch, strategy):
    import json

    from mana_agent.agent.plan_service import PlanService
    from mana_agent.llm.messages import ToolCall

    if strategy == "tool":
        native_profile.facts.pop("native_structured_output")

    request_log = []

    class Adapter:
        async def generate(self, request):
            request_log.append(request)
            arguments = {"decision": {
                "action": "investigate", "reason": "Inspect evidence", "objective": "Explain the behavior",
                "scope": "a.py", "unknown-private-key": "rejected-private-value",
            }}
            if strategy == "tool":
                return ModelResponse(tool_calls=[ToolCall(id="decision", name="record_decision", args=arguments)])
            return ModelResponse(content=json.dumps(arguments))

    model = engine(native_profile, monkeypatch, Adapter())
    with pytest.raises(RuntimeError, match=r"ModeDecision.*decision\..*extra forbidden") as exc:
        PlanService(tmp_path, llm=model).next_decision("plan", {"request": "Inspect the entry point"})
    assert len(request_log) == 2
    assert "unknown-private-key" not in str(exc.value)
    assert "rejected-private-value" not in str(exc.value)
    assert "rejected-private-value" not in request_log[1].messages[-1].content
    assert "decision.investigate.<key>: extra forbidden" in request_log[1].messages[-1].content


@pytest.mark.parametrize("recover", [False, True])
def test_mode_decision_tool_accepts_scoped_investigation_and_recovery(tmp_path, native_profile, monkeypatch, recover):
    from mana_agent.agent.plan_service import PlanService
    from mana_agent.llm.messages import ToolCall

    decision = {
        "action": "investigate", "reason": "Trace the handler and its caller.",
        "objective": "Explain the dispatch path.", "scope": "Trace two known files.",
        "paths": ["app.py", "src/caller.py"], "task_kind": "understanding",
    }
    if recover:
        decision["recovery"] = {
            "decision_id": "retry-saved", "task_id": "task-1", "action": "retry",
            "reason": "Continue the same investigation.", "same_task_retry_authorized": True,
            "safe_to_continue": True,
        }
    requests = []

    class Adapter:
        async def generate(self, request):
            requests.append(request)
            tool, = request.tools
            assert tool["name"] == "record_decision"
            assert request.response_schema is None
            fields = tool["parameters"]["$defs"]["InvestigateDecision"]["properties"]
            assert set(decision) <= set(fields)
            return ModelResponse(tool_calls=[ToolCall(id="decision", name="record_decision", args={"decision": decision})])

    model = NativeLLMEngine(provider="openrouter", model="anthropic/claude-haiku-5.5", api_key="fixture",
                            base_url="https://provider.example/api/v1", endpoint="anthropic_messages", adapter=Adapter())
    monkeypatch.setattr(model, "profile", lambda *args, **kwargs: native_profile)

    selected = PlanService(tmp_path, llm=model).next_decision("debug", {"request": "Explain dispatch"})

    assert len(requests) == 1
    assert selected.paths == ("app.py", "src/caller.py")
    assert selected.task_kind == "understanding"
    if recover:
        assert selected.recovery.task_id == "task-1"
    else:
        assert selected.recovery is None


@pytest.mark.parametrize("corrected", [False, True])
def test_mode_decision_tool_rejects_extra_top_level_arguments(tmp_path, native_profile, monkeypatch, corrected):
    from mana_agent.agent.plan_service import PlanService
    from mana_agent.llm.messages import ToolCall

    native_profile.facts.pop("native_structured_output")
    requests = []

    class Adapter:
        async def generate(self, request):
            requests.append(request)
            args = {"decision": {"action": "block", "reason": "More evidence is required."}}
            if len(requests) == 1 or not corrected:
                args["PRIVATE_EXTRA"] = "PRIVATE_VALUE"
            return ModelResponse(tool_calls=[ToolCall(id="decision", name="record_decision", args=args)])

    service = PlanService(tmp_path, llm=engine(native_profile, monkeypatch, Adapter()))
    if corrected:
        assert service.next_decision("debug", {"request": "Explain dispatch"}).action == "block"
    else:
        with pytest.raises(RuntimeError, match=r"arguments: unregistered arguments.*<key>: extra forbidden") as error:
            service.next_decision("debug", {"request": "Explain dispatch"})
        assert "PRIVATE_" not in str(error.value)

    assert len(requests) == 2
    assert requests[0].tools == requests[1].tools
    assert "only top-level field is decision" in requests[0].messages[0].content
    assert "Use only those top-level fields" in requests[0].messages[-1].content
    assert "arguments: unregistered arguments" in requests[1].messages[-1].content
    assert "PRIVATE_" not in requests[1].messages[-1].content


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


def test_anthropic_mode_decision_uses_messages_and_preserves_validation(tmp_path, native_profile, monkeypatch):
    import json
    from types import SimpleNamespace

    from mana_agent.agent.plan_service import PlanService
    from mana_agent.llm.engine import create_model

    requests = []

    class Client:
        def __init__(self, **options):
            assert options["base_url"] == "https://api.anthropic.com"
            assert options["api_key"] == "fixture-anthropic-key"
            self.messages = SimpleNamespace(create=self.create)

        async def create(self, **body):
            requests.append(body)
            objective = "x" * 2001 if len(requests) == 1 else "Trace the return value"
            decision = {"action": "investigate", "reason": "Read the caller",
                        "objective": objective, "scope": "app.py", "initial_repository_scope": "",
                        "paths": ["app.py"], "task_kind": "investigation", "recovery": None}
            return {"stop_reason": "end_turn", "content": [
                {"type": "text", "text": json.dumps({"decision": decision})},
            ]}

        async def close(self):
            pass

    monkeypatch.setattr("anthropic.AsyncAnthropic", Client)
    model = create_model(provider="anthropic", model="anthropic/claude-haiku-5-5",
                         api_key="fixture-anthropic-key", base_url="https://api.anthropic.com")
    monkeypatch.setattr(model, "profile", lambda *args, **kwargs: native_profile)
    service = PlanService(tmp_path, llm=model)

    decision = service.next_decision("plan", {"request": "Explain app.py"})

    assert model.endpoint == "anthropic_messages"
    assert decision.objective == "Trace the return value"
    assert len(requests) == 2
    for body in requests:
        assert body["model"] == "claude-haiku-5-5"
        assert body["system"]
        assert "response_format" not in body
        output_format = body["extra_body"]["output_config"]["format"]
        assert output_format["type"] == "json_schema"
        investigate = output_format["schema"]["$defs"]["InvestigateDecision"]
        assert investigate["properties"]["action"]["const"] == "investigate"
        objective = investigate["properties"]["objective"]
        assert "maxLength" not in objective
        assert "maxLength: 2000" in objective["description"]
    assert requests[0]["extra_body"] == requests[1]["extra_body"]


@pytest.mark.parametrize("status", [200, 400])
@pytest.mark.parametrize("entry_route", [False, True])
def test_openrouter_claude_uses_messages_decision_tool(tmp_path, native_profile, monkeypatch, status, entry_route):
    import json

    import httpx
    from anthropic import AsyncAnthropic

    from mana_agent.agent.plan_service import PlanService
    from mana_agent.gateway.entry_routing import _invoke_routing_model
    from mana_agent.llm.engine import create_model

    requests = []
    payload = (
        {"route": "conversation", "confidence": 0.95, "reason": "Answer directly.", "required_sources": ["none"]}
        if entry_route else {"decision": {"action": "block", "reason": "The supplied facts are insufficient."}}
    )

    def respond(request):
        requests.append(request)
        if status == 400:
            return httpx.Response(status, json={"error": {"message": "fixture rejection"}})
        return httpx.Response(200, json={
            "id": "message-1", "type": "message", "role": "assistant",
            "model": "anthropic/claude-sonnet-5.5", "stop_reason": "tool_use", "stop_sequence": None,
            "content": [{"type": "tool_use", "id": "toolu-decision", "name": "record_decision", "input": payload}],
            "usage": {"input_tokens": 40, "output_tokens": 10},
        })

    def sdk_client(**options):
        return AsyncAnthropic(**options, http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-unrelated-anthropic-key")
    monkeypatch.setattr("anthropic.AsyncAnthropic", sdk_client)
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **options: pytest.fail("Expected the native Messages client"))
    model = create_model(provider="openrouter", model="openrouter/anthropic/claude-sonnet-5.5",
                         api_key="fixture-openrouter-key", base_url="https://openrouter.ai/api/v1",
                         n=1, output_token_parameter="max_completion_tokens",
                         extra_body={"provider": {"order": ["Anthropic"]}})
    monkeypatch.setattr(model, "profile", lambda *args, **kwargs: native_profile)

    def decide():
        if entry_route:
            return _invoke_routing_model(model, [UserMessage("Hello")])[1]["route"]
        return PlanService(tmp_path, llm=model).next_decision("debug", {"request": "Explain app.py"}).action

    if status == 400:
        with pytest.raises(RuntimeError, match=r"openrouter rejected the request \(HTTP 400\)"):
            decide()
    else:
        assert decide() == ("conversation" if entry_route else "block")

    assert model.provider == "openrouter"
    assert model.endpoint == "anthropic_messages"
    assert len(requests) == 1
    request = requests[0]
    assert request.url == "https://openrouter.ai/api/v1/messages"
    assert request.headers["authorization"] == "Bearer fixture-openrouter-key"
    assert "x-api-key" not in request.headers
    body = json.loads(request.content)
    assert body["model"] == "anthropic/claude-sonnet-5.5"
    assert body["provider"] == {"order": ["Anthropic"]}
    assert body["max_tokens"] > 0
    assert "response_format" not in body
    assert "n" not in body
    assert "output_token_parameter" not in body
    assert "output_config" not in body
    tool, = body["tools"]
    assert tool["name"] == "record_decision"
    assert not tool.get("strict")
    if entry_route:
        assert tool["input_schema"]["properties"]["coding_workflow"]
    else:
        assert tool["input_schema"]["$defs"]["BlockDecision"]["properties"]["reason"]["minLength"] == 1


@pytest.mark.parametrize(("model", "mode", "expected"), [
    ("anthropic/claude-haiku-5.5", "auto", "anthropic_messages"),
    ("openrouter/~anthropic/claude-haiku-latest", "auto", "anthropic_messages"),
    ("anthropic/claude-haiku-5.5", "chat_completions", "chat_completions"),
    ("deepseek/deepseek-v4.1-flash", "auto", "chat_completions"),
])
def test_openrouter_model_protocol_agrees_with_capability_lookup(model, mode, expected, monkeypatch):
    from mana_agent.config.model_capabilities import resolve_model_capability
    from mana_agent.llm.engine import create_model
    from mana_agent.llm.registry import resolve_profile

    monkeypatch.setattr("mana_agent.config.user_config.get_setting",
                        lambda name, default=None: mode if name == "MANA_LLM_API_MODE" else default)

    engine = create_model(provider="openrouter", model=model, api_key="fixture-openrouter-key")

    assert engine.endpoint == expected
    assert resolve_profile("openrouter", model, catalog_records=[]).transport == expected
    assert resolve_model_capability("openrouter", model, catalog_records=[]).transport == expected
