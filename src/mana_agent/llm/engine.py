"""Native inference execution. No tools, routes, tasks or agent loops execute here."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any
from pydantic import BaseModel, ValidationError
from mana_agent.llm.capabilities import ModelProfile
from mana_agent.llm.errors import (
    CapabilityError, InvalidResponseError, InferenceTimeoutError, ProviderRequestError,
    StructuredOutputError, describe_decision_validation,
)
from mana_agent.llm.messages import messages_from_records, validate_conversation
from mana_agent.llm.registry import metadata_binding, resolve_profile
from mana_agent.llm.tools import tool_specification
from mana_agent.llm.types import InferenceRequest, ModelResponse, StreamEvent, ContinuationState


def _conversation_fingerprint(messages: list) -> str:
    records = [message.model_dump(mode="json") for message in messages]
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _emit_inference_activity(event_type: str, **fields: Any) -> None:
    from mana_agent.services.activity_context import current_activity_context, emit_activity
    context = current_activity_context()
    if context is not None and context.execution_lane == "coding":
        fields["backend"] = "internal"
    emit_activity(event_type, **fields)


class NativeLLMEngine:
    def __init__(self, *, provider: str, model: str, api_key: str, base_url: str,
                 endpoint: str = "", headers: dict[str, str] | None = None,
                 query: dict[str, str] | None = None, parameters: dict[str, Any] | None = None,
                 timeout: float = 600, max_attempts: int = 2, adapter: Any = None):
        if not provider or not model:
            raise ValueError("Inference requires an explicitly configured provider and model")
        import math
        if not math.isfinite(timeout) or timeout <= 0 or not 1 <= max_attempts <= 3:
            raise ValueError("Inference requires a finite positive timeout and one to three transport attempts")
        self.provider, self.model, self.api_key, self.base_url = provider, model, api_key, base_url
        self.selected_provider, self.model_name = provider, model
        self.endpoint, self.headers, self.query = endpoint or base_url, headers or {}, query or {}
        self.parameters, self.timeout, self.max_attempts = parameters or {}, timeout, max_attempts
        self.adapter = adapter
        self.tools: list[dict[str, Any]] = []
        self.tool_choice: str | dict[str, Any] | None = None
        self._tool_argument_validators: dict[str, Callable[[dict[str, Any]], None]] = {}
        self.context_cost_governor = None
        self._volatile_continuations: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    def with_tools(
        self, tools: list[Any], *, tool_choice: Any = None,
        argument_validators: dict[str, Callable[[dict[str, Any]], None]] | None = None,
    ) -> NativeLLMEngine:
        """Bind provider schemas and optional host-side argument contracts."""
        selected = copy.copy(self)
        selected.tools = [tool_specification(tool) for tool in tools]
        selected.tool_choice = tool_choice
        selected._tool_argument_validators = dict(argument_validators or {})
        names = {tool["name"] for tool in selected.tools if "name" in tool and "parameters" in tool}
        if selected._tool_argument_validators.keys() - names:
            raise ValueError("Argument validators must belong to offered function tools")
        if not all(callable(validator) for validator in selected._tool_argument_validators.values()):
            raise TypeError("Tool argument validators must be callable")
        return selected

    def update_model_assignment(self, provider: str, model: str, *, settings: Any,
                                model_configuration: dict[str, Any] | None = None) -> None:
        from mana_agent.config.inference_provider import resolve_inference_connection
        connection = resolve_inference_connection(settings, provider=provider)
        updated = create_model(provider=connection.provider, model=model, api_key=connection.api_key,
            base_url=connection.base_url, default_headers=connection.headers, model_configuration=model_configuration,
            timeout=self.timeout, max_retries=self.max_attempts - 1)
        updated.context_cost_governor = self.context_cost_governor
        self.__dict__.update(updated.__dict__)

    def profile(self, parameters: dict[str, Any] | None = None):
        scoped_parameters = dict(self.parameters if parameters is None else parameters)
        if parameters is not None:
            for name in ("stream", "max_output_tokens", "max_tokens", "max_completion_tokens"):
                scoped_parameters.pop(name, None)
                if name in self.parameters:
                    scoped_parameters[name] = self.parameters[name]
        return resolve_profile(self.provider, self.model, endpoint=self.base_url,
            api_version=str(self.headers.get("anthropic-version") or self.query.get("api-version") or ""),
            transport=self.endpoint, configuration={"parameters": scoped_parameters,
                "headers_revision": hashlib.sha256(json.dumps(self.headers, sort_keys=True).encode()).hexdigest(),
                "query_revision": hashlib.sha256(json.dumps(self.query, sort_keys=True).encode()).hexdigest()},
            credential_revision=hashlib.sha256(self.api_key.encode()).hexdigest())

    def _request(self, messages: Any, *, schema: Any = None, continuation: Any = None, **options: Any) -> InferenceRequest:
        options.pop("config", None)
        options.pop("callbacks", None)
        output_fields = ("max_tokens", "max_output_tokens", "max_completion_tokens")
        parameters = dict(self.parameters)
        if any(options.get(name) is not None for name in output_fields):
            for name in output_fields:
                parameters.pop(name, None)
        parameters.update(options)
        reserved = {"model", "messages", "input", "tools", "tool_choice", "response_format", "text", "previous_response_id", "conversation"}
        if reserved & parameters.keys():
            raise CapabilityError("Model identity, messages, tools, schemas and continuation must use native request fields")
        extra = parameters.get("extra_body") or {}
        if not isinstance(extra, dict) or (reserved | {"max_tokens", "max_output_tokens", "max_completion_tokens", "stream"}) & extra.keys():
            raise CapabilityError("extra_body cannot override native identity, history, tools, schema, stream or output-budget fields")
        native = messages_from_records(messages)
        validate_conversation(native)
        profile = self.profile(parameters)
        if continuation is not None:
            continuation = ContinuationState.model_validate(continuation)
            if (continuation.provider, continuation.model, continuation.endpoint) != (self.provider, self.model, self.endpoint):
                raise InvalidResponseError("Continuation belongs to a different provider/model/endpoint; reconcile before continuing")
            if continuation.message_count > len(native):
                raise InvalidResponseError("Continuation history is incomplete; reconcile its checkpoint")
            if not continuation.message_fingerprint or _conversation_fingerprint(native[:continuation.message_count]) != continuation.message_fingerprint:
                raise InvalidResponseError("Continuation history changed; reconcile the checkpoint before continuing")
            if continuation.configuration_fingerprint and continuation.configuration_fingerprint != profile.configuration_fingerprint:
                raise InvalidResponseError("Continuation configuration changed; reconcile the checkpoint before continuing")
        required = ["input_text"]
        if not parameters.get("modalities"):
            required.append("output_text")
        if self.tools:
            required.append("tool_calling")
            if any(tool.get("type") == "programmatic_tool_calling" for tool in self.tools):
                required.append("programmatic_tool_calling")
        if self.tool_choice is not None:
            if not self.tools:
                raise CapabilityError("A tool choice requires registered tools")
            required.append("tool_choice")
            mode = self.tool_choice if isinstance(self.tool_choice, str) else "function"
            if mode == "function":
                choice = self.tool_choice
                if not isinstance(choice, dict) or choice.get("type") != "function":
                    raise CapabilityError("A named tool choice must specify a registered function")
                function = choice.get("function")
                name = choice.get("name") or (function.get("name") if isinstance(function, dict) else None)
                if not any(tool.get("name") == name and name for tool in self.tools):
                    raise CapabilityError("The named tool choice is outside the registered tool set")
            elif mode not in {"auto", "required", "none"}:
                raise CapabilityError("Choose auto, required, none, or a registered function")
            if not profile.supports_value("tool_choice", mode):
                raise CapabilityError(f"Tool-choice mode {mode} lacks current evidence. Configure or probe it in model settings.")
        if schema is not None:
            required.append("native_structured_output")
        if parameters.get("reasoning_effort") is not None:
            required.extend(["reasoning", "reasoning_effort"])
            if not profile.supports_value("reasoning_effort", parameters["reasoning_effort"]):
                raise CapabilityError("Requested reasoning effort is absent from the selected model's capability evidence")
        reasoning = parameters.get("reasoning") or extra.get("reasoning")
        if reasoning is not None:
            required.append("reasoning")
            if not isinstance(reasoning, dict):
                raise CapabilityError("Reasoning configuration must be an explicit object")
            if reasoning.get("effort") is not None and not profile.supports_value("reasoning_effort", reasoning["effort"]):
                raise CapabilityError("Requested reasoning effort lacks current evidence")
            if reasoning.get("summary") is not None:
                required.append("public_reasoning_summaries")
        if parameters.get("thinking") is not None or extra.get("thinking") is not None:
            required.extend(["reasoning", "thinking_control"])
        if parameters.get("parallel_tool_calls") is not None or extra.get("parallel_tool_calls") is not None:
            required.append("parallel_tool_calls")
        modalities = parameters.get("modalities") or extra.get("modalities") or []
        if not isinstance(modalities, list) or any(not isinstance(value, str) for value in modalities):
            raise CapabilityError("Requested output modalities must be a list of names")
        for modality in modalities:
            required.append(f"output_{modality}")
        for message in native:
            if isinstance(message.content, list):
                for block in message.content:
                    kind = block.get("type")
                    capability = {"text": "input_text", "input_text": "input_text", "image_url": "input_image", "input_image": "input_image", "image": "input_image", "input_audio": "input_audio", "audio": "input_audio", "file": "input_pdf", "input_file": "input_pdf", "document": "input_pdf", "video": "input_video"}.get(kind)
                    if capability is None:
                        raise CapabilityError(f"Unrecognized input modality {kind}")
                    required.append(capability)
        limits = [parameters[name] for name in output_fields if parameters.get(name) is not None]
        output = limits[0] if limits else 0
        if self.adapter is None:
            profile.require(*required)
        else:
            non_text_required = [c for c in required if c not in {"input_text", "output_text"}]
            if non_text_required:
                profile.require(*non_text_required)
            if not profile.supports("output_text") and not profile.supports("input_text"):
                profile.require(*required)
        if self.adapter is None:
            profile.require("context_window")
            if type(profile.context_window) is not int or profile.context_window <= 0:
                raise CapabilityError("The context limit must be an explicitly configured positive integer")
            profile.require("max_output_tokens")
            if profile.max_output_tokens is None:
                raise CapabilityError("Output limit evidence must contain a positive integer")
            if not any(parameters.get(name) is not None for name in output_fields):
                from mana_agent.config.settings import Settings
                ratio = Settings().mana_context_default_output_ratio
                parameters["max_output_tokens"] = max(1, int(profile.max_output_tokens * ratio))
            limits = [parameters[name] for name in output_fields if parameters.get(name) is not None]
            if not limits or any(type(value) is not int or value < 1 for value in limits):
                raise CapabilityError("Output limits must be explicitly supported positive integers")
            if len(limits) != 1:
                raise CapabilityError("Specify one output-limit parameter for the selected transport")
            output = limits[0]
            if output > profile.max_output_tokens:
                raise CapabilityError("Requested output budget exceeds the discovered model limit")
        elif profile.max_output_tokens is not None and not any(parameters.get(name) is not None for name in output_fields):
            parameters["max_output_tokens"] = profile.max_output_tokens
            output = profile.max_output_tokens
        from mana_agent.context_cost.estimator import estimate_value_tokens
        serialized = [message.model_dump(mode="json") for message in native]
        if continuation is not None and continuation.mode == "stateless":
            serialized = [*continuation.items, *serialized[continuation.message_count:]]
        response_schema = schema.model_json_schema() if isinstance(schema, type) and issubclass(schema, BaseModel) else schema
        tokens = estimate_value_tokens(serialized) + estimate_value_tokens(self.tools)
        if response_schema:
            tokens += estimate_value_tokens(response_schema)
        input_limit = profile.value("max_input_tokens")
        if profile.context_window is not None and type(profile.context_window) is int and profile.context_window > 0:
            if tokens + output > profile.context_window or (type(input_limit) is int and tokens > input_limit):
                raise CapabilityError("Request exceeds the selected model context budget; compact context before inference")
        elif type(input_limit) is int and tokens > input_limit:
            raise CapabilityError("Request exceeds the selected model context budget; compact context before inference")
        if parameters.get("n", 1) not in (None, 1):
            raise CapabilityError("Native inference requires one response choice")
        from mana_agent.services.activity_context import current_activity_context
        activity = current_activity_context()
        return InferenceRequest(provider=self.provider, model=self.model, messages=native, tools=self.tools,
            tool_choice=self.tool_choice, response_schema=response_schema,
            parameters=parameters, continuation=continuation, request_id=uuid.uuid4().hex,
            configuration_fingerprint=profile.configuration_fingerprint,
            task_id=str(getattr(activity, "task_id", "") or ""), attempt_id=str(getattr(activity, "attempt_id", "") or ""))

    def _adapter(self):
        if self.adapter is not None:
            return self.adapter
        from mana_agent.llm.providers.openai import OpenAIProvider
        from mana_agent.llm.providers.anthropic import AnthropicProvider
        adapter = {"responses": OpenAIProvider, "chat_completions": OpenAIProvider, "anthropic_messages": AnthropicProvider}.get(self.endpoint)
        if adapter is None:
            raise CapabilityError(f"No native adapter registered for protocol {self.endpoint}")
        options = dict(provider=self.provider, api_key=self.api_key, base_url=self.base_url,
            endpoint=self.endpoint, headers=self.headers, query=self.query, timeout=self.timeout)
        if self.endpoint == "anthropic_messages":
            options["volatile_continuations"] = self._volatile_continuations
        return adapter(**options)

    def _reserve(self, request: InferenceRequest) -> str:
        governor = self.context_cost_governor
        if governor is None or not governor.enabled:
            return ""
        from mana_agent.context_cost.models import ContextSegment
        from mana_agent.context_cost.estimator import estimate_value_tokens
        segments = [ContextSegment(kind="system" if m.role in {"system", "developer"} else "user" if m.role == "user" else "tool_result" if m.role == "tool" else "history",
            content=m.model_dump(mode="json"), token_estimate=estimate_value_tokens(m.model_dump(mode="json")),
            protected=True, source_id=f"message:{index}") for index, m in enumerate(request.messages)]
        segments.extend(ContextSegment(kind="schema", content=tool, token_estimate=estimate_value_tokens(tool), protected=True, source_id=f"tool:{index}") for index, tool in enumerate(request.tools))
        if request.response_schema:
            segments.append(ContextSegment(kind="schema", content=request.response_schema,
                token_estimate=estimate_value_tokens(request.response_schema), protected=True, source_id="response-schema"))
        identity, _ = governor.before_model_call(segments, provider=request.provider, model=request.model,
            task_id=request.task_id, observation_boundary="native_inference",
            expected_output_tokens=next((int(request.parameters[k]) for k in ("max_output_tokens", "max_tokens", "max_completion_tokens") if request.parameters.get(k)), None))
        return identity

    def _complete(self, request: InferenceRequest, response: ModelResponse, reservation: str, started: float) -> None:
        if response.refusal:
            detail = "Selected provider refused the request"
            error = InvalidResponseError(detail, public_detail=detail)
            error.provider_usage, error.response_accepted = response.usage.model_dump(), True
            raise error
        try:
            from mana_agent.multi_agent.tools.operation_policy import validate_arguments
            offered = {tool["name"]: tool["parameters"] for tool in request.tools if tool.get("name") and "parameters" in tool}
            if offered.keys() == {"record_decision"} and "record_decision" in self._tool_argument_validators:
                calls = response.tool_calls
                if len(calls) != 1 or calls[0].name != "record_decision":
                    detail = "Return exactly one record_decision call."
                    raise StructuredOutputError(detail, public_detail=detail)
            if not response.content and not response.tool_calls and not response.protocol_pending:
                raise InvalidResponseError("Empty inference output is not completion evidence",
                                           public_detail="The provider returned an empty response.")
            seen = {call.id for message in request.messages for call in message.tool_calls}
            for call in response.tool_calls:
                if call.id in seen:
                    detail = "The provider reused a tool call ID."
                    raise InvalidResponseError(detail, public_detail=detail, reason_code="tool_call_id_reused")
                if call.name not in offered:
                    detail = "The provider called a tool outside the offered set."
                    raise InvalidResponseError(detail, public_detail=detail, reason_code="tool_not_offered")
                validator = self._tool_argument_validators.get(call.name)
                try:
                    if validator is not None:
                        validator(call.args)
                    else:
                        validate_arguments(call.args, offered[call.name], redact_keys=True)
                except StructuredOutputError:
                    raise
                except ValueError as exc:
                    detail = f"Provider tool arguments violate the registered schema for {call.name}"
                    if validator is None:
                        detail += f" ({exc})"
                    raise StructuredOutputError(detail, public_detail=detail) from None
                seen.add(call.id)
        except InvalidResponseError as exc:
            exc.provider_usage, exc.response_accepted = response.usage.model_dump(), True
            raise
        if reservation:
            self.context_cost_governor.record_model_call(reservation, usage=response.usage.model_dump(),
                provider=request.provider, model=request.model, task_id=request.task_id,
                estimated_input=[m.content for m in request.messages])
        if response.continuation is not None:
            # Output-budget defaults are not caller configuration changes.
            response.continuation.configuration_fingerprint = request.configuration_fingerprint
            response.continuation.message_fingerprint = _conversation_fingerprint([*request.messages, response.message])
        from mana_agent.evals.recorder import record_current
        record_current("model.call", {"boundary": "native_inference", "provider": request.provider,
            "model": request.model, "task_id": request.task_id, "request_id": request.request_id,
            "usage": response.usage.model_dump(), "latency_seconds": time.perf_counter() - started})
        _emit_inference_activity("model.completed", title="Model responded", status="success", operation_id=request.request_id,
                      duration_ms=int((time.perf_counter() - started) * 1000),
                      metadata={"provider": request.provider, "model": request.model,
                                "duration_ms": int((time.perf_counter() - started) * 1000)})
        if response.public_summaries and self.profile().supports("public_reasoning_summaries"):
            for index, summary in enumerate(response.public_summaries):
                _emit_inference_activity("agent.summary", title="Reasoning summary", summary=summary, status="success",
                    event_id=f"summary-{response.id}-{index}", operation_id=request.request_id,
                    metadata={"summary_source": "provider", "summary_trusted": True, "summary_complete": True,
                        "provider": self.provider, "transport": "direct_responses" if self.endpoint == "responses" else self.endpoint})

    def _failed(
        self, request: InferenceRequest, reservation: str, exc: BaseException, started: float,
        *, decision_correction_available: bool = False,
    ) -> ProviderRequestError | None:
        if reservation:
            if getattr(exc, "response_accepted", False):
                self.context_cost_governor.record_model_call(reservation, usage=getattr(exc, "provider_usage", None),
                    provider=request.provider, model=request.model, task_id=request.task_id,
                    estimated_input=[message.content for message in request.messages])
            else:
                self.context_cost_governor.release_reservation(reservation, reason=f"Inference failed: {type(exc).__name__}")
        request_error = None
        status_code = getattr(exc, "status_code", None)
        if not isinstance(exc, ProviderRequestError) and type(status_code) is int and 400 <= status_code < 600:
            from mana_agent.integrations.provider_failure import classify_http_status

            failure = classify_http_status(status_code, provider=request.provider, model=request.model,
                operation="native_inference", endpoint=self.endpoint)
            request_error = ProviderRequestError(failure.safe_message, provider=request.provider,
                status_code=status_code, failure=failure)
        error = request_error or exc
        metadata = {"provider": request.provider, "model": request.model, "transport": self.endpoint,
                    "error_type": type(error).__name__,
                    "duration_ms": int((time.perf_counter() - started) * 1000)}
        title, status = "Model request failed", "failed"
        summary = error.public_detail if isinstance(error, (InvalidResponseError, ProviderRequestError)) else ""
        if isinstance(error, StructuredOutputError) and getattr(error, "response_accepted", False):
            title = "Model response failed validation"
            metadata["reason_code"] = "response_schema_invalid"
            if decision_correction_available:
                title, status = "Decision needs correction", "rejected"
        elif isinstance(error, InvalidResponseError):
            if getattr(error, "response_accepted", False):
                title = "Model response rejected"
                metadata["reason_code"] = error.reason_code or "provider_response_invalid"
            elif error.reason_code:
                metadata["reason_code"] = error.reason_code
        elif isinstance(error, asyncio.TimeoutError):
            title = "Model request timed out"
            summary = f"The provider response did not complete within the {self.timeout:g}-second request deadline."
            metadata["reason_code"] = "inference_timeout"
        elif isinstance(error, asyncio.CancelledError):
            title, status = "Model request cancelled", "cancelled"
        if isinstance(error, ProviderRequestError) and error.failure is not None:
            metadata["reason_code"] = error.failure.error_code
        _emit_inference_activity("model.failed", title=title, status=status,
            operation_id=request.request_id, duration_ms=metadata["duration_ms"], metadata=metadata, summary=summary)
        return request_error

    async def _refresh_metadata(
        self, parameters: dict[str, Any] | None = None, *, required_capabilities: tuple[str, ...] = (),
    ) -> ModelProfile:
        selected = self.profile(parameters)
        required = ("input_text", "output_text", "context_window", "max_output_tokens", *required_capabilities)
        if self.adapter is not None or all(selected.supports(name) for name in required):
            return selected
        from mana_agent.llm.discovery import ModelDiscoveryService
        from mana_agent.config.user_config import load_model_cache
        from datetime import datetime, timedelta, timezone
        cached = load_model_cache(self.provider, self.base_url)
        if cached is not None and cached.created_at:
            try:
                created = datetime.fromisoformat(cached.created_at)
                binding = metadata_binding(self.provider, self.base_url, hashlib.sha256(self.api_key.encode()).hexdigest(),
                    hashlib.sha256(json.dumps(self.headers, sort_keys=True).encode()).hexdigest(),
                    hashlib.sha256(json.dumps(self.query, sort_keys=True).encode()).hexdigest())
                if (created.tzinfo and datetime.now(timezone.utc) - created < timedelta(hours=24)
                    and any(isinstance(record, dict) and record.get("_mana_catalog_binding") == binding for record in cached.models)):
                    return selected
            except ValueError:
                pass
        await ModelDiscoveryService().discover(provider=self.provider, base_url=self.base_url, api_key=self.api_key,
            default_headers=self.headers, query_params=self.query)
        return self.profile(parameters)

    async def generate(
        self, messages: Any, *, schema: Any = None, continuation: Any = None,
        _decision_correction_available: bool = False, **options: Any,
    ) -> ModelResponse:
        if options.get("stream", self.parameters.get("stream", False)):
            result = None
            async for event in self.stream(
                messages, schema=schema, continuation=continuation,
                _decision_correction_available=_decision_correction_available, **options,
            ):
                if event.type == "completed":
                    result = event.response
            if result is None:
                raise InvalidResponseError("Stream did not return a completed response")
            return result
        await self._refresh_metadata()
        request = self._request(messages, schema=schema, continuation=continuation, **options)
        reservation, started = self._reserve(request), time.perf_counter()
        _emit_inference_activity("model.started", title="Waiting for model", status="running", operation_id=request.request_id,
                      metadata={"provider": request.provider, "model": request.model, "timeout_seconds": self.timeout})
        try:
            for attempt in range(self.max_attempts):
                adapter = self._adapter()
                try:
                    response = await asyncio.wait_for(adapter.generate(request), timeout=self.timeout)
                    self._complete(request, response, reservation, started)
                    return response
                except Exception as exc:
                    # Only a connection failure before provider acceptance is safe to retry.
                    import httpx
                    cause = exc.__cause__ or exc
                    safe = isinstance(cause, (httpx.ConnectError, httpx.ConnectTimeout))
                    if not safe or request.tools and any(t.get("type") for t in request.tools) or attempt + 1 >= self.max_attempts:
                        raise
                    await asyncio.sleep(min(.25 * 2 ** attempt, 2))
                finally:
                    if self.adapter is None:
                        await adapter.close()
        except BaseException as exc:
            request_error = self._failed(request, reservation, exc, started,
                                         decision_correction_available=_decision_correction_available)
            if request_error is not None:
                raise request_error from None
            if isinstance(exc, asyncio.TimeoutError):
                raise InferenceTimeoutError("Selected inference request timed out; no alternate action was attempted") from exc
            raise
        raise InvalidResponseError("Inference attempt budget is empty")

    def generate_sync(self, messages: Any, **options: Any) -> ModelResponse:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.generate(messages, **options))
        # Legacy synchronous controllers can be called by an async host. Keep
        # that boundary synchronous while the provider uses its own event loop.
        # Context variables carry activity, accounting and authorization identity.
        import contextvars
        from concurrent.futures import ThreadPoolExecutor
        context = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mana-inference-sync") as pool:
            return pool.submit(context.run, lambda: asyncio.run(self.generate(messages, **options))).result()

    async def decide(
        self, messages: Any, schema: type[BaseModel] | dict[str, Any], *,
        repair_decision: bool = False, validation_context: dict[str, Any] | None = None,
        complex_schema: bool = False, correction_hint: str | Callable[[ValueError], str] = "", **options: Any,
    ) -> BaseModel | dict[str, Any]:
        from mana_agent.llm.messages import UserMessage
        from mana_agent.multi_agent.tools.operation_policy import validate_arguments

        require_native = options.pop("require_native_structured_output", False)
        # Use Claude's ordinary client tool for composite workflow decisions.
        # Local validation retains the full contract without compiling a provider grammar.
        use_tool = (complex_schema and not require_native and self.provider == "openrouter"
                    and self.endpoint == "anthropic_messages")
        required = ("native_structured_output",) if require_native else ("tool_calling",) if use_tool else ()
        # Refresh capability evidence before choosing or validating the request format.
        prof = await self._refresh_metadata(options.get("parameters"), required_capabilities=required)
        prof.require(*required)
        schema_dict = schema if isinstance(schema, dict) else schema.model_json_schema()
        decision_messages = messages_from_records(messages)
        native = not use_tool and prof.supports("native_structured_output")
        tool_based = not native and prof.supports("tool_calling")
        model = self
        if tool_based:
            def validate_decision_arguments(arguments: dict[str, Any]) -> None:
                try:
                    validate_arguments(arguments, schema_dict, redact_keys=True)
                except ValueError as exc:
                    detail = f"Provider tool arguments violate the registered schema for record_decision ({exc})"
                    if isinstance(schema, type) and issubclass(schema, BaseModel):
                        try:
                            schema.model_validate_json(json.dumps(arguments), strict=True, context=validation_context)
                        except ValidationError as error:
                            detail += ". " + describe_decision_validation(error, schema_dict)
                    raise StructuredOutputError(detail, public_detail=detail) from None

            record_tool = {
                "name": "record_decision",
                "description": "Submit the complete decision in the tool arguments.",
                "parameters": schema_dict,
            }
            choice = {"type": "function", "name": "record_decision"} if prof.supports_value("tool_choice", "function") else None
            model = self.with_tools([record_tool], tool_choice=choice,
                                    argument_validators={"record_decision": validate_decision_arguments})
            instruction = "Return exactly one record_decision call with the complete decision in its arguments."
            required_fields = schema_dict.get("required", [])
            if required_fields:
                instruction += f" Required top-level fields are {', '.join(required_fields)}."
                if schema_dict.get("additionalProperties") is False and set(required_fields) == set(schema_dict.get("properties", {})):
                    instruction += " Use only those top-level fields."
            decision_messages.append(UserMessage(instruction))
        if not native and not tool_based:
            if not (prof.supports("output_text") or prof.supports("input_text") or prof.supports("text_generation")):
                prof.require("output_text")
            decision_messages.append(UserMessage(
                "Return only a JSON object matching this schema.\n" + json.dumps(schema_dict, indent=2)
            ))

        attempts = 2 if repair_decision or (not native and not tool_based) else 1
        for attempt in range(attempts):
            try:
                response = await model.generate(
                    decision_messages, schema=schema if native else None,
                    _decision_correction_available=attempt + 1 < attempts, **options,
                )
            except StructuredOutputError as exc:
                failure = exc
            else:
                try:
                    if tool_based:
                        content = json.dumps(response.tool_calls[0].args)
                    else:
                        content = response.content.strip() if isinstance(response.content, str) else ""
                        if not native and content.startswith("```"):
                            lines = content.splitlines()
                            if len(lines) >= 2 and lines[-1].strip() == "```":
                                content = "\n".join(lines[1:-1]).strip()
                            elif lines[0].startswith("```"):
                                content = "\n".join(lines[1:]).strip()
                    if isinstance(schema, dict):
                        payload = json.loads(content)
                        validate_arguments(payload, schema_dict)
                        return payload
                    # These strategies all return JSON. Strict Python validation
                    # would reject JSON arrays that represent typed tuples.
                    return schema.model_validate_json(content, strict=True, context=validation_context)
                except ValueError as exc:
                    failure = exc

            if attempts == 1:
                raise failure
            if isinstance(failure, ValidationError):
                detail = describe_decision_validation(failure, schema_dict)
            elif isinstance(failure, StructuredOutputError):
                detail = str(failure)
            else:
                detail = "The response did not match the required JSON fields or types."
            if attempt + 1 >= attempts:
                raise StructuredOutputError(f"Decision still invalid after one correction (failed schema validation). {detail}") from None
            correction = (
                f"The decision was invalid. {detail}\n"
                "Correct the response for the original request using the same response format."
            )
            hint = correction_hint(failure) if callable(correction_hint) else correction_hint
            if hint:
                correction += f"\n{hint}"
            decision_messages.append(UserMessage(correction))
        raise AssertionError("Decision validation attempts were not configured")

    def decide_sync(self, messages: Any, schema: type[BaseModel] | dict[str, Any], **options: Any) -> BaseModel | dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.decide(messages, schema, **options))
        import contextvars
        from concurrent.futures import ThreadPoolExecutor
        context = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mana-inference-decide-sync") as pool:
            return pool.submit(context.run, lambda: asyncio.run(self.decide(messages, schema, **options))).result()

    async def stream(
        self, messages: Any, *, _decision_correction_available: bool = False, **options: Any,
    ) -> AsyncIterator[StreamEvent]:
        await self._refresh_metadata()
        request = self._request(messages, **options)
        self.profile(request.parameters).require("streaming", *(["tool_call_streaming"] if request.tools else []))
        reservation, started = self._reserve(request), time.perf_counter()
        adapter = None
        events = None
        completed = False
        deadline = time.monotonic() + self.timeout
        _emit_inference_activity("model.started", title="Waiting for model stream", status="running",
            operation_id=request.request_id, metadata={"provider": self.provider, "model": self.model,
                                                        "timeout_seconds": self.timeout})
        try:
            adapter = self._adapter()
            events = adapter.stream(request).__aiter__()
            while True:
                try:
                    event = await asyncio.wait_for(events.__anext__(), timeout=max(.001, deadline - time.monotonic()))
                except StopAsyncIteration:
                    break
                if completed:
                    raise InvalidResponseError("Provider emitted data after the terminal stream event")
                if event.type in {"text.delta", "tool.arguments.delta"}:
                    _emit_inference_activity("model.stream", title="Receiving model response", status="running",
                        summary="Receiving text" if event.type == "text.delta" else "Receiving tool arguments",
                        operation_id=request.request_id, event_id=f"{request.request_id}:{event.sequence}",
                        metadata={"provider": self.provider, "model": self.model, "call_id": event.call_id,
                            "stream_type": event.type, "received_characters": len(event.delta)})
                if event.type == "completed":
                    if event.response is None:
                        raise InvalidResponseError("Completed stream has no response")
                    self._complete(request, event.response, reservation, started)
                    reservation = ""
                    completed = True
                yield event
            if not completed:
                raise InvalidResponseError("Interrupted stream; reconcile checkpoint before continuing")
        except BaseException as exc:
            request_error = self._failed(request, reservation, exc, started,
                                         decision_correction_available=_decision_correction_available)
            if request_error is not None:
                raise request_error from None
            if isinstance(exc, asyncio.TimeoutError):
                raise InferenceTimeoutError("Inference stream timed out; reconcile before continuing") from exc
            raise
        finally:
            close_stream = getattr(events, "aclose", None)
            if close_stream:
                await close_stream()
            if self.adapter is None and adapter is not None:
                await adapter.close()


def create_model(*, api_key: str, model: str, base_url: str | None = None, provider: str | None = None,
                 default_headers: dict[str, str] | None = None, default_query: dict[str, str] | None = None,
                 model_configuration: dict[str, Any] | None = None,
                 **options: Any) -> NativeLLMEngine:
    from mana_agent.config.provider_registry import PROVIDERS, inference_protocol_for_model, model_id_for_provider
    from mana_agent.config.user_config import get_setting
    from mana_agent.config.inference_provider import resolve_inference_connection
    from mana_agent.config.settings import Settings
    selected = str(provider or get_setting("MANA_AI_PROVIDER", "") or "")
    if not selected:
        raise ValueError("Configure an inference provider before creating a native model")
    connection = resolve_inference_connection(Settings(), provider=selected, require_api_key=False)
    definition = PROVIDERS.get(selected)
    from urllib.parse import urlsplit
    effective_base_url = (base_url or connection.base_url).rstrip("/")
    parsed = urlsplit(effective_base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Configure an HTTP(S) provider base URL without credentials, query fields, or a fragment")
    bare_model = model_id_for_provider(definition.id, model)
    mode = str(get_setting("MANA_LLM_API_MODE", "auto") or "auto")
    endpoint = inference_protocol_for_model(definition.id, bare_model, mode=mode)
    timeout = float(options.pop("timeout", options.pop("request_timeout", 600)))
    attempts = int(options.pop("max_retries", 1)) + 1
    from mana_agent.model_routing.models import provider_request_overrides_from_configuration
    if "model_kwargs" in (model_configuration or {}) or "model_kwargs" in options:
        raise ValueError("Retired model_kwargs setting; move explicit provider fields into extra_body")
    parameters = {**provider_request_overrides_from_configuration(model_configuration), **options}
    effort = get_setting("MANA_LLM_REASONING_EFFORT", "")
    if effort and "reasoning_effort" not in parameters:
        parameters["reasoning_effort"] = effort
    return NativeLLMEngine(provider=definition.id, model=bare_model, api_key=api_key, base_url=effective_base_url,
        endpoint=endpoint, headers={**connection.headers, **(default_headers or {})},
        query={**connection.query_params, **(default_query or {})},
        parameters=parameters, timeout=timeout, max_attempts=attempts)
