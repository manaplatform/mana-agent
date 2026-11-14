"""Native inference execution. No tools, routes, tasks or agent loops execute here."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any
from pydantic import BaseModel
from mana_agent.llm.errors import CapabilityError, InvalidResponseError, InferenceTimeoutError, ProviderRequestError, format_provider_error
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
                 endpoint: str, headers: dict[str, str] | None = None,
                 query: dict[str, str] | None = None, parameters: dict[str, Any] | None = None,
                 timeout: float = 600, max_attempts: int = 2, adapter: Any = None):
        if not provider or not model:
            raise ValueError("Inference requires an explicitly configured provider and model")
        import math
        if not math.isfinite(timeout) or timeout <= 0 or not 1 <= max_attempts <= 3:
            raise ValueError("Inference requires a finite positive timeout and one to three transport attempts")
        self.provider, self.model, self.api_key, self.base_url = provider, model, api_key, base_url
        self.selected_provider, self.model_name = provider, model
        self.endpoint, self.headers, self.query = endpoint, headers or {}, query or {}
        self.parameters, self.timeout, self.max_attempts = parameters or {}, timeout, max_attempts
        self.adapter = adapter
        self.tools: list[dict[str, Any]] = []
        self.tool_choice: str | dict[str, Any] | None = None
        self.context_cost_governor = None
        self._volatile_continuations: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    def with_tools(self, tools: list[Any], *, tool_choice: Any = None) -> NativeLLMEngine:
        selected = copy.copy(self)
        selected.tools = [tool_specification(tool) for tool in tools]
        selected.tool_choice = tool_choice
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
        profile.require(*required, "context_window")
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
        from mana_agent.context_cost.estimator import estimate_value_tokens
        serialized = [message.model_dump(mode="json") for message in native]
        if continuation is not None and continuation.mode == "stateless":
            serialized = [*continuation.items, *serialized[continuation.message_count:]]
        response_schema = schema.model_json_schema() if isinstance(schema, type) and issubclass(schema, BaseModel) else schema
        tokens = estimate_value_tokens(serialized) + estimate_value_tokens(self.tools)
        if response_schema:
            tokens += estimate_value_tokens(response_schema)
        input_limit = profile.value("max_input_tokens")
        if tokens + output > profile.context_window or (type(input_limit) is int and tokens > input_limit):
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
            error = InvalidResponseError("Selected provider refused the request")
            error.provider_usage, error.response_accepted = response.usage.model_dump(), True
            raise error
        if not response.content and not response.tool_calls and not response.protocol_pending:
            error = InvalidResponseError("Empty inference output is not completion evidence")
            error.provider_usage, error.response_accepted = response.usage.model_dump(), True
            raise error
        try:
            from mana_agent.multi_agent.tools.operation_policy import validate_arguments
            offered = {tool["name"]: tool["parameters"] for tool in request.tools if tool.get("name") and "parameters" in tool}
            seen = {call.id for message in request.messages for call in message.tool_calls}
            for call in response.tool_calls:
                if call.id in seen or call.name not in offered:
                    raise InvalidResponseError("Provider returned a reused call ID or a tool outside the offered set")
                validate_arguments(call.args, offered[call.name])
                seen.add(call.id)
        except ValueError as exc:
            error = InvalidResponseError("Provider tool arguments violate the registered schema")
            error.provider_usage, error.response_accepted = response.usage.model_dump(), True
            raise error from exc
        except InvalidResponseError as exc:
            exc.provider_usage, exc.response_accepted = response.usage.model_dump(), True
            raise
        if reservation:
            self.context_cost_governor.record_model_call(reservation, usage=response.usage.model_dump(),
                provider=request.provider, model=request.model, estimated_input=[m.content for m in request.messages])
        if response.continuation is not None:
            # Output-budget defaults are not caller configuration changes.
            response.continuation.configuration_fingerprint = request.configuration_fingerprint
            response.continuation.message_fingerprint = _conversation_fingerprint([*request.messages, response.message])
        from mana_agent.evals.recorder import record_current
        record_current("model.call", {"boundary": "native_inference", "provider": request.provider,
            "model": request.model, "task_id": request.task_id, "request_id": request.request_id,
            "usage": response.usage.model_dump(), "latency_seconds": time.perf_counter() - started})
        _emit_inference_activity("model.completed", title="Model responded", status="success", operation_id=request.request_id,
                      metadata={"provider": request.provider, "model": request.model})
        if response.public_summaries and self.profile().supports("public_reasoning_summaries"):
            for index, summary in enumerate(response.public_summaries):
                _emit_inference_activity("agent.summary", title="Reasoning summary", summary=summary, status="success",
                    event_id=f"summary-{response.id}-{index}", operation_id=request.request_id,
                    metadata={"summary_source": "provider", "summary_trusted": True, "summary_complete": True,
                        "provider": self.provider, "transport": "direct_responses" if self.endpoint == "responses" else self.endpoint})

    def _failed(self, request: InferenceRequest, reservation: str, exc: BaseException) -> None:
        if reservation:
            if getattr(exc, "response_accepted", False):
                self.context_cost_governor.record_model_call(reservation, usage=getattr(exc, "provider_usage", None),
                    provider=request.provider, model=request.model, estimated_input=[message.content for message in request.messages])
            else:
                self.context_cost_governor.release_reservation(reservation, reason=f"Inference failed: {type(exc).__name__}")
        _emit_inference_activity("model.failed", title="Model request failed", status="failed", operation_id=request.request_id,
            metadata={"provider": request.provider, "model": request.model, "error_type": type(exc).__name__})

    async def _refresh_metadata(self) -> None:
        selected = self.profile()
        if self.adapter is not None:
            return
        if selected.supports("input_text") and selected.supports("output_text") and selected.context_window and selected.max_output_tokens:
            return
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
                    return
            except ValueError:
                pass
        await ModelDiscoveryService().discover(provider=self.provider, base_url=self.base_url, api_key=self.api_key,
            default_headers=self.headers, query_params=self.query)

    async def generate(self, messages: Any, *, schema: Any = None, continuation: Any = None, **options: Any) -> ModelResponse:
        if options.get("stream", self.parameters.get("stream", False)):
            result = None
            async for event in self.stream(messages, schema=schema, continuation=continuation, **options):
                if event.type == "completed":
                    result = event.response
            if result is None:
                raise InvalidResponseError("Stream did not return a completed response")
            return result
        await self._refresh_metadata()
        request = self._request(messages, schema=schema, continuation=continuation, **options)
        reservation, started = self._reserve(request), time.perf_counter()
        _emit_inference_activity("model.started", title="Waiting for model", status="running", operation_id=request.request_id,
                      metadata={"provider": request.provider, "model": request.model})
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
            self._failed(request, reservation, exc)
            if isinstance(exc, asyncio.TimeoutError):
                raise InferenceTimeoutError("Selected inference request timed out; no alternate action was attempted") from exc
            if isinstance(getattr(exc, "status_code", None), int):
                from mana_agent.integrations.provider_failure import classify_http_status
                failure = classify_http_status(exc.status_code, provider=self.provider, model=self.model,
                    operation="native_inference", endpoint=self.endpoint)
                error = ProviderRequestError(failure.safe_message or failure.user_status(), provider=self.provider,
                    status_code=exc.status_code, retryable=False)
                error.failure = failure
                raise error from exc
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

    async def decide(self, messages: Any, schema: type[BaseModel] | dict[str, Any], **options: Any) -> BaseModel | dict[str, Any]:
        response = await self.generate(messages, schema=schema, **options)
        if not isinstance(response.content, str):
            raise InvalidResponseError("Structured decision did not return JSON text")
        if isinstance(schema, dict):
            from mana_agent.multi_agent.tools.operation_policy import validate_arguments
            value = json.loads(response.content)
            validate_arguments(value, schema)
            return value
        return schema.model_validate_json(response.content, strict=True)

    def decide_sync(self, messages: Any, schema: type[BaseModel] | dict[str, Any], **options: Any) -> BaseModel | dict[str, Any]:
        response = self.generate_sync(messages, schema=schema, **options)
        if not isinstance(response.content, str):
            raise InvalidResponseError("Structured decision did not return JSON text")
        if isinstance(schema, dict):
            from mana_agent.multi_agent.tools.operation_policy import validate_arguments
            value = json.loads(response.content)
            validate_arguments(value, schema)
            return value
        return schema.model_validate_json(response.content, strict=True)

    async def stream(self, messages: Any, **options: Any) -> AsyncIterator[StreamEvent]:
        await self._refresh_metadata()
        request = self._request(messages, **options)
        self.profile(request.parameters).require("streaming", *(["tool_call_streaming"] if request.tools else []))
        reservation, started = self._reserve(request), time.perf_counter()
        adapter = None
        events = None
        completed = False
        deadline = time.monotonic() + self.timeout
        _emit_inference_activity("model.started", title="Waiting for model stream", status="running",
            operation_id=request.request_id, metadata={"provider": self.provider, "model": self.model})
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
            self._failed(request, reservation, exc)
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
    from mana_agent.config.provider_registry import PROVIDERS, model_id_for_provider
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
    endpoint = definition.inference_protocol if mode == "auto" else mode
    if definition.inference_protocol == "anthropic_messages":
        endpoint = "anthropic_messages"
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
