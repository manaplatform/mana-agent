"""Loopback ASGI server exposing POST /v1/responses for Codex.

Critical lifecycle rule
-----------------------
Never return ``HTTP 200 text/event-stream`` before the upstream Chat Completions
provider has accepted the request.

Incorrect (historical bug):
  return StreamingResponse → yield response.created → open NVIDIA → HTTP 400
  → generator fails / socket closes → Codex sees responseStreamDisconnected
  → Reconnecting 1/5

Correct:
  open NVIDIA upstream → inspect HTTP status → on non-2xx return JSON error
  with the real status → only on success begin Responses SSE.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from mana_agent.integrations.codex.responses_bridge.models import (
    BridgeUpstreamConfig,
    ResponsesBridgeError,
    UpstreamProviderError,
)
from mana_agent.integrations.codex.responses_bridge.request_adapter import (
    convert_responses_request_to_chat,
)
from mana_agent.integrations.codex.responses_bridge.response_adapter import (
    convert_chat_completion_to_response,
    responses_error_body,
)
from mana_agent.integrations.codex.responses_bridge.stream_adapter import (
    ChatToResponsesStreamAdapter,
)
from mana_agent.integrations.provider_failure import (
    PROVIDER_CIRCUIT_BREAKER,
    ProviderFailure,
    ProviderFailureKind,
    RetryOwner,
    circuit_scope_key,
    classify_http_status,
    classify_stream_interrupt,
    classify_transport_exception,
    log_provider_failure,
)

logger = logging.getLogger(__name__)

AuthChecker = Callable[[str | None], bool]

# Bridge never multiplies retries. Codex owns stream reconnect; the supervisor
# owns task-level recovery. See mana_agent.integrations.provider_failure.
BRIDGE_TRANSPORT_MAX_ATTEMPTS = 1


def _bearer_token(header_value: str | None) -> str | None:
    if not header_value:
        return None
    text = str(header_value).strip()
    if text.lower().startswith("bearer "):
        return text[7:].strip()
    return text or None


def _maybe_invalidate_model_cache(failure: ProviderFailure) -> None:
    """On model retired / not found, drop cached catalog so UI can refresh."""
    if failure.kind not in {
        ProviderFailureKind.MODEL_RETIRED,
        ProviderFailureKind.MODEL_NOT_FOUND,
    }:
        return
    try:
        from mana_agent.config.user_config import invalidate_model_cache

        invalidate_model_cache()
        logger.info(
            "provider_model_cache_invalidated provider=%s model=%s kind=%s",
            failure.provider,
            failure.model,
            failure.kind.value,
        )
    except Exception:
        logger.debug(
            "provider_model_cache_invalidation_failed provider=%s",
            failure.provider,
            exc_info=True,
        )


def _failure_to_upstream_error(failure: ProviderFailure) -> UpstreamProviderError:
    log_provider_failure(failure)
    _maybe_invalidate_model_cache(failure)
    return UpstreamProviderError(
        failure.safe_message,
        provider=failure.provider,
        status_code=failure.http_status,
        error_kind=failure.kind.value,
        failure=failure,
    )


def _error_status_for_failure(failure: ProviderFailure) -> int:
    if failure.http_status is not None and 400 <= int(failure.http_status) < 600:
        return int(failure.http_status)
    if failure.kind is ProviderFailureKind.AUTHENTICATION:
        return 401
    if failure.kind is ProviderFailureKind.PERMISSION:
        return 403
    if failure.kind is ProviderFailureKind.MODEL_NOT_FOUND:
        return 404
    if failure.kind is ProviderFailureKind.MODEL_RETIRED:
        return 410
    if failure.kind is ProviderFailureKind.INVALID_REQUEST:
        return 400
    if failure.kind is ProviderFailureKind.RATE_LIMITED:
        return 429
    if failure.kind is ProviderFailureKind.CIRCUIT_OPEN:
        return 503
    if failure.kind in {
        ProviderFailureKind.CONNECT_TIMEOUT,
        ProviderFailureKind.DNS_FAILURE,
        ProviderFailureKind.CONNECTION_RESET,
        ProviderFailureKind.READ_TIMEOUT,
    }:
        return 502
    return 502


def build_bridge_app(
    *,
    upstream: BridgeUpstreamConfig,
    expected_token: str,
    circuit_breaker=PROVIDER_CIRCUIT_BREAKER,
) -> FastAPI:
    """Create a FastAPI app that never exposes the upstream API key."""

    app = FastAPI(title="Mana Responses Bridge", docs_url=None, redoc_url=None)
    expected = str(expected_token or "")
    breaker = circuit_breaker
    transport_max_attempts = max(1, int(getattr(upstream, "transport_max_attempts", BRIDGE_TRANSPORT_MAX_ATTEMPTS) or 1))

    def _authorized(request: Request) -> bool:
        token = _bearer_token(request.headers.get("authorization"))
        return bool(expected) and token == expected

    def _circuit_key() -> str:
        return circuit_scope_key(
            provider=upstream.provider,
            endpoint=upstream.base_url.rstrip("/") + "/chat/completions",
        )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "mana_responses_bridge",
            "upstream": upstream.public_dict(),
            "transport": "codex_responses_bridge",
            "circuit": breaker.state(_circuit_key()).value,
            "retry_ownership": {
                "bridge_transport_max_attempts": transport_max_attempts,
                "bridge_http_retries": 0,
                "codex_stream_reconnect_owner": RetryOwner.CODEX_STREAM.value,
                "task_recovery_owner": RetryOwner.SUPERVISOR.value,
            },
        }

    @app.post("/v1/responses")
    async def create_response(request: Request) -> Response:
        if not _authorized(request):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Mana Responses bridge authentication failed.",
                        "type": "authentication_error",
                        "code": "bridge_unauthorized",
                    }
                },
            )
        try:
            body = await request.json()
        except Exception as exc:
            raise ResponsesBridgeError("Request body must be valid JSON.", status_code=400) from exc
        if not isinstance(body, dict):
            raise ResponsesBridgeError("Request body must be a JSON object.", status_code=400)

        try:
            chat_payload = convert_responses_request_to_chat(body, upstream=upstream)
        except ResponsesBridgeError:
            raise
        except Exception as exc:
            raise ResponsesBridgeError(
                f"Failed to convert Responses request: {type(exc).__name__}.",
                status_code=400,
            ) from exc

        stream = bool(chat_payload.get("stream"))
        url = upstream.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {upstream.api_key}",
            "Content-Type": "application/json",
            **dict(upstream.headers or {}),
        }
        model = str(chat_payload.get("model") or upstream.model or "")
        template = chat_payload.get("chat_template_kwargs")
        template_summary = ""
        if isinstance(template, dict):
            # Log non-secret template knobs only (thinking / reasoning_effort).
            thinking = template.get("thinking")
            effort = template.get("reasoning_effort")
            template_summary = f"thinking={thinking!r} reasoning_effort={effort!r}"
        logger.info(
            "responses_bridge.upstream_request provider=%s model=%s stream=%s host=%s "
            "has_tools=%s has_chat_template_kwargs=%s chat_template_kwargs=%s "
            "transport_max_attempts=%s",
            upstream.provider,
            model,
            stream,
            httpx.URL(url).host,
            bool(chat_payload.get("tools")),
            isinstance(template, dict),
            template_summary or "none",
            transport_max_attempts,
        )

        scope = _circuit_key()
        if not breaker.allow_request(scope):
            failure = ProviderFailure(
                kind=ProviderFailureKind.CIRCUIT_OPEN,
                provider=upstream.provider,
                model=model,
                http_status=503,
                retryable=True,
                safe_message=(
                    f"{upstream.display_name} temporarily unavailable (circuit open). "
                    "Not probing yet."
                ),
                operation="chat_completion_stream" if stream else "chat_completion",
                endpoint=url,
                retry_owner=RetryOwner.TRANSPORT,
                error_code="upstream_circuit_open",
                attempt=1,
                max_attempts=transport_max_attempts,
            )
            raise _failure_to_upstream_error(failure)

        if stream:
            # Open upstream BEFORE returning SSE so non-2xx never becomes a
            # fake stream disconnect / Codex reconnect.
            client: httpx.AsyncClient | None = None
            response: httpx.Response | None = None
            try:
                client = httpx.AsyncClient(timeout=upstream.timeout_seconds)
                http_request = client.build_request(
                    "POST", url, headers=headers, json=chat_payload
                )
                response = await client.send(http_request, stream=True)
            except httpx.TimeoutException as exc:
                if client is not None:
                    await client.aclose()
                failure = classify_transport_exception(
                    exc,
                    provider=upstream.provider,
                    model=model,
                    operation="chat_completion_stream",
                    endpoint=url,
                    attempt=1,
                    max_attempts=transport_max_attempts,
                    display_name=upstream.display_name,
                )
                breaker.record_failure(scope, failure)
                raise _failure_to_upstream_error(failure) from exc
            except httpx.HTTPError as exc:
                if client is not None:
                    await client.aclose()
                failure = classify_transport_exception(
                    exc,
                    provider=upstream.provider,
                    model=model,
                    operation="chat_completion_stream",
                    endpoint=url,
                    attempt=1,
                    max_attempts=transport_max_attempts,
                    display_name=upstream.display_name,
                )
                breaker.record_failure(scope, failure)
                raise _failure_to_upstream_error(failure) from exc

            assert response is not None and client is not None
            if response.status_code >= 400:
                body_bytes = await response.aread()
                await response.aclose()
                await client.aclose()
                failure = classify_http_status(
                    response.status_code,
                    provider=upstream.provider,
                    model=model,
                    body=body_bytes,
                    headers=response.headers,
                    operation="chat_completion_stream",
                    endpoint=url,
                    attempt=1,
                    max_attempts=transport_max_attempts,
                    display_name=upstream.display_name,
                )
                breaker.record_failure(scope, failure)
                raise _failure_to_upstream_error(failure)

            # Upstream accepted the stream. Only now begin Responses SSE.
            breaker.record_success(scope)
            return StreamingResponse(
                _stream_accepted_upstream(
                    client=client,
                    response=response,
                    chat_payload=chat_payload,
                    upstream=upstream,
                    url=url,
                    circuit_key=scope,
                    circuit_breaker=breaker,
                ),
                media_type="text/event-stream",
            )

        # Non-streaming path: one transport attempt, classify, no nested retries.
        try:
            async with httpx.AsyncClient(timeout=upstream.timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=chat_payload)
        except httpx.TimeoutException as exc:
            failure = classify_transport_exception(
                exc,
                provider=upstream.provider,
                model=model,
                operation="chat_completion",
                endpoint=url,
                attempt=1,
                max_attempts=transport_max_attempts,
                display_name=upstream.display_name,
            )
            breaker.record_failure(scope, failure)
            raise _failure_to_upstream_error(failure) from exc
        except httpx.HTTPError as exc:
            failure = classify_transport_exception(
                exc,
                provider=upstream.provider,
                model=model,
                operation="chat_completion",
                endpoint=url,
                attempt=1,
                max_attempts=transport_max_attempts,
                display_name=upstream.display_name,
            )
            breaker.record_failure(scope, failure)
            raise _failure_to_upstream_error(failure) from exc

        if response.status_code >= 400:
            failure = classify_http_status(
                response.status_code,
                provider=upstream.provider,
                model=model,
                body=response.text,
                headers=response.headers,
                operation="chat_completion",
                endpoint=url,
                attempt=1,
                max_attempts=transport_max_attempts,
                display_name=upstream.display_name,
            )
            breaker.record_failure(scope, failure)
            raise _failure_to_upstream_error(failure)

        breaker.record_success(scope)
        try:
            chat = response.json()
        except Exception as exc:
            raise ResponsesBridgeError(
                "Upstream Chat Completions response was not valid JSON.",
                status_code=502,
            ) from exc
        if not isinstance(chat, dict):
            raise ResponsesBridgeError(
                "Upstream Chat Completions response had an unexpected shape.",
                status_code=502,
            )
        converted = convert_chat_completion_to_response(
            chat, model=str(chat_payload.get("model") or upstream.model)
        )
        return JSONResponse(converted)

    @app.exception_handler(ResponsesBridgeError)
    async def bridge_error_handler(_request: Request, exc: ResponsesBridgeError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=responses_error_body(
                message=str(exc),
                code="mana_responses_bridge_error",
                status_code=exc.status_code,
            ),
        )

    @app.exception_handler(UpstreamProviderError)
    async def upstream_error_handler(_request: Request, exc: UpstreamProviderError) -> JSONResponse:
        failure = exc.failure
        status = _error_status_for_failure(failure)
        headers: dict[str, str] = {}
        if failure.retry_after is not None and failure.retryable:
            headers["Retry-After"] = str(int(max(1, round(failure.retry_after))))
        body = responses_error_body(
            message=failure.safe_message or str(exc),
            code=failure.error_code or f"upstream_{failure.kind.value}",
            status_code=status,
            extra={
                "provider": failure.provider,
                "model": failure.model,
                "failure_kind": failure.kind.value,
                "retryable": failure.retryable,
                "retry_owner": failure.retry_owner.value,
                "attempts": failure.attempt,
                "max_attempts": failure.max_attempts,
                "http_status": failure.http_status,
                "provider_request_id": failure.provider_request_id,
                # Sanitized snippet only — never secrets.
                "upstream_body_snippet": failure.upstream_body_snippet[:1024] or None,
            },
        )
        return JSONResponse(status_code=status, content=body, headers=headers)

    return app


async def _stream_accepted_upstream(
    *,
    client: httpx.AsyncClient,
    response: httpx.Response,
    chat_payload: dict[str, Any],
    upstream: BridgeUpstreamConfig,
    url: str,
    circuit_key: str,
    circuit_breaker,
):
    """Yield Responses SSE from an already-accepted (2xx) upstream stream.

    Once SSE has started, never let raw exceptions escape the generator.
    Emit ``response.failed`` and terminate cleanly so Codex receives a protocol
    failure rather than an unexplained socket EOF.
    """
    model = str(chat_payload.get("model") or upstream.model)
    adapter = ChatToResponsesStreamAdapter(model=model)
    try:
        for event in adapter.open_events():
            yield event

        buffer: list[str] = []
        async for line in response.aiter_lines():
            if line == "":
                if not buffer:
                    continue
                payload = "\n".join(
                    part[5:].lstrip() if part.startswith("data:") else part
                    for part in buffer
                    if part.startswith("data:")
                )
                buffer = []
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk, dict):
                    for event in adapter.ingest_chat_chunk(chunk):
                        yield event
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data = line[5:].lstrip()
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    buffer.append(line)
                    continue
                if isinstance(chunk, dict):
                    for event in adapter.ingest_chat_chunk(chunk):
                        yield event
                continue
            buffer.append(line)

        for event in adapter.close_events():
            yield event
        circuit_breaker.record_success(circuit_key)
    except httpx.TimeoutException as exc:
        failure = classify_transport_exception(
            exc,
            provider=upstream.provider,
            model=model,
            operation="chat_completion_stream",
            endpoint=url,
            received_stream_data=adapter.received_stream_data,
            tool_side_effects=adapter.tool_side_effects,
            display_name=upstream.display_name,
        )
        circuit_breaker.record_failure(circuit_key, failure)
        log_provider_failure(failure)
        for event in adapter.close_events(
            failed=True,
            error={
                "code": failure.error_code,
                "message": failure.safe_message,
                "type": failure.kind.value,
                "retryable": failure.retryable,
            },
        ):
            yield event
    except httpx.HTTPError as exc:
        failure = classify_transport_exception(
            exc,
            provider=upstream.provider,
            model=model,
            operation="chat_completion_stream",
            endpoint=url,
            received_stream_data=adapter.received_stream_data,
            tool_side_effects=adapter.tool_side_effects,
            display_name=upstream.display_name,
        )
        # Unexpected mid-stream close after data is a stream interrupt.
        if adapter.received_stream_data and failure.kind is not ProviderFailureKind.CANCELLED:
            failure = classify_stream_interrupt(
                provider=upstream.provider,
                model=model,
                received_stream_data=adapter.received_stream_data,
                tool_side_effects=adapter.tool_side_effects,
                display_name=upstream.display_name,
                detail=str(exc),
            )
        circuit_breaker.record_failure(circuit_key, failure)
        log_provider_failure(failure)
        for event in adapter.close_events(
            failed=True,
            error={
                "code": failure.error_code,
                "message": failure.safe_message,
                "type": failure.kind.value,
                "retryable": failure.retryable,
            },
        ):
            yield event
    except Exception as exc:
        logger.error(
            "responses_bridge.stream_failed provider=%s error_type=%s received_data=%s tools=%s",
            upstream.provider,
            type(exc).__name__,
            adapter.received_stream_data,
            adapter.tool_side_effects,
        )
        failure = classify_stream_interrupt(
            provider=upstream.provider,
            model=model,
            received_stream_data=adapter.received_stream_data,
            tool_side_effects=adapter.tool_side_effects,
            display_name=upstream.display_name,
            detail=type(exc).__name__,
        )
        circuit_breaker.record_failure(circuit_key, failure)
        log_provider_failure(failure)
        for event in adapter.close_events(
            failed=True,
            error={
                "code": "mana_responses_bridge_error",
                "message": failure.safe_message or "Mana Responses bridge stream conversion failed.",
                "type": failure.kind.value,
                "retryable": failure.retryable,
            },
        ):
            yield event
    finally:
        try:
            await response.aclose()
        except Exception:
            pass
        try:
            await client.aclose()
        except Exception:
            pass


__all__ = ["BRIDGE_TRANSPORT_MAX_ATTEMPTS", "build_bridge_app"]
