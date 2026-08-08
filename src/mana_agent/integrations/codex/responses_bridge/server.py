"""Loopback ASGI server exposing POST /v1/responses for Codex."""

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

logger = logging.getLogger(__name__)

AuthChecker = Callable[[str | None], bool]


def _bearer_token(header_value: str | None) -> str | None:
    if not header_value:
        return None
    text = str(header_value).strip()
    if text.lower().startswith("bearer "):
        return text[7:].strip()
    return text or None


def _classify_upstream_status(status_code: int) -> str:
    if status_code in {401, 403}:
        return "authentication"
    if status_code in {404, 410}:
        return "not_found"
    if status_code == 422:
        return "invalid_request"
    if status_code == 429:
        return "rate_limit"
    if status_code >= 500:
        return "provider_service"
    return "provider_error"


def _safe_upstream_message(
    *,
    provider: str,
    status_code: int,
    body_text: str,
    model: str | None = None,
) -> str:
    kind = _classify_upstream_status(status_code)
    model_part = f" model={model}" if model else ""
    labels = {
        "authentication": (
            f"{provider} authentication or permission failed (HTTP {status_code}).{model_part}"
        ),
        "not_found": (
            f"{provider} endpoint or model is unavailable (HTTP {status_code}).{model_part} "
            "Confirm the model id is enabled for this NVIDIA account and that the request "
            "uses Chat Completions with provider-correct options."
        ),
        "invalid_request": (
            f"{provider} rejected the request (HTTP {status_code}).{model_part}"
        ),
        "rate_limit": f"{provider} rate limit or quota exceeded (HTTP {status_code}).{model_part}",
        "provider_service": f"{provider} service failure (HTTP {status_code}).{model_part}",
        "provider_error": f"{provider} request failed (HTTP {status_code}).{model_part}",
    }
    # Never include raw provider body in user-facing text (may leak request fragments).
    _ = body_text
    return labels[kind]


def build_bridge_app(
    *,
    upstream: BridgeUpstreamConfig,
    expected_token: str,
) -> FastAPI:
    """Create a FastAPI app that never exposes the upstream API key."""

    app = FastAPI(title="Mana Responses Bridge", docs_url=None, redoc_url=None)
    expected = str(expected_token or "")

    def _authorized(request: Request) -> bool:
        token = _bearer_token(request.headers.get("authorization"))
        return bool(expected) and token == expected

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "mana_responses_bridge",
            "upstream": upstream.public_dict(),
            "transport": "codex_responses_bridge",
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
        # Never log secrets or full authorization headers.
        template = chat_payload.get("chat_template_kwargs")
        logger.info(
            "responses_bridge.upstream_request provider=%s model=%s stream=%s host=%s "
            "has_tools=%s has_chat_template_kwargs=%s",
            upstream.provider,
            chat_payload.get("model"),
            stream,
            httpx.URL(url).host,
            bool(chat_payload.get("tools")),
            isinstance(template, dict),
        )

        if stream:
            return StreamingResponse(
                _stream_upstream(
                    url=url,
                    headers=headers,
                    chat_payload=chat_payload,
                    upstream=upstream,
                ),
                media_type="text/event-stream",
            )

        try:
            async with httpx.AsyncClient(timeout=upstream.timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=chat_payload)
        except httpx.TimeoutException as exc:
            raise UpstreamProviderError(
                f"{upstream.display_name} request timed out.",
                provider=upstream.provider,
                error_kind="timeout",
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamProviderError(
                f"{upstream.display_name} network error.",
                provider=upstream.provider,
                error_kind="network",
            ) from exc

        if response.status_code >= 400:
            message = _safe_upstream_message(
                provider=upstream.display_name,
                status_code=response.status_code,
                body_text=response.text,
                model=str(chat_payload.get("model") or upstream.model or ""),
            )
            logger.error(
                "responses_bridge.upstream_failed provider=%s model=%s status=%s kind=%s",
                upstream.provider,
                chat_payload.get("model"),
                response.status_code,
                _classify_upstream_status(response.status_code),
            )
            raise UpstreamProviderError(
                message,
                provider=upstream.provider,
                status_code=response.status_code,
                error_kind=_classify_upstream_status(response.status_code),
            )
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
        status = int(exc.status_code or 502)
        return JSONResponse(
            status_code=status if 400 <= status < 600 else 502,
            content=responses_error_body(
                message=str(exc),
                code=f"upstream_{exc.error_kind}",
                status_code=status,
            ),
        )

    return app


async def _stream_upstream(
    *,
    url: str,
    headers: dict[str, str],
    chat_payload: dict[str, Any],
    upstream: BridgeUpstreamConfig,
):
    adapter = ChatToResponsesStreamAdapter(
        model=str(chat_payload.get("model") or upstream.model)
    )
    for event in adapter.open_events():
        yield event
    try:
        async with httpx.AsyncClient(timeout=upstream.timeout_seconds) as client:
            async with client.stream(
                "POST", url, headers=headers, json=chat_payload
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    message = _safe_upstream_message(
                        provider=upstream.display_name,
                        status_code=response.status_code,
                        body_text=body.decode("utf-8", errors="replace"),
                        model=str(chat_payload.get("model") or upstream.model or ""),
                    )
                    logger.error(
                        "responses_bridge.upstream_stream_failed provider=%s model=%s status=%s",
                        upstream.provider,
                        chat_payload.get("model"),
                        response.status_code,
                    )
                    for event in adapter.close_events(
                        failed=True,
                        error={
                            "code": f"upstream_{_classify_upstream_status(response.status_code)}",
                            "message": message,
                        },
                    ):
                        yield event
                    return

                async def line_iter():
                    async for line in response.aiter_lines():
                        yield line

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
                        # Single-line SSE frames without a blank terminator.
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
    except httpx.TimeoutException:
        for event in adapter.close_events(
            failed=True,
            error={
                "code": "upstream_timeout",
                "message": f"{upstream.display_name} request timed out.",
            },
        ):
            yield event
    except httpx.HTTPError:
        for event in adapter.close_events(
            failed=True,
            error={
                "code": "upstream_network",
                "message": f"{upstream.display_name} network error.",
            },
        ):
            yield event
    except Exception as exc:
        logger.error(
            "responses_bridge.stream_failed provider=%s error_type=%s",
            upstream.provider,
            type(exc).__name__,
        )
        for event in adapter.close_events(
            failed=True,
            error={
                "code": "mana_responses_bridge_error",
                "message": "Mana Responses bridge stream conversion failed.",
            },
        ):
            yield event


__all__ = ["build_bridge_app"]
