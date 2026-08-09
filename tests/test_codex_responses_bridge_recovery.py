"""Regression tests for Responses bridge recovery and HTTP lifecycle.

Covers the reported bug:
  NVIDIA HTTP 400 → bridge returned SSE 200 → Codex responseStreamDisconnected
  → Reconnecting 1/5

After fix: HTTP 400 is returned as a non-retryable Responses error without SSE.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from mana_agent.integrations.codex.responses_bridge.models import BridgeUpstreamConfig
from mana_agent.integrations.codex.responses_bridge.server import (
    BRIDGE_TRANSPORT_MAX_ATTEMPTS,
    build_bridge_app,
)
from mana_agent.integrations.codex.responses_bridge.stream_adapter import (
    ChatToResponsesStreamAdapter,
)
from mana_agent.integrations.provider_failure import ProviderFailureKind


def _upstream(**updates: Any) -> BridgeUpstreamConfig:
    values: dict[str, Any] = {
        "provider": "nvidia",
        "display_name": "NVIDIA",
        "api_key": "nvapi-test-secret-key",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "deepseek-ai/deepseek-v4-flash-0731",
        "transport_max_attempts": 1,
    }
    values.update(updates)
    return BridgeUpstreamConfig(**values)


@pytest.fixture
def bridge_token() -> str:
    return "bridge-token"


def test_streaming_http_400_returns_error_without_sse_or_reconnect(
    bridge_token: str,
) -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "invalid messages sequence for deepseek chat template",
                    "type": "invalid_request_error",
                }
            },
            headers={"content-type": "application/json"},
        )

    transport = httpx.MockTransport(handler)

    class _Client(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    with patch(
        "mana_agent.integrations.codex.responses_bridge.server.httpx.AsyncClient",
        _Client,
    ):
        app = build_bridge_app(upstream=_upstream(), expected_token=bridge_token)
        client = TestClient(app)
        response = client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {bridge_token}"},
            json={
                "model": "deepseek-ai/deepseek-v4-flash-0731",
                "input": "fix astropy__astropy-12907",
                "stream": True,
            },
        )

    assert response.status_code == 400
    assert "text/event-stream" not in (response.headers.get("content-type") or "")
    body = response.json()
    assert body["status"] == "failed"
    error = body["error"]
    assert error["failure_kind"] == ProviderFailureKind.INVALID_REQUEST.value
    assert error["retryable"] is False
    assert error["http_status"] == 400
    assert error["attempts"] == 1
    assert "invalid messages sequence" in (error.get("message") or "").lower() or (
        "diagnostic" in (error.get("message") or "").lower()
    )
    assert "responseStreamDisconnected" not in json.dumps(body)
    assert "Reconnecting" not in json.dumps(body)
    # Exactly one upstream request — no bridge-level retry multiplication.
    assert call_count["n"] == 1
    assert BRIDGE_TRANSPORT_MAX_ATTEMPTS == 1


def test_streaming_open_timeout_returns_typed_error_without_starting_sse(
    bridge_token: str,
) -> None:
    """A provider that never accepts SSE must not pin a coding turn for 10 minutes."""

    class _SlowClient(httpx.AsyncClient):
        async def send(self, *args: Any, **kwargs: Any) -> httpx.Response:
            await asyncio.sleep(0.05)
            return httpx.Response(200, headers={"content-type": "text/event-stream"})

    upstream = _upstream(stream_open_timeout_seconds=0.001)
    with patch(
        "mana_agent.integrations.codex.responses_bridge.server.httpx.AsyncClient",
        _SlowClient,
    ):
        app = build_bridge_app(upstream=upstream, expected_token=bridge_token)
        client = TestClient(app)
        response = client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {bridge_token}"},
            json={"model": "m", "input": "hi", "stream": True},
        )

    assert response.status_code == 502
    assert "text/event-stream" not in (response.headers.get("content-type") or "")
    error = response.json()["error"]
    assert error["failure_kind"] == ProviderFailureKind.READ_TIMEOUT.value
    assert error["retryable"] is True


def test_streaming_http_410_model_retired_invalidates_cache(bridge_token: str, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana"))
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(410, json={"error": {"message": "model retired"}})

    transport = httpx.MockTransport(handler)

    class _Client(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    with patch(
        "mana_agent.integrations.codex.responses_bridge.server.httpx.AsyncClient",
        _Client,
    ), patch(
        "mana_agent.config.user_config.invalidate_model_cache"
    ) as invalidate:
        app = build_bridge_app(upstream=_upstream(), expected_token=bridge_token)
        client = TestClient(app)
        response = client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {bridge_token}"},
            json={"model": "deepseek-ai/deepseek-v4-flash-0731", "input": "hi", "stream": True},
        )

    assert response.status_code == 410
    error = response.json()["error"]
    assert error["failure_kind"] == ProviderFailureKind.MODEL_RETIRED.value
    assert error["retryable"] is False
    assert call_count["n"] == 1
    invalidate.assert_called_once()


def test_streaming_http_401_no_retry(bridge_token: str) -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    transport = httpx.MockTransport(handler)

    class _Client(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    with patch(
        "mana_agent.integrations.codex.responses_bridge.server.httpx.AsyncClient",
        _Client,
    ):
        app = build_bridge_app(upstream=_upstream(), expected_token=bridge_token)
        client = TestClient(app)
        response = client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {bridge_token}"},
            json={"model": "m", "input": "hi", "stream": True},
        )

    assert response.status_code == 401
    assert response.json()["error"]["retryable"] is False
    assert call_count["n"] == 1


def test_streaming_success_starts_sse_after_accept(bridge_token: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(lines).encode("utf-8"),
        )

    transport = httpx.MockTransport(handler)

    class _Client(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    with patch(
        "mana_agent.integrations.codex.responses_bridge.server.httpx.AsyncClient",
        _Client,
    ):
        app = build_bridge_app(upstream=_upstream(), expected_token=bridge_token)
        client = TestClient(app)
        with client.stream(
            "POST",
            "/v1/responses",
            headers={"Authorization": f"Bearer {bridge_token}"},
            json={"model": "m", "input": "hi", "stream": True},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in (response.headers.get("content-type") or "")
            text = "".join(response.iter_text())
    assert "response.created" in text
    assert "response.completed" in text
    assert "ok" in text


def test_active_sse_provider_failure_emits_response_failed() -> None:
    """After SSE has started, failures must close with response.failed (not raw EOF)."""
    adapter = ChatToResponsesStreamAdapter(model="deepseek-ai/deepseek-v4-flash")
    events = list(adapter.open_events())
    events.extend(adapter.ingest_chat_chunk({"choices": [{"delta": {"content": "partial"}}]}))
    # Simulate the bridge path after a mid-stream transport error.
    events.extend(
        adapter.close_events(
            failed=True,
            error={
                "code": "upstream_stream_interrupted",
                "message": "NVIDIA stream interrupted after partial response.",
                "type": "stream_interrupted",
                "retryable": True,
            },
        )
    )
    joined = "".join(events)
    assert "response.created" in joined
    assert "response.output_text.delta" in joined
    assert "response.failed" in joined
    assert "upstream_stream_interrupted" in joined
    assert adapter.received_stream_data is True


def test_non_stream_http_400_preserves_body_snippet(bridge_token: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "max_tokens too large for model"}},
        )

    transport = httpx.MockTransport(handler)

    class _Client(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    with patch(
        "mana_agent.integrations.codex.responses_bridge.server.httpx.AsyncClient",
        _Client,
    ):
        app = build_bridge_app(upstream=_upstream(), expected_token=bridge_token)
        client = TestClient(app)
        response = client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {bridge_token}"},
            json={"model": "m", "input": "hi", "stream": False},
        )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["retryable"] is False
    assert "max_tokens" in (error.get("upstream_body_snippet") or error.get("message") or "")


def test_nested_retry_layers_do_not_multiply(bridge_token: str) -> None:
    """Bridge performs exactly one upstream attempt even if Codex would retry."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    transport = httpx.MockTransport(handler)

    class _Client(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    with patch(
        "mana_agent.integrations.codex.responses_bridge.server.httpx.AsyncClient",
        _Client,
    ):
        app = build_bridge_app(upstream=_upstream(), expected_token=bridge_token)
        client = TestClient(app)
        # Simulate what would look like nested retries: caller retries 5 times.
        for _ in range(5):
            response = client.post(
                "/v1/responses",
                headers={"Authorization": f"Bearer {bridge_token}"},
                json={"model": "m", "input": "hi", "stream": True},
            )
            assert response.status_code == 503
            assert response.json()["error"]["retryable"] is True

    # Five caller attempts × 1 bridge transport attempt each = 5, not 5×N nested.
    assert call_count["n"] == 5
    assert BRIDGE_TRANSPORT_MAX_ATTEMPTS == 1


def test_stream_adapter_tracks_tool_side_effects() -> None:
    adapter = ChatToResponsesStreamAdapter(model="m")
    assert adapter.received_stream_data is False
    assert adapter.tool_side_effects is False
    adapter.ingest_chat_chunk(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "shell", "arguments": "{}"},
                            }
                        ]
                    }
                }
            ]
        }
    )
    assert adapter.received_stream_data is True
    assert adapter.tool_side_effects is True
    assert adapter.progress_snapshot()["tool_call_count"] == 1


def test_stream_adapter_failed_close_emits_response_failed() -> None:
    adapter = ChatToResponsesStreamAdapter(model="m")
    events = list(adapter.open_events())
    events.extend(
        adapter.close_events(
            failed=True,
            error={"code": "upstream_invalid_request", "message": "bad", "retryable": False},
        )
    )
    joined = "".join(events)
    assert "response.failed" in joined
    assert "upstream_invalid_request" in joined
    # Second close is a no-op.
    assert adapter.close_events(failed=True) == []


def test_health_documents_retry_ownership(bridge_token: str) -> None:
    app = build_bridge_app(upstream=_upstream(), expected_token=bridge_token)
    client = TestClient(app)
    health = client.get("/health").json()
    assert health["ok"] is True
    ownership = health["retry_ownership"]
    assert ownership["bridge_http_retries"] == 0
    assert ownership["bridge_transport_max_attempts"] == 1
