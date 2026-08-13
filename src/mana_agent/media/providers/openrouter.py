from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import Any

from mana_agent.media.errors import MediaProviderError
from mana_agent.media.models import ImageGenerationRequest
from mana_agent.media.providers.openai import _TRANSIENT_HTTP, OpenAIMediaProvider


class OpenRouterMediaProvider(OpenAIMediaProvider):
    provider_id = "openrouter"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: int = 120,
        http_referer: str = "https://github.com/mana-agent/mana-agent",
        title: str = "Mana-Agent",
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        self.http_referer = http_referer
        self.title = title

    @staticmethod
    def _image_payload(request: ImageGenerationRequest) -> dict[str, Any]:
        model_name = request.model.split("/")[-1]
        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "n": request.count,
        }
        if model_name.startswith("dall-e"):
            if request.output_format != "png" or request.background:
                raise MediaProviderError(
                    "media_provider_parameter_rejected",
                    "DALL-E models require PNG output and do not support background control.",
                )
            if model_name == "dall-e-3" and request.count != 1:
                raise MediaProviderError(
                    "media_provider_parameter_rejected",
                    "DALL-E 3 accepts one image per request.",
                )
            allowed_sizes = (
                {"256x256", "512x512", "1024x1024"}
                if model_name == "dall-e-2"
                else {"1024x1024", "1024x1792", "1792x1024"}
            )
            if request.size != "auto" and request.size not in allowed_sizes:
                raise MediaProviderError(
                    "media_provider_parameter_rejected",
                    "The selected DALL-E model does not support the requested size.",
                )
            if model_name == "dall-e-2" and request.quality != "auto":
                raise MediaProviderError(
                    "media_provider_parameter_rejected",
                    "DALL-E 2 does not accept a quality setting.",
                )
            if model_name == "dall-e-3" and request.quality not in {
                "auto",
                "standard",
                "hd",
            }:
                raise MediaProviderError(
                    "media_provider_parameter_rejected",
                    "DALL-E 3 quality must be auto, standard, or hd.",
                )
            payload["response_format"] = "b64_json"
            if request.size != "auto":
                payload["size"] = request.size
            if request.quality != "auto" and model_name == "dall-e-3":
                payload["quality"] = request.quality
            return payload

        # Non-DALL-E models
        if request.size != "auto":
            payload["size"] = request.size
        if request.quality != "auto":
            payload["quality"] = request.quality
        payload["response_format"] = "b64_json"
        return payload

    def _request_bytes(
        self,
        method: str,
        path: str,
        data: bytes | None,
        *,
        content_type: str = "",
        idempotency_key: str = "",
        absolute: bool = False,
    ) -> tuple[bytes, str, str]:
        url = path if absolute else f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": self.http_referer,
            "X-OpenRouter-Title": self.title,
        }
        if content_type:
            headers["Content-Type"] = content_type
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        for attempt in range(3):
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return (
                        response.read(),
                        str(response.headers.get("x-request-id") or ""),
                        str(response.headers.get("content-type") or ""),
                    )
            except urllib.error.HTTPError as exc:
                if exc.code in _TRANSIENT_HTTP and attempt < 2:
                    time.sleep(0.25 * (2**attempt))
                    continue
                code = (
                    "media_authentication_failed"
                    if exc.code in {401, 403}
                    else "media_rate_limited"
                    if exc.code == 429
                    else "media_provider_rejected"
                )
                raise MediaProviderError(
                    code,
                    f"The media provider rejected the request (HTTP {exc.code}).",
                    retryable=exc.code in _TRANSIENT_HTTP,
                ) from exc
            except TimeoutError as exc:
                raise MediaProviderError(
                    "media_generation_timeout",
                    "The media generation timed out.",
                    retryable=True,
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < 2:
                    time.sleep(0.25 * (2**attempt))
                    continue
                raise MediaProviderError(
                    "media_provider_unavailable",
                    "The media provider could not be reached.",
                    retryable=True,
                ) from exc
        raise MediaProviderError(
            "media_provider_unavailable", "The media provider could not be reached."
        )
