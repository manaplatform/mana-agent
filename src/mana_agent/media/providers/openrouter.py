from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from mana_agent.config.model_catalog import ModelCapability, normalize_capabilities
from mana_agent.media.artifacts import _detected_mime
from mana_agent.media.errors import MediaProviderError
from mana_agent.media.models import (
    GenerationStatus,
    ImageGenerationRequest,
    MediaArtifact,
)
from mana_agent.media.providers.base import ProviderOutput
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
        self._image_model_cache: list[dict[str, Any]] | None = None

    def capabilities(self, model: str) -> frozenset[ModelCapability]:
        lowered = str(model or "").lower()
        # Non-text and dedicated image model family indicators
        if any(
            marker in lowered
            for marker in (
                "dall-e",
                "image-gen",
                "image_generation",
                "flux",
                "stable-diffusion",
                "midjourney",
                "recraft",
                "imagen",
                "seedance",
                "ideogram",
            )
        ):
            return frozenset({ModelCapability.IMAGE_GENERATION})

        # Known text-only reasoning/chat families must not be inferred as image generation
        if any(
            marker in lowered
            for marker in (
                "grok",
                "claude",
                "deepseek",
                "llama",
                "mistral",
                "qwen",
                "gpt-4",
                "gpt-3",
                "o1",
                "o3",
                "o4",
                "gemini",
                "command-r",
            )
        ) and not any(marker in lowered for marker in ("image", "dall-e", "flux")):
            return normalize_capabilities(self.provider_id, model)

        caps = normalize_capabilities(self.provider_id, model)
        return caps

    def list_image_models(self) -> list[dict[str, Any]]:
        """Fetch the dedicated OpenRouter image model catalog."""
        raw_bytes, _, _ = self._request_bytes("GET", "/images/models", None)
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MediaProviderError(
                "media_catalog_invalid",
                "Failed to parse OpenRouter image model catalog.",
            ) from exc

        data = payload.get("data")
        if not isinstance(data, list):
            return []

        models: list[dict[str, Any]] = []
        for raw in data:
            if not isinstance(raw, dict) or not str(raw.get("id") or "").strip():
                continue
            arch = raw.get("architecture") if isinstance(raw.get("architecture"), dict) else {}
            output_modalities = arch.get("output_modalities") if isinstance(arch, dict) else []
            if not isinstance(output_modalities, list):
                output_modalities = []
            out_mods_lower = [str(m).lower() for m in output_modalities]

            # Only retain models with image output
            lowered_id = str(raw["id"]).lower()
            if "image" not in out_mods_lower and not any(
                marker in lowered_id
                for marker in (
                    "dall-e",
                    "image-gen",
                    "image_generation",
                    "flux",
                    "stable-diffusion",
                    "midjourney",
                    "recraft",
                    "imagen",
                    "ideogram",
                )
            ):
                continue

            models.append(
                {
                    "id": str(raw["id"]).strip(),
                    "name": str(raw.get("name") or raw["id"]).strip(),
                    "architecture": {
                        "input_modalities": arch.get("input_modalities")
                        if isinstance(arch.get("input_modalities"), list)
                        else ["text"],
                        "output_modalities": output_modalities
                        if output_modalities
                        else ["image"],
                    },
                    "supported_parameters": list(raw.get("supported_parameters") or []),
                    "supports_streaming": bool(raw.get("supports_streaming", False)),
                    "endpoints": list(raw.get("endpoints") or []),
                }
            )
        self._image_model_cache = models
        return models

    @staticmethod
    def _image_payload(request: ImageGenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "response_format": "b64_json",
        }
        if request.count > 1:
            payload["n"] = request.count
        if request.size and request.size != "auto":
            payload["size"] = request.size
        if request.resolution:
            payload["resolution"] = request.resolution
        if request.aspect_ratio:
            payload["aspect_ratio"] = request.aspect_ratio
        if request.quality and request.quality != "auto":
            payload["quality"] = request.quality
        if request.output_format and request.output_format != "png":
            payload["output_format"] = request.output_format
        if request.background:
            payload["background"] = request.background

        model_name = request.model.split("/")[-1]
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
        return payload

    def generate_image(
        self,
        request: ImageGenerationRequest,
        reference_artifacts: tuple[MediaArtifact, ...] = (),
    ) -> ProviderOutput:
        if reference_artifacts:
            raise MediaProviderError(
                "media_reference_unsupported",
                "OpenRouter image generation does not currently support reference-image editing.",
            )

        payload = self._image_payload(request)
        body = json.dumps(payload).encode("utf-8")

        response_bytes, request_id, _ = self._request_bytes(
            "POST",
            "/images",
            body,
            content_type="application/json",
            idempotency_key=request.idempotency_key,
        )

        try:
            response = json.loads(response_bytes.decode("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MediaProviderError(
                "media_provider_invalid_output",
                "The image provider returned invalid JSON output.",
            ) from exc

        if not isinstance(response, dict):
            raise MediaProviderError(
                "media_provider_invalid_output",
                "The image provider returned an unexpected response structure.",
            )

        content: list[bytes] = []
        urls: list[str] = []
        mime_types: list[str] = []

        data = response.get("data")
        if not isinstance(data, list) or not data:
            raise MediaProviderError(
                "media_provider_empty_output",
                "The image provider returned no image data.",
            )

        for item in data:
            if not isinstance(item, dict):
                continue
            encoded = str(item.get("b64_json") or "").strip()
            if encoded:
                try:
                    decoded = base64.b64decode(encoded, validate=True)
                except ValueError as exc:
                    raise MediaProviderError(
                        "media_provider_invalid_output",
                        "The image provider returned invalid base64-encoded output.",
                    ) from exc
                if len(decoded) == 0:
                    raise MediaProviderError(
                        "media_provider_empty_output",
                        "The image provider returned empty image bytes.",
                    )
                content.append(decoded)
                declared_mime = str(item.get("media_type") or "").strip()
                detected = _detected_mime(decoded)
                effective_mime = (
                    declared_mime
                    if declared_mime and declared_mime.startswith("image/")
                    else detected
                    if detected != "application/octet-stream"
                    else f"image/{request.output_format}"
                )
                mime_types.append(effective_mime)
            elif item.get("url"):
                urls.append(str(item["url"]))

        if not content and not urls:
            raise MediaProviderError(
                "media_provider_empty_output",
                "The image provider returned no downloadable output.",
            )

        dimensions = request.size.split("x", 1)
        metadata: dict[str, Any] = {
            "provider": self.provider_id,
            "model": request.model,
        }
        if len(dimensions) == 2 and all(v.isdigit() for v in dimensions):
            metadata["width"] = int(dimensions[0])
            metadata["height"] = int(dimensions[1])

        usage_dict = response.get("usage")
        if isinstance(usage_dict, dict):
            metadata["usage"] = usage_dict
            if "cost" in usage_dict:
                metadata["cost"] = usage_dict["cost"]

        return ProviderOutput(
            provider_request_id=request_id or str(response.get("id") or ""),
            status=GenerationStatus.COMPLETED,
            content=tuple(content),
            mime_types=tuple(mime_types),
            remote_urls=tuple(urls),
            metadata=metadata,
        )

    def _redact(self, text: str) -> str:
        if not text:
            return ""
        if self._api_key and self._api_key in text:
            text = text.replace(self._api_key, "[REDACTED]")
        # Redact generic bearer tokens or key patterns
        return re.sub(r"sk-[a-zA-Z0-9_-]{10,}", "[REDACTED]", text)

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

                error_detail = ""
                try:
                    if hasattr(exc, "read") and callable(exc.read):
                        raw_read = exc.read()
                        if raw_read:
                            error_body = raw_read.decode("utf-8", errors="replace")
                            try:
                                parsed = json.loads(error_body)
                                if isinstance(parsed, dict) and "error" in parsed:
                                    err_obj = parsed["error"]
                                    if isinstance(err_obj, dict):
                                        error_detail = str(err_obj.get("message") or "")
                                    elif isinstance(err_obj, str):
                                        error_detail = err_obj
                                elif error_body:
                                    error_detail = error_body
                            except json.JSONDecodeError:
                                error_detail = error_body
                except Exception:
                    pass

                if not error_detail and getattr(exc, "msg", None):
                    error_detail = str(exc.msg)
                elif not error_detail and getattr(exc, "reason", None):
                    error_detail = str(exc.reason)

                code = (
                    "media_provider_auth_required"
                    if exc.code in {401, 403}
                    else "media_rate_limited"
                    if exc.code == 429
                    else "media_provider_parameter_rejected"
                    if exc.code == 400
                    else "media_provider_rejected"
                )
                safe_detail = self._redact(error_detail) if error_detail else f"The media provider rejected the request (HTTP {exc.code})."
                raise MediaProviderError(
                    code,
                    safe_detail,
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
                safe_reason = self._redact(str(getattr(exc, "reason", exc)))
                raise MediaProviderError(
                    "media_provider_unavailable",
                    f"The media provider could not be reached: {safe_reason}",
                    retryable=True,
                ) from exc
        raise MediaProviderError(
            "media_provider_unavailable", "The media provider could not be reached."
        )
