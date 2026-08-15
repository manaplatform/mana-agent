from __future__ import annotations

import base64
import difflib
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from mana_agent.config.model_catalog import ModelCapability
from mana_agent.media.artifacts import _detected_mime
from mana_agent.media.errors import (
    MediaCapabilityError,
    MediaModelNotFoundError,
    MediaProviderError,
)
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
        cache_ttl_seconds: float = 300.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        self.http_referer = http_referer
        self.title = title
        self._image_model_cache: dict[str, dict[str, Any]] | None = None
        self._image_model_cache_fetched_at: float = 0.0
        self._cache_ttl_seconds = float(cache_ttl_seconds)

    def list_image_models(self, *, force_refresh: bool = False) -> list[dict[str, Any]]:
        """Fetch the dedicated OpenRouter image model catalog (GET /api/v1/images/models)."""
        if (
            not force_refresh
            and self._image_model_cache is not None
            and (time.monotonic() - self._image_model_cache_fetched_at) < self._cache_ttl_seconds
        ):
            return list(self._image_model_cache.values())

        try:
            raw_bytes, _, _ = self._request_bytes("GET", "/images/models", None)
        except MediaProviderError as exc:
            if exc.code in {"media_provider_auth_required", "media_image_provider_auth_required"}:
                raise MediaProviderError(
                    "media_image_provider_auth_required",
                    exc.detail,
                    retryable=False,
                ) from exc
            raise MediaProviderError(
                "media_image_provider_unavailable",
                exc.detail,
                retryable=True,
            ) from exc

        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MediaProviderError(
                "media_image_provider_unavailable",
                "Failed to parse OpenRouter image model catalog.",
            ) from exc

        data = payload.get("data")
        if not isinstance(data, list):
            self._image_model_cache = {}
            self._image_model_cache_fetched_at = time.monotonic()
            return []

        models: dict[str, dict[str, Any]] = {}
        for raw in data:
            if not isinstance(raw, dict):
                continue
            model_id = str(raw.get("id") or "").strip()
            if not model_id:
                continue

            arch = raw.get("architecture") if isinstance(raw.get("architecture"), dict) else {}
            input_modalities = (
                list(arch.get("input_modalities") or [])
                if isinstance(arch.get("input_modalities"), list)
                else ["text"]
            )
            output_modalities = (
                list(arch.get("output_modalities") or [])
                if isinstance(arch.get("output_modalities"), list)
                else []
            )
            supported_parameters = (
                list(raw.get("supported_parameters") or [])
                if isinstance(raw.get("supported_parameters"), list)
                else []
            )
            supports_streaming = bool(raw.get("supports_streaming", False))
            endpoints = (
                list(raw.get("endpoints") or [])
                if isinstance(raw.get("endpoints"), list)
                else []
            )

            models[model_id] = {
                "id": model_id,
                "name": str(raw.get("name") or model_id).strip(),
                "architecture": {
                    "input_modalities": input_modalities,
                    "output_modalities": output_modalities,
                },
                "supported_parameters": supported_parameters,
                "supports_streaming": supports_streaming,
                "endpoints": endpoints,
            }

        self._image_model_cache = models
        self._image_model_cache_fetched_at = time.monotonic()
        return list(models.values())

    def get_image_model(self, model: str) -> dict[str, Any]:
        """Fetch and validate model from OpenRouter image catalog with single retry on miss."""
        target = str(model or "").strip()
        if not target:
            raise MediaCapabilityError(
                "media_image_model_not_configured",
                "No image generation model configured.",
            )

        if (
            self._image_model_cache is None
            or (time.monotonic() - self._image_model_cache_fetched_at) >= self._cache_ttl_seconds
        ):
            self.list_image_models(force_refresh=False)

        entry = self._image_model_cache.get(target) if self._image_model_cache is not None else None

        # On cache miss: refresh catalog once and retry exact lookup once
        if entry is None:
            self.list_image_models(force_refresh=True)
            entry = (
                self._image_model_cache.get(target)
                if self._image_model_cache is not None
                else None
            )

        if entry is None:
            available_ids = list(self._image_model_cache.keys()) if self._image_model_cache else []
            closest = difflib.get_close_matches(target, available_ids, n=3, cutoff=0.2)
            suggested = closest[0] if closest else ""
            detail = (
                f"The requested image model {target!r} was not found in OpenRouter's image model catalog."
            )
            if suggested:
                detail += f" Suggested model: {suggested!r}."
            diag_meta = {
                "requested_model": target,
                "provider": self.provider_id,
                "image_catalog_loaded": self._image_model_cache is not None,
                "closest_model_ids": closest,
                "suggested_model": suggested,
            }
            raise MediaModelNotFoundError(
                "media_image_model_not_found",
                detail,
                metadata=diag_meta,
            )

        output_modalities = [
            str(m).lower()
            for m in entry.get("architecture", {}).get("output_modalities", [])
        ]
        if "image" not in output_modalities:
            detail = (
                f"The model {target!r} exists in OpenRouter's catalog but does not support image output modalities."
            )
            diag_meta = {
                "requested_model": target,
                "provider": self.provider_id,
                "output_modalities": entry.get("architecture", {}).get("output_modalities", []),
            }
            raise MediaCapabilityError(
                "media_image_model_unsupported",
                detail,
                metadata=diag_meta,
            )

        return entry

    def capabilities(self, model: str) -> frozenset[ModelCapability]:
        """Validate image model capability dynamically against OpenRouter's image catalog."""
        self.get_image_model(model)
        return frozenset({ModelCapability.IMAGE_GENERATION})

    def _image_payload(
        self,
        request: ImageGenerationRequest,
        model_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        info = model_info or self.get_image_model(request.model)
        supported = set(info.get("supported_parameters") or [])

        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
        }

        if not supported or "response_format" in supported:
            payload["response_format"] = "b64_json"

        if not supported or "n" in supported:
            if request.count > 1 or "n" in supported:
                payload["n"] = request.count

        if (not supported or "size" in supported) and request.size and request.size != "auto":
            payload["size"] = request.size

        if (not supported or "resolution" in supported) and request.resolution:
            payload["resolution"] = request.resolution

        if (not supported or "aspect_ratio" in supported) and request.aspect_ratio:
            payload["aspect_ratio"] = request.aspect_ratio

        if (not supported or "quality" in supported) and request.quality and request.quality != "auto":
            payload["quality"] = request.quality

        if (not supported or "output_format" in supported) and request.output_format and request.output_format != "png":
            payload["output_format"] = request.output_format

        if (not supported or "background" in supported) and request.background:
            payload["background"] = request.background

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

        model_info = self.get_image_model(request.model)
        payload = self._image_payload(request, model_info=model_info)
        body = json.dumps(payload).encode("utf-8")

        endpoint_path = "/images"
        endpoints = model_info.get("endpoints")
        if isinstance(endpoints, list) and endpoints:
            first_ep = endpoints[0]
            if isinstance(first_ep, dict) and first_ep.get("url"):
                ep_url = str(first_ep["url"]).strip()
                if ep_url.startswith("http://") or ep_url.startswith("https://"):
                    endpoint_path = ep_url
                elif ep_url.startswith("/"):
                    endpoint_path = ep_url

        absolute = endpoint_path.startswith("http://") or endpoint_path.startswith("https://")
        try:
            response_bytes, request_id, _ = self._request_bytes(
                "POST",
                endpoint_path,
                body,
                content_type="application/json",
                idempotency_key=request.idempotency_key,
                absolute=absolute,
            )
        except MediaProviderError as exc:
            if exc.code in {"media_provider_auth_required", "media_image_provider_auth_required"}:
                raise MediaProviderError(
                    "media_image_provider_auth_required", exc.detail, retryable=False
                ) from exc
            if exc.code in {"media_provider_parameter_rejected", "media_rate_limited", "media_generation_timeout"}:
                raise
            if exc.code == "media_provider_unavailable":
                raise MediaProviderError(
                    "media_image_provider_unavailable", exc.detail, retryable=True
                ) from exc
            raise MediaProviderError(
                "media_image_generation_failed", exc.detail, retryable=exc.retryable
            ) from exc

        try:
            response = json.loads(response_bytes.decode("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MediaProviderError(
                "media_image_generation_failed",
                "The image provider returned invalid JSON output.",
            ) from exc

        if not isinstance(response, dict):
            raise MediaProviderError(
                "media_image_generation_failed",
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
                        "media_image_generation_failed",
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
            "image_count": len(content) or len(urls),
        }
        if len(dimensions) == 2 and all(v.isdigit() for v in dimensions):
            metadata["width"] = int(dimensions[0])
            metadata["height"] = int(dimensions[1])

        usage_dict = response.get("usage")
        if isinstance(usage_dict, dict):
            metadata["usage"] = usage_dict
            metadata["provider_usage"] = usage_dict
            if "cost" in usage_dict:
                metadata["cost"] = usage_dict["cost"]
                metadata["actual_cost"] = usage_dict["cost"]

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
                    "media_image_provider_auth_required"
                    if exc.code in {401, 403}
                    else "media_rate_limited"
                    if exc.code == 429
                    else "media_provider_parameter_rejected"
                    if exc.code == 400
                    else "media_image_generation_failed"
                )
                safe_detail = (
                    self._redact(error_detail)
                    if error_detail
                    else f"The media provider rejected the request (HTTP {exc.code})."
                )
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
                    "media_image_provider_unavailable",
                    f"The media provider could not be reached: {safe_reason}",
                    retryable=True,
                ) from exc
        raise MediaProviderError(
            "media_image_provider_unavailable", "The media provider could not be reached."
        )
