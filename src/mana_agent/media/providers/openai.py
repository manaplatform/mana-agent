from __future__ import annotations

import base64
import ipaddress
import json
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from mana_agent.config.model_catalog import ModelCapability, normalize_capabilities
from mana_agent.media.errors import MediaProviderError
from mana_agent.media.models import (
    GenerationStatus,
    ImageGenerationRequest,
    MediaArtifact,
    VideoGenerationRequest,
    VoiceGenerationRequest,
)
from mana_agent.media.providers.base import ProviderOutput


_TRANSIENT_HTTP = {408, 409, 429, 500, 502, 503, 504}
_VIDEO_STATUS = {
    "queued": GenerationStatus.QUEUED,
    "in_progress": GenerationStatus.GENERATING,
    "processing": GenerationStatus.GENERATING,
    "completed": GenerationStatus.COMPLETED,
    "failed": GenerationStatus.FAILED,
    "cancelled": GenerationStatus.CANCELLED,
}


class OpenAIMediaProvider:
    provider_id = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: int = 120,
    ) -> None:
        if not api_key.strip():
            raise MediaProviderError(
                "media_authentication_missing",
                "Media provider authentication is not configured.",
            )
        self._api_key = api_key.strip()
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        parsed_base = urllib.parse.urlparse(self.base_url)
        if (
            parsed_base.scheme not in {"http", "https"}
            or not parsed_base.hostname
            or parsed_base.username
            or parsed_base.password
        ):
            raise MediaProviderError(
                "media_base_url_invalid",
                "The configured media provider base URL is invalid.",
            )
        self.timeout_seconds = max(1, int(timeout_seconds))

    def capabilities(self, model: str) -> frozenset[ModelCapability]:
        return normalize_capabilities(self.provider_id, model)

    def generate_image(
        self,
        request: ImageGenerationRequest,
        reference_artifacts: tuple[MediaArtifact, ...] = (),
    ) -> ProviderOutput:
        payload = self._image_payload(request, for_edit=bool(reference_artifacts))
        is_gpt_image = self._is_gpt_image_model(request.model)

        if reference_artifacts:
            if self._is_dalle3_model(request.model):
                raise MediaProviderError(
                    "media_reference_unsupported",
                    "DALL-E 3 does not support reference-image editing.",
                )

            # GPT Image supports one or more reference images. Legacy DALL-E 2
            # editing accepts a single image.
            if not is_gpt_image and len(reference_artifacts) != 1:
                raise MediaProviderError(
                    "media_reference_count_rejected",
                    "The selected legacy image model accepts one image reference per request.",
                )

            multipart_fields = {
                key: self._multipart_scalar(value)
                for key, value in payload.items()
            }
            image_field = "image[]" if is_gpt_image else "image"
            files = tuple(
                (
                    image_field,
                    Path(reference.local_path).name,
                    reference.mime_type,
                    self._reference_bytes(reference),
                )
                for reference in reference_artifacts
            )
            body, boundary = self._multipart(multipart_fields, files=files)
            response, request_id, _ = self._request_json_bytes(
                "POST",
                "/images/edits",
                body,
                content_type=f"multipart/form-data; boundary={boundary}",
                idempotency_key=request.idempotency_key,
            )
        else:
            wire_payload = dict(payload)
            if is_gpt_image:
                # GPT Image returns base64 data natively. Keep response_format
                # in _image_payload() only for compatibility with existing
                # callers/tests, but never send the legacy DALL-E parameter.
                wire_payload.pop("response_format", None)

            response, request_id, _ = self._request_json(
                "POST",
                "/images/generations",
                wire_payload,
                idempotency_key=request.idempotency_key,
            )

        content: list[bytes] = []
        urls: list[str] = []
        revised_prompts: list[str] = []

        for item in response.get("data") or []:
            if not isinstance(item, dict):
                continue

            encoded = str(item.get("b64_json") or "")
            if encoded:
                try:
                    content.append(base64.b64decode(encoded, validate=True))
                except (ValueError, TypeError) as exc:
                    raise MediaProviderError(
                        "media_provider_invalid_output",
                        "The image provider returned invalid encoded output.",
                    ) from exc
            elif item.get("url"):
                urls.append(str(item["url"]))

            revised_prompt = str(item.get("revised_prompt") or "").strip()
            if revised_prompt:
                revised_prompts.append(revised_prompt)

        if not content and not urls:
            raise MediaProviderError(
                "media_provider_empty_output",
                "The image provider returned no downloadable output.",
            )

        mime = self._image_mime_type(request, payload)
        metadata = self._image_metadata(request, payload, response, revised_prompts)

        return ProviderOutput(
            provider_request_id=request_id,
            status=GenerationStatus.COMPLETED,
            content=tuple(content),
            mime_types=tuple(mime for _ in content),
            remote_urls=tuple(urls),
            metadata=metadata,
        )

    @staticmethod
    def _is_gpt_image_model(model: str) -> bool:
        normalized = str(model or "").strip().lower()
        return normalized.startswith("gpt-image-") or normalized == "chatgpt-image-latest"

    @staticmethod
    def _is_gpt_image_25_model(model: str) -> bool:
        return str(model or "").strip().lower().startswith("gpt-image-2.5-")

    @staticmethod
    def _is_gpt_image_2_model(model: str) -> bool:
        normalized = str(model or "").strip().lower()
        return normalized == "gpt-image-2" or normalized.startswith("gpt-image-2-202")

    @staticmethod
    def _is_dalle2_model(model: str) -> bool:
        return str(model or "").strip().lower() == "dall-e-2"

    @classmethod
    def _is_dalle3_model(cls, model: str) -> bool:
        normalized = str(model or "").strip().lower()
        return normalized.startswith("dall-e") and not cls._is_dalle2_model(normalized)

    @staticmethod
    def _resolve_orientation(aspect_ratio: str, size: str) -> str:
        ar = str(aspect_ratio or "").lower().strip()
        if ar in {"16:9", "16/9", "3:2", "3/2", "4:3", "4/3", "landscape", "wide"}:
            return "landscape"
        if ar in {"9:16", "9/16", "2:3", "2/3", "3:4", "3/4", "portrait", "tall"}:
            return "portrait"
        if ar in {"1:1", "1/1", "square"}:
            return "square"

        s = str(size or "").lower().strip()
        if "x" in s:
            parts = s.split("x", 1)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                width, height = int(parts[0]), int(parts[1])
                if width > height:
                    return "landscape"
                if height > width:
                    return "portrait"
                return "square"
        return "square"

    @staticmethod
    def _valid_flexible_gpt_image_size(size: str) -> bool:
        value = str(size or "").lower().strip()
        if value == "auto":
            return True
        if "x" not in value:
            return False

        width_text, height_text = value.split("x", 1)
        if not width_text.isdigit() or not height_text.isdigit():
            return False

        width, height = int(width_text), int(height_text)
        if width <= 0 or height <= 0:
            return False
        if width > 3840 or height > 3840:
            return False
        if width % 16 != 0 or height % 16 != 0:
            return False

        short_edge = min(width, height)
        long_edge = max(width, height)
        if short_edge == 0 or long_edge / short_edge > 3:
            return False

        pixels = width * height
        return 655_360 <= pixels <= 8_294_400

    @classmethod
    def _resolve_gpt_image_size(
        cls,
        *,
        model: str,
        requested_size: str,
        orientation: str,
    ) -> str:
        size = str(requested_size or "").lower().strip()

        # Normalize legacy DALL-E 3 portrait/landscape dimensions to the
        # recommended GPT Image dimensions. The 2.5 models also accept custom
        # sizes, but these two legacy values are normalized intentionally.
        legacy_size_map = {
            "1792x1024": "1536x1024",
            "1024x1792": "1024x1536",
        }
        if size in legacy_size_map:
            return legacy_size_map[size]

        # Preserve deterministic provider behavior when an older caller uses
        # size="auto" together with an aspect-ratio hint.
        if size == "auto":
            if orientation == "landscape":
                return "1536x1024"
            if orientation == "portrait":
                return "1024x1536"
            return "1024x1024"

        # GPT Image 2/2.5 accept valid custom dimensions too.
        if cls._is_gpt_image_25_model(model) or cls._is_gpt_image_2_model(model):
            if cls._valid_flexible_gpt_image_size(size):
                return size
        elif size in {"1024x1024", "1024x1536", "1536x1024"}:
            return size

        if orientation == "landscape":
            return "1536x1024"
        if orientation == "portrait":
            return "1024x1536"
        return "1024x1024"

    @classmethod
    def _normalize_gpt_image_quality(cls, model: str, quality: str) -> str | None:
        value = str(quality or "").lower().strip()

        # Omit auto/empty so OpenAI applies the model-native default.
        if not value or value == "auto":
            return None

        aliases = {
            "standard": "medium",
            "hd": "high",
        }
        value = aliases.get(value, value)

        if cls._is_gpt_image_25_model(model):
            allowed = {"low", "medium", "high", "xhigh", "max"}
        else:
            allowed = {"low", "medium", "high"}

        return value if value in allowed else None

    @staticmethod
    def _normalize_background(background: str) -> str:
        value = str(background or "").lower().strip()
        return value if value in {"transparent", "opaque", "auto"} else ""

    @staticmethod
    def _normalize_output_format(output_format: str) -> str:
        value = str(output_format or "").lower().strip()
        return value if value in {"png", "jpeg", "webp"} else "png"

    @staticmethod
    def _optional_int(request: ImageGenerationRequest, name: str) -> int | None:
        value = getattr(request, name, None)
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _image_payload(
        cls,
        request: ImageGenerationRequest,
        *,
        for_edit: bool = False,
    ) -> dict[str, Any]:
        model = str(request.model or "").strip()
        orientation = cls._resolve_orientation(request.aspect_ratio, request.size)

        if cls._is_dalle2_model(model):
            payload: dict[str, Any] = {
                "model": model,
                "prompt": request.prompt,
                "response_format": "b64_json",
                "n": max(1, min(4, request.count)),
            }
            payload["size"] = (
                request.size
                if request.size in {"256x256", "512x512", "1024x1024"}
                else "1024x1024"
            )
            return payload

        if cls._is_dalle3_model(model):
            payload = {
                "model": model,
                "prompt": request.prompt,
                "response_format": "b64_json",
                "n": 1,
            }
            if request.size in {"1024x1024", "1024x1792", "1792x1024"}:
                payload["size"] = request.size
            elif orientation == "landscape":
                payload["size"] = "1792x1024"
            elif orientation == "portrait":
                payload["size"] = "1024x1792"
            else:
                payload["size"] = "1024x1024"

            quality = str(request.quality or "").lower().strip()
            if quality in {"hd", "high"}:
                payload["quality"] = "hd"
            elif quality in {"standard", "medium", "low"}:
                payload["quality"] = "standard"
            return payload

        # Current GPT Image API.
        #
        # Do not send the legacy DALL-E response_format parameter here.
        # GPT Image returns base64-encoded image data in data[].b64_json.
        payload = {
            "model": model,
            "prompt": request.prompt,
            "n": max(1, min(4, request.count)),
            "size": cls._resolve_gpt_image_size(
                model=model,
                requested_size=request.size,
                orientation=orientation,
            ),
            # Backward-compatible helper output. generate_image() removes this
            # legacy field before sending GPT Image requests to OpenAI.
            "response_format": "b64_json",
        }

        quality = cls._normalize_gpt_image_quality(model, request.quality)
        if quality is not None:
            payload["quality"] = quality

        output_format = cls._normalize_output_format(request.output_format)
        background = cls._normalize_background(request.background)

        # Transparent outputs require PNG or WebP. Prefer PNG rather than
        # submitting a provider-invalid transparent JPEG request.
        if background == "transparent" and output_format == "jpeg":
            output_format = "png"

        payload["output_format"] = output_format
        if background:
            payload["background"] = background

        output_compression = cls._optional_int(request, "output_compression")
        if output_format in {"jpeg", "webp"} and output_compression is not None:
            payload["output_compression"] = max(0, min(100, output_compression))

        moderation = str(getattr(request, "moderation", "") or "").lower().strip()
        if moderation in {"auto", "low"}:
            payload["moderation"] = moderation

        # Only forward input_fidelity for edit requests on older GPT Image
        # models where the request schema exposes it. GPT Image 2 rejects
        # this parameter, and the GPT Image 2.5 guide does not document a
        # caller-selectable input_fidelity setting.
        input_fidelity = str(getattr(request, "input_fidelity", "") or "").lower().strip()
        if (
            for_edit
            and input_fidelity in {"low", "high"}
            and not cls._is_gpt_image_2_model(model)
            and not cls._is_gpt_image_25_model(model)
        ):
            payload["input_fidelity"] = input_fidelity

        return payload

    @staticmethod
    def _multipart_scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @classmethod
    def _image_mime_type(
        cls,
        request: ImageGenerationRequest,
        payload: dict[str, Any],
    ) -> str:
        if str(request.model or "").lower().startswith("dall-e"):
            return "image/png"

        output_format = str(payload.get("output_format") or "png").lower()
        if output_format == "jpeg":
            return "image/jpeg"
        if output_format == "webp":
            return "image/webp"
        return "image/png"

    @staticmethod
    def _image_metadata(
        request: ImageGenerationRequest,
        payload: dict[str, Any],
        response: dict[str, Any],
        revised_prompts: list[str],
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "model": str(request.model or ""),
            "size": str(payload.get("size") or request.size or ""),
            "quality": str(payload.get("quality") or request.quality or ""),
        }

        dimensions = str(payload.get("size") or request.size or "").split("x", 1)
        if len(dimensions) == 2 and all(value.isdigit() for value in dimensions):
            metadata["width"] = int(dimensions[0])
            metadata["height"] = int(dimensions[1])

        output_format = payload.get("output_format")
        if output_format:
            metadata["output_format"] = str(output_format)

        background = payload.get("background")
        if background:
            metadata["background"] = str(background)

        if revised_prompts:
            metadata["revised_prompts"] = tuple(revised_prompts)

        usage = response.get("usage")
        if isinstance(usage, dict):
            metadata["usage"] = usage

        return metadata

    def generate_speech(self, request: VoiceGenerationRequest) -> ProviderOutput:
        payload: dict[str, Any] = {
            "model": request.model,
            "input": request.text,
            "voice": request.voice,
            "response_format": request.output_format,
            "speed": request.speed,
        }
        if request.instructions and request.model not in {"tts-1", "tts-1-hd"}:
            payload["instructions"] = request.instructions
        content, request_id, content_type = self._request_bytes(
            "POST",
            "/audio/speech",
            json.dumps(payload).encode("utf-8"),
            content_type="application/json",
            idempotency_key=request.idempotency_key,
        )
        expected = {
            "mp3": "audio/mpeg",
            "opus": "audio/ogg",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/L16",
        }[request.output_format]
        return ProviderOutput(
            provider_request_id=request_id,
            status=GenerationStatus.COMPLETED,
            content=(content,),
            mime_types=((content_type.split(";", 1)[0] if content_type.startswith("audio/") else expected),),
        )

    def generate_video(
        self,
        request: VideoGenerationRequest,
        reference_artifacts: tuple[MediaArtifact, ...] = (),
    ) -> ProviderOutput:
        orientation = self._resolve_orientation(request.aspect_ratio, request.resolution)
        if request.aspect_ratio and orientation == "landscape":
            resolved_resolution = (
                "1792x1024" if request.resolution == "1024x1792" else "1280x720"
            )
        elif request.aspect_ratio and orientation == "portrait":
            resolved_resolution = (
                "1024x1792" if request.resolution == "1792x1024" else "720x1280"
            )
        elif request.resolution in {
            "720x1280",
            "1280x720",
            "1024x1792",
            "1792x1024",
        }:
            resolved_resolution = request.resolution
        elif orientation == "landscape":
            resolved_resolution = "1280x720"
        else:
            resolved_resolution = "720x1280"

        duration = request.duration_seconds
        if duration not in {4, 8, 12}:
            if duration <= 5:
                duration = 4
            elif duration <= 10:
                duration = 8
            else:
                duration = 12

        fields = {
            "model": request.model,
            "prompt": request.prompt,
            "seconds": str(duration),
            "size": resolved_resolution,
        }
        if len(reference_artifacts) > 1:
            raise MediaProviderError(
                "media_reference_count_rejected",
                "The selected provider accepts one video image reference per request.",
            )
        files: tuple[tuple[str, str, str, bytes], ...] = ()
        if reference_artifacts:
            reference = reference_artifacts[0]
            files = (
                (
                    "input_reference",
                    Path(reference.local_path).name,
                    reference.mime_type,
                    self._reference_bytes(reference),
                ),
            )
        body, boundary = self._multipart(fields, files=files)
        payload, request_id, _ = self._request_json_bytes(
            "POST",
            "/videos",
            body,
            content_type=f"multipart/form-data; boundary={boundary}",
            idempotency_key=request.idempotency_key,
        )
        provider_id = str(payload.get("id") or request_id).strip()
        if not provider_id:
            raise MediaProviderError(
                "media_provider_invalid_response",
                "The video provider returned no generation identifier.",
            )
        status = self._video_status(payload)
        return ProviderOutput(
            provider_request_id=provider_id,
            status=status,
            progress=self._progress(payload),
            metadata=self._safe_video_metadata(payload),
        )

    def get_generation_status(self, provider_request_id: str) -> ProviderOutput:
        payload, request_id, _ = self._request_json(
            "GET", f"/videos/{urllib.parse.quote(provider_request_id, safe='')}", None
        )
        status = self._video_status(payload)
        return ProviderOutput(
            provider_request_id=str(payload.get("id") or request_id or provider_request_id),
            status=status,
            progress=self._progress(payload),
            metadata=self._safe_video_metadata(payload),
        )

    def cancel_generation(self, provider_request_id: str) -> ProviderOutput:
        raise MediaProviderError(
            "media_cancellation_unsupported",
            "The configured OpenAI video provider does not support cancelling an active generation.",
        )

    def download_result(self, provider_request_id: str) -> ProviderOutput:
        content, request_id, content_type = self._request_bytes(
            "GET",
            f"/videos/{urllib.parse.quote(provider_request_id, safe='')}/content",
            None,
        )
        return ProviderOutput(
            provider_request_id=request_id or provider_request_id,
            status=GenerationStatus.COMPLETED,
            content=(content,),
            mime_types=((content_type.split(";", 1)[0] or "video/mp4"),),
        )

    def download_url(self, url: str) -> tuple[bytes, str]:
        parsed = urllib.parse.urlparse(url)
        unsafe_address = False
        try:
            unsafe_address = not ipaddress.ip_address(parsed.hostname or "").is_global
        except ValueError:
            pass
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.hostname.casefold() == "localhost"
            or parsed.hostname.endswith(".localhost")
            or unsafe_address
        ):
            raise MediaProviderError(
                "media_download_url_invalid",
                "The provider returned an unsafe download URL.",
            )
        content, _, content_type = self._request_bytes("GET", url, None, absolute=True)
        return content, content_type.split(";", 1)[0]

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
        *,
        idempotency_key: str = "",
    ) -> tuple[dict[str, Any], str, str]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        return self._request_json_bytes(
            method,
            path,
            data,
            content_type="application/json",
            idempotency_key=idempotency_key,
        )

    def _request_json_bytes(
        self,
        method: str,
        path: str,
        data: bytes | None,
        *,
        content_type: str,
        idempotency_key: str = "",
    ) -> tuple[dict[str, Any], str, str]:
        body, request_id, response_type = self._request_bytes(
            method,
            path,
            data,
            content_type=content_type,
            idempotency_key=idempotency_key,
        )
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MediaProviderError(
                "media_provider_invalid_response",
                "The media provider returned an invalid response.",
            ) from exc
        if not isinstance(payload, dict):
            raise MediaProviderError(
                "media_provider_invalid_response",
                "The media provider returned an invalid response.",
            )
        return payload, request_id, response_type

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
        headers = {"Authorization": f"Bearer {self._api_key}"}
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

    @staticmethod
    def _multipart(
        fields: dict[str, str],
        *,
        files: tuple[tuple[str, str, str, bytes], ...] = (),
    ) -> tuple[bytes, str]:
        boundary = f"mana-{secrets.token_hex(16)}"
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    str(value).encode("utf-8"),
                    b"\r\n",
                ]
            )
        for field_name, filename, mime_type, content in files:
            safe_filename = Path(filename).name.replace('"', "")
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    (
                        f'Content-Disposition: form-data; name="{field_name}"; '
                        f'filename="{safe_filename}"\r\n'
                    ).encode(),
                    f"Content-Type: {mime_type}\r\n\r\n".encode(),
                    content,
                    b"\r\n",
                ]
            )
        chunks.append(f"--{boundary}--\r\n".encode())
        return b"".join(chunks), boundary

    @staticmethod
    def _reference_bytes(artifact: MediaArtifact) -> bytes:
        try:
            return Path(artifact.local_path).read_bytes()
        except OSError as exc:
            raise MediaProviderError(
                "media_reference_unavailable",
                "The managed reference artifact could not be read.",
            ) from exc

    @staticmethod
    def _progress(payload: dict[str, Any]) -> float | None:
        value = payload.get("progress")
        try:
            progress = float(value)
        except (TypeError, ValueError):
            return None
        return min(1.0, max(0.0, progress / 100 if progress > 1 else progress))

    @staticmethod
    def _video_status(payload: dict[str, Any]) -> GenerationStatus:
        value = str(payload.get("status") or "").strip()
        try:
            return _VIDEO_STATUS[value]
        except KeyError as exc:
            raise MediaProviderError(
                "media_provider_invalid_response",
                "The video provider returned an unknown generation status.",
            ) from exc

    @staticmethod
    def _safe_video_metadata(payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {"created_at", "completed_at", "expires_at", "seconds", "size"}
        result = {key: payload[key] for key in allowed if key in payload}
        try:
            result["duration_seconds"] = float(payload["seconds"])
        except (KeyError, TypeError, ValueError):
            pass
        size = str(payload.get("size") or "")
        dimensions = size.split("x", 1)
        if len(dimensions) == 2 and all(value.isdigit() for value in dimensions):
            result["width"] = int(dimensions[0])
            result["height"] = int(dimensions[1])
        return result
