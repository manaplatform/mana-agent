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
        payload = self._image_payload(request)
        if reference_artifacts:
            if request.model == "dall-e-3":
                raise MediaProviderError(
                    "media_reference_unsupported",
                    "DALL-E 3 does not support reference-image editing.",
                )
            if len(reference_artifacts) != 1:
                raise MediaProviderError(
                    "media_reference_count_rejected",
                    "The selected provider accepts one image reference per request.",
                )
            reference = reference_artifacts[0]
            multipart_fields = {
                key: str(value)
                for key, value in payload.items()
                if key not in {"n"} or int(value) != 1
            }
            body, boundary = self._multipart(
                multipart_fields,
                files=(
                    (
                        "image",
                        Path(reference.local_path).name,
                        reference.mime_type,
                        self._reference_bytes(reference),
                    ),
                ),
            )
            response, request_id, _ = self._request_json_bytes(
                "POST",
                "/images/edits",
                body,
                content_type=f"multipart/form-data; boundary={boundary}",
                idempotency_key=request.idempotency_key,
            )
        else:
            response, request_id, _ = self._request_json(
                "POST",
                "/images/generations",
                payload,
                idempotency_key=request.idempotency_key,
            )
        content: list[bytes] = []
        urls: list[str] = []
        for item in response.get("data") or []:
            if not isinstance(item, dict):
                continue
            encoded = str(item.get("b64_json") or "")
            if encoded:
                try:
                    content.append(base64.b64decode(encoded, validate=True))
                except ValueError as exc:
                    raise MediaProviderError(
                        "media_provider_invalid_output",
                        "The image provider returned invalid encoded output.",
                    ) from exc
            elif item.get("url"):
                urls.append(str(item["url"]))
        if not content and not urls:
            raise MediaProviderError(
                "media_provider_empty_output",
                "The image provider returned no downloadable output.",
            )
        mime = f"image/{'jpeg' if request.output_format == 'jpeg' else request.output_format}"
        dimensions = request.size.split("x", 1)
        metadata: dict[str, Any] = {}
        if len(dimensions) == 2 and all(value.isdigit() for value in dimensions):
            metadata = {"width": int(dimensions[0]), "height": int(dimensions[1])}
        return ProviderOutput(
            provider_request_id=request_id,
            status=GenerationStatus.COMPLETED,
            content=tuple(content),
            mime_types=tuple(mime for _ in content),
            remote_urls=tuple(urls),
            metadata=metadata,
        )

    @staticmethod
    def _image_payload(request: ImageGenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "n": request.count,
        }
        if request.model.startswith("dall-e"):
            if request.output_format != "png" or request.background:
                raise MediaProviderError(
                    "media_provider_parameter_rejected",
                    "DALL-E models require PNG output and do not support background control.",
                )
            if request.model == "dall-e-3" and request.count != 1:
                raise MediaProviderError(
                    "media_provider_parameter_rejected",
                    "DALL-E 3 accepts one image per request.",
                )
            allowed_sizes = (
                {"256x256", "512x512", "1024x1024"}
                if request.model == "dall-e-2"
                else {"1024x1024", "1024x1792", "1792x1024"}
            )
            if request.size != "auto" and request.size not in allowed_sizes:
                raise MediaProviderError(
                    "media_provider_parameter_rejected",
                    "The selected DALL-E model does not support the requested size.",
                )
            if request.model == "dall-e-2" and request.quality != "auto":
                raise MediaProviderError(
                    "media_provider_parameter_rejected",
                    "DALL-E 2 does not accept a quality setting.",
                )
            if request.model == "dall-e-3" and request.quality not in {
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
            if request.quality != "auto" and request.model == "dall-e-3":
                payload["quality"] = request.quality
            return payload
        if request.size not in {"auto", "1024x1024", "1024x1536", "1536x1024"}:
            raise MediaProviderError(
                "media_provider_parameter_rejected",
                "The GPT image model does not support the requested size.",
            )
        if request.quality not in {"auto", "low", "medium", "high"}:
            raise MediaProviderError(
                "media_provider_parameter_rejected",
                "GPT image quality must be auto, low, medium, or high.",
            )
        payload.update(
            {
                "size": request.size,
                "quality": request.quality,
                "output_format": request.output_format,
            }
        )
        if request.background:
            payload["background"] = request.background
        return payload

    def generate_speech(self, request: VoiceGenerationRequest) -> ProviderOutput:
        if request.instructions and request.model in {"tts-1", "tts-1-hd"}:
            raise MediaProviderError(
                "media_provider_parameter_rejected",
                "The selected TTS model does not support voice instructions.",
            )
        payload: dict[str, Any] = {
            "model": request.model,
            "input": request.text,
            "voice": request.voice,
            "response_format": request.output_format,
            "speed": request.speed,
        }
        if request.instructions:
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
        if request.aspect_ratio:
            raise MediaProviderError(
                "media_provider_parameter_rejected",
                "The OpenAI video endpoint selects framing through resolution and does not accept a separate aspect ratio.",
            )
        if request.duration_seconds not in {4, 8, 12}:
            raise MediaProviderError(
                "media_provider_duration_rejected",
                "The provider accepts video durations of 4, 8, or 12 seconds.",
            )
        if request.resolution not in {
            "720x1280",
            "1280x720",
            "1024x1792",
            "1792x1024",
        }:
            raise MediaProviderError(
                "media_provider_resolution_rejected",
                "The provider rejected the requested video resolution.",
            )
        fields = {
            "model": request.model,
            "prompt": request.prompt,
            "seconds": str(request.duration_seconds),
            "size": request.resolution,
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
