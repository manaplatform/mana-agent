from __future__ import annotations

import os
import secrets
import tempfile
from pathlib import Path
from typing import Any, Callable

from mana_agent.config.model_catalog import ModelCapability
from mana_agent.media.artifacts import MediaArtifactStore
from mana_agent.media.config import MediaConfig, MediaModalityConfig
from mana_agent.media.errors import (
    MediaCapabilityError,
    MediaConfigurationError,
    MediaError,
    MediaValidationError,
)
from mana_agent.media.models import (
    GenerationResult,
    GenerationStatus,
    ImageGenerationRequest,
    MediaArtifact,
    MediaType,
    VideoGenerationRequest,
    VoiceGenerationRequest,
    utc_now_iso,
)
from mana_agent.media.providers.base import MediaProvider, ProviderOutput
from mana_agent.media.registry import MEDIA_PROVIDERS, MediaProviderRegistry


EventSink = Callable[..., None]


_REQUIRED_CAPABILITY = {
    MediaType.IMAGE: ModelCapability.IMAGE_GENERATION,
    MediaType.VOICE: ModelCapability.TEXT_TO_SPEECH,
    MediaType.VIDEO: ModelCapability.VIDEO_GENERATION,
}

_PERMISSION = {
    MediaType.IMAGE: "media.image.generate",
    MediaType.VOICE: "media.voice.generate",
    MediaType.VIDEO: "media.video.generate",
}


class MediaService:
    """Provider-neutral generation facade used by gateway, tools, and jobs."""

    def __init__(
        self,
        *,
        config: MediaConfig | None = None,
        artifact_store: MediaArtifactStore | None = None,
        provider_registry: MediaProviderRegistry | None = None,
        event_sink: EventSink | None = None,
        settings_values: dict[str, Any] | None = None,
        workspace_root: str | Path | None = None,
    ) -> None:
        self.config = config or MediaConfig.load(settings_values)
        image_output_root = (
            Path(workspace_root).expanduser().resolve()
            if workspace_root is not None
            else Path.cwd().resolve()
        )
        self.artifacts = artifact_store or MediaArtifactStore(
            image_output_root=image_output_root
        )
        self.providers = provider_registry or MEDIA_PROVIDERS
        self.event_sink = event_sink
        self.settings_values = settings_values

    def availability(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for media_type in MediaType:
            modality = self.config.modality(media_type)
            result[media_type.value] = {
                "enabled": modality.enabled,
                "configured": bool(modality.provider and modality.model),
                "provider": modality.provider,
                "model": modality.model,
            }
        return result

    def generate_image(
        self, request: ImageGenerationRequest, *, session_id: str, turn_id: str = ""
    ) -> GenerationResult:
        return self._generate(
            MediaType.IMAGE,
            request,
            session_id=session_id,
            turn_id=turn_id,
            call=lambda provider, item, references: provider.generate_image(
                item, references
            ),
        )

    def generate_speech(
        self, request: VoiceGenerationRequest, *, session_id: str, turn_id: str = ""
    ) -> GenerationResult:
        return self._generate(
            MediaType.VOICE,
            request,
            session_id=session_id,
            turn_id=turn_id,
            call=lambda provider, item, _references: provider.generate_speech(item),
        )

    def generate_video(
        self, request: VideoGenerationRequest, *, session_id: str, turn_id: str = ""
    ) -> GenerationResult:
        return self._generate(
            MediaType.VIDEO,
            request,
            session_id=session_id,
            turn_id=turn_id,
            call=lambda provider, item, references: provider.generate_video(
                item, references
            ),
        )

    def get_generation_status(
        self, generation_id: str, *, session_id: str, turn_id: str = ""
    ) -> GenerationResult:
        self._require_permission("media.status.read")
        result = GenerationResult.model_validate(
            self.artifacts.load_generation(session_id, generation_id)
        )
        if result.status in {
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
            GenerationStatus.CANCELLED,
        }:
            return result
        modality, provider = self._provider_for_result(result)
        try:
            output = provider.get_generation_status(result.provider_request_id)
            completed_artifacts: tuple[MediaArtifact, ...] = ()
            if output.status is GenerationStatus.COMPLETED:
                downloaded = provider.download_result(result.provider_request_id)
                downloaded.metadata.update(output.metadata)
                completed_artifacts = self._persist_outputs(
                    result.generation_id,
                    result.media_type,
                    downloaded,
                    modality,
                    provider,
                )
        except MediaError as exc:
            failed = result.model_copy(
                update={
                    "status": GenerationStatus.FAILED,
                    "error_code": exc.code,
                    "error_detail": exc.detail,
                    "updated_at": utc_now_iso(),
                }
            )
            self._save(session_id, failed)
            self._emit("media_generation_failed", failed, turn_id=turn_id)
            raise
        updated = result.model_copy(
            update={
                "status": output.status,
                "progress": output.progress,
                "error_code": "media_provider_generation_failed"
                if output.status is GenerationStatus.FAILED
                else "",
                "error_detail": "The media provider reported that generation failed."
                if output.status is GenerationStatus.FAILED
                else "",
                "updated_at": utc_now_iso(),
            }
        )
        if output.status is GenerationStatus.COMPLETED:
            updated = updated.model_copy(
                update={"artifacts": completed_artifacts}
            )
            self._emit("media_generation_completed", updated, turn_id=turn_id)
        elif output.status is GenerationStatus.FAILED:
            self._emit("media_generation_failed", updated, turn_id=turn_id)
        elif output.status is GenerationStatus.CANCELLED:
            self._emit("media_generation_cancelled", updated, turn_id=turn_id)
        else:
            self._emit("media_generation_progress", updated, turn_id=turn_id)
        self._save(session_id, updated)
        return updated

    def cancel_generation(
        self, generation_id: str, *, session_id: str, turn_id: str = ""
    ) -> GenerationResult:
        self._require_permission("media.generation.cancel")
        result = GenerationResult.model_validate(
            self.artifacts.load_generation(session_id, generation_id)
        )
        if result.status in {
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
            GenerationStatus.CANCELLED,
        }:
            return result
        _, provider = self._provider_for_result(result)
        output = provider.cancel_generation(result.provider_request_id)
        updated = result.model_copy(
            update={"status": output.status, "updated_at": utc_now_iso()}
        )
        self._save(session_id, updated)
        self._emit("media_generation_cancelled", updated, turn_id=turn_id)
        return updated

    def get_artifact(self, artifact_id: str, *, session_id: str = "") -> MediaArtifact:
        return self.artifacts.load(artifact_id, session_id=session_id)

    def export_artifact(
        self,
        artifact_id: str,
        *,
        session_id: str,
        workspace_root: str | Path,
        relative_destination: str,
    ) -> Path:
        """Copy an artifact only into an explicitly selected workspace path."""
        self._require_permission("media.artifact.write")
        artifact = self.get_artifact(artifact_id, session_id=session_id)
        root = Path(workspace_root).expanduser().resolve()
        relative = Path(relative_destination)
        if relative.is_absolute() or ".." in relative.parts:
            raise MediaValidationError(
                "media_export_path_invalid",
                "Media export destination must be a relative workspace path.",
            )
        destination = (root / relative).resolve()
        try:
            destination.relative_to(root)
        except ValueError as exc:
            raise MediaValidationError(
                "media_export_path_invalid",
                "Media export destination escapes the active workspace.",
            ) from exc
        source = Path(artifact.local_path)
        if destination.suffix.lower() != source.suffix.lower():
            raise MediaValidationError(
                "media_export_extension_invalid",
                f"Media exports must preserve the {source.suffix} extension.",
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=str(destination.parent)
        )
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as output, source.open("rb") as input_stream:
                for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp, destination)
        except OSError as exc:
            temp.unlink(missing_ok=True)
            raise MediaValidationError(
                "media_export_failed",
                "The media artifact could not be exported to the workspace.",
            ) from exc
        return destination

    def _generate(
        self,
        media_type: MediaType,
        request: Any,
        *,
        session_id: str,
        turn_id: str,
        call: Callable[[MediaProvider, Any, tuple[MediaArtifact, ...]], ProviderOutput],
    ) -> GenerationResult:
        self._require_permission(_PERMISSION[media_type])
        self._require_permission("media.artifact.write")
        modality = self.config.require(media_type)
        model = str(request.model or modality.model).strip()
        reference_artifacts = tuple(
            self.artifacts.load(artifact_id, session_id=session_id)
            for artifact_id in getattr(request, "reference_artifact_ids", ())
        )
        for artifact in reference_artifacts:
            if not artifact.mime_type.startswith("image/"):
                raise MediaValidationError(
                    "media_reference_mime_invalid",
                    "Image and video references must be managed image artifacts.",
                )
        if not model:
            raise MediaConfigurationError(
                f"media_{media_type.value}_not_configured",
                f"{media_type.value.capitalize()} generation model is not configured.",
            )
        if (
            media_type is MediaType.VIDEO
            and modality.max_duration_seconds is not None
            and request.duration_seconds > modality.max_duration_seconds
        ):
            raise MediaValidationError(
                "media_duration_exceeds_limit",
                "The requested video duration exceeds the configured maximum.",
            )
        api_key = self.config.api_key(media_type, self.settings_values)
        if not api_key:
            raise MediaConfigurationError(
                "media_provider_auth_required",
                f"Media provider authentication is not configured for {modality.provider}.",
            )
        provider = self.providers.create(modality, api_key)
        capabilities = provider.capabilities(model)
        required = _REQUIRED_CAPABILITY[media_type]
        if required not in capabilities:
            err_code = (
                "media_image_model_unsupported"
                if media_type is MediaType.IMAGE
                else "media_model_capability_unsupported"
            )
            raise MediaCapabilityError(
                err_code,
                f"The selected model {model!r} does not support {media_type.value} generation.",
            )
        generation_id = f"media_{secrets.token_urlsafe(18)}"
        request = request.model_copy(
            update={
                "model": model,
                "idempotency_key": request.idempotency_key
                or f"{session_id}:{turn_id or generation_id}",
            }
        )
        result = GenerationResult(
            generation_id=generation_id,
            media_type=media_type,
            provider=modality.provider,
            model=model,
            status=GenerationStatus.QUEUED,
            request=self._safe_request(request),
        )
        self._save(session_id, result)
        self._emit("media_generation_requested", result, turn_id=turn_id)
        self._emit("media_generation_queued", result, turn_id=turn_id)
        result = result.model_copy(
            update={"status": GenerationStatus.GENERATING, "updated_at": utc_now_iso()}
        )
        self._emit("media_generation_started", result, turn_id=turn_id)
        try:
            output = call(provider, request, reference_artifacts)
            artifacts: tuple[MediaArtifact, ...] = ()
            if output.status is GenerationStatus.COMPLETED:
                artifacts = self._persist_outputs(
                    generation_id,
                    media_type,
                    output,
                    modality,
                    provider,
                    session_id=session_id,
                    model=model,
                )
            usage_dict = dict(output.metadata.get("usage") or {})
            if "cost" in output.metadata and "cost" not in usage_dict:
                usage_dict["cost"] = output.metadata["cost"]
            if "actual_cost" in output.metadata and "actual_cost" not in usage_dict:
                usage_dict["actual_cost"] = output.metadata["actual_cost"]
            if "image_count" in output.metadata and "image_count" not in usage_dict:
                usage_dict["image_count"] = output.metadata["image_count"]
            if "provider_usage" in output.metadata and "provider_usage" not in usage_dict:
                usage_dict["provider_usage"] = output.metadata["provider_usage"]
            result = result.model_copy(
                update={
                    "status": output.status,
                    "artifacts": artifacts,
                    "provider_request_id": output.provider_request_id,
                    "progress": output.progress,
                    "usage": usage_dict,
                    "error_code": "media_provider_generation_failed"
                    if output.status is GenerationStatus.FAILED
                    else "",
                    "error_detail": "The media provider reported that generation failed."
                    if output.status is GenerationStatus.FAILED
                    else "",
                    "updated_at": utc_now_iso(),
                }
            )
            self._save(session_id, result)
            self._emit(
                "media_generation_completed"
                if result.status is GenerationStatus.COMPLETED
                else "media_generation_failed"
                if result.status is GenerationStatus.FAILED
                else "media_generation_cancelled"
                if result.status is GenerationStatus.CANCELLED
                else "media_generation_progress",
                result,
                turn_id=turn_id,
            )
            return result
        except MediaError as exc:
            failed = result.model_copy(
                update={
                    "status": GenerationStatus.FAILED,
                    "error_code": exc.code,
                    "error_detail": exc.detail,
                    "updated_at": utc_now_iso(),
                }
            )
            self._save(session_id, failed)
            self._emit("media_generation_failed", failed, turn_id=turn_id)
            raise

    def _provider_for_result(
        self, result: GenerationResult
    ) -> tuple[MediaModalityConfig, MediaProvider]:
        modality = self.config.require(result.media_type)
        if modality.provider != result.provider:
            raise MediaConfigurationError(
                "media_generation_configuration_changed",
                "The persisted generation requires its original configured provider.",
            )
        api_key = self.config.api_key(result.media_type, self.settings_values)
        return modality, self.providers.create(modality, api_key)

    def _persist_outputs(
        self,
        generation_id: str,
        media_type: MediaType,
        output: ProviderOutput,
        modality: MediaModalityConfig,
        provider: MediaProvider,
        *,
        session_id: str = "",
        model: str = "",
    ) -> tuple[MediaArtifact, ...]:
        content = list(output.content)
        mime_types = list(output.mime_types)
        for url in output.remote_urls:
            downloader = getattr(provider, "download_url", None)
            if not callable(downloader):
                raise MediaValidationError(
                    "media_download_unsupported",
                    "The output could not be downloaded by the selected provider.",
                )
            data, mime = downloader(url)
            content.append(data)
            mime_types.append(mime)
        if not content:
            raise MediaValidationError(
                "media_output_missing", "The provider returned no media output."
            )
        saved_artifacts: list[MediaArtifact] = []
        meta = {
            **output.metadata,
            "session_id": session_id,
            "provider": modality.provider,
            "model": model or modality.model,
        }
        for index, data in enumerate(content):
            if not data or len(data) == 0:
                raise MediaValidationError(
                    "media_output_empty", "The provider returned empty media content."
                )
            artifact = self.artifacts.save(
                generation_id=generation_id,
                media_type=media_type,
                data=data,
                declared_mime=mime_types[index] if index < len(mime_types) else "",
                max_bytes=modality.max_output_bytes,
                index=index,
                metadata=meta,
            )
            # Verify artifact was persisted and can be read
            artifact_file = Path(artifact.local_path)
            if not artifact_file.is_file() or artifact_file.stat().st_size == 0:
                raise MediaValidationError(
                    "media_artifact_verification_failed",
                    "The generated media artifact could not be verified on disk.",
                )
            saved_artifacts.append(artifact)
        return tuple(saved_artifacts)

    def _save(self, session_id: str, result: GenerationResult) -> None:
        self.artifacts.save_generation(session_id, result.model_dump(mode="json"))

    def _require_permission(self, scope: str) -> None:
        decision = str(self.config.permissions.get(scope) or "deny").strip().lower()
        if decision != "allow":
            raise MediaValidationError(
                "media_permission_denied",
                f"Permission {scope} is not granted.",
            )

    @staticmethod
    def _safe_request(request: Any) -> dict[str, Any]:
        payload = request.model_dump(mode="json")
        payload.pop("idempotency_key", None)
        text = payload.get("text")
        prompt = payload.get("prompt")
        instructions = payload.get("instructions")
        if text:
            payload["text_length"] = len(text)
            payload.pop("text", None)
        if prompt:
            payload["prompt_length"] = len(prompt)
            payload.pop("prompt", None)
        if instructions:
            payload["instructions_length"] = len(instructions)
            payload.pop("instructions", None)
        return payload

    def _emit(self, event_type: str, result: GenerationResult, *, turn_id: str) -> None:
        if not callable(self.event_sink):
            return
        metadata = {
            "job_id": result.generation_id,
            "media_type": result.media_type.value,
            "provider": result.provider,
            "model": result.model,
            "status": result.status.value,
            "progress": result.progress,
            "artifact_id": result.primary_artifact.artifact_id
            if result.primary_artifact
            else "",
            "error_code": result.error_code,
            "error": result.error_detail,
            "turn_id": turn_id,
        }
        try:
            self.event_sink(
                event_type,
                f"{result.media_type.value.title()} generation {result.status.value}",
                metadata=metadata,
            )
        except TypeError:
            try:
                self.event_sink(event_type, metadata)
            except TypeError:
                self.event_sink(
                    {
                        "event_type": event_type,
                        "backend": "media",
                        "model": result.model,
                        "status": result.status.value,
                        "title": f"{result.media_type.value.title()} generation",
                        "metadata": metadata,
                        **metadata,
                    }
                )
