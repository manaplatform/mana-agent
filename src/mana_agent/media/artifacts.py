from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from mana_agent.media.errors import MediaArtifactError
from mana_agent.media.models import MediaArtifact, MediaType
from mana_agent.workspaces.paths import mana_home


_MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/L16": ".pcm",
    "video/mp4": ".mp4",
    "application/octet-stream": ".bin",
}


def _detected_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"ID3") or data[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
        return "audio/mpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "audio/wav"
    if data.startswith(b"fLaC"):
        return "audio/flac"
    if data.startswith(b"OggS"):
        return "audio/ogg"
    if len(data) > 12 and data[4:8] == b"ftyp":
        return "video/mp4"
    return "application/octet-stream"


class MediaArtifactStore:
    """User-level metadata storage with an optional workspace image output."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        image_output_root: Path | None = None,
    ) -> None:
        self.root = (root or (mana_home() / "artifacts" / "media")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.image_output_root = (
            image_output_root.expanduser().resolve()
            if image_output_root is not None
            else None
        )
        if self.image_output_root is not None and not self.image_output_root.is_dir():
            raise MediaArtifactError(
                "media_image_output_root_invalid",
                "The image output directory does not exist.",
            )

    def save(
        self,
        *,
        generation_id: str,
        media_type: MediaType,
        data: bytes,
        declared_mime: str,
        max_bytes: int,
        index: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> MediaArtifact:
        if not data:
            raise MediaArtifactError(
                "media_output_empty", "The provider returned an empty media output."
            )
        if len(data) > max_bytes:
            raise MediaArtifactError(
                "media_output_too_large",
                "The generated output exceeds the configured maximum size.",
            )
        detected = _detected_mime(data)
        expected_prefix = {
            MediaType.IMAGE: "image/",
            MediaType.VOICE: "audio/",
            MediaType.VIDEO: "video/",
        }[media_type]
        declared = str(declared_mime or "").split(";", 1)[0]
        effective = detected if detected != "application/octet-stream" else declared
        if not effective.startswith(expected_prefix):
            raise MediaArtifactError(
                "media_output_mime_invalid",
                f"The generated output is not valid {media_type.value} media.",
            )
        if detected != "application/octet-stream" and declared and declared != detected:
            compatible_audio = {declared, detected} <= {"audio/ogg", "audio/opus"}
            if not compatible_audio:
                raise MediaArtifactError(
                    "media_output_mime_mismatch",
                    "The generated output MIME type does not match its content.",
                )
        extension = _MIME_EXTENSIONS.get(effective)
        if extension is None:
            raise MediaArtifactError(
                "media_output_mime_unsupported",
                f"Generated MIME type {effective!r} is not supported.",
            )
        digest = hashlib.sha256(data).hexdigest()
        artifact_id = f"media_{digest[:24]}"
        metadata_directory = self._generation_dir(generation_id)
        metadata_directory.mkdir(parents=True, exist_ok=True)
        if media_type is MediaType.IMAGE and self.image_output_root is not None:
            job_fragment = generation_id.removeprefix("media_")
            destination = (
                self.image_output_root
                / f"media_{job_fragment}-{index + 1}-{digest[:12]}{extension}"
            )
        else:
            destination = (
                metadata_directory
                / f"{media_type.value}-{index + 1}-{digest[:12]}{extension}"
            )
        self._atomic_write(destination, data)
        artifact = MediaArtifact(
            artifact_id=artifact_id,
            local_path=str(destination),
            mime_type=effective,
            size_bytes=len(data),
            sha256=digest,
            width=(metadata or {}).get("width"),
            height=(metadata or {}).get("height"),
            duration_seconds=(metadata or {}).get("duration_seconds"),
        )
        self._atomic_json(
            metadata_directory / f"{artifact_id}.json",
            artifact.model_dump(mode="json"),
        )
        return artifact

    def load(self, artifact_id: str, *, session_id: str = "") -> MediaArtifact:
        self._validate_opaque_id(artifact_id)
        if session_id:
            self._validate_opaque_id(session_id)
            for record in (self.root / session_id).glob("*/generation.json"):
                try:
                    payload = json.loads(record.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                for item in payload.get("artifacts") or []:
                    if isinstance(item, dict) and item.get("artifact_id") == artifact_id:
                        return self._validated_artifact(item)
            raise MediaArtifactError(
                "media_artifact_not_found",
                "The requested media artifact was not found in this session.",
            )
        candidates = list(self.root.glob(f"*/*/{artifact_id}.json"))
        if len(candidates) != 1:
            raise MediaArtifactError(
                "media_artifact_not_found", "The requested media artifact was not found."
            )
        try:
            payload = json.loads(candidates[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MediaArtifactError(
                "media_artifact_corrupt", "The media artifact metadata is invalid."
            ) from exc
        return self._validated_artifact(payload)

    def _validated_artifact(self, payload: dict[str, Any]) -> MediaArtifact:
        try:
            artifact = MediaArtifact.model_validate(payload)
        except ValueError as exc:
            raise MediaArtifactError(
                "media_artifact_corrupt", "The media artifact metadata is invalid."
            ) from exc
        path = Path(artifact.local_path).resolve()
        self._confine_artifact(path)
        if not path.is_file():
            raise MediaArtifactError(
                "media_artifact_not_found", "The requested media artifact was not found."
            )
        return artifact

    def save_generation(self, session_id: str, result: dict[str, Any]) -> Path:
        generation_id = str(result.get("generation_id") or "")
        self._validate_opaque_id(session_id)
        self._validate_opaque_id(generation_id)
        directory = self.root / session_id / generation_id
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "generation.json"
        self._atomic_json(destination, result)
        return destination

    def load_generation(self, session_id: str, generation_id: str) -> dict[str, Any]:
        self._validate_opaque_id(session_id)
        self._validate_opaque_id(generation_id)
        path = (self.root / session_id / generation_id / "generation.json").resolve()
        self._confine(path)
        if not path.is_file():
            raise MediaArtifactError(
                "media_generation_not_found", "The requested media generation was not found."
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MediaArtifactError(
                "media_generation_corrupt", "The media generation record is invalid."
            ) from exc
        if not isinstance(payload, dict):
            raise MediaArtifactError(
                "media_generation_corrupt", "The media generation record is invalid."
            )
        return payload

    def cleanup_expired(self, *, retention_days: int) -> int:
        """Remove only completed managed generation directories older than policy."""
        cutoff = time.time() - max(1, int(retention_days)) * 86400
        removed = 0
        for session in self.root.iterdir():
            if not session.is_dir() or session.name == "_objects":
                continue
            for generation in session.iterdir():
                record = generation / "generation.json"
                if not generation.is_dir() or not record.is_file() or record.stat().st_mtime >= cutoff:
                    continue
                try:
                    payload = json.loads(record.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if payload.get("status") not in {"completed", "failed", "cancelled"}:
                    continue
                for artifact in payload.get("artifacts") or []:
                    if not isinstance(artifact, dict):
                        continue
                    local_path = Path(str(artifact.get("local_path") or "")).resolve()
                    self._confine_artifact(local_path)
                    local_path.unlink(missing_ok=True)
                object_directory = self._generation_dir(generation.name).resolve()
                self._confine(object_directory)
                if object_directory.is_dir():
                    shutil.rmtree(object_directory)
                self._confine(generation.resolve())
                shutil.rmtree(generation)
                removed += 1
        return removed

    def _generation_dir(self, generation_id: str) -> Path:
        self._validate_opaque_id(generation_id)
        # Generation IDs include their owning session in the persisted path at
        # service level; content-addressed search keeps writes private here.
        return self.root / "_objects" / generation_id

    def _confine(self, path: Path) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise MediaArtifactError(
                "media_artifact_path_invalid", "Media artifact path escapes managed storage."
            ) from exc

    def _confine_artifact(self, path: Path) -> None:
        try:
            path.relative_to(self.root)
            return
        except ValueError:
            pass
        if self.image_output_root is not None:
            if (
                path.parent == self.image_output_root
                and path.name.startswith("media_")
                and path.suffix.lower() in {".png", ".jpg", ".webp", ".gif"}
            ):
                return
        raise MediaArtifactError(
            "media_artifact_path_invalid",
            "Media artifact path escapes managed output locations.",
        )

    @staticmethod
    def _validate_opaque_id(value: str) -> None:
        if not value or "/" in value or "\\" in value or ".." in value:
            raise MediaArtifactError(
                "media_identifier_invalid", "Media identifiers must be opaque values."
            )

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        temp = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        except OSError as exc:
            temp.unlink(missing_ok=True)
            raise MediaArtifactError(
                "media_artifact_write_failed",
                "The generated output could not be saved to managed artifact storage.",
            ) from exc

    @classmethod
    def _atomic_json(cls, path: Path, payload: dict[str, Any]) -> None:
        cls._atomic_write(
            path,
            (json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n").encode("utf-8"),
        )
