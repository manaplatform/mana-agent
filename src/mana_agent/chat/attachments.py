"""Canonical chat attachment models, validation, and storage for Mana-Agent.

Provides a single shared attachment pipeline for TUI, Dashboard, API,
gateway, and session persistence.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from mana_agent.config.user_config import get_setting
from mana_agent.workspaces.paths import session_dir


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AttachmentCategory(str, Enum):
    IMAGE = "image"
    DOCUMENT = "document"
    CODE = "code"
    AUDIO = "audio"
    VIDEO = "video"
    OTHER = "other"


class AttachmentStatus(str, Enum):
    READY = "ready"
    ERROR = "error"
    EXPIRED = "expired"


# Default limits
DEFAULT_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
DEFAULT_MAX_ATTACHMENT_COUNT = 10
DEFAULT_MAX_TOTAL_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

# Magic bytes signatures for MIME verification
_MAGIC_SIGNATURES: list[tuple[bytes, str, AttachmentCategory]] = [
    # Images
    (b"\x89PNG\r\n\x1a\n", "image/png", AttachmentCategory.IMAGE),
    (b"\xff\xd8\xff", "image/jpeg", AttachmentCategory.IMAGE),
    (b"GIF87a", "image/gif", AttachmentCategory.IMAGE),
    (b"GIF89a", "image/gif", AttachmentCategory.IMAGE),
    # PDF
    (b"%PDF-", "application/pdf", AttachmentCategory.DOCUMENT),
    # Audio
    (b"ID3", "audio/mp3", AttachmentCategory.AUDIO),
    (b"\xff\xfb", "audio/mp3", AttachmentCategory.AUDIO),
    (b"\xff\xf3", "audio/mp3", AttachmentCategory.AUDIO),
    (b"\xff\xf2", "audio/mp3", AttachmentCategory.AUDIO),
    (b"OggS", "audio/ogg", AttachmentCategory.AUDIO),
]

_CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".scss",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".java", ".kt",
    ".rb", ".php", ".sh", ".bash", ".zsh", ".sql", ".yaml", ".yml",
    ".toml", ".ini", ".xml", ".dockerfile", ".r", ".swift", ".scala",
})

_DOCUMENT_EXTENSIONS = frozenset({
    ".pdf", ".txt", ".md", ".json", ".csv", ".tsv", ".rst", ".log",
    ".docx", ".xlsx", ".pptx",
})

_MEDIA_EXTENSIONS = {
    ".png": ("image/png", AttachmentCategory.IMAGE),
    ".jpg": ("image/jpeg", AttachmentCategory.IMAGE),
    ".jpeg": ("image/jpeg", AttachmentCategory.IMAGE),
    ".webp": ("image/webp", AttachmentCategory.IMAGE),
    ".gif": ("image/gif", AttachmentCategory.IMAGE),
    ".svg": ("image/svg+xml", AttachmentCategory.IMAGE),
    ".mp3": ("audio/mp3", AttachmentCategory.AUDIO),
    ".wav": ("audio/wav", AttachmentCategory.AUDIO),
    ".ogg": ("audio/ogg", AttachmentCategory.AUDIO),
    ".m4a": ("audio/m4a", AttachmentCategory.AUDIO),
    ".mp4": ("video/mp4", AttachmentCategory.VIDEO),
    ".webm": ("video/webm", AttachmentCategory.VIDEO),
    ".mov": ("video/quicktime", AttachmentCategory.VIDEO),
}


@dataclass(frozen=True)
class ChatAttachment:
    """Canonical chat attachment model shared across all frontends and services."""

    attachment_id: str
    filename: str
    mime_type: str
    size_bytes: int
    category: str
    storage_path: str
    preview_metadata: dict[str, Any] = field(default_factory=dict)
    status: str = AttachmentStatus.READY.value
    error_message: str | None = None
    created_at: str = field(default_factory=_utc_now)
    sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatAttachment":
        return cls(
            attachment_id=str(data.get("attachment_id") or f"att_{uuid.uuid4().hex[:16]}"),
            filename=str(data.get("filename") or "unnamed_file"),
            mime_type=str(data.get("mime_type") or "application/octet-stream"),
            size_bytes=int(data.get("size_bytes") or 0),
            category=str(data.get("category") or AttachmentCategory.OTHER.value),
            storage_path=str(data.get("storage_path") or ""),
            preview_metadata=dict(data.get("preview_metadata") or {}),
            status=str(data.get("status") or AttachmentStatus.READY.value),
            error_message=data.get("error_message"),
            created_at=str(data.get("created_at") or _utc_now()),
            sha256=str(data.get("sha256") or ""),
        )

    def is_image(self) -> bool:
        return self.category == AttachmentCategory.IMAGE.value

    def is_text_or_code(self) -> bool:
        return self.category in {
            AttachmentCategory.DOCUMENT.value,
            AttachmentCategory.CODE.value,
        }

    def format_size(self) -> str:
        """Human-readable size format."""
        return format_size(self.size_bytes)


def format_size(size_bytes: int) -> str:
    """Human-readable size format for bytes."""
    b = int(size_bytes or 0)
    if b < 1024:
        return f"{b} B"
    elif b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    else:
        return f"{b / (1024 * 1024):.1f} MB"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and shell injection."""
    basename = Path(filename).name
    # Strip dangerous characters, keep alphanumerics, dots, hyphens, underscores
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", basename)
    # Remove leading dots to avoid hidden files or directory traversals
    sanitized = sanitized.lstrip(".").strip()
    return sanitized or f"file_{uuid.uuid4().hex[:8]}"


def detect_mime_and_category(
    filename: str, content: bytes
) -> tuple[str, AttachmentCategory]:
    """Validate MIME type and category using magic bytes and file content."""
    ext = Path(filename).suffix.lower()

    # 1. Check magic bytes
    for magic, mime, cat in _MAGIC_SIGNATURES:
        if content.startswith(magic):
            return mime, cat

    # RIFF container (WEBP or WAV)
    if content.startswith(b"RIFF") and len(content) >= 12:
        tag = content[8:12]
        if tag == b"WEBP":
            return "image/webp", AttachmentCategory.IMAGE
        if tag == b"WAVE":
            return "audio/wav", AttachmentCategory.AUDIO

    # MP4 / M4A (ftyp box)
    if len(content) >= 12 and content[4:8] == b"ftyp":
        major_brand = content[8:12].lower()
        if major_brand in {b"m4a ", b"mp42", b"isom"}:
            if ext == ".m4a":
                return "audio/m4a", AttachmentCategory.AUDIO
            return "video/mp4", AttachmentCategory.VIDEO

    # 2. Check known media extensions
    if ext in _MEDIA_EXTENSIONS:
        expected_mime, cat = _MEDIA_EXTENSIONS[ext]
        # Basic check for image: reject obviously binary executables disguised as image
        if cat == AttachmentCategory.IMAGE and content.startswith(b"MZ"):
            raise ValueError(f"File content does not match image extension '{ext}'")
        return expected_mime, cat

    # 3. Code files
    if ext in _CODE_EXTENSIONS:
        _check_text_content(content, filename)
        mime = mimetypes.guess_type(filename)[0] or "text/plain"
        return mime, AttachmentCategory.CODE

    # 4. Document files
    if ext in _DOCUMENT_EXTENSIONS:
        if ext == ".pdf":
            # PDF was already checked via magic bytes. If not matching %PDF-, reject!
            if not content.startswith(b"%PDF-"):
                raise ValueError("Corrupted or invalid PDF document")
            return "application/pdf", AttachmentCategory.DOCUMENT
        _check_text_content(content, filename)
        mime = mimetypes.guess_type(filename)[0] or "text/plain"
        return mime, AttachmentCategory.DOCUMENT

    # 5. Generic text check
    try:
        content[:4096].decode("utf-8")
        return "text/plain", AttachmentCategory.DOCUMENT
    except UnicodeDecodeError:
        pass

    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return mime, AttachmentCategory.OTHER


def _check_text_content(content: bytes, filename: str) -> None:
    """Ensure a supposedly text/code file does not contain null bytes."""
    sample = content[:8192]
    if b"\x00" in sample:
        raise ValueError(f"Binary content detected in text file '{filename}'")
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"File '{filename}' must be valid UTF-8 text") from exc


class AttachmentValidator:
    """Configurable limits and security validator for chat attachments."""

    def __init__(
        self,
        max_file_size_bytes: int | None = None,
        max_count: int | None = None,
        max_total_size_bytes: int | None = None,
    ) -> None:
        self.max_file_size_bytes = int(
            max_file_size_bytes
            or get_setting("MANA_ATTACHMENT_MAX_FILE_SIZE_BYTES", DEFAULT_MAX_FILE_SIZE_BYTES)
            or DEFAULT_MAX_FILE_SIZE_BYTES
        )
        self.max_count = int(
            max_count
            or get_setting("MANA_ATTACHMENT_MAX_COUNT", DEFAULT_MAX_ATTACHMENT_COUNT)
            or DEFAULT_MAX_ATTACHMENT_COUNT
        )
        self.max_total_size_bytes = int(
            max_total_size_bytes
            or get_setting("MANA_ATTACHMENT_MAX_TOTAL_SIZE_BYTES", DEFAULT_MAX_TOTAL_SIZE_BYTES)
            or DEFAULT_MAX_TOTAL_SIZE_BYTES
        )

    def validate_file(
        self, filename: str, content: bytes, current_count: int = 0, current_total_size: int = 0
    ) -> tuple[str, str, AttachmentCategory]:
        """Validate an attachment before storing.

        Returns (sanitized_filename, mime_type, category).
        Raises ValueError if validation fails.
        """
        if current_count >= self.max_count:
            raise ValueError(f"Maximum attachment limit of {self.max_count} reached")

        size = len(content)
        if size == 0:
            raise ValueError("Attachment file cannot be empty")

        if size > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            raise ValueError(
                f"File '{filename}' ({size / (1024 * 1024):.1f} MB) exceeds limit of {max_mb:.1f} MB"
            )

        if current_total_size + size > self.max_total_size_bytes:
            max_mb = self.max_total_size_bytes / (1024 * 1024)
            raise ValueError(
                f"Total attachment size exceeds maximum allowed of {max_mb:.1f} MB"
            )

        sanitized = sanitize_filename(filename)
        mime, category = detect_mime_and_category(sanitized, content)
        return sanitized, mime, category


class AttachmentStore:
    """Manages file storage for chat attachments under the session directory."""

    @staticmethod
    def get_session_attachments_dir(session_id: str) -> Path:
        target = session_dir(session_id) / "attachments"
        target.mkdir(parents=True, exist_ok=True)
        return target

    @classmethod
    def save(
        cls,
        session_id: str,
        filename: str,
        content: bytes,
        *,
        attachment_id: str | None = None,
        validator: AttachmentValidator | None = None,
    ) -> ChatAttachment:
        """Validate, hash, and persist an attachment file."""
        val = validator or AttachmentValidator()
        sanitized_name, mime, category = val.validate_file(filename, content)

        att_id = attachment_id or f"att_{uuid.uuid4().hex[:16]}"
        target_dir = cls.get_session_attachments_dir(session_id) / att_id
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / sanitized_name

        file_path.write_bytes(content)
        sha = hashlib.sha256(content).hexdigest()

        # Build preview metadata
        preview: dict[str, Any] = {"sha256": sha}
        if category == AttachmentCategory.IMAGE:
            # Try to extract simple dimensions if possible
            preview.update(_extract_image_dimensions(content, mime))
        elif category in {AttachmentCategory.DOCUMENT, AttachmentCategory.CODE}:
            try:
                decoded = content[:2000].decode("utf-8", errors="replace")
                preview["line_count"] = content.count(b"\n") + 1
                preview["snippet"] = decoded[:200]
            except Exception:
                pass

        return ChatAttachment(
            attachment_id=att_id,
            filename=sanitized_name,
            mime_type=mime,
            size_bytes=len(content),
            category=category.value,
            storage_path=str(file_path),
            preview_metadata=preview,
            status=AttachmentStatus.READY.value,
            created_at=_utc_now(),
            sha256=sha,
        )

    @classmethod
    def save_from_path(
        cls,
        session_id: str,
        source_path: str | Path,
        *,
        attachment_id: str | None = None,
        validator: AttachmentValidator | None = None,
    ) -> ChatAttachment:
        """Create attachment from local file path."""
        p = Path(source_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Attachment file not found: {source_path}")
        content = p.read_bytes()
        return cls.save(
            session_id=session_id,
            filename=p.name,
            content=content,
            attachment_id=attachment_id,
            validator=validator,
        )

    @classmethod
    def get_path(cls, session_id: str, attachment_id: str, filename: str) -> Path:
        """Resolve attachment path ensuring path traversal protection."""
        sanitized_filename = sanitize_filename(filename)
        safe_att_id = re.sub(r"[^a-zA-Z0-9_-]", "", attachment_id)
        base = cls.get_session_attachments_dir(session_id) / safe_att_id
        path = (base / sanitized_filename).resolve()
        if not str(path).startswith(str(base.resolve())):
            raise ValueError("Path traversal detected")
        return path


def _extract_image_dimensions(content: bytes, mime: str) -> dict[str, Any]:
    """Lightweight dimension extraction without heavy external dependencies."""
    meta: dict[str, Any] = {}
    try:
        if mime == "image/png" and len(content) >= 24:
            import struct
            width, height = struct.unpack(">II", content[16:24])
            meta["width"] = width
            meta["height"] = height
        elif mime == "image/gif" and len(content) >= 10:
            import struct
            width, height = struct.unpack("<HH", content[6:10])
            meta["width"] = width
            meta["height"] = height
    except Exception:
        pass
    return meta
