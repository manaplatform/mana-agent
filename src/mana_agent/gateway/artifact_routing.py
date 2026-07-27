"""Shared, provenance-aware evidence for artifact routing.

This module deliberately does not select a lane from user text.  It prepares
validated attachment and target evidence for the entry-routing model, which is
the authority that chooses the ``artifact`` route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Literal, Sequence


ArtifactFamily = Literal["spreadsheet", "document", "presentation", "pdf", "image", "conversion"]


@dataclass(frozen=True, slots=True)
class ArtifactHandler:
    family: ArtifactFamily
    extensions: tuple[str, ...]
    mime_prefixes: tuple[str, ...]
    handler: str
    tools: tuple[str, ...]


ARTIFACT_HANDLERS: tuple[ArtifactHandler, ...] = (
    ArtifactHandler("spreadsheet", (".xls", ".xlsx", ".xlsm", ".csv", ".ods"), ("application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml", "text/csv", "application/vnd.oasis.opendocument.spreadsheet"), "spreadsheet", ("document_detect", "document_read", "document_create", "document_update")),
    ArtifactHandler("document", (".doc", ".docx", ".odt", ".rtf", ".txt"), ("application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml", "application/vnd.oasis.opendocument.text", "application/rtf", "text/plain"), "document", ("document_detect", "document_read", "document_create", "document_update")),
    ArtifactHandler("presentation", (".ppt", ".pptx", ".odp"), ("application/vnd.ms-powerpoint", "application/vnd.openxmlformats-officedocument.presentationml", "application/vnd.oasis.opendocument.presentation"), "presentation", ()),
    ArtifactHandler("pdf", (".pdf",), ("application/pdf",), "pdf", ("document_detect", "document_read", "document_create", "document_update")),
    ArtifactHandler("image", (".png", ".jpg", ".jpeg", ".webp", ".gif"), ("image/",), "image", ()),
)

_FILENAME = re.compile(r"(?<![\w.-])([\w][\w .-]{0,180}\.[A-Za-z0-9]{1,8})(?![\w.-])")


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    path: str
    filename: str
    mime_type: str
    provenance: str
    repository_member: bool
    extension: str
    family: ArtifactFamily | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def artifact_routing_evidence(
    *, root: Path, user_prompt: str, attachments: Sequence[object] = (), target_files: Sequence[object] = ()
) -> dict[str, Any]:
    """Return redacted, centralized artifact evidence for a model decision."""
    references = [
        _reference(value, root=root, provenance="attachment") for value in attachments
    ] + [
        _reference(value, root=root, provenance="target") for value in target_files
    ]
    known = {item.filename.casefold() for item in references if item.filename}
    for filename in _FILENAME.findall(user_prompt):
        if filename.casefold() not in known:
            references.append(_reference(filename, root=root, provenance="mentioned"))
            known.add(filename.casefold())
    families = sorted({item.family for item in references if item.family})
    return {
        "references": [item.to_dict() for item in references],
        "artifact_families": families,
        "detected_extensions": sorted({item.extension for item in references if item.extension}),
        "has_user_artifact": any(item.provenance == "attachment" and not item.repository_member for item in references),
        "available_handlers": [
            {"family": handler.family, "handler": handler.handler, "tools": list(handler.tools)}
            for handler in ARTIFACT_HANDLERS
        ],
    }


def artifact_handler_availability(evidence: dict[str, Any]) -> tuple[bool, str]:
    """Validate registered handler support before model or lock reservation."""
    families = set(evidence.get("artifact_families") or [])
    if not families:
        return False, "No supported artifact family was resolved from the request."
    handlers = {handler.family: handler for handler in ARTIFACT_HANDLERS}
    unavailable = sorted(family for family in families if not handlers[family].tools)
    if unavailable:
        return False, f"No configured artifact handler can execute: {', '.join(unavailable)}."
    return True, ""


def _reference(value: object, *, root: Path, provenance: str) -> ArtifactReference:
    payload = value if isinstance(value, dict) else {"path": str(value)}
    path_text = str(payload.get("path") or payload.get("filename") or payload.get("name") or "").strip()
    filename = Path(path_text).name
    mime_type = str(payload.get("mime_type") or payload.get("mime") or "").strip().lower()
    extension = Path(filename).suffix.lower()
    resolved = Path(path_text).expanduser().resolve() if path_text else None
    repository_member = bool(resolved and (resolved == root or root in resolved.parents))
    family = next((handler.family for handler in ARTIFACT_HANDLERS if extension in handler.extensions or any(mime_type.startswith(prefix) for prefix in handler.mime_prefixes)), None)
    return ArtifactReference(path=path_text, filename=filename, mime_type=mime_type, provenance=provenance, repository_member=repository_member, extension=extension, family=family)
