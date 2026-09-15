"""Multimodal input normalization and capability validation for Mana-Agent chat.

Converts canonical ChatAttachment objects into the multimodal representation
expected by the selected model/provider, preserving text + attachment ordering
and enforcing explicit capability checks.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Sequence

from mana_agent.chat.attachments import AttachmentCategory, ChatAttachment
from mana_agent.config.catalog_service import ModelCatalogService
from mana_agent.config.model_capabilities import resolve_model_capability
from mana_agent.config.model_catalog import (
    ModelCapability,
    ModelDescriptor,
    ModelPurpose,
    descriptors_from_catalog,
    extract_capabilities_from_record,
    normalize_capabilities,
)


class UnsupportedAttachmentError(ValueError):
    """Raised when the selected model/provider does not support an attachment."""

    def __init__(self, provider: str, model: str, attachment: ChatAttachment, reason: str = "") -> None:
        detail = reason or (
            f"The selected model '{provider}/{model}' does not support "
            f"{attachment.category} attachments ('{attachment.filename}' with MIME '{attachment.mime_type}')."
        )
        super().__init__(detail)
        self.provider = provider
        self.model = model
        self.attachment = attachment


def _get_model_descriptor(
    provider: str,
    model: str,
    *,
    catalog_service: ModelCatalogService | None = None,
    catalog_records: Sequence[dict[str, Any]] | None = None,
) -> ModelDescriptor | None:
    """Retrieve model descriptor dynamically from API metadata or catalog cache.

    Avoids hardcoded model lists by querying the provider's API / catalog service.
    """
    target = str(model or "").strip()
    if not target:
        return None
    p_lower = str(provider or "").strip().lower()

    # 1. Check if caller provided explicit API catalog records (e.g. active turn or tests)
    if catalog_records:
        descriptors = descriptors_from_catalog(p_lower, catalog_records, source="api_records")
        for desc in descriptors:
            if desc.id == target or desc.qualified_id == target or target.endswith(f"/{desc.id}") or desc.id.endswith(f"/{target}"):
                return desc

    # 2. Query catalog service (queries cached API response, then live provider API)
    service = catalog_service or ModelCatalogService()
    try:
        desc = service.get_descriptor(p_lower, target)
        if desc is not None:
            return desc
    except Exception:
        pass

    # 3. Check maintained capabilities descriptor
    try:
        caps_desc = resolve_model_capability(p_lower, target)
        if caps_desc.is_known:
            meta = getattr(caps_desc, "metadata", {}) or {}
            caps_set = set(extract_capabilities_from_record(meta)) if isinstance(meta, dict) else set()
            return ModelDescriptor(
                provider=p_lower,
                id=target,
                capabilities=frozenset(caps_set) if caps_set else normalize_capabilities(p_lower, target),
                source="capability_registry",
                metadata=meta,
            )
    except Exception:
        pass

    # 4. Fallback to normalize_capabilities baseline
    caps = normalize_capabilities(p_lower, target)
    if caps:
        return ModelDescriptor(
            provider=p_lower,
            id=target,
            capabilities=caps,
            source="inferred",
        )

    return None


def _model_supports_vision(
    provider: str,
    model: str,
    *,
    catalog_service: ModelCatalogService | None = None,
    catalog_records: Sequence[dict[str, Any]] | None = None,
) -> bool:
    """Check whether a model candidate supports image / vision input from API metadata."""
    desc = _get_model_descriptor(provider, model, catalog_service=catalog_service, catalog_records=catalog_records)
    if desc is None:
        return False
    if desc.supports(ModelPurpose.MULTIMODAL_INPUT):
        return True
    arch = desc.metadata.get("architecture") if isinstance(desc.metadata.get("architecture"), dict) else {}
    mods = arch.get("input_modalities") or desc.metadata.get("input_modalities") or []
    return any(str(m).lower() in {"image", "image_url", "vision"} for m in mods)


def _model_supports_audio(
    provider: str,
    model: str,
    *,
    catalog_service: ModelCatalogService | None = None,
    catalog_records: Sequence[dict[str, Any]] | None = None,
) -> bool:
    """Check whether a model candidate supports audio input from API metadata."""
    desc = _get_model_descriptor(provider, model, catalog_service=catalog_service, catalog_records=catalog_records)
    if desc is None:
        return False
    if desc.supports(ModelPurpose.AUDIO_INPUT):
        return True
    arch = desc.metadata.get("architecture") if isinstance(desc.metadata.get("architecture"), dict) else {}
    mods = arch.get("input_modalities") or desc.metadata.get("input_modalities") or []
    return any(str(m).lower() in {"audio", "voice"} for m in mods)


def _model_supports_video(
    provider: str,
    model: str,
    *,
    catalog_service: ModelCatalogService | None = None,
    catalog_records: Sequence[dict[str, Any]] | None = None,
) -> bool:
    """Check whether a model candidate supports video input from API metadata."""
    desc = _get_model_descriptor(provider, model, catalog_service=catalog_service, catalog_records=catalog_records)
    if desc is None:
        return False
    if desc.supports(ModelPurpose.VIDEO_INPUT):
        return True
    arch = desc.metadata.get("architecture") if isinstance(desc.metadata.get("architecture"), dict) else {}
    mods = arch.get("input_modalities") or desc.metadata.get("input_modalities") or []
    return any(str(m).lower() == "video" for m in mods)


def validate_model_attachment_support(
    provider: str,
    model: str,
    attachments: Sequence[ChatAttachment],
    *,
    catalog_service: ModelCatalogService | None = None,
    catalog_records: Sequence[dict[str, Any]] | None = None,
) -> None:
    """Explicitly check model capabilities for all attached inputs.

    Determines capabilities dynamically from provider API metadata / catalog responses,
    avoiding hardcoded model lists. Fails gracefully with UnsupportedAttachmentError
    if the model cannot handle any attachment.
    """
    if not attachments:
        return

    prov = str(provider or "").strip().lower() or "openai"
    mod = str(model or "").strip()
    if not mod or mod.lower() in {"default", "openai/default"}:
        try:
            from mana_agent.config.user_config import load_effective_settings
            from mana_agent.config.provider_registry import split_qualified_model_id

            eff = load_effective_settings(include_env=True)
            raw_cfg = str(
                eff.get("OPENAI_CHAT_MODEL")
                or eff.get("MANA_PRIMARY_MODEL")
                or eff.get("LLM_MODEL")
                or "gpt-4.1-mini"
            ).strip()
            if raw_cfg.lower() == "default":
                raw_cfg = "gpt-4.1-mini"
            prov, mod = split_qualified_model_id(raw_cfg, default_provider=prov)
        except Exception:
            mod = "gpt-4.1-mini"

    for att in attachments:
        if att.category == AttachmentCategory.IMAGE.value:
            if not _model_supports_vision(prov, mod, catalog_service=catalog_service, catalog_records=catalog_records):
                raise UnsupportedAttachmentError(prov, mod, att)
        elif att.category == AttachmentCategory.AUDIO.value:
            if not _model_supports_audio(prov, mod, catalog_service=catalog_service, catalog_records=catalog_records):
                raise UnsupportedAttachmentError(prov, mod, att)
        elif att.category == AttachmentCategory.VIDEO.value:
            if not _model_supports_video(prov, mod, catalog_service=catalog_service, catalog_records=catalog_records):
                raise UnsupportedAttachmentError(prov, mod, att)
        # Document and Code attachments are always supported (either as multimodal blocks
        # or extracted text chunks).


def normalize_multimodal_content(
    text: str, attachments: Sequence[ChatAttachment]
) -> str | list[dict[str, Any]]:
    """Convert user text and attachments into target LLM message content.

    If no attachments exist, returns the plain text string unmodified.
    If attachments exist, returns an ordered list of content parts
    preserving text and attachment sequence.
    """
    if not attachments:
        return text

    blocks: list[dict[str, Any]] = []

    # If there is accompanying text, add it first (or keep ordering)
    if text.strip():
        blocks.append({"type": "text", "text": text})

    for att in attachments:
        storage_path = Path(att.storage_path)
        if not storage_path.is_file():
            # If the stored file is missing, add placeholder text rather than crashing
            blocks.append({
                "type": "text",
                "text": f"\n[Attachment '{att.filename}' not found at storage location]\n",
            })
            continue

        if att.category == AttachmentCategory.IMAGE.value:
            # Base64 data URI for vision models
            data = storage_path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{att.mime_type};base64,{b64}"},
            })
        elif att.category in {AttachmentCategory.DOCUMENT.value, AttachmentCategory.CODE.value}:
            # For documents and code, read text or extract content
            file_text = _extract_attachment_text(storage_path, att)
            blocks.append({
                "type": "text",
                "text": f"\n\n[Attached {att.category.capitalize()}: {att.filename}]\n```\n{file_text}\n```\n",
            })
        else:
            blocks.append({
                "type": "text",
                "text": f"\n\n[Attached file: {att.filename} ({att.mime_type}, {att.format_size()})]\n",
            })

    # If blocks contain only text blocks, combine or return structured list
    return blocks


def _extract_attachment_text(path: Path, att: ChatAttachment) -> str:
    """Safely extract readable text from document or code file."""
    if att.mime_type == "application/pdf" or path.suffix.lower() == ".pdf":
        try:
            from mana_agent.documents.readers import read_pdf
            parsed = read_pdf(path, max_chunks=50, max_chars_per_chunk=3000)
            text_chunks = [c.content for c in parsed.chunks if c.content]
            if text_chunks:
                return "\n\n".join(text_chunks)
        except Exception:
            pass

    # Standard text/code read
    try:
        raw = path.read_bytes()
        return raw.decode("utf-8", errors="replace")
    except Exception as exc:
        return f"(Could not read text content: {exc})"
