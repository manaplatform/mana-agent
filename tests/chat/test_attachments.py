"""Tests for chat attachments: detection, validation, normalization, and persistence."""

from __future__ import annotations

import base64
from pathlib import Path
import pytest

from mana_agent.chat.attachments import (
    AttachmentCategory,
    AttachmentStatus,
    AttachmentStore,
    AttachmentValidator,
    ChatAttachment,
    detect_mime_and_category,
    format_size,
    sanitize_filename,
)
from mana_agent.chat.events import UserMessageEvent
from mana_agent.chat.normalization import (
    UnsupportedAttachmentError,
    normalize_multimodal_content,
    validate_model_attachment_support,
)
from mana_agent.gateway.chat_turn_store import ChatTurnStore
from mana_agent.gateway.envelope import build_routing_execution_envelope
from mana_agent.services.chat_session_history import ChatSessionHistory


# Minimal valid file fixtures
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)
MINIMAL_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\xff\xd9"
MINIMAL_GIF = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
MINIMAL_PDF = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\nxref\n0 2\ntrailer<</Size 2/Root 1 0 R>>\nstartxref\n9\n%%EOF"
MINIMAL_WEBP = b"RIFF\x1a\x00\x00\x00WEBPVP8L\x0e\x00\x00\x00/\x00\x00\x00\x00\x07\x00\x08\x88\x01\x00"


def test_mime_and_category_detection() -> None:
    # PNG
    mime, cat = detect_mime_and_category("test.png", MINIMAL_PNG)
    assert mime == "image/png"
    assert cat == AttachmentCategory.IMAGE

    # JPEG
    mime, cat = detect_mime_and_category("photo.jpg", MINIMAL_JPEG)
    assert mime == "image/jpeg"
    assert cat == AttachmentCategory.IMAGE

    # GIF
    mime, cat = detect_mime_and_category("anim.gif", MINIMAL_GIF)
    assert mime == "image/gif"
    assert cat == AttachmentCategory.IMAGE

    # PDF
    mime, cat = detect_mime_and_category("document.pdf", MINIMAL_PDF)
    assert mime == "application/pdf"
    assert cat == AttachmentCategory.DOCUMENT

    # WEBP
    mime, cat = detect_mime_and_category("image.webp", MINIMAL_WEBP)
    assert mime == "image/webp"
    assert cat == AttachmentCategory.IMAGE

    # Code: Python
    py_content = b"def hello():\n    print('world')\n"
    mime, cat = detect_mime_and_category("script.py", py_content)
    assert cat == AttachmentCategory.CODE

    # Code: JSON
    json_content = b'{"name": "mana", "active": true}\n'
    mime, cat = detect_mime_and_category("data.json", json_content)
    assert cat == AttachmentCategory.CODE

    # Document: Markdown
    md_content = b"# Architecture\n\nThis is a plan.\n"
    mime, cat = detect_mime_and_category("README.md", md_content)
    assert cat == AttachmentCategory.DOCUMENT


def test_tampered_mime_detection_rejected() -> None:
    # Executable disguised as PNG
    fake_png = b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 50
    with pytest.raises(ValueError, match="does not match image extension"):
        detect_mime_and_category("payload.png", fake_png)

    # Invalid PDF disguised with .pdf extension
    fake_pdf = b"NOT A REAL PDF FILE"
    with pytest.raises(ValueError, match="Corrupted or invalid PDF"):
        detect_mime_and_category("fake.pdf", fake_pdf)

    # Text file containing binary null bytes
    binary_text = b"hello\x00world"
    with pytest.raises(ValueError, match="Binary content detected"):
        detect_mime_and_category("notes.txt", binary_text)


def test_attachment_validator_limits() -> None:
    validator = AttachmentValidator(
        max_file_size_bytes=100,
        max_count=2,
        max_total_size_bytes=150,
    )

    # File exceeding max_file_size_bytes
    big_content = b"a" * 101
    with pytest.raises(ValueError, match="exceeds maximum allowed"):
        validator.validate_file("big.txt", big_content)

    # Validate valid file
    name, mime, cat = validator.validate_file("ok.txt", b"hello")
    assert name == "ok.txt"

    # Validate collection limit
    att1 = ChatAttachment(
        attachment_id="1", filename="1.txt", mime_type="text/plain",
        size_bytes=80, category="document", storage_path="/tmp/1.txt",
    )
    att2 = ChatAttachment(
        attachment_id="2", filename="2.txt", mime_type="text/plain",
        size_bytes=80, category="document", storage_path="/tmp/2.txt",
    )
    # Total size exceeds 150
    with pytest.raises(ValueError, match="Total attachment size"):
        validator.validate_collection([att1, att2])

    # Count exceeds 2
    att3 = ChatAttachment(
        attachment_id="3", filename="3.txt", mime_type="text/plain",
        size_bytes=10, category="document", storage_path="/tmp/3.txt",
    )
    with pytest.raises(ValueError, match="Too many attachments"):
        validator.validate_collection([att1, att2, att3])


def test_attachment_store_persistence_and_traversal_protection(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    session_id = "test_session_123"

    # Normal save
    attachment = AttachmentStore.save(
        session_id=session_id,
        filename="diagram.png",
        content=MINIMAL_PNG,
    )
    assert attachment.attachment_id.startswith("att_")
    assert attachment.filename == "diagram.png"
    assert attachment.category == "image"
    assert attachment.status == AttachmentStatus.READY.value
    assert Path(attachment.storage_path).is_file()
    assert attachment.preview_metadata.get("width") == 1
    assert attachment.preview_metadata.get("height") == 1

    # Retrieval
    resolved = AttachmentStore.get_path(session_id, attachment.attachment_id, "diagram.png")
    assert resolved == Path(attachment.storage_path)

    # Path traversal protection in sanitize_filename
    assert sanitize_filename("../../../etc/passwd") == "etc_passwd"
    assert sanitize_filename("foo/bar/baz.txt") == "baz.txt"

    # Path traversal in get_path
    with pytest.raises(ValueError, match="Path traversal"):
        AttachmentStore.get_path(session_id, attachment.attachment_id, "../../../outside.txt")


def test_model_attachment_support_validation() -> None:
    img_att = ChatAttachment(
        attachment_id="1", filename="cat.png", mime_type="image/png",
        size_bytes=100, category=AttachmentCategory.IMAGE.value, storage_path="/tmp/cat.png",
    )
    doc_att = ChatAttachment(
        attachment_id="2", filename="report.pdf", mime_type="application/pdf",
        size_bytes=500, category=AttachmentCategory.DOCUMENT.value, storage_path="/tmp/report.pdf",
    )

    # Vision capable model from catalog (openai gpt-4o) allows images and docs
    validate_model_attachment_support("openai", "gpt-4o", [img_att, doc_att])

    # Dynamic model capabilities retrieved from API catalog metadata
    api_records = [
        {
            "id": "anthropic/claude-3-5-sonnet",
            "architecture": {"input_modalities": ["text", "image"]},
        },
        {
            "id": "google/gemini-2.0-flash",
            "architecture": {"input_modalities": ["text", "image", "audio", "video"]},
        },
        {
            "id": "meta-llama/llama-3-8b-instruct",
            "architecture": {"input_modalities": ["text"]},
        },
    ]

    # Vision capable model from API metadata allows image
    validate_model_attachment_support(
        "openrouter", "anthropic/claude-3-5-sonnet", [img_att], catalog_records=api_records
    )

    # Fully multimodal model from API metadata allows images, audio, and video
    audio_att = ChatAttachment(
        attachment_id="3", filename="voice.mp3", mime_type="audio/mp3",
        size_bytes=1000, category=AttachmentCategory.AUDIO.value, storage_path="/tmp/voice.mp3",
    )
    video_att = ChatAttachment(
        attachment_id="4", filename="clip.mp4", mime_type="video/mp4",
        size_bytes=2000, category=AttachmentCategory.VIDEO.value, storage_path="/tmp/clip.mp4",
    )
    validate_model_attachment_support(
        "openrouter", "google/gemini-2.0-flash", [img_att, audio_att, video_att], catalog_records=api_records
    )

    # Non-vision model from API metadata rejects images
    with pytest.raises(UnsupportedAttachmentError, match="does not support image attachments"):
        validate_model_attachment_support(
            "openrouter", "meta-llama/llama-3-8b-instruct", [img_att], catalog_records=api_records
        )

    # Model without audio input in API metadata rejects audio
    with pytest.raises(UnsupportedAttachmentError, match="does not support audio attachments"):
        validate_model_attachment_support(
            "openrouter", "anthropic/claude-3-5-sonnet", [audio_att], catalog_records=api_records
        )

    # Model without video input in API metadata rejects video
    with pytest.raises(UnsupportedAttachmentError, match="does not support video attachments"):
        validate_model_attachment_support(
            "openrouter", "anthropic/claude-3-5-sonnet", [video_att], catalog_records=api_records
        )

    # Non-vision model (openai gpt-3.5-turbo) rejects images
    with pytest.raises(UnsupportedAttachmentError, match="does not support image attachments"):
        validate_model_attachment_support("openai", "gpt-3.5-turbo", [img_att])


def test_normalize_multimodal_content(tmp_path: Path) -> None:
    # Save a temporary image and a document
    img_path = tmp_path / "sample.png"
    img_path.write_bytes(MINIMAL_PNG)

    doc_path = tmp_path / "notes.txt"
    doc_path.write_text("Detailed meeting notes.\nAction items included.", encoding="utf-8")

    img_att = ChatAttachment(
        attachment_id="1", filename="sample.png", mime_type="image/png",
        size_bytes=len(MINIMAL_PNG), category=AttachmentCategory.IMAGE.value,
        storage_path=str(img_path),
    )
    doc_att = ChatAttachment(
        attachment_id="2", filename="notes.txt", mime_type="text/plain",
        size_bytes=len(doc_path.read_bytes()), category=AttachmentCategory.DOCUMENT.value,
        storage_path=str(doc_path),
    )

    # Multi-attachment normalization
    blocks = normalize_multimodal_content("Explain this diagram and review notes", [img_att, doc_att])
    assert len(blocks) == 3
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "Explain this diagram and review notes"

    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")

    assert blocks[2]["type"] == "text"
    assert "Attached Document: notes.txt" in blocks[2]["text"]
    assert "Detailed meeting notes." in blocks[2]["text"]

    # Empty text prompt with attachment
    empty_prompt_blocks = normalize_multimodal_content("", [img_att])
    assert len(empty_prompt_blocks) == 1
    assert empty_prompt_blocks[0]["type"] == "image_url"


def test_chat_session_history_persistence_and_reload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    session_id = "session_history_test"
    history = ChatSessionHistory(session_id)

    att_dict = {
        "attachment_id": "att_test_1",
        "filename": "chart.png",
        "mime_type": "image/png",
        "size_bytes": 1024,
        "category": "image",
        "storage_path": "/path/to/chart.png",
        "status": "ready",
    }

    # Append user message with attachments
    history.append(
        role="user",
        content="Look at this chart",
        turn_id="turn_1",
        attachments=[att_dict],
    )

    # Append assistant response (text only)
    history.append(
        role="assistant",
        content="I see the upward trend in the chart.",
        turn_id="turn_1",
    )

    # Reload from disk
    reloaded = ChatSessionHistory(session_id)
    messages = reloaded.list()
    assert len(messages) == 2

    user_msg = messages[0]
    assert user_msg.role == "user"
    assert user_msg.content == "Look at this chart"
    assert len(user_msg.attachments) == 1
    assert user_msg.attachments[0]["filename"] == "chart.png"
    assert user_msg.attachments[0]["mime_type"] == "image/png"

    assistant_msg = messages[1]
    assert assistant_msg.role == "assistant"
    assert assistant_msg.attachments == ()


def test_chat_turn_store_fingerprint_with_attachments(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MANA_HOME", str(tmp_path / "mana_home"))
    store = ChatTurnStore("session_fingerprints")

    att_a = ChatAttachment(
        attachment_id="att_a", filename="a.png", mime_type="image/png",
        size_bytes=100, category="image", storage_path="/tmp/a.png",
    )
    att_b = ChatAttachment(
        attachment_id="att_b", filename="b.png", mime_type="image/png",
        size_bytes=100, category="image", storage_path="/tmp/b.png",
    )

    # Turn with att_a
    rec1, dup1 = store.create_or_get(
        conversation_id="conv_1",
        user_message_id="msg_1",
        turn_id="turn_1",
        text="analyze image",
        attachments=[att_a],
    )
    assert dup1 is False

    # Identical retry of turn 1 with att_a -> duplicate reused
    rec1_dup, dup1_retry = store.create_or_get(
        conversation_id="conv_1",
        user_message_id="msg_1",
        turn_id="turn_1_retry",
        text="analyze image",
        attachments=[att_a],
    )
    assert dup1_retry is True
    assert rec1_dup.turn_id == rec1.turn_id

    # Same text but with different attachment att_b -> separate turn
    rec2, dup2 = store.create_or_get(
        conversation_id="conv_1",
        user_message_id="msg_2",
        turn_id="turn_2",
        text="analyze image",
        attachments=[att_b],
    )
    assert dup2 is False
    assert rec2.turn_id == "turn_2"


def test_build_routing_execution_envelope_retains_attachments() -> None:
    from mana_agent.gateway.routing_decision import RoutingDecision

    decision = RoutingDecision(
        route="conversation",
        confidence=1.0,
        reasoning="Conversational message with file",
    )
    att_dict = {"attachment_id": "att_1", "filename": "file.txt"}
    envelope = build_routing_execution_envelope(
        turn_id="turn_env_1",
        session_id="session_env_1",
        user_text="review attached file",
        decision=decision,
        attachments=[att_dict],
    )
    assert envelope.attachments == (att_dict,)
    payload = envelope.to_dict()
    assert payload["attachments"] == (att_dict,)


def test_format_size() -> None:
    assert format_size(500) == "500 B"
    assert format_size(2048) == "2.0 KB"
    assert format_size(1048576 * 3) == "3.0 MB"


def test_extract_capabilities_from_dictionary_modalities_and_text_card() -> None:
    from mana_agent.config.model_catalog import (
        ModelCapability,
        extract_capabilities_from_record,
        parse_model_metadata_text,
    )

    # 1. Dictionary modalities
    record = {
        "id": "test-model",
        "modalities": {
            "Text": "Input and output",
            "Image": "Input only",
            "Audio": "Not supported",
            "Video": "Not supported",
        },
        "features": {
            "Streaming": "Supported",
            "Function calling": "Supported",
            "Structured outputs": "Supported",
            "Fine-tuning": "Not supported",
        },
        "tools": {
            "Web search": "Supported",
            "Code interpreter": "Supported",
        },
    }
    caps = extract_capabilities_from_record(record)
    assert ModelCapability.IMAGE_INPUT in caps
    assert ModelCapability.TEXT_GENERATION in caps
    assert ModelCapability.TOOL_CALLING in caps
    assert ModelCapability.STRUCTURED_OUTPUT in caps
    assert ModelCapability.CODE in caps
    assert ModelCapability.AUDIO_INPUT not in caps
    assert ModelCapability.VIDEO_INPUT not in caps

    # 2. Raw platform model card text parsing
    raw_text = """
Modalities
Text
Input and output
Image
Input only
Audio
Not supported
Video
Not supported
Endpoints
Live
v1/live/sessions
Chat Completions
v1/chat/completions
Responses
v1/responses
Realtime
v1/realtime
Features
Streaming
Supported
Function calling
Supported
Structured outputs
Supported
Fine-tuning
Not supported
Tools
Tools supported by this model when using the Responses API.
Web search
Supported
File search
Supported
Image generation
Supported
Code interpreter
Supported
"""
    parsed = parse_model_metadata_text(raw_text)
    assert parsed["modalities"]["Text"] == "Input and output"
    assert parsed["modalities"]["Image"] == "Input only"
    assert parsed["modalities"]["Audio"] == "Not supported"
    assert "v1/chat/completions" in parsed["endpoints"]
    assert "v1/realtime" in parsed["endpoints"]
    assert parsed["features"]["Function calling"] == "Supported"
    assert parsed["tools"]["Image generation"] == "Supported"

    text_caps = extract_capabilities_from_record({"id": "raw-card-model", "raw_metadata": raw_text})
    assert ModelCapability.IMAGE_INPUT in text_caps
    assert ModelCapability.TEXT_GENERATION in text_caps
    assert ModelCapability.TOOL_CALLING in text_caps
    assert ModelCapability.STRUCTURED_OUTPUT in text_caps
    assert ModelCapability.REALTIME in text_caps
    assert ModelCapability.IMAGE_GENERATION in text_caps
    assert ModelCapability.AUDIO_INPUT not in text_caps
    assert ModelCapability.VIDEO_INPUT not in text_caps


def test_validate_model_attachment_support_with_dict_modalities_and_card() -> None:
    img_att = ChatAttachment(
        attachment_id="1", filename="photo.jpg", mime_type="image/jpeg",
        size_bytes=200, category=AttachmentCategory.IMAGE.value, storage_path="/tmp/photo.jpg",
    )
    audio_att = ChatAttachment(
        attachment_id="2", filename="clip.mp3", mime_type="audio/mp3",
        size_bytes=500, category=AttachmentCategory.AUDIO.value, storage_path="/tmp/clip.mp3",
    )

    api_records = [
        {
            "id": "openai/custom-gpt-4o",
            "modalities": {
                "Text": "Input and output",
                "Image": "Input only",
                "Audio": "Not supported",
                "Video": "Not supported",
            },
        },
        {
            "id": "openai/card-model",
            "description": "Modalities\nText\nInput and output\nImage\nInput only\nAudio\nNot supported\n",
        },
    ]

    # Model with dict modalities allows image
    validate_model_attachment_support(
        "openai", "openai/custom-gpt-4o", [img_att], catalog_records=api_records
    )

    # Model with text card metadata allows image
    validate_model_attachment_support(
        "openai", "openai/card-model", [img_att], catalog_records=api_records
    )

    # Rejects audio since Audio is "Not supported"
    with pytest.raises(UnsupportedAttachmentError, match="does not support audio attachments"):
        validate_model_attachment_support(
            "openai", "openai/custom-gpt-4o", [audio_att], catalog_records=api_records
        )


def test_configured_agent_models_preserves_vision_capabilities() -> None:
    from mana_agent.config.model_catalog import ModelCapability
    from mana_agent.tui.model_management import configured_agent_models

    models = configured_agent_models(
        {
            "MANA_AI_PROVIDER": "openai",
            "MANA_CONFIGURED_PROVIDERS": ["openai"],
            "OPENAI_CHAT_MODEL": "gpt-4o",
        }
    )
    assert any(m.id == "gpt-4o" for m in models)
    target = next(m for m in models if m.id == "gpt-4o")
    assert ModelCapability.IMAGE_INPUT in target.capabilities


def test_validate_model_attachment_support_resolves_default_from_config(monkeypatch) -> None:
    # When model is 'default' or empty, resolve from configured OPENAI_CHAT_MODEL
    monkeypatch.setattr(
        "mana_agent.config.user_config.load_effective_settings",
        lambda **kwargs: {"OPENAI_CHAT_MODEL": "gpt-4o", "MANA_AI_PROVIDER": "openai"},
    )
    img_att = ChatAttachment(
        attachment_id="1", filename="logo.png", mime_type="image/png",
        size_bytes=100, category=AttachmentCategory.IMAGE.value, storage_path="/tmp/logo.png",
    )
    # Calling with 'default' must resolve gpt-4o and succeed without raising UnsupportedAttachmentError
    validate_model_attachment_support("openai", "default", [img_att])
    # Calling with 'openai/default' must resolve gpt-4o and succeed
    validate_model_attachment_support("openai", "openai/default", [img_att])


def test_image_attachment_chat_turn_does_not_trigger_unconfigured_artifact_handler(tmp_path: Path) -> None:
    from mana_agent.gateway.artifact_routing import artifact_routing_evidence

    img_path = tmp_path / "logo.png"
    img_path.write_bytes(MINIMAL_PNG)

    img_att = ChatAttachment(
        attachment_id="att_logo",
        filename="logo.png",
        mime_type="image/png",
        size_bytes=len(MINIMAL_PNG),
        category=AttachmentCategory.IMAGE.value,
        storage_path=str(img_path),
    )

    evidence = artifact_routing_evidence(
        root=tmp_path,
        user_prompt="check what is this?",
        attachments=[img_att.to_dict()],
    )

    # Must not be flagged as a document artifact family
    assert evidence["artifact_families"] == []
    assert evidence["has_user_artifact"] is False

    # Multimodal normalization must succeed for vision model execution
    blocks = normalize_multimodal_content("check what is this?", [img_att])
    assert len(blocks) == 2
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "check what is this?"
    assert blocks[1]["type"] == "image_url"
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")



