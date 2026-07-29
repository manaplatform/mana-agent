from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from mana_agent.media.errors import MediaError
from mana_agent.media.models import (
    ImageGenerationRequest,
    VideoGenerationRequest,
    VoiceGenerationRequest,
)
from mana_agent.media.service import MediaService


class _Context(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_decision_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class _Image(_Context):
    prompt: str = Field(min_length=1, max_length=32_000)
    model: str = ""
    size: str = Field(default="1024x1024", pattern=r"^(auto|\d{2,5}x\d{2,5})$")
    count: int = Field(default=1, ge=1, le=4)
    quality: str = Field(default="auto", max_length=32)
    output_format: str = Field(default="png", pattern=r"^(png|jpeg|webp)$")
    background: str | None = Field(default=None, max_length=32)
    reference_artifact_ids: tuple[str, ...] = ()
    idempotency_key: str = Field(default="", max_length=200)


class _Voice(_Context):
    text: str = Field(min_length=1, max_length=4096)
    model: str = ""
    voice: str = Field(default="alloy", min_length=1, max_length=160)
    output_format: str = Field(
        default="mp3", pattern=r"^(mp3|opus|aac|flac|wav|pcm)$"
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    instructions: str = Field(default="", max_length=4096)
    idempotency_key: str = Field(default="", max_length=200)


class _Video(_Context):
    prompt: str = Field(min_length=1, max_length=32_000)
    model: str = ""
    duration_seconds: int = Field(default=4, ge=1, le=120)
    aspect_ratio: str = Field(default="", max_length=32)
    resolution: str = Field(
        default="720x1280", pattern=r"^(auto|\d{2,5}x\d{2,5})$"
    )
    reference_artifact_ids: tuple[str, ...] = ()
    idempotency_key: str = Field(default="", max_length=200)


class _Generation(_Context):
    generation_id: str = Field(min_length=1)


def _json(operation: Any) -> str:
    try:
        result = operation()
        return json.dumps(
            {"ok": True, "result": result.model_dump(mode="json")},
            ensure_ascii=False,
        )
    except MediaError as exc:
        return json.dumps(
            {
                "ok": False,
                "error_code": exc.code,
                "message": exc.detail,
            },
            ensure_ascii=False,
        )
    except Exception:
        return json.dumps(
            {
                "ok": False,
                "error_code": "media_generation_failed",
                "message": "The media operation failed before a safe result was produced.",
            },
            ensure_ascii=False,
        )


def build_media_langchain_tools(
    root: str | Path, *, service: MediaService | None = None
) -> list[Any]:
    workspace_root = Path(root).resolve()
    media = service or MediaService(workspace_root=workspace_root)
    common = (
        "Use only after a validated model decision selects the media route. "
        "Returns compact metadata and managed artifact references, never binary data."
    )
    return [
        StructuredTool.from_function(
            name="generate_image",
            description=f"Generate an image with the configured image model. {common}",
            args_schema=_Image,
            func=lambda source_decision_id, session_id, **kwargs: _json(
                lambda: media.generate_image(
                    ImageGenerationRequest(**kwargs),
                    session_id=session_id,
                    turn_id=source_decision_id,
                )
            ),
        ),
        StructuredTool.from_function(
            name="generate_voice",
            description=f"Generate speech audio with the configured voice model. {common}",
            args_schema=_Voice,
            func=lambda source_decision_id, session_id, **kwargs: _json(
                lambda: media.generate_speech(
                    VoiceGenerationRequest(**kwargs),
                    session_id=session_id,
                    turn_id=source_decision_id,
                )
            ),
        ),
        StructuredTool.from_function(
            name="generate_video",
            description=f"Create or queue video generation with the configured video model. {common}",
            args_schema=_Video,
            func=lambda source_decision_id, session_id, **kwargs: _json(
                lambda: media.generate_video(
                    VideoGenerationRequest(**kwargs),
                    session_id=session_id,
                    turn_id=source_decision_id,
                )
            ),
        ),
        StructuredTool.from_function(
            name="get_media_generation_status",
            description="Get durable status and artifact metadata for one media generation.",
            args_schema=_Generation,
            func=lambda source_decision_id, session_id, generation_id: _json(
                lambda: media.get_generation_status(
                    generation_id,
                    session_id=session_id,
                    turn_id=source_decision_id,
                )
            ),
        ),
        StructuredTool.from_function(
            name="cancel_media_generation",
            description="Cancel the exact pending media generation when its provider supports cancellation.",
            args_schema=_Generation,
            func=lambda source_decision_id, session_id, generation_id: _json(
                lambda: media.cancel_generation(
                    generation_id,
                    session_id=session_id,
                    turn_id=source_decision_id,
                )
            ),
        ),
    ]


MEDIA_TOOL_NAMES = (
    "generate_image",
    "generate_voice",
    "generate_video",
    "get_media_generation_status",
    "cancel_media_generation",
)
