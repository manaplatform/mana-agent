"""Normal model-tool surface for agent and workflow Canvas operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from mana_agent.canvas.models import CanvasSource, OwnerRef
from mana_agent.canvas.service import canvas_service_for_root


class _Decision(BaseModel):
    source_decision_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)


class _Surface(_Decision):
    surface_id: str = Field(min_length=1, max_length=160)


class _Create(_Surface):
    owner: OwnerRef
    retain_on_complete: bool = True


class _Components(_Surface):
    components: list[dict[str, Any]] = Field(min_length=1)


class _Data(_Surface):
    value: dict[str, Any]
    path: str = "/"


class _List(_Decision):
    include_deleted: bool = False


class _Wait(_Surface):
    action_name: str = Field(min_length=1, max_length=128)
    timeout_seconds: float | None = Field(default=None, gt=0)


def _json(operation) -> str:
    try:
        value = operation()
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return json.dumps(
            {"ok": True, "result": value}, ensure_ascii=False, default=str
        )
    except (ValueError, RuntimeError, TimeoutError) as exc:
        return json.dumps(
            {"ok": False, "error_code": "canvas_operation_failed", "message": str(exc)}
        )


def build_canvas_langchain_tools(root: str | Path) -> list[Any]:
    service = canvas_service_for_root(root)
    common = (
        "Use only after a structured model decision selects Canvas. All UI is validated against "
        "the allowlisted catalog; never include HTML, JavaScript, CSS, commands, prompts, or secrets."
    )
    return [
        StructuredTool.from_function(
            name="canvas_create_surface",
            description=f"Create a durable A2UI surface. {common}",
            args_schema=_Create,
            func=lambda source_decision_id, session_id, conversation_id, surface_id, owner, retain_on_complete: (
                _json(
                    lambda: service.create_surface(
                        session_id=session_id,
                        conversation_id=conversation_id,
                        surface_id=surface_id,
                        owner=owner,
                        correlation_id=source_decision_id,
                        source=CanvasSource.AGENT,
                        retain_on_complete=retain_on_complete,
                    )
                )
            ),
        ),
        StructuredTool.from_function(
            name="canvas_update_components",
            description=f"Add or replace validated components. {common}",
            args_schema=_Components,
            func=lambda source_decision_id, session_id, conversation_id, surface_id, components: (
                _json(
                    lambda: service.update_components(
                        session_id=session_id,
                        conversation_id=conversation_id,
                        surface_id=surface_id,
                        components=components,
                        correlation_id=source_decision_id,
                    )
                )
            ),
        ),
        StructuredTool.from_function(
            name="canvas_update_data",
            description=f"Update a validated surface data model. {common}",
            args_schema=_Data,
            func=lambda source_decision_id, session_id, conversation_id, surface_id, value, path: (
                _json(
                    lambda: service.update_data(
                        session_id=session_id,
                        conversation_id=conversation_id,
                        surface_id=surface_id,
                        value=value,
                        path=path,
                        correlation_id=source_decision_id,
                    )
                )
            ),
        ),
        StructuredTool.from_function(
            name="canvas_delete_surface",
            description="Delete a surface from its owning session.",
            args_schema=_Surface,
            func=lambda source_decision_id, session_id, conversation_id, surface_id: (
                _json(
                    lambda: service.delete_surface(
                        session_id=session_id,
                        conversation_id=conversation_id,
                        surface_id=surface_id,
                        correlation_id=source_decision_id,
                    )
                )
            ),
        ),
        StructuredTool.from_function(
            name="canvas_get_surface",
            description="Load one durable Canvas snapshot.",
            args_schema=_Surface,
            func=lambda source_decision_id, session_id, conversation_id, surface_id: (
                _json(lambda: service.get_surface(session_id, surface_id))
            ),
        ),
        StructuredTool.from_function(
            name="canvas_list_surfaces",
            description="List surfaces owned by a session.",
            args_schema=_List,
            func=lambda source_decision_id, session_id, conversation_id, include_deleted: (
                _json(
                    lambda: [
                        item.model_dump(mode="json")
                        for item in service.list_surfaces(
                            session_id, include_deleted=include_deleted
                        )
                    ]
                )
            ),
        ),
        StructuredTool.from_function(
            name="canvas_wait_for_action",
            description="Pause the owning runtime until the exact declared renderer action arrives.",
            args_schema=_Wait,
            func=lambda source_decision_id, session_id, conversation_id, surface_id, action_name, timeout_seconds: (
                _json(
                    lambda: service.wait_for_action(
                        session_id=session_id,
                        surface_id=surface_id,
                        action_name=action_name,
                        timeout=timeout_seconds,
                    )
                )
            ),
        ),
    ]


CANVAS_TOOL_NAMES = (
    "canvas_create_surface",
    "canvas_update_components",
    "canvas_update_data",
    "canvas_delete_surface",
    "canvas_get_surface",
    "canvas_list_surfaces",
    "canvas_wait_for_action",
)
