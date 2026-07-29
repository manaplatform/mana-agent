"""Deterministic event reducer for recoverable Live Canvas state."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from mana_agent.canvas.catalog import validate_components, validate_data_model
from mana_agent.canvas.config import CanvasConfig
from mana_agent.canvas.models import (
    CanvasEventEnvelope,
    CanvasEventType,
    OwnerRef,
    SurfaceSnapshot,
)


class CanvasStateError(ValueError):
    """Raised when an event would create an invalid surface transition."""


def reduce_canvas_event(
    snapshot: SurfaceSnapshot | None,
    event: CanvasEventEnvelope,
    *,
    config: CanvasConfig,
) -> SurfaceSnapshot:
    """Apply exactly one ordered event; invalid transitions always raise."""
    if snapshot is None:
        if event.event_type is not CanvasEventType.CREATE:
            raise CanvasStateError(
                "Unknown surface; createSurface must be the first event."
            )
        body = _body(event, "createSurface")
        catalog_id = str(body.get("catalogId") or "")
        if catalog_id not in config.allowed_catalogs:
            raise CanvasStateError("Surface catalog is not allowlisted.")
        owner = OwnerRef(
            agent_id=event.agent_id,
            task_id=event.task_id,
            workflow_id=event.workflow_id,
            node_id=event.node_id,
            automation_id=event.automation_id,
        )
        expires_at = event.timestamp + timedelta(seconds=config.surface_expiry_seconds)
        return SurfaceSnapshot(
            session_id=event.session_id,
            conversation_id=event.conversation_id,
            surface_id=event.surface_id,
            catalog_id=catalog_id,
            owner=owner,
            version=1,
            last_sequence=event.sequence,
            created_at=event.timestamp,
            updated_at=event.timestamp,
            expires_at=expires_at,
            retain_on_complete=(
                True if event.retain_on_complete is None else event.retain_on_complete
            ),
        )

    if (
        event.session_id != snapshot.session_id
        or event.conversation_id != snapshot.conversation_id
    ):
        raise CanvasStateError(
            "Canvas event ownership does not match the surface session."
        )
    if event.surface_id != snapshot.surface_id:
        raise CanvasStateError("Canvas event targets a different surface.")
    if event.sequence != snapshot.last_sequence + 1:
        raise CanvasStateError(
            f"Out-of-order canvas event: expected {snapshot.last_sequence + 1}, got {event.sequence}."
        )
    if snapshot.deleted:
        raise CanvasStateError("Deleted surfaces cannot be updated.")

    values = snapshot.model_dump()
    values["version"] = snapshot.version + 1
    values["last_sequence"] = event.sequence
    values["updated_at"] = event.timestamp

    if event.event_type is CanvasEventType.CREATE:
        raise CanvasStateError("createSurface cannot replace an existing surface.")
    if event.event_type is CanvasEventType.COMPONENTS:
        body = _body(event, "updateComponents")
        incoming = validate_components(
            body.get("components") or (),
            surface_id=event.surface_id,
            config=config,
            require_root=not bool(snapshot.components),
        )
        merged = {item.id: item for item in snapshot.components}
        merged.update({item.id: item for item in incoming})
        values["components"] = validate_components(
            merged.values(), surface_id=event.surface_id, config=config
        )
    elif event.event_type is CanvasEventType.DATA:
        body = _body(event, "updateDataModel")
        values["data_model"] = _update_data_model(
            snapshot.data_model,
            path=str(body.get("path") or "/"),
            value=body.get("value", _MISSING),
            surface_id=event.surface_id,
        )
    elif event.event_type is CanvasEventType.DELETE:
        _body(event, "deleteSurface")
        values["deleted"] = True
    elif event.event_type is CanvasEventType.COMPLETE:
        values["completed"] = True
        if not snapshot.retain_on_complete:
            values["deleted"] = True
    elif event.event_type in {CanvasEventType.ACTION, CanvasEventType.ERROR}:
        pass
    else:  # pragma: no cover - enum protects this boundary
        raise CanvasStateError(f"Unsupported canvas event type: {event.event_type}.")
    return SurfaceSnapshot.model_validate(values)


_MISSING = object()


def _update_data_model(
    current: dict[str, Any], *, path: str, value: Any, surface_id: str
) -> dict[str, Any]:
    if path in {"", "/"}:
        if value is _MISSING:
            return {}
        return validate_data_model(value, surface_id=surface_id)
    if not path.startswith("/"):
        raise CanvasStateError("Data model path must be a JSON Pointer.")
    result = _deep_copy(current)
    tokens = [_unescape(token) for token in path[1:].split("/")]
    cursor: dict[str, Any] = result
    for token in tokens[:-1]:
        child = cursor.get(token)
        if child is None:
            child = {}
            cursor[token] = child
        if not isinstance(child, dict):
            raise CanvasStateError("Data model update traverses a non-object value.")
        cursor = child
    leaf = tokens[-1]
    if value is _MISSING:
        if leaf not in cursor:
            raise CanvasStateError("Data model delete targets an unknown path.")
        del cursor[leaf]
    else:
        cursor[leaf] = value
    return validate_data_model(result, surface_id=surface_id)


def _body(event: CanvasEventEnvelope, key: str) -> dict[str, Any]:
    if event.payload.get("version") != event.protocol_version:
        raise CanvasStateError(
            "A2UI payload version does not match its event envelope."
        )
    body = event.payload.get(key)
    if not isinstance(body, dict) or body.get("surfaceId") != event.surface_id:
        raise CanvasStateError(f"Invalid {key} payload or surface identifier.")
    return body


def _unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _deep_copy(value: dict[str, Any]) -> dict[str, Any]:
    import json

    return json.loads(json.dumps(value, ensure_ascii=False))
