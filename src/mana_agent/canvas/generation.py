"""Provider-independent parse, prompt, validation, and bounded correction boundary."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from mana_agent.canvas.catalog import catalog_metadata
from mana_agent.canvas.config import CanvasConfig, IMPLEMENTATION_VERSION, WIRE_VERSION
from mana_agent.canvas.models import RendererCapabilities, SurfaceSnapshot


class CanvasGenerationError(ValueError):
    pass


def generation_context(
    capabilities: RendererCapabilities, *, current: SurfaceSnapshot | None = None
) -> dict[str, Any]:
    return {
        "implementationVersion": IMPLEMENTATION_VERSION,
        "wireVersion": WIRE_VERSION,
        "rendererCapabilities": capabilities.model_dump(mode="json"),
        "catalog": catalog_metadata(),
        "rules": [
            "Return JSON only.",
            "Use only catalog components and declared actions.",
            "Never emit HTML, JavaScript, CSS, executable code, commands, prompts, or secrets.",
            "Use A2UI v0.9 createSurface/updateComponents/updateDataModel/deleteSurface messages.",
        ],
        "currentSurface": current.model_dump(mode="json") if current else None,
    }


def parse_generated_messages(
    raw: str,
    *,
    config: CanvasConfig,
    correct: Callable[[str, list[str]], str] | None = None,
) -> list[dict[str, Any]]:
    """Strictly parse model output with at most the configured correction attempts."""
    candidate = raw
    attempts = 0
    while True:
        errors: list[str] = []
        try:
            value = json.loads(candidate)
            rows = value if isinstance(value, list) else [value]
            if not rows or not all(isinstance(row, dict) for row in rows):
                raise ValueError(
                    "A2UI output must be a JSON object or non-empty object list."
                )
            for index, row in enumerate(rows):
                if row.get("version") != WIRE_VERSION:
                    errors.append(f"/{index}/version must equal {WIRE_VERSION}")
                keys = set(row) - {"version"}
                if len(keys) != 1 or next(iter(keys), "") not in {
                    "createSurface",
                    "updateComponents",
                    "updateDataModel",
                    "deleteSurface",
                }:
                    errors.append(
                        f"/{index} must contain exactly one supported A2UI message"
                    )
            encoded = json.dumps(rows, ensure_ascii=False).encode("utf-8")
            if len(encoded) > config.max_event_payload_bytes:
                errors.append("A2UI output exceeds the configured payload limit")
            if not errors:
                return rows
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(str(exc))
        if correct is None or attempts >= config.validation_retry_limit:
            raise CanvasGenerationError(
                "Model-produced A2UI failed validation: " + "; ".join(errors)
            )
        attempts += 1
        candidate = correct(candidate, errors)
