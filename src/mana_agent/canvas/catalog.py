"""Allowlisted native component catalog and semantic validation."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

from mana_agent.canvas.config import CanvasConfig, MANA_CATALOG_ID
from mana_agent.canvas.models import Component, ValidationError


COMPONENT_CATALOG: dict[str, dict[str, Any]] = {
    "Text": {"actions": set(), "required": {"text"}, "optional": {"variant"}},
    "Heading": {"actions": set(), "required": {"text"}, "optional": {"level"}},
    "Markdown": {"actions": set(), "required": {"text"}, "optional": set()},
    "Button": {
        "actions": {"press"},
        "required": {"label"},
        "optional": {"disabled", "variant"},
    },
    "TextField": {
        "actions": {"change", "submit"},
        "required": {"label", "value"},
        "optional": {"placeholder", "inputType"},
    },
    "TextArea": {
        "actions": {"change", "submit"},
        "required": {"label", "value"},
        "optional": {"placeholder", "rows"},
    },
    "Select": {
        "actions": {"change"},
        "required": {"label", "options", "value"},
        "optional": {"placeholder"},
    },
    "Checkbox": {
        "actions": {"change"},
        "required": {"label", "value"},
        "optional": set(),
    },
    "RadioGroup": {
        "actions": {"change"},
        "required": {"label", "options", "value"},
        "optional": set(),
    },
    "Form": {"actions": {"submit"}, "required": {"children"}, "optional": set()},
    "Row": {
        "actions": set(),
        "required": {"children"},
        "optional": {"justify", "align"},
    },
    "Column": {
        "actions": set(),
        "required": {"children"},
        "optional": {"justify", "align"},
    },
    "Card": {
        "actions": set(),
        "required": set(),
        "optional": {"child", "children", "title"},
    },
    "Divider": {"actions": set(), "required": set(), "optional": {"axis"}},
    "Tabs": {"actions": {"change"}, "required": {"tabs"}, "optional": {"value"}},
    "List": {"actions": {"select"}, "required": {"items"}, "optional": {"ordered"}},
    "Table": {
        "actions": {"select", "sort"},
        "required": {"columns", "rows"},
        "optional": {"caption"},
    },
    "Badge": {"actions": set(), "required": {"text"}, "optional": {"status"}},
    "Progress": {"actions": set(), "required": {"value"}, "optional": {"max", "label"}},
    "Image": {
        "actions": set(),
        "required": {"url", "description"},
        "optional": {"fit"},
    },
    "Artifact": {
        "actions": {"open", "download"},
        "required": {"label", "url"},
        "optional": {"mediaType", "description"},
    },
    "ErrorState": {
        "actions": {"retry"},
        "required": {"message"},
        "optional": {"title"},
    },
    "EmptyState": {"actions": set(), "required": {"message"}, "optional": {"title"}},
}

_EXECUTABLE = re.compile(
    r"<\s*(script|iframe|object|embed)|javascript:|data:text/html|on\w+\s*=", re.I
)
_CHILD_FIELDS = ("child", "children")


class CatalogValidationError(ValueError):
    def __init__(self, errors: list[ValidationError]) -> None:
        self.errors = errors
        super().__init__("; ".join(error.message for error in errors))


def catalog_metadata() -> dict[str, Any]:
    return {
        "catalogId": MANA_CATALOG_ID,
        "components": {
            name: {
                "required": sorted(spec["required"]),
                "optional": sorted(spec["optional"]),
                "actions": sorted(spec["actions"]),
            }
            for name, spec in COMPONENT_CATALOG.items()
        },
    }


def validate_components(
    components: Iterable[Component | dict[str, Any]],
    *,
    surface_id: str,
    config: CanvasConfig,
    require_root: bool = True,
) -> tuple[Component, ...]:
    rows = tuple(
        item if isinstance(item, Component) else Component.model_validate(item)
        for item in components
    )
    errors: list[ValidationError] = []
    if not rows:
        errors.append(
            _error(surface_id, "/components", "Component update must not be empty.")
        )
    if len(rows) > config.max_components_per_surface:
        errors.append(
            _error(
                surface_id,
                "/components",
                "Component count exceeds the configured limit.",
            )
        )
    ids = [item.id for item in rows]
    if len(ids) != len(set(ids)):
        errors.append(
            _error(surface_id, "/components", "Component identifiers must be unique.")
        )
    if require_root and rows and "root" not in ids:
        errors.append(
            _error(
                surface_id, "/components", "The component tree must contain id 'root'."
            )
        )
    known = set(ids)
    for index, item in enumerate(rows):
        raw = item.model_dump(mode="json")
        spec = COMPONENT_CATALOG.get(item.component)
        if spec is None:
            errors.append(
                _error(
                    surface_id,
                    f"/components/{index}/component",
                    f"Unsupported component: {item.component}.",
                )
            )
            continue
        missing = spec["required"] - set(raw)
        if missing:
            errors.append(
                _error(
                    surface_id,
                    f"/components/{index}",
                    f"Missing required properties: {', '.join(sorted(missing))}.",
                )
            )
        common = {"id", "component", "actions", "ariaLabel", "weight"}
        unknown_properties = set(raw) - common - spec["required"] - spec["optional"]
        if unknown_properties:
            errors.append(
                _error(
                    surface_id,
                    f"/components/{index}",
                    f"Unsupported properties for {item.component}: {', '.join(sorted(unknown_properties))}.",
                )
            )
        supported_actions = spec["actions"]
        for action in item.actions:
            family = action.name.rsplit(".", 1)[-1]
            if family not in supported_actions:
                errors.append(
                    _error(
                        surface_id,
                        f"/components/{index}/actions",
                        f"Action '{action.name}' is not supported by {item.component}.",
                    )
                )
        for field in _CHILD_FIELDS:
            refs = raw.get(field)
            if isinstance(refs, str) and refs not in known:
                errors.append(
                    _error(
                        surface_id,
                        f"/components/{index}/{field}",
                        f"Unknown component reference: {refs}.",
                    )
                )
            if isinstance(refs, list):
                for ref in refs:
                    if not isinstance(ref, str) or ref not in known:
                        errors.append(
                            _error(
                                surface_id,
                                f"/components/{index}/{field}",
                                f"Unknown component reference: {ref}.",
                            )
                        )
        _validate_strings(raw, surface_id, f"/components/{index}", errors)
        _validate_bindings(raw, surface_id, f"/components/{index}", errors)
        if item.component == "Image":
            _validate_url(
                raw.get("url"),
                config.allowed_image_schemes,
                surface_id,
                f"/components/{index}/url",
                errors,
            )
        if item.component == "Artifact":
            _validate_url(
                raw.get("url"),
                config.allowed_artifact_schemes,
                surface_id,
                f"/components/{index}/url",
                errors,
            )
    if rows:
        _validate_depth(rows, surface_id, config.max_component_depth, errors)
    if errors:
        raise CatalogValidationError(errors)
    return rows


def validate_data_model(value: Any, *, surface_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogValidationError(
            [_error(surface_id, "/value", "The root data model must be an object.")]
        )
    errors: list[ValidationError] = []
    _validate_strings(value, surface_id, "/value", errors)
    if errors:
        raise CatalogValidationError(errors)
    return value


def _validate_strings(
    value: Any, surface_id: str, path: str, errors: list[ValidationError]
) -> None:
    if isinstance(value, str) and _EXECUTABLE.search(value):
        errors.append(
            _error(
                surface_id,
                path,
                "Executable HTML, script URLs, and inline event handlers are forbidden.",
            )
        )
    elif isinstance(value, dict):
        for key, child in value.items():
            _validate_strings(child, surface_id, f"{path}/{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_strings(child, surface_id, f"{path}/{index}", errors)


def _validate_bindings(
    value: Any, surface_id: str, path: str, errors: list[ValidationError]
) -> None:
    if isinstance(value, dict):
        if "call" in value:
            errors.append(
                _error(
                    surface_id,
                    path,
                    "Client-side function calls are not enabled in this catalog.",
                )
            )
        if "path" in value:
            pointer = value.get("path")
            if not isinstance(pointer, str) or not pointer.startswith("/"):
                errors.append(
                    _error(
                        surface_id,
                        f"{path}/path",
                        "Data binding must be an absolute JSON Pointer.",
                    )
                )
            elif re.search(r"~(?![01])", pointer):
                errors.append(
                    _error(
                        surface_id,
                        f"{path}/path",
                        "Data binding contains an invalid JSON Pointer escape.",
                    )
                )
        for key, child in value.items():
            _validate_bindings(child, surface_id, f"{path}/{key}", errors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_bindings(child, surface_id, f"{path}/{index}", errors)


def _validate_url(
    value: Any,
    allowed: tuple[str, ...],
    surface_id: str,
    path: str,
    errors: list[ValidationError],
) -> None:
    if isinstance(value, dict) and set(value) == {"path"}:
        return
    parsed = urlsplit(str(value or ""))
    if parsed.scheme not in allowed or (parsed.scheme == "https" and not parsed.netloc):
        errors.append(
            _error(surface_id, path, "URL does not satisfy the configured allowlist.")
        )


def _validate_depth(
    rows: tuple[Component, ...],
    surface_id: str,
    limit: int,
    errors: list[ValidationError],
) -> None:
    by_id = {item.id: item.model_dump(mode="json") for item in rows}

    def visit(component_id: str, depth: int, trail: frozenset[str]) -> None:
        if component_id in trail:
            errors.append(
                _error(surface_id, "/components", "Component tree contains a cycle.")
            )
            return
        if depth > limit:
            errors.append(
                _error(
                    surface_id,
                    "/components",
                    "Component tree exceeds the configured depth.",
                )
            )
            return
        row = by_id.get(component_id, {})
        refs: list[str] = []
        if isinstance(row.get("child"), str):
            refs.append(row["child"])
        if isinstance(row.get("children"), list):
            refs.extend(ref for ref in row["children"] if isinstance(ref, str))
        for ref in refs:
            visit(ref, depth + 1, trail | {component_id})

    visit("root", 1, frozenset())


def _error(surface_id: str, path: str, message: str) -> ValidationError:
    return ValidationError(surface_id=surface_id, path=path, message=message)
