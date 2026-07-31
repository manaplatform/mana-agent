"""Small, dependency-free JSON Schema validation for API requests."""

from __future__ import annotations

import re
from typing import Any

from mana_agent.api_manager.errors import RequestValidationError


def validate_json(value: Any, schema: dict[str, Any], *, path: str = "$") -> None:
    """Validate the request-oriented JSON Schema subset emitted by OpenAPI.

    Unsupported assertion keywords are preserved in normalized definitions but
    never treated as permission to accept structurally invalid input.
    """

    if not schema:
        return
    if value is None:
        if schema.get("nullable") or "null" in schema.get("type", []):
            return
        if schema.get("type") == "null":
            return
    expected = schema.get("type")
    if isinstance(expected, list):
        if any(_matches_type(value, item) for item in expected):
            pass
        else:
            _fail(path, f"expected one of {expected}")
    elif expected and not _matches_type(value, expected):
        _fail(path, f"expected {expected}")

    if "enum" in schema and value not in schema["enum"]:
        _fail(path, f"value is not one of {schema['enum']!r}")
    if "const" in schema and value != schema["const"]:
        _fail(path, "value does not match const")

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        missing = [name for name in required if name not in value]
        if missing:
            _fail(path, f"missing required properties: {', '.join(missing)}")
        additional = schema.get(
            "additionalProperties",
            False if properties else True,
        )
        unknown = sorted(set(value).difference(properties))
        if unknown and additional is False:
            _fail(path, f"unknown properties: {', '.join(unknown)}")
        for name, item in value.items():
            child_schema = properties.get(name)
            if child_schema:
                validate_json(item, child_schema, path=f"{path}.{name}")
            elif isinstance(additional, dict):
                validate_json(item, additional, path=f"{path}.{name}")
        if "minProperties" in schema and len(value) < int(schema["minProperties"]):
            _fail(path, "object has too few properties")
        if "maxProperties" in schema and len(value) > int(schema["maxProperties"]):
            _fail(path, "object has too many properties")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            _fail(path, "array has too few items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            _fail(path, "array has too many items")
        item_schema = schema.get("items") or {}
        for index, item in enumerate(value):
            validate_json(item, item_schema, path=f"{path}[{index}]")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            _fail(path, "string is too short")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            _fail(path, "string is too long")
        if schema.get("pattern"):
            try:
                matched = re.search(str(schema["pattern"]), value)
            except re.error as exc:
                _fail(path, f"schema contains an invalid pattern: {exc}")
            if not matched:
                _fail(path, "string does not match the required pattern")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            _fail(path, f"value is below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            _fail(path, f"value is above maximum {schema['maximum']}")

    for alternative_keyword in ("oneOf", "anyOf"):
        alternatives = schema.get(alternative_keyword)
        if alternatives:
            successes = 0
            for alternative in alternatives:
                try:
                    validate_json(value, alternative, path=path)
                    successes += 1
                except RequestValidationError:
                    continue
            if (alternative_keyword == "oneOf" and successes != 1) or (
                alternative_keyword == "anyOf" and successes < 1
            ):
                _fail(path, f"value does not satisfy {alternative_keyword}")
    for requirement in schema.get("allOf") or ():
        validate_json(value, requirement, path=path)


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(str(expected), True)


def _fail(path: str, reason: str) -> None:
    raise RequestValidationError(
        f"Request validation failed at {path}: {reason}.",
        details={"path": path, "reason": reason},
    )
