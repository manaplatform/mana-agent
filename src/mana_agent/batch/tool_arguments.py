"""Validate registered JSON-schema inputs before a deferred local invocation."""

from __future__ import annotations

from typing import Any


def validate_arguments(value: Any, schema: dict[str, Any], *, root: dict[str, Any] | None = None) -> None:
    root = root or schema
    if "$ref" in schema:
        target: Any = root
        reference = schema["$ref"]
        if not reference.startswith("#/"):
            raise ValueError("External tool schema references are unsupported")
        for part in reference[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        return validate_arguments(value, target, root=root)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            matches = 0
            for option in schema[key]:
                try:
                    validate_arguments(value, option, root=root)
                    matches += 1
                except ValueError:
                    pass
            if matches < 1 or (key == "oneOf" and matches != 1):
                raise ValueError("Tool argument does not satisfy its registered union type")
    expected = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool,
             "number": (int, float), "integer": int, "null": type(None)}
    if isinstance(expected, str) and expected in types:
        if not isinstance(value, types[expected]) or (expected in {"integer", "number"} and isinstance(value, bool)):
            raise ValueError(f"Tool argument must be {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("Tool argument is outside the registered enum")
    if isinstance(value, dict):
        if not set(schema.get("required", [])).issubset(value):
            raise ValueError("Tool call is missing required arguments")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                validate_arguments(item, properties[key], root=root)
            elif schema.get("additionalProperties") is False:
                raise ValueError("Tool call contains unregistered arguments")
            elif isinstance(schema.get("additionalProperties"), dict):
                validate_arguments(item, schema["additionalProperties"], root=root)
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")):
            raise ValueError("Tool argument list exceeds its registered bounds")
        for item in value:
            validate_arguments(item, schema.get("items", {}), root=root)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema.get("minimum", -float("inf")) or value > schema.get("maximum", float("inf")):
            raise ValueError("Tool argument exceeds its registered numeric bounds")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", float("inf")):
            raise ValueError("Tool argument exceeds its registered text bounds")
