"""Registered execution effects, independent of model intent routing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class OperationPolicy:
    capability: str
    effect: Literal["read_only", "mutation", "command", "external"]
    repository: bool = True


READ_TOOLS = frozenset({
    "read_file", "read_skill", "list_files", "repo_search", "repo_batch_read",
    "repo_batch_search", "batch_read_files", "batch_search_files", "batch_list_files",
    "find_symbols", "call_graph", "git_status", "git_diff", "git_log",
    "git_show", "git_help", "conversation_context_read", "memory_read",
})
WRITE_TOOLS = frozenset({
    "write_file", "create_file", "delete_file", "edit_file", "multi_edit_file", "apply_patch",
    "apply_patch_batch", "batch_write_files", "batch_apply_patches", "batch_move_files", "batch_delete_files",
})
COMMAND_TOOLS = frozenset({"run_command", "run_script_once", "shell", "verify_project"})
CONTROL_TOOLS = frozenset({"approval_request", "request_user_approval", "wait_for_approval"})


def builtin_policy(name: str) -> OperationPolicy | None:
    """Only exact registered names confer effects; unknown names fail closed."""
    name = name.replace(".", "_")
    if name in READ_TOOLS:
        return OperationPolicy("repository_read", "read_only")
    if name in WRITE_TOOLS:
        return OperationPolicy("repository_write", "mutation")
    if name in COMMAND_TOOLS:
        return OperationPolicy("test_execution" if name == "verify_project" else "shell_execution", "command")
    if name in CONTROL_TOOLS:
        return OperationPolicy("approval", "external", repository=False)
    # Git's own exact command registry and transactional gate validate argv.
    if name in {
        "git_generic", "git_branch", "git_create_branch", "git_add", "git_commit", "git_push",
        "git_pull", "git_fetch", "git_remote", "git_merge", "git_rebase", "git_revert",
        "git_reset", "git_clean", "git_tag", "git_config", "git_checkout", "git_switch",
    }:
        return OperationPolicy("git_execution", "command")
    return None


def validate_arguments(value: Any, schema: dict[str, Any], path: str = "arguments", *,
                       _root: dict[str, Any] | None = None, _depth: int = 0) -> None:
    """Validate registry schemas without coercing types or fetching remote refs."""
    import math
    import re
    if _depth > 64:
        raise ValueError(f"{path}: registered schema exceeds validation depth")
    root = schema if _root is None else _root
    def validate(item, spec, location=path):
        validate_arguments(item, spec, location, _root=root, _depth=_depth + 1)
    if schema is False:
        raise ValueError(f"{path}: disallowed by the registered schema")
    if schema is True:
        return
    if "$ref" in schema:
        reference = schema["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/"):
            raise ValueError(f"{path}: only local registered schema references are allowed")
        target = root
        try:
            for part in reference[2:].split("/"):
                target = target[part.replace("~1", "/").replace("~0", "~")]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}: unresolved registered schema reference") from exc
        validate(value, target)
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: value differs from the registered constant")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value is not in the registered enum")
    for spec in schema.get("allOf", []):
        validate(value, spec)
    if "not" in schema:
        try:
            validate(value, schema["not"])
        except ValueError:
            pass
        else:
            raise ValueError(f"{path}: disallowed by the registered schema")
    for union in ("anyOf", "oneOf"):
        if union in schema:
            matches = 0
            for candidate in schema[union]:
                try:
                    validate(value, candidate)
                    matches += 1
                except ValueError:
                    pass
            if not matches or union == "oneOf" and matches != 1:
                raise ValueError(f"{path}: no unambiguous registered schema")
    kind = schema.get("type")
    if isinstance(kind, list):
        validate(value, {"anyOf": [{**schema, "type": k} for k in kind]})
        return
    checks = {"object": isinstance(value, dict), "array": isinstance(value, list),
              "string": isinstance(value, str), "integer": type(value) is int,
              "number": type(value) in (int, float), "boolean": type(value) is bool, "null": value is None}
    if kind and not checks.get(kind, False):
        raise ValueError(f"{path}: expected {kind}")
    if isinstance(value, dict):
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            raise ValueError(f"{path}: missing {', '.join(sorted(missing))}")
        props = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        if additional is False and set(value) - set(props):
            raise ValueError(f"{path}: unregistered arguments {', '.join(sorted(set(value) - set(props)))}")
        for key, item in value.items():
            if key in props:
                validate(item, props[key], f"{path}.{key}")
            elif isinstance(additional, dict):
                validate(item, additional, f"{path}.{key}")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", float("inf")):
            raise ValueError(f"{path}: outside registered array bounds")
        for i, item in enumerate(value):
            validate(item, schema.get("items", {}), f"{path}[{i}]")
        if schema.get("uniqueItems") and any(item in value[:i] for i, item in enumerate(value)):
            raise ValueError(f"{path}: registered array requires unique items")
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", float("inf")):
            raise ValueError(f"{path}: outside registered string bounds")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ValueError(f"{path}: value does not match the registered pattern")
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite numbers are not JSON arguments")
        if ("minimum" in schema and value < schema["minimum"] or "maximum" in schema and value > schema["maximum"]
                or "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]
                or "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]):
            raise ValueError(f"{path}: outside registered bounds")
