"""Responses → Chat Completions tool conversion with fail-fast diagnostics.

Codex emits a mix of function tools, namespaced function collections, and
Responses *host* tools (local_shell, web_search, freeform/custom apply_patch,
built-in apply_patch). Chat Completions only understands function tools, so the
bridge must:

1. Convert every representable tool into a Chat Completions function tool.
2. Preserve origin metadata so streaming responses can emit the matching
   Responses item type (function_call vs custom_tool_call, etc.).
3. Fail explicitly only when a tool truly cannot be represented — never
   silently drop required coding tools such as apply_patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Responses host / built-in tool types that Codex may emit and that we can
# represent as Chat Completions function tools for NVIDIA and peers.
_HOST_TOOL_TYPES = frozenset(
    {
        "local_shell",
        "web_search",
        "web_search_preview",
        "web_search_preview_2025_03_11",
        "apply_patch",
        "custom",  # freeform tools (Codex apply_patch_tool_type=freeform)
    }
)

_NAMESPACE_TOOL_TYPE = "namespace"
_CHAT_FUNCTION_NAME_SEPARATOR = "__"

# Names treated as freeform string-input tools when origin is custom/apply_patch.
_FREEFORM_ARGUMENT_NAMES = frozenset(
    {
        "input",
        "command",
        "patch",
        "content",
        "text",
    }
)


@dataclass(frozen=True, slots=True)
class ToolConversionReport:
    original_count: int
    converted_count: int
    converted_tools: list[dict[str, Any]] = field(default_factory=list)
    unsupported: list[dict[str, Any]] = field(default_factory=list)
    skipped_empty: int = 0
    # name → original Responses tool type (function, custom, local_shell, …)
    tool_origins: dict[str, str] = field(default_factory=dict)
    # Chat function name → original Responses function name. These differ for
    # expanded namespace tools, whose upstream name includes the namespace.
    response_tool_names: dict[str, str] = field(default_factory=dict)
    # Chat function name → original Responses namespace for expanded tools.
    tool_namespaces: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        # A namespace expands one source entry into multiple Chat functions.
        return not self.unsupported


class BridgeToolCompatibilityError(RuntimeError):
    """Raised when Codex tools cannot be represented for the upstream provider."""

    def __init__(
        self,
        message: str,
        *,
        report: ToolConversionReport,
        provider: str = "",
        model: str = "",
        transport: str = "responses_bridge",
    ) -> None:
        super().__init__(message)
        self.report = report
        self.provider = str(provider or "")
        self.model = str(model or "")
        self.transport = str(transport or "responses_bridge")

    def diagnostics(self) -> dict[str, Any]:
        return {
            "code": "bridge_tool_compatibility_error",
            "message": str(self),
            "provider": self.provider,
            "model": self.model,
            "transport": self.transport,
            "original_tool_count": self.report.original_count,
            "converted_tool_count": self.report.converted_count,
            "unsupported_tool_count": len(self.report.unsupported),
            "unsupported_tools": list(self.report.unsupported),
            "skipped_empty": self.report.skipped_empty,
            "converted_tool_names": sorted(self.report.tool_origins.keys()),
        }


def convert_responses_tools(
    tools: Any,
    *,
    provider: str = "",
    model: str = "",
    transport: str = "responses_bridge",
    fail_on_unsupported: bool = True,
) -> ToolConversionReport:
    """Convert Responses-style tools to Chat Completions function tools.

    Supported shapes:
    * ``{"type":"function","function":{...}}`` (already Chat-shaped)
    * ``{"type":"function","name":...,"parameters":...}`` (Responses flat function)
    * ``{"type":"namespace","name":...,"tools":[...]}`` (expanded to functions)
    * Host tools: ``local_shell``, ``web_search*``, ``apply_patch``, ``custom`` freeform
    """
    if not isinstance(tools, list):
        return ToolConversionReport(original_count=0, converted_count=0, converted_tools=[])

    converted: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    skipped_empty = 0
    tool_origins: dict[str, str] = {}
    response_tool_names: dict[str, str] = {}
    tool_namespaces: dict[str, str] = {}

    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            unsupported.append(
                {
                    "index": index,
                    "type": type(tool).__name__,
                    "reason": "tool_entry_not_object",
                }
            )
            continue

        tool_type = str(tool.get("type") or "").strip() or "function"
        namespace = _convert_namespace_tool(tool, tool_type=tool_type, index=index)
        if namespace is not None:
            if namespace.get("error"):
                unsupported.append(namespace["error"])
                continue
            for entry in namespace["tools"]:
                fn_tool = entry["tool"]
                chat_name = str(fn_tool["function"]["name"])
                if chat_name in tool_origins:
                    unsupported.append(
                        {
                            "index": index,
                            "type": tool_type,
                            "name": chat_name,
                            "reason": "duplicate_chat_function_name",
                        }
                    )
                    continue
                converted.append(fn_tool)
                tool_origins[chat_name] = "function"
                response_tool_names[chat_name] = str(entry["response_name"])
                tool_namespaces[chat_name] = str(entry["namespace"])
            continue

        host = _convert_host_tool(tool, tool_type=tool_type, index=index)
        if host is not None:
            if host.get("error"):
                unsupported.append(host["error"])
                continue
            fn_tool = host["tool"]
            name = str(fn_tool["function"]["name"])
            converted.append(fn_tool)
            tool_origins[name] = str(host.get("origin_type") or tool_type)
            response_tool_names[name] = name
            continue

        if tool_type != "function":
            unsupported.append(
                {
                    "index": index,
                    "type": tool_type,
                    "name": str(
                        tool.get("name")
                        or (tool.get("function") or {}).get("name")
                        or ""
                    ),
                    "reason": "unsupported_tool_type_for_chat_completions",
                }
            )
            continue

        if isinstance(tool.get("function"), dict):
            function = dict(tool["function"])
            name = str(function.get("name") or tool.get("name") or "").strip()
            if not name:
                unsupported.append(
                    {
                        "index": index,
                        "type": tool_type,
                        "name": "",
                        "reason": "function_tool_missing_name",
                    }
                )
                continue
            function["name"] = name
            if "parameters" not in function:
                function["parameters"] = (
                    tool.get("parameters")
                    or tool.get("input_schema")
                    or {"type": "object", "properties": {}}
                )
            if "description" not in function and tool.get("description") is not None:
                function["description"] = str(tool.get("description") or "")
            converted.append({"type": "function", "function": function})
            tool_origins[name] = "function"
            response_tool_names[name] = name
            continue

        name = str(tool.get("name") or "").strip()
        if not name:
            if (
                not tool.get("description")
                and not tool.get("parameters")
                and not tool.get("input_schema")
            ):
                skipped_empty += 1
                continue
            unsupported.append(
                {
                    "index": index,
                    "type": tool_type,
                    "name": "",
                    "reason": "function_tool_missing_name",
                }
            )
            continue
        function = {
            "name": name,
            "description": str(tool.get("description") or ""),
            "parameters": tool.get("parameters")
            or tool.get("input_schema")
            or {"type": "object", "properties": {}},
        }
        if "strict" in tool:
            function["strict"] = tool["strict"]
        converted.append({"type": "function", "function": function})
        tool_origins[name] = "function"
        response_tool_names[name] = name

    report = ToolConversionReport(
        original_count=len(tools),
        converted_count=len(converted),
        converted_tools=converted,
        unsupported=unsupported,
        skipped_empty=skipped_empty,
        tool_origins=tool_origins,
        response_tool_names=response_tool_names,
        tool_namespaces=tool_namespaces,
    )
    if fail_on_unsupported and unsupported:
        unsupported_summary = ", ".join(
            f"{item.get('type') or '?'}({item.get('name') or item.get('reason') or 'unnamed'})"
            for item in unsupported[:8]
        )
        raise BridgeToolCompatibilityError(
            (
                "Codex Responses tools cannot be represented on the Chat Completions "
                f"upstream ({provider or 'unknown'}/{model or 'unknown'}). "
                f"original={report.original_count} converted={report.converted_count} "
                f"unsupported={len(unsupported)} [{unsupported_summary}]. "
                "No tools were silently dropped."
            ),
            report=report,
            provider=provider,
            model=model,
            transport=transport,
        )
    return report


def _convert_namespace_tool(
    tool: dict[str, Any],
    *,
    tool_type: str,
    index: int,
) -> dict[str, Any] | None:
    """Expand one Responses namespace into explicit Chat Completions functions."""
    if tool_type.lower() != _NAMESPACE_TOOL_TYPE:
        return None

    namespace = str(tool.get("name") or "").strip()
    children = tool.get("tools")
    if not namespace:
        return {
            "error": {
                "index": index,
                "type": tool_type,
                "name": "",
                "reason": "namespace_tool_missing_name",
            }
        }
    if not isinstance(children, list) or not children:
        return {
            "error": {
                "index": index,
                "type": tool_type,
                "name": namespace,
                "reason": "namespace_tool_missing_functions",
            }
        }

    expanded: list[dict[str, Any]] = []
    names: set[str] = set()
    for child_index, child in enumerate(children):
        if not isinstance(child, dict):
            return {
                "error": {
                    "index": index,
                    "type": tool_type,
                    "name": namespace,
                    "reason": f"namespace_function_{child_index}_not_object",
                }
            }
        child_type = str(child.get("type") or "function").strip().lower()
        if child_type != "function":
            return {
                "error": {
                    "index": index,
                    "type": tool_type,
                    "name": namespace,
                    "reason": f"namespace_function_{child_index}_unsupported_type_{child_type}",
                }
            }
        function = dict(child.get("function") or child)
        response_name = str(function.get("name") or "").strip()
        chat_name = f"{namespace}{_CHAT_FUNCTION_NAME_SEPARATOR}{response_name}"
        if not response_name:
            return {
                "error": {
                    "index": index,
                    "type": tool_type,
                    "name": namespace,
                    "reason": f"namespace_function_{child_index}_missing_name",
                }
            }
        if chat_name in names:
            return {
                "error": {
                    "index": index,
                    "type": tool_type,
                    "name": chat_name,
                    "reason": "duplicate_namespace_function_name",
                }
            }
        names.add(chat_name)
        schema = function.get("parameters") or function.get("input_schema") or {
            "type": "object",
            "properties": {},
        }
        description = function.get("description") or child.get("description") or ""
        chat_function: dict[str, Any] = {
            "name": chat_name,
            "description": str(description),
            "parameters": schema,
        }
        if "strict" in function:
            chat_function["strict"] = function["strict"]
        expanded.append(
            {
                "tool": {"type": "function", "function": chat_function},
                "response_name": response_name,
                "namespace": namespace,
            }
        )
    return {"tools": expanded}


def _convert_host_tool(
    tool: dict[str, Any],
    *,
    tool_type: str,
    index: int,
) -> dict[str, Any] | None:
    """Return conversion result for host tools, or None if not a host type."""
    lowered = tool_type.lower()
    if lowered not in _HOST_TOOL_TYPES and not lowered.startswith("web_search"):
        return None

    if lowered == "local_shell":
        return {
            "origin_type": "local_shell",
            "tool": {
                "type": "function",
                "function": {
                    "name": "local_shell",
                    "description": str(
                        tool.get("description")
                        or "Run a local shell command in the workspace."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Argv list for the shell command.",
                            },
                            "working_directory": {"type": "string"},
                            "timeout_ms": {"type": "integer"},
                            "env": {
                                "type": "object",
                                "additionalProperties": {"type": "string"},
                            },
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                },
            },
        }

    if lowered.startswith("web_search"):
        name = "web_search"
        return {
            "origin_type": lowered,
            "tool": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(
                        tool.get("description") or "Search the web for information."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "additionalProperties": True,
                    },
                },
            },
        }

    if lowered == "apply_patch":
        # Built-in Responses apply_patch tool.
        return {
            "origin_type": "apply_patch",
            "tool": _freeform_function_tool(
                name="apply_patch",
                description=str(
                    tool.get("description")
                    or "Apply a structured patch to files under the workspace."
                ),
                input_description=(
                    "Patch body in apply_patch / freeform diff format "
                    "(*** Begin Patch … *** End Patch)."
                ),
            ),
        }

    if lowered == "custom":
        # Freeform custom tools (Codex freeform apply_patch, etc.).
        name = str(tool.get("name") or "").strip()
        if not name:
            return {
                "error": {
                    "index": index,
                    "type": tool_type,
                    "name": "",
                    "reason": "custom_tool_missing_name",
                }
            }
        description = str(tool.get("description") or f"Custom freeform tool {name}.")
        return {
            "origin_type": "custom",
            "tool": _freeform_function_tool(
                name=name,
                description=description,
                input_description="Freeform tool input text.",
            ),
        }

    return {
        "error": {
            "index": index,
            "type": tool_type,
            "name": str(tool.get("name") or ""),
            "reason": "unsupported_host_tool_variant",
        }
    }


def _freeform_function_tool(
    *,
    name: str,
    description: str,
    input_description: str,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": input_description,
                    }
                },
                "required": ["input"],
                "additionalProperties": False,
            },
        },
    }


def responses_item_type_for_tool_call(
    name: str,
    *,
    tool_origins: dict[str, str] | None = None,
) -> str:
    """Choose Responses output item type for a Chat Completions function call."""
    origins = tool_origins or {}
    origin = str(origins.get(name) or "function").lower()
    if origin == "custom" or origin == "apply_patch":
        # Freeform / built-in apply_patch round-trip as custom_tool_call so Codex
        # freeform handlers receive a string input channel.
        return "custom_tool_call"
    if origin == "local_shell":
        # Codex accepts function_call for shell-like tools when declared as
        # functions; keep function_call for maximum Chat Completions fidelity.
        return "function_call"
    if origin.startswith("web_search"):
        return "function_call"
    return "function_call"


def responses_tool_call_identity(
    name: str,
    *,
    response_tool_names: dict[str, str] | None = None,
    tool_namespaces: dict[str, str] | None = None,
) -> tuple[str, str | None]:
    """Recover the original Responses function identity for a Chat tool call."""
    response_name = str((response_tool_names or {}).get(name) or name)
    namespace = str((tool_namespaces or {}).get(name) or "").strip() or None
    return response_name, namespace


def freeform_input_from_function_arguments(arguments: str) -> str:
    """Extract freeform tool input from JSON function arguments when possible."""
    import json

    raw = arguments if isinstance(arguments, str) else str(arguments or "")
    stripped = raw.strip()
    if not stripped:
        return ""
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return raw
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        for key in _FREEFORM_ARGUMENT_NAMES:
            value = parsed.get(key)
            if isinstance(value, str) and value:
                return value
        # Single string property fallback.
        string_values = [v for v in parsed.values() if isinstance(v, str) and v]
        if len(string_values) == 1:
            return string_values[0]
    return raw


def write_tools_survived_conversion(report: ToolConversionReport) -> bool:
    """True when conversion produced at least one actionable function tool."""
    return report.converted_count > 0 and not report.unsupported


__all__ = [
    "BridgeToolCompatibilityError",
    "ToolConversionReport",
    "convert_responses_tools",
    "freeform_input_from_function_arguments",
    "responses_item_type_for_tool_call",
    "responses_tool_call_identity",
    "write_tools_survived_conversion",
]
