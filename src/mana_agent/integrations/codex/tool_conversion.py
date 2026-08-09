"""Responses → Chat Completions tool conversion with fail-fast diagnostics.

Codex may emit multiple tool shapes. Chat Completions only represents function
tools. Silently dropping an unrepresentable tool is forbidden: the bridge must
fail with structured compatibility diagnostics (no secrets).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolConversionReport:
    original_count: int
    converted_count: int
    converted_tools: list[dict[str, Any]] = field(default_factory=list)
    unsupported: list[dict[str, Any]] = field(default_factory=list)
    skipped_empty: int = 0

    @property
    def ok(self) -> bool:
        return not self.unsupported and self.converted_count == (
            self.original_count - self.skipped_empty
        )


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
    * ``{"type":"function", ...}`` with nested function dict variants

    Any other tool type, or a function tool missing a name, is unsupported.
    Empty / non-list tools yield an empty conversion (not an error).
    """
    if not isinstance(tools, list):
        return ToolConversionReport(original_count=0, converted_count=0, converted_tools=[])

    converted: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    skipped_empty = 0

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
        # Only function tools are representable on Chat Completions.
        if tool_type != "function":
            # Already Chat-shaped with nested function may still carry type "function".
            # Non-function types (web_search, file_search, computer_use, …) fail.
            if isinstance(tool.get("function"), dict) and tool.get("name") is None:
                # Some clients set type incorrectly but nest a function — still reject
                # unless type is function to avoid inventing compatibility.
                pass
            unsupported.append(
                {
                    "index": index,
                    "type": tool_type,
                    "name": str(tool.get("name") or (tool.get("function") or {}).get("name") or ""),
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
            continue

        name = str(tool.get("name") or "").strip()
        if not name:
            # Empty function shell with no name — skip only truly empty placeholders.
            if not tool.get("description") and not tool.get("parameters") and not tool.get("input_schema"):
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

    report = ToolConversionReport(
        original_count=len(tools),
        converted_count=len(converted),
        converted_tools=converted,
        unsupported=unsupported,
        skipped_empty=skipped_empty,
    )
    if fail_on_unsupported and unsupported:
        raise BridgeToolCompatibilityError(
            (
                "Codex Responses tools cannot be represented on the Chat Completions "
                f"upstream ({provider or 'unknown'}/{model or 'unknown'}). "
                f"original={report.original_count} converted={report.converted_count} "
                f"unsupported={len(unsupported)}. No tools were silently dropped."
            ),
            report=report,
            provider=provider,
            model=model,
            transport=transport,
        )
    return report


def write_tools_survived_conversion(report: ToolConversionReport) -> bool:
    """True when conversion produced at least one actionable function tool."""
    return report.converted_count > 0 and not report.unsupported


__all__ = [
    "BridgeToolCompatibilityError",
    "ToolConversionReport",
    "convert_responses_tools",
    "write_tools_survived_conversion",
]
