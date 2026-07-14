"""
ToolCard

Beautiful collapsible card for ToolCall + ToolResult using Textual + Rich.

Features:
- Collapsible container (click header to expand/collapse)
- Call phase: yellow accent, pretty-printed JSON args
- Result phase: green (success) / red (error) accent + result summary
- Syntax highlighting for JSON and (if result looks like code) other content
- Compact one-line header with tool name + status
- Designed to be mounted into ChatLog
"""

from __future__ import annotations

import json
from typing import Any

from rich.json import JSON as RichJSON
from rich.syntax import Syntax
from rich.text import Text
from textual.containers import Vertical
from textual.widgets import Collapsible, Static

from mana_agent.chat.events import ToolCallEvent, ToolResultEvent


class ToolCard(Vertical):
    """
    A self-contained collapsible card representing one tool invocation.

    You can construct it from a ToolCallEvent (pending result) and later
    feed it a ToolResultEvent via .set_result(...).
    """

    DEFAULT_CSS = """
    ToolCard {
        margin: 0 0 1 0;
        padding: 0;
    }
    ToolCard Collapsible {
        border: round $primary;
    }
    .tool-call-header {
        color: $warning;  /* yellow-ish */
        text-style: bold;
    }
    .tool-result-success {
        color: $success;
    }
    .tool-result-error {
        color: $error;
    }
    .tool-args {
        padding: 0 1;
    }
    .tool-result-body {
        padding: 0 1;
    }
    """

    def __init__(
        self,
        call_event: ToolCallEvent,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id or f"tool-{call_event.call_id}")
        self.call_event = call_event
        self.result_event: ToolResultEvent | None = None
        self._collapsible: Collapsible | None = None
        self.content_container: Vertical | None = None

    def compose(self):
        """Build the collapsible card using proper compose context managers.

        All initial children are *yielded* (not mounted via .mount()) so that
        Textual can attach them during the normal mount phase. This avoids
        "Can't mount widget(s) before Vertical() is mounted".
        """
        header = self._build_call_header()

        # Use 'with' so yields inside become children of the Collapsible,
        # and the Collapsible itself becomes a child of this ToolCard (Vertical).
        with Collapsible(collapsed=True, title=header) as collapsible:
            self._collapsible = collapsible

            # Yield the call args as a normal child (no .mount() call here)
            call_body = self._build_call_body()
            yield Static(call_body, classes="tool-args")

            # Empty container for future result(s). We will .mount() into it
            # later (from set_result), after this whole card has been mounted.
            self.content_container = Vertical(classes="tool-result-body")
            yield self.content_container

    def _build_call_header(self) -> str:
        name = self.call_event.tool_name or "tool"
        summary = self.call_event.summary or ""
        return f"🔧 {name}" + (f"  {summary}" if summary else "")

    def _build_call_body(self) -> Text | Syntax | str:
        args = self.call_event.args
        if args is None or args == {} or args == "":
            return Text("(no arguments)", style="dim")
        try:
            if isinstance(args, str):
                # try parse for pretty
                parsed = json.loads(args)
                return RichJSON.from_data(parsed, indent=2)
            else:
                return RichJSON.from_data(args, indent=2)
        except Exception:
            # fallback to raw repr
            return Syntax(str(args), "json", theme="ansi_dark", line_numbers=False)

    def set_result(self, result_event: ToolResultEvent) -> None:
        """Update the card when the matching result arrives. Safe to call multiple times."""
        self.result_event = result_event

        # Update collapsible title with status
        status_icon = "✅" if result_event.success else "❌"
        color_class = "tool-result-success" if result_event.success else "tool-result-error"
        header = f"{status_icon} {result_event.tool_name}"
        if result_event.summary:
            header += f"  {result_event.summary}"
        if result_event.duration_ms is not None:
            header += f"  ({result_event.duration_ms}ms)"

        # Rebuild or append result section
        try:
            if self._collapsible:
                # Do not overwrite the original call header (keeps "🔧 toolname" visible).
                # This prevents the "tool call immediately gone" visual after result.
                # Status is shown inside the result body instead.
                # Open it automatically on result for visibility (premium feel)
                self._collapsible.collapsed = False
        except Exception:
            pass

        result_widget = Static(
            self._build_result_body(result_event),
            classes="tool-result-body " + color_class,
        )

        # Mount the result into the dedicated container.
        # This is safe because set_result is called from a deferred _render_event
        # (via call_after_refresh in ChatLog) after the ToolCard has been mounted.
        if self.content_container:
            # Remove any previous result children (keep the container itself)
            for child in list(self.content_container.children):
                child.remove()
            self.content_container.mount(result_widget)

    def _build_result_body(self, res: ToolResultEvent) -> Text | Syntax | str:
        if not res.success:
            err = res.error or str(res.result) or "Unknown error"
            return Text(f"Error: {err}", style="bold red")

        result = res.result
        if result is None:
            return Text("(no output)", style="dim")

        # Pretty JSON when possible
        if isinstance(result, (dict, list)):
            try:
                return RichJSON.from_data(result, indent=2)
            except Exception:
                pass

        text = str(result)
        # If it looks like code or long structured output, syntax highlight
        if len(text) > 120 or "\n" in text:
            # Heuristic: try json first, else python
            try:
                json.loads(text)
                return Syntax(text, "json", theme="ansi_dark", line_numbers=False)
            except Exception:
                return Syntax(text[:2000], "python", theme="ansi_dark", line_numbers=False)

        return Text(text)

    def __rich__(self) -> Text:
        # Fallback rich representation
        name = self.call_event.tool_name
        return Text(f"tool:{name}", style="yellow")
