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

from rich.text import Text
from textual.containers import Vertical
from textual.widgets import Collapsible, Static

from mana_agent.chat.events import ToolCallEvent, ToolResultEvent
from mana_agent.tui.widgets.selectable_text import SelectableText


def _safe_textual_id(raw: str | None) -> str | None:
    """Return a Textual-safe widget id.

    Textual DOM ids must contain only letters, numbers, underscores or hyphens
    and must not begin with a digit. Call ids from workers can contain ":"
    (e.g. "5061ef...:1"), which we map to "-".
    """
    if not raw:
        return None
    s = str(raw)
    # Keep only allowed chars; map others to hyphen
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in s)
    # Collapse runs of separators
    while "--" in safe:
        safe = safe.replace("--", "-")
    safe = safe.strip("-_")
    if not safe:
        return None
    if safe[0].isdigit():
        safe = "x" + safe
    return safe


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
        height: auto;
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
        height: auto;
    }
    .tool-result-body.-empty {
        display: none;
    }
    .tool-card-header {
        text-style: bold;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        call_event: ToolCallEvent,
        *,
        id: str | None = None,
    ) -> None:
        raw = id or f"tool-{call_event.call_id}"
        super().__init__(id=_safe_textual_id(raw))
        self.call_event = call_event
        self.result_event: ToolResultEvent | None = None
        self._collapsible: Collapsible | None = None
        self.header_line: Static | None = None  # always-visible summary line with full key data
        self.result_body: SelectableText | None = None

    def compose(self):
        """Build the collapsible card using proper compose context managers.

        All initial children are *yielded* (not mounted via .mount()) so that
        Textual can attach them during the normal mount phase. This avoids
        "Can't mount widget(s) before Vertical() is mounted".
        """
        header = self._build_call_header()

        # Always-visible compact line with "full data" (tool + summary).
        # This stays visible even if the details "menu" (Collapsible) is collapsed.
        self.header_line = Static(header, classes="tool-card-header")
        yield self.header_line

        # Collapsible "menu" for verbose/full details (args + result body).
        # Start collapsed. The header_line above ensures data is not lost on collapse.
        with Collapsible(collapsed=True, title="details") as collapsible:
            self._collapsible = collapsible

            # Yield the call args as a normal child (no .mount() call here)
            call_body, call_language = self._build_call_body()
            yield SelectableText(call_body, language=call_language, classes="tool-args")

            # Keep one result widget for the card's lifetime. Replacing a mounted
            # subtree during live updates can leave Textual with stale measurements.
            self.result_body = SelectableText("", classes="tool-result-body -empty")
            yield self.result_body

    def _build_call_header(self) -> str:
        name = self.call_event.tool_name or "tool"
        summary = self.call_event.summary or ""
        return f"🔧 {name}" + (f"  {summary}" if summary else "")

    def _build_call_body(self) -> tuple[str, str | None]:
        args = self.call_event.args
        if args is None or args == {} or args == "":
            return "(no arguments)", None
        try:
            if isinstance(args, str):
                # try parse for pretty
                parsed = json.loads(args)
                return json.dumps(parsed, indent=2, ensure_ascii=False), "json"
            else:
                return json.dumps(args, indent=2, ensure_ascii=False), "json"
        except Exception:
            return str(args), "json"

    def set_result(self, result_event: ToolResultEvent) -> None:
        """Update the card when the matching result arrives. Safe to call multiple times.

        The title is updated with a compact summary of BOTH call + result so that
        even when the Collapsible is collapsed, the "full data" (tool + args summary + result)
        remains visible in the header. Detailed content is still inside when expanded.
        """
        self.result_event = result_event

        status_icon = "✅" if result_event.success else "❌"
        # Build a title that carries the full relevant data even when collapsed.
        # Format: "✅ toolname [call_summary] → [result_summary] (time)"
        call_part = self.call_event.tool_name
        if self.call_event.summary:
            call_part += f" {self.call_event.summary}"

        result_part = result_event.summary or ("success" if result_event.success else "error")
        header = f"{status_icon} {call_part} → {result_part}"
        if result_event.duration_ms is not None:
            header += f" ({result_event.duration_ms}ms)"

        if self._collapsible:
            self._collapsible.title = "details"
        if self.header_line:
            self.header_line.update(header)
        if self.result_body:
            result_text, result_language = self._build_result_body(result_event)
            self.result_body.load_text(result_text)
            self.result_body.language = result_language
            self.result_body.remove_class("-empty")
            self.result_body.set_class(result_event.success, "tool-result-success")
            self.result_body.set_class(not result_event.success, "tool-result-error")
        # Do not force a completed card closed: a user may have expanded it to
        # inspect live output. The initial state is already collapsed.
        self._invalidate_layout()

    def on_collapsible_toggled(self, _: Collapsible.Toggled) -> None:
        """Invalidate this card and its scrollable ancestors after a state change."""
        self._invalidate_layout()

    def _invalidate_layout(self) -> None:
        """Request fresh measurement from the card through the chat timeline."""
        try:
            self.refresh(layout=True)
            for ancestor in self.ancestors:
                ancestor.refresh(layout=True)
        except Exception:
            # A result can arrive before mount; first mount will measure it.
            pass

    def _build_result_body(self, res: ToolResultEvent) -> tuple[str, str | None]:
        if not res.success:
            err = res.error or str(res.result) or "Unknown error"
            return f"Error: {err}", "python" if "Traceback" in err else None

        result = res.result
        if result is None:
            return "(no output)", None

        # Pretty JSON when possible
        if isinstance(result, (dict, list)):
            try:
                return json.dumps(result, indent=2, ensure_ascii=False), "json"
            except Exception:
                return str(result), None

        text = str(result)
        # If it looks like code or long structured output, syntax highlight
        if len(text) > 120 or "\n" in text:
            # Heuristic: try json first, else python
            try:
                json.loads(text)
                return text, "json"
            except Exception:
                return text, "python"

        return text, None

    def __rich__(self) -> Text:
        # Fallback rich representation
        name = self.call_event.tool_name
        return Text(f"tool:{name}", style="yellow")
