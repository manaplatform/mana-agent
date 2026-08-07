import time
from dataclasses import dataclass
from typing import Any, Optional

from rich.live import Live
from rich.text import Text
from langchain_core.callbacks.base import BaseCallbackHandler


@dataclass
class ToolRunState:
    current_tool: Optional[str] = None
    current_input: Optional[str] = None
    started_at: float = 0.0
    last_event: str = ""


class RichToolCallbackHandler(BaseCallbackHandler):
    """Updates a Rich Live display when tools start/end."""

    def __init__(self, state: ToolRunState, live: Live, *, show_inputs: bool = True) -> None:
        self.state = state
        self.live = live
        self.show_inputs = show_inputs

    def _render(self) -> None:
        if not self.state.current_tool:
            self.live.update(Text("Thinking…"))
            return
        elapsed = time.time() - self.state.started_at
        msg = f"Using tool: {self.state.current_tool}  ({elapsed:0.1f}s)"
        if self.show_inputs and self.state.current_input:
            # keep short to avoid destroying the UI
            inp = self.state.current_input
            if len(inp) > 140:
                inp = inp[:140] + "…"
            msg += f"\n↳ {inp}"
        self.live.update(Text(msg))

    # LangChain tool hooks
    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        name = serialized.get("name") or serialized.get("id") or "tool"
        self.state.current_tool = str(name)
        self.state.current_input = input_str
        self.state.started_at = time.time()
        self.state.last_event = "start"
        self._render()

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self.state.last_event = "end"
        # clear current tool, show brief completion
        tool = self.state.current_tool or "tool"
        self.state.current_tool = None
        self.state.current_input = None
        self.live.update(Text(f"Finished: {tool}"))