from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import select
import shutil
import subprocess
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from mana_agent.config.settings import default_diagrams_dir
from mana_agent.llm.ask_agent import AskAgent
from mana_agent.llm.run_logger import LlmRunLogger
from mana_agent.services.coding_memory_service import CodingMemoryService

logger = logging.getLogger('mana_agent.commands.cli')
UNLIMITED_AGENT_MAX_STEPS = 1_000_000_000


def _public_cli_symbol(name: str, default: Any) -> Any:
    public_cli = sys.modules.get("mana_agent.commands.cli")
    if public_cli is None or not hasattr(public_cli, name):
        return default
    public_value = getattr(public_cli, name)
    original_value = globals().get(name, None)
    if default is not original_value and public_value is original_value:
        return default
    return public_value


def _sanitize_full_auto_answer_text(
    text: str,
    *,
    changed_files_count: int = 0,
    terminal_reason: str = "",
) -> str:
    """Replace interactive full-auto prompts with executable status text."""
    raw = str(text or "").strip()
    lower = raw.lower()
    prompt_signals = (
        "if you want",
        "reply yes",
        "please choose",
        "need a scope choice",
        "please share permission",
        "need to read the current repository files",
    )
    if (
        str(terminal_reason or "").strip().lower() == "pass_cap_reached"
        and lower.startswith("auto-execute ended without a direct answer from tool runs.")
    ):
        return (
            "Status: executing full-auto workflow. "
            f"Changed files so far: {changed_files_count}. "
            f"Terminal reason: {terminal_reason or 'unknown'}."
        )
    if any(signal in lower for signal in prompt_signals):
        return (
            "Status: executing full-auto workflow. "
            f"Changed files so far: {changed_files_count}. "
            f"Terminal reason: {terminal_reason or 'unknown'}."
        )
    return raw


class RichToolCallbackHandler(BaseCallbackHandler):
    """Stream tool start/end/error into the active chat log."""

    def __init__(self, *, show_inputs: bool = True) -> None:
        self.show_inputs = show_inputs
        self._tool: str | None = None
        self._t0: float = 0.0
        self._event_id: str | None = None

    def on_tool_start(self, serialized, input_str: str, **kwargs) -> None:
        name = (serialized or {}).get("name") or "tool"
        self._tool = str(name)
        self._t0 = time.time()
        self._event_id = f"callback-{uuid.uuid4().hex}"
        args = ""
        if self.show_inputs and input_str:
            args = input_str.strip().replace("\n", " ")
        emit_tool_event("start", self._tool, args=args, event_id=self._event_id)

    def on_tool_end(self, output: str, **kwargs) -> None:
        tool = self._tool or "tool"
        dt = max(0.0, time.time() - self._t0)
        event_id = self._event_id
        self._tool = None
        self._event_id = None
        emit_tool_event("end", tool, duration=dt, event_id=event_id)

    def on_tool_error(self, error: BaseException, **kwargs) -> None:
        tool = self._tool or "tool"
        event_id = self._event_id
        self._tool = None
        self._event_id = None
        emit_tool_event("error", tool, error=str(error), event_id=event_id)


# -----------------------------------------
# Chat log timeline
# -----------------------------------------

@dataclass
class ChatLogEntry:
    role: str
    content: str = ""
    status: str = ""
    tool_name: str = ""
    tool_args: str = ""
    duration: float | None = None
    run_id: str = ""
    tool_call_id: str = ""
    timestamp: float = field(default_factory=time.time)
    error: str = ""


def _compact_display_text(value: Any, limit: int = 80) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if parsed is not None:
        if isinstance(parsed, dict):
            candidates: list[str] = []
            for key in ("path", "file", "query", "pattern", "command", "url", "glob"):
                raw = parsed.get(key)
                if isinstance(raw, (str, int, float)) and str(raw).strip():
                    candidates.append(str(raw).strip())
            text = " ".join(candidates) if candidates else json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        elif isinstance(parsed, list):
            text = json.dumps(parsed[:3], ensure_ascii=False, separators=(",", ":"))
    text = text.replace("\n", " ")
    text = re.sub(r"https?://[^\s'\"]+", lambda match: _summary(match.group(0), 28), text)
    text = re.sub(r"\s+", " ", text).strip()
    return _summary(text, limit)


def _looks_like_raw_log_record(text: str) -> bool:
    return bool(re.match(r"^\s*(?:\[[A-Z]+\]|\b(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\b)\s+", str(text or "")))


class ChatLog:
    """One ordered chat transcript with stable, updatable tool entries."""

    def __init__(self) -> None:
        self.entries: list[ChatLogEntry] = []
        self._tool_entries: dict[str, ChatLogEntry] = {}

    def add_user(self, content: str) -> ChatLogEntry:
        return self._append(ChatLogEntry(role="user", content=_compact_display_text(content, 220)))

    def add_thinking(self, content: str) -> ChatLogEntry:
        return self._append(ChatLogEntry(role="thinking", content=_compact_display_text(content, 220), status="running"))

    def start_tool(
        self,
        tool_name: str,
        *,
        tool_args: str = "",
        run_id: str = "",
        tool_call_id: str = "",
    ) -> ChatLogEntry:
        key = self._tool_key(tool_name, tool_args, run_id, tool_call_id)
        existing = self._tool_entries.get(key)
        if existing is not None:
            existing.status = "running"
            existing.tool_args = _compact_display_text(tool_args)
            existing.timestamp = time.time()
            return existing
        entry = ChatLogEntry(
            role="tool",
            status="running",
            tool_name=str(tool_name or "tool").strip() or "tool",
            tool_args=_compact_display_text(tool_args),
            run_id=str(run_id or "").strip(),
            tool_call_id=str(tool_call_id or "").strip(),
        )
        self._tool_entries[key] = entry
        self.entries.append(entry)
        return entry

    def finish_tool(
        self,
        tool_name: str,
        *,
        duration: float | None = None,
        run_id: str = "",
        tool_call_id: str = "",
        tool_args: str = "",
    ) -> ChatLogEntry:
        entry = self._resolve_tool(tool_name, tool_args, run_id, tool_call_id)
        entry.status = "success"
        entry.duration = duration
        if tool_args and not entry.tool_args:
            entry.tool_args = _compact_display_text(tool_args)
        return entry

    def fail_tool(
        self,
        tool_name: str,
        *,
        error: str = "",
        duration: float | None = None,
        run_id: str = "",
        tool_call_id: str = "",
        tool_args: str = "",
    ) -> ChatLogEntry:
        entry = self._resolve_tool(tool_name, tool_args, run_id, tool_call_id)
        entry.status = "failure"
        entry.duration = duration
        entry.error = _compact_display_text(error, 120)
        if tool_args and not entry.tool_args:
            entry.tool_args = _compact_display_text(tool_args)
        return entry

    def add_assistant(self, content: str) -> ChatLogEntry:
        return self._append(ChatLogEntry(role="assistant", content=str(content or "").strip(), status="done"))

    def add_error(self, content: str) -> ChatLogEntry:
        if _looks_like_raw_log_record(content):
            return ChatLogEntry(role="error", status="filtered")
        return self._append(ChatLogEntry(role="error", content=_compact_display_text(content, 180), status="failure"))

    def _append(self, entry: ChatLogEntry) -> ChatLogEntry:
        if entry.content:
            self.entries.append(entry)
        return entry

    @staticmethod
    def _tool_key(tool_name: str, tool_args: str, run_id: str, tool_call_id: str) -> str:
        explicit = str(tool_call_id or run_id or "").strip()
        if explicit:
            return explicit
        return f"{str(tool_name or 'tool').strip()}:{str(tool_args or '').strip()}"

    def _resolve_tool(
        self,
        tool_name: str,
        tool_args: str,
        run_id: str,
        tool_call_id: str,
    ) -> ChatLogEntry:
        key = self._tool_key(tool_name, tool_args, run_id, tool_call_id)
        existing = self._tool_entries.get(key)
        if existing is not None:
            return existing
        if not (run_id or tool_call_id or tool_args):
            for entry in reversed(self.entries):
                if entry.role == "tool" and entry.tool_name == str(tool_name or "tool").strip() and entry.status == "running":
                    return entry
        return self.start_tool(tool_name, tool_args=tool_args, run_id=run_id, tool_call_id=tool_call_id)


class ChatLogRenderer:
    def __init__(self, chat_log: ChatLog, *, spinner_text: str = "Working…", max_rows: int = 20) -> None:
        self.chat_log = chat_log
        self.spinner_text = spinner_text
        self.max_rows = max(1, int(max_rows))

    def __rich__(self):
        parts: list[Any] = []
        non_tools = [entry for entry in self.chat_log.entries if entry.role != "tool"]
        tools = [entry for entry in self.chat_log.entries if entry.role == "tool"]
        for entry in non_tools:
            if entry.role == "user":
                parts.append(self._text_panel(entry.content, "user", "green"))
            elif entry.role == "thinking":
                parts.append(self._text_panel(entry.content or self.spinner_text, "thinking", "cyan"))
            elif entry.role == "assistant":
                parts.append(self._text_panel(entry.content, "assistant", "blue"))
            elif entry.role == "error":
                parts.append(self._text_panel(entry.content, "error", "red"))
        if tools:
            parts.append(self._tools_panel(tools[-self.max_rows :]))
        if not parts:
            parts.append(self._text_panel(self.spinner_text, "thinking", "cyan"))
        return Group(*parts)

    @staticmethod
    def _text_panel(content: str, title: str, border_style: str) -> Panel:
        return Panel(
            Text(str(content or "").strip() or "-", overflow="fold"),
            title=title,
            title_align="left",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(0, 1),
        )

    @staticmethod
    def _duration_text(duration: float | None) -> str:
        return f"{float(duration):0.1f}s" if isinstance(duration, (int, float)) else ""

    def _tools_panel(self, tools: list[ChatLogEntry]) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column(no_wrap=True)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(overflow="fold", style="dim")
        table.add_column(justify="right", no_wrap=True, style="dim")
        for entry in tools:
            if entry.status == "success":
                status = Text("✓", style="green")
                detail = entry.tool_args
            elif entry.status == "failure":
                status = Text("✗", style="red")
                detail = entry.error or entry.tool_args
            else:
                status = Text("⠙", style="cyan")
                detail = entry.tool_args
            table.add_row(status, entry.tool_name or "tool", detail, self._duration_text(entry.duration))
        return Panel(table, title="tools", title_align="left", border_style="cyan", box=box.ROUNDED, padding=(0, 1))


class LiveToolActivity:
    """Compatibility wrapper that now renders the chat log tools timeline."""

    def __init__(
        self,
        *,
        spinner_text: str = "Working…",
        show_all_logs: bool = False,
        max_rows: int = 10,
    ) -> None:
        self.spinner_text = spinner_text
        self.show_all_logs = show_all_logs
        self.max_rows = max(1, int(max_rows))
        self.log = ChatLog()
        self._started_at: dict[str, float] = {}
        self._running_meta: dict[str, dict[str, str]] = {}
        self._log_lines: deque[str] = deque(maxlen=14)
        self._ok = 0
        self._failed = 0

    # -- event ingestion --------------------------------------------------
    def handle(
        self,
        kind: str,
        tool: str,
        *,
        args: str = "",
        duration: float | None = None,
        error: str = "",
        event_id: str | None = None,
    ) -> None:
        tool = str(tool or "tool")
        args = str(args or "")
        key = self._event_key(tool, args, event_id)
        if kind == "start":
            self._started_at[key] = time.time()
            self._running_meta[key] = {"tool": tool, "args": args}
            self.log.start_tool(tool, tool_args=args, tool_call_id=event_id or "")
        elif kind == "end":
            key = self._matching_running_key(key, tool, args, event_id)
            self._finish(key, tool, duration=duration, ok=True, event_id=event_id)
        elif kind == "error":
            key = self._matching_running_key(key, tool, args, event_id)
            self._finish(key, tool, duration=duration, ok=False, error=str(error or ""), event_id=event_id)

    @staticmethod
    def _event_key(tool: str, args: str, event_id: str | None) -> str:
        explicit_id = str(event_id or "").strip()
        if explicit_id:
            return explicit_id
        return f"{str(tool or 'tool').strip()}:{str(args or '').strip()}"

    def _matching_running_key(
        self,
        key: str,
        tool: str,
        args: str,
        event_id: str | None,
    ) -> str:
        if key in self._started_at or event_id or args:
            return key
        for running_key, running in reversed(list(self._running_meta.items())):
            if str(running.get("tool", "") or "") == tool:
                return running_key
        return key

    def _finish(
        self,
        key: str,
        tool: str,
        *,
        duration: float | None,
        ok: bool,
        error: str = "",
        event_id: str | None = None,
    ) -> None:
        started = self._started_at.pop(key, None)
        meta = self._running_meta.pop(key, None) or {}
        if duration is None and started is not None:
            duration = max(0.0, time.time() - float(started))
        args = str(meta.get("args", ""))
        if ok:
            self._ok += 1
            self.log.finish_tool(tool, duration=duration, tool_call_id=event_id or key, tool_args=args)
        else:
            self._failed += 1
            self.log.fail_tool(tool, error=error, duration=duration, tool_call_id=event_id or key, tool_args=args)

    def add_log_line(self, line: str) -> None:
        text = str(line or "").strip()
        if text:
            self._log_lines.append(text)

    # -- rendering --------------------------------------------------------
    def __rich__(self):
        return ChatLogRenderer(self.log, spinner_text=self.spinner_text, max_rows=self.max_rows).__rich__()


# Currently-active activity panel, set while a tool run is in progress.
_ACTIVE_TOOL_ACTIVITY: LiveToolActivity | None = None


def _env_flag(name: str) -> bool | None:
    value = str(os.getenv(name, "") or "").strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _use_live_tool_activity(console: Console) -> bool:
    """Return true when Rich Live can update one terminal region cleanly."""
    override = _env_flag("MANA_LIVE_TOOL_ACTIVITY")
    if override is not None:
        return override
    if os.getenv("CI"):
        return False
    if str(os.getenv("TERM", "") or "").strip().lower() in {"", "dumb"}:
        return False
    if bool(getattr(console, "record", False)):
        return False
    if bool(getattr(console, "is_jupyter", False)):
        return False
    return bool(getattr(console, "is_terminal", False)) and bool(
        getattr(console, "is_interactive", False)
    )


def set_active_tool_activity(activity: LiveToolActivity | None) -> None:
    """Register (or clear) the active chat log that tool events should stream into."""
    global _ACTIVE_TOOL_ACTIVITY
    _ACTIVE_TOOL_ACTIVITY = activity


def emit_tool_event(
    kind: str,
    tool: str,
    *,
    args: str = "",
    duration: float | None = None,
    error: str = "",
    event_id: str | None = None,
) -> None:
    """Send a tool start/end/error event to the active chat log, if any."""
    activity = _ACTIVE_TOOL_ACTIVITY
    if activity is None:
        return
    activity.handle(kind, tool, args=args, duration=duration, error=error, event_id=event_id)


# -----------------------------------------
# "Full logging" helpers (added, non-breaking)
# -----------------------------------------


def _make_log_formatter() -> logging.Formatter:
    """Return the default formatter used by live chat log tails."""
    return logging.Formatter("%(levelname)s %(name)s: %(message)s")

def _now_iso() -> str:
    """Return local wall-clock timestamp in second precision for log metadata."""
    return datetime.now().isoformat(timespec="seconds")


def _log_call(fn_name: str, **fields: object) -> None:
    """Emit a structured debug log for function entry."""
    try:
        logger.debug("CALL %s", fn_name, extra={**fields, "ts": _now_iso()})
    except Exception:
        logger.debug("CALL %s (extra logging failed)", fn_name)


def _log_return(fn_name: str, **fields: object) -> None:
    """Emit a structured debug log for function exit."""
    try:
        logger.debug("RETURN %s", fn_name, extra={**fields, "ts": _now_iso()})
    except Exception:
        logger.debug("RETURN %s (extra logging failed)", fn_name)


def _log_exception(fn_name: str, exc: Exception, **fields: object) -> None:
    """Emit a structured exception log, preserving original traceback."""
    try:
        logger.exception("EXCEPTION %s: %s", fn_name, exc, extra={**fields, "ts": _now_iso()})
    except Exception:
        logger.exception("EXCEPTION %s: %s", fn_name, exc)


def _register_tool_if_missing(agent: AskAgent, tool: object) -> None:
    """Register a tool on an AskAgent by name if it is not already present."""
    name = str(getattr(tool, "name", "") or "").strip()
    if not name:
        return
    existing = {str(getattr(item, "name", "")) for item in getattr(agent, "tools", []) or []}
    if name in existing:
        return
    agent.tools.append(tool)


def _resolve_agent_max_steps(
    agent_max_steps: int,
    *,
    agent_unlimited: bool,
    min_steps: int = 1,
    cap: int | None = None,
) -> int:
    """Resolve effective tool-step budget, including optional unlimited mode."""
    if agent_unlimited:
        return max(min_steps, UNLIMITED_AGENT_MAX_STEPS)
    effective = max(min_steps, int(agent_max_steps))
    if cap is not None:
        effective = min(effective, int(cap))
    return effective


_EDIT_INTENT_TOKENS = (
    "integrate",
    "patch",
    "modify",
    "edit",
    "update",
    "rewrite",
    "refactor",
    "fix",
    "implement",
    "build",
    "create",
    "add",
    "remove",
    "delete",
    "edit file",
    "edit this",
    "change file",
    "change this",
    "write file",
    "update file",
    "apply patch",
    "fix this",
    "implement this",
)

_EDIT_TARGET_PATTERN = re.compile(
    r"\b(readme(?:\.md)?|[\w./-]+\.(?:py|md|js|jsx|ts|tsx|json|toml|ya?ml|ini|cfg|txt|sh|go|rs|java|kt|rb|php|swift|sql|c|cc|cpp|h|hpp|cs))\b"
)

_PLAN_TRIGGER_PATTERN = re.compile(
    r"\b("
    r"plan|planning|roadmap|checklist|execution\s+plan|implementation\s+plan|"
    r"auto[-\s]?execute|execute\s+(?:the\s+)?plan|run\s+(?:the\s+)?plan"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_plan_trigger_request(question: str) -> bool:
    text = str(question or "").strip()
    if not text:
        return False
    return bool(_PLAN_TRIGGER_PATTERN.search(text))


def _looks_like_edit_request(question: str) -> bool:
    text = str(question or "").strip()
    if not text:
        return False

    lowered = text.lower()
    has_intent = any(token in lowered for token in _EDIT_INTENT_TOKENS)
    if not has_intent:
        return False

    if _EDIT_TARGET_PATTERN.search(text):
        return True

    directive_prefixes = (
        "add ",
        "build ",
        "create ",
        "delete ",
        "edit ",
        "fix ",
        "implement ",
        "integrate ",
        "modify ",
        "patch ",
        "refactor ",
        "remove ",
        "rewrite ",
        "update ",
    )
    return lowered.startswith(directive_prefixes) or any(
        marker in lowered
        for marker in (
            "apply patch",
            "patch this",
            "write file",
            "edit this",
            "fix this",
            "implement this",
            "implement plan",
        )
    )


# Simple health/chat commands answered directly, without FAISS / RAG / CodingAgent.
_DIRECT_COMMANDS: frozenset[str] = frozenset({"ping", "hello", "hi", "status", "help"})

# Explicit "exact search" requests routed straight to ripgrep/static search.
_EXACT_SEARCH_PATTERN = re.compile(
    r"^\s*(?:search(?:\s+for)?|grep|find|locate|where\s+is)\b[:\s]+(.+)$",
    re.IGNORECASE,
)


def _extract_exact_search_query(question: str) -> str | None:
    """Return the search term for an explicit exact-search request, else None.

    Matches grep/find-style prompts like ``grep foo``, ``search for "bar"`` and
    ``where is parse_config``. Returns the stripped (and unquoted) term.
    """
    text = str(question or "").strip()
    if not text:
        return None
    match = _EXACT_SEARCH_PATTERN.match(text)
    if not match:
        return None
    if _looks_like_edit_request(text):
        return None
    term = match.group(1).strip()
    # Strip a single matching pair of surrounding quotes.
    if len(term) >= 2 and term[0] == term[-1] and term[0] in "\"'":
        term = term[1:-1].strip()
    return term or None


def _classify_direct_command(question: str) -> str | None:
    """Return the direct-command name for trivial health/chat input, else None.

    Matches only when the entire (trimmed) message is a single known token so
    that real questions like "help me fix the parser" are not intercepted.
    """
    text = str(question or "").strip().lower().rstrip("!.?")
    if text in _DIRECT_COMMANDS:
        return text
    return None


# -----------------------------------------
# Chat console chrome (banner / status / headers)
# -----------------------------------------

# ANSI sequence for a colored readline prompt. The \001/\002 guards tell
# readline these bytes are non-printing so line-wrapping math stays correct.
CHAT_PROMPT = "\001\033[1;96m\002💬 ❯ \001\033[0m\002"


def _render_chat_banner(console: Console, *, subtitle: str = "") -> None:
    """Render the welcome banner shown once when chat starts."""
    title = Text()
    title.append("◆ ", style="bright_cyan")
    title.append("mana-agent", style="bold bright_white")
    title.append(" chat", style="bright_cyan")

    body = Text()
    if subtitle:
        body.append(subtitle + "\n\n", style="dim")
    body.append("Ask about this project, or request an edit and I'll dig in.\n")
    body.append("Type ", style="dim")
    body.append("help", style="bold yellow")
    body.append(" for commands · ", style="dim")
    body.append("exit", style="bold yellow")
    body.append("/", style="dim")
    body.append("quit", style="bold yellow")
    body.append(" to leave.", style="dim")

    console.print(
        Panel(
            body,
            title=title,
            title_align="left",
            border_style="bright_cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def _render_chat_status(
    console: Console,
    rows: list[tuple[str, str]],
    *,
    title: str = "session",
) -> None:
    """Render the session configuration as a compact key/value panel."""
    if not rows:
        return
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", justify="right", no_wrap=True)
    grid.add_column(style="white")
    for key, value in rows:
        grid.add_row(str(key), str(value))
    console.print(
        Panel(
            grid,
            title=f"[dim]{title}[/dim]",
            title_align="left",
            border_style="bright_black",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def _render_answer_header(console: Console, title: str = "Answer") -> None:
    """Render a styled separator/header above an answer block."""
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]", style="cyan", align="left"))


def _render_direct_command(
    console: Console,
    command: str,
    *,
    project_root: Path | str,
    index_available: bool,
    coding_agent_active: bool,
    tool_worker_active: bool,
) -> str:
    """Answer a direct command without any search/index dependency.

    Returns the plain-text answer (also printed to ``console``).
    """
    if command == "ping":
        answer = "pong"
    elif command in {"hello", "hi"}:
        answer = "Hello! Ask me about this project, or request an edit/fix and I'll dig in."
    elif command == "help":
        answer = (
            "mana-agent chat commands:\n"
            "- ping / hello / hi — quick health check\n"
            "- status — show index, project root, coding-agent and tool-worker status\n"
            "- help — show this message\n"
            "- exit / quit — leave chat\n"
            "\n"
            "Otherwise just ask: I search the project (semantic index when available,\n"
            "ripgrep/static search otherwise), read files, and can edit/fix code."
        )
    elif command == "status":
        index_line = (
            "available (semantic search enabled)"
            if index_available
            else "missing (using direct project search fallback; run `mana-agent index` to enable)"
        )
        answer = (
            "mana-agent chat status\n"
            f"- project root: {project_root}\n"
            f"- semantic index: {index_line}\n"
            f"- coding agent: {'active' if coding_agent_active else 'inactive'}\n"
            f"- tool worker: {'active' if tool_worker_active else 'inactive'}"
        )
    else:  # pragma: no cover - defensive
        answer = "Unknown command."

    _render_answer_header(console)
    console.print(answer)
    return answer


def _run_with_live_buffer(
    console: Console,
    *,
    spinner_text: str,
    fn,  # callable
    callbacks: list[BaseCallbackHandler] | None = None,
    show_all_logs: bool = False,
    activity: LiveToolActivity | None = None,
    manage_live: bool = True,
) -> tuple[object, str]:
    """
    Runs fn() while collecting tool events into one chat-log transcript.
    Returns (result, debug_tail_text). The visible debug tail is intentionally
    empty; full logger output belongs in the configured log files, not chat UI.
    """
    live_activity = activity or LiveToolActivity(spinner_text=spinner_text, show_all_logs=show_all_logs)
    log_buf = LiveLogBuffer(
        live_activity,
        capacity=250,
        show_lines=14,
        show_all_logs=show_all_logs,
    )
    log_buf.setFormatter(_make_log_formatter())

    # capture everything (root) so verbose mode can surface internal debug logs
    root_logger = logging.getLogger()
    root_logger.addHandler(log_buf)
    if manage_live:
        set_active_tool_activity(live_activity)
    use_live_activity = bool(manage_live and _use_live_tool_activity(console))

    try:
        if manage_live:
            if use_live_activity:
                with Live(live_activity, console=console, refresh_per_second=12, transient=True):
                    result = fn(callbacks=callbacks or [])
            else:
                result = fn(callbacks=callbacks or [])
        else:
            result = fn(callbacks=callbacks or [])
    finally:
        root_logger.removeHandler(log_buf)
        if manage_live:
            set_active_tool_activity(None)
            console.print(live_activity)

    return result, ""


def _summary(text: str, limit: int = 260) -> str:
    """Return one-line preview text capped at ``limit`` characters."""
    t = (text or "").strip().replace("\n", " ")
    return t if len(t) <= limit else (t[:limit].rstrip() + "…")


def _summary_block(text: str, *, limit: int = 520, max_lines: int = 12) -> str:
    """Return a readable multiline preview capped by character and line count."""
    raw = (text or "").strip()
    if not raw:
        return "-"
    if len(raw) > limit:
        raw = raw[:limit].rstrip() + "…"
    lines = raw.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append("…")
    return "\n".join(lines)


def _history_time(timestamp: str) -> str:
    """Compact ISO timestamps for the current chat session history table."""
    raw = str(timestamp or "").strip()
    if "T" not in raw:
        return raw or "-"
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).strftime("%H:%M:%S")
    except ValueError:
        return raw


def _render_turn_summary(
    *,
    answer: str,
    sources_count: int,
    warnings_count: int,
    tool_steps: int,
    changed_files_count: int = 0,
    has_diff: bool = False,
) -> str:
    """Build the markdown summary block shown for each chat turn."""
    answer_text = (answer or "").strip()
    preview = _summary(answer_text)
    truncated = len(preview) < len(answer_text)
    lines = [
        "Summary",
        f"Preview: {preview or '-'}",
        (
            "Stats: "
            f"{len(answer_text)} chars | "
            f"{sources_count} sources | "
            f"{warnings_count} warnings | "
            f"{tool_steps} tool steps"
        ),
        f"Preview truncated: {'yes' if truncated else 'no'}",
    ]
    if changed_files_count:
        lines.append(f"Changed files: {changed_files_count}")
    if has_diff:
        lines.append("Diff: yes")
    return "\n".join(lines)


def _render_summary_section(
    console: Console,
    *,
    turn: "ChatTurnTelemetry",
) -> None:
    """Render a readable per-turn answer summary."""
    answer_text = (turn.answer_text or "").strip()
    preview = _summary_block(answer_text)
    tool_steps = turn.tool_steps_total if isinstance(turn.tool_steps_total, int) else len(turn.trace)

    metrics = Table.grid(expand=True, padding=(0, 1))
    metrics.add_column(justify="right", ratio=1)
    metrics.add_column(ratio=1)
    metrics.add_column(justify="right", ratio=1)
    metrics.add_column(ratio=1)
    metrics.add_row("[bold]Answer[/bold]", f"{len(answer_text)} chars", "[bold]Sources[/bold]", str(len(turn.sources)))
    metrics.add_row("[bold]Tool steps[/bold]", str(tool_steps), "[bold]Warnings[/bold]", str(len(turn.warnings)))
    if turn.changed_files or turn.has_diff:
        diff_text = "yes" if turn.has_diff else "no"
        metrics.add_row("[bold]Changed files[/bold]", str(len(turn.changed_files)), "[bold]Diff[/bold]", diff_text)

    body = Table.grid(padding=(0, 1), expand=True)
    body.add_column(ratio=1)
    body.add_row("[bold]Answer preview[/bold]")
    body.add_row(Markdown(preview))
    body.add_row("")
    body.add_row(metrics)

    console.print()
    console.print(Panel(body, title="Summary", border_style="cyan", box=box.ROUNDED, expand=True))


@dataclass(slots=True)
class ChatTurnTelemetry:
    """Normalized telemetry payload persisted/rendered for a single chat turn."""

    turn_index: int
    timestamp: str
    question: str
    answer_text: str
    sources: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    tool_steps_total: int | None = None
    decisions: list[dict[str, str]] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    has_diff: bool = False
    coding_state: dict[str, Any] = field(default_factory=dict)


def _coerce_trace_items(items: list | None) -> list[dict[str, Any]]:
    """Normalize trace rows from dict/object payloads into a stable dict schema."""
    normalized: list[dict[str, Any]] = []
    for item in list(items or []):
        if isinstance(item, dict):
            tool = str(item.get("tool_name", "")).strip()
            status = str(item.get("status", "")).strip()
            args_summary = str(item.get("args_summary", "")).strip()
            duration = item.get("duration_ms", "")
        else:
            tool = str(getattr(item, "tool_name", "")).strip()
            status = str(getattr(item, "status", "")).strip()
            args_summary = str(getattr(item, "args_summary", "")).strip()
            duration = getattr(item, "duration_ms", "")
        normalized.append(
            {
                "tool_name": tool or "-",
                "status": status or "-",
                "args_summary": args_summary or "-",
                "duration_ms": duration,
            }
        )
    return normalized


def _merge_warnings(warnings: list[str] | None, payload: dict | None) -> list[str]:
    """Merge warning lists from caller and structured payload without duplicates."""
    merged = [str(item).strip() for item in list(warnings or []) if str(item).strip()]
    payload_warnings = payload.get("warnings", []) if isinstance(payload, dict) else []
    if isinstance(payload_warnings, list):
        for item in payload_warnings:
            text = str(item).strip()
            if text and text not in merged:
                merged.append(text)
    return merged


def _decision_rows_from_raw(raw: Any, fallback_rationale: str) -> list[dict[str, str]]:
    """Extract decision rows from list payloads containing dict/str entries."""
    rows: list[dict[str, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                decision = str(item.get("decision", "") or item.get("title", "")).strip()
                rationale = str(item.get("rationale", "") or item.get("reason", "")).strip()
                if decision:
                    rows.append({"decision": decision[:200], "rationale": (rationale or fallback_rationale)[:220]})
            else:
                decision = str(item).strip()
                if decision:
                    rows.append({"decision": decision[:200], "rationale": fallback_rationale[:220]})
    return rows


def _extract_decisions(
    *,
    answer_text: str,
    warnings: list[str],
    payload: dict | None = None,
    result_payload: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Collect and dedupe decision rows from payloads, answer text, and warnings."""
    rows: list[dict[str, str]] = []
    if isinstance(payload, dict):
        rows.extend(_decision_rows_from_raw(payload.get("decisions"), "Provided in structured payload"))
        rows.extend(_decision_rows_from_raw(payload.get("recent_decisions"), "Provided in structured payload"))
    if isinstance(result_payload, dict):
        rows.extend(_decision_rows_from_raw(result_payload.get("decisions"), "Provided in coding result"))
        rows.extend(_decision_rows_from_raw(result_payload.get("recent_decisions"), "Provided in coding result"))

    for raw_line in (answer_text or "").splitlines():
        line = raw_line.strip()
        if line.lower().startswith("decision:"):
            decision = line.split(":", 1)[1].strip()
            if decision:
                rows.append({"decision": decision[:200], "rationale": "Provided in assistant answer"})

    for warning in warnings:
        lowered = warning.lower()
        if "write_file fallback" in lowered or "mutation_failed_no_changes" in lowered:
            rows.append({"decision": "Stop after failed mutation", "rationale": warning[:220]})
        elif "patch-only loop" in lowered or "patch-style retry" in lowered:
            rows.append({"decision": "Stop patch-only retries", "rationale": warning[:220]})

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["decision"], row["rationale"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped[:20]


def _render_steps_section(console: Console, trace: list[dict[str, Any]]) -> None:
    """Render tool-step trace table for the current turn."""
    if not trace:
        console.print(Panel("No tool steps ran for this answer.", title="Steps", border_style="dim", box=box.ROUNDED))
        return
    trace_table = Table(show_header=True, header_style="bold", show_lines=False, box=box.SIMPLE, expand=True)
    trace_table.add_column("Tool")
    trace_table.add_column("Status")
    trace_table.add_column("Duration (ms)", justify="right")
    trace_table.add_column("Args", overflow="fold")
    for item in trace:
        duration = item.get("duration_ms", "")
        if isinstance(duration, (int, float)):
            duration_text = f"{float(duration):.1f}"
        else:
            duration_text = str(duration or "-")
        trace_table.add_row(
            str(item.get("tool_name", "-") or "-"),
            str(item.get("status", "-") or "-"),
            duration_text,
            str(item.get("args_summary", "-") or "-"),
        )
    console.print(Panel(trace_table, title="Steps", border_style="blue", box=box.ROUNDED))


def _render_decisions_section(console: Console, decisions: list[dict[str, str]]) -> None:
    """Render parsed decision/rationale rows for the current turn."""
    if not decisions:
        console.print(Panel("No decisions were recorded for this turn.", title="Decisions", border_style="dim", box=box.ROUNDED))
        return
    decisions_table = Table(show_header=True, header_style="bold", show_lines=False, box=box.SIMPLE, expand=True)
    decisions_table.add_column("Decision", overflow="fold", ratio=2)
    decisions_table.add_column("Rationale", overflow="fold", ratio=3)
    for item in decisions:
        decision = str(item.get("decision", "")).strip()
        rationale = str(item.get("rationale", "")).strip()
        if decision:
            decisions_table.add_row(decision, rationale or "-")
    console.print(Panel(decisions_table, title="Decisions", border_style="magenta", box=box.ROUNDED))


def _render_history_section(console: Console, turns: list[ChatTurnTelemetry]) -> None:
    """Render compact session history for transparency/debugging."""
    if not turns:
        console.print(Panel("No prior turns in this session.", title="History", border_style="dim", box=box.ROUNDED))
        return
    history_table = Table(title="Session History", show_lines=False, box=box.SIMPLE, expand=True)
    history_table.add_column("#", justify="right", no_wrap=True)
    history_table.add_column("Time", no_wrap=True)
    history_table.add_column("Question", overflow="fold", ratio=2)
    history_table.add_column("Answer Preview", overflow="fold", ratio=3)
    history_table.add_column("Signals", overflow="fold", ratio=1)
    for turn in turns:
        signals = (
            f"steps {len(turn.trace)} | "
            f"warn {len(turn.warnings)} | "
            f"dec {len(turn.decisions)}"
        )
        history_table.add_row(
            str(turn.turn_index),
            _history_time(turn.timestamp),
            _summary(turn.question, limit=100),
            _summary(turn.answer_text, limit=120),
            signals,
        )
    console.print(Panel(history_table, title="History", border_style="green", box=box.ROUNDED))


def _render_turn_transparency(
    console: Console,
    *,
    turn: ChatTurnTelemetry,
    history: list[ChatTurnTelemetry],
) -> None:
    """Render summary, steps, decisions, and history blocks for one turn."""
    _render_summary_section(console, turn=turn)
    _render_steps_section(console, turn.trace)
    _render_decisions_section(console, turn.decisions)
    _render_history_section(console, history)


def _log_chat_turn(
    run_logger: LlmRunLogger | None,
    *,
    turn: ChatTurnTelemetry,
    mode: str,
    dir_mode: bool,
    coding_agent: bool,
    flow_id: str | None = None,
    render_mode: str | None = None,
    fallback_reason: str | None = None,
    planning_mode: bool = False,
    planning_question_source: str | None = None,
    planning_question_index: int = 0,
    auto_execute_plan: bool = False,
    auto_execute_passes: int = 0,
    auto_execute_terminal_reason: str | None = None,
    toolsmanager_requests_count: int = 0,
    auto_execute_pass_logs: list[dict[str, Any]] | None = None,
    planner_decisions: list[dict[str, Any]] | None = None,
    prechecklist_source: str | None = None,
    prechecklist_steps_count: int = 0,
    prechecklist_warning: str | None = None,
    tool_execution_backend: str = "",
    tool_execution_run_id: str = "",
    tool_execution_duration_ms: float = 0.0,
    tool_execution_requests_ok: int = 0,
    tool_execution_requests_failed: int = 0,
    full_auto_resume_cycles: int = 0,
    full_auto_passes_total: int = 0,
    full_auto_pass_checkpoints_emitted: int = 0,
    resumed_from_pass_cap: bool = False,
    multiline_input: bool | None = None,
    multiline_terminator: str | None = None,
) -> None:
    """Persist a structured chat turn log record through ``LlmRunLogger``."""
    if run_logger is None:
        return
    try:
        run_logger.log(
            {
                "flow": "chat",
                "mode": mode,
                "dir_mode": bool(dir_mode),
                "coding_agent": bool(coding_agent),
                "flow_id": flow_id,
                "turn_index": int(turn.turn_index),
                "question": turn.question,
                "answer": turn.answer_text,
                "answer_chars": len(turn.answer_text or ""),
                "sources_count": len(turn.sources or []),
                "warnings_count": len(turn.warnings or []),
                "tool_steps": turn.tool_steps_total if isinstance(turn.tool_steps_total, int) else len(turn.trace or []),
                "trace": list(turn.trace or []),
                "decisions": list(turn.decisions or []),
                "changed_files": list(turn.changed_files or []),
                "has_diff": bool(turn.has_diff),
                "coding_state": dict(turn.coding_state or {}),
                "render_mode": str(render_mode or "default"),
                "fallback_reason": str(fallback_reason or ""),
                "planning_mode": bool(planning_mode),
                "planning_question_source": str(planning_question_source or "none"),
                "planning_question_index": int(planning_question_index),
                "auto_execute_plan": bool(auto_execute_plan),
                "auto_execute_passes": int(auto_execute_passes),
                "auto_execute_terminal_reason": str(auto_execute_terminal_reason or ""),
                "toolsmanager_requests_count": int(toolsmanager_requests_count),
                "auto_execute_pass_logs": list(auto_execute_pass_logs or []),
                "planner_decisions": list(planner_decisions or []),
                "prechecklist_source": str(prechecklist_source or ""),
                "prechecklist_steps_count": int(prechecklist_steps_count),
                "prechecklist_warning": str(prechecklist_warning or ""),
                "tool_execution_backend": str(tool_execution_backend or ""),
                "tool_execution_run_id": str(tool_execution_run_id or ""),
                "tool_execution_duration_ms": float(tool_execution_duration_ms or 0.0),
                "tool_execution_requests_ok": int(tool_execution_requests_ok or 0),
                "tool_execution_requests_failed": int(tool_execution_requests_failed or 0),
                "full_auto_resume_cycles": int(full_auto_resume_cycles or 0),
                "full_auto_passes_total": int(full_auto_passes_total or 0),
                "full_auto_pass_checkpoints_emitted": int(full_auto_pass_checkpoints_emitted or 0),
                "resumed_from_pass_cap": bool(resumed_from_pass_cap),
                "multiline_input": bool(multiline_input) if multiline_input is not None else None,
                "multiline_terminator": str(multiline_terminator or ""),
            }
        )
    except Exception:
        logger.debug("Failed to write chat run log", exc_info=True)


def _extract_structured_answer(answer: str) -> tuple[str, dict | None]:
    """Split plain-text answer from optional JSON wrapper payload."""

    def _extract_text_from_blocks(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            text_value = value.get("text")
            if isinstance(text_value, str):
                return text_value.strip()
            if isinstance(text_value, dict):
                nested = text_value.get("value")
                if isinstance(nested, str):
                    return nested.strip()
            for key in ("content", "value"):
                nested = value.get(key)
                if nested is not None:
                    extracted = _extract_text_from_blocks(nested).strip()
                    if extracted:
                        return extracted
            return ""
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    block_type = str(item.get("type", "")).strip().lower()
                    if block_type and block_type not in {"text", "output_text"}:
                        continue
                extracted = _extract_text_from_blocks(item).strip()
                if extracted:
                    parts.append(extracted)
            return "\n".join(parts).strip()
        return ""

    raw = (answer or "").strip()
    if not raw:
        return "", None

    candidates = [raw]
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidates.insert(0, "\n".join(lines[1:-1]).strip())

    for candidate in candidates:
        if not candidate.startswith("{"):
            continue
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            nested_answer = payload.get("answer")
            if isinstance(nested_answer, str):
                return nested_answer.strip(), payload
            nested_text = _extract_text_from_blocks(nested_answer).strip()
            if nested_text:
                return nested_text, payload
            return candidate, payload
        if isinstance(payload, list):
            extracted = _extract_text_from_blocks(payload).strip()
            if extracted:
                return extracted, None

    if raw.startswith("["):
        try:
            payload = ast.literal_eval(raw)
        except Exception:
            payload = None
        if isinstance(payload, list):
            extracted = _extract_text_from_blocks(payload).strip()
            if extracted:
                return extracted, None

    return raw, None


def _normalize_ui_option(item: dict[str, Any], index: int) -> dict[str, Any] | None:
    label = str(item.get("label", "") or "").strip()
    if not label:
        return None
    raw_id = str(item.get("id", "") or "").strip()
    option_id = raw_id or f"option_{index + 1}"
    description = str(item.get("description", "") or "").strip()
    value = item.get("value", option_id)
    return {
        "id": option_id,
        "label": label,
        "value": value,
        "description": description,
    }


def _normalize_ui_block(block: dict[str, Any]) -> dict[str, Any] | None:
    block_type = str(block.get("type", "") or "").strip().lower()
    if block_type == "plan":
        title = str(block.get("title", "") or "").strip() or "Plan"
        objective = str(block.get("objective", "") or "").strip()
        raw_steps = block.get("steps")
        steps: list[dict[str, Any]] = []
        if isinstance(raw_steps, list):
            for item in raw_steps:
                if not isinstance(item, dict):
                    continue
                step_title = str(item.get("title", "") or "").strip()
                if not step_title:
                    continue
                status = str(item.get("status", "pending") or "pending").strip().lower()
                if status not in {"pending", "in_progress", "done", "blocked"}:
                    status = "pending"
                detail = str(item.get("detail", "") or "").strip()
                steps.append({"status": status, "title": step_title, "detail": detail})
        return {
            "type": "plan",
            "title": title,
            "objective": objective,
            "steps": steps,
        }

    if block_type == "diagram":
        title = str(block.get("title", "") or "").strip() or "Diagram"
        fmt = str(block.get("format", "text") or "text").strip().lower()
        if fmt not in {"mermaid", "text"}:
            fmt = "text"
        content = str(block.get("content", "") or "").strip()
        if not content:
            return None
        return {
            "type": "diagram",
            "title": title,
            "format": fmt,
            "content": content,
        }

    if block_type in {"selection", "continue"}:
        prompt = str(block.get("prompt", "") or "").strip()
        if not prompt:
            return None
        raw_id = str(block.get("id", "") or "").strip()
        selection_id = raw_id or ("continue_selection" if block_type == "continue" else "")
        if not selection_id:
            return None
        raw_options = block.get("options")
        if not isinstance(raw_options, list):
            return None
        options: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_options):
            if not isinstance(item, dict):
                continue
            normalized = _normalize_ui_option(item, idx)
            if normalized is not None:
                options.append(normalized)
        if not options:
            return None
        title = str(block.get("title", "") or "").strip()
        if not title:
            title = "Continue" if block_type == "continue" else "Selection"
        return {
            "type": block_type,
            "id": selection_id,
            "title": title,
            "prompt": prompt,
            "options": options,
            "allow_free_text": bool(block.get("allow_free_text", False)),
        }

    return None


def _extract_ui_blocks(payload: dict | None) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw_blocks = payload.get("ui_blocks")
    if not isinstance(raw_blocks, list):
        return []
    blocks: list[dict[str, Any]] = []
    for item in raw_blocks:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_ui_block(item)
        if normalized is not None:
            blocks.append(normalized)
    return blocks


def _infer_ui_blocks_from_answer_text(answer_text: str) -> list[dict[str, Any]]:
    text = str(answer_text or "")
    if not text:
        return []
    blocks: list[dict[str, Any]] = []
    for match in re.finditer(r"```mermaid\s*(.*?)```", text, re.IGNORECASE | re.DOTALL):
        content = str(match.group(1) or "").strip()
        if not content:
            continue
        blocks.append(
            {
                "type": "diagram",
                "title": "Diagram",
                "format": "mermaid",
                "content": content,
            }
        )
    return blocks


def _effective_ui_blocks(answer_text: str, payload: dict | None) -> list[dict[str, Any]]:
    blocks = _extract_ui_blocks(payload if isinstance(payload, dict) else None)
    if blocks:
        return blocks
    return _infer_ui_blocks_from_answer_text(answer_text)


def _render_plan_block(console: Console, block: dict[str, Any]) -> None:
    console.print(f"\n[bold]{block.get('title', 'Plan')}[/bold]")
    objective = str(block.get("objective", "") or "").strip()
    if objective:
        console.print(f"- objective: {objective}")
    steps = block.get("steps")
    if isinstance(steps, list):
        for item in steps:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "pending") or "pending")
            title = str(item.get("title", "") or "step")
            detail = str(item.get("detail", "") or "").strip()
            console.print(f"- [{status}] {title}")
            if detail:
                console.print(f"  {detail}")


def _slugify_filename(text: str, fallback: str = "diagram") -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(text or "")).strip("-_.").lower()
    return slug or fallback


def _detect_mermaid_cli(project_root: Path | None = None) -> list[str] | None:
    mmdc = shutil.which("mmdc")
    if mmdc:
        return [mmdc]
    local_candidates = [Path.cwd() / "node_modules" / ".bin" / "mmdc"]
    if project_root is not None:
        local_candidates.append(project_root / "node_modules" / ".bin" / "mmdc")
    for candidate in local_candidates:
        if candidate.exists() and candidate.is_file():
            return [str(candidate.resolve())]
    return None


def _render_mermaid_artifact(
    content: str,
    *,
    output_dir: Path,
    title: str,
    image_format: str,
    timeout_seconds: int,
    project_root: Path | None = None,
) -> tuple[Path | None, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cli_cmd = _detect_mermaid_cli(project_root=project_root)
    if not cli_cmd:
        return (
            None,
            "Mermaid CLI not found. Install `@mermaid-js/mermaid-cli` and ensure `mmdc` is on PATH.",
        )

    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:10]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = _slugify_filename(title, fallback="diagram")
    src_path = output_dir / f"{stem}-{stamp}-{digest}.mmd"
    out_path = output_dir / f"{stem}-{stamp}-{digest}.{image_format}"

    try:
        src_path.write_text(content, encoding="utf-8")
    except Exception as exc:
        return None, f"Failed to write Mermaid source file: {exc}"

    cmd = [
        *cli_cmd,
        "-i",
        str(src_path),
        "-o",
        str(out_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(5, int(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"Mermaid render timed out after {timeout_seconds}s."
    except Exception as exc:
        return None, f"Mermaid render failed to execute: {exc}"
    finally:
        try:
            src_path.unlink(missing_ok=True)
        except Exception:
            pass

    if proc.returncode != 0:
        stderr = str(proc.stderr or "").strip()
        stdout = str(proc.stdout or "").strip()
        detail = stderr or stdout or f"exit code {proc.returncode}"
        return None, f"Mermaid render failed: {detail}"
    if not out_path.exists():
        return None, f"Mermaid render reported success but output file is missing: {out_path}"

    return out_path.resolve(), None


def _open_artifact_with_default_app(path: Path) -> str | None:
    resolved = path.resolve()
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(resolved)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return None
        if os.name == "nt":
            os.startfile(str(resolved))  # type: ignore[attr-defined]
            return None
        opener = shutil.which("xdg-open")
        if opener:
            subprocess.Popen([opener, str(resolved)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return None
        return "No supported opener found (tried `open` and `xdg-open`)."
    except Exception as exc:
        return f"Failed to open artifact: {exc}"


def _render_diagram_block(
    console: Console,
    block: dict[str, Any],
    *,
    render_images: bool = False,
    output_dir: Path | None = None,
    image_format: str = "svg",
    open_artifact: bool = False,
    timeout_seconds: int = 25,
    project_root: Path | None = None,
) -> None:
    title = str(block.get("title", "Diagram") or "Diagram")
    fmt = str(block.get("format", "text") or "text").lower()
    content = str(block.get("content", "") or "")
    lexer = "mermaid" if fmt == "mermaid" else "text"
    syntax = Syntax(content, lexer, word_wrap=True)
    console.print(Panel(syntax, title=title, border_style="cyan"))
    if fmt != "mermaid" or not render_images:
        return

    effective_dir = output_dir or default_diagrams_dir((project_root or Path.cwd()).resolve())
    artifact_path, error = _public_cli_symbol("_render_mermaid_artifact", _render_mermaid_artifact)(
        content,
        output_dir=effective_dir,
        title=title,
        image_format=image_format,
        timeout_seconds=timeout_seconds,
        project_root=project_root,
    )
    if artifact_path is not None:
        note = f"Rendered `{image_format}` diagram: {artifact_path}"
        if open_artifact:
            open_error = _open_artifact_with_default_app(artifact_path)
            if open_error:
                note += f"\nOpen failed: {open_error}"
            else:
                note += "\nOpened in default app."
        console.print(Panel(note, title="Diagram Artifact", border_style="green"))
        return
    if error:
        console.print(Panel(error, title="Diagram Render Fallback", border_style="yellow"))


def _render_selection_block(console: Console, block: dict[str, Any]) -> None:
    title = str(block.get("title", "Selection") or "Selection")
    prompt = str(block.get("prompt", "") or "").strip()
    options = block.get("options")
    if not isinstance(options, list):
        return
    lines: list[str] = [prompt, ""]
    for idx, item in enumerate(options, start=1):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "") or "").strip() or f"Option {idx}"
        option_id = str(item.get("id", "") or "").strip()
        description = str(item.get("description", "") or "").strip()
        suffix = f" ({option_id})" if option_id else ""
        lines.append(f"{idx}. {label}{suffix}")
        if description:
            lines.append(f"   {description}")
    if bool(block.get("allow_free_text", False)):
        lines.append("")
        lines.append("Or type free text.")
    console.print(Panel("\n".join(lines), title=title, border_style="blue"))


def _render_dynamic_blocks(
    console: Console,
    ui_blocks: list[dict[str, Any]],
    *,
    diagram_render_images: bool = False,
    diagram_output_dir: Path | None = None,
    diagram_format: str = "svg",
    diagram_open_artifact: bool = False,
    diagram_timeout_seconds: int = 25,
    project_root: Path | None = None,
) -> dict[str, bool]:
    rendered: dict[str, bool] = {}
    for block in ui_blocks:
        block_type = str(block.get("type", "") or "")
        if block_type == "plan":
            _render_plan_block(console, block)
            rendered["plan"] = True
            continue
        if block_type == "diagram":
            _render_diagram_block(
                console,
                block,
                render_images=diagram_render_images,
                output_dir=diagram_output_dir,
                image_format=diagram_format,
                open_artifact=diagram_open_artifact,
                timeout_seconds=diagram_timeout_seconds,
                project_root=project_root,
            )
            rendered["diagram"] = True
            continue
        if block_type in {"selection", "continue"}:
            _render_selection_block(console, block)
            rendered[block_type] = True
            continue
    return rendered


def _pending_ui_selection_from_blocks(ui_blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    for block in ui_blocks:
        block_type = str(block.get("type", "") or "")
        if block_type in {"selection", "continue"}:
            return block
    return None


def _normalize_choice_token(text: str) -> str:
    token = str(text or "").strip().lower()
    token = re.sub(r"\s+", " ", token)
    token = re.sub(r"[^a-z0-9 _-]", "", token)
    return token


def _resolve_ui_selection_input(block: dict[str, Any], user_input: str) -> tuple[str, dict[str, Any] | None]:
    options = block.get("options")
    if not isinstance(options, list) or not options:
        return "invalid", None

    raw = str(user_input or "").strip()
    if raw.isdigit():
        index = int(raw) - 1
        if 0 <= index < len(options):
            choice = options[index]
            if isinstance(choice, dict):
                return "option", choice

    token = _normalize_choice_token(raw)
    for item in options:
        if not isinstance(item, dict):
            continue
        option_id = _normalize_choice_token(str(item.get("id", "") or ""))
        label = _normalize_choice_token(str(item.get("label", "") or ""))
        if token and (token == option_id or token == label):
            return "option", item

    if bool(block.get("allow_free_text", False)) and raw:
        return "free_text", {"text": raw}

    return "invalid", None


def _stdin_has_buffered_data(timeout_seconds: float) -> bool:
    """Return True when stdin has immediate buffered bytes available."""
    timeout = max(0.0, float(timeout_seconds))
    try:
        fileno = sys.stdin.fileno()
    except Exception:
        return False
    try:
        ready, _, _ = select.select([fileno], [], [], timeout)
    except Exception:
        return False
    return bool(ready)


def _read_chat_input(
    console: Console,
    *,
    prompt: str,
    multiline_enabled: bool,
    multiline_terminator: str,
) -> str:
    """Read one chat question.

    Prefers the prompt_toolkit input box (Enter sends, Shift+Enter / Alt+Enter /
    Ctrl+J insert a newline) when running in an interactive terminal. Falls back
    to plain ``input()`` collection (tests, pipes, CI, or when prompt_toolkit is
    unavailable), preserving the legacy ``/paste`` + terminator behavior.
    """
    if multiline_enabled:
        try:
            from mana_agent.commands.chat_input import (
                prompt_toolkit_available,
                read_chat_input,
            )

            if prompt_toolkit_available():
                return read_chat_input()
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("prompt_toolkit input unavailable; using plain input", extra={"error": str(exc)})

    first_line = input(prompt)
    normalized_first_line = first_line.strip()
    force_multiline = bool(multiline_enabled and normalized_first_line == "/paste")
    try:
        stdin_is_tty = bool(sys.stdin.isatty())
    except Exception:
        stdin_is_tty = False
    auto_multiline = bool(
        multiline_enabled
        and not force_multiline
        and stdin_is_tty
        and _stdin_has_buffered_data(timeout_seconds=0.02)
    )
    if not (force_multiline or auto_multiline):
        return normalized_first_line

    console.print(
        Panel(
            f"Multiline input active.\nSubmit with `{multiline_terminator}` on its own line.",
            title="Multiline Input",
            border_style="blue",
        )
    )
    lines: list[str] = [] if force_multiline else [first_line]
    while True:
        line = input("... ")
        if line == multiline_terminator:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _render_answer_sections(
    console: Console,
    *,
    answer: str,
    title: str = "Answer",
    sources: list | None = None,
    warnings: list[str] | None = None,
    trace: list | None = None,
    show_warnings: bool = True,
    show_trace: bool = True,
) -> None:
    answer_text, payload = _extract_structured_answer(answer)

    payload_sources = payload.get("sources", []) if isinstance(payload, dict) else []
    payload_trace = payload.get("trace", []) if isinstance(payload, dict) else []
    payload_warnings = payload.get("warnings", []) if isinstance(payload, dict) else []

    merged_sources = list(sources or [])
    if not merged_sources and isinstance(payload_sources, list):
        merged_sources = payload_sources

    merged_trace = list(trace or [])
    if not merged_trace and isinstance(payload_trace, list):
        merged_trace = payload_trace

    merged_warnings = list(warnings or [])
    if isinstance(payload_warnings, list):
        for item in payload_warnings:
            txt = str(item).strip()
            if txt and txt not in merged_warnings:
                merged_warnings.append(txt)

    _render_answer_header(console, title)
    if answer_text:
        console.print(Markdown(answer_text))
    else:
        console.print("[dim](no answer text)[/dim]")

    if merged_sources:
        table = Table(title="Sources", show_lines=False)
        table.add_column("File", overflow="fold")
        table.add_column("Lines", justify="right")
        table.add_column("Symbol", overflow="fold")
        table.add_column("Score", justify="right")

        for item in merged_sources[:12]:
            if isinstance(item, dict):
                file_path = str(item.get("file_path", ""))
                start_line = int(item.get("start_line", 0) or 0)
                end_line = int(item.get("end_line", 0) or 0)
                symbol_name = str(item.get("symbol_name", ""))
                score_val = item.get("score", "")
            else:
                file_path = str(getattr(item, "file_path", ""))
                start_line = int(getattr(item, "start_line", 0) or 0)
                end_line = int(getattr(item, "end_line", 0) or 0)
                symbol_name = str(getattr(item, "symbol_name", ""))
                score_val = getattr(item, "score", "")

            lines = f"{start_line}-{end_line}" if start_line or end_line else "-"
            score = "-"
            if isinstance(score_val, (int, float)):
                score = f"{float(score_val):.3f}"
            elif str(score_val).strip():
                score = str(score_val)
            table.add_row(file_path or "-", lines, symbol_name or "-", score)

        console.print(table)

    if show_warnings and merged_warnings:
        warning_lines = "\n".join(f"- {w}" for w in merged_warnings[:12])
        console.print(Panel(warning_lines, title="Warnings", border_style="yellow"))

    if show_trace and merged_trace:
        trace_table = Table(title="Tool Trace", show_lines=False)
        trace_table.add_column("Tool")
        trace_table.add_column("Status")
        trace_table.add_column("Duration (ms)", justify="right")
        trace_table.add_column("Args", overflow="fold")
        for item in merged_trace[:12]:
            if isinstance(item, dict):
                tool = str(item.get("tool_name", ""))
                status = str(item.get("status", ""))
                duration = item.get("duration_ms", "")
                args_summary = str(item.get("args_summary", ""))
            else:
                tool = str(getattr(item, "tool_name", ""))
                status = str(getattr(item, "status", ""))
                duration = getattr(item, "duration_ms", "")
                args_summary = str(getattr(item, "args_summary", ""))
            if isinstance(duration, (int, float)):
                duration_text = f"{float(duration):.1f}"
            else:
                duration_text = str(duration or "-")
            trace_table.add_row(tool or "-", status or "-", duration_text, args_summary or "-")
        console.print(trace_table)


def _render_coding_sections(
    console: Console,
    result: dict[str, Any],
    *,
    rendered_dynamic: dict[str, bool] | None = None,
    show_actions: bool = True,
    show_warnings: bool = True,
) -> None:
    plan = result.get("plan")
    progress = result.get("progress") or {}
    checklist = result.get("checklist") or {}
    actions = result.get("actions_taken") or []
    raw_actions_total = result.get("actions_taken_total", len(actions))
    actions_total = int(raw_actions_total) if isinstance(raw_actions_total, int) else len(actions)
    actions_truncated = bool(result.get("actions_taken_truncated", actions_total > len(actions)))
    changed_files = result.get("changed_files") or []
    static_analysis = result.get("static_analysis") or {}
    next_step = str(result.get("next_step", "") or "").strip()
    warnings = result.get("warnings") or []

    plan_already_rendered = bool((rendered_dynamic or {}).get("plan", False))
    if isinstance(plan, dict) and not plan_already_rendered:
        console.print("\n[bold]Plan[/bold]")
        objective = str(plan.get("objective", "")).strip()
        if objective:
            console.print(f"- objective: {objective}")
        steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
        for step in steps[:12]:
            if isinstance(step, dict):
                status = str(step.get("status", "pending"))
                title = str(step.get("title", "") or "step")
                console.print(f"- [{status}] {title}")

    if progress:
        console.print("\n[bold]Progress[/bold]")
        console.print(f"- phase: {progress.get('phase', '-')}")
        console.print(f"- why: {progress.get('why', '-')}")
        budgets = progress.get("budgets")
        if isinstance(budgets, dict):
            console.print(
                f"- search: {budgets.get('search_used', 0)}/{budgets.get('search_budget', 0)} | "
                f"read: {budgets.get('read_used', 0)}/{budgets.get('read_budget', 0)} | "
                f"read-files: {budgets.get('read_files_observed', 0)}/{budgets.get('required_read_files', 0)}"
            )
            if "read_line_window" in budgets:
                console.print(
                    f"- read-window: {budgets.get('read_line_window', 0)} | "
                    f"dynamic: {bool(budgets.get('dynamic_read_budget_used', False))} | "
                    f"fallback: {bool(budgets.get('dynamic_read_budget_fallback_used', False))}"
                )

    if checklist:
        console.print("\n[bold]Checklist[/bold]")
        console.print(
            f"- done: {checklist.get('done', 0)} | pending: {checklist.get('pending', 0)} | "
            f"blocked: {checklist.get('blocked', 0)} | total: {checklist.get('total', 0)}"
        )

    if show_actions and actions:
        if actions_truncated and actions_total > len(actions):
            console.print(f"- actions shown: {len(actions)}/{actions_total}")
        trace_table = Table(title="Actions Taken", show_lines=False)
        trace_table.add_column("Tool")
        trace_table.add_column("Status")
        trace_table.add_column("Duration (ms)", justify="right")
        trace_table.add_column("Args", overflow="fold")
        for item in actions[:12]:
            if isinstance(item, dict):
                tool = str(item.get("tool_name", ""))
                status = str(item.get("status", ""))
                duration = item.get("duration_ms", "")
                args_summary = str(item.get("args_summary", ""))
                if isinstance(duration, (int, float)):
                    duration_text = f"{float(duration):.1f}"
                else:
                    duration_text = str(duration or "-")
                trace_table.add_row(tool or "-", status or "-", duration_text, args_summary or "-")
        console.print(trace_table)

    if changed_files:
        console.print("\n[bold]Files Changed[/bold]")
        for file_path in changed_files[:30]:
            console.print(f"- {file_path}")

    if static_analysis:
        console.print("\n[bold]Verification[/bold]")
        console.print(f"- finding_count: {static_analysis.get('finding_count', 0)}")

    if show_warnings and warnings:
        warning_lines = "\n".join(f"- {w}" for w in warnings[:12])
        console.print(Panel(warning_lines, title="Warnings", border_style="yellow"))

    console.print("\n[bold]Next Step[/bold]")
    console.print(next_step or "-")


def _build_flow_summary_payload(
    coding_memory_service: CodingMemoryService,
    flow_id: str,
) -> dict[str, Any] | None:
    summary = coding_memory_service.get_flow_summary(flow_id)
    if summary is None:
        return None
    return {
        "flow_id": summary.flow_id,
        "objective": summary.objective,
        "updated_at": summary.updated_at,
        "constraints": summary.constraints,
        "acceptance": summary.acceptance,
        "open_tasks": summary.open_tasks,
        "recent_decisions": summary.recent_decisions,
        "last_changed_files": summary.last_changed_files,
        "unresolved_static_findings": summary.unresolved_static_findings,
        "checklist": summary.checklist,
        "transitions": summary.transitions,
        "last_blocked_reason": summary.last_blocked_reason,
        "recent_turns": coding_memory_service.list_recent_turns(summary.flow_id),
    }


def _render_flow_summary(
    console: Console,
    summary: dict[str, Any],
    *,
    include_checklist: bool,
    include_transitions: bool,
    include_recent_turns: bool,
) -> None:
    console.print(f"[bold]Flow[/bold]: {summary['flow_id']}")
    console.print(f"[bold]Objective[/bold]: {summary['objective']}")
    updated_at = str(summary.get("updated_at", "") or "").strip()
    if updated_at:
        console.print(f"[bold]Updated[/bold]: {updated_at}")

    constraints = summary.get("constraints", []) or []
    if constraints:
        console.print("[bold]Constraints[/bold]")
        for item in constraints:
            console.print(f"- {item}")

    acceptance = summary.get("acceptance", []) or []
    if acceptance:
        console.print("[bold]Acceptance Criteria[/bold]")
        for item in acceptance:
            console.print(f"- {item}")

    tasks = summary.get("open_tasks", []) or []
    if tasks:
        console.print("[bold]Open tasks[/bold]")
        for item in tasks:
            console.print(f"- [ ] {item}")

    decisions = summary.get("recent_decisions", []) or []
    if decisions:
        console.print("[bold]Recent decisions[/bold]")
        for item in decisions:
            if isinstance(item, dict):
                decision = str(item.get("decision", "")).strip()
                rationale = str(item.get("rationale", "")).strip()
                if decision and rationale:
                    console.print(f"- {decision} ({rationale})")
                elif decision:
                    console.print(f"- {decision}")

    files = summary.get("last_changed_files", []) or []
    if files:
        console.print("[bold]Last changed files[/bold]")
        for item in files[:20]:
            console.print(f"- {item}")

    static_findings = summary.get("unresolved_static_findings", []) or []
    if static_findings:
        console.print("[bold]Unresolved static findings[/bold]")
        for item in static_findings[:20]:
            console.print(f"- {item}")

    blocked_reason = str(summary.get("last_blocked_reason", "") or "").strip()
    if blocked_reason:
        console.print(f"[bold]Last blocked reason[/bold]: {blocked_reason}")

    if include_checklist:
        checklist = summary.get("checklist")
        if isinstance(checklist, dict):
            _render_flow_checklist(console, checklist)

    if include_transitions:
        transitions = summary.get("transitions", [])
        if isinstance(transitions, list) and transitions:
            console.print("[bold]Recent transitions[/bold]")
            for item in transitions[:20]:
                if isinstance(item, dict):
                    from_phase = str(item.get("from_phase", "")).strip() or "?"
                    to_phase = str(item.get("to_phase", "")).strip() or "?"
                    reason = str(item.get("reason", "")).strip()
                    if reason:
                        console.print(f"- {from_phase} -> {to_phase}: {reason}")
                    else:
                        console.print(f"- {from_phase} -> {to_phase}")

    if include_recent_turns:
        turns = summary.get("recent_turns", [])
        if isinstance(turns, list) and turns:
            console.print("[bold]Recent turns[/bold]")
            for item in turns[:20]:
                if not isinstance(item, dict):
                    continue
                created_at = str(item.get("created_at", "")).strip()
                request = str(item.get("user_request", "")).strip()
                changed_count = len(item.get("changed_files", []) or [])
                warnings_count = len(item.get("warnings", []) or [])
                console.print(
                    f"- {created_at or '-'} | request: {request[:80] or '-'}"
                    f" | changed_files={changed_count} warnings={warnings_count}"
                )


def _render_flow_checklist(console: Console, checklist: dict[str, Any]) -> None:
    console.print("[bold]Flow Checklist[/bold]")
    objective = str(checklist.get("objective", "")).strip()
    if objective:
        console.print(f"- objective: {objective}")
    steps = checklist.get("steps") if isinstance(checklist.get("steps"), list) else []
    for step in steps[:20]:
        if isinstance(step, dict):
            console.print(
                f"- [{step.get('status', 'pending')}] {step.get('title', 'step')}"
            )


def _derive_checklist_counts_from_steps(steps: list[Any]) -> dict[str, int] | None:
    if not steps:
        return None
    done = 0
    blocked = 0
    pending = 0
    total = 0
    for item in steps:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "pending") or "pending").strip().lower()
        if status not in {"pending", "in_progress", "done", "blocked"}:
            status = "pending"
        total += 1
        if status == "done":
            done += 1
        elif status == "blocked":
            blocked += 1
        else:
            pending += 1
    if total <= 0:
        return None
    return {"done": done, "pending": pending, "blocked": blocked, "total": total}


def _resolve_payload_checklist_counts(payload: dict[str, Any]) -> dict[str, int] | None:
    checklist = payload.get("checklist")
    if isinstance(checklist, dict):
        done = int(checklist.get("done", 0) or 0)
        pending = int(checklist.get("pending", 0) or 0)
        blocked = int(checklist.get("blocked", 0) or 0)
        total_raw = checklist.get("total")
        if total_raw is None:
            total = max(0, done + pending + blocked)
        else:
            total = max(0, int(total_raw or 0))
        if total > 0 or done > 0 or pending > 0 or blocked > 0:
            return {"done": done, "pending": pending, "blocked": blocked, "total": total}

    plan = payload.get("plan")
    if isinstance(plan, dict):
        steps = plan.get("steps")
        if isinstance(steps, list):
            return _derive_checklist_counts_from_steps(steps)
    return None


def _checkpoint_decisions_from_pass_window(
    planner_decisions: list[dict[str, Any]],
    pass_logs: list[dict[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in reversed(planner_decisions):
        decision = str(item.get("decision", "") or "").strip()
        rationale = str(item.get("decision_reason", "") or item.get("rationale", "") or "").strip()
        if not decision:
            continue
        key = (decision, rationale)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"decision": decision[:200], "rationale": rationale[:220]})
        if len(rows) >= 3:
            return rows

    for item in reversed(pass_logs):
        decision = str(item.get("planner_decision", "") or "").strip()
        rationale = str(item.get("planner_decision_reason", "") or "").strip()
        if not decision:
            continue
        key = (decision, rationale)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"decision": decision[:200], "rationale": rationale[:220]})
        if len(rows) >= 3:
            break
    return rows


def _render_full_auto_checkpoint(
    console: Console,
    *,
    decision_rows: list[dict[str, str]],
    checklist_counts: dict[str, int] | None,
    window_passes: int,
    pass_total: int,
    resume_cycles: int,
) -> None:
    console.print("\n[bold cyan]Full-auto Checkpoint[/bold cyan]")
    decisions = list(decision_rows or [])[:3]
    if decisions:
        for item in decisions:
            decision = str(item.get("decision", "")).strip()
            rationale = str(item.get("rationale", "")).strip()
            if decision and rationale:
                console.print(f"- decision: {decision} ({rationale})")
            elif decision:
                console.print(f"- decision: {decision}")
    else:
        console.print("- decision: none")

    if checklist_counts is not None:
        console.print(
            f"- checklist: done {checklist_counts['done']} | pending {checklist_counts['pending']} | "
            f"blocked {checklist_counts['blocked']} | total {checklist_counts['total']}"
        )
    else:
        console.print("- checklist: unavailable")

    console.print(
        f"- status: resuming full-auto (resume_cycle={resume_cycles}; window_passes={window_passes}; passes_total={pass_total})"
    )


class LiveLogBuffer(logging.Handler):
    """Keep compatibility with the old live-buffer hook without rendering logs."""

    def __init__(
        self,
        activity: LiveToolActivity | None = None,
        capacity: int = 250,
        show_lines: int = 14,
        show_all_logs: bool = False,
    ) -> None:
        super().__init__(level=logging.DEBUG)
        self.activity = activity
        self.records: deque[str] = deque(maxlen=capacity)
        self.show_lines = show_lines
        self.show_all_logs = show_all_logs

    def emit(self, record: logging.LogRecord) -> None:
        _ = record
        return

    def tail(self, n: int = 35) -> str:
        _ = n
        return ""
