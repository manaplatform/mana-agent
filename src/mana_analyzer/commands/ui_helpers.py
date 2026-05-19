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
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks.base import BaseCallbackHandler
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from mana_analyzer.config.settings import default_diagrams_dir
from mana_analyzer.llm.ask_agent import AskAgent
from mana_analyzer.llm.run_logger import LlmRunLogger
from mana_analyzer.services.coding_memory_service import CodingMemoryService

logger = logging.getLogger('mana_analyzer.commands.cli')
UNLIMITED_AGENT_MAX_STEPS = 1_000_000_000


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
    if any(signal in lower for signal in prompt_signals):
        return (
            "Status: executing full-auto workflow. "
            f"Changed files so far: {changed_files_count}. "
            f"Terminal reason: {terminal_reason or 'unknown'}."
        )
    return raw


class RichToolCallbackHandler(BaseCallbackHandler):
    """Logs tool start/end so LiveLogBuffer can show it."""

    def __init__(self, *, show_inputs: bool = True) -> None:
        self.show_inputs = show_inputs
        self._tool: str | None = None
        self._t0: float = 0.0

    def on_tool_start(self, serialized, input_str: str, **kwargs) -> None:
        name = (serialized or {}).get("name") or "tool"
        self._tool = str(name)
        self._t0 = time.time()
        msg = f"TOOL start: {self._tool}"
        if self.show_inputs and input_str:
            inp = input_str.strip().replace("\n", " ")
            if len(inp) > 160:
                inp = inp[:160] + "…"
            msg += f" | args: {inp}"
        logger.info(msg)

    def on_tool_end(self, output: str, **kwargs) -> None:
        tool = self._tool or "tool"
        dt = max(0.0, time.time() - self._t0)
        self._tool = None
        logger.info(f"TOOL end: {tool} ({dt:0.1f}s)")

    def on_tool_error(self, error: BaseException, **kwargs) -> None:
        tool = self._tool or "tool"
        self._tool = None
        logger.info(f"TOOL error: {tool} - {error}")


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


def _run_with_live_buffer(
    console: Console,
    *,
    spinner_text: str,
    fn,  # callable
    callbacks: list[BaseCallbackHandler] | None = None,
    show_all_logs: bool = False,
) -> tuple[object, str]:
    """
    Runs fn() while streaming logs into a Live panel.
    Returns (result, debug_tail_text).
    """
    spinner = Spinner("dots", text=spinner_text)
    with Live(spinner, console=console, refresh_per_second=12, transient=True) as live:
        log_buf = LiveLogBuffer(
            live,
            capacity=250,
            show_lines=14,
            show_all_logs=show_all_logs,
        )
        log_buf.setFormatter(_make_log_formatter())

        # capture everything (root) so you see internal debug + tool messages
        root_logger = logging.getLogger()
        root_logger.addHandler(log_buf)

        try:
            result = fn(callbacks=callbacks or [])
        finally:
            root_logger.removeHandler(log_buf)

    return result, log_buf.tail(35).strip()


def _summary(text: str, limit: int = 260) -> str:
    """Return one-line preview text capped at ``limit`` characters."""
    t = (text or "").strip().replace("\n", " ")
    return t if len(t) <= limit else (t[:limit].rstrip() + "…")


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
        "[bold]Summary[/bold]",
        f"- preview: {preview}",
        f"- answer_chars: {len(answer_text)}",
        f"- preview_truncated: {'yes' if truncated else 'no'}",
        f"- sources: {sources_count}",
        f"- warnings: {warnings_count}",
        f"- tool steps: {tool_steps}",
    ]
    if changed_files_count:
        lines.append(f"- changed_files: {changed_files_count}")
    if has_diff:
        lines.append("- diff: yes")
    return "\n".join(lines)


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
        if "write_file fallback" in lowered:
            rows.append({"decision": "Use write_file fallback", "rationale": warning[:220]})
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
    console.print("\n[bold]Steps[/bold]")
    if not trace:
        console.print("- none")
        return
    trace_table = Table(title="Steps", show_lines=False)
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
    console.print(trace_table)


def _render_decisions_section(console: Console, decisions: list[dict[str, str]]) -> None:
    """Render parsed decision/rationale rows for the current turn."""
    console.print("\n[bold]Decisions[/bold]")
    if not decisions:
        console.print("- none")
        return
    for item in decisions:
        decision = str(item.get("decision", "")).strip()
        rationale = str(item.get("rationale", "")).strip()
        if decision and rationale:
            console.print(f"- {decision} ({rationale})")
        elif decision:
            console.print(f"- {decision}")


def _render_history_section(console: Console, turns: list[ChatTurnTelemetry]) -> None:
    """Render compact session history for transparency/debugging."""
    console.print("\n[bold]History[/bold]")
    if not turns:
        console.print("- none")
        return
    history_table = Table(title="Session History", show_lines=False)
    history_table.add_column("Turn", justify="right")
    history_table.add_column("Time")
    history_table.add_column("Question", overflow="fold")
    history_table.add_column("Answer Preview", overflow="fold")
    history_table.add_column("Steps", justify="right")
    history_table.add_column("Warnings", justify="right")
    history_table.add_column("Decisions", justify="right")
    for turn in turns:
        history_table.add_row(
            str(turn.turn_index),
            turn.timestamp,
            _summary(turn.question, limit=100),
            _summary(turn.answer_text, limit=120),
            str(len(turn.trace)),
            str(len(turn.warnings)),
            str(len(turn.decisions)),
        )
    console.print(history_table)


def _render_turn_transparency(
    console: Console,
    *,
    turn: ChatTurnTelemetry,
    history: list[ChatTurnTelemetry],
) -> None:
    """Render summary, steps, decisions, and history blocks for one turn."""
    console.print(
        "\n"
        + _render_turn_summary(
            answer=turn.answer_text,
            sources_count=len(turn.sources),
            warnings_count=len(turn.warnings),
            tool_steps=turn.tool_steps_total if isinstance(turn.tool_steps_total, int) else len(turn.trace),
            changed_files_count=len(turn.changed_files),
            has_diff=turn.has_diff,
        )
    )
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
    artifact_path, error = _render_mermaid_artifact(
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
    """Read one chat question with optional multiline collection."""
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

    console.print(f"\n[bold]{title}[/bold]")
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
    """Capture recent log records and stream a tail into Rich Live."""

    def __init__(
        self,
        live: Live,
        capacity: int = 250,
        show_lines: int = 14,
        show_all_logs: bool = False,
    ) -> None:
        super().__init__(level=logging.DEBUG)
        self.live = live
        self.records: deque[str] = deque(maxlen=capacity)
        self.show_lines = show_lines
        self.show_all_logs = show_all_logs

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if not self.show_all_logs and not message.startswith("TOOL "):
            return
        try:
            msg = self.format(record)
        except Exception:
            msg = message
        self.records.append(msg)

        tail = list(self.records)[-self.show_lines :]
        body = "\n".join(tail) if tail else ""
        self.live.update(Panel(body, title="Live events", border_style="dim"))

    def tail(self, n: int = 35) -> str:
        return "\n".join(list(self.records)[-n:])
