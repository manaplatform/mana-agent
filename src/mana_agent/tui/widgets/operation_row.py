"""Compact operation summary with selectable, paged details."""

from __future__ import annotations

import asyncio
from typing import Any

from rich.markup import escape
from textual.containers import HorizontalScroll
from textual.widgets import Button, Collapsible

from mana_agent.chat.activity import is_mutation, safe_text
from mana_agent.chat.operations import operation_details
from mana_agent.tui.presentation import glyph
from mana_agent.tui.widgets.selectable_text import SelectableText


class OperationOutput(SelectableText):
    diff_lines: frozenset[int] = frozenset()

    def get_line(self, line_index):
        line = super().get_line(line_index)
        sign = line.plain[:1]
        if line_index in self.diff_lines and sign in {"+", "-"} and not line.plain.startswith(("+++", "---")):
            colors = self.app.get_css_variables()
            line.stylize(colors["success" if sign == "+" else "error"])
        return line


def observed_changes(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    meta = event.get("metadata") or {}
    if event.get("execution_mode", meta.get("phase")) in {"plan", "plan_only"}:
        return {}
    changes = {
        row["path"]: row for row in (meta.get("mutations") or [])
        if isinstance(row, dict) and isinstance(row.get("path"), str) and row.get("status") in {"applied", "partial"}
    }
    mutation = meta.get("mutation")
    if isinstance(mutation, dict) and isinstance(mutation.get("path"), str) and mutation.get("status") in {"applied", "partial"}:
        changes.setdefault(mutation["path"], mutation)
    if is_mutation(event):
        path = event.get("path") or meta.get("path")
        if isinstance(path, str) and path:
            changes.setdefault(path, meta)
        for path in event.get("changed_files") or meta.get("changed_files") or ():
            if isinstance(path, str) and path:
                changes.setdefault(path, {})
    return changes


def _change_label(path: str, change: dict[str, Any]) -> str:
    verb = {"create": "Created", "delete": "Deleted", "rename": "Renamed"}.get(change.get("operation"), "Edited")
    label = f"{verb} {path}"
    added, removed = change.get("added_lines"), change.get("removed_lines")
    if type(added) is int and type(removed) is int and not change.get("binary"):
        label += f" (+{added} -{removed})"
    return label


def operation_title(event: dict[str, Any]) -> str:
    meta = event.get("metadata") or {}
    kind = str(event.get("type") or "")
    status = str(event.get("status") or "running")
    command = event.get("command") or meta.get("command")
    path = event.get("path") or meta.get("path")
    title = str(event.get("title") or event.get("tool_name") or kind.replace(".", " "))
    changes = observed_changes(event)
    file_labels = [_change_label(file_path, entry) for file_path, entry in list(changes.items())[:3]]
    if len(changes) > 3:
        file_labels.append(f"+{len(changes) - 3} more files")
    if command:
        title = f"{'Running' if status == 'running' else 'Ran'} {command}"
    elif file_labels:
        title = file_labels.pop(0)
    elif path and (kind.startswith("file.read") or event.get("tool_name") in {"read_file", "repo_batch_read"}):
        title = f"{'Reading' if status == 'running' else 'Read'} {path}"
    elif path and path not in title:
        title = f"{title} {path}"
    if status in {"failed", "error", "failure", "timeout", "timed_out", "cancelled", "interrupted", "unverified", "rejected"}:
        title = f"{title} [{status}]"
    summary = str(meta.get("result_summary") or event.get("summary") or "")
    if summary and summary not in {title, status} and not any(word in str(summary).lower() for word in ("command output", "invocation")):
        if str(summary).lower() not in {str(event.get("title", "")).lower(), str(event.get("detail", "")).lower()} or command:
            title = f"{title} · {summary}"
    return "\n  ".join(safe_text(line, 190).replace("\n", " ") for line in (title, *file_labels))


class OperationRow(Collapsible):
    DEFAULT_CSS = """
    OperationRow { padding: 0; margin: 0; height: auto; background: transparent; border-top: none; }
    OperationRow CollapsibleTitle { padding: 0; height: auto; color: $text; background: transparent; }
    OperationRow Contents { margin: 0 0 0 2; padding: 0; background: transparent; }
    OperationRow OperationOutput { max-height: 14; }
    OperationRow HorizontalScroll { height: 2; }
    OperationRow Button { height: 1; border: none; min-width: 1; padding: 0 1; }
    """

    def __init__(self, event: dict[str, Any]) -> None:
        self.event = event
        self.output = OperationOutput()
        self.copy_command = Button("Copy command", classes="operation-copy-command")
        self.copy_diff = Button("Copy diff", classes="operation-copy-diff")
        self.retained = Button("Retained output", classes="operation-retained-output")
        self._loaded: dict[str, str] = {}
        self._offsets: dict[str, int] = {}
        self._complete: set[str] = set()
        self._truncated = False
        self._load_error = ""
        self._pending_text: str | None = None
        super().__init__(
            self.output, HorizontalScroll(self.copy_command, self.copy_diff, self.retained),
            title=escape(operation_title(event)), collapsed=True,
            collapsed_symbol=glyph("•", "*"), expanded_symbol=glyph("▾", "-"),
        )

    def update_operation(self, event: dict[str, Any]) -> None:
        if self.event != event:
            self.event = event
            self.title = escape(operation_title(event))
            self._update_details()

    def on_collapsible_expanded(self, _: Collapsible.Expanded) -> None:
        self._update_details()

    def on_text_area_selection_changed(self, event) -> None:
        if self._pending_text is not None and event.selection.is_empty:
            self._update_details()

    def _update_details(self) -> None:
        if self.collapsed:
            return
        event = self.event
        meta = event.get("metadata") or {}
        body = [operation_details(event)]
        for key, label in (
            ("duration_ms", "Duration (ms)"), ("exit_code", "Exit code"), ("display_arguments", "Arguments"),
            ("display_argv", "Command arguments"), ("tool_name", "Tool"), ("backend", "Backend"),
            ("agent_id", "Agent"), ("parent_agent_id", "Parent agent"), ("task_id", "Task"),
            ("attempt_id", "Attempt"), ("operation_attempt_id", "Operation attempt"),
            ("result_summary", "Result"), ("error", "Error"), ("termination_reason", "Termination"),
        ):
            value = meta.get(key, event.get(key))
            if value is not None and value != "" and value != [] and value != {}:
                body.append(f"{label}: {safe_text(str(value))}")
        if event.get("detail"):
            body.append(str(event["detail"]))
        if event.get("output_preview") and not meta.get("streams"):
            body.append(str(event["output_preview"]))
        body.extend(f"--- retained {channel} ---\n{content}" for channel, content in self._loaded.items() if content)
        if self._truncated:
            body.append("Retained trace is incomplete.")
        if self._load_error:
            body.append(self._load_error)
        content = "\n".join(part for part in body if part) or "No details recorded."
        self.copy_command.display = bool(event.get("command") or meta.get("command"))
        self.copy_diff.display = bool(self._loaded.get("diff") or meta.get("diff") or (meta.get("streams") or {}).get("diff"))
        self.retained.display = all(event.get(key) for key in ("conversation_id", "execution_id", "operation_id", "repository_id"))
        if self.output.selected_text:
            self._pending_text = content
        elif content != self.output.text:
            self._pending_text = None
            position = self.output.scroll_offset
            in_diff = False
            diff_lines = set()
            for index, line in enumerate(content.splitlines()):
                if line.startswith("--- ") and line.endswith(" ---"):
                    in_diff = "diff" in line
                if in_diff:
                    diff_lines.add(index)
            self.output.diff_lines = frozenset(diff_lines)
            self.output.load_text(content)
            if self.output.is_mounted:
                self.output.scroll_to(x=position.x, y=position.y, animate=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button not in (self.copy_command, self.copy_diff, self.retained):
            return
        event.stop()
        meta = self.event.get("metadata") or {}
        if event.button is self.copy_command:
            self.app.copy_to_clipboard(str(self.event.get("command") or meta.get("command") or ""))
        elif event.button is self.copy_diff:
            self.app.copy_to_clipboard(str(self._loaded.get("diff") or (meta.get("streams") or {}).get("diff") or meta.get("diff") or ""))
        else:
            self.retained.disabled = True
            self.run_worker(self.load_retained(), group="retained-output", exclusive=True)

    async def load_retained(self) -> None:
        from mana_agent.services.conversation_service import ConversationService
        from mana_agent.services.operation_trace import operation_content

        event = self.event
        conversation = str(event.get("conversation_id") or "")
        execution = str(event.get("execution_id") or "")
        meta = event.get("metadata") or {}
        self._load_error = ""
        try:
            service = ConversationService(repository_id=str(event.get("repository_id") or ""))
            for channel in ("stdout", "stderr", "combined", "diff", "result"):
                if channel in self._complete:
                    continue
                page = await asyncio.to_thread(
                    operation_content, service, conversation, execution,
                    str(event.get("operation_id") or ""), str(event.get("attempt_id") or ""),
                    operation_attempt_id=str(event.get("operation_attempt_id") or meta.get("operation_attempt_id") or ""),
                    channel=channel, offset=self._offsets.get(channel, 0), limit=16000,
                )
                if not self.is_mounted:
                    return
                previous_offset = self._offsets.get(channel, 0)
                self._offsets[channel] = page["next_offset"]
                if not page["has_more"] or page["next_offset"] <= previous_offset:
                    self._complete.add(channel)
                self._truncated |= not page["trace_complete"] or page["has_more"] and page["next_offset"] <= previous_offset
                self._loaded[channel] = page["content"]
        except (OSError, ValueError, PermissionError) as exc:
            self._load_error = f"Retained output unavailable: {safe_text(exc, 220)}"
        finally:
            if self.is_mounted:
                self.retained.label = "More retained output" if self._offsets else "Retained output"
                self.retained.disabled = len(self._complete) == 5
                self._update_details()
