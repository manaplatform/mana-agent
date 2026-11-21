"""One inline execution stream for a conversation turn."""

from __future__ import annotations

from collections import OrderedDict, deque
from typing import Any

from textual.containers import Vertical
from textual.widgets import Button, Static

from mana_agent.chat.activity import (
    TERMINAL, accepts_activity_update, delivery_key, operation_key,
    project_activity, visible_activity,
)
from mana_agent.chat.operations import merge_operation
from mana_agent.tui.presentation import glyph
from mana_agent.tui.widgets.operation_row import OperationRow, observed_changes


ROOT_END = {
    "turn.finished", "turn.completed", "turn.cancelled", "turn.failed", "turn.timeout",
    "execution.cancelled", "execution.failed", "execution.timeout",
}
WAITING = {"waiting", "waiting_approval", "pending_approval", "waiting_permission", "blocked"}


class ExecutionPanel(Vertical):
    """Bounded, expandable observations. Task control stays in the gateway."""

    DEFAULT_CSS = """
    ExecutionPanel { height: auto; padding: 0 1; }
    ExecutionPanel > Static { height: auto; }
    ExecutionPanel > .changed-files { color: $text-muted; }
    ExecutionPanel > .operation-group { color: $text-muted; }
    ExecutionPanel > Button { height: 1; min-width: 1; border: none; padding: 0 1; }
    """
    PAGE_SIZE = 20
    RECENT_LIMIT = 200

    def __init__(self, *, turn_id: str) -> None:
        super().__init__()
        self.turn_id = turn_id
        self.events_by_key: dict[str, dict[str, Any]] = {}
        self._deliveries: OrderedDict[str, None] = OrderedDict()
        self._legacy_keys: dict[str, str] = {}
        self._runtime_keys: dict[str, str | None] = {}
        self._rows: dict[str, OperationRow] = {}
        self._groups: dict[str, Static] = {}
        self._render_pending = False
        self._mounted = False
        self._evicted = 0
        self._page = 0
        self._archive: dict[str, dict[str, Any]] = {}
        self._paging = False
        self._archive_cursor: int | None = None
        self._archive_exhausted = False
        self._terminal = False
        self._waiting = False
        self._outcome = ""
        self.changed_files: dict[str, None] = {}
        self.context_meter: dict[str, Any] = {}
        self.workflow_mode = ""
        self.summary = Static("", classes="changed-files", markup=False)
        self.earlier = Button("Earlier operations", classes="earlier-operations")
        self.newer = Button("Recent operations", classes="recent-operations")
        for widget in (self.summary, self.earlier, self.newer):
            widget.display = False

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self.events_by_key.values())

    def compose(self):
        yield self.earlier
        yield self.newer
        yield self.summary

    def on_mount(self) -> None:
        self._mounted = True
        self._schedule_render()

    def on_unmount(self) -> None:
        self._mounted = False

    def update_trace_event(self, event: dict[str, Any]) -> None:
        self.update_event(event)

    def update_event(self, event: dict[str, Any], *, projected: bool = False, legacy: bool = False) -> None:
        event = event if projected else project_activity(event)
        if event is None:
            return
        receipt = delivery_key(event)
        if receipt in self._deliveries:
            return
        self._deliveries[receipt] = None
        if len(self._deliveries) > 4096:
            self._deliveries.popitem(last=False)

        meta = event.get("metadata") or {}
        kind = event["type"]
        status = str(event.get("status") or "running")
        mode = event.get("execution_mode")
        if isinstance(mode, str) and mode.lower() in {"auto", "act", "debug", "plan"}:
            self.workflow_mode = mode.upper()
        if kind.startswith("context.") and isinstance(meta.get("context_window"), (int, float)) and meta["context_window"] > 0:
            self.context_meter = {field: meta[field] for field in ("context_window", "used_tokens", "utilization_ratio", "estimated") if field in meta}

        key = operation_key(event)
        correlation = str(event.get("tool_call_id") or event.get("call_id") or meta.get("tool_call_id") or meta.get("call_id") or event.get("operation_id") or "")
        if correlation and legacy:
            if correlation in self._runtime_keys:
                return
            self._legacy_keys[correlation] = key
        elif correlation and (kind.startswith(("tool.", "command.")) or event.get("semantic_kind") == "tool_execution"):
            previous_key = self._runtime_keys.setdefault(correlation, key)
            if previous_key != key:
                self._runtime_keys[correlation] = None
            old_key = self._legacy_keys.pop(correlation, None)
            if old_key is not None and old_key != key and old_key in self.events_by_key:
                # Keep the original widget and chronological position when a legacy record gains provenance.
                self.events_by_key = {key if current == old_key else current: value
                                      for current, value in self.events_by_key.items()}
                if old_key in self._rows:
                    self._rows[key] = self._rows.pop(old_key)
            elif old_key == key:
                self.events_by_key.pop(key, None)

        previous = self.events_by_key.get(key)
        if previous is not None and not accepts_activity_update(previous, event):
            return
        if previous and (event.get("metadata") or {}).get("streams"):
            # ChatHistory already assembled this bounded snapshot.
            event = {**previous, **event, "metadata": {**(previous.get("metadata") or {}), **meta}}
        elif previous:
            event = merge_operation(previous, event)
        self.events_by_key[key] = event

        root = (
            kind in ROOT_END and not event.get("subagent_id")
            and (kind != "turn.completed" or not event.get("backend"))
        ) or (
            kind in {"task.completed", "task.failed", "task.cancelled"}
            and event.get("task_id") and event.get("task_id") == event.get("root_task_id")
            and meta.get("execution_supervisor")
        )
        if root:
            self._waiting = status in WAITING
            if status in TERMINAL:
                self._terminal = True
                self._outcome = status
        elif kind in {"turn.started", "turn.resume_requested"}:
            self._terminal = self._waiting = False
            self._outcome = ""

        for path in observed_changes(event):
            self.changed_files[path] = None

        if len(self.events_by_key) > self.RECENT_LIMIT:
            completed = [entry for entry, record in self.events_by_key.items() if self._status(record) in TERMINAL]
            evicted = set(completed[:-self.RECENT_LIMIT])
            if evicted:
                for entry in evicted:
                    del self.events_by_key[entry]
                self._evicted += len(evicted)
                self._legacy_keys = {cid: entry for cid, entry in self._legacy_keys.items() if entry not in evicted}
                self._runtime_keys = {cid: entry for cid, entry in self._runtime_keys.items() if entry not in evicted}
        self._schedule_render()

    def resolve_approvals(self, request_ids: set[str], *, approved: bool, expired: bool) -> None:
        for key, row in list(self.events_by_key.items()):
            meta = row.get("metadata") or {}
            ids = {str(meta.get(field) or row.get(field) or "") for field in (
                "request_id", "inbox_item_id", "permission_request_id", "approval_request_id", "action_id"
            )}
            if ids.intersection(request_ids) and row.get("status") in WAITING:
                label = "Approval expired" if expired else "Approval granted" if approved else "Approval denied"
                self.events_by_key[key] = {**row, "type": "approval.resolved", "event_type": "approval.resolved",
                                           "title": label, "status": "completed" if approved else "rejected"}
        self._schedule_render()

    def _status(self, record: dict[str, Any]) -> str:
        status = str(record.get("status") or "running")
        if self._terminal and status not in TERMINAL:
            return "unverified"
        return status

    def live_label(self) -> str:
        if self._terminal:
            return ""
        live = ""
        for event in reversed(self.events):
            status = self._status(event)
            kind = event["type"]
            if status in WAITING and ("approval" in kind or status in {"waiting_approval", "pending_approval", "waiting_permission"}):
                return "Waiting for approval"
            if live or status in TERMINAL or kind.startswith(("context.", "cost.", "budget.", "steering.")):
                if kind == "steering.delivered" and not live:
                    live = "Applying steering instruction"
                continue
            command = event.get("command") or (event.get("metadata") or {}).get("command")
            if kind.startswith(("test.", "verification.")):
                live = "Running tests" if kind.startswith("test.") else "Verifying"
            elif kind.startswith(("search.", "discovery.")):
                live = "Searching repository"
            elif kind.startswith("file.read") or event.get("tool_name") in {"read_file", "repo_batch_read"}:
                live = "Reading files"
            elif command:
                live = "Running command"
            else:
                live = str(event.get("title") or "Working")
        return live or ("Waiting for input" if self._waiting else "")

    def _schedule_render(self) -> None:
        if self._mounted and not self._render_pending:
            self._render_pending = True
            self.set_timer(0.05, self._render_operations)

    def _showable(self, event: dict[str, Any]) -> bool:
        kind = event["type"]
        if event.get("tool_name") in {"approval_request", "request_user_approval", "wait_for_approval"}:
            return False
        if kind in ROOT_END or kind.startswith(("context.", "cost.", "steering.")):
            return False
        if self._status(event) in WAITING and "approval" in kind:
            return False
        visibility = getattr(self.app, "activity_visibility", "normal") if self.is_mounted else "normal"
        if visibility != "verbose" and self._status(event) not in {"failed", "error", "rejected"}:
            if kind.startswith(("loop.iteration.", "loop.taskboard.", "routing.", "routing_", "backend.selected",
                                "model.", "turn.started")):
                return False
        return visible_activity(event, visibility)

    @staticmethod
    def _is_exploration(event: dict[str, Any]) -> bool:
        return event["type"].startswith(("search.", "file.read", "discovery.")) or event.get("tool_name") in {
            "repo_search", "grep_search", "find_by_name", "read_file", "repo_batch_read",
        }

    def _render_operations(self) -> None:
        self._render_pending = False
        log = self.parent
        follow = bool(getattr(log, "following", False))
        records = self.events_by_key
        completed = [(key, event) for key, event in records.items()
                     if self._showable(event) and self._status(event) in TERMINAL]
        end = len(completed) - self._page * self.PAGE_SIZE
        shown = dict(completed[max(0, end - self.PAGE_SIZE):max(0, end)]) if end > 0 else self._archive
        shown = {**shown, **{key: event for key, event in records.items()
                            if key in self._rows and not self._rows[key].collapsed}}
        active = {key: event for key, event in records.items()
                  if self._showable(event) and self._status(event) not in TERMINAL}
        shown.update(active)
        ordered = {key: event for key, event in records.items() if key in shown}
        ordered.update({key: event for key, event in shown.items() if key not in ordered})
        for key in list(self._rows):
            if key not in ordered:
                self._rows.pop(key).remove()
        for key, event in ordered.items():
            row_event = {**event, "status": self._status(event)}
            if key not in self._rows:
                self._rows[key] = OperationRow(row_event)
                self.mount(self._rows[key], before=self.summary)
            else:
                self._rows[key].update_operation(row_event)
        keys = list(ordered)
        groups = set()
        for index, key in enumerate(keys[:-1]):
            previous_is_exploration = index > 0 and self._is_exploration(ordered[keys[index - 1]])
            if not previous_is_exploration and self._is_exploration(ordered[key]) and self._is_exploration(ordered[keys[index + 1]]):
                groups.add(key)
        for key in self._groups.keys() - groups:
            self._groups.pop(key).remove()
        children = []
        for key in keys:
            if key in groups:
                if key not in self._groups:
                    self._groups[key] = Static(f"{glyph('•', '*')} Explored", classes="operation-group", markup=False)
                    self.mount(self._groups[key], before=self.summary)
                children.append(self._groups[key])
            children.append(self._rows[key])
        previous = self.newer
        for child in children:
            if child in self.children and previous in self.children and self.children.index(child) != self.children.index(previous) + 1:
                self.move_child(child, after=previous)
            previous = child

        self.earlier.display = end > self.PAGE_SIZE or bool(self._evicted and not self._archive_exhausted)
        self.newer.display = self._page > 0
        files = list(self.changed_files)
        message = f"Changed {len(files)} file{'s' if len(files) != 1 else ''} · {', '.join(files[:3])}" if files else ""
        if len(files) > 3:
            message += f" · +{len(files) - 3} more"
        if self._outcome in {"failed", "error", "cancelled", "interrupted", "timeout", "timed_out"}:
            label = {"cancelled": "Interrupted", "interrupted": "Interrupted", "timeout": "Timed out",
                     "timed_out": "Timed out"}.get(self._outcome, "Execution failed")
            message = f"{label} · {message}" if message else label
        self.summary.update(message)
        self.summary.display = bool(self._terminal and message)
        self.display = bool(ordered or self.summary.display or self.earlier.display or self.newer.display)
        if hasattr(log, "refresh_activity_status"):
            log.refresh_activity_status()
            if follow:
                log.follow_after_refresh()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button not in (self.earlier, self.newer):
            return
        event.stop()
        if event.button is self.newer:
            self._page = 0
            self._archive.clear()
            self._archive_cursor = None
            self._archive_exhausted = False
        elif not self._paging:
            self._page += 1
            completed = sum(self._showable(row) and self._status(row) in TERMINAL for row in self.events)
            if self._page * self.PAGE_SIZE >= completed and self._evicted:
                self._paging = True
                self.earlier.disabled = True
                try:
                    await self._load_older_page()
                finally:
                    self._paging = False
                    if self.is_mounted:
                        self.earlier.disabled = False
        self._schedule_render()

    async def _load_older_page(self) -> None:
        import asyncio

        from mana_agent.services.conversation_service import ConversationService

        anchor = next((row for row in self.events if row.get("conversation_id") and row.get("repository_id") and row.get("sequence")), None)
        if anchor is None:
            self._archive_exhausted = True
            self.app.notify("Older operations are unavailable for this turn.", severity="warning")
            return
        sequence = self._archive_cursor or min((int(row["sequence"]) for row in self.events_by_key.values()
                        if row.get("sequence")), default=int(anchor["sequence"]))
        try:
            service = ConversationService(repository_id=str(anchor["repository_id"]))
            records = await asyncio.to_thread(
                service.list_events, str(anchor["conversation_id"]),
                execution_id=str(anchor.get("execution_id") or self.turn_id),
                before_sequence=sequence, limit=200,
            )
        except (OSError, ValueError, PermissionError) as exc:
            self.app.notify(f"Older operations unavailable: {exc}", severity="warning")
            return
        if not self.is_mounted:
            return
        archive: dict[str, dict[str, Any]] = {}
        for record in sorted(records, key=lambda row: int(row.get("sequence") or 0)):
            projected = project_activity(record)
            if projected is None:
                continue
            key = operation_key(projected)
            if key not in self.events_by_key:
                archive[key] = merge_operation(archive.get(key, {}), projected)
        visible = [(key, row) for key, row in archive.items() if self._showable(row)]
        visible = visible[-self.PAGE_SIZE:]
        self._archive = dict(visible)
        if visible:
            self._archive_cursor = min(int(row.get("sequence") or sequence) for _, row in visible)
        elif records:
            self._archive_cursor = min(int(row.get("sequence") or sequence) for row in records)
        self._archive_exhausted = not records or self._archive_cursor == sequence
        if self._archive_exhausted and not self._archive:
            self.app.notify("No earlier retained operations.", severity="information")
