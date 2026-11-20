"""Textual conversation frontend for Mana-Agent."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import getpass
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any

logger = logging.getLogger(__name__)

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Static

from mana_agent.chat.events import (
    AssistantMessageEvent,
    CodingActivityEvent,
    StreamTokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from mana_agent.chat.history import ChatHistory, get_history
from mana_agent.chat.activity import delivery_key, project_activity, safe_text
from mana_agent.tui.presentation import glyph
from mana_agent.tui.widgets.attachment_bar import AttachmentBar
from mana_agent.tui.widgets.chat_log import ChatLog
from mana_agent.tui.widgets.message_input import MessageInput


class ManaChatApp(App):
    """Main Textual application for the enhanced mana-agent chat TUI."""

    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("q", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear_log", "Clear", show=False),
        Binding("ctrl+r", "simulate_response", "Simulate Agent", show=False),
        Binding("ctrl+o", "attach_file", "Attach File", show=False),
        Binding("ctrl+end", "latest_activity", "Latest activity", show=False),
        Binding("escape", "interrupt_execution", "Interrupt", show=False),
        Binding("ctrl+g", "focus_approval", "Focus approval", show=False),
    ]

    # Reactive status for footer
    status_text: reactive[str] = reactive("Ready")
    token_count: reactive[int] = reactive(0)

    def __init__(
        self,
        history: ChatHistory | None = None,
        *,
        repo_root: str | Path | None = None,
        model: str | None = None,
        initial_prompt: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        chat_service: Any = None,
        coding_agent: Any = None,
        tools_orchestrator: Any = None,
        # Full context for exact parity with console coding-agent path (dir-mode, auto-execute, planning, flows, etc.)
        # These are forwarded from chat_cli.py so generate* calls use identical args.
        dir_mode: bool = False,
        index_dir: str | Path | None = None,
        index_dirs: list[str | Path] | None = None,
        auto_execute_plan: bool = True,
        auto_execute_max_passes: int = 4,
        coding_agent_max_steps: int = 200,
        resolved_k: int = 6,
        agent_timeout_seconds: int = 600,
        # Gateway (when provided) is the authoritative connection from TUI to agents.
        # chat_cli creates it and passes it so that "tui chat need connect with gateway to agent".
        # The individual objects + parity context are still accepted for full backward
        # compatibility with the exact console parity implementation.
        gateway: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Use `is not None` — empty ChatHistory is falsy (__len__==0) and must still be kept.
        self.history = history if history is not None else get_history()
        self.chat_log: ChatLog | None = None
        self.input: MessageInput | None = None
        self.attachment_bar: AttachmentBar | None = None
        self._turn_counter = 0
        self._tool_cid_map: dict[str, str] = {}  # key -> call_id for reliable start/end pairing in bridge

        # Configuration for real agent behavior
        self.repo_root: Path = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
        self.model: str | None = model
        if not self.model or self.model == "default":
            try:
                from mana_agent.config.user_config import load_effective_settings

                eff = load_effective_settings()
                cfg_model = str(
                    eff.get("OPENAI_CHAT_MODEL")
                    or eff.get("MANA_PRIMARY_MODEL")
                    or eff.get("LLM_MODEL")
                    or ""
                ).strip()
                if cfg_model and cfg_model != "default":
                    self.model = cfg_model
            except Exception:
                pass
        self.initial_prompt: str | None = initial_prompt
        self.api_key: str | None = api_key
        self.base_url: str | None = base_url
        # Full multi-agent objects (wired from chat_cli after complete setup)
        self.chat_service = chat_service
        self.coding_agent = coding_agent
        self.tools_orchestrator = tools_orchestrator
        self._llm = None  # lazy

        # Exact-parity context (used to build identical calls to coding_agent.generate* as the console path)
        self.dir_mode = bool(dir_mode)
        self.index_dir: str | None = str(index_dir) if index_dir else None
        self.index_dirs: list[str] | None = [str(p) for p in index_dirs] if index_dirs else None
        self.auto_execute_plan = bool(auto_execute_plan)
        self.auto_execute_max_passes = int(auto_execute_max_passes)
        self.coding_agent_max_steps = int(coding_agent_max_steps)
        self.resolved_k = int(resolved_k)
        self.agent_timeout_seconds = int(agent_timeout_seconds)

        # Gateway connection (optional, preferred path going forward)
        self.gateway = gateway
        self._gateway_session_id: str | None = None
        if self.gateway is not None and hasattr(self.gateway, "create_session"):
            configured_session = getattr(getattr(self.gateway, "config", None), "session_id", None)
            self._gateway_session_id = self.gateway.create_session(
                frontend="tui",
                session_id=configured_session,
            )

        # Minimal per-session state to support full flows (updated from generate results)
        self.active_flow_id: str | None = None
        self._in_planning_collection: bool = False
        self._planning_request: str | None = None
        self._planning_questions: list[str] = []
        self._planning_answers: list[str] = []
        self._turn_in_progress = False
        self.interrupt_requested = False
        self._active_approval = None
        self._approval_drawer = Vertical(id="approval-drawer")
        self._approval_drawer.display = False
        self._footer = Static("", id="chat-status", markup=False)
        self._branch = ""
        self._permission_text = ""
        self._mode_text = ""
        self._computer_permission_requests_shown: set[str] = set()
        self._transactional_modal_queue: list[str] = []
        self._transactional_modal_active = False
        self._active_transactional_id = ""
        self._resolved_inline_approvals: set[str] = set()
        self._drawer_request_ids: set[str] = set()
        self._approval_drawer_generation = 0
        self._saved_modal_previews: dict[str, str] = {}
        self._modal_screen_active: bool = False
        self._active_modal_request_id: str = ""
        self._modal_request_queue: list[Any] = []
        self._unsubscribe_computer_permissions = None
        self._unsubscribe_session_events = None
        self._unsubscribe_api_approval_events = None
        self._delivered_coding_event_ids: set[str] = set()
        self._activity_revisions: set[tuple[str, str, str, str]] = set()
        self._hub_receipt_lock = threading.Lock()
        self._replay_deliveries: list[dict[str, Any]] | None = None
        self._transcript_ready = asyncio.Event()
        self._transcript_ready.set()
        self.activity_visibility = "normal"
        self._execution_to_frontend_turn: dict[str, str] = {}
        self._current_frontend_turn_id: str | None = None
        self._ui_thread_id: int | None = None

    @property
    def _resolved_approvals(self) -> set[str]:
        return self._resolved_inline_approvals

    def compose(self) -> ComposeResult:
        with Vertical(id="body"):
            self.chat_log = ChatLog(history=self.history, id="chat-log")
            yield self.chat_log
            yield self._approval_drawer
            with Vertical(id="input-bar"):
                self.attachment_bar = AttachmentBar(id="attachment-bar")
                yield self.attachment_bar
                with Horizontal(id="composer"):
                    yield Static(glyph("❯", ">"), id="prompt-indicator", markup=False)
                    self.input = MessageInput(id="chat-input")
                    yield self.input
                yield Static("", id="command-menu", markup=False)
                yield self._footer

    @staticmethod
    def _permission_label() -> str:
        from mana_agent.approval.policy import get_approval_policy

        return {
            "full_access": "Full access",
            "auto_approve_safe": "Auto-approve safe",
            "default": "Approval required",
        }[get_approval_policy().mode.value]

    @property
    def has_pending_execution_approval(self) -> bool:
        return self._active_approval is not None

    @property
    def can_interrupt(self) -> bool:
        return bool(self._gateway_session_id and callable(getattr(self.gateway, "cancel", None))
                    and (self._turn_in_progress or self.has_pending_execution_approval))

    def _update_startup(self) -> None:
        from mana_agent import __version__

        if self.chat_log is None:
            return
        sep = glyph(" · ", " | ")
        fields = [self.model, self._permission_label()]
        if self._branch:
            fields.append(self._branch)
        self.chat_log.startup.update(
            f"Mana-Agent {__version__}\n{self.repo_root}\n" + sep.join(field for field in fields if field)
        )
        self.chat_log.startup.display = True

    async def _load_startup_branch(self) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                "git", "symbolic-ref", "--quiet", "--short", "HEAD",
                cwd=self.repo_root, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await process.communicate()
        except OSError:
            return
        if process.returncode == 0:
            self._branch = safe_text(stdout.decode(errors="replace").strip(), 120)
            self._update_startup()

    ACTIVE_CHAT_COMMANDS = frozenset({"cancel", "status", "help", "steer"})
    LOCAL_COMMANDS = {
        "attach": "Attach a file to the next message",
        "detach": "Remove a pending attachment",
        "clear-attachments": "Remove all pending attachments",
        "models": "Manage configured models",
        "steer": "Steer the current execution",
    }

    def _refresh_command_menu(self) -> None:
        if self.input is None:
            return
        definitions = getattr(getattr(self.gateway, "command_registry", None), "definitions", None)
        local = dict(self.LOCAL_COMMANDS)
        if self._turn_in_progress:
            local.pop("models")
        rows = [(name, description, ()) for name, description in local.items()]
        if definitions:
            rows.extend(
                (definition.canonical_name, definition.description, definition.aliases)
                for definition in definitions()
                if "tui" in definition.frontends
                and (not self._turn_in_progress or definition.canonical_name in self.ACTIVE_CHAT_COMMANDS)
            )
        self.input.set_commands(rows)

    def on_mount(self) -> None:
        if not os.environ.get("COLORTERM") and os.environ.get("TERM") in {"linux", "vt100", "ansi", "xterm", "dumb"}:
            self.theme = "textual-ansi"
        self._ui_thread_id = threading.get_ident()
        if self.input:
            self.input.menu = self.query_one("#command-menu", Static)
            self._refresh_command_menu()
            self.input.focus()
        self._unsubscribe_computer_permissions = self.history.subscribe(self._handle_computer_permission_event)
        if self._gateway_session_id:
            self._attach_session_events(self._gateway_session_id, restore_transcript=not bool(self.history))
        self._permission_text = self._permission_label()
        self._update_startup()
        from mana_agent.config.user_config import get_setting

        self.activity_visibility = str(get_setting("MANA_ACTIVITY_VISIBILITY", "normal"))
        if self.is_running:
            self.run_worker(self._load_startup_branch(), group="startup")
        self._refresh_runtime_labels()
        self.call_after_refresh(self._queue_outstanding_transactional_approvals)
        if self.initial_prompt:
            self.run_worker(self._send_initial_prompt(), group="chat-turn", exclusive=True)

    def _handle_computer_permission_event(self, event: Any) -> None:
        if not isinstance(event, CodingActivityEvent):
            return
        activity = event.activity
        event_type = activity.get("event_type")
        if event_type in {
            "action.approval.expired",
            "action.approval.invalidated",
            "inbox.item.expired",
        }:
            metadata = activity.get("metadata") or {}
            request_ids = {str(metadata.get(key) or "") for key in ("permission_request_id", "approval_request_id", "inbox_item_id", "action_id")} - {""}
            if request_ids:
                from mana_agent.tui.computer_permission import ApprovalDecided
                self.post_message(ApprovalDecided(request_ids, approved=False, expired=True))
                self.update_status("Approval request expired or invalidated")
            return
        if event_type not in {
            "computer.waiting_permission",
            "remote_execution.waiting_permission",
            "server.waiting_approval",
            "api.waiting_approval",
            "action.approval.required",
        }:
            return
        metadata = activity.get("metadata") or {}
        exp = metadata.get("expires_at")
        expires_at_dt = None
        if isinstance(exp, datetime):
            expires_at_dt = exp
        elif isinstance(exp, str) and exp:
            try:
                expires_at_dt = datetime.fromisoformat(exp)
            except Exception:
                pass
        if expires_at_dt and expires_at_dt <= datetime.now(timezone.utc):
            return

        request_id = str(metadata.get("permission_request_id") or "")
        if bool(metadata.get("transactional_action_approval")) or activity.get("event_type") == "action.approval.required":
            inbox_item_id = str(metadata.get("inbox_item_id") or request_id)
            if inbox_item_id:
                self._enqueue_transactional_modal(inbox_item_id, metadata=metadata, title=str(activity.get("title") or ""))
            return
        if not request_id or request_id in self._computer_permission_requests_shown:
            return
        self._computer_permission_requests_shown.add(request_id)
        from mana_agent.tui.computer_permission import ComputerPermissionRequested

        # Chat history listeners run synchronously on the gateway/tool thread.
        # Posting a Textual message is thread-safe and non-blocking; call_from_thread
        # would wait for the UI callback and can deadlock the active tool turn.
        self.post_message(ComputerPermissionRequested(
            request_id=request_id,
            scope=str(metadata.get("permission_scope") or ""),
            preview=(
                json.dumps(
                    metadata.get("approval_display") or metadata.get("preview"),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
                if metadata.get("approval_display") or isinstance(metadata.get("preview"), dict)
                else str(metadata.get("preview") or activity.get("title") or "")
            ),
            remote=bool(metadata.get("remote_permission")),
            server=bool(metadata.get("server_approval") or activity.get("event_type") == "server.waiting_approval"),
            api=bool(metadata.get("api_approval") or activity.get("event_type") == "api.waiting_approval"),
            transactional=bool(metadata.get("transactional_action_approval")),
            expires_at=expires_at_dt,
        ))

    def _subscribe_session_events(self, session_id: str = "") -> None:
        """Subscribe the active TUI session to general execution events from ExecutionEventHub."""
        if self._unsubscribe_session_events is not None:
            try:
                self._unsubscribe_session_events()
            except Exception:
                pass
            self._unsubscribe_session_events = None
        self._unsubscribe_api_approval_events = None

        sid = str(session_id or self._gateway_session_id or "").strip()
        if not sid:
            return

        try:
            from mana_agent.services.execution_event_hub import get_execution_event_hub

            from mana_agent.workspaces.paths import repository_id_for_path
            hub = get_execution_event_hub()
            generation = self._approval_drawer_generation

            def receive(event: dict[str, Any]) -> None:
                with self._hub_receipt_lock:
                    if generation != self._approval_drawer_generation:
                        return
                    if self._replay_deliveries is not None:
                        self._replay_deliveries.append(event)
                        return
                self._handle_hub_session_event(event)

            unsub = hub.subscribe(
                sid,
                receive,
                repository_id=repository_id_for_path(self.repo_root),
                buffered=True,
            )
            self._unsubscribe_session_events = unsub
            self._unsubscribe_api_approval_events = unsub
        except Exception:
            logger.debug("Failed to subscribe to ExecutionEventHub for session %s", session_id, exc_info=True)

    def _handle_hub_session_event(self, event: dict[str, Any]) -> None:
        """Handle execution events from ExecutionEventHub as durable fallback and source of truth."""
        sid = str(event.get("conversation_id") or "")
        if self._gateway_session_id:
            if sid and sid != str(self._gateway_session_id):
                return
        elif sid:
            return

        event_type = str(event.get("type") or event.get("event_type") or "").strip()
        event_id = str(event.get("event_id") or event.get("id") or "").strip()
        if event_type == "mode.input_ready":
            prompt = {"task_id": event.get("task_id"), "inbox_id": (event.get("metadata") or {}).get("inbox_id")}
            if self.is_running and threading.get_ident() != getattr(self, "_ui_thread_id", None):
                self.call_from_thread(self._show_mode_prompt, prompt)
            elif self.is_running:
                self._show_mode_prompt(prompt)
            return
        if event_type in {"activity.degraded", "activity.backpressure"} and not event.get("execution_id"):
            title = safe_text(event.get("title") or "Activity stream degraded", 120)
            if self.is_running and threading.get_ident() != getattr(self, "_ui_thread_id", None):
                self.call_from_thread(self.update_status, title)
            else:
                self.update_status(title)
            return

        # Handle API, server, and transactional action approval modal trigger
        if event_type in {"api.waiting_approval", "server.waiting_approval", "action.approval.required"}:
            if event_id:
                if event_id in self._delivered_coding_event_ids:
                    return
                self._delivered_coding_event_ids.add(event_id)
            execution_id = str(event.get("execution_id") or "")
            target_turn_id = (
                self._execution_to_frontend_turn.get(execution_id)
                or self._current_frontend_turn_id
                or execution_id
            )
            metadata = dict(event.get("metadata") or {})
            exp = metadata.get("expires_at")
            expires_at_dt = None
            if isinstance(exp, datetime):
                expires_at_dt = exp
            elif isinstance(exp, str) and exp:
                try:
                    expires_at_dt = datetime.fromisoformat(exp)
                except Exception:
                    pass
            if expires_at_dt and expires_at_dt <= datetime.now(timezone.utc):
                return

            backend = event.get("backend") or metadata.get("backend")
            if backend:
                metadata["backend"] = backend
            act_title = str(
                event.get("title")
                or (
                    "Server action approval required"
                    if event_type == "server.waiting_approval"
                    else (
                        "Transactional action approval required"
                        if event_type == "action.approval.required"
                        else "API request approval required"
                    )
                )
            )
            act = {
                "event_id": event_id,
                "event_type": event_type,
                "title": act_title,
                "status": "waiting_approval",
                "backend": backend or "",
                "metadata": metadata,
            }
            self._safe_post_activity(act, target_turn_id)

            # Trigger approval modal directly in real time
            from mana_agent.approval.policy import get_approval_policy

            if get_approval_policy().is_full_access():
                return

            req_id = str(metadata.get("permission_request_id") or metadata.get("inbox_item_id") or event_id or "")
            if event_type == "action.approval.required" or bool(metadata.get("transactional_action_approval")):
                inbox_item_id = str(metadata.get("inbox_item_id") or req_id)
                if inbox_item_id:
                    self._enqueue_transactional_modal(inbox_item_id, metadata=metadata, title=act_title)
            elif req_id and req_id not in self._computer_permission_requests_shown:
                self._computer_permission_requests_shown.add(req_id)
                from mana_agent.tui.computer_permission import ComputerPermissionRequested
                preview_val = metadata.get("approval_display") or metadata.get("preview") or act_title
                preview_str = (
                    json.dumps(preview_val, indent=2, ensure_ascii=False, default=str)
                    if isinstance(preview_val, (dict, list))
                    else str(preview_val)
                )
                self.post_message(ComputerPermissionRequested(
                    request_id=req_id,
                    scope=str(metadata.get("permission_scope") or ""),
                    preview=preview_str,
                    server=bool(event_type == "server.waiting_approval" or metadata.get("server_approval")),
                    api=bool(event_type == "api.waiting_approval" or metadata.get("api_approval")),
                    expires_at=expires_at_dt,
                ))
            return

        if event_type in {
            "server.approval_decided",
            "api.approval_decided",
            "computer.permission_decided",
            "action.approval.granted",
            "action.approval.denied",
            "action.approval.expired",
            "action.approval.invalidated",
            "inbox.item.expired",
        }:
            metadata = dict(event.get("metadata") or {})
            decision = str(metadata.get("decision") or "")
            is_expired = (
                event_type in {"action.approval.expired", "action.approval.invalidated", "inbox.item.expired"}
                or decision == "expired"
            )
            from mana_agent.tui.computer_permission import ApprovalDecided
            request_ids = {str(metadata.get(key) or "") for key in ("permission_request_id", "approval_request_id", "inbox_item_id", "action_id")} - {""}
            if request_ids:
                self.post_message(
                    ApprovalDecided(
                        request_ids,
                        not is_expired and (
                            event_type == "action.approval.granted"
                            or decision in {"approve", "allow_once", "allow_session", "always"}
                        ),
                        expired=is_expired,
                    )
                )
            if is_expired:
                self.update_status("Approval request expired or invalidated")
            elif decision == "deny" or event_type == "action.approval.denied":
                self.update_status(
                    "Server action denied"
                    if event_type == "server.approval_decided"
                    else (
                        "Computer action denied"
                        if event_type == "computer.permission_decided"
                        else (
                            "Action denied"
                            if event_type == "action.approval.denied"
                            else "API request denied"
                        )
                    )
                )
            return

        if event_type == "turn.resume_requested":
            metadata = dict(event.get("metadata") or {})
            is_server = bool(metadata.get("server_approval"))
            self.update_status(
                "Resuming server execution…" if is_server else "Resuming execution…"
            )
            return

        key = delivery_key(event)
        meta = event.get("metadata") or event.get("payload") or {}
        revision = (event_id, event_type, str(event.get("status") or ""), str(meta.get("stream_offset") or ""))
        safe = project_activity(event)
        if safe is None:
            return
        with self._hub_receipt_lock:
            if key in self._delivered_coding_event_ids or (event_id and revision in self._activity_revisions):
                return
            self._delivered_coding_event_ids.add(key)
            if event_id:
                self._activity_revisions.add(revision)
            if len(self._delivered_coding_event_ids) > 4000:
                self._delivered_coding_event_ids.pop()
            if len(self._activity_revisions) > 4000:
                self._activity_revisions.pop()
        execution_id = str(event.get("execution_id") or event.get("turn_id") or "")
        target_turn_id = self._execution_to_frontend_turn.get(execution_id) or execution_id
        if not target_turn_id:
            # An unbound event cannot attach itself to whichever turn is current.
            return
        self._safe_post_activity(safe, target_turn_id)

    def _handle_api_approval_event(self, event: dict[str, Any]) -> None:
        """Backward-compatible hook delegating to _handle_hub_session_event."""
        self._handle_hub_session_event(event)

    def _safe_post_activity(self, activity: dict[str, Any], turn_id: str) -> None:
        projected = project_activity(activity)
        if projected is not None:
            projected.setdefault("conversation_id", self._gateway_session_id)
            projected.setdefault("execution_id", turn_id)
            if projected["type"].startswith("steering."):
                metadata = projected.get("metadata") or {}
                request_id = str(metadata.get("request_id") or projected.get("request_id") or "")
                if request_id and metadata.get("text") and not any(
                    isinstance(row, UserMessageEvent) and row.is_steering and row.request_id == request_id
                    for row in self.history.get_events()
                ):
                    self.history.add(UserMessageEvent(content=metadata["text"], turn_id=turn_id,
                        event_id=f"steering-{request_id}", is_steering=True, request_id=request_id,
                        steering_status=projected["type"].split(".")[-1], steering_recorded=bool(projected.get("sequence"))))
            # ChatLog marshals history notifications through nonblocking messages.
            self.history.add(CodingActivityEvent(activity=projected, turn_id=turn_id))

    def _handle_session_replacement(self, new_session_id: str, *, clear_view: bool = True) -> str:
        """Atomically detach previous chat/session, reset state, and attach fresh session."""
        if self._unsubscribe_session_events is not None:
            try:
                self._unsubscribe_session_events()
            except Exception:
                pass
            self._unsubscribe_session_events = None
        self._unsubscribe_api_approval_events = None

        self._execution_to_frontend_turn.clear()
        with self._hub_receipt_lock:
            self._delivered_coding_event_ids.clear()
            self._activity_revisions.clear()
            self._approval_drawer_generation += 1
        self._current_frontend_turn_id = None
        self._turn_in_progress = False
        self._computer_permission_requests_shown.clear()
        self._transactional_modal_queue.clear()
        self._transactional_modal_active = False
        self._active_transactional_id = ""
        self._modal_request_queue.clear()
        self._active_modal_request_id = ""
        self._resolved_inline_approvals.clear()
        self._drawer_request_ids.clear()
        self._saved_modal_previews.clear()
        self.interrupt_requested = False
        if self._active_approval is not None:
            if self._active_approval.is_mounted:
                self._active_approval.remove()
            self._active_approval = None
        self._approval_drawer.display = False
        self.active_flow_id = None
        self._tool_cid_map.clear()

        if clear_view:
            self._clear_conversation_view()

        self._gateway_session_id = new_session_id
        self._attach_session_events(new_session_id)
        self._refresh_runtime_labels()
        return new_session_id

    def _attach_session_events(self, session_id: str, *, restore_transcript: bool = False) -> None:
        replay = bool(self.is_running and session_id)
        with self._hub_receipt_lock:
            self._replay_deliveries = [] if replay else None
        self._subscribe_session_events(session_id)
        if replay:
            generation = self._approval_drawer_generation
            saved_history = getattr(getattr(self.gateway, "session_service", None), "history", None)
            load_messages = getattr(saved_history, "list", None) if restore_transcript else None
            if callable(load_messages):
                self._transcript_ready.clear()

            async def replay_and_release() -> None:
                try:
                    if callable(load_messages):
                        try:
                            messages = await asyncio.to_thread(load_messages, session_id, limit=5000)
                            if generation == self._approval_drawer_generation and messages:
                                self._replace_timeline([message.to_dict() for message in messages])
                        except (OSError, ValueError) as exc:
                            self.notify(f"Conversation history unavailable: {safe_text(exc, 140)}", severity="warning")
                        finally:
                            self._transcript_ready.set()
                    await self._replay_durable_session_events(session_id)
                finally:
                    if generation == self._approval_drawer_generation:
                        while True:
                            with self._hub_receipt_lock:
                                deliveries = self._replay_deliveries
                                self._replay_deliveries = [] if deliveries else None
                            if not deliveries:
                                break
                            for event in deliveries:
                                self._handle_hub_session_event(event)
                            await asyncio.sleep(0)

            self.run_worker(replay_and_release, group="session-replay", exclusive=True)

    async def _replay_durable_session_events(self, session_id: str) -> None:
        """Replay into the live projection through a fixed high-water mark."""
        from mana_agent.services.execution_event_hub import get_execution_event_hub
        from mana_agent.workspaces.paths import repository_id_for_path

        generation = self._approval_drawer_generation
        hub = get_execution_event_hub()
        repository = repository_id_for_path(self.repo_root)
        scope = dict(conversation_id=session_id, repository_id=repository)
        try:
            high = await asyncio.to_thread(hub.history, **scope, limit=1)
            high_water = int(high[-1].get("sequence") or 0) if high else 0
            cursor = 0
            while cursor < high_water and generation == self._approval_drawer_generation:
                page = await asyncio.to_thread(hub.history, **scope, after_sequence=cursor, forward=True, limit=500)
                if not page or generation != self._approval_drawer_generation:
                    break
                for event in page:
                    sequence = int(event.get("sequence") or 0)
                    if sequence > high_water:
                        break
                    cursor = sequence
                    self._handle_hub_session_event(event)
                if int(page[-1].get("sequence") or 0) >= high_water:
                    break
                await asyncio.sleep(0)
        except (OSError, ValueError) as exc:
            logger.debug("Failed to replay durable events for session %s", session_id, exc_info=True)
            self.notify(f"Activity history unavailable: {safe_text(exc, 140)}", severity="warning")

    def _queue_outstanding_transactional_approvals(self) -> None:
        if self.gateway is None or not getattr(self.gateway, "human_inbox_service", None):
            return
        try:
            from mana_agent.human_inbox.models import InboxQuery, InboxStatus

            try:
                self.gateway.human_inbox_service.expire_due()
            except Exception:
                pass

            rows = self.gateway.human_inbox_service.list(
                InboxQuery(statuses={InboxStatus.PENDING, InboxStatus.DELIVERED}),
                actor_id=getpass.getuser(),
            )
        except Exception:
            return
        now = datetime.now(timezone.utc)
        for item in rows:
            if item.request_type.value == "approval" and item.action_intent_id:
                if item.status.value not in {"pending", "delivered"}:
                    continue
                if item.expires_at and item.expires_at <= now:
                    continue
                self._enqueue_transactional_modal(item.inbox_item_id)

    def _enqueue_transactional_modal(
        self,
        inbox_item_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        title: str = "",
    ) -> None:
        from mana_agent.approval.policy import get_approval_policy

        if get_approval_policy().is_full_access():
            return
        action_id = str((metadata or {}).get("action_id") or "")
        if inbox_item_id in self._computer_permission_requests_shown:
            return
        if action_id and action_id in self._computer_permission_requests_shown:
            return
        self._computer_permission_requests_shown.add(inbox_item_id)
        if action_id:
            self._computer_permission_requests_shown.add(action_id)
        item = None
        if self.gateway is not None:
            try:
                item = self.gateway.human_inbox_service.get(
                    inbox_item_id,
                    actor_id=getpass.getuser(),
                )
            except Exception:
                try:
                    item = self.gateway.human_inbox_service.repository.get(inbox_item_id)
                except Exception:
                    pass
                if item is None:
                    try:
                        matches = self.gateway.human_inbox_service.repository.find_for_action(inbox_item_id)
                        if matches:
                            item = matches[0]
                    except Exception:
                        pass
                if item is None and action_id:
                    try:
                        matches = self.gateway.human_inbox_service.repository.find_for_action(action_id)
                        if matches:
                            item = matches[0]
                    except Exception:
                        pass

        if item is None:
            try:
                from mana_agent.human_inbox import default_human_inbox_service
                service = default_human_inbox_service()
                try:
                    item = service.get(inbox_item_id, actor_id=getpass.getuser())
                except Exception:
                    try:
                        item = service.repository.get(inbox_item_id)
                    except Exception:
                        matches = service.repository.find_for_action(inbox_item_id)
                        if matches:
                            item = matches[0]
                if item is None and action_id:
                    try:
                        matches = service.repository.find_for_action(action_id)
                        if matches:
                            item = matches[0]
                    except Exception:
                        pass
            except Exception:
                pass

        now = datetime.now(timezone.utc)
        if item is not None and (
            item.request_type.value != "approval"
            or item.status.value not in {"pending", "delivered"}
            or (item.expires_at and item.expires_at <= now)
        ):
            return

        preview_data = None
        if item is not None and hasattr(item, "card"):
            preview_data = item.card()
        elif metadata:
            preview_data = metadata.get("approval_display") or metadata.get("preview") or {
                "command": metadata.get("command"),
                "reason": metadata.get("reason"),
                "title": title or metadata.get("title") or "User approval required",
            }
        elif title:
            preview_data = {"title": title}

        preview_str = (
            json.dumps(preview_data, indent=2, ensure_ascii=False, default=str)
            if isinstance(preview_data, (dict, list))
            else str(preview_data or f"Approval required for {inbox_item_id}")
        )
        self._saved_modal_previews[inbox_item_id] = preview_str
        target_id = item.inbox_item_id if item is not None else inbox_item_id
        if target_id != inbox_item_id:
            self._saved_modal_previews[target_id] = preview_str
            self._computer_permission_requests_shown.add(target_id)

        self._computer_permission_requests_shown.add(inbox_item_id)
        if action_id:
            self._computer_permission_requests_shown.add(action_id)
        if item is not None:
            self._computer_permission_requests_shown.add(item.inbox_item_id)
            if item.action_intent_id:
                self._computer_permission_requests_shown.add(item.action_intent_id)

        if target_id not in self._transactional_modal_queue:
            self._transactional_modal_queue.append(target_id)
        self._show_next_transactional_modal()
        self._sync_approval_drawer()

    def _show_next_transactional_modal(self) -> None:
        if self._transactional_modal_active or not self._transactional_modal_queue:
            return

        inbox_item_id = ""
        item = None
        while self._transactional_modal_queue:
            candidate_id = self._transactional_modal_queue.pop(0)
            candidate_item = None
            if self.gateway is not None and getattr(self.gateway, "human_inbox_service", None):
                try:
                    candidate_item = self.gateway.human_inbox_service.get(candidate_id, actor_id=getpass.getuser())
                except Exception:
                    try:
                        candidate_item = self.gateway.human_inbox_service.repository.get(candidate_id)
                    except Exception:
                        pass
                    if candidate_item is None:
                        try:
                            matches = self.gateway.human_inbox_service.repository.find_for_action(candidate_id)
                            if matches:
                                candidate_item = matches[0]
                        except Exception:
                            pass
            if candidate_item is None:
                try:
                    from mana_agent.human_inbox import default_human_inbox_service
                    service = default_human_inbox_service()
                    try:
                        candidate_item = service.repository.get(candidate_id)
                    except Exception:
                        matches = service.repository.find_for_action(candidate_id)
                        if matches:
                            candidate_item = matches[0]
                except Exception:
                    pass

            now = datetime.now(timezone.utc)
            if candidate_item is not None and (
                candidate_item.status.value not in {"pending", "delivered"}
                or (candidate_item.expires_at and candidate_item.expires_at <= now)
            ):
                # Item is already resolved or expired, skip redundant modal
                continue

            if candidate_item is not None and candidate_item.action_intent_id:
                try:
                    from mana_agent.transactional_actions.runtime import default_action_gateway
                    act_gw = default_action_gateway(self.repo_root)
                    act = act_gw.store.get_action(candidate_item.action_intent_id)
                    from mana_agent.transactional_actions.models import utc_now
                    u_now = utc_now()
                    if act is not None and (
                        act.state.value not in {"awaiting_approval"}
                        or (act.expires_at and act.expires_at <= u_now)
                        or (act.policy_decision and act.policy_decision.expires_at and act.policy_decision.expires_at <= u_now)
                    ):
                        # Action is already approved/executing/committed or expired, skip redundant modal
                        continue
                except Exception:
                    pass
                if self.gateway is not None and hasattr(self.gateway, "_transactional_runtime"):
                    try:
                        gw_act = self.gateway._transactional_runtime.store.get_action(candidate_item.action_intent_id)
                        from mana_agent.transactional_actions.models import utc_now
                        u_now = utc_now()
                        if gw_act is not None and (
                            gw_act.state.value not in {"awaiting_approval"}
                            or (gw_act.expires_at and gw_act.expires_at <= u_now)
                            or (gw_act.policy_decision and gw_act.policy_decision.expires_at and gw_act.policy_decision.expires_at <= u_now)
                        ):
                            continue
                    except Exception:
                        pass

            if candidate_item is None and candidate_id not in self._saved_modal_previews:
                continue

            inbox_item_id = candidate_id
            item = candidate_item
            break

        if not inbox_item_id:
            return

        preview_str = self._saved_modal_previews.get(inbox_item_id)
        if item is not None and hasattr(item, "card"):
            preview_str = json.dumps(item.card(), indent=2, ensure_ascii=False, default=str)
        if not preview_str:
            preview_str = f"Approval required for: {inbox_item_id}"

        pending_count = len(self._transactional_modal_queue) + 1
        self._transactional_modal_active = True
        self._active_transactional_id = inbox_item_id
        from mana_agent.tui.computer_permission import ComputerPermissionRequested

        expires_at_dt = getattr(item, "expires_at", None) if item is not None else None
        self.post_message(ComputerPermissionRequested(
            request_id=inbox_item_id,
            scope="transactional_action.once",
            preview=preview_str,
            transactional=True,
            pending_count=pending_count,
            expires_at=expires_at_dt,
        ))

    def on_computer_permission_requested(self, event: Any) -> None:
        self._show_inline_approval(event)

    def _show_inline_approval(self, event: Any) -> None:
        """Validate a request and show it in the shared inline approval area."""

        if event.request_id in self._resolved_inline_approvals or event.request_id == self._active_modal_request_id:
            return

        now = datetime.now(timezone.utc)
        if getattr(event, "expires_at", None) and event.expires_at <= now:
            self._discard_approval(event.request_id, expired=True)
            return

        item = None
        try:
            from mana_agent.human_inbox import default_human_inbox_service
            service = (getattr(self.gateway, "human_inbox_service", None) if self.gateway else None) or default_human_inbox_service()
            try:
                item = service.repository.get(event.request_id)
            except Exception:
                matches = service.repository.find_for_action(event.request_id)
                if matches:
                    item = matches[0]
        except Exception:
            pass

        if item is not None:
            if item.status.value not in {"pending", "delivered"}:
                self._discard_approval(event.request_id, expired=item.status.value == "expired")
                return
            if item.expires_at and item.expires_at <= now:
                self._discard_approval(event.request_id, expired=True)
                return
            if item.action_intent_id:
                try:
                    from mana_agent.transactional_actions.runtime import default_action_gateway
                    act_gw = default_action_gateway(self.repo_root)
                    act = act_gw.store.get_action(item.action_intent_id)
                    from mana_agent.transactional_actions.models import utc_now
                    u_now = utc_now()
                    if act is not None and (
                        act.state.value not in {"awaiting_approval"}
                        or (act.expires_at and act.expires_at <= u_now)
                        or (act.policy_decision and act.policy_decision.expires_at and act.policy_decision.expires_at <= u_now)
                    ):
                        self._discard_approval(event.request_id, expired=bool(act.expires_at and act.expires_at <= u_now))
                        return
                except Exception:
                    pass

        if self._active_approval is not None:
            if event.request_id not in self._drawer_request_ids:
                self._modal_request_queue.append(event)
                self._drawer_request_ids.add(event.request_id)
                self._sync_approval_drawer()
            return

        from mana_agent.tui.computer_permission import InlineApproval

        self._active_modal_request_id = event.request_id
        self._drawer_request_ids.add(event.request_id)
        self._active_approval = InlineApproval(event, self._apply_computer_permission)
        if self._approval_drawer.is_mounted:
            self._approval_drawer.mount(self._active_approval)
        self._sync_approval_drawer()

    def _discard_approval(self, request_id: str, *, expired: bool) -> None:
        from mana_agent.tui.computer_permission import InlineApprovalDecided

        self.on_inline_approval_decided(InlineApprovalDecided({request_id}, False, expired=expired))

    def _show_next_approval(self) -> None:
        if self._active_approval is not None:
            return
        while self._modal_request_queue and self._active_approval is None:
            event = self._modal_request_queue.pop(0)
            self._drawer_request_ids.discard(event.request_id)
            self._show_inline_approval(event)

    def _sync_approval_drawer(self) -> None:
        controls = self._active_approval
        self._approval_drawer.display = controls is not None
        if controls is not None:
            ids = self._drawer_request_ids | set(self._transactional_modal_queue)
            controls.set_pending_count(len(ids))
        if self.chat_log is not None:
            self.chat_log.refresh_activity_status()

    def action_focus_approval(self) -> None:
        if self._active_approval is not None:
            self._active_approval.focus_choices()

    def on_inline_approval_decided(self, event: Any) -> None:
        self._resolved_inline_approvals.update(event.request_ids)
        self._drawer_request_ids.difference_update(event.request_ids)
        self._modal_request_queue = [req for req in self._modal_request_queue if req.request_id not in event.request_ids]
        self._transactional_modal_queue[:] = [item for item in self._transactional_modal_queue if item not in event.request_ids]
        if self._active_transactional_id in event.request_ids:
            self._transactional_modal_active = False
            self._active_transactional_id = ""
        if self._active_modal_request_id in event.request_ids and self._active_approval is not None:
            restore_focus = self._active_approval.has_focus_within
            if self._active_approval.is_mounted:
                self._active_approval.remove()
            self._active_approval = None
            self._active_modal_request_id = ""
            if restore_focus and self.input:
                self.input.focus()
        if self.chat_log is not None:
            for panel in self.chat_log._execution_panels.values():
                panel.resolve_approvals(event.request_ids, approved=event.approved, expired=event.expired)
        self._sync_approval_drawer()
        self.call_after_refresh(self._show_next_approval)
        self.call_after_refresh(self._show_next_transactional_modal)

    on_approval_decided = on_inline_approval_decided

    def _apply_computer_permission(self, choice: Any, *, controls: Any = None) -> None:
        if choice is None:
            return
        controls = controls or self._active_approval
        batch_ids = list(self._transactional_modal_queue) if getattr(choice, "approve_all", False) else []
        generation = self._approval_drawer_generation

        async def complete_and_advance() -> None:
            success = await self._complete_computer_permission(choice, batch_ids=batch_ids)
            if generation != self._approval_drawer_generation:
                return
            if success and choice.request_id not in self._resolved_inline_approvals:
                from mana_agent.tui.computer_permission import InlineApprovalDecided

                # The resolver removes additional requests only after the matching decision commits.
                resolved = {choice.request_id} | (set(batch_ids) - set(self._transactional_modal_queue))
                self.on_inline_approval_decided(InlineApprovalDecided(resolved, choice.decision is not None))
            elif not success and controls is not None and controls is self._active_approval:
                controls.enable_choices()

        self.run_worker(complete_and_advance(), group="execution-approval", name=f"approval-{choice.request_id}")

    async def _complete_computer_permission(self, choice: Any, *, batch_ids: list[str] | None = None, is_batch: bool = False) -> bool:
        from mana_agent.integrations.computer_control.cancellation import (
            decide_computer_permission,
            deny_computer_permission,
        )

        generation = self._approval_drawer_generation
        try:
            if choice.transactional:
                is_approve = choice.decision is not None
                handled = False
                msg = ""
                # Snapshot only requests already queued when the user decided.
                if batch_ids is None:
                    batch_ids = list(self._transactional_modal_queue) if getattr(choice, "approve_all", False) and is_approve else []
                from mana_agent.human_inbox import default_human_inbox_service
                service = (getattr(self.gateway, "human_inbox_service", None) if self.gateway else None) or default_human_inbox_service()
                try:
                    reference = service.repository.get(choice.request_id)
                except Exception:
                    try:
                        matches = service.repository.find_for_action(choice.request_id)
                        reference = matches[0] if matches else None
                    except Exception:
                        reference = None
                if reference is None:
                    fallback = default_human_inbox_service()
                    try:
                        reference = fallback.repository.get(choice.request_id)
                        service = fallback
                    except Exception:
                        pass
                now = datetime.now(timezone.utc)
                if reference is not None and (
                    reference.status.value not in {"pending", "delivered"}
                    or (reference.expires_at and reference.expires_at <= now)
                ):
                    if is_batch:
                        return False
                    is_expired = reference.status.value == "expired" or bool(reference.expires_at and reference.expires_at <= now)
                    logger.info("Inbox item %s is already terminal/expired (%s)", reference.inbox_item_id, reference.status.value)
                    self.notify("Approval request expired." if is_expired else "Approval request already decided.", severity="information")
                    from mana_agent.tui.computer_permission import ApprovalDecided
                    self.on_approval_decided(ApprovalDecided({choice.request_id}, approved=False, expired=is_expired))
                    return True
                if self.gateway is not None:
                    command = (
                        self.gateway.transactional_action_approval_command
                        if is_approve
                        else self.gateway.deny_transactional_action_command
                    )
                    try:
                        extra_args = {}
                        if getattr(choice, "expected_version", None) is not None:
                            extra_args["expected_version"] = choice.expected_version
                        result = await asyncio.to_thread(
                            command,
                            choice.request_id,
                            session_id=self._gateway_session_id or "",
                            **extra_args,
                        )
                        msg = str(result.get("message") or "")
                        handled = True
                    except Exception as exc:
                        exc_str = str(exc).lower()
                        if is_approve and (
                            "already terminal: approved" in exc_str
                            or "already approved" in exc_str
                            or "already processed" in exc_str
                            or "only an action awaiting approval may receive a grant" in exc_str
                            or "is already in approved" in exc_str
                            or "action is already" in exc_str
                            or "is no longer approvable" in exc_str
                        ):
                            msg = "Transactional action approved."
                            handled = True
                        elif not is_approve and (
                            "already terminal: denied" in exc_str
                            or "already denied" in exc_str
                        ):
                            msg = "Transactional action denied."
                            handled = True
                        elif (
                            "already terminal" in exc_str
                            or "is terminal" in exc_str
                            or "expired" in exc_str
                            or "no longer approvable" in exc_str
                            or "no longer accepts" in exc_str
                        ):
                            if is_batch:
                                return False
                            is_expired = "expired" in exc_str
                            logger.info("Gateway transactional approval command error (terminal/expired): %s", exc)
                            self.notify("Approval request expired." if is_expired else "Approval request already decided.", severity="information")
                            from mana_agent.tui.computer_permission import ApprovalDecided
                            self.on_approval_decided(ApprovalDecided({choice.request_id}, approved=False, expired=is_expired))
                            return True
                        else:
                            logger.warning("Gateway transactional approval command error: %s", exc)

                if not handled:
                    try:
                        from mana_agent.human_inbox import default_human_inbox_service
                        from mana_agent.human_inbox.models import (
                            InboxStatus,
                            ResponseOperation,
                            ResponseSubmission,
                            UNRESOLVED_STATUSES,
                        )
                        from mana_agent.transactional_actions.runtime import default_action_gateway

                        service = (
                            self.gateway.human_inbox_service
                            if (self.gateway and getattr(self.gateway, "human_inbox_service", None))
                            else default_human_inbox_service()
                        )
                        item = None
                        try:
                            item = service.repository.get(choice.request_id)
                        except Exception:
                            try:
                                matches = service.repository.find_for_action(choice.request_id)
                                if matches:
                                    item = matches[0]
                            except Exception:
                                pass
                        if item is None and service is not default_human_inbox_service():
                            try:
                                fallback_svc = default_human_inbox_service()
                                item = fallback_svc.repository.get(choice.request_id)
                                if item is not None:
                                    service = fallback_svc
                            except Exception:
                                try:
                                    fallback_svc = default_human_inbox_service()
                                    matches = fallback_svc.repository.find_for_action(choice.request_id)
                                    if matches:
                                        item = matches[0]
                                        service = fallback_svc
                                except Exception:
                                    pass

                        if item is None:
                            raise LookupError("Approval request not found")
                        expected_version = getattr(choice, "expected_version", None)
                        if expected_version is not None and item.version != expected_version:
                            raise ValueError("approval request changed after batch judgment")
                        if expected_version is not None and item.action_intent_id:
                            from mana_agent.transactional_actions.models import ActionState, PolicyOutcome, utc_now

                            current = default_action_gateway(self.repo_root).store.get_action(item.action_intent_id)
                            if (current is None
                                    or (current.state is not ActionState.AWAITING_APPROVAL
                                        and current.state not in {ActionState.APPROVED, ActionState.EXECUTING, ActionState.COMMITTED})
                                    or current.policy_decision is None
                                    or current.policy_decision.outcome is not PolicyOutcome.REQUIRE_APPROVAL
                                    or (current.state is ActionState.AWAITING_APPROVAL and current.expires_at <= utc_now())
                                    or (current.state is ActionState.AWAITING_APPROVAL and current.policy_decision.expires_at <= utc_now())
                                    or current.approval_digest() != item.action_digest):
                                raise PermissionError("transactional action changed or is no longer approvable")

                        actor_id = getpass.getuser()
                        operation = ResponseOperation.APPROVE if is_approve else ResponseOperation.DENY
                        if item is not None:
                            if item.status in UNRESOLVED_STATUSES:
                                service.respond(ResponseSubmission(
                                    inbox_item_id=item.inbox_item_id,
                                    operation=operation,
                                    actor_id=actor_id,
                                    channel="tui-local",
                                    idempotency_key=f"tui-resp:{item.inbox_item_id}",
                                    expected_version=expected_version if expected_version is not None else item.version,
                                    current_action_digest=item.action_digest,
                                ))
                            elif (is_approve and item.status == InboxStatus.APPROVED) or (not is_approve and item.status == InboxStatus.DENIED):
                                logger.info(
                                    "Inbox item %s is already terminal (%s); treating as idempotent",
                                    item.inbox_item_id,
                                    item.status.value,
                                )
                            else:
                                service.respond(ResponseSubmission(
                                    inbox_item_id=item.inbox_item_id,
                                    operation=operation,
                                    actor_id=actor_id,
                                    channel="tui-local",
                                    idempotency_key=f"tui-resp:{item.inbox_item_id}",
                                    expected_version=expected_version if expected_version is not None else item.version,
                                    current_action_digest=item.action_digest,
                                ))

                        if is_approve:
                            from mana_agent.transactional_actions.models import ActionState

                            act_gw = default_action_gateway(self.repo_root)
                            act_id = getattr(item, "action_intent_id", "") or choice.request_id
                            action = act_gw.store.get_action(act_id) if act_id else None
                            if action is not None:
                                if action.state is ActionState.AWAITING_APPROVAL:
                                    act_gw.approvals.issue(action, approved_by=actor_id, ttl_seconds=300)
                                elif action.state in {ActionState.APPROVED, ActionState.EXECUTING, ActionState.COMMITTED}:
                                    logger.info("Action %s is already %s, approval preserved", act_id, action.state.value)
                            if self.gateway is not None and hasattr(self.gateway, "_transactional_runtime"):
                                try:
                                    gw_runtime = self.gateway._transactional_runtime
                                    gw_act = gw_runtime.store.get_action(act_id) if act_id else None
                                    if gw_act is not None:
                                        if gw_act.state is ActionState.AWAITING_APPROVAL:
                                            gw_runtime.gateway.approvals.issue(gw_act, approved_by=actor_id, ttl_seconds=300)
                                        elif gw_act.state in {ActionState.APPROVED, ActionState.EXECUTING, ActionState.COMMITTED}:
                                            logger.info("Gateway action %s is already %s", act_id, gw_act.state.value)
                                except Exception as gw_appr_exc:
                                    logger.debug("Could not issue approval on gateway runtime: %s", gw_appr_exc)

                        msg = f"Transactional action {'approved' if is_approve else 'denied'}."
                        handled = True
                    except Exception as exc:
                        exc_str = str(exc).lower()
                        if is_approve and (
                            "already terminal: approved" in exc_str
                            or "already approved" in exc_str
                            or "already processed" in exc_str
                            or "only an action awaiting approval may receive a grant" in exc_str
                            or "is already in approved" in exc_str
                            or "action is already" in exc_str
                            or "is no longer approvable" in exc_str
                        ):
                            logger.info("Transactional action approval was already recorded: %s", exc)
                            msg = "Transactional action approved."
                            handled = True
                        elif not is_approve and (
                            "already terminal: denied" in exc_str
                            or "already denied" in exc_str
                        ):
                            logger.info("Transactional action denial was already recorded: %s", exc)
                            msg = "Transactional action denied."
                            handled = True
                        elif (
                            "already terminal" in exc_str
                            or "is terminal" in exc_str
                            or "expired" in exc_str
                            or "no longer approvable" in exc_str
                            or "no longer accepts" in exc_str
                        ):
                            if is_batch:
                                return False
                            is_expired = "expired" in exc_str
                            logger.info("Transactional action approval was terminal/expired: %s", exc)
                            self.notify("Approval request expired." if is_expired else "Approval request already decided.", severity="information")
                            from mana_agent.tui.computer_permission import ApprovalDecided
                            self.on_approval_decided(ApprovalDecided({choice.request_id}, approved=False, expired=is_expired))
                            return True
                        else:
                            logger.exception("Failed to resolve transactional action approval directly")
                            msg = f"Failed to record approval decision: {exc}"

                # Match only the queued snapshot, never the entire inbox or future requests.
                if (handled and is_approve and reference is not None and batch_ids
                        and service.repository.get(reference.inbox_item_id).status.value == "approved"):
                    from mana_agent.human_inbox.equivalence import same_approval_request
                    from mana_agent.human_inbox.models import UNRESOLVED_STATUSES
                    from mana_agent.transactional_actions.runtime import default_action_gateway
                    runtime = getattr(self.gateway, "_transactional_runtime", None) if self.gateway else None
                    store = runtime.store if runtime is not None else default_action_gateway(self.repo_root).store
                    extra_approved = 0
                    from mana_agent.tui.computer_permission import ComputerPermissionChoice
                    for rem_id in batch_ids:
                        try:
                            candidate = service.repository.get(rem_id)
                            if candidate.status not in UNRESOLVED_STATUSES or not same_approval_request(reference, candidate, action_store=store):
                                continue
                            rem_choice = ComputerPermissionChoice(
                                request_id=rem_id,
                                decision=choice.decision,
                                remote=choice.remote,
                                server=choice.server,
                                api=choice.api,
                                transactional=choice.transactional,
                                approve_all=False,
                                expected_version=candidate.version,
                            )
                            await self._complete_computer_permission(rem_choice, is_batch=True)
                            if service.repository.get(rem_id).status.value == "approved":
                                if rem_id in self._transactional_modal_queue:
                                    self._transactional_modal_queue.remove(rem_id)
                                extra_approved += 1
                        except Exception as extra_exc:
                            logger.debug("Failed approving additional queued item %s: %s", rem_id, extra_exc)
                    if extra_approved > 0:
                        msg = f"Transactional actions approved ({extra_approved + 1} total)."

                if not handled:
                    self.notify(msg or "Failed to record approval decision.", severity="warning")
                    return False
                if generation != self._approval_drawer_generation:
                    return True
                decided_event = "action.approval.granted" if is_approve else "action.approval.denied"
                self.history.add(
                    CodingActivityEvent(
                        turn_id=self._current_frontend_turn_id or "",
                        activity={
                            "event_type": decided_event,
                            "title": f"Transactional action {'approved' if is_approve else 'denied'}",
                            "status": "completed" if is_approve else "failed",
                            "metadata": {
                                "decision": "approve" if is_approve else "deny",
                                "inbox_item_id": choice.request_id,
                                "permission_request_id": choice.request_id,
                            },
                        }
                    )
                )
                if is_approve:
                    self.notify("Approved", severity="information")
                else:
                    self.notify("Denied", severity="warning")
                return True
            if choice.server:
                command = (
                    self.gateway.server_approval_command
                    if choice.decision is not None
                    else self.gateway.deny_server_approval_command
                )
                result = await asyncio.to_thread(
                    command,
                    choice.request_id,
                    session_id=self._gateway_session_id or "",
                    client_type="tui",
                )
                if generation == self._approval_drawer_generation:
                    self._record_server_approval_completion(choice.request_id, result)
                return True
            if choice.api:
                command = (
                    self.gateway.api_approval_command
                    if choice.decision is not None
                    else self.gateway.deny_api_approval_command
                )
                result = await asyncio.to_thread(
                    command,
                    choice.request_id,
                    session_id=self._gateway_session_id or "",
                )
                if generation == self._approval_drawer_generation:
                    self._record_api_approval_completion(choice.request_id, result)
                return True
            if choice.remote:
                if choice.decision is None:
                    self.notify("Remote SSH request denied.", severity="warning")
                    return True
                result = await asyncio.to_thread(self.gateway.remote_permission_command, choice.request_id)
                if generation == self._approval_drawer_generation:
                    self.notify(result["message"])
                    self.history.add(AssistantMessageEvent(content=result["message"]))
                return True
            if choice.decision is None:
                await asyncio.to_thread(
                    deny_computer_permission,
                    choice.request_id,
                    client_type="tui",
                )
                self.notify("Computer action denied.", severity="warning")
                return True
            result = await asyncio.to_thread(
                decide_computer_permission,
                choice.request_id,
                decision=choice.decision,
                client_type="tui",
            )
            if generation == self._approval_drawer_generation:
                self.notify(result.message)
                self.history.add(AssistantMessageEvent(
                    content=f"Computer action completed after permission approval: {result.message}"
                ))
            return True
        except Exception as exc:
            if generation != self._approval_drawer_generation:
                return False
            exc_str = str(exc).lower()
            if "expired" in exc_str or "already terminal" in exc_str or "is terminal" in exc_str:
                if is_batch:
                    return False
                action_name = (
                    "Server approval"
                    if choice.server
                    else "API approval"
                    if choice.api
                    else "Computer permission"
                )
                self.notify(f"{action_name} request expired.", severity="information")
                from mana_agent.tui.computer_permission import ApprovalDecided
                self.on_approval_decided(ApprovalDecided({choice.request_id}, approved=False, expired="expired" in exc_str))
                return True
            action_name = (
                "Server approval"
                if choice.server
                else "API approval"
                if choice.api
                else "Computer permission"
            )
            self.notify(f"{action_name} action failed: {exc}", severity="error")
            if choice.api:
                self.history.add(AssistantMessageEvent(
                    content=f"API approval action failed: {exc}",
                    turn_id=choice.request_id,
                ))
            elif choice.server:
                self.history.add(AssistantMessageEvent(
                    content=f"Server approval action failed: {exc}",
                    turn_id=choice.request_id,
                ))
            return False

    def _record_server_approval_completion(
        self,
        approval_request_id: str,
        result: dict[str, Any],
    ) -> None:
        """Show the validated approved server outcome as the TUI's terminal assistant message."""
        message = str(result.get("answer") or result.get("message") or "").strip()
        if not message:
            raise ValueError("Server approval completed without a validated result message.")
        status = str(result.get("status") or "")
        self.history.add(AssistantMessageEvent(
            content=message,
            turn_id=approval_request_id,
        ))
        if status == "denied":
            self.notify("Server action denied. No server command was executed.", severity="warning")
            self.update_status("Server action denied")
            return
        if status in {"failed", "verification_failed"}:
            self.notify(f"Server action failed: {message}", severity="error")
            self.update_status("Server action failed")
            return
        self.notify("Server action completed. Response added to chat.")
        self.update_status("Server action completed")

    def _record_api_approval_completion(
        self,
        approval_request_id: str,
        result: dict[str, Any],
    ) -> None:
        """Show the validated approved API outcome as the TUI's terminal assistant message."""
        message = str(result.get("message") or "").strip()
        if not message:
            raise ValueError("API approval completed without a validated result message.")
        status = str(result.get("status") or "")
        self.history.add(AssistantMessageEvent(
            content=message,
            turn_id=approval_request_id,
        ))
        if status == "denied":
            self.notify("API request denied. No API request was executed.", severity="warning")
            self.update_status("API request denied")
            return
        if status in {"failed", "approved_not_executed"}:
            self.notify(f"API request failed: {message}", severity="error")
            self.update_status("API request failed")
            return
        self.notify("API request completed. Response added to chat.")
        self.update_status("API request completed")

    def _show_mode_prompt(self, prompt) -> None:
        if not prompt or not prompt.get("task_id") or not prompt.get("inbox_id"):
            return
        from mana_agent.tui.mode_prompt import ModePrompt

        if isinstance(self.screen, ModePrompt):
            return
        workflow = self.gateway.mode_workflow
        prompt = workflow.pending_prompt(self._gateway_session_id, prompt["task_id"], prompt["inbox_id"])
        if prompt is None:
            return

        def finish(result):
            if result:
                self.run_worker(self._respond_to_mode_prompt(prompt, result))

        self.push_screen(ModePrompt(prompt), finish)

    async def _respond_to_mode_prompt(self, prompt, result) -> None:
        from mana_agent.human_inbox.models import ResponseOperation

        action, value = result
        operation = {"approve": ResponseOperation.APPROVE, "reject": ResponseOperation.DENY,
                     "revise": ResponseOperation.DENY, "answer": ResponseOperation.ANSWER}[action]
        try:
            response = await asyncio.to_thread(
                self.gateway.mode_workflow.respond, self._gateway_session_id,
                prompt["inbox_id"], operation, frontend="tui",
                answer=value if action == "answer" else None,
                comment="Revise plan:\n" + value if action == "revise" else "",
            )
            self.history.add(AssistantMessageEvent(content=response.message))
        except (PermissionError, ValueError, RuntimeError) as exc:
            self.notify(str(exc), severity="error")
            if action == "answer":
                self._show_mode_prompt(prompt)

    def update_status(self, text: str) -> None:
        """Update the status reactive. The watcher + refresh_footer will keep the footer in sync."""
        self.status_text = text
        # Do not call refresh_footer directly here — the @reactive watcher
        # (watch_status_text) will invoke it. This prevents duplicate work
        # and ordering problems during early app lifecycle.

    def _refresh_runtime_labels(self) -> None:
        self._permission_text = self._permission_label()
        mode_reader = getattr(self.gateway, "get_session_execution_mode", None)
        mode = mode_reader(self._gateway_session_id) if mode_reader and self._gateway_session_id else ""
        self._mode_text = str(mode).upper() if mode else ""
        self.refresh_footer()

    def refresh_footer(self) -> None:
        if not self._footer.is_mounted:
            return
        sep = glyph(" · ", " | ")
        panel = self.chat_log.get_execution_panel(self._current_frontend_turn_id) if self.chat_log and self._current_frontend_turn_id else None
        mode = (panel.workflow_mode if panel else "") or self._mode_text
        parts = [part for part in (self.model, mode, self._permission_text) if part]
        meters = [panel.context_meter for panel in self.chat_log._execution_panels.values() if panel.context_meter] if self.chat_log else []
        metric = meters[-1] if meters else {}
        used, window = metric.get("used_tokens"), metric.get("context_window")
        ratio = metric.get("utilization_ratio")
        if isinstance(used, (float, int)) and window:
            ratio = used / window
        if isinstance(ratio, (float, int)) and 0 <= ratio <= 1:
            marker = "~" if metric.get("estimated") else ""
            parts.append(f"Context {marker}{100 * (1 - ratio):.0f}% left")
        value = sep.join(parts)
        if str(self._footer.renderable) != value:
            self._footer.update(Text(value, no_wrap=True, overflow="ellipsis"))

    def on_resize(self, _: events.Resize) -> None:
        self.call_after_refresh(self.refresh_footer)

    async def action_interrupt_execution(self) -> None:
        if not self.can_interrupt or self.interrupt_requested or len(self.screen_stack) > 1:
            return
        session_id = self._gateway_session_id
        self.interrupt_requested = True
        if self.chat_log:
            self.chat_log.refresh_activity_status()

        async def interrupt() -> None:
            try:
                accepted = await asyncio.to_thread(self.gateway.cancel, session_id, source="tui_escape")
            except Exception as exc:
                if self._gateway_session_id == session_id:
                    self.interrupt_requested = False
                    self.notify(f"Unable to interrupt: {safe_text(exc, 160)}", severity="error")
                return
            if self._gateway_session_id == session_id and not accepted:
                self.interrupt_requested = False
                self.notify("No cancellable turn is active.", severity="information")
            if self._gateway_session_id == session_id and self.chat_log:
                self.chat_log.refresh_activity_status()

        self.run_worker(interrupt(), group="chat-interrupt")

    def action_latest_activity(self) -> None:
        if self.chat_log is not None:
            self.chat_log._scroll_to_latest()

    def action_clear_log(self) -> None:
        if self.chat_log:
            self.chat_log.clear_log()
        # keep history but visually reset (or call self.history.clear() for full reset)
        self.update_status("Log cleared (history preserved)")

    async def action_simulate_response(self) -> None:
        """Demo: inject a complete realistic flow (user + call + result + final answer)."""
        self.update_status("Simulating agent turn with tool...")

        # 1. User message (as if typed)
        user_msg = UserMessageEvent(content="Show me the structure of the main agent and list any recent changes.")
        self.history.add(user_msg)

        await asyncio.sleep(0.15)

        # 2. Tool call (agent decides)
        call = ToolCallEvent(
            tool_name="read_file",
            args={"path": "src/mana_agent/multi_agent/agents/main_agent.py"},
            call_id="demo-call-001",
            summary="Read main_agent.py",
        )
        self.history.add(call)

        await asyncio.sleep(0.35)

        # 3. Tool result
        result = ToolResultEvent(
            call_id="demo-call-001",
            tool_name="read_file",
            success=True,
            result=(
                "class MainAgent(BaseAgent):\n"
                "    async def handle(self, task: str, context: dict):\n"
                "        ...\n"
                "# (truncated for demo)"
            ),
            summary="Read 142 lines",
            duration_ms=87,
        )
        self.history.add(result)

        await asyncio.sleep(0.2)

        # 4. Streaming assistant response
        assistant = AssistantMessageEvent(
            content="",
            is_streaming=True,
            event_id="demo-asst-001",
            turn_id=user_msg.turn_id,
        )
        self.history.add(assistant)

        demo_text = (
            "The main agent orchestrates planning, tool selection and verification.\n"
            "It recently gained improved memory and multi-agent routing.\n"
            "\n"
            "Key files involved:\n"
            "- `multi_agent/agents/main_agent.py`\n"
            "- `multi_agent/runtime/`\n"
            "- `chat/history.py` (new subscription system)"
        )

        for token in demo_text.split(" "):
            self.history.add(
                StreamTokenEvent(
                    token=token + " ",
                    assistant_event_id="demo-asst-001",
                    turn_id=user_msg.turn_id,
                )
            )
            await asyncio.sleep(0.02)

        # Finalize the assistant message (non-streaming version replaces/ends)
        final = AssistantMessageEvent(
            content=demo_text,
            is_streaming=False,
            event_id="demo-asst-001",
            turn_id=user_msg.turn_id,
        )
        self.history.add(final)

        self.token_count += 42
        self.update_status("Agent response complete")

    def action_attach_file(self) -> None:
        """Open the file attachment picker modal."""
        from mana_agent.tui.attachment_picker import FileAttachmentModal

        self.push_screen(FileAttachmentModal(self.repo_root), self._apply_file_attachment)

    def action_steer_execution(self) -> None:
        """Open modal to steer currently active execution."""
        from mana_agent.tui.steer_modal import SteerModal

        self.push_screen(SteerModal(), self._handle_steer_modal_result)

    def _handle_steer_modal_result(self, result: str | None) -> None:
        if not result or not result.strip():
            return
        asyncio.create_task(self._submit_live_steering(result.strip()))

    async def _submit_live_steering(self, instruction: str) -> None:
        import uuid
        target_turn_id = self._current_frontend_turn_id or getattr(self.chat_log, "_current_turn_id", "")
        session_id = self._gateway_session_id
        if not target_turn_id or not self.gateway or not session_id:
            self.notify("Steering unavailable: no active execution in this session.", severity="warning")
            return
        request_id = f"steer_{uuid.uuid4().hex}"
        message = UserMessageEvent(content=instruction, turn_id=target_turn_id,
                                   event_id=f"steering-{request_id}", is_steering=True,
                                   steering_status="pending", steering_instruction=instruction,
                                   request_id=request_id)
        self.history.add(message)
        try:
            result = await asyncio.to_thread(self.gateway.submit_steering,
                                             session_id=session_id, text=instruction,
                                             scope="execution", request_id=request_id, execution_id=target_turn_id)
            if self._gateway_session_id != session_id:
                return
            status = str(result.get("status") or "queued")
            self._safe_post_activity({"event_type": "steering.queued", "title": "Live steering queued",
                                      "status": "queued", "operation_id": request_id, "detail": instruction,
                                      "metadata": {"request_id": request_id, "text": instruction,
                                                   "steering_sequence": result.get("sequence", 0),
                                                   "status": status}}, target_turn_id)
        except Exception as exc:
            if self._gateway_session_id != session_id:
                return
            status = "rejected" if isinstance(exc, (ValueError, PermissionError)) else "failed"
            self._safe_post_activity({"event_type": f"steering.{status}", "title": f"Live steering {status}",
                                      "status": status, "summary": safe_text(exc), "operation_id": request_id,
                                      "metadata": {"request_id": request_id}}, target_turn_id)
            self.notify(f"Steering {status}: {safe_text(exc, 120)}", severity="error")

    def _apply_file_attachment_path(self, target_path: Path | str) -> None:
        p = Path(target_path).expanduser()
        if not p.is_absolute():
            p = (self.repo_root / p).resolve()
        else:
            p = p.resolve()
        if not p.is_file():
            self.notify(f"Attachment not found: {p}", severity="error")
            return
        self._apply_file_attachment(p)

    def _apply_file_attachment(self, path: Path | None) -> None:
        if path is None:
            return
        if not self._gateway_session_id and self.gateway is not None:
            self._gateway_session_id = self.gateway.create_session(frontend="tui")
        sid = self._gateway_session_id or "tui_session"
        from mana_agent.chat.attachments import AttachmentStore, AttachmentValidator

        validator = AttachmentValidator()
        current = self.attachment_bar.attachments if self.attachment_bar else []
        if len(current) >= validator.max_count:
            self.notify(f"Cannot attach more than {validator.max_count} files.", severity="error")
            return
        try:
            attachment = AttachmentStore.save_from_path(
                session_id=sid,
                source_path=path,
                validator=validator,
            )
            if self.attachment_bar:
                self.attachment_bar.add_attachment(attachment)
            self.notify(f"Attached {attachment.filename} ({attachment.format_size()})", severity="information")
        except Exception as exc:
            self.notify(f"Failed to attach file: {exc}", severity="error")

    async def on_message_input_submitted(self, event: MessageInput.Submitted) -> None:
        """Handle user pressing Enter in the input box. Always uses the real turn handler."""
        # Preserve message whitespace and explicit line breaks. A final newline is
        # an editing artifact from the composer, not part of the submitted turn.
        text = event.value.rstrip("\r\n")
        await self._transcript_ready.wait()

        # Check attachment commands before regular turn execution
        if text == "/attach" or text.startswith("/attach "):
            parts = text.split(" ", 1)
            if len(parts) > 1 and parts[1].strip():
                self._apply_file_attachment_path(parts[1].strip())
            else:
                self.action_attach_file()
            if self.input and self.input.value == event.value:
                self.input.reset()
            return

        if text == "/clear-attachments":
            if self.attachment_bar:
                removed = self.attachment_bar.clear_attachments()
                self.notify(f"Removed {len(removed)} attachments.", severity="information")
            if self.input and self.input.value == event.value:
                self.input.reset()
            return

        if text == "/detach" or text.startswith("/detach "):
            parts = text.split(" ", 1)
            if self.attachment_bar:
                if len(parts) > 1 and parts[1].strip():
                    arg = parts[1].strip()
                    if arg.isdigit():
                        idx = int(arg) - 1
                        removed = self.attachment_bar.remove_attachment(idx)
                    else:
                        removed = self.attachment_bar.remove_by_name(arg)
                else:
                    count = len(self.attachment_bar.attachments)
                    removed = self.attachment_bar.remove_attachment(count - 1) if count > 0 else None
                if removed:
                    self.notify(f"Removed attachment: {removed.filename}", severity="information")
                else:
                    self.notify("No matching attachment to remove.", severity="warning")
            if self.input and self.input.value == event.value:
                self.input.reset()
            return

        pending_atts = self.attachment_bar.attachments if self.attachment_bar else []
        if not text.strip() and not pending_atts:
            return

        if self._turn_in_progress and text.startswith("/"):
            name, *arguments = text.split()
            name = name.lower()
            if name == "/cancel" and not arguments:
                self.input.reset()
                await self.action_interrupt_execution()
                return
            if name in {"/status", "/help"} and not arguments and self.gateway and hasattr(self.gateway, "dispatch_command"):
                self.input.reset()
                result = self.gateway.dispatch_command(text, session_id=self._gateway_session_id, frontend="tui")
                if result and result.message:
                    self.notify(result.message)
                return
            if name != "/steer":
                self.notify("This command is unavailable during execution. Press Esc to interrupt.", severity="warning")
                return

        if self._turn_in_progress:
            # Plain Enter steers the existing turn.
            instruction = text.strip()
            if instruction and instruction.split(maxsplit=1)[0].lower() == "/steer":
                instruction = instruction[len("/steer"):].strip()
            elif not instruction:
                self.notify("Pending attachments will be sent with the next turn.")
                return
            if self.input and self.input.value == event.value:
                self.input.reset()
            if instruction:
                asyncio.create_task(self._submit_live_steering(instruction))
            else:
                self.action_steer_execution()
            return

        if text == "/steer" or text.startswith("/steer "):
            instruction = text[len("/steer"):].strip()
            if not instruction:
                self.action_steer_execution()
            else:
                asyncio.create_task(self._submit_live_steering(instruction))
            if self.input and self.input.value == event.value:
                self.input.reset()
            return

        if text == "/models" or text.startswith("/models "):
            if text == "/models":
                from mana_agent.tui.model_management import ModelManagementScreen

                self.push_screen(ModelManagementScreen(current_model=self.model or "default"), self._apply_model_selection)
            else:
                from mana_agent.tui.model_management import plain_models_command

                try:
                    message, selection = await asyncio.to_thread(
                        plain_models_command,
                        text,
                        current_model=self.model or "default",
                    )
                except Exception as exc:
                    message, selection = f"Model command failed: {exc}", None
                self.history.add(AssistantMessageEvent(content=message))
                self._apply_model_selection(selection)
            if self.input and self.input.value == event.value:
                self.input.reset()
            return

        if text.startswith("/") and self.gateway is not None and hasattr(self.gateway, "dispatch_command"):
            if not self._gateway_session_id:
                self._gateway_session_id = self.gateway.create_session(frontend="tui")
            command_name = text.split(maxsplit=1)[0].lower()
            plan_command = command_name == "/plan"
            approval_command = command_name in {"/approve", "/accept", "/act", "/revise", "/reject"}
            activity_turn = UserMessageEvent(content=text) if plan_command or approval_command else None
            if activity_turn is not None:
                self.history.add(activity_turn)

            if plan_command:
                self._turn_in_progress = True
                self._current_frontend_turn_id = activity_turn.turn_id
                self._refresh_command_menu()
                if self.chat_log:
                    self.chat_log.set_active_turn(activity_turn.turn_id)
                self.update_status("Preparing plan…")
                await asyncio.sleep(0)

                def _plan_activity(activity: dict[str, Any]) -> None:
                    self._safe_post_activity(activity, activity_turn.turn_id)
                    title = str(activity.get("title") or "Preparing plan")
                    try:
                        self.call_from_thread(self.update_status, f"{title}…")
                    except Exception:
                        pass

                try:
                    result = await asyncio.to_thread(
                        self.gateway.dispatch_command,
                        text,
                        session_id=self._gateway_session_id,
                        frontend="tui",
                        frontend_data={"activity_sink": _plan_activity, "activity_turn_id": activity_turn.turn_id},
                    )
                except Exception as exc:
                    result = None
                    self._safe_post_activity(
                        {
                            "event_id": f"plan-preparation-{self._gateway_session_id}",
                            "event_type": "agent.planning",
                            "title": "Plan preparation failed",
                            "status": "failed",
                            "detail": str(exc),
                        },
                        activity_turn.turn_id,
                    )
                finally:
                    self._finish_active_turn()
            elif approval_command:
                self._turn_in_progress = True
                self._current_frontend_turn_id = activity_turn.turn_id
                self._refresh_command_menu()
                if self.chat_log:
                    self.chat_log.set_active_turn(activity_turn.turn_id)
                self.update_status("Approving plan…")
                await asyncio.sleep(0)
                try:
                    result = await asyncio.to_thread(
                        self.gateway.dispatch_command,
                        text,
                        session_id=self._gateway_session_id,
                        frontend="tui",
                    )
                except Exception as exc:
                    result = None
                    self.history.add(
                        AssistantMessageEvent(
                            content=f"Plan approval failed: {exc}",
                            turn_id=activity_turn.turn_id,
                        )
                    )
                finally:
                    self._finish_active_turn()
            else:
                result = self.gateway.dispatch_command(
                    text, session_id=self._gateway_session_id, frontend="tui"
                )
            if result is None and (plan_command or approval_command):
                if plan_command:
                    self.history.add(AssistantMessageEvent(
                        content="Plan preparation failed. See Activity for details.",
                        turn_id=activity_turn.turn_id,
                    ))
                if self.input and self.input.value == event.value:
                    self.input.reset()
                self.update_status("Ready")
                self._turn_in_progress = False
                return
            if plan_command and result is not None and result.status == "error":
                self._safe_post_activity(
                    {
                        "event_id": f"plan-preparation-{self._gateway_session_id}",
                        "event_type": "agent.planning",
                        "title": "Plan preparation failed",
                        "status": "failed",
                        "detail": result.message,
                    },
                    activity_turn.turn_id,
                )
            if result is not None:
                if result.data.get("activity_visibility") in {"minimal", "normal", "verbose"}:
                    self.activity_visibility = result.data["activity_visibility"]
                    if self.chat_log:
                        for panel in self.chat_log._execution_panels.values():
                            panel._schedule_render()
                new_session_id = str(result.data.get("session_id") or "")
                for response_event in result.events:
                    if response_event.get("type") == "timeline.replace":
                        self._replace_timeline(response_event.get("messages") or [])
                    elif response_event.get("type") == "session.picker":
                        from mana_agent.tui.session_management import SessionPickerScreen

                        self.push_screen(SessionPickerScreen(response_event.get("sessions") or []), self._apply_session_action)
                if new_session_id:
                    self._handle_session_replacement(new_session_id, clear_view=False)
                if result.status == "input_required" and (result.next_prompt or {}).get("kind") == "connector.telegram.setup":
                    from mana_agent.tui.session_management import TelegramSetupScreen

                    self.push_screen(TelegramSetupScreen(self.repo_root), self._apply_telegram_setup)
                if result.message:
                    self.history.add(AssistantMessageEvent(
                        content=result.message,
                        turn_id=activity_turn.turn_id if activity_turn is not None else "",
                    ))
                self._show_mode_prompt(result.next_prompt)
                if self.input and self.input.value == event.value:
                    self.input.reset()
                waiting_for_answers = (result.next_prompt or {}).get("kind") == "mode.clarification"
                self.update_status("Waiting for your answers" if waiting_for_answers else "Ready")
                self._refresh_runtime_labels()
                return
            if approval_command:
                self._turn_in_progress = False

        if text.strip() == "/new" and self.gateway is not None and hasattr(self.gateway, "start_new_conversation"):
            if not self._gateway_session_id:
                self._gateway_session_id = self.gateway.create_session(frontend="tui")
            self._start_new_conversation()
            if self.input and self.input.value == event.value:
                self.input.reset()
            self.update_status("Ready")
            return

        # Clear input immediately (premium feel)
        if self.input and self.input.value == event.value:
            self.input.reset()

        self.update_status("Thinking...")
        self.token_count += len(text.split())

        att_dicts = tuple(a.to_dict() for a in pending_atts)
        if self.attachment_bar:
            self.attachment_bar.clear_attachments()

        # Record user message first so ChatLog can paint the bubble immediately
        user_event = UserMessageEvent(content=text, attachments=att_dicts)
        self.history.add(user_event)

        self._turn_in_progress = True
        self._current_frontend_turn_id = user_event.turn_id
        if self.chat_log:
            self.chat_log.set_active_turn(user_event.turn_id)
        self._refresh_command_menu()
        await asyncio.sleep(0)

        self.run_worker(self._handle_real_turn_guarded(user_event), group="chat-turn", exclusive=True)

    def _start_new_conversation(self) -> str:
        """Apply the same gateway-owned conversation boundary as the plain CLI."""
        if self.gateway is None or not hasattr(self.gateway, "start_new_conversation"):
            raise RuntimeError("A gateway session is required to start a new conversation.")
        if not self._gateway_session_id:
            self._gateway_session_id = self.gateway.create_session(frontend="tui")
        new_sid = self.gateway.start_new_conversation(
            self._gateway_session_id, frontend="tui"
        )
        return self._handle_session_replacement(new_sid, clear_view=True)

    def _replace_timeline(self, messages: list[dict[str, Any]]) -> None:
        """Replace visible state from canonical chronological durable messages."""
        self._clear_conversation_view()
        for message in messages:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            turn_id = str(message.get("turn_id") or "")
            identity = {"event_id": str(message["message_id"])} if message.get("message_id") else {}
            if role == "user":
                attachments = tuple(message.get("attachments") or ())
                self.history.add(UserMessageEvent(content=content, turn_id=turn_id, attachments=attachments, **identity))
            elif role == "assistant":
                self.history.add(AssistantMessageEvent(content=content, turn_id=turn_id, **identity))

    def _clear_conversation_view(self) -> None:
        """Clear both canonical TUI events and already-mounted timeline widgets."""
        self.history.clear()
        if self.chat_log is not None:
            self.chat_log.clear_log()
        self._tool_cid_map.clear()
        self.token_count = 0

    def _apply_session_action(self, action: Any | None) -> None:
        if action is None or self.gateway is None or not self._gateway_session_id:
            return
        import shlex

        command = f"/session {action.action} {shlex.quote(action.session_id)}"
        if action.action == "rename":
            command += " " + shlex.quote(action.title)
        result = self.gateway.dispatch_command(
            command,
            session_id=self._gateway_session_id,
            frontend="tui",
            confirmed=action.action == "delete",
        )
        if result.status == "error":
            self.notify(result.message, severity="error")
            return
        replacement_id = str(result.data.get("session_id") or "")
        for event in result.events:
            if event.get("type") == "timeline.replace":
                self._replace_timeline(event.get("messages") or [])
        if replacement_id or action.action == "switch":
            self._handle_session_replacement(replacement_id or action.session_id, clear_view=False)
        if result.message:
            self.notify(result.message)

    def _apply_telegram_setup(self, setup: Any | None) -> None:
        if setup is None:
            return
        self.run_worker(self._complete_telegram_setup(setup), exclusive=True)

    async def _complete_telegram_setup(self, setup: Any) -> None:
        from mana_agent.connectors.service import TelegramConnectRequest

        self.update_status("Validating Telegram…")
        try:
            result = await asyncio.to_thread(
                self.gateway.connector_service.connect_telegram,
                TelegramConnectRequest(
                    transport=setup.transport, repository=setup.repository,
                    allowed_users=setup.allowed_users, allowed_chats=setup.allowed_chats,
                    webhook_url=setup.webhook_url, secret_source=setup.secret_source,
                ),
                token=setup.token,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self.notify(str(exc), severity="error")
        else:
            self.notify(result.message)
        finally:
            self.update_status("Ready")

    def _finish_active_turn(self) -> None:
        self._current_frontend_turn_id = None
        self._turn_in_progress = False
        self.interrupt_requested = False
        self._refresh_command_menu()
        self._refresh_runtime_labels()
        if self.chat_log is not None:
            self.chat_log.set_active_turn(None)

    async def _handle_real_turn_guarded(self, user_event: UserMessageEvent) -> None:
        self._current_frontend_turn_id = user_event.turn_id
        self._turn_in_progress = True
        if self.chat_log is not None:
            self.chat_log.set_active_turn(user_event.turn_id)
        self._refresh_command_menu()
        try:
            await self._handle_real_turn(user_event)
        finally:
            self._finish_active_turn()
            self.call_after_refresh(self._queue_outstanding_transactional_approvals)

    def _apply_model_selection(self, selection: Any) -> None:
        if selection is None:
            return
        self.model = selection.model_id
        if self.gateway is not None:
            try:
                from dataclasses import replace

                from mana_agent.gateway import AgentChatGateway

                old_gateway = self.gateway
                old_stack = old_gateway.get_stack()
                worker = getattr(old_stack, "tool_worker_client", None)
                if worker is not None and hasattr(worker, "stop"):
                    worker.stop()
                # Rebuilding model clients must preserve the active conversation identity.
                config = replace(
                    old_gateway.config,
                    model=selection.model_id,
                    session_id=self._gateway_session_id,
                )
                self.gateway = AgentChatGateway(self.repo_root, config=config, settings=old_gateway.settings)
                self._gateway_session_id = self.gateway.create_session(
                    frontend="tui", session_id=self._gateway_session_id
                )
                rich = self.gateway.get_rich_context(self._gateway_session_id)
                self.chat_service = rich.chat_service
                self.coding_agent = rich.coding_agent
                self.tools_orchestrator = rich.tools_orchestrator
            except Exception as exc:
                self.history.add(AssistantMessageEvent(content=f"Model selected, but the runtime could not be rebuilt: {exc}"))
        self._refresh_command_menu()
        self._refresh_runtime_labels()
        self.update_status(f"Model changed to {selection.qualified_id}")
        if selection.persist:
            self.history.add(AssistantMessageEvent(content=f"Default model saved: `{selection.qualified_id}`"))

    async def _send_initial_prompt(self) -> None:
        """Send the prompt that was passed on the command line (e.g. mana-agent chat "foo")."""
        if not self.initial_prompt:
            return
        await self._transcript_ready.wait()
        prompt = self.initial_prompt
        self.initial_prompt = None  # only once
        # Seed the user message (it may have already been conceptually sent)
        # Use the same path as normal input for consistency
        user_event = UserMessageEvent(content=prompt)
        # Avoid duplicate if somehow already added
        if not any(
            isinstance(e, UserMessageEvent) and e.content == prompt for e in self.history.get_events()
        ):
            self.history.add(user_event)
        await self._handle_real_turn_guarded(user_event)

    @staticmethod
    def _extract_answer(result: Any) -> str:
        """Normalize coding-agent / chat-service results into plain answer text."""
        if result is None:
            return ""
        if isinstance(result, dict):
            for key in ("answer", "content", "text", "message", "output"):
                value = result.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
            return str(result).strip()
        for attr in ("answer", "content", "text"):
            value = getattr(result, attr, None)
            if value is not None and str(value).strip():
                return str(value).strip()
        text = str(result).strip()
        return text if text and text != "None" else ""

    def _count_tool_events_for_turn(self, turn_id: str) -> int:
        """Count ToolCall/ToolResult events already recorded for this turn."""
        n = 0
        for event in self.history.get_events():
            is_runtime_tool = isinstance(event, CodingActivityEvent) and str(event.activity.get("event_type") or "").startswith(("tool.", "command."))
            if not isinstance(event, (ToolCallEvent, ToolResultEvent)) and not is_runtime_tool:
                continue
            if str(getattr(event, "turn_id", "") or "") == str(turn_id or ""):
                n += 1
        return n

    def _make_tool_emit_bridge(self, *, original_emit: Any, turn_id: str):
        """Wrap emit_tool_event so the transcript can correlate legacy tool start and end events."""

        def bridged_emit(kind, tool, *, args="", duration=None, error="", event_id=None, **kwargs):
            try:
                original_emit(kind, tool, args=args, duration=duration, error=error, event_id=event_id, **kwargs)
            except Exception:
                pass
            kind_l = str(kind).lower().strip()
            key = str(event_id).strip() if event_id else f"{tool}:{str(args)[:80]}"
            is_start = (
                kind_l in {"start", "started", "tool_start", "worker_request_start"}
                or kind_l.endswith("_start")
            )
            is_end = (
                kind_l in {"end", "finished", "done", "success", "tool_end", "worker_request_end"}
                or kind_l.endswith("_end")
            )
            is_error = (
                kind_l in {"error", "fail", "failed", "tool_error", "worker_request_error"}
                or "error" in kind_l
                or "fail" in kind_l
            )
            if is_start:
                call_kwargs: dict[str, Any] = {
                    "tool_name": str(tool),
                    "args": args or {},
                    "summary": f"{tool} started",
                    "turn_id": turn_id,
                }
                if event_id and str(event_id).strip():
                    call_kwargs["call_id"] = str(event_id).strip()
                tcall = ToolCallEvent(**call_kwargs)
                self._tool_cid_map[key] = tcall.call_id
                if event_id and str(event_id).strip():
                    self._tool_cid_map[str(event_id).strip()] = tcall.call_id
                self.history.add(tcall)
            elif is_end and not is_error:
                cid = self._tool_cid_map.get(key) or self._tool_cid_map.get(str(event_id or "").strip()) or (
                    str(event_id) if event_id else f"tool-{tool}"
                )
                duration_ms = None
                if isinstance(duration, (int, float)):
                    duration_ms = int(duration * 1000) if duration < 50 else int(duration)
                tres = ToolResultEvent(
                    call_id=cid,
                    tool_name=str(tool),
                    success=True,
                    result={"duration": duration} if duration is not None else "(ok)",
                    summary=f"{tool} completed",
                    duration_ms=duration_ms,
                    turn_id=turn_id,
                )
                self.history.add(tres)
            elif is_error:
                cid = self._tool_cid_map.get(key) or self._tool_cid_map.get(str(event_id or "").strip()) or (
                    str(event_id) if event_id else f"tool-{tool}"
                )
                tres = ToolResultEvent(
                    call_id=cid,
                    tool_name=str(tool),
                    success=False,
                    error=str(error or "failed"),
                    summary=f"{tool} failed",
                    turn_id=turn_id,
                )
                self.history.add(tres)

        return bridged_emit

    def _replay_tool_traces_from_result(self, result: Any, *, turn_id: str) -> None:
        """Replay tool traces onto ChatHistory when live emit did not fire."""
        rows: list[Any] = []
        payload = getattr(result, "payload", None) or {}
        if isinstance(payload, dict):
            rows.extend(payload.get("actions_taken") or [])
            if not rows:
                rows.extend(payload.get("trace") or [])
        if not rows:
            rows.extend(list(getattr(result, "trace", None) or []))

        for idx, row in enumerate(rows[:40]):
            if row is None:
                continue
            if hasattr(row, "to_dict"):
                try:
                    row = row.to_dict()
                except Exception:
                    row = None
            if not isinstance(row, dict):
                continue
            if row.get("backend") in {"codex", "internal"}:
                # Already rendered live in the backend-neutral execution panel.
                continue
            tname = str(row.get("tool_name") or row.get("tool") or "tool")
            args_summary = f"{tname} observed invocation"
            cid = str(row.get("call_id") or row.get("event_id") or f"gw-{tname}-{turn_id}-{idx}")
            status = str(row.get("status", "ok") or "ok").lower()
            preview = row.get("output_preview") or row.get("result_summary") or row.get("result") or "(ok)"
            duration_ms = row.get("duration_ms")
            try:
                duration_ms_int = int(float(duration_ms)) if duration_ms is not None else None
            except (TypeError, ValueError):
                duration_ms_int = None
            self.history.add(
                ToolCallEvent(
                    tool_name=tname,
                    args=args_summary,
                    call_id=cid,
                    summary=args_summary[:60],
                    turn_id=turn_id,
                )
            )
            self.history.add(
                ToolResultEvent(
                    call_id=cid,
                    tool_name=tname,
                    success=status not in {"failed", "error", "blocked"},
                    result=preview,
                    summary=f"{tname} {status}",
                    duration_ms=duration_ms_int,
                    turn_id=turn_id,
                    error="" if status not in {"failed", "error", "blocked"} else str(preview)[:300],
                )
            )

    async def _handle_real_turn(self, user_event: UserMessageEvent) -> None:
        """Real turn handler with *exact* parity to the console coding-agent path.

        All decisions, generate* calls, kwargs, resume cycles, flow state,
        callbacks, and tool emission are constructed to be identical to the
        logic in chat_cli.py so that:
          - behavior (edits, memory, plans, auto-execute, verification) is unchanged
          - observable tools appear in the turn’s inline execution stream

        Planning collection state is also supported for interactive pre-questions.
        """
        question = user_event.content
        turn_id = user_event.turn_id

        def _on_coding_event(event: Any) -> None:
            from mana_agent.services.execution_event_hub import get_execution_event_hub
            from mana_agent.workspaces.paths import repository_id_for_path
            payload = project_activity(event)
            if payload is None:
                return
            task_id = str(payload.get("task_id") or "")
            if task_id:
                self._execution_to_frontend_turn[task_id] = turn_id
            get_execution_event_hub().publish(payload, conversation_id=self._gateway_session_id,
                                              execution_id=turn_id, repository_id=repository_id_for_path(self.repo_root))

        self.update_status("Thinking with multi-agent flow...")

        # Legacy observers are scoped to this turn; never replace a process-global
        # emitter while another session or agent may be using it.
        from mana_agent.commands.ui_helpers import tool_event_scope
        tool_scope = tool_event_scope(self._make_tool_emit_bridge(original_emit=lambda *a, **k: None, turn_id=turn_id))
        tool_scope.__enter__()

        answer = ""
        used_full_flow = False
        ctx_result = {"root": str(self.repo_root)}

        try:
            # Prefer gateway-owned turn engine (auto-chat + coding agent + model decision).
            # Gateway routes connector queries (Gmail, MCP, web) via auto-chat / ChatService.ask
            # and only uses CodingAgent for edit/plan. TUI must not bypass this routing.
            if self.gateway is not None and hasattr(self.gateway, "process_turn"):
                try:
                    if not self._gateway_session_id:
                        if hasattr(self.gateway, "create_session"):
                            self._gateway_session_id = self.gateway.create_session(frontend="tui")
                        else:
                            self._gateway_session_id = f"tui-{id(self)}"
                    sid = self._gateway_session_id

                    # Sync indexes so agent-tools ask (auto-chat) can retrieve context
                    if hasattr(self.gateway, "set_index_dirs"):
                        try:
                            self.gateway.set_index_dirs(
                                index_dir=self.index_dir,
                                index_dirs=self.index_dirs,
                            )
                        except Exception:
                            pass

                    tool_cb = None
                    try:
                        from mana_agent.commands.ui_helpers import RichToolActivityObserver

                        tool_cb = RichToolActivityObserver(show_inputs=True)
                    except Exception:
                        tool_cb = None
                    ask_callbacks = [tool_cb] if tool_cb is not None else []

                    def _on_gateway_sink_event(event_type: str, title: str = "", **kwargs: Any) -> None:
                        metadata = dict(kwargs.get("metadata") or {})
                        display_title = title or event_type.replace("_", " ").title()
                        event_payload = {
                            "event_type": event_type,
                            "title": display_title,
                            "turn_id": turn_id,
                            "status": kwargs.get("status") or metadata.get("status") or "running",
                            **kwargs,
                        }
                        from mana_agent.services.execution_event_hub import get_execution_event_hub
                        from mana_agent.workspaces.paths import repository_id_for_path
                        get_execution_event_hub().publish(event_payload, conversation_id=sid, execution_id=turn_id, repository_id=repository_id_for_path(self.repo_root))
                        try:
                            if hasattr(self, "call_from_thread"):
                                self.call_from_thread(self.update_status, f"{display_title}…")
                            else:
                                self.update_status(f"{display_title}…")
                        except Exception:
                            pass

                    def _run_gateway_turn() -> Any:
                        return self.gateway.process_turn(
                            sid,
                            question,
                            model=self.model,
                            index_dir=self.index_dir,
                            index_dirs=self.index_dirs,
                            callbacks=ask_callbacks or None,
                            turn_id=turn_id,
                            user_message_id=user_event.event_id,
                            event_sink=_on_gateway_sink_event,
                            attachments=list(getattr(user_event, "attachments", ()) or []),
                        )

                    tools_before = self._count_tool_events_for_turn(turn_id)
                    self.update_status("Routing…")
                    from mana_agent.coding.live_events import coding_event_scope

                    with coding_event_scope(_on_coding_event):
                        try:
                            result = await asyncio.to_thread(_run_gateway_turn)
                        except Exception as exc:
                            self._safe_post_activity(
                                {
                                    "event_type": "error",
                                    "title": "Execution failed",
                                    "status": "failed",
                                    "error": str(exc),
                                    "detail": str(exc),
                                },
                                turn_id,
                            )
                            raise
                    answer = str(getattr(result, "answer", "") or "")
                    result_payload = dict(getattr(result, "payload", {}) or {})
                    if getattr(result, "error", None) and not answer:
                        self._safe_post_activity(
                            {
                                "event_type": "error",
                                "title": "Execution failed",
                                "status": "failed",
                                "error": str(result.error),
                                "detail": str(result.error),
                            },
                            turn_id,
                        )
                    if str(getattr(result, "mode", "") or "") == "command":
                        active_session_id = str(result_payload.get("session_id") or "")
                        if active_session_id:
                            self._gateway_session_id = active_session_id
                        command_result = dict(result_payload.get("command_result") or {})
                        for command_event in command_result.get("events") or []:
                            if command_event.get("type") == "timeline.replace":
                                self._replace_timeline(command_event.get("messages") or [])
                    if getattr(result, "error", None) and not answer:
                        answer = f"(Gateway error: {result.error})"
                    flow_id = getattr(result, "flow_id", None)
                    if isinstance(flow_id, str) and flow_id.strip():
                        self.active_flow_id = flow_id.strip()

                    routing = dict((getattr(result, "payload", {}) or {}).get("routing_decision") or {})
                    if routing:
                        self.update_status(
                            f"{routing.get('provider')}/{routing.get('model')} · "
                            f"{routing.get('routing_mode', 'single')} · "
                            f"{float(routing.get('confidence') or 0):.0%}"
                        )

                    # Surface tool/mode hints when auto-chat (e.g. gmail) was used
                    route_mode = str(getattr(result, "mode", "") or "")
                    auto_mode = str(getattr(result, "auto_chat_mode", "") or "")
                    if auto_mode and not getattr(result, "used_coding_agent", False):
                        self.update_status(f"Auto-chat ({auto_mode})")
                    elif getattr(result, "used_coding_agent", False):
                        self.update_status("Coding agent")

                    # Live callbacks should have painted tools; if not, replay
                    # serialized traces from the auto-chat AskAgent response.
                    if self._count_tool_events_for_turn(turn_id) <= tools_before:
                        self._replay_tool_traces_from_result(result, turn_id=turn_id)

                    if answer or str(getattr(result, "mode", "") or "") != "command":
                        self.history.add(
                            AssistantMessageEvent(
                                content=answer or "(No response)",
                                turn_id=turn_id,
                            )
                        )
                    next_prompt = result_payload.get("next_prompt")
                    self._show_mode_prompt(next_prompt)
                    if (next_prompt or {}).get("kind") == "mode.clarification":
                        self.update_status("Waiting for your answers")
                    elif not route_mode:
                        self.update_status("Ready")
                    else:
                        self.update_status(f"Ready ({route_mode})")
                    return
                except Exception as exc:
                    answer = f"Gateway execution failed: {exc}. No direct model fallback was executed."
                    self.history.add(AssistantMessageEvent(
                        content=answer,
                        turn_id=turn_id,
                    ))
                    self.update_status("Gateway routing failed")
                    return

            if not answer and self.coding_agent is not None:
                try:
                    idx = self.index_dir or str(self.repo_root)
                    payload = await asyncio.to_thread(
                        self.coding_agent.generate,
                        question,
                        index_dir=idx,
                        k=self.resolved_k,
                        max_steps=self.coding_agent_max_steps,
                        timeout_seconds=min(max(self.agent_timeout_seconds, 60), 600),
                    )
                    answer = self._extract_answer(payload) or str(payload)
                    used_full_flow = True
                except Exception:
                    pass

            if not answer and self.tools_orchestrator is not None:
                try:
                    if hasattr(self.tools_orchestrator, "run"):
                        result = await asyncio.to_thread(self.tools_orchestrator.run, question)
                        extracted = self._extract_answer(result)
                        if extracted:
                            answer = extracted
                            used_full_flow = True
                except Exception:
                    pass

            if not answer and self.chat_service is not None:
                try:
                    if hasattr(self.chat_service, "ask"):
                        resp = await asyncio.to_thread(self.chat_service.ask, question)
                        extracted = self._extract_answer(resp)
                        if extracted:
                            answer = extracted
                            used_full_flow = True
                except Exception:
                    pass

            if not answer:
                try:
                    from mana_agent.config.settings import Settings
                    from mana_agent.commands.cli_internal import _build_ask_service_compat
                    from mana_agent.context_cost.governor import ContextCostGovernor

                    settings = Settings()
                    governor = ContextCostGovernor(
                        session_id=str(self._gateway_session_id or f"tui-{id(self)}"),
                        settings=settings,
                    )
                    ask = _build_ask_service_compat(
                        settings,
                        model_override=self.model,
                        context_cost_governor=governor,
                        project_root=self.repo_root,
                    )
                    if hasattr(ask, "ask_with_tools"):
                        resp = await asyncio.to_thread(
                            ask.ask_with_tools,
                            index_dir=str(self.repo_root),
                            question=question,
                            k=self.resolved_k or 6,
                        )
                        answer = self._extract_answer(resp) or str(resp)
                    else:
                        resp = await asyncio.to_thread(ask.ask, str(self.repo_root), question, self.resolved_k or 6)
                        answer = self._extract_answer(resp) or str(resp)
                except Exception:
                    pass

            if not answer:
                try:
                    answer = await self._call_llm(question, extra_context=str(ctx_result))
                except Exception as exc:
                    answer = f"Understood: {question[:80]}. (All paths failed: {exc})"
        finally:
            tool_scope.__exit__(None, None, None)

        if not answer:
            answer = f"Understood: {question[:80]}. (No answer produced.)"

        if used_full_flow:
            self.update_status("Ready (real multi-agent tools)")

        # Final assistant answer streamed (tokens appear in chat log)
        assistant = AssistantMessageEvent(
            content="",
            is_streaming=True,
            turn_id=turn_id,
        )
        self.history.add(assistant)
        await asyncio.sleep(0)

        for i, tok in enumerate(answer.split(" ")):
            self.history.add(StreamTokenEvent(
                token=tok + " ",
                assistant_event_id=assistant.event_id,
                turn_id=turn_id,
            ))
            if i % 5 == 0:
                await asyncio.sleep(0.01)

        self.history.add(AssistantMessageEvent(
            content=answer,
            is_streaming=False,
            turn_id=turn_id,
        ))

        self.token_count += max(1, len(answer.split()))
        self.update_status("Ready")

    async def _call_llm(self, question: str, extra_context: str = "") -> str:
        """Use the project's chat model (with proper credentials from CLI launch).

        This now correctly passes api_key/base_url like the rest of the agent system.
        For full multi-agent flow (routing, coding agent, full tool registry, auto-execute,
        memory, verification) the TUI should be passed the orchestrator/coding_agent
        objects (future enhancement). Currently uses direct model + context for responses.
        """
        try:
            from mana_agent.llm.engine import create_model

            model_name = self.model or "gpt-4o-mini"
            from mana_agent.config.settings import Settings

            settings = Settings()
            from mana_agent.config.inference_provider import (
                ProviderConfigurationError,
                resolve_inference_connection,
            )

            try:
                connection = resolve_inference_connection(settings)
            except ProviderConfigurationError as exc:
                raise RuntimeError(str(exc)) from exc
            api_key = self.api_key or connection.api_key
            base_url = self.base_url or connection.base_url

            if not api_key:
                raise RuntimeError(
                    f"No {connection.display_name} API key is saved in ~/.mana/secrets.toml"
                )

            llm = create_model(
                api_key=api_key,
                model=model_name,
                base_url=base_url,
                provider=connection.provider,
                default_headers=connection.headers,
            )

            from mana_agent.spirit.adapter import apply_spirit_instruction

            system = apply_spirit_instruction(
                "You are a helpful repository-aware coding assistant called mana-agent. "
                "Answer concisely and usefully. Use any provided repo context. "
                "If edits or tools would help, describe what you would do.",
                agent_name="tui-agent",
                agent_role="main",
                provider=connection.provider,
                model=model_name,
            )
            user_content = f"Repo context: {extra_context}\n\nUser question: {question}"

            # Run off the event loop (the model invoke may be sync)
            response = await asyncio.to_thread(
                llm.generate_sync,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
            )

            content = getattr(response, "content", None) or str(response)
            return str(content).strip()
        except Exception as exc:
            # Helpful message instead of generic "llm unavailable"
            return (
                f"I understood the request: {question[:100]}.\n\n"
                "The TUI is connected to the mana-agent environment, but the direct LLM call failed.\n"
                f"Details: {type(exc).__name__}: {exc}\n\n"
                "To use the full multi-agent flow (routing via route_for_turn, CodingAgent, "
                "proper tool execution via QueueManager/ToolsManager, memory, planning, verification, etc.) "
                "like the classic `mana-agent chat`, we need to pass the prepared agent objects "
                "from chat_cli.py into the TUI handler (in progress).\n\n"
                "For now, basic responses + visible tool cards work when credentials are present."
            )

    def action_quit(self) -> None:
        if self.gateway is not None and hasattr(self.gateway, "request_shutdown"):
            self.gateway.request_shutdown(source="tui_quit", session_id=self._gateway_session_id)
        elif self.gateway is not None and hasattr(self.gateway, "close_session"):
            self.gateway.close_session(self._gateway_session_id)
        self.exit()

    def on_unmount(self) -> None:
        """Use the same idempotent finalization path for all TUI shutdowns."""
        if self._unsubscribe_computer_permissions is not None:
            self._unsubscribe_computer_permissions()
            self._unsubscribe_computer_permissions = None
        if self._unsubscribe_session_events is not None:
            self._unsubscribe_session_events()
            self._unsubscribe_session_events = None
        self._unsubscribe_api_approval_events = None
        if self.gateway is not None and hasattr(self.gateway, "request_shutdown"):
            self.gateway.request_shutdown(source="tui_quit", session_id=self._gateway_session_id)
        elif self.gateway is not None and hasattr(self.gateway, "close_session"):
            self.gateway.close_session(self._gateway_session_id)


    def watch_status_text(self, value: str) -> None:
        self.refresh_footer()

    def watch_token_count(self, value: int) -> None:
        self.refresh_footer()


# Convenience launcher
def run_chat_tui(
    history: ChatHistory | None = None,
    *,
    repo_root: str | Path | None = None,
    model: str | None = None,
    initial_prompt: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    chat_service: Any = None,
    coding_agent: Any = None,
    tools_orchestrator: Any = None,
    # Extended context for 100% parity with console coding agent behavior
    dir_mode: bool = False,
    index_dir: str | Path | None = None,
    index_dirs: list[str | Path] | None = None,
    auto_execute_plan: bool = True,
    auto_execute_max_passes: int = 4,
    coding_agent_max_steps: int = 200,
    resolved_k: int = 6,
    agent_timeout_seconds: int = 600,
    # Gateway connection (passed through from chat_cli)
    gateway: Any = None,
) -> None:
    """Launch the TUI as the default chat experience.

    Connected to the real multi-agent flow (coding_agent, tools_orchestrator,
    chat_service) when launched from chat_cli.py. All coding-agent control
    flags are forwarded so generate/generate_dir_mode/generate_auto_execute
    are invoked with exactly the same arguments as the classic console path.

    When a gateway is supplied, the TUI has an explicit connection object
    to the agent runtime ("tui chat connect with gateway to agent").
    """
    app = ManaChatApp(
        history=history,
        repo_root=repo_root,
        model=model,
        initial_prompt=initial_prompt,
        api_key=api_key,
        base_url=base_url,
        chat_service=chat_service,
        coding_agent=coding_agent,
        tools_orchestrator=tools_orchestrator,
        dir_mode=dir_mode,
        index_dir=index_dir,
        index_dirs=index_dirs,
        auto_execute_plan=auto_execute_plan,
        auto_execute_max_passes=auto_execute_max_passes,
        coding_agent_max_steps=coding_agent_max_steps,
        resolved_k=resolved_k,
        agent_timeout_seconds=agent_timeout_seconds,
        gateway=gateway,
    )
    app.run()


if __name__ == "__main__":
    run_chat_tui()
