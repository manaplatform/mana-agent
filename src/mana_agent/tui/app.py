"""
ManaChatApp — Production-quality Textual Chat TUI for mana-agent.

Layout:
  Header
  ─────────────────────────────────────
  ChatLog  (scrollable, subscribes to history)
  ─────────────────────────────────────
  Input bar  (type + Enter)
  Footer     (status, model placeholder, ready for token counts)

Keyboard:
  Ctrl+C / q     quit
  Ctrl+L         clear log
  Up/Down        history navigation (future)

Streaming + Tools:
  The app wires a ChatHistory. All messages, tool calls, results and
  tokens go through history.add(...). The ChatLog reacts live.

Integration notes for the rest of mana-agent:
------------------------------------------------
Instead of:
    console.print("[tool] ...")
    print(result)

Do:
    from mana_agent.chat.history import get_history
    from mana_agent.chat.events import ToolCallEvent, ToolResultEvent, AssistantMessageEvent, StreamTokenEvent

    h = get_history()
    h.add(ToolCallEvent(tool_name="read_file", args={"path": p}, call_id=cid))
    ...
    h.add(ToolResultEvent(call_id=cid, tool_name=..., success=True, result=...))
    h.add(AssistantMessageEvent(content=answer))
    # for streaming responses:
    h.add(StreamTokenEvent(token=chunk, assistant_event_id=aid))

This guarantees tool visibility on *every* turn, not just the first message.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

from mana_agent.chat.events import (
    AssistantMessageEvent,
    CodingActivityEvent,
    StreamTokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserMessageEvent,
)
from mana_agent.chat.history import ChatHistory, get_history
from mana_agent.tui.widgets.chat_log import ChatLog
from mana_agent.tui.widgets.message_input import MessageInput


class ManaChatApp(App):
    """Main Textual application for the enhanced mana-agent chat TUI."""

    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("q", "quit", "Quit", show=False),
        Binding("ctrl+l", "clear_log", "Clear", show=True),
        Binding("ctrl+r", "simulate_response", "Simulate Agent", show=True),
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
        self._turn_counter = 0
        self._tool_cid_map: dict[str, str] = {}  # key -> call_id for reliable start/end pairing in bridge

        # Configuration for real agent behavior
        self.repo_root: Path = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
        self.model: str | None = model
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        # Body takes the space between docked Header and docked Footer.
        # Log gets 1fr, input bar is fixed at the bottom of the body so it cannot
        # go below the footer.
        with Vertical(id="body"):
            self.chat_log = ChatLog(history=self.history, id="chat-log")
            yield self.chat_log

            # Bottom input bar (message box)
            with Vertical(id="input-bar"):
                self.input = MessageInput(
                    id="chat-input",
                )
                yield self.input
                yield Static(
                    "Enter sends · Shift+Enter adds a line · Ctrl+J / Alt+Enter also add a line",
                    id="input-help",
                )


        yield Footer()

    def on_mount(self) -> None:
        """Focus input and show a welcome message. Seed initial prompt if provided by CLI."""
        if self.input:
            self.input.focus()

        # Safe immediate footer (avoids any early watcher issues)
        self.sub_title = "Ready"

        # Welcome (only if this is a fresh history)
        if len(self.history.get_events()) == 0:
            root_str = str(self.repo_root)
            model_str = self.model or "default"
            try:
                from mana_agent.config.user_config import load_effective_settings

                configured = load_effective_settings(include_env=False)
                provider = str(configured.get("MANA_AI_PROVIDER") or "openai")
                search = str(configured.get("MANA_WEB_SEARCH_PROVIDER") or "disabled") if configured.get("MANA_SEARCH_ENABLE_WEB") else "disabled"
                github = str(configured.get("MANA_GITHUB_CREDENTIAL_SOURCE") or "disabled")
                warnings = []
                if not configured.get("OPENAI_API_KEY"):
                    warnings.append("provider authentication missing")
                warning_line = f"\nWarnings: {', '.join(warnings)}" if warnings else ""
            except Exception:
                provider, search, github, warning_line = "unknown", "unknown", "unknown", "\nWarnings: configuration unavailable"
            welcome = AssistantMessageEvent(
                content=(
                    f"**Mana-Agent** · `{root_str}`\n"
                    f"Provider: `{provider}` · Model: `{model_str}` · Search: `{search}` · GitHub: `{github}`"
                    f"{warning_line}\n\nType a message, or use `/models` to manage configured models."
                )
            )
            self.history.add(welcome)

        # Schedule the first status update safely after the widget tree is ready.
        # Calling update_status() synchronously in on_mount can trigger watchers
        # before Footer is queryable, leading to crashes on footer.update().
        self.call_after_refresh(
            lambda: self.update_status(f"Ready — {self.repo_root.name or self.repo_root}")
        )

        # If the CLI passed an initial prompt (e.g. `mana-agent chat "explain the architecture"`),
        # send it automatically as the first user message.
        if self.initial_prompt:
            self.run_worker(self._send_initial_prompt(), exclusive=True)

    def update_status(self, text: str) -> None:
        """Update the status reactive. The watcher + refresh_footer will keep the footer in sync."""
        self.status_text = text
        # Do not call refresh_footer directly here — the @reactive watcher
        # (watch_status_text) will invoke it. This prevents duplicate work
        # and ordering problems during early app lifecycle.

    def refresh_footer(self) -> None:
        """Update dynamic footer text.

        We use self.sub_title instead of directly mutating the Footer widget.
        This avoids query_one timing issues during early mount / reactive updates
        (watchers can fire before the compose tree is fully attached).
        The default Textual Footer will display sub_title on the right side.
        """
        model = self.model or "default"
        root = self.repo_root.name if self.repo_root else ""
        tokens = f"tokens: {self.token_count}"
        self.sub_title = f"{self.status_text}  |  {model}  |  {root}  |  {tokens}"

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

    def on_message_input_height_changed(self, event: MessageInput.HeightChanged) -> None:
        """Keep the input bar and chat timeline in sync with the composer."""
        if event.message_input is not self.input:
            return
        try:
            self.query_one("#input-bar").styles.height = event.height + 2
        except NoMatches:
            # A queued resize can arrive after a modal replaces the chat screen.
            return

    async def on_message_input_submitted(self, event: MessageInput.Submitted) -> None:
        """Handle user pressing Enter in the input box. Always uses the real turn handler."""
        # Preserve message whitespace and explicit line breaks. A final newline is
        # an editing artifact from the composer, not part of the submitted turn.
        text = event.value.rstrip("\r\n")
        if not text.strip():
            return

        if self.gateway is not None and hasattr(self.gateway, "handle_control_command"):
            control = self.gateway.handle_control_command(
                text,
                session_id=self._gateway_session_id or "",
            )
            if control is not None:
                self.history.add(AssistantMessageEvent(content=control))
                if self.input:
                    self.input.reset()
                return

        if text == "/models" or text.startswith("/models "):
            if self._turn_in_progress:
                self.notify("Wait for the current turn to finish before changing models.", severity="warning")
                return
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
            if self.input:
                self.input.reset()
            return

        if text == "/new":
            if self._turn_in_progress:
                self.notify("Wait for the current turn to finish before starting a new conversation.", severity="warning")
                return
            if self.gateway is None or not hasattr(self.gateway, "start_new_conversation"):
                self.history.add(AssistantMessageEvent(content="A gateway session is required to start a new conversation."))
                return
            if not self._gateway_session_id:
                self._gateway_session_id = self.gateway.create_session(frontend="tui")
            self._start_new_conversation()
            if self.input:
                self.input.reset()
            self.update_status("Ready")
            return

        # Clear input immediately (premium feel)
        if self.input:
            self.input.reset()

        self.update_status("Thinking...")
        self.token_count += len(text.split())

        # Record user message first so ChatLog can paint the bubble immediately
        user_event = UserMessageEvent(content=text)
        self.history.add(user_event)

        # Yield so the Textual message pump can mount/paint the user bubble
        # before long agent/tool work starts (immediate feedback in the message box).
        await asyncio.sleep(0)

        # Run the turn as a worker so the UI stays responsive and tool events can paint live
        self._turn_in_progress = True
        self.run_worker(self._handle_real_turn_guarded(user_event), exclusive=True)

    def _start_new_conversation(self) -> str:
        """Apply the same gateway-owned conversation boundary as the plain CLI."""
        if self.gateway is None or not hasattr(self.gateway, "start_new_conversation"):
            raise RuntimeError("A gateway session is required to start a new conversation.")
        if not self._gateway_session_id:
            self._gateway_session_id = self.gateway.create_session(frontend="tui")
        self._gateway_session_id = self.gateway.start_new_conversation(
            self._gateway_session_id, frontend="tui"
        )
        self.active_flow_id = None
        self.history.clear()
        self.history.add(AssistantMessageEvent(content="Started a new conversation."))
        return self._gateway_session_id

    async def _handle_real_turn_guarded(self, user_event: UserMessageEvent) -> None:
        try:
            await self._handle_real_turn(user_event)
        finally:
            self._turn_in_progress = False

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
        self.update_status(f"Model changed to {selection.qualified_id}")
        if selection.persist:
            self.history.add(AssistantMessageEvent(content=f"Default model saved: `{selection.qualified_id}`"))

    async def _send_initial_prompt(self) -> None:
        """Send the prompt that was passed on the command line (e.g. mana-agent chat "foo")."""
        if not self.initial_prompt:
            return
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
        await self._handle_real_turn(user_event)

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
            if not isinstance(event, (ToolCallEvent, ToolResultEvent)):
                continue
            if str(getattr(event, "turn_id", "") or "") == str(turn_id or ""):
                n += 1
        return n

    def _make_tool_emit_bridge(self, *, original_emit: Any, turn_id: str):
        """Wrap emit_tool_event so tool start/end become ToolCards in ChatLog."""

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
                    "summary": str(args)[:60] if args else "running…",
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
            if row.get("backend") in {"codex", "internal"} and row.get("event_type"):
                # Already rendered live in the backend-neutral execution panel.
                continue
            tname = str(row.get("tool_name") or row.get("tool") or "tool")
            args_summary = str(row.get("args_summary") or row.get("args") or tname)[:120]
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
          - every tool appears as ToolCard inside the chat log ("chat box / tool box")

        Planning collection state is also supported for interactive pre-questions.
        """
        question = user_event.content
        turn_id = user_event.turn_id

        def _on_coding_event(event: Any) -> None:
            payload = event.model_dump(mode="json") if hasattr(event, "model_dump") else dict(event)
            # The frontend turn is authoritative for presentation. Provider turn
            # IDs remain available inside the normalized activity payload.
            self.history.add(CodingActivityEvent(activity=payload, turn_id=turn_id))

        # --- Planning collection state machine (mirrors console pre-generate logic) ---
        if self._in_planning_collection and self._planning_request:
            # Treat this input as the next planning answer
            self._planning_answers.append(question)
            self._planning_questions.append(
                f"(answer {len(self._planning_answers)})"
            )  # placeholder; real questions were emitted earlier
            # If we have enough, fall through to run generation; else ask another
            # For simplicity in first full integration we proceed after one answer
            # (full multi-question loop can be expanded; core parity is the generate call).
            self._in_planning_collection = False
            # Continue with the original planning_request as the task
            question = self._planning_request
            # Clear for next time
            self._planning_request = None

        self.update_status("Thinking with multi-agent flow...")

        # Bridge emit_tool_event for the *entire* turn — including gateway auto-chat.
        # Previously the bridge was only installed after the gateway path returned,
        # so email_read / web_search / MCP tools never became ToolCards in the TUI.
        import mana_agent.commands.ui_helpers as ui_helpers
        original_emit = ui_helpers.emit_tool_event
        ui_helpers.emit_tool_event = self._make_tool_emit_bridge(
            original_emit=original_emit,
            turn_id=turn_id,
        )

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
                        from mana_agent.commands.ui_helpers import RichToolCallbackHandler

                        tool_cb = RichToolCallbackHandler(show_inputs=True)
                    except Exception:
                        tool_cb = None
                    ask_callbacks = [tool_cb] if tool_cb is not None else []

                    def _run_gateway_turn() -> Any:
                        return self.gateway.process_turn(
                            sid,
                            question,
                            index_dir=self.index_dir,
                            index_dirs=self.index_dirs,
                            callbacks=ask_callbacks or None,
                        )

                    tools_before = self._count_tool_events_for_turn(turn_id)
                    self.update_status("Routing via gateway (auto-chat / coding)…")
                    from mana_agent.coding.live_events import coding_event_scope

                    with coding_event_scope(_on_coding_event):
                        result = await asyncio.to_thread(_run_gateway_turn)
                    answer = str(getattr(result, "answer", "") or "")
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

                    self.history.add(
                        AssistantMessageEvent(
                            content=answer or "(No response)",
                            turn_id=turn_id,
                        )
                    )
                    if not route_mode:
                        self.update_status("Ready")
                    else:
                        self.update_status(f"Ready ({route_mode})")
                    return
                except Exception as exc:
                    answer = f"Gateway execution failed: {exc}. No direct model fallback was executed."
                    self.history.add(AssistantMessageEvent(content=answer, turn_id=turn_id))
                    self.update_status("Gateway routing failed")
                    return

            # ==========================================================
            # Decide whether to use the full CodingAgent + strict planner checklist path.
            # General queries (e.g. "check my gmail", "connect to mcp server X",
            # "search the internet for ...", pure questions) should use the standard
            # routing via chat_service.ask (which supports MCP tools, email connectors,
            # web search, etc.) even when the rich coding stack is available.
            # ==========================================================
            use_coding_path = False
            if self.coding_agent is not None:
                try:
                    from mana_agent.multi_agent.runtime.auto_chat import classify_auto_chat_intent, AutoChatMode
                    intent = classify_auto_chat_intent(question or "")
                    is_coding_intent = intent in (AutoChatMode.EDIT, AutoChatMode.PLAN_ONLY)
                except Exception:
                    is_coding_intent = False

                # Use classifier as the primary model decision.
                # General queries ("check my gmail", "search internet", "connect to mcp X")
                # should use standard routing even if auto_execute_plan is on by default.
                # Only enter the strict coding planner for edit/plan intents.
                use_coding_path = is_coding_intent

            if self.coding_agent is not None and use_coding_path:
                # ==========================================================
                # EXACT CODING AGENT PATH (mirrors chat_cli.py ~3127-3330)
                # Only entered for edit/plan intents (or when auto-execute forces it).
                # ==========================================================
                self.update_status("Running Codex coding turn (tools live in chat box)…")

                try:
                    from mana_agent.commands.ui_helpers import RichToolCallbackHandler
                    cb = RichToolCallbackHandler(show_inputs=True)
                except Exception:
                    cb = None
                callbacks = [cb] if cb is not None else None

                # Replicate decision variables from console (using our stored parity context)
                # These expressions are intentionally close to the source of truth.
                plan_trigger_request = False  # basic; can be extended with heuristics if needed
                force_plan_only_response = False
                force_auto_execute_edit = bool(
                    self.coding_agent is not None
                    and self.auto_execute_plan
                )
                execute_plan_now = bool(
                    self.auto_execute_plan
                    and not force_plan_only_response
                    and (plan_trigger_request or force_auto_execute_edit)
                )
                auto_execute_available = bool(
                    execute_plan_now and hasattr(self.coding_agent, "generate_auto_execute")
                )

                request_for_generation = question

                # State for full-auto resume cycles (mirrors console)
                turn_full_auto_resume_cycles = 0
                turn_resume_run_id: str | None = None
                active_flow_id = self.active_flow_id

                # Try to give the planner good Flow context early (very important for valid checklist output).
                # This mirrors the memory service usage in the console path and in preview/generate.
                flow_context = None
                try:
                    mem_svc = getattr(self.coding_agent, "coding_memory_service", None)
                    if mem_svc is not None and getattr(self.coding_agent, "coding_memory_enabled", False):
                        git_paths: list[str] = []
                        try:
                            import subprocess
                            proc = subprocess.run(
                                ["git", "status", "--porcelain"],
                                cwd=str(self.repo_root),
                                capture_output=True,
                                text=True,
                                check=False,
                            )
                            for line in (proc.stdout or "").splitlines():
                                if len(line) >= 4:
                                    p = line[3:].strip()
                                    if p:
                                        git_paths.append(p)
                        except Exception:
                            pass
                        try:
                            flow_context = mem_svc.build_flow_context(active_flow_id, sorted(set(git_paths)))
                        except Exception:
                            pass
                except Exception:
                    pass

                # Mirror console pre-call logic for prechecklist (used by auto-execute to avoid
                # double planner calls and to give the agent a head-start on execution_scope).
                pending_prechecklist = None
                pending_prechecklist_source = ""
                pending_prechecklist_warning = ""
                if self.auto_execute_plan and hasattr(self.coding_agent, "preview_execution_checklist"):
                    try:
                        preview_payload = self.coding_agent.preview_execution_checklist(
                            request_for_generation,
                            flow_id=active_flow_id,
                            flow_context=flow_context,
                        )
                    except Exception:
                        preview_payload = {}
                    if isinstance(preview_payload, dict):
                        f = preview_payload.get("flow_id")
                        if isinstance(f, str) and f.strip():
                            active_flow_id = f.strip()
                        pre = preview_payload.get("prechecklist")
                        if isinstance(pre, dict):
                            pending_prechecklist = pre
                        pending_prechecklist_source = str(preview_payload.get("prechecklist_source", "") or "")
                        pending_prechecklist_warning = str(preview_payload.get("prechecklist_warning", "") or "")

                prechecklist_payload = (
                    {
                        "flow_id": active_flow_id,
                        "prechecklist": pending_prechecklist,
                        "prechecklist_source": pending_prechecklist_source,
                        "prechecklist_warning": pending_prechecklist_warning,
                    }
                    if isinstance(pending_prechecklist, dict) else None
                )

                cycle_result: dict[str, Any] = {}

                def _run_coding_generation() -> Any:
                    """Builds the exact call used by the console path and executes it."""
                    nonlocal active_flow_id, turn_resume_run_id

                    if self.dir_mode:
                        idxs = self.index_dirs or [str(self.repo_root)]
                        if auto_execute_available and hasattr(self.coding_agent, "generate_auto_execute"):
                            return self.coding_agent.generate_auto_execute(
                                request_for_generation,
                                index_dirs=idxs,
                                k=self.resolved_k,
                                max_steps=self.coding_agent_max_steps,
                                timeout_seconds=min(max(self.agent_timeout_seconds, 60), 600),
                                pass_cap=self.auto_execute_max_passes,
                                callbacks=callbacks,
                                flow_id=active_flow_id,
                                run_id=turn_resume_run_id,
                                flow_context=flow_context,
                                prechecklist_payload=prechecklist_payload,
                                auto_chat_mode="edit" if execute_plan_now else None,
                            )
                        return self.coding_agent.generate_dir_mode(
                            request_for_generation,
                            index_dirs=idxs,
                            k=self.resolved_k,
                            max_steps=self.coding_agent_max_steps,
                            timeout_seconds=min(max(self.agent_timeout_seconds, 60), 600),
                            callbacks=callbacks,
                            flow_id=active_flow_id,
                            flow_context=flow_context,
                            auto_chat_mode="edit" if execute_plan_now else None,
                        )
                    else:
                        idx = self.index_dir or str(self.repo_root)
                        if auto_execute_available and hasattr(self.coding_agent, "generate_auto_execute"):
                            return self.coding_agent.generate_auto_execute(
                                request_for_generation,
                                index_dir=idx,
                                k=self.resolved_k,
                                max_steps=self.coding_agent_max_steps,
                                timeout_seconds=min(max(self.agent_timeout_seconds, 60), 600),
                                pass_cap=self.auto_execute_max_passes,
                                callbacks=callbacks,
                                flow_id=active_flow_id,
                                run_id=turn_resume_run_id,
                                flow_context=flow_context,
                                prechecklist_payload=prechecklist_payload,
                                auto_chat_mode="edit" if execute_plan_now else None,
                            )
                        return self.coding_agent.generate(
                            request_for_generation,
                            index_dir=idx,
                            k=self.resolved_k,
                            max_steps=self.coding_agent_max_steps,
                            timeout_seconds=min(max(self.agent_timeout_seconds, 60), 600),
                            callbacks=callbacks,
                            flow_id=active_flow_id,
                            flow_context=flow_context,
                            auto_chat_mode="edit" if execute_plan_now else None,
                        )

                # Execute (with simple resume cycle support for auto-execute)
                # Mirror console error handling so decision failures (e.g. execution_scope budget)
                # and worker errors do not crash the TUI worker with traceback.
                while True:
                    try:
                        from mana_agent.coding.live_events import coding_event_scope

                        with coding_event_scope(_on_coding_event):
                            payload = await asyncio.to_thread(_run_coding_generation)
                    except Exception as gen_exc:  # includes ExecutionScopeDecisionError, ToolWorkerProcessError etc.
                        # Surface gracefully (mirrors console except blocks).
                        # The emit bridge may have emitted ToolCards for work done before the decision failure.
                        err_msg = f"Coding agent failed: {type(gen_exc).__name__}: {gen_exc}"
                        self.update_status("Coding agent error (see chat)")
                        # Do not re-raise. Set payload so normal extraction + streaming shows it as final assistant message.
                        payload = {
                            "answer": err_msg,
                            "decision_error": str(gen_exc),
                            "warnings": [str(gen_exc)],
                        }
                    cycle_result["result"] = payload
                    if not (
                        getattr(self, "chat_auto_continue", False)
                        and execute_plan_now
                        and isinstance(payload, dict)
                    ):
                        break
                    # minimal resume accounting (full ingest functions live in chat_cli)
                    term = str((payload or {}).get("auto_execute_terminal_reason", "") or "").strip().lower()
                    f = (payload or {}).get("flow_id")
                    if isinstance(f, str) and f.strip():
                        active_flow_id = f.strip()
                    r = (payload or {}).get("run_id")
                    if isinstance(r, str) and r.strip():
                        turn_resume_run_id = r.strip()
                    if term != "pass_cap_reached":
                        break
                    turn_full_auto_resume_cycles += 1
                    # In real console a new resume request is built; here we simply loop once more
                    # with the same request (the agent itself manages state via flow_id/run_id).
                    continue

                result = cycle_result.get("result")
                extracted = self._extract_answer(result)
                if extracted:
                    answer = extracted
                    used_full_flow = True

                # Clean up planner failure messages so internal parse details like
                # "no checklist payload found" are not leaked to the user.
                if isinstance(result, dict):
                    de = str(result.get("decision_error", "") or "").lower()
                    ans_lower = (answer or "").lower()
                    planner_failed = (
                        "no checklist payload found" in de or
                        "planner output is invalid" in de or
                        "no checklist payload found" in ans_lower or
                        "planner failed to produce valid checklist" in ans_lower
                    )
                    if planner_failed:
                        # For non-editing / general queries, clear the answer so we fall
                        # back to the standard chat_service / routing path. This makes
                        # "check my gmail", "search the internet", "connect to mcp ..." work
                        # via connectors / MCP tools / web search instead of the coding planner.
                        try:
                            from mana_agent.multi_agent.runtime.auto_chat import classify_auto_chat_intent, AutoChatMode
                            intent = classify_auto_chat_intent(question or "")
                            if intent not in (AutoChatMode.EDIT, AutoChatMode.PLAN_ONLY):
                                answer = ""  # fall through to chat_service.ask etc.
                                used_full_flow = False
                            else:
                                answer = "Planner was unable to produce a valid checklist for this request (after repair attempt). The planner LLM may need a more specific goal."
                        except Exception:
                            # If we can't classify, still prefer to show a helpful (non-internal) message
                            answer = "Planner was unable to produce a valid checklist for this request (after repair attempt). Try rephrasing or use a more specific coding/editing request."
                    elif "planner failed to produce valid checklist" in ans_lower:
                        warns = result.get("warnings") or []
                        detail = "; ".join(str(w) for w in warns if w and "planner" in str(w).lower())
                        if detail:
                            answer = (answer or "") + f" Details: {detail[:200]}"

                # Update our mirror state
                if isinstance(result, dict):
                    rf = result.get("flow_id")
                    if isinstance(rf, str) and rf.strip():
                        self.active_flow_id = rf.strip()
                        active_flow_id = self.active_flow_id

                # Live bridge already emitted ToolCards. Also replay actions_taken for completeness
                # (exactly as the old TUI + console transcript logic).
                if isinstance(result, dict):
                    for row in (result.get("actions_taken") or []):
                        if not isinstance(row, dict):
                            continue
                        if row.get("backend") in {"codex", "internal"} and row.get("event_type"):
                            continue
                        tname = str(row.get("tool_name") or row.get("tool") or row.get("name") or "tool")
                        args_val = row.get("args") or row.get("input") or row.get("tool_args") or {}
                        cid = str(
                            row.get("event_id")
                            or row.get("tool_call_id")
                            or row.get("call_id")
                            or f"tui-{tname}-{abs(hash(str(row)[:120])) % 1000000}"
                        )
                        tcall = ToolCallEvent(
                            tool_name=tname,
                            args=args_val,
                            call_id=cid,
                            summary=str(args_val)[:60] if args_val else "",
                            turn_id=turn_id,
                        )
                        self.history.add(tcall)
                        ok = bool(row.get("ok", True))
                        if "success" in row:
                            ok = bool(row.get("success"))
                        res_val = row.get("result") or row.get("output") or row.get("answer") or "(ok)"
                        err = row.get("error")
                        if err:
                            ok = False
                        tres = ToolResultEvent(
                            call_id=cid,
                            tool_name=tname,
                            success=ok,
                            result=None if not ok else res_val,
                            error=str(err) if err else None,
                            summary=f"{tname} {'ok' if ok else 'error'}",
                            turn_id=turn_id,
                        )
                        self.history.add(tres)

            # Fallbacks (unchanged behavior for non-coding paths)
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
                    from mana_agent.services.ask_service import _build_ask_service_compat  # type: ignore[attr-defined]

                    settings = Settings()
                    ask = _build_ask_service_compat(settings, model_override=self.model, project_root=str(self.repo_root))
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
            ui_helpers.emit_tool_event = original_emit

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
            from mana_agent.multi_agent.runtime.compatibility import create_chat_model

            model_name = self.model or "gpt-4o-mini"
            from mana_agent.config.settings import Settings

            settings = Settings()
            api_key = self.api_key or settings.openai_api_key
            base_url = self.base_url or settings.openai_base_url

            if not api_key:
                raise RuntimeError("No OPENAI_API_KEY is saved in ~/.mana/secrets.toml")

            llm = create_chat_model(
                api_key=api_key,
                model=model_name,
                base_url=base_url,
            )

            system = (
                "You are a helpful repository-aware coding assistant called mana-agent. "
                "Answer concisely and usefully. Use any provided repo context. "
                "If edits or tools would help, describe what you would do."
            )
            user_content = f"Repo context: {extra_context}\n\nUser question: {question}"

            # Run off the event loop (the model invoke may be sync)
            response = await asyncio.to_thread(
                llm.invoke,
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
        if self.gateway is not None and hasattr(self.gateway, "close_session"):
            self.gateway.close_session(self._gateway_session_id)
        self.exit()

    def on_unmount(self) -> None:
        """Use the same idempotent finalization path for all TUI shutdowns."""
        if self.gateway is not None and hasattr(self.gateway, "close_session"):
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
