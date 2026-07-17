"""Central AgentChatGateway for mana-agent.

Responsibilities:
- Own construction of the chat / coding-agent stack for a repository.
- Provide a single point that TUI, Telegram, Dashboard, and CLI use to reach agents.
- Simple path (send/ask) for connectors.
- Rich path (get_rich_context / process_turn) for full auto-chat + coding agent parity.

All frontends should go through an instance of this (or a thin adapter) rather than
building AskService / CodingAgent directly.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from mana_agent.config.settings import Settings
from mana_agent.gateway.config import ChatGatewayConfig
from mana_agent.gateway.stack import ChatStack, build_chat_stack
from mana_agent.gateway.turn_engine import ChatTurnResult, load_analysis_context, process_chat_turn
from mana_agent.services.chat_service import ChatService
from mana_agent.services.chat_session_history import ChatSessionHistory
from mana_agent.workspaces.service import WorkspaceService

logger = logging.getLogger(__name__)


@dataclass
class RichChatContext:
    """Objects and flags needed by rich clients (TUI, full console loop).

    Populated by the gateway after it owns stack construction.
    """

    chat_service: ChatService | Any
    coding_agent: Any | None = None
    tools_orchestrator: Any | None = None
    dir_mode: bool = False
    index_dir: str | None = None
    index_dirs: list[str] | None = None
    auto_execute_plan: bool = False
    auto_execute_max_passes: int = 3
    coding_agent_max_steps: int = 200
    resolved_k: int = 6
    agent_timeout_seconds: int = 600
    root: Path | None = None
    session_id: str | None = None
    event_sink: Callable[..., None] | None = None
    ask_service: Any | None = None
    tool_worker_client: Any | None = None
    coding_memory_service: Any | None = None
    coding_agent_is_custom: bool = False
    execution_profile: str = "balanced"
    auto_continue: bool = True
    agent_tools: bool = True
    config: ChatGatewayConfig | None = None


class AgentChatGateway:
    """Gateway for all agent (multi-agent) chat connections.

    Typical usage::

        gw = AgentChatGateway(root=repo_root, coding_agent=True, ...)
        sid = gw.create_session(frontend="tui")
        result = gw.process_turn(sid, "explain the architecture")
        ctx = gw.get_rich_context(sid)
    """

    def __init__(
        self,
        root: str | Path,
        *,
        config: ChatGatewayConfig | None = None,
        # Core model / index config (subset of chat() flags; also accepted as kwargs)
        model: str | None = None,
        index_dir: str | Path | None = None,
        dir_mode: bool = False,
        max_indexes: int = 0,
        auto_index_missing: bool = True,
        k: int | None = None,
        agent_tools: bool = True,
        coding_agent: bool = True,
        tool_worker_process: bool = True,
        tool_worker_strict: bool = True,
        tool_exec_backend: str = "local",
        redis_url: str | None = None,
        toolsmanager_parallel_requests: int = 3,
        redis_queue_name: str = "mana-tools",
        redis_ttl_seconds: int = 86_400,
        coding_memory: bool = True,
        flow_id: str | None = None,
        coding_plan_max_steps: int = 8,
        coding_search_budget: int = 4,
        coding_read_budget: int = 6,
        coding_require_read_files: int = 2,
        auto_execute_plan: bool = True,
        auto_execute_max_passes: int = 4,
        auto_continue: bool = True,
        execution_profile: str = "balanced",
        full_auto: bool = False,
        full_auto_status_every: int = 10,
        agent_max_steps: int = 6,
        agent_unlimited: bool = False,
        agent_timeout_seconds: int = 30,
        session_id: str | None = None,
        event_sink: Callable[..., None] | None = None,
        # Allow passing pre-built objects (tests / transitional)
        chat_service: Any = None,
        coding_agent_instance: Any = None,
        tools_orchestrator: Any = None,
        settings: Settings | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.settings = settings or Settings()
        self._workspaces = WorkspaceService()

        if config is None:
            config = ChatGatewayConfig(
                model=model,
                index_dir=index_dir,
                dir_mode=dir_mode,
                max_indexes=max_indexes,
                auto_index_missing=auto_index_missing,
                k=k,
                agent_tools=agent_tools,
                coding_agent=coding_agent,
                tool_worker_process=tool_worker_process,
                tool_worker_strict=tool_worker_strict,
                tool_exec_backend=tool_exec_backend,
                redis_url=redis_url,
                toolsmanager_parallel_requests=toolsmanager_parallel_requests,
                redis_queue_name=redis_queue_name,
                redis_ttl_seconds=redis_ttl_seconds,
                coding_memory=coding_memory,
                flow_id=flow_id,
                coding_plan_max_steps=coding_plan_max_steps,
                coding_search_budget=coding_search_budget,
                coding_read_budget=coding_read_budget,
                coding_require_read_files=coding_require_read_files,
                auto_execute_plan=auto_execute_plan,
                auto_execute_max_passes=auto_execute_max_passes,
                auto_continue=auto_continue,
                execution_profile=execution_profile,
                full_auto=full_auto,
                full_auto_status_every=full_auto_status_every,
                agent_max_steps=agent_max_steps,
                agent_unlimited=agent_unlimited,
                agent_timeout_seconds=agent_timeout_seconds,
                session_id=session_id,
                chat_service=chat_service,
                coding_agent_instance=coding_agent_instance,
                tools_orchestrator=tools_orchestrator,
                event_sink=event_sink,
            )
        else:
            # Allow kwargs to override injected objects when config already set
            if chat_service is not None:
                config.chat_service = chat_service
            if coding_agent_instance is not None:
                config.coding_agent_instance = coding_agent_instance
            if tools_orchestrator is not None:
                config.tools_orchestrator = tools_orchestrator
            if event_sink is not None:
                config.event_sink = event_sink

        self.config = config.normalized()
        self._event_sink = self.config.event_sink

        # Parity flags
        self._dir_mode = bool(self.config.dir_mode)
        self._index_dir = str(self.config.index_dir) if self.config.index_dir else None
        self._index_dirs: list[str] = []
        self._max_indexes = int(self.config.max_indexes)
        self._auto_index_missing = bool(self.config.auto_index_missing)
        self._auto_execute_plan = bool(self.config.auto_execute_plan)
        self._auto_execute_max_passes = int(self.config.auto_execute_max_passes)
        self._agent_timeout_seconds = int(self.config.agent_timeout_seconds)

        self._sessions: dict[str, dict[str, Any]] = {}
        self._active: set[str] = set()
        self._history_store = ChatSessionHistory()

        # Build stack (full coding stack when coding_agent=True)
        self._stack: ChatStack = build_chat_stack(
            self.root, self.config, settings=self.settings
        )
        self._chat_service = self._stack.chat_service
        self._coding_agent = self._stack.coding_agent
        self._tools_orchestrator = self._stack.tools_orchestrator
        self._coding_agent_max_steps = self._stack.coding_agent_max_steps
        self._resolved_k = self._stack.resolved_k
        self._coding_agent_is_custom = self._stack.coding_agent_is_custom

        # Default session state seed
        self._default_flow_id = self.config.flow_id
        try:
            from mana_agent.commands.cli_internal import _record_multi_agent_request

            _record_multi_agent_request(
                self.root,
                "gateway:init",
                entrypoint="gateway",
                command_scope=False,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def create_session(self, *, frontend: str = "cli", session_id: str | None = None) -> str:
        """Create a logical session id for a frontend conversation."""
        if session_id:
            sid = session_id
        else:
            try:
                ws = self._workspaces.create_session(self.root)
                sid = ws.session_id
            except Exception:
                sid = f"gw-{frontend}-{id(self)}-{len(self._sessions)}"

        if sid not in self._sessions:
            self._sessions[sid] = self._new_session_state(sid, frontend=frontend)
        return sid

    def _new_session_state(self, session_id: str, *, frontend: str = "cli") -> dict[str, Any]:
        analysis = None
        try:
            analysis = load_analysis_context(self.root)
        except Exception:
            analysis = None
        messages = self._history_store.list(session_id)
        completed: dict[str, dict[str, str]] = {}
        for message in messages:
            if message.role in {"user", "assistant"}:
                completed.setdefault(message.turn_id, {})[message.role] = message.content
        history = [
            (turn["user"], turn["assistant"])
            for turn in completed.values()
            if turn.get("user") and turn.get("assistant")
        ]
        return {
            "frontend": frontend,
            "history": history[-40:],
            "messages": [message.to_dict() for message in messages],
            "root": str(self.root),
            "active_flow_id": self._default_flow_id,
            "auto_chat_state": None,
            "analysis_context": analysis,
            "pending_prechecklist": None,
            "pending_prechecklist_source": "",
            "pending_prechecklist_warning": "",
        }

    def _session(self, session_id: str) -> dict[str, Any]:
        if session_id not in self._sessions:
            # An explicitly supplied/restored workspace session owns its persisted history.
            self._sessions[session_id] = self._new_session_state(session_id)
        return self._sessions[session_id]

    def start_new_conversation(self, session_id: str, *, frontend: str | None = None) -> str:
        """Archive the current conversation and return a fresh isolated session."""
        state = self._session(session_id)
        self.start_new_topic(session_id)
        try:
            self._workspaces.archive_session(session_id)
        except (FileNotFoundError, ValueError):
            pass
        created = self._workspaces.create_session(self.root)
        sid = created.session_id
        self._sessions[sid] = self._new_session_state(
            sid, frontend=frontend or str(state.get("frontend") or "cli")
        )
        return sid

    def session_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Return the durable chronological message log for diagnostics and UIs."""
        return [message.to_dict() for message in self._history_store.list(session_id)]

    def _append_session_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        turn_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        message = self._history_store.append(
            session_id,
            role=role,
            content=content,
            turn_id=turn_id,
            metadata=metadata,
        )
        self._session(session_id).setdefault("messages", []).append(message.to_dict())

    def start_new_topic(self, session_id: str) -> str | None:
        """Reset coding flow for a session (keeps conversation history)."""
        state = self._session(session_id)
        reset_id: str | None = None
        active = state.get("active_flow_id")
        if self._coding_agent is not None:
            target = active or (
                self._coding_agent.get_active_flow_id()
                if hasattr(self._coding_agent, "get_active_flow_id")
                else None
            )
            if isinstance(target, str) and target.strip():
                if hasattr(self._coding_agent, "reset_flow"):
                    reset_id = self._coding_agent.reset_flow(target.strip())
                else:
                    reset_id = target.strip()
        state["active_flow_id"] = None
        return reset_id

    def set_index_dirs(
        self,
        *,
        index_dir: str | Path | None = None,
        index_dirs: list[str | Path] | None = None,
    ) -> None:
        """Attach resolved indexes for classic / dir-mode turns."""
        if index_dir is not None:
            self._index_dir = str(index_dir)
            if hasattr(self._chat_service, "set_index_dir"):
                try:
                    self._chat_service.set_index_dir(index_dir)
                except Exception:
                    pass
        if index_dirs is not None:
            self._index_dirs = [str(p) for p in index_dirs]
            if hasattr(self._chat_service, "set_index_dirs"):
                try:
                    self._chat_service.set_index_dirs(index_dirs)
                except Exception:
                    pass

    def refresh_analysis_context(self, session_id: str | None = None) -> str | None:
        text = load_analysis_context(self.root)
        if session_id:
            self._session(session_id)["analysis_context"] = text
        else:
            for state in self._sessions.values():
                state["analysis_context"] = text
        return text

    # ------------------------------------------------------------------
    # Simple path (Telegram, basic dashboard, API)
    # ------------------------------------------------------------------

    def send(self, session_id: str, text: str) -> str:
        """Synchronous send — full process_turn when stack is rich, else ask."""
        return asyncio.run(self.send_async(session_id, text))

    async def send_async(self, session_id: str, text: str) -> str:
        """Primary simple-path entry used by gateway-connected frontends."""
        self._active.add(session_id)
        try:
            # Prefer full turn engine when coding stack or agent tools are active
            if self._coding_agent is not None or bool(self.config.agent_tools):
                result = await asyncio.to_thread(self.process_turn, session_id, text)
                if result.error and not result.answer:
                    return f"(Gateway error: {result.error})"
                return result.answer or "(No response from agent)"

            # Minimal ChatService-only path
            state = self._session(session_id)
            turn_id = f"turn_{uuid.uuid4().hex[:20]}"
            self._append_session_message(session_id, role="user", content=text, turn_id=turn_id)
            hist = state.get("history", [])[-20:]
            question = text
            if hist:
                transcript = "\n\n".join(
                    f"User: {q}\nMana-Agent: {a}" for q, a in hist
                )
                question = (
                    f"Conversation history for continuity:\n{transcript[-20000:]}\n\n"
                    f"Current user message:\n{text}"
                )
            try:
                resp = self._chat_service.ask(question, k=getattr(self._chat_service, "_k", 6))
            except Exception as exc:
                self._append_session_message(
                    session_id,
                    role="system",
                    content=f"Turn failed: {exc}",
                    turn_id=turn_id,
                    metadata={"state": "failed", "error_type": type(exc).__name__},
                )
                raise
            answer = getattr(resp, "answer", resp)
            if not isinstance(answer, str):
                answer = str(answer or "").strip()
            result = (answer or "").strip() or "(No response from agent)"
            state.setdefault("history", []).append((text, result))
            state["history"] = state["history"][-40:]
            self._append_session_message(session_id, role="assistant", content=result, turn_id=turn_id)
            return result
        finally:
            self._active.discard(session_id)

    def status(self, session_id: str) -> str:
        return "running" if session_id in self._active else "ready"

    def cancel(self, session_id: str) -> bool:
        return False

    # ------------------------------------------------------------------
    # Full turn engine (auto-chat + coding agent + model decision)
    # ------------------------------------------------------------------

    def process_turn(
        self,
        session_id: str,
        text: str,
        *,
        planning_answers: list[str] | None = None,
        event_sink: Callable[..., None] | None = None,
        **options: Any,
    ) -> ChatTurnResult:
        """Run one full chat turn through the gateway-owned engine."""
        self._active.add(session_id)
        turn_id = str(options.pop("turn_id", "") or f"turn_{uuid.uuid4().hex[:20]}")
        self._append_session_message(session_id, role="user", content=text, turn_id=turn_id)
        try:
            state = self._session(session_id)
            sink = event_sink or self._event_sink
            ask_service = self.get_ask_service()
            result = process_chat_turn(
                root=self.root,
                text=text,
                chat_service=self._chat_service,
                ask_service=ask_service,
                coding_agent=self._coding_agent,
                config=self.config,
                session_state=state,
                coding_agent_is_custom=self._coding_agent_is_custom,
                resolved_k=self._resolved_k,
                coding_agent_max_steps=self._coding_agent_max_steps,
                index_dir=options.get("index_dir", self._index_dir),
                index_dirs=options.get("index_dirs", self._index_dirs or None),
                event_sink=sink,
                # Optional LangChain callbacks (e.g. RichToolCallbackHandler) so
                # auto-chat tool start/end events reach the TUI emit bridge.
                callbacks=options.get("callbacks"),
            )
            # Sync flow id back if coding agent advanced it
            if result.flow_id:
                state["active_flow_id"] = result.flow_id
            for index, trace in enumerate(result.trace or []):
                if not isinstance(trace, dict):
                    continue
                summary = str(
                    trace.get("result_summary")
                    or trace.get("output_preview")
                    or trace.get("status")
                    or ""
                ).strip()
                self._append_session_message(
                    session_id,
                    role="tool",
                    content=summary[:4000],
                    turn_id=turn_id,
                    metadata={
                        "tool_name": str(trace.get("tool_name") or "tool"),
                        "sequence": index,
                    },
                )
            if result.answer:
                self._append_session_message(
                    session_id,
                    role="assistant",
                    content=result.answer,
                    turn_id=turn_id,
                    metadata={"model": self.config.model, "mode": result.mode},
                )
            else:
                self._append_session_message(
                    session_id,
                    role="system",
                    content=result.error or "Turn interrupted before an assistant response.",
                    turn_id=turn_id,
                    metadata={"state": "failed" if result.error else "interrupted"},
                )
            return result
        except BaseException as exc:
            self._append_session_message(
                session_id,
                role="system",
                content=f"Turn failed: {exc}",
                turn_id=turn_id,
                metadata={"state": "failed", "error_type": type(exc).__name__},
            )
            raise
        finally:
            self._active.discard(session_id)

    async def process_turn_async(
        self,
        session_id: str,
        text: str,
        **kwargs: Any,
    ) -> ChatTurnResult:
        return await asyncio.to_thread(self.process_turn, session_id, text, **kwargs)

    # ------------------------------------------------------------------
    # Rich path (TUI + full console chat from chat_cli)
    # ------------------------------------------------------------------

    def get_rich_context(self, session_id: str | None = None) -> RichChatContext:
        """Return the objects + parity flags expected by TUI / console."""
        return RichChatContext(
            chat_service=self._chat_service,
            coding_agent=self._coding_agent,
            tools_orchestrator=self._tools_orchestrator,
            dir_mode=self._dir_mode,
            index_dir=self._index_dir,
            index_dirs=list(self._index_dirs) if self._index_dirs else None,
            auto_execute_plan=self._auto_execute_plan,
            auto_execute_max_passes=self._auto_execute_max_passes,
            coding_agent_max_steps=self._coding_agent_max_steps,
            resolved_k=self._resolved_k,
            agent_timeout_seconds=self._agent_timeout_seconds,
            root=self.root,
            session_id=session_id,
            event_sink=self._event_sink,
            ask_service=self.get_ask_service(),
            tool_worker_client=self._stack.tool_worker_client,
            coding_memory_service=self._stack.coding_memory_service,
            coding_agent_is_custom=self._coding_agent_is_custom,
            execution_profile=self.config.execution_profile,
            auto_continue=bool(self.config.auto_continue),
            agent_tools=bool(self.config.agent_tools),
            config=self.config,
        )

    def get_stack(self) -> ChatStack:
        return self._stack

    def get_ask_service(self) -> Any:
        return getattr(self._chat_service, "_ask_service", None) or self._stack.ask_service

    def owns_coding_stack(self) -> bool:
        return self._coding_agent is not None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AgentChatGateway root={self.root} coding={self.owns_coding_stack()}>"


# Convenience alias for the Telegram protocol expectation
ChatGateway = AgentChatGateway
