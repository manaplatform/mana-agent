"""Official-SDK ACP agent backed by one AgentChatGateway."""

from __future__ import annotations

import asyncio
from typing import Any

from mana_agent._version import get_version
from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.evals.recorder import record_current

from .event_mapper import AcpEventMapper
from .permissions import AcpPermissionBroker
from .session_adapter import AcpSessionAdapter


class ManaAcpAgent:
    MODES = (
        ("ask", "Ask", "Conversation and repository questions."),
        ("coding", "Coding", "Model-decided repository changes."),
        ("planning", "Planning", "Implementation planning."),
        ("review", "Review", "Read-only code review."),
        ("verification", "Verification", "Run model-selected verification."),
        ("read-only", "Read only", "Repository inspection without mutation."),
    )

    def __init__(
        self,
        gateway: AgentChatGateway,
        *,
        allowed_roots: tuple[Any, ...] = (),
        mcp_forwarding: bool = True,
        session_load: bool = True,
    ) -> None:
        self.gateway = gateway
        self.sessions = AcpSessionAdapter(gateway, allowed_roots=allowed_roots)
        self.mapper = AcpEventMapper()
        self.mcp_forwarding = mcp_forwarding
        self.session_load_enabled = session_load
        self.client: Any = None
        self.permissions: AcpPermissionBroker | None = None
        self._turns: dict[str, asyncio.Task[Any]] = {}
        self._turn_lock = asyncio.Lock()

    def on_connect(self, conn: Any) -> None:
        self.client = conn
        self.permissions = AcpPermissionBroker(conn)

    async def initialize(self, protocol_version: int, client_capabilities: Any = None, client_info: Any = None, **_: Any) -> Any:
        from acp import PROTOCOL_VERSION
        from acp.schema import (
            AgentCapabilities,
            Implementation,
            InitializeResponse,
            McpCapabilities,
            PromptCapabilities,
            SessionCapabilities,
            SessionCloseCapabilities,
            SessionListCapabilities,
        )

        _ = client_capabilities, client_info
        negotiated = min(int(protocol_version), int(PROTOCOL_VERSION))
        return InitializeResponse(
            protocol_version=negotiated,
            agent_info=Implementation(name="mana-agent", title="Mana Agent", version=get_version()),
            agent_capabilities=AgentCapabilities(
                load_session=self.session_load_enabled,
                prompt_capabilities=PromptCapabilities(image=False, audio=False, embedded_context=False),
                mcp_capabilities=McpCapabilities(http=True, sse=True, acp=False),
                session_capabilities=SessionCapabilities(
                    list=SessionListCapabilities(),
                    close=SessionCloseCapabilities(),
                ),
            ),
        )

    async def new_session(self, cwd: str, additional_directories: list[str] | None = None, mcp_servers: list[Any] | None = None, **_: Any) -> Any:
        from acp.schema import NewSessionResponse

        if mcp_servers and not self.mcp_forwarding:
            raise ValueError("ACP MCP forwarding is disabled by local policy.")
        state = self.sessions.create(cwd=cwd, additional_directories=additional_directories, mcp_servers=mcp_servers)
        record_current("protocol.acp.session.created", {"session_id": state.acp_session_id})
        await self._send_commands(state.acp_session_id)
        return NewSessionResponse(session_id=state.acp_session_id, modes=self._mode_state(state.mode), config_options=self._config_options(state))

    async def load_session(self, cwd: str, session_id: str, mcp_servers: list[Any] | None = None, additional_directories: list[str] | None = None, **_: Any) -> Any:
        from acp.helpers import update_agent_message_text, update_user_message_text
        from acp.schema import LoadSessionResponse

        if not self.session_load_enabled:
            raise ValueError("ACP session loading is disabled by local policy.")
        if mcp_servers and not self.mcp_forwarding:
            raise ValueError("ACP MCP forwarding is disabled by local policy.")
        state = self.sessions.load_session(session_id, cwd=cwd, additional_directories=additional_directories, mcp_servers=mcp_servers)
        record_current("protocol.acp.session.loaded", {"session_id": session_id})
        for message in self.sessions.history(session_id):
            update = update_user_message_text(message["content"]) if message.get("role") == "user" else update_agent_message_text(message["content"])
            if message.get("role") in {"user", "assistant"}:
                await self.client.session_update(session_id=session_id, update=update)
        await self._send_commands(session_id)
        return LoadSessionResponse(modes=self._mode_state(state.mode), config_options=self._config_options(state))

    async def list_sessions(self, cwd: str | None = None, cursor: str | None = None, **_: Any) -> Any:
        from acp.schema import ListSessionsResponse, SessionInfo

        states = self.sessions.list_states(cwd=cwd)
        start = max(0, int(cursor or 0))
        page = states[start : start + 100]
        next_cursor = str(start + len(page)) if start + len(page) < len(states) else None
        return ListSessionsResponse(
            sessions=[SessionInfo(session_id=item.acp_session_id, cwd=item.cwd, additional_directories=list(item.additional_directories), title=f"Mana Agent ({item.mode})") for item in page],
            next_cursor=next_cursor,
        )

    async def prompt(self, session_id: str, prompt: list[Any], **_: Any) -> Any:
        from acp.helpers import update_agent_message_text
        from acp.schema import PromptResponse

        state = self.sessions.get(session_id)
        if state.closed:
            raise ValueError("ACP session is closed.")
        text = self._prompt_text(prompt)
        record_current("protocol.acp.prompt.started", {"session_id": session_id})
        loop = asyncio.get_running_loop()

        def sink(event: Any) -> None:
            if isinstance(event, dict) and str(event.get("type") or event.get("event_type") or "") in {"approval.required", "approval_required"}:
                broker = self.permissions
                if broker is None:
                    allowed = False
                else:
                    future = asyncio.run_coroutine_threadsafe(
                        broker.request(
                            session_id=session_id,
                            call_id=str(event.get("call_id") or event.get("id") or "approval"),
                            title=str(event.get("title") or "Tool permission"),
                            tool_name=str(event.get("tool_name") or "tool"),
                            raw_input=event.get("args"),
                        ),
                        loop,
                    )
                    allowed = bool(future.result())
                resolver = event.get("resolve")
                if callable(resolver):
                    resolver(allowed)
                return
            for update in self.mapper.map(event):
                loop.call_soon_threadsafe(asyncio.create_task, self.client.session_update(session_id=session_id, update=update))

        async with self._turn_lock:
            ask_service = self.gateway.get_ask_service()
            prior_mcp = list(getattr(ask_service, "mcp_server_overrides", []) or [])
            ask_service.mcp_server_overrides = list(state.mcp_overrides)
            task = asyncio.create_task(
                self.gateway.process_turn_async(
                    state.mana_session_id,
                    text,
                    event_sink=sink,
                    target_files=[],
                    protocol_read_only=state.read_only,
                )
            )
            self._turns[session_id] = task
            try:
                result = await task
            except asyncio.CancelledError:
                return PromptResponse(stop_reason="cancelled")
            finally:
                self._turns.pop(session_id, None)
                ask_service.mcp_server_overrides = prior_mcp
        if result.answer:
            await self.client.session_update(session_id=session_id, update=update_agent_message_text(result.answer))
        if result.error:
            record_current("protocol.acp.prompt.finished", {"session_id": session_id, "stop_reason": "refusal"})
            return PromptResponse(stop_reason="refusal")
        record_current("protocol.acp.prompt.finished", {"session_id": session_id, "stop_reason": "end_turn"})
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id: str, **_: Any) -> None:
        task = self._turns.get(session_id)
        if task is not None:
            task.cancel()
        if self.permissions is not None:
            self.permissions.clear_session(session_id)
        state = self.sessions.get(session_id)
        self.gateway.cancel(state.mana_session_id)
        record_current("protocol.acp.session.cancelled", {"session_id": session_id})

    async def close_session(self, session_id: str, **_: Any) -> Any:
        from acp.schema import CloseSessionResponse

        await self.cancel(session_id)
        self.sessions.close(session_id)
        record_current("protocol.acp.session.closed", {"session_id": session_id})
        if self.permissions is not None:
            self.permissions.clear_session(session_id)
        return CloseSessionResponse()

    async def set_session_mode(self, session_id: str, mode_id: str, **_: Any) -> Any:
        from acp.helpers import update_current_mode
        from acp.schema import SetSessionModeResponse

        if mode_id not in {item[0] for item in self.MODES}:
            raise ValueError("Unsupported Mana session mode.")
        state = self.sessions.get(session_id)
        state.mode = mode_id
        state.read_only = mode_id in {"review", "read-only"}
        self.sessions._save()  # noqa: SLF001
        await self.client.session_update(session_id=session_id, update=update_current_mode(mode_id))
        return SetSessionModeResponse()

    async def set_config_option(self, config_id: str, session_id: str, value: str | bool, **_: Any) -> Any:
        from acp.schema import SetSessionConfigOptionResponse

        state = self.sessions.get(session_id)
        if config_id != "read-only" or not isinstance(value, bool):
            raise ValueError("Unsupported ACP session configuration option.")
        state.read_only = value
        self.sessions._save()  # noqa: SLF001
        return SetSessionConfigOptionResponse(config_options=self._config_options(state))

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise ValueError(f"Unsupported ACP extension method: {method}")

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        _ = method, params

    async def shutdown(self) -> None:
        for session_id in list(self._turns):
            await self.cancel(session_id)
        for state in list(self.sessions.sessions.values()):
            if not state.closed:
                self.sessions.close(state.acp_session_id)

    @classmethod
    def _mode_state(cls, current: str) -> Any:
        from acp.schema import SessionMode, SessionModeState

        return SessionModeState(
            current_mode_id=current,
            available_modes=[SessionMode(id=mode_id, name=name, description=description) for mode_id, name, description in cls.MODES],
        )

    @staticmethod
    def _config_options(state: Any) -> list[Any]:
        from acp.schema import SessionConfigOptionBoolean

        return [SessionConfigOptionBoolean(id="read-only", name="Read only", description="Deny mutation-capable routes.", current_value=state.read_only, type="boolean")]

    async def _send_commands(self, session_id: str) -> None:
        from acp.helpers import update_available_commands
        from acp.schema import AvailableCommand

        commands = [
            AvailableCommand(name=name, description=description)
            for name, description in (
                ("new", "Start a new conversation."),
                ("models", "Show configured models."),
                ("tools", "Show available tools."),
                ("status", "Show session and task status."),
                ("clear", "Clear the visible conversation."),
            )
        ]
        await self.client.session_update(session_id=session_id, update=update_available_commands(commands))

    @staticmethod
    def _prompt_text(blocks: list[Any]) -> str:
        parts: list[str] = []
        for block in blocks:
            kind = str(getattr(block, "type", ""))
            if kind == "text":
                parts.append(str(getattr(block, "text", "")))
            elif kind == "resource_link":
                name = str(getattr(block, "name", "resource"))
                uri = str(getattr(block, "uri", ""))
                parts.append(f"Referenced resource: {name} ({uri})")
            else:
                raise ValueError(f"Unsupported ACP prompt content type: {kind or 'unknown'}")
        text = "\n\n".join(item for item in parts if item).strip()
        if not text:
            raise ValueError("ACP prompt must contain text or a resource link.")
        return text
