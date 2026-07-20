"""A2A 1.0 HTTP server built from official SDK routes."""

from __future__ import annotations

from pathlib import Path

from mana_agent.gateway.chat_gateway import AgentChatGateway
from mana_agent.protocols.common.auth import require_bearer_token
from mana_agent.protocols.common.exceptions import OptionalProtocolDependencyError, ProtocolAuthenticationError

from .agent_card import build_agent_card
from .executor import ManaA2AExecutor
from .task_adapter import ManaA2ATaskStore


def a2a_sdk_info() -> dict[str, str | bool]:
    try:
        from importlib.metadata import version
        import a2a  # noqa: F401
    except ImportError:
        return {"installed": False, "protocol_version": "1.0", "sdk_version": ""}
    return {"installed": True, "protocol_version": "1.0", "sdk_version": version("a2a-sdk")}


def create_a2a_app(
    *,
    root: str | Path,
    public_base_url: str,
    token: str,
    enabled_skills: set[str] | None = None,
    max_concurrent_tasks: int = 4,
    max_request_bytes: int = 1_048_576,
    gateway_instance: AgentChatGateway | None = None,
) -> object:
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        from a2a.server.context import ServerCallContext
        from a2a.server.request_handlers.default_request_handler_v2 import DefaultRequestHandlerV2
        from a2a.server.routes.agent_card_routes import create_agent_card_routes
        from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
        from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
        from a2a.server.routes.rest_routes import create_rest_routes
        from a2a.auth.user import User
    except ImportError as exc:
        raise OptionalProtocolDependencyError.for_protocol("a2a") from exc

    if not str(token or "").strip():
        raise ValueError("A2A bearer token is required; unauthenticated network serving is disabled.")
    gateway = gateway_instance or AgentChatGateway(Path(root).expanduser().resolve())
    card = build_agent_card(public_base_url=public_base_url, enabled_skills=enabled_skills)
    executor = ManaA2AExecutor(gateway, max_concurrent_tasks=max_concurrent_tasks)
    task_store = ManaA2ATaskStore()
    handler = DefaultRequestHandlerV2(executor, task_store, card)

    class _User(User):
        def __init__(self, name: str) -> None:
            self.name = name

        @property
        def is_authenticated(self) -> bool:
            return True

        @property
        def user_name(self) -> str:
            return self.name

    class _ContextBuilder:
        def build(self, request: Request) -> ServerCallContext:
            identity = require_bearer_token(request.headers.get("authorization"), token)
            return ServerCallContext(user=_User(identity.caller_id), state={"headers": dict(request.headers)})

    context_builder = _ContextBuilder()
    app = FastAPI(title="Mana-Agent A2A", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def secure_a2a(request: Request, call_next):
        if request.url.path == "/.well-known/agent-card.json":
            return await call_next(request)
        try:
            require_bearer_token(request.headers.get("authorization"), token)
        except ProtocolAuthenticationError:
            return JSONResponse({"error": "authentication required"}, status_code=401)
        length = int(request.headers.get("content-length") or 0)
        if length > max(1, int(max_request_bytes)):
            return JSONResponse({"error": "request too large"}, status_code=413)
        return await call_next(request)

    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url="/a2a", context_builder=context_builder),
        rest_routes=create_rest_routes(handler, context_builder=context_builder),
    )
    app.state.mana_gateway = gateway
    app.state.a2a_executor = executor
    app.state.a2a_task_store = task_store
    return app
