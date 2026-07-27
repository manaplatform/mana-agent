from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mana_agent import __version__
from mana_agent.api.exceptions import ManaApiError
from mana_agent.api.routes.analyze import router as analyze_router
from mana_agent.api.routes.conversations import router as conversations_router
from mana_agent.api.routes.control import router as control_router
from mana_agent.api.routes.events_ws import router as events_ws_router
from mana_agent.api.routes.fleet import router as fleet_router
from mana_agent.api.routes.repository_analyze import router as repository_analyze_router
from mana_agent.api.routes.workspaces import router as workspaces_router
from mana_agent.api.routes.teach import router as teach_router
from mana_agent.config.user_config import load_effective_settings, validate_bool


def create_app(
    *,
    telegram_config: Any | None = None,
    telegram_gateway: Any | None = None,
    chat_gateway: Any | None = None,
    github_autopilot: Any | None = None,
) -> FastAPI:
    from mana_agent.remote_execution.gateway import WorkerGateway, WorkerGatewayConfig, build_worker_router
    from mana_agent.config.settings import Settings
    from mana_agent.fleet import FleetConfig, FleetRegistry, FleetStore

    gateway_settings = load_effective_settings(include_env=True)
    fleet_registry = getattr(chat_gateway, "fleet_registry", None)
    if fleet_registry is None:
        # The API server is commonly started without a ChatGateway. Reverse
        # workers still publish their Fleet capability inventory over this API,
        # so the standalone coordinator must own the same persistent registry.
        fleet_config = FleetConfig.from_settings(Settings())
        fleet_registry = FleetRegistry(FleetStore(fleet_config.root), fleet_config)
    worker_gateway = WorkerGateway(WorkerGatewayConfig(
        enabled=validate_bool(gateway_settings["MANA_WORKER_GATEWAY_ENABLED"]),
        public_url=str(gateway_settings["MANA_WORKER_GATEWAY_PUBLIC_URL"]),
        allow_insecure_http=validate_bool(
            gateway_settings["MANA_WORKER_GATEWAY_ALLOW_INSECURE_HTTP"]
        ),
        allow_insecure_local_development=validate_bool(
            gateway_settings["MANA_WORKER_GATEWAY_LOCAL_DEV"]
        ),
    ), fleet_registry=fleet_registry)
    telegram_connector = None
    if telegram_config is None:
        from mana_agent.connectors.telegram.config import load_telegram_config
        telegram_config = load_telegram_config()
    effective_telegram_gateway = telegram_gateway or chat_gateway
    if telegram_config.enabled and telegram_config.effective_transport == "webhook":
        from mana_agent.connectors.telegram.connector import TelegramConnector
        telegram_connector = TelegramConnector(telegram_config, gateway=effective_telegram_gateway)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if telegram_connector is not None:
            await telegram_connector.initialize()
            assert telegram_connector.task_queue is not None
            await telegram_connector.task_queue.start()
            await telegram_connector.register_webhook()
            application.state.telegram_connector = telegram_connector
        if github_autopilot is not None:
            await github_autopilot.start()
            application.state.github_autopilot = github_autopilot
        try:
            yield
        finally:
            if telegram_connector is not None:
                await telegram_connector.stop(remove_webhook=False)
            if github_autopilot is not None:
                await github_autopilot.stop()

    app = FastAPI(
        title="Mana-Agent API",
        version=__version__,
        description="HTTP API for Mana-Agent repository intelligence workflows.",
        lifespan=lifespan,
    )

    @app.exception_handler(ManaApiError)
    async def _mana_api_error_handler(_request: Request, exc: ManaApiError) -> JSONResponse:
        payload: dict[str, str] = {"detail": exc.detail}
        if exc.error:
            payload["error"] = exc.error
        return JSONResponse(status_code=exc.status_code, content=payload)

    app.include_router(analyze_router)
    app.include_router(repository_analyze_router)
    app.include_router(conversations_router)
    app.include_router(control_router)
    app.include_router(events_ws_router)
    app.include_router(fleet_router)
    app.include_router(workspaces_router)
    app.include_router(teach_router)
    app.include_router(build_worker_router(worker_gateway))
    if github_autopilot is not None:
        from mana_agent.github_autopilot.webhook import router as github_autopilot_router
        app.include_router(github_autopilot_router)

    # Make the central chat gateway (if provided) available to routes / services
    if chat_gateway is not None:
        app.state.chat_gateway = chat_gateway
    app.state.worker_gateway = worker_gateway
    app.state.fleet_registry = fleet_registry
    if telegram_connector is not None:
        from fastapi import Response

        async def telegram_webhook(request: Request) -> Response:
            return await telegram_connector.webhook_receiver().receive(request)

        app.add_api_route(
            telegram_config.webhook.path,
            telegram_webhook,
            methods=["POST"],
            include_in_schema=False,
            tags=["telegram"],
        )
    return app


def _configured_github_autopilot() -> Any | None:
    from mana_agent.config.settings import Settings
    from mana_agent.github_autopilot import GitHubAutopilotService, GitHubAutopilotSettings

    settings = GitHubAutopilotSettings.from_mana_settings(Settings())
    return GitHubAutopilotService(settings) if settings.enabled else None


app = create_app(github_autopilot=_configured_github_autopilot())
