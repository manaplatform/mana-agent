from __future__ import annotations

import getpass
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mana_agent import __version__
from mana_agent.api.exceptions import ManaApiError
from mana_agent.api.routes.analyze import router as analyze_router
from mana_agent.api.routes.conversations import router as conversations_router
from mana_agent.api.routes.control import router as control_router
from mana_agent.api.routes.events_ws import router as events_ws_router
from mana_agent.api.routes.canvas import router as canvas_router
from mana_agent.api.routes.fleet import router as fleet_router
from mana_agent.api.routes.repository_analyze import router as repository_analyze_router
from mana_agent.api.routes.workspaces import router as workspaces_router
from mana_agent.api.routes.teach import router as teach_router
from mana_agent.api.routes.servers import router as servers_router
from mana_agent.api.routes.tasks import router as tasks_router
from mana_agent.api.routes.memory_capsules import router as memory_capsules_router
from mana_agent.human_inbox.api import router as human_inbox_router
from mana_agent.config.user_config import load_effective_settings, validate_bool


def _local_capsule_identity_resolver(root: Path) -> Any:
    """Resolve the fixed identity for one trusted local repository host.

    The standalone server has no upstream identity provider.  Its process
    owner, configured dashboard root, and repository identity are therefore
    fixed when the app starts; request data never contributes to this binding.
    Private and parent-child capsules remain unavailable because their durable
    task identities cannot be established by this local host.
    """
    from mana_agent.config.user_config import resolve_local_user_id
    from mana_agent.memory import CapsuleTaskContext, MemoryPrincipal
    from mana_agent.workspaces.paths import repository_id_for_path

    user_id = resolve_local_user_id()
    project_id = repository_id_for_path(root)
    task_id = f"api-{project_id}"
    principal = MemoryPrincipal(
        user_id=user_id,
        project_id=project_id,
        task_id=task_id,
        agent_id="api:local",
        capabilities=frozenset({
            "memory.capsule.read.project",
            "memory.capsule.read.user",
        }),
    )
    context = CapsuleTaskContext(
        user_id=user_id,
        organisation_id=None,
        project_id=project_id,
        team_ids=frozenset(),
        task_id=task_id,
        agent_id="api:local",
    )

    def resolve_local_identity(_request: Request) -> tuple[MemoryPrincipal, CapsuleTaskContext]:
        return principal, context

    return resolve_local_identity


def _local_capsule_bindings() -> tuple[Any, Any]:
    """Bind the standalone API to one trusted local repository identity."""
    from mana_agent.memory import CapsuleService
    from mana_agent.memory.config import CapsuleConfig

    root = Path(os.getenv("MANA_DASHBOARD_ROOT") or Path.cwd()).expanduser().resolve()
    settings = load_effective_settings(include_env=True)
    service = CapsuleService(
        root,
        config=CapsuleConfig.load(settings),
        provider=str(settings.get("MANA_MEMORY_PROVIDER") or "mana"),
    )
    return service, _local_capsule_identity_resolver(root)


def create_app(
    *,
    telegram_config: Any | None = None,
    telegram_gateway: Any | None = None,
    chat_gateway: Any | None = None,
    github_autopilot: Any | None = None,
    capsule_identity_resolver: Any | None = None,
    capsule_service: Any | None = None,
    human_inbox_identity_resolver: Any | None = None,
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
    shared_remote_execution = getattr(chat_gateway, "remote_execution_service", None)
    worker_gateway = WorkerGateway(
        WorkerGatewayConfig(
            enabled=validate_bool(gateway_settings["MANA_WORKER_GATEWAY_ENABLED"]),
            public_url=str(gateway_settings["MANA_WORKER_GATEWAY_PUBLIC_URL"]),
            allow_insecure_http=validate_bool(
                gateway_settings["MANA_WORKER_GATEWAY_ALLOW_INSECURE_HTTP"]
            ),
            allow_insecure_local_development=validate_bool(
                gateway_settings["MANA_WORKER_GATEWAY_LOCAL_DEV"]
            ),
        ),
        registry=(
            shared_remote_execution.workers
            if shared_remote_execution is not None
            else None
        ),
        execution=shared_remote_execution,
        fleet_registry=fleet_registry,
    )
    from mana_agent.execution_supervisor import ExecutionSupervisor, ExecutionSupervisorConfig
    from mana_agent.services.execution_event_hub import get_execution_event_hub

    gateway_memory_service = getattr(getattr(chat_gateway, "_stack", None), "memory_service", None)
    gateway_capsule_service = (
        getattr(gateway_memory_service, "capsules", None)
        if gateway_memory_service is not None
        else None
    )
    if capsule_service is None and gateway_capsule_service is not None:
        capsule_service = gateway_capsule_service

    def supervisor_event(event_type: str, payload: dict[str, Any]) -> None:
        get_execution_event_hub().publish({
            "type": event_type,
            "kind": event_type,
            "status": "success" if event_type == "task_completed" else "running",
            "message": event_type.replace("_", " "),
            "execution_id": payload.get("task_id", ""),
            "metadata": {"execution_supervisor": True, **payload},
        }, persist=False)

    execution_supervisor = ExecutionSupervisor(
        ExecutionSupervisorConfig.from_settings(Settings()),
        event_sink=supervisor_event,
    )
    from mana_agent.human_inbox import default_human_inbox_service
    human_inbox = default_human_inbox_service(branch_controller=execution_supervisor)
    worker_gateway.execution.attach_inbox(human_inbox)
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
        if execution_supervisor.config.startup_recovery:
            execution_supervisor.reconnect_tree()
            execution_supervisor.recover()
        human_inbox.expire_due()
        application.state.human_inbox_reconciliation = human_inbox.reconcile()
        application.state.execution_supervisor = execution_supervisor
        application.state.human_inbox = human_inbox
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

    if capsule_service is None and capsule_identity_resolver is None:
        local_capsule_service, local_capsule_identity_resolver = _local_capsule_bindings()
        capsule_service = local_capsule_service
        capsule_identity_resolver = local_capsule_identity_resolver
    elif capsule_identity_resolver is None and capsule_service is gateway_capsule_service:
        capsule_identity_resolver = _local_capsule_identity_resolver(
            Path(getattr(chat_gateway, "root", Path.cwd())).expanduser().resolve()
        )

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
    app.include_router(canvas_router)
    app.include_router(fleet_router)
    app.include_router(workspaces_router)
    app.include_router(teach_router)
    app.include_router(servers_router)
    app.include_router(tasks_router)
    app.include_router(memory_capsules_router)
    app.include_router(human_inbox_router)
    app.include_router(build_worker_router(worker_gateway))
    if github_autopilot is not None:
        from mana_agent.github_autopilot.webhook import router as github_autopilot_router
        app.include_router(github_autopilot_router)

    # Make the central chat gateway (if provided) available to routes / services
    if chat_gateway is not None:
        app.state.chat_gateway = chat_gateway
    if capsule_identity_resolver is not None:
        app.state.capsule_identity_resolver = capsule_identity_resolver
    if capsule_service is not None:
        app.state.capsule_service = capsule_service
    app.state.worker_gateway = worker_gateway
    app.state.fleet_registry = fleet_registry
    app.state.execution_supervisor = execution_supervisor
    app.state.human_inbox = human_inbox
    if human_inbox_identity_resolver is not None:
        app.state.human_inbox_identity_resolver = human_inbox_identity_resolver
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
