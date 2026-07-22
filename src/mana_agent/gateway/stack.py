"""Chat / coding-agent stack construction for the gateway.

Owns the same construction path that chat_cli historically performed so all
frontends share one coding agent, tool worker, and queue manager setup.
"""

from __future__ import annotations

import logging
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mana_agent.commands.cli_internal import (
    build_ask_service as _ORIGINAL_BUILD_ASK_SERVICE,
)
from mana_agent.config.settings import Settings, default_logs_dir
from mana_agent.config.inference_provider import resolve_inference_connection
from mana_agent.gateway.config import ChatGatewayConfig
from mana_agent.integrations.codex.coding_agent_shim import CodexCodingAgentShim
from mana_agent.integrations.codex.config import CodexSettings
from mana_agent.coding.selection import resolve_coding_backend
from mana_agent.coding.internal_agent_shim import InternalCodingAgentShim
from mana_agent.multi_agent.core.types import AgentRole
from mana_agent.multi_agent.runtime.model_levels import resolve_model_for_role
from mana_agent.model_routing.repository import RepositoryMetadataInspector
from mana_agent.gateway.routing import GatewayRoutingAuthority
from mana_agent.multi_agent.runtime.tool_worker_process import ToolWorkerClient
from mana_agent.multi_agent.runtime.tools_executor import (
    LocalToolsExecutor,
    RedisRQToolsExecutor,
    ToolsExecutionConfig,
    build_tools_executor_with_fallback,
)
from mana_agent.multi_agent.runtime.agent_work_queue import QueueManager
from mana_agent.services.chat_service import ChatService
from mana_agent.memory import CodingMemoryService, MemoryService
from mana_agent.execution import ExecutionManager, build_execution_manager
from mana_agent.workspaces.preparation import PreparedRepository
from mana_agent.workspaces.service import WorkspaceService

logger = logging.getLogger(__name__)

# Public compatibility name used by CLI/TUI integrations and test injection.
CodingAgent = CodexCodingAgentShim

# _ORIGINAL_BUILD_ASK_SERVICE is bound at import time so later monkeypatches of
# cli_internal.build_ask_service (and re-exports on chat_cli/cli) can be detected
# by identity comparison in _resolve_build_ask_service.


def _public_symbol(name: str, default: Any) -> Any:
    """Prefer symbols patched on chat_cli / cli / this module (test fakes).

    Existing smoke tests monkeypatch ``mana_agent.commands.chat_cli.CodingAgent``
    (and similar). Gateway unit tests may patch ``mana_agent.gateway.stack.*``.
    Prefer any symbol that differs from the cli_internal original import.
    """
    try:
        from mana_agent.commands import cli_internal as _ci

        original = getattr(_ci, name, None) if hasattr(_ci, name) else None
    except Exception:
        original = None

    for mod_name in ("mana_agent.commands.chat_cli", "mana_agent.commands.cli"):
        mod = sys.modules.get(mod_name)
        if mod is None or not hasattr(mod, name):
            continue
        public_value = getattr(mod, name)
        if original is not None and public_value is not original:
            return public_value

    if original is not None and default is not original:
        return default
    return default


def _resolve_build_ask_service() -> Any:
    """Return build_ask_service, honoring chat_cli/cli/cli_internal monkeypatches.

    Gateway unit tests patch ``mana_agent.commands.cli_internal.build_ask_service``.
    CLI smoke tests patch the public re-export on ``chat_cli`` or ``cli``.
    Re-exports keep a stale reference when only ``cli_internal`` is patched, so we
    compare against the import-time original and prefer any replaced callable.
    """
    import mana_agent.commands.cli_internal as cli_internal_mod

    for mod_name in (
        "mana_agent.commands.chat_cli",
        "mana_agent.commands.cli",
        "mana_agent.commands.cli_internal",
    ):
        if mod_name == "mana_agent.commands.cli_internal":
            mod = cli_internal_mod
        else:
            mod = sys.modules.get(mod_name)
        if mod is None or not hasattr(mod, "build_ask_service"):
            continue
        candidate = getattr(mod, "build_ask_service")
        if candidate is not _ORIGINAL_BUILD_ASK_SERVICE:
            return candidate
    return _ORIGINAL_BUILD_ASK_SERVICE


def _resolve_agent_max_steps(
    agent_max_steps: int,
    *,
    agent_unlimited: bool,
    min_steps: int = 1,
    cap: int | None = None,
) -> int:
    try:
        from mana_agent.commands.ui_helpers import _resolve_agent_max_steps as helper

        return helper(
            agent_max_steps,
            agent_unlimited=agent_unlimited,
            min_steps=min_steps,
            cap=cap,
        )
    except Exception:
        if agent_unlimited:
            return max(min_steps, 1_000_000_000)
        effective = max(min_steps, int(agent_max_steps))
        if cap is not None:
            effective = min(effective, int(cap))
        return effective


@dataclass
class ChatStack:
    """Fully wired chat runtime objects owned by the gateway."""

    settings: Settings
    ask_service: Any
    chat_service: ChatService | Any
    memory_service: MemoryService | Any
    coding_agent: Any | None = None
    coding_memory_service: Any | None = None
    tool_worker_client: Any | None = None
    tools_orchestrator: Any | None = None
    tools_executor: Any | None = None
    tools_execution_config: ToolsExecutionConfig | None = None
    tools_execution_boot_warnings: list[str] = field(default_factory=list)
    coding_agent_is_custom: bool = False
    effective_model: str | None = None
    chat_agent_max_steps: int = 6
    coding_agent_max_steps: int = 200
    resolved_k: int = 6
    session_id: str = ""
    workspace_id: str | None = None
    repository_id: str | None = None
    log_path: Path | None = None
    execution_manager: ExecutionManager | None = None
    routing_authority: GatewayRoutingAuthority | None = None
    prepared_repository: PreparedRepository | None = None


def build_chat_stack(
    root: Path,
    config: ChatGatewayConfig,
    *,
    settings: Settings | None = None,
    prepared_repository: PreparedRepository | None = None,
) -> ChatStack:
    """Build ask/chat and optional coding stack for *root*.

    When config injects pre-built chat_service / coding_agent_instance /
    tools_orchestrator (tests or transition), those are used and remaining
    objects are still filled where possible.
    """
    root = Path(root).expanduser().resolve()
    cfg = config.normalized()
    settings = settings or Settings()

    if cfg.coding_agent and prepared_repository is None:
        workspace_service = WorkspaceService()
        expected_workspace_id: str | None = None
        if cfg.session_id:
            try:
                expected_workspace_id = workspace_service.store.get_session(
                    cfg.session_id
                ).workspace_id
            except FileNotFoundError:
                expected_workspace_id = None
        prepared_repository = workspace_service.prepare_repository(
            root,
            allow_create=not bool(cfg.session_id),
            initialize_if_missing=True,
            expected_workspace_id=expected_workspace_id,
            entry_point="gateway-stack",
        )

    # --- steps / k ---
    chat_agent_max_steps = _resolve_agent_max_steps(
        cfg.agent_max_steps,
        agent_unlimited=cfg.agent_unlimited,
        min_steps=1,
    )
    coding_agent_max_steps = _resolve_agent_max_steps(
        cfg.agent_max_steps,
        agent_unlimited=cfg.agent_unlimited,
        min_steps=8,
        cap=200,
    )
    resolved_k = int(cfg.k or settings.default_top_k)

    # --- tools execution config ---
    resolved_tool_exec_backend = str(
        (cfg.tool_exec_backend or getattr(settings, "tool_exec_backend", "local")) or "local"
    ).strip().lower()
    if resolved_tool_exec_backend not in {"local", "redis"}:
        resolved_tool_exec_backend = "local"
    resolved_redis_url = str(
        (cfg.redis_url or getattr(settings, "redis_url", "redis://127.0.0.1:6379/0"))
        or "redis://127.0.0.1:6379/0"
    ).strip()
    resolved_parallel_requests = max(
        1,
        int(
            cfg.toolsmanager_parallel_requests
            or getattr(settings, "toolsmanager_parallel_requests", 3)
            or 3
        ),
    )
    resolved_redis_queue_name = (
        str((cfg.redis_queue_name or getattr(settings, "redis_queue_name", "mana-tools")) or "mana-tools").strip()
        or "mana-tools"
    )
    resolved_redis_ttl_seconds = max(
        60,
        int(cfg.redis_ttl_seconds or getattr(settings, "redis_ttl_seconds", 86_400) or 86_400),
    )
    tools_execution_config = ToolsExecutionConfig(
        backend=resolved_tool_exec_backend,
        redis_url=resolved_redis_url,
        queue_name=resolved_redis_queue_name,
        parallel_requests=resolved_parallel_requests,
        ttl_seconds=resolved_redis_ttl_seconds,
    )
    tools_execution_boot_warnings: list[str] = []
    execution_manager = build_execution_manager(settings, event_sink=cfg.event_sink)

    session_id = cfg.session_id or f"sess-{uuid.uuid4().hex}"
    log_path = default_logs_dir(root) / f"mana_agent_{__import__('datetime').datetime.now().strftime('%Y%m%d')}.log"

    workspace_id = prepared_repository.workspace_id if prepared_repository else None
    repository_id = prepared_repository.repository_id if prepared_repository else None
    if cfg.session_id and workspace_id is None:
        try:
            ctx = WorkspaceService().context_for_session(cfg.session_id)
            workspace_id = ctx.workspace.workspace_id
            repository_id = ctx.session.primary_repository_id
        except (FileNotFoundError, ValueError):
            workspace_id = None
            repository_id = None

    routing_authority = GatewayRoutingAuthority(
        root,
        settings=settings,
        event_sink=cfg.event_sink,
    )

    # --- ask + chat service ---
    if cfg.chat_service is not None:
        chat_service = cfg.chat_service
        ask_service = getattr(chat_service, "_ask_service", None) or getattr(
            chat_service, "ask_service", None
        )
    else:
        build_ask = _resolve_build_ask_service()
        try:
            ask_service = build_ask(
                settings,
                model_override=cfg.model,
                project_root=root,
                routing_authority=routing_authority,
            )
        except TypeError:
            try:
                ask_service = build_ask(settings, cfg.model, project_root=root)
            except TypeError:
                ask_service = build_ask(settings, cfg.model)

        chat_service_cls = _public_symbol("ChatService", ChatService)
        chat_service = chat_service_cls(
            ask_service=ask_service,
            settings=settings,
            model_override=cfg.model,
            index_dir=cfg.index_dir,
            dir_mode=cfg.dir_mode,
            root_dir=str(root),
            k=resolved_k,
            agent_tools=bool(cfg.agent_tools),
            agent_max_steps=chat_agent_max_steps,
            agent_timeout_seconds=cfg.agent_timeout_seconds,
            max_indexes=cfg.max_indexes,
            auto_index_missing=cfg.auto_index_missing,
        )

    gateway_ask_agent = getattr(ask_service, "ask_agent", None)
    if gateway_ask_agent is not None and hasattr(gateway_ask_agent, "execution_manager"):
        gateway_ask_agent.execution_manager = execution_manager

    routing_repository = RepositoryMetadataInspector().inspect(root)
    route_context = {
        "routing_authority": routing_authority,
        "session_id": session_id,
        "workspace_id": str(workspace_id or ""),
        "repository_id": str(repository_id or ""),
    }
    effective_model = resolve_model_for_role(
        AgentRole.MAIN,
        global_model=cfg.model or settings.openai_chat_model,
        repository=routing_repository,
        **route_context,
    ).resolved_model
    router_model_assignment = resolve_model_for_role(
        AgentRole.HEAD_DECISION,
        global_model=effective_model,
        repository=routing_repository,
        **route_context,
    )
    coding_model_assignment = resolve_model_for_role(
        AgentRole.CODING,
        global_model=getattr(settings, "mana_codex_model", None) or effective_model,
        repository=routing_repository,
        **route_context,
    )
    planner_model_assignment = resolve_model_for_role(
        AgentRole.PLANNER,
        global_model=settings.openai_coding_planner_model or effective_model,
        repository=routing_repository,
        **route_context,
    )
    tool_worker_model_assignment = resolve_model_for_role(
        AgentRole.TOOL_WORKER,
        global_model=settings.openai_tool_worker_model or effective_model,
        repository=routing_repository,
        **route_context,
    )
    effective_tool_worker_model = tool_worker_model_assignment.resolved_model
    # Stack construction is deliberately credential-free for diagnostics and
    # injected test services. The actual model construction validates keys.
    inference_connection = resolve_inference_connection(settings, require_api_key=False)
    effective_base_url = inference_connection.base_url

    coding_agent_instance = cfg.coding_agent_instance
    coding_memory_service = None
    tool_worker_client = None
    tools_manager_orchestrator = cfg.tools_orchestrator
    tools_executor_instance = None
    public_coding_agent_cls = _public_symbol("CodingAgent", CodingAgent)
    coding_selection = resolve_coding_backend(settings)
    coding_agent_cls = (
        public_coding_agent_cls
        if public_coding_agent_cls is not CodexCodingAgentShim
        else (CodexCodingAgentShim if coding_selection.backend == "codex" else InternalCodingAgentShim)
    )
    coding_agent_is_custom = public_coding_agent_cls is not CodexCodingAgentShim

    def _build_tools_executor(worker_client: Any) -> Any:
        helper = _public_symbol("build_tools_executor_with_fallback", build_tools_executor_with_fallback)
        init_payload = (
            worker_client.init_payload_dict()
            if hasattr(worker_client, "init_payload_dict")
            else {}
        )
        return helper(
            worker_client=worker_client,
            config=tools_execution_config,
            worker_init_payload=init_payload,
            warnings=tools_execution_boot_warnings,
            warning_key=f"gateway:{root}:{tools_execution_config.redis_url}:{tools_execution_config.queue_name}",
            local_executor_cls=_public_symbol("LocalToolsExecutor", LocalToolsExecutor),
            redis_executor_cls=_public_symbol("RedisRQToolsExecutor", RedisRQToolsExecutor),
        )

    if coding_agent_instance is None and cfg.coding_agent:
        coding_repository_root = (
            prepared_repository.repository_root if prepared_repository else root
        )
        coding_working_directory = (
            prepared_repository.working_directory if prepared_repository else root
        )
        if coding_agent_cls is CodexCodingAgentShim:
            codex_settings = CodexSettings.from_mana_settings(
                settings,
                provider=(
                    coding_model_assignment.routing_decision.provider
                    if coding_model_assignment.routing_decision is not None
                    else None
                ),
            ).model_copy(
                update={"model": coding_model_assignment.resolved_model}
            )
            coding_agent_instance = coding_agent_cls(
                repo_root=coding_repository_root,
                working_directory=coding_working_directory,
                codex_settings=codex_settings,
                repository_id=repository_id,
                session_id=session_id,
                event_sink=cfg.event_sink,
                routing_authority=routing_authority,
                workspace_id=workspace_id,
            )
        else:
            if not cfg.agent_tools:
                raise ValueError("internal/custom coding_agent requires agent_tools (needs tool loop).")
            if ask_service is None or getattr(ask_service, "ask_agent", None) is None:
                raise ValueError("custom coding_agent requires AskService.ask_agent to be configured.")

            if cfg.coding_memory:
                coding_memory_service = CodingMemoryService(
                    project_root=root,
                    max_turns=settings.coding_flow_max_turns,
                    max_tasks=settings.coding_flow_max_tasks,
                    session_id=session_id,
                )

            if hasattr(ask_service.ask_agent, "update_model"):
                ask_service.ask_agent.update_model(coding_model_assignment.resolved_model)
            elif hasattr(ask_service.ask_agent, "model"):
                ask_service.ask_agent.model = coding_model_assignment.resolved_model

            if cfg.tool_worker_process:
                tool_worker_client_cls = _public_symbol("ToolWorkerClient", ToolWorkerClient)
                tool_worker_client = tool_worker_client_cls(
                    api_key=inference_connection.api_key,
                    model=effective_tool_worker_model,
                    base_url=effective_base_url,
                    repo_root=coding_repository_root,
                    project_root=coding_working_directory,
                    allowed_prefixes=None,
                    tools_only_strict=cfg.tool_worker_strict,
                    model_level=tool_worker_model_assignment.model_level,
                    workspace_id=workspace_id,
                    repository_id=repository_id,
                )

            coding_agent_instance = coding_agent_cls(
                api_key=inference_connection.api_key,
                base_url=effective_base_url,
                repo_root=coding_repository_root,
                project_root=coding_working_directory,
                ask_agent=ask_service.ask_agent,
                allowed_prefixes=None,
                coding_memory_service=coding_memory_service,
                coding_memory_enabled=cfg.coding_memory,
                plan_max_steps=max(1, int(cfg.coding_plan_max_steps or settings.coding_plan_max_steps)),
                search_budget=max(1, int(cfg.coding_search_budget or settings.coding_search_budget)),
                read_budget=max(1, int(cfg.coding_read_budget or settings.coding_read_budget)),
                require_read_files=max(
                    1, int(cfg.coding_require_read_files or settings.coding_require_read_files)
                ),
                tool_worker_client=tool_worker_client,
                full_auto_mode=(cfg.execution_profile == "full-auto"),
                planner_model=planner_model_assignment.resolved_model,
            )

        if (
            coding_agent_instance is not None
            and cfg.auto_execute_plan
            and tool_worker_client is not None
            and tools_manager_orchestrator is None
        ):
            tools_executor_instance = _build_tools_executor(tool_worker_client)
            tools_manager_orchestrator_cls = _public_symbol("QueueManager", QueueManager)
            tools_manager_orchestrator = tools_manager_orchestrator_cls(
                api_key=inference_connection.api_key,
                model=effective_model,
                base_url=effective_base_url,
                worker_client=tool_worker_client,
                repo_root=root,
                execution_config=tools_execution_config,
                executor=tools_executor_instance,
                coding_memory_service=coding_memory_service,
                workspace_id=workspace_id,
                repository_id=repository_id,
                session_id=session_id,
            )
            if hasattr(coding_agent_instance, "set_tools_manager_orchestrator"):
                coding_agent_instance.set_tools_manager_orchestrator(tools_manager_orchestrator)
            if hasattr(tools_manager_orchestrator, "attach_decision_provider"):
                tools_manager_orchestrator.attach_decision_provider(coding_agent_instance)

    elif coding_agent_instance is not None:
        coding_memory_service = getattr(coding_agent_instance, "coding_memory_service", None)
        tool_worker_client = getattr(coding_agent_instance, "tool_worker_client", None)

    if isinstance(coding_agent_instance, CodexCodingAgentShim):
        coding_backend = "codex"
        coding_model = str(getattr(settings, "mana_codex_model", "") or "router-managed")
        routed_coding_model = coding_agent_instance.codex_settings.model or "app-server-default"
        planner_model = "codex-owned"
    elif coding_agent_instance is not None:
        coding_backend = "internal" if isinstance(coding_agent_instance, InternalCodingAgentShim) else type(coding_agent_instance).__name__
        coding_model = coding_model_assignment.resolved_model
        routed_coding_model = coding_model
        planner_model = planner_model_assignment.resolved_model
    else:
        coding_backend = "disabled"
        coding_model = "disabled"
        routed_coding_model = "disabled"
        planner_model = "disabled"
    tool_worker_model = (
        tool_worker_model_assignment.resolved_model
        if tool_worker_client is not None
        else "disabled"
    )
    logger.info(
        "Resolved chat runtime models: main=%s; router=%s; coding_backend=%s; "
        "coding=%s; coding_routed=%s; planner=%s; tool_worker=%s",
        effective_model,
        router_model_assignment.resolved_model,
        coding_backend,
        coding_model,
        routed_coding_model,
        planner_model,
        tool_worker_model,
    )

    memory_service = MemoryService(
        root=root,
        session_id=cfg.session_id or "",
        workspace_id=workspace_id,
        repository_id=repository_id,
        enable_compatibility=False,
    )

    return ChatStack(
        settings=settings,
        ask_service=ask_service,
        chat_service=chat_service,
        memory_service=memory_service,
        coding_agent=coding_agent_instance,
        coding_memory_service=coding_memory_service,
        tool_worker_client=tool_worker_client,
        tools_orchestrator=tools_manager_orchestrator,
        tools_executor=tools_executor_instance,
        tools_execution_config=tools_execution_config,
        tools_execution_boot_warnings=tools_execution_boot_warnings,
        execution_manager=execution_manager,
        routing_authority=routing_authority,
        prepared_repository=prepared_repository,
        coding_agent_is_custom=coding_agent_is_custom,
        effective_model=effective_model,
        chat_agent_max_steps=chat_agent_max_steps,
        coding_agent_max_steps=coding_agent_max_steps,
        resolved_k=resolved_k,
        session_id=session_id,
        workspace_id=workspace_id,
        repository_id=repository_id,
        log_path=log_path,
    )
