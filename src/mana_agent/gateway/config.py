"""Configuration for AgentChatGateway runtime.

Mirrors the runtime-relevant flags from the chat CLI so all frontends
(console, TUI, Telegram, dashboard) can share one construction path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ChatGatewayConfig:
    """Runtime config for chat / coding-agent stack construction and turns."""

    # Model / index
    model: str | None = None
    # When True, use the explicit model fields below instead of resolving through
    # user MODEL_LEVEL_* / MANA_MODEL_* settings. Required for eval isolation so
    # suite variant models are not rewritten by operator preferences.
    pin_models: bool = False
    router_model: str | None = None
    coding_model: str | None = None
    reviewer_model: str | None = None
    verifier_model: str | None = None
    index_dir: str | Path | None = None
    dir_mode: bool = False
    max_indexes: int = 0
    auto_index_missing: bool = True
    k: int | None = None

    # Agent behavior
    agent_tools: bool = True
    coding_agent: bool = True
    coding_memory: bool = True
    flow_id: str | None = None

    # Tool worker / executor
    tool_worker_process: bool = True
    tool_worker_strict: bool = True
    tool_exec_backend: str = "local"
    redis_url: str | None = None
    toolsmanager_parallel_requests: int = 3
    redis_queue_name: str = "mana-tools"
    redis_ttl_seconds: int = 86_400

    # Coding budgets
    coding_plan_max_steps: int = 8
    coding_search_budget: int = 4
    coding_read_budget: int = 6
    coding_require_read_files: int = 2

    # Auto-execute / profile
    auto_execute_plan: bool = True
    auto_execute_max_passes: int = 4
    auto_continue: bool = True
    execution_profile: str = "balanced"  # full-auto | balanced | conservative
    full_auto: bool = False
    full_auto_status_every: int = 10

    # Step / timeout budgets
    agent_max_steps: int = 6
    agent_unlimited: bool = False
    agent_timeout_seconds: int = 30

    # Gateway specialist-lane coordinator
    lane_overrides: dict[str, Any] = field(default_factory=dict)
    lane_global_worker_limit: int = 8
    lane_provider_limits: dict[str, int] = field(default_factory=dict)
    lane_session_token_budget: int | None = None
    lane_global_token_budget: int | None = None

    # Session
    session_id: str | None = None
    memory_user_id: str = ""

    # Optional injection (tests / transitional)
    chat_service: Any = None
    coding_agent_instance: Any = None
    tools_orchestrator: Any = None
    event_sink: Any = None

    # Extra free-form overrides
    extra: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "ChatGatewayConfig":
        """Return a copy with profile / pass-cap rules applied."""
        profile = str(self.execution_profile or "balanced").strip().lower()
        if self.full_auto:
            profile = "full-auto"
        if profile not in {"full-auto", "balanced", "conservative"}:
            profile = "balanced"

        auto_execute = bool(self.auto_execute_plan)
        max_passes = max(1, min(int(self.auto_execute_max_passes), 12))
        if profile == "full-auto":
            auto_execute = True
            if int(self.auto_execute_max_passes) == 4:
                max_passes = 10

        return ChatGatewayConfig(
            model=self.model,
            pin_models=bool(self.pin_models),
            router_model=self.router_model,
            coding_model=self.coding_model,
            reviewer_model=self.reviewer_model,
            verifier_model=self.verifier_model,
            index_dir=self.index_dir,
            dir_mode=bool(self.dir_mode),
            max_indexes=int(self.max_indexes),
            auto_index_missing=bool(self.auto_index_missing),
            k=self.k,
            agent_tools=bool(self.agent_tools),
            coding_agent=bool(self.coding_agent),
            coding_memory=bool(self.coding_memory),
            flow_id=self.flow_id,
            tool_worker_process=bool(self.tool_worker_process),
            tool_worker_strict=bool(self.tool_worker_strict),
            tool_exec_backend=str(self.tool_exec_backend or "local").strip().lower() or "local",
            redis_url=self.redis_url,
            toolsmanager_parallel_requests=max(1, int(self.toolsmanager_parallel_requests or 3)),
            redis_queue_name=str(self.redis_queue_name or "mana-tools").strip() or "mana-tools",
            redis_ttl_seconds=max(60, int(self.redis_ttl_seconds or 86_400)),
            coding_plan_max_steps=int(self.coding_plan_max_steps),
            coding_search_budget=int(self.coding_search_budget),
            coding_read_budget=int(self.coding_read_budget),
            coding_require_read_files=int(self.coding_require_read_files),
            auto_execute_plan=auto_execute,
            auto_execute_max_passes=max_passes,
            auto_continue=bool(self.auto_continue and auto_execute),
            execution_profile=profile,
            full_auto=profile == "full-auto",
            full_auto_status_every=max(0, int(self.full_auto_status_every)),
            agent_max_steps=int(self.agent_max_steps),
            agent_unlimited=bool(self.agent_unlimited),
            agent_timeout_seconds=int(self.agent_timeout_seconds),
            lane_overrides=dict(self.lane_overrides or {}),
            lane_global_worker_limit=max(1, int(self.lane_global_worker_limit or 8)),
            lane_provider_limits={
                str(key): max(1, int(value))
                for key, value in (self.lane_provider_limits or {}).items()
            },
            # 0 means unlimited (product policy); only positive values cap lanes.
            lane_session_token_budget=(
                None
                if self.lane_session_token_budget is None
                or int(self.lane_session_token_budget) <= 0
                else max(1, int(self.lane_session_token_budget))
            ),
            lane_global_token_budget=(
                None
                if self.lane_global_token_budget is None
                or int(self.lane_global_token_budget) <= 0
                else max(1, int(self.lane_global_token_budget))
            ),
            session_id=self.session_id,
            memory_user_id=str(self.memory_user_id or "").strip(),
            chat_service=self.chat_service,
            coding_agent_instance=self.coding_agent_instance,
            tools_orchestrator=self.tools_orchestrator,
            event_sink=self.event_sink,
            extra=dict(self.extra or {}),
        )
