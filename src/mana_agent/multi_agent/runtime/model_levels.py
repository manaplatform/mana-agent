from __future__ import annotations

import os
from dataclasses import dataclass

from mana_agent.config.user_config import get_setting
from mana_agent.multi_agent.core.types import AgentRole

MODEL_LEVEL_3_HIGH_REASONING = "MODEL_LEVEL_3_HIGH_REASONING"
MODEL_LEVEL_2_CODING = "MODEL_LEVEL_2_CODING"
MODEL_LEVEL_1_FAST_TOOL = "MODEL_LEVEL_1_FAST_TOOL"

_DEFAULT_MODEL_LEVELS = {
    AgentRole.MAIN: ("MANA_MODEL_MAIN", MODEL_LEVEL_3_HIGH_REASONING),
    AgentRole.HEAD_DECISION: ("MANA_MODEL_HEAD_DECISION", MODEL_LEVEL_3_HIGH_REASONING),
    AgentRole.PLANNER: ("MANA_MODEL_PLANNER", MODEL_LEVEL_3_HIGH_REASONING),
    AgentRole.CODING: ("MANA_MODEL_CODING", MODEL_LEVEL_2_CODING),
    AgentRole.VERIFIER: ("MANA_MODEL_VERIFIER", MODEL_LEVEL_2_CODING),
    AgentRole.REVIEWER: ("MANA_MODEL_REVIEWER", MODEL_LEVEL_3_HIGH_REASONING),
    AgentRole.TOOL: ("MANA_MODEL_TOOL", MODEL_LEVEL_1_FAST_TOOL),
    AgentRole.TOOL_WORKER: ("MANA_MODEL_TOOL_WORKER", MODEL_LEVEL_1_FAST_TOOL),
    AgentRole.RESEARCH: ("MANA_MODEL_TOOL", MODEL_LEVEL_1_FAST_TOOL),
    AgentRole.SUMMARIZER: ("MANA_MODEL_SUMMARIZER", MODEL_LEVEL_1_FAST_TOOL),
}


@dataclass(frozen=True)
class ModelLevelAssignment:
    role: AgentRole
    env_var: str
    model_level: str


def model_level_for_role(role: AgentRole) -> ModelLevelAssignment:
    env_var, default = _DEFAULT_MODEL_LEVELS[role]
    return ModelLevelAssignment(role=role, env_var=env_var, model_level=str(get_setting(env_var, default) or default))


@dataclass(frozen=True)
class ResolvedModelAssignment:
    role: AgentRole
    env_var: str
    model_level: str
    resolved_model: str


def _is_symbolic_model_level(value: str) -> bool:
    return str(value or "").strip().startswith("MODEL_LEVEL_")


def resolve_model_for_role(role: AgentRole, *, global_model: str) -> ResolvedModelAssignment:
    env_var, default_level = _DEFAULT_MODEL_LEVELS[role]
    role_value = str(get_setting(env_var, "") or "").strip()
    fallback = str(global_model or "").strip()
    if role_value and not _is_symbolic_model_level(role_value):
        return ResolvedModelAssignment(
            role=role,
            env_var=env_var,
            model_level=default_level,
            resolved_model=role_value,
        )
    configured = role_value or default_level
    resolved = str(get_setting(configured, "") or os.getenv(configured, "") or "").strip()
    return ResolvedModelAssignment(
        role=role,
        env_var=env_var,
        model_level=configured,
        resolved_model=resolved or fallback,
    )
