from __future__ import annotations

import os
from dataclasses import dataclass

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
    return ModelLevelAssignment(role=role, env_var=env_var, model_level=os.getenv(env_var, default))
