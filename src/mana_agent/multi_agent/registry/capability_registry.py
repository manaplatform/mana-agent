from __future__ import annotations

from mana_agent.multi_agent.core.types import AgentRole

DEFAULT_CAPABILITIES: dict[AgentRole, list[str]] = {
    AgentRole.MAIN: ["conversation"],
    AgentRole.HEAD_DECISION: ["decision", "risk_analysis"],
    AgentRole.PLANNER: ["planning"],
    AgentRole.RESEARCH: ["repo_search", "repo_read"],
    AgentRole.CODING: ["coding", "patch_generation"],
    AgentRole.TOOL: ["tool_execution"],
    AgentRole.TOOL_WORKER: ["tool_execution"],
    AgentRole.VERIFIER: ["verification"],
    AgentRole.REVIEWER: ["review", "risk_analysis"],
    AgentRole.SUMMARIZER: ["summarization"],
}
