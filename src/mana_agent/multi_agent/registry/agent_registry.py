from __future__ import annotations

from mana_agent.multi_agent.core.errors import AgentRegistryError
from mana_agent.multi_agent.core.ids import new_agent_id, new_subagent_id
from mana_agent.multi_agent.core.types import AgentNode, AgentRole, AgentState
from mana_agent.multi_agent.registry.capability_registry import DEFAULT_CAPABILITIES
from mana_agent.multi_agent.runtime.model_levels import model_level_for_role


class AgentRegistry:
    def __init__(self, *, max_depth: int = 5, max_active_agents: int = 16) -> None:
        self.max_depth = max_depth
        self.max_active_agents = max_active_agents
        self.agents: dict[str, AgentNode] = {}
        self._bootstrapping_defaults = False
        self.register_defaults()

    def register_defaults(self) -> None:
        if self.agents:
            return
        self._bootstrapping_defaults = True
        main = self.register(AgentRole.MAIN)
        head = self.register(AgentRole.HEAD_DECISION, parent_agent_id=main.agent_id)
        for role in (
            AgentRole.PLANNER,
            AgentRole.RESEARCH,
            AgentRole.CODING,
            AgentRole.TOOL,
            AgentRole.VERIFIER,
            AgentRole.REVIEWER,
            AgentRole.SUMMARIZER,
        ):
            self.register(role, parent_agent_id=head.agent_id)
        self._bootstrapping_defaults = False

    def register(
        self,
        role: AgentRole,
        *,
        parent_agent_id: str | None = None,
        capabilities: list[str] | None = None,
        agent_id: str | None = None,
    ) -> AgentNode:
        if not self._bootstrapping_defaults and len(self.agents) >= self.max_active_agents:
            raise AgentRegistryError("maximum active agents reached")
        if parent_agent_id and parent_agent_id not in self.agents:
            raise AgentRegistryError(f"unknown parent agent: {parent_agent_id}")
        node = AgentNode(
            agent_id=agent_id or new_agent_id(role.value),
            role=role,
            parent_agent_id=parent_agent_id,
            capabilities=capabilities or list(DEFAULT_CAPABILITIES.get(role, [])),
            model_level=model_level_for_role(role).model_level,
        )
        self._validate_depth(node.parent_agent_id)
        self.agents[node.agent_id] = node
        return node

    def create_subagent(self, role: AgentRole, parent_agent_id: str, capabilities: list[str]) -> AgentNode:
        active_subagents = sum(1 for agent_id in self.agents if agent_id.startswith("subagent_"))
        if active_subagents >= self.max_active_agents:
            raise AgentRegistryError("maximum active agents reached")
        if parent_agent_id not in self.agents:
            raise AgentRegistryError(f"unknown parent agent: {parent_agent_id}")
        node = AgentNode(
            agent_id=new_subagent_id(role.value),
            role=role,
            parent_agent_id=parent_agent_id,
            capabilities=capabilities,
            model_level=model_level_for_role(role).model_level,
        )
        self._validate_depth(node.parent_agent_id)
        self.agents[node.agent_id] = node
        return node

    def deactivate(self, agent_id: str) -> None:
        if agent_id not in self.agents:
            raise AgentRegistryError(f"unknown agent: {agent_id}")
        self.agents[agent_id].state = AgentState.DONE

    def set_parent(self, agent_id: str, parent_agent_id: str | None) -> None:
        if agent_id == parent_agent_id:
            raise AgentRegistryError("agent cannot be its own parent")
        node = self.agents[agent_id]
        original = node.parent_agent_id
        node.parent_agent_id = parent_agent_id
        try:
            self._validate_no_cycle(agent_id)
            self._validate_depth(parent_agent_id)
        except Exception:
            node.parent_agent_id = original
            raise

    def find_by_capability(self, capability: str) -> list[AgentNode]:
        return [node for node in self.agents.values() if capability in node.capabilities]

    def find_by_role(self, role: AgentRole) -> AgentNode:
        for node in self.agents.values():
            if node.role == role:
                return node
        raise AgentRegistryError(f"agent role not registered: {role.value}")

    def hierarchy(self) -> dict[str, list[str]]:
        tree: dict[str, list[str]] = {}
        for node in self.agents.values():
            tree.setdefault(node.parent_agent_id or "root", []).append(node.agent_id)
        return tree

    def _validate_no_cycle(self, agent_id: str) -> None:
        seen: set[str] = set()
        cursor: str | None = agent_id
        while cursor:
            if cursor in seen:
                raise AgentRegistryError("circular hierarchy is not allowed")
            seen.add(cursor)
            cursor = self.agents[cursor].parent_agent_id if cursor in self.agents else None

    def _validate_depth(self, parent_agent_id: str | None) -> None:
        depth = 1
        cursor = parent_agent_id
        seen: set[str] = set()
        while cursor:
            if cursor in seen:
                raise AgentRegistryError("circular hierarchy is not allowed")
            seen.add(cursor)
            depth += 1
            if depth > self.max_depth:
                raise AgentRegistryError("maximum hierarchy depth exceeded")
            cursor = self.agents[cursor].parent_agent_id
