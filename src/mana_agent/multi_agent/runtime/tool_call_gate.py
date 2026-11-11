"""Bind an existing specialist tool catalog to the common deterministic runner."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import uuid
from mana_agent.coding.models import WorkspaceContext
from mana_agent.multi_agent.agents.tool_runner_agent import ToolRunnerAgent, bind_tool_authority
from mana_agent.multi_agent.core.types import AgentRole
from mana_agent.multi_agent.registry.agent_registry import AgentRegistry
from mana_agent.multi_agent.tools.operation_policy import OperationPolicy
from mana_agent.services.activity_context import current_activity_context
from mana_agent.tools.contracts import coding_tool_contracts


class ToolCallGate:
    def __init__(self, *, root: Path, tools: dict[str, Any], task_id: str, transport=None, repository_read_only: bool = False, externally_locked: bool = False):
        self.root, self.tools = Path(root).resolve(), tools
        self.task_id = task_id
        self.transport = transport
        self.repository_read_only = repository_read_only
        self.externally_locked = externally_locked
        context = current_activity_context()
        self.registry = getattr(context, "execution_registry", None) or AgentRegistry()
        self.runners: dict[str, ToolRunnerAgent] = {}
        self.contracts = {c.name.replace(".", "_"): c for c in coding_tool_contracts()}

    def invoke(self, name: str, arguments: dict, *, call_id: str, config: dict | None = None):
        if not self.task_id:
            context = current_activity_context()
            self.task_id = str(getattr(context, "task_id", "") or f"task_{uuid.uuid4().hex[:8]}")
        tool = self.tools[name]
        contract = self.contracts.get(name.replace(".", "_"))
        metadata = dict(getattr(tool, "metadata", None) or {})
        if metadata.get("execution_capability") and metadata.get("execution_effect") in {"read_only", "mutation", "command", "external"}:
            policy = OperationPolicy(metadata["execution_capability"], metadata["execution_effect"], bool(metadata.get("repository_operation")))
        elif contract is not None and getattr(contract, "effect", None):
            policy = OperationPolicy(contract.required_capability or "repository_execution", contract.effect, bool(getattr(contract, "repository_operation", True)))
        else:
            effect = "read_only" if metadata.get("read_only") else "external"
            policy = OperationPolicy("tool_execution", effect, bool(metadata.get("repository_operation", False)))
        key = "coding" if policy.repository else "specialist"
        if key not in self.runners:
            node = self.registry.find_by_role(AgentRole.CODING if policy.repository else AgentRole.TOOL)
            if node is None:
                node = self.registry.register(AgentRole.CODING if policy.repository else AgentRole.TOOL)
            context = current_activity_context()
            workspace = WorkspaceContext(repository_path=self.root, worktree_path=self.root, allow_in_place_write=True,
                sandbox="readOnly" if (policy.repository and self.repository_read_only) or str(getattr(context, "execution_mode", "")) in {"plan", "plan_only", "read_only"} else "workspaceWrite")
            # Bind only the selected catalog, with schemas originating in its registry.
            policies, schemas = {}, {}
            for tool_name, candidate in self.tools.items():
                item = self.contracts.get(tool_name.replace(".", "_"))
                info = getattr(candidate, "metadata", None) or {}
                if info.get("execution_capability") and info.get("execution_effect") in {"read_only", "mutation", "command", "external"}:
                    spec = OperationPolicy(info["execution_capability"], info["execution_effect"], bool(info.get("repository_operation")))
                elif item is not None and getattr(item, "effect", None):
                    spec = OperationPolicy(item.required_capability or "repository_execution", item.effect, bool(getattr(item, "repository_operation", True)))
                else:
                    eff = "read_only" if info.get("read_only") else "external"
                    spec = OperationPolicy("tool_execution", eff, bool(info.get("repository_operation", False)))
                if spec.repository != policy.repository:
                    continue
                policies[tool_name] = spec
                schema = getattr(candidate, "args_schema", None)
                schemas[tool_name] = schema if isinstance(schema, dict) else schema.model_json_schema() if schema is not None else item.input_schema if item else {"type": "object"}
            board = getattr(context, "taskboard", None)
            if board is not None:
                board.assign(self.task_id, node.agent_id)
            authority = bind_tool_authority(registry=self.registry, actor_id=node.agent_id,
                task_id=self.task_id, workspace=workspace, policies=policies, schemas=schemas,
                authorize=getattr(context, "authorize_tool", None), taskboard=board, externally_locked=self.externally_locked)
            self.runners[key] = ToolRunnerAgent(authority)
        runner = self.runners[key]
        request = runner.request(name, arguments, call_id=call_id or uuid.uuid4().hex,
            decision_id=call_id or uuid.uuid4().hex)
        def execute(selected):
            if self.transport is not None and policy.repository:
                return self.transport(request)
            value = tool.invoke(selected, config=config)
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except ValueError:
                    return {"ok": True, "result": value}
                return parsed if isinstance(parsed, dict) else {"ok": True, "result": parsed}
            return value if isinstance(value, dict) else {"ok": True, "result": value}
        result = runner.execute(request, execute)
        from mana_agent.multi_agent.agents.tool_runner_agent import model_result
        return json.dumps(model_result(result, runner.authority), ensure_ascii=False, default=str)


def execute_host_operation(*, root: Path, task_id: str, tool_name: str, arguments: dict,
                           implementation, call_id: str = "", effect: str | None = None):
    """Adapt an already selected legacy CodingAgent operation, without planning."""
    from mana_agent.multi_agent.tools.operation_policy import builtin_policy
    context = current_activity_context()
    identity = str(getattr(context, "task_id", "") or task_id)
    if not identity:
        raise PermissionError("Legacy repository execution requires an assigned task")
    policy = builtin_policy(tool_name)
    if policy is None:
        if effect not in {"read_only", "command", "mutation"}:
            raise PermissionError("Legacy operation has no registered execution effects")
        policy = OperationPolicy("repository_execution", effect)
    registry = getattr(context, "execution_registry", None) or AgentRegistry()
    actor = registry.find_by_role(AgentRole.CODING)
    board = getattr(context, "taskboard", None)
    if board is not None:
        board.assign(identity, actor.agent_id)
    authority = bind_tool_authority(registry=registry, actor_id=actor.agent_id, task_id=identity,
        workspace=WorkspaceContext(repository_path=root, worktree_path=root, allow_in_place_write=True,
            sandbox="readOnly" if str(getattr(context, "execution_mode", "")) in {"plan", "plan_only", "read_only"} else "workspaceWrite"),
        policies={tool_name: policy}, schemas={tool_name: {"type": "object"}},
        authorize=getattr(context, "authorize_tool", None), taskboard=board)
    runner = ToolRunnerAgent(authority)
    selected = runner.request(tool_name, arguments, call_id=call_id or uuid.uuid4().hex,
        decision_id=call_id or uuid.uuid4().hex)
    return runner.execute(selected, implementation).result
