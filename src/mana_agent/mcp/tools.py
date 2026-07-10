from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import create_model

from .client import McpClient
from .config import McpConfigError, load_mcp_servers


def mcp_model_tool_name(server_id: str, tool_name: str) -> str:
    """Encode an MCP tool as an OpenAI-compatible function name."""
    return f"mcp__{server_id}__{tool_name}".replace(".", "_")


def discovered_mcp_langchain_tools(*, overrides: list[str] | None = None) -> tuple[list[Any], list[str]]:
    """Build model-visible MCP tools from configured providers."""
    try:
        selected_overrides = list(overrides or [])
        if not selected_overrides and os.getenv("MANA_MCP_SERVER_OVERRIDES"):
            raw = json.loads(str(os.environ["MANA_MCP_SERVER_OVERRIDES"]))
            if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
                raise McpConfigError("MANA_MCP_SERVER_OVERRIDES must be a JSON list of server definitions")
            selected_overrides = raw
        client = McpClient(load_mcp_servers(overrides=selected_overrides))
        discovery = client.discover()
    except (McpConfigError, RuntimeError, ValueError) as exc:
        return [], [f"MCP discovery failed; no MCP tools were registered: {exc}"]
    tools: list[Any] = []
    for descriptor in discovery.get("tools", []):
        qualified = str(descriptor["qualified_name"])
        model_name = mcp_model_tool_name(str(descriptor["server_id"]), str(descriptor["name"]))
        schema = descriptor.get("input_schema") if isinstance(descriptor.get("input_schema"), dict) else {}
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = set(schema.get("required") or [])
        fields = {str(name): (Any, ... if str(name) in required else None) for name in properties}
        args_schema = create_model("Mcp_" + model_name.replace("-", "_"), **fields)

        def call(_tool: str = qualified, **kwargs: Any) -> str:
            return json.dumps(client.call_tool(_tool, {key: value for key, value in kwargs.items() if value is not None}), ensure_ascii=False, default=str)

        tools.append(StructuredTool.from_function(func=call, name=model_name, description=str(descriptor.get("description") or f"MCP tool {qualified}"), args_schema=args_schema))
    return tools, []


def discovered_mcp_tool_names() -> list[str]:
    tools, _warnings = discovered_mcp_langchain_tools()
    return sorted(str(tool.name) for tool in tools)
