from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from .config import McpServerConfig


@dataclass(frozen=True)
class McpToolDescriptor:
    server_id: str
    name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def qualified_name(self) -> str:
        return f"mcp.{self.server_id}.{self.name}"


@dataclass(frozen=True)
class McpResourceDescriptor:
    server_id: str
    uri: str
    name: str
    description: str = ""
    mime_type: str = ""


class McpClient:
    """Short-lived MCP sessions, safe for sync worker processes and queues."""

    def __init__(self, servers: list[McpServerConfig]) -> None:
        self.servers = {server.id: server for server in servers}

    def discover(self) -> dict[str, Any]:
        return self._run(self._discover())

    def call_tool(self, qualified_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        server_id, name = self._parse_tool_name(qualified_name)
        return self._run(self._call_tool(server_id, name, arguments))

    def read_resource(self, server_id: str, uri: str) -> dict[str, Any]:
        if server_id not in self.servers:
            raise ValueError(f"unknown MCP server: {server_id}")
        return self._run(self._read_resource(server_id, uri))

    @staticmethod
    def _run(coro: Any) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError("MCP client cannot run synchronously inside an active event loop")

    def _parse_tool_name(self, value: str) -> tuple[str, str]:
        bits = str(value).split(".", 2)
        if len(bits) != 3 or bits[0] != "mcp" or not bits[1] or not bits[2]:
            raise ValueError("MCP tool must be named mcp.<server_id>.<tool_name>")
        if bits[1] not in self.servers:
            raise ValueError(f"unknown MCP server: {bits[1]}")
        return bits[1], bits[2]

    async def _discover(self) -> dict[str, Any]:
        tools: list[dict[str, Any]] = []
        resources: list[dict[str, Any]] = []
        for server in self.servers.values():
            async with self._session(server) as session:
                listed_tools = await asyncio.wait_for(session.list_tools(), timeout=server.timeout_seconds)
                listed_resources = await asyncio.wait_for(session.list_resources(), timeout=server.timeout_seconds)
                for tool in listed_tools.tools:
                    tools.append({"server_id": server.id, "name": tool.name, "qualified_name": f"mcp.{server.id}.{tool.name}", "description": tool.description or "", "input_schema": dict(tool.inputSchema or {})})
                for resource in listed_resources.resources:
                    resources.append({"server_id": server.id, "uri": str(resource.uri), "name": resource.name, "description": resource.description or "", "mime_type": resource.mimeType or ""})
        names = [item["qualified_name"] for item in tools]
        if len(names) != len(set(names)):
            raise ValueError("MCP discovery returned duplicate qualified tool names")
        return {"ok": True, "tools": sorted(tools, key=lambda item: item["qualified_name"]), "resources": sorted(resources, key=lambda item: (item["server_id"], item["uri"]))}

    async def _call_tool(self, server_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        server = self.servers[server_id]
        started = time.perf_counter()
        async with self._session(server) as session:
            result = await asyncio.wait_for(session.call_tool(name, arguments=arguments), timeout=server.timeout_seconds)
        content = [item.model_dump(mode="json") if hasattr(item, "model_dump") else str(item) for item in result.content]
        return {"ok": not bool(getattr(result, "isError", False)), "server_id": server_id, "tool_name": name, "transport": server.transport, "content": content, "structured_content": getattr(result, "structuredContent", None), "is_error": bool(getattr(result, "isError", False)), "duration_ms": round((time.perf_counter() - started) * 1000, 2)}

    async def _read_resource(self, server_id: str, uri: str) -> dict[str, Any]:
        server = self.servers[server_id]
        started = time.perf_counter()
        async with self._session(server) as session:
            result = await asyncio.wait_for(session.read_resource(uri), timeout=server.timeout_seconds)
        content = [item.model_dump(mode="json") if hasattr(item, "model_dump") else str(item) for item in result.contents]
        return {"ok": True, "server_id": server_id, "uri": uri, "transport": server.transport, "content": content, "duration_ms": round((time.perf_counter() - started) * 1000, 2)}

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _session(self, server: McpServerConfig):
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamable_http_client
            from mcp.client.sse import sse_client
        except ImportError as exc:
            raise RuntimeError("MCP support requires the 'mcp' package; reinstall mana-agent") from exc
        if server.transport == "stdio":
            params = StdioServerParameters(command=server.command, args=server.args, env=server.resolved_env())
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=server.timeout_seconds)
                    yield session
            return
        if server.transport == "streamable_http":
            import httpx

            async with httpx.AsyncClient(
                headers=server.resolved_headers() or None,
                timeout=server.timeout_seconds,
            ) as http_client:
                async with streamable_http_client(server.url, http_client=http_client) as streams:
                    read, write = streams[0], streams[1]
                    async with ClientSession(read, write) as session:
                        await asyncio.wait_for(session.initialize(), timeout=server.timeout_seconds)
                        yield session
            return
        async with sse_client(
            server.url,
            headers=server.resolved_headers() or None,
            timeout=server.timeout_seconds,
        ) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=server.timeout_seconds)
                yield session
