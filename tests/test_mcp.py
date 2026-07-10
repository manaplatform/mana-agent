from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from mana_agent.mcp.client import McpClient
from mana_agent.mcp.tools import mcp_model_tool_name
from mana_agent.mcp.config import McpConfigError, McpServerConfig, load_mcp_servers, load_mcp_token, parse_mcp_server_json, save_mcp_server, save_mcp_token
from mana_agent.mcp.server import protected_http_app
from mana_agent.multi_agent.core.types import QueueJob, QueueJobType
from mana_agent.multi_agent.tools.tool_manager import ToolsManager
from mana_agent.multi_agent.runtime.entry_router import EntryRouter, RouteDecision
from mana_agent.multi_agent.runtime.route_executor import RouteExecutionContext, RouteExecutor
from mana_agent.services.ask_service import AskService
from mana_agent.commands.chat_cli import _explicit_mcp_server_request


def test_mcp_config_loads_servers_and_rejects_duplicate_ids(tmp_path):
    config = tmp_path / "mcp.toml"
    config.write_text('[[servers]]\nid = "local"\ntransport = "stdio"\ncommand = "python"\nargs = ["server.py"]\n', encoding="utf-8")
    servers = load_mcp_servers(config)
    assert servers[0].namespace == "mcp.local"
    with pytest.raises(McpConfigError, match="duplicate"):
        load_mcp_servers(config, [json.dumps({"id": "local", "transport": "stdio", "command": "other"})])


def test_mcp_config_rejects_invalid_inline_definition():
    with pytest.raises(McpConfigError, match="object"):
        parse_mcp_server_json("[]")
    with pytest.raises(McpConfigError, match="require command"):
        parse_mcp_server_json('{"id":"x","transport":"stdio"}')


def test_mcp_queue_job_uses_namespaced_tool(monkeypatch, tmp_path):
    calls = []

    class FakeClient:
        def __init__(self, servers):
            assert servers == []
        def call_tool(self, name, args):
            calls.append((name, args))
            return {"ok": True, "server_id": "demo", "tool_name": "echo"}

    monkeypatch.setattr("mana_agent.multi_agent.tools.tool_manager.McpClient", FakeClient)
    manager = ToolsManager(tmp_path)
    job = QueueJob("job", "task", "agent", QueueJobType.MCP_TOOL, {"tool_name": "mcp.demo.echo", "args": {"value": 1}})
    result = manager.execute_job(job)
    assert result.ok is True
    assert calls == [("mcp.demo.echo", {"value": 1})]


def test_mcp_model_tool_name_is_openai_compatible():
    assert mcp_model_tool_name("context7", "query-docs") == "mcp__context7__query-docs"


def test_mcp_stdio_discovers_calls_tool_and_reads_resource(tmp_path):
    server = tmp_path / "server.py"
    server.write_text(
        """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("fixture")
@mcp.tool()
def echo(value: str) -> str:
    return value
@mcp.resource("fixture://status")
def status() -> str:
    return "ready"
mcp.run(transport="stdio")
""",
        encoding="utf-8",
    )
    config = McpServerConfig(id="fixture", transport="stdio", command=str(__import__("sys").executable), args=[str(server)])
    client = McpClient([config])
    discovery = client.discover()
    assert discovery["tools"][0]["qualified_name"] == "mcp.fixture.echo"
    assert client.call_tool("mcp.fixture.echo", {"value": "ok"})["ok"] is True
    assert client.read_resource("fixture", "fixture://status")["ok"] is True


def test_mcp_http_requires_bearer_token(tmp_path):
    client = TestClient(protected_http_app(repo_root=tmp_path, token="secret"))
    assert client.post("/mcp", json={}).status_code == 401
    assert client.post("/mcp", headers={"Authorization": "Bearer secret"}, json={}).status_code != 401


def test_mcp_token_is_stored_outside_server_config(monkeypatch, tmp_path):
    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    save_mcp_token("remote", "secret-value")
    config = McpServerConfig(id="remote", transport="streamable_http", url="https://example.test/mcp")
    assert config.resolved_headers()["Authorization"] == "Bearer secret-value"
    assert "secret-value" not in config.model_dump_json()


def test_context7_stdio_receives_its_stored_token(monkeypatch, tmp_path):
    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    save_mcp_token("context7", "ctx7-token")
    config = McpServerConfig(id="context7", transport="stdio", command="npx")
    assert config.resolved_env()["CONTEXT7_API_KEY"] == "ctx7-token"


def test_mcp_server_registration_persists_a_chat_usable_definition(monkeypatch, tmp_path):
    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    path = save_mcp_server(McpServerConfig(id="demo", transport="stdio", command="python", args=["server.py"]))
    assert path.exists()
    assert load_mcp_servers()[0].id == "demo"


def test_mcp_server_registration_can_replace_a_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    save_mcp_server(McpServerConfig(id="demo", transport="stdio", command="python"))
    save_mcp_server(
        McpServerConfig(id="demo", transport="streamable_http", url="https://example.test/mcp"),
        replace=True,
    )
    servers = load_mcp_servers()
    assert len(servers) == 1
    assert servers[0].transport == "streamable_http"


def test_mcp_token_set_interactively_selects_configured_server(monkeypatch, tmp_path):
    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    save_mcp_server(McpServerConfig(id="first", transport="stdio", command="python"))
    save_mcp_server(McpServerConfig(id="second", transport="stdio", command="python"))
    monkeypatch.setattr("mana_agent.tui.menu.select_option", lambda **kwargs: "second")
    from mana_agent.commands.cli_internal import mcp_token_set_command
    mcp_token_set_command(server_id=None, token="token-value")
    assert load_mcp_token("second") == "token-value"


def test_explicit_mcp_provider_blocks_web_search_substitution(monkeypatch, tmp_path):
    monkeypatch.setenv("MANA_HOME", str(tmp_path))
    save_mcp_server(McpServerConfig(id="context7", transport="stdio", command="python"))
    assert AskService._requested_mcp_server("Use Context7 for this") == "context7"
    executor = RouteExecutor(router=EntryRouter(llm=object()), store=None, qna_chain=None)
    decision = RouteDecision(kind="web_search", confidence=1.0, reason="wrong route")
    context = RouteExecutionContext(question="use Context7", index_dir=None, required_mcp_server="context7")
    assert "explicitly required MCP provider" in str(executor._validate(decision, context))


def test_chat_fast_path_detects_explicit_mcp_provider():
    class AskServiceStub:
        @staticmethod
        def _requested_mcp_server(question):
            return "context7" if "Context7" in question else None

    assert _explicit_mcp_server_request(ask_service=AskServiceStub(), question="use Context7") == "context7"
