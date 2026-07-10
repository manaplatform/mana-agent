# MCP Interoperability

Mana-Agent can consume tools and resources from Model Context Protocol servers,
and can expose its own repository tools through MCP.

## Configure external servers

Create `~/.mana/mcp.toml` (or set `MANA_MCP_CONFIG_PATH`):

    [[servers]]
    id = "filesystem"
    transport = "stdio"
    command = "npx"
    args = ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]

Imported tools are namespaced as `mcp.<server_id>.<tool_name>`. Invalid
configuration or failed discovery disables that provider; no local substitute is used.

For a remote bearer token, run `mana-agent mcp token-set` with no server id.
Use arrow keys and Enter to select a configured MCP server, then enter the
hidden token. Supplying a server id remains available for scripts. Tokens are
stored in mode-0600 `~/.mana/mcp_secrets.toml`, never MCP tool metadata or
server configuration. Alternatively set `token_env` in a server definition.

For the configured `context7` stdio provider, `mcp token-set context7` is passed
to the subprocess as `CONTEXT7_API_KEY`, as required by Context7.

If local stdio Context7 is blocked or times out, switch the same configured
provider to its hosted Streamable HTTP endpoint:

    mana-agent mcp add context7 --replace --transport streamable_http --url https://mcp.context7.com/mcp

For one chat session, repeat `--mcp-server-json` with a JSON server definition.
Legacy `sse` is supported only when explicitly selected.

## Serve Mana-Agent

    export MANA_MCP_SERVER_TOKEN='choose-a-long-random-token'
    mana-agent mcp serve --root-dir /path/to/repository

The Streamable HTTP endpoint is `http://127.0.0.1:8765/mcp` and requires a bearer
token. Use `--transport stdio` for local-process integration. Calls run through
Mana-Agent's existing path, document, shell, and Git safeguards.
