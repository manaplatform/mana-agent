# Agent Client Protocol (ACP)

ACP v1 lets compatible editors use Mana-Agent as a coding agent over stdio. Install `mana-agent[acp]`, then run:

```bash
mana-agent acp info
mana-agent acp doctor
mana-agent acp serve --repo /absolute/path/to/repository
```

`serve` reserves stdout exclusively for ACP JSON-RPC. Diagnostics go to stderr. One gateway is constructed for the repository; each ACP session maps one-to-one to a durable Mana session and subsequent prompts reuse it. Loading a retained ACP session reopens the same Mana identity and replays its user/assistant history as session updates.

The adapter supports initialize, new/load/list sessions, prompts, cancellation notifications, session updates, modes, configuration options, text/resource-link prompts, per-session MCP definitions, tool updates, and durable history. Read-only mode rejects mutation routes after the routing model decides the route and before execution.

## Editor setup

Use an executable command and arguments; do not put API keys in editor settings:

```json
{
  "command": "mana-agent",
  "args": ["acp", "serve", "--repo", "/absolute/path/to/repository"]
}
```

- Zed: add the command as a custom ACP agent in the editor agent settings.
- JetBrains: configure the same command in an ACP-capable plugin.
- Neovim: pass the command/argument array to the ACP client plugin.
- Other clients: launch it as an ACP v1 stdio agent.

Compatibility depends on the editor or extension implementing ACP. If initialization fails, run `mana-agent acp doctor`, confirm the `acp` extra is installed, and ensure no wrapper writes banners or log messages to stdout.
