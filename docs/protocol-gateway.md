# Protocol gateway

Mana-Agent exposes three distinct protocol roles through one runtime:

- MCP connects Mana-Agent to tools and resources.
- ACP connects editor clients to Mana-Agent.
- A2A connects Mana-Agent to other autonomous agents.

ACP and A2A are transport adapters around `AgentChatGateway`. They reuse its workspace/session records, durable history, memory scope, lane coordinator, task board, routing decisions, tool policy, and verification path. They do not call models, coding agents, or tools directly. Asynchronous turns are serialized at the shared gateway boundary because the underlying memory and tool stack is session-bound.

Install protocol support without adding it to core installations:

```bash
pip install "mana-agent[acp]"
pip install "mana-agent[a2a]"
# or both
pip install "mana-agent[protocols]"
```

Missing SDKs fail with an installation instruction. Mana-Agent does not substitute a handwritten or reduced protocol implementation.

Security boundaries include absolute workspace-scoped paths, fail-closed bearer authentication for A2A, caller-scoped task reads, request/artifact limits, secret redaction, HTTPS/public-address validation for remote agents, explicit context packages, and delegation-loop limits. The public Agent Card contains capabilities, not local paths, model names, prompts, or credentials.

Features intentionally not advertised are embedded ACP media/resources, ACP client filesystem/terminal delegation, A2A push notifications, authenticated extended cards, and unrestricted local-file artifacts. These remain disabled until their full approval, persistence, and access-control paths are available.
