# Agent2Agent Protocol (A2A)

Mana-Agent serves A2A 1.0 through the official Python SDK using JSON-RPC and HTTP+JSON. Install `mana-agent[a2a]` and configure a bearer token outside normal command output:

```bash
export MANA_A2A_SERVER_TOKEN='replace-with-a-secret'
mana-agent a2a doctor --public-base-url https://agent.example
mana-agent a2a card --public-base-url https://agent.example
mana-agent a2a serve --repo /srv/repository \
  --host 127.0.0.1 --port 8766 \
  --public-base-url https://agent.example
```

Terminate TLS in a production reverse proxy. The default loopback bind is deliberate; Mana-Agent rejects non-local deployment metadata that does not name an HTTPS public URL. The public Agent Card is served at `/.well-known/agent-card.json`. Push notifications and extended cards are not advertised.

A2A context IDs map to Mana sessions. A2A task IDs map to durable Mana task-board records plus caller-scoped SDK task records. Submitted, working, completed, failed, canceled, input-required, authentication-required, and rejected states have explicit mappings; an unknown internal state is an error, never success. Text answers are returned as artifacts.

## Remote agents

```bash
mana-agent a2a agents add https://remote.example/.well-known/agent-card.json \
  --name reviewer --skill code-review --trust
mana-agent a2a agents list
mana-agent a2a agents inspect reviewer --refresh
mana-agent a2a delegate reviewer "Review this explicitly supplied patch summary" \
  --skill code-review
mana-agent a2a agents remove reviewer
```

The registry stores non-secret metadata under the Mana user state directory. Credentials must be supplied from the secret/configuration layer and are never stored in the registry. Discovery requires A2A 1.0, validates HTTPS and resolved addresses, disables redirects, and caches validated cards. Local development endpoints require the explicit `--allow-local` delegation flag.

Delegation sends only the explicit approved context package. Repository files, memory, chat history, secrets, and credentials are not added implicitly. Policy verifies the selected skill, workspace allowlist, authentication availability, concurrency limits, hop depth, visited agents, and task fingerprint before a handoff. After a remote accepts work, failures are reported rather than silently switching to a local agent.
